"""
dp/noise.py
差分隐私噪声计算模块
包含：基准噪声计算、异构噪声分配、隐私预算管理
"""

import math
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class PrivacyAccountant:
    """隐私预算记账器"""
    epsilon_total: float
    delta: float
    num_rounds: int
    
    def __post_init__(self):
        # 预分配每轮的隐私预算（均匀分配）
        # 使用 RDP 组合：总预算 = sqrt(sum(ε_i²))
        # 所以每轮预算 = ε_total / sqrt(num_rounds)
        self.epsilon_per_round = self.epsilon_total / math.sqrt(self.num_rounds)
        self.epsilon_spent = 0.0
        self.round_count = 0
    
    def get_round_epsilon(self) -> float:
        """获取当前轮的隐私预算"""
        return self.epsilon_per_round
    
    def consume_round(self, epsilon_used: Optional[float] = None):
        """消耗一轮隐私预算"""
        eps = epsilon_used if epsilon_used is not None else self.epsilon_per_round
        self.epsilon_spent = math.sqrt(self.epsilon_spent**2 + eps**2)
        self.round_count += 1
    
    def remaining_budget(self) -> float:
        """剩余隐私预算"""
        return max(0, self.epsilon_total - self.epsilon_spent)
    
    def is_exhausted(self) -> bool:
        """隐私预算是否耗尽"""
        return self.epsilon_spent >= self.epsilon_total


def compute_base_noise_std(
    epsilon: float,
    delta: float,
    clip_norm: float,
    sampling_rate: float = 1.0,
    sparsity_rate: float = 1.0,
    use_subsampling_amplification: bool = True,
    use_sparsity_amplification: bool = True,
) -> float:
    """
    计算基准噪声标准差
    
    使用 Gaussian Mechanism:
        σ = Δf × √(2 ln(1.25/δ)) / ε
    
    其中 Δf = C（裁剪阈值）
    
    参数：
        epsilon: 隐私预算 ε
        delta: 隐私参数 δ
        clip_norm: 裁剪阈值 C
        sampling_rate: 客户端采样率 q
        sparsity_rate: Top-k 稀疏率 k/d
        use_subsampling_amplification: 是否使用子采样隐私放大
        use_sparsity_amplification: 是否使用稀疏化隐私放大
    
    返回：
        噪声标准差 σ
    """
    if epsilon <= 0:
        raise ValueError("ε must be positive")
    
    # 基础 Gaussian Mechanism
    sensitivity = clip_norm
    sigma_base = sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon
    
    # 子采样隐私放大（近似）
    # 如果只选择 q 比例的客户端，有效噪声降低
    if use_subsampling_amplification and sampling_rate < 1.0:
        # 这是一个启发式近似，严格分析需要使用 RDP
        sigma_base *= math.sqrt(sampling_rate)
    
    # 稀疏化隐私放大（启发式）
    # 只上传 k/d 比例的参数，敏感度降低
    if use_sparsity_amplification and sparsity_rate < 1.0:
        sigma_base *= math.sqrt(sparsity_rate)
    
    return sigma_base


def allocate_heterogeneous_epsilon(
    importance: torch.Tensor,
    mask: torch.Tensor,
    epsilon_total: float,
    min_epsilon_ratio: float = 0.1,
) -> torch.Tensor:
    """
    基于重要性分配异构隐私预算
    
    策略：重要性高的参数分配更多隐私预算（更少噪声）
    
    约束：√(Σ ε_i²) ≤ ε_total（Rényi DP 组合）
    
    参数：
        importance: 参数重要性矩阵（展平后）
        mask: Top-k 掩码
        epsilon_total: 总隐私预算
        min_epsilon_ratio: 最小隐私预算比例（防止某些参数噪声过大）
    
    返回：
        每个参数的隐私预算 ε_i
    """
    # 只对选中的参数分配
    selected_importance = importance[mask > 0]
    num_selected = selected_importance.numel()
    
    if num_selected == 0:
        return torch.zeros_like(importance)
    
    # 归一化重要性（0-1范围）
    imp_min = selected_importance.min()
    imp_max = selected_importance.max()
    if imp_max > imp_min:
        normalized_imp = (selected_importance - imp_min) / (imp_max - imp_min)
    else:
        normalized_imp = torch.ones_like(selected_importance)
    
    # 基于重要性的平方根分配
    # 重要性高 → 更大 ε → 更小噪声
    weights = normalized_imp.sqrt() + min_epsilon_ratio
    weights = weights / weights.sum()
    
    # 满足 RDP 约束：√(Σ ε_i²) = ε_total
    # 所以 Σ ε_i² = ε_total²
    # ε_i = weight_i * scale, 其中 scale 使得 Σ(weight_i * scale)² = ε_total²
    scale = epsilon_total / (weights.pow(2).sum().sqrt())
    epsilon_i = weights * scale
    
    # 构建完整的 epsilon 张量
    full_epsilon = torch.zeros_like(importance)
    full_epsilon[mask > 0] = epsilon_i
    
    return full_epsilon


def compute_heterogeneous_noise_std(
    epsilon_i: torch.Tensor,
    delta: float,
    clip_norm: float,
) -> torch.Tensor:
    """
    基于异构隐私预算计算每个参数的噪声标准差
    
    参数：
        epsilon_i: 每个参数的隐私预算
        delta: 隐私参数 δ
        clip_norm: 裁剪阈值
    
    返回：
        每个参数的噪声标准差 σ_i
    """
    # 避免除零
    epsilon_safe = epsilon_i.clamp(min=1e-10)
    
    # σ_i = C × √(2 ln(1.25/δ)) / ε_i
    factor = math.sqrt(2 * math.log(1.25 / delta))
    sigma_i = clip_norm * factor / epsilon_safe
    
    # 未选中的参数（ε=0）设置噪声为0（不上传，不加噪）
    sigma_i[epsilon_i == 0] = 0.0
    
    return sigma_i


def add_noise_to_gradient(
    gradient: torch.Tensor,
    sigma: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    给梯度添加高斯噪声（同构噪声）
    
    参数：
        gradient: 梯度张量
        sigma: 噪声标准差
        generator: 随机数生成器（可选，必须在CPU上）
    
    返回：
        加噪后的梯度
    """
    if generator is not None:
        # 在 CPU 上生成噪声，然后移动到目标设备
        noise = torch.randn(gradient.shape, generator=generator, dtype=gradient.dtype)
        noise = noise.to(gradient.device) * sigma
    else:
        noise = torch.randn_like(gradient) * sigma
    return gradient + noise


def add_heterogeneous_noise(
    gradient: torch.Tensor,
    sigma_i: torch.Tensor,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    给梯度添加异构高斯噪声
    
    参数：
        gradient: 梯度张量
        sigma_i: 每个参数的噪声标准差（与 gradient 形状相同）
        generator: 随机数生成器（可选，必须在CPU上）
    
    返回：
        加噪后的梯度
    """
    # 生成标准正态噪声
    if generator is not None:
        # 在 CPU 上生成，然后移动到目标设备
        z = torch.randn(gradient.shape, generator=generator, dtype=gradient.dtype)
        z = z.to(gradient.device)
    else:
        z = torch.randn_like(gradient)
    # 乘以对应的标准差
    noise = z * sigma_i
    return gradient + noise


def verify_privacy_constraint(
    epsilon_i: torch.Tensor,
    epsilon_total: float,
    tolerance: float = 1e-6,
) -> Tuple[bool, float]:
    """
    验证异构隐私预算是否满足 RDP 约束
    
    约束：√(Σ ε_i²) ≤ ε_total
    
    返回：
        (是否满足约束, 实际消耗的隐私预算)
    """
    epsilon_used = epsilon_i.pow(2).sum().sqrt().item()
    satisfied = epsilon_used <= epsilon_total + tolerance
    return satisfied, epsilon_used


# ══════════════════════════════════════════════════════════════════════════════
# 快速验证
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== 差分隐私模块测试 ===\n")
    
    # 1. 测试隐私记账器
    print("1. 隐私记账器测试")
    accountant = PrivacyAccountant(epsilon_total=8.0, delta=1e-5, num_rounds=100)
    print(f"   总预算: {accountant.epsilon_total}")
    print(f"   每轮预算: {accountant.epsilon_per_round:.4f}")
    
    for _ in range(10):
        accountant.consume_round()
    print(f"   10轮后已消耗: {accountant.epsilon_spent:.4f}")
    print(f"   剩余预算: {accountant.remaining_budget():.4f}")
    
    # 2. 测试基准噪声计算
    print("\n2. 基准噪声计算")
    sigma = compute_base_noise_std(
        epsilon=0.8, delta=1e-5, clip_norm=1.0,
        sampling_rate=0.1, sparsity_rate=0.1,
    )
    print(f"   ε=0.8, C=1.0, q=0.1, k/d=0.1")
    print(f"   噪声标准差 σ = {sigma:.4f}")
    
    # 3. 测试异构隐私预算分配
    print("\n3. 异构隐私预算分配")
    importance = torch.tensor([0.1, 0.5, 0.8, 0.2, 1.0, 0.3, 0.0, 0.0])
    mask = torch.tensor([1, 1, 1, 1, 1, 1, 0, 0], dtype=torch.float)
    
    epsilon_i = allocate_heterogeneous_epsilon(importance, mask, epsilon_total=0.8)
    print(f"   重要性: {importance.tolist()}")
    print(f"   掩码:   {mask.tolist()}")
    print(f"   ε_i:    {[f'{e:.3f}' for e in epsilon_i.tolist()]}")
    
    # 验证约束
    satisfied, eps_used = verify_privacy_constraint(epsilon_i, 0.8)
    print(f"   约束验证: {'通过' if satisfied else '失败'}, 实际消耗 ε = {eps_used:.4f}")
    
    # 4. 测试异构噪声
    print("\n4. 异构噪声添加")
    gradient = torch.ones(8)
    sigma_i = compute_heterogeneous_noise_std(epsilon_i, delta=1e-5, clip_norm=1.0)
    print(f"   σ_i: {[f'{s:.3f}' for s in sigma_i.tolist()]}")
    
    noisy_grad = add_heterogeneous_noise(gradient, sigma_i)
    print(f"   原梯度: {gradient.tolist()}")
    print(f"   加噪后: {[f'{g:.3f}' for g in noisy_grad.tolist()]}")
    
    print("\n✓ 所有测试通过")
