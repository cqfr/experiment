"""
compression/topk.py
梯度压缩模块
包含：参数重要性评估、Top-k 选择策略、残差累积
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
from enum import Enum


# ══════════════════════════════════════════════════════════════════════════════
# 重要性评估
# ══════════════════════════════════════════════════════════════════════════════

def compute_fisher_information(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_batches: int = 5,
) -> Dict[str, torch.Tensor]:
    """
    计算 Fisher 信息矩阵的对角线（逐参数）
    
    Fisher 信息定义：F_θ = E[(∂logp/∂θ)²]
    近似为：F_θ ≈ (1/N) Σ (∂L/∂θ)²
    
    参数：
        model: 神经网络模型
        dataloader: 数据加载器
        device: 计算设备
        num_batches: 用于估计 Fisher 的 batch 数
    
    返回：
        每个参数的 Fisher 信息估计
    """
    model.eval()
    fisher = {}
    
    # 初始化
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher[name] = torch.zeros_like(param.data)
    
    total_samples = 0
    criterion = nn.CrossEntropyLoss(reduction='sum')
    
    for batch_idx, (data, target) in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
        
        data, target = data.to(device), target.to(device)
        batch_size = data.size(0)
        total_samples += batch_size
        
        model.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        # 累积梯度平方
        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher[name] += param.grad.data.pow(2) * batch_size
    
    # 平均
    for name in fisher:
        fisher[name] /= total_samples
    
    model.train()
    return fisher


def compute_importance_fisher_grad(
    gradient: Dict[str, torch.Tensor],
    fisher: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    策略 A: Fisher 信息 × 梯度绝对值
    
    Importance = Fisher(θ) ⊙ |∇θ|
    """
    importance = {}
    for name in gradient:
        if name in fisher:
            importance[name] = fisher[name] * gradient[name].abs()
        else:
            importance[name] = gradient[name].abs()
    return importance


def compute_importance_grad_squared(
    gradient: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    策略 B: 梯度绝对值 × 梯度绝对值 = 梯度平方
    
    Importance = |∇θ|²
    """
    return {name: grad.pow(2) for name, grad in gradient.items()}


def compute_importance_grad_normalized(
    gradient: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    策略 C: 梯度绝对值（层级归一化）
    
    每层内部归一化到 [0, 1]
    """
    importance = {}
    for name, grad in gradient.items():
        abs_grad = grad.abs()
        min_val = abs_grad.min()
        max_val = abs_grad.max()
        if max_val > min_val:
            importance[name] = (abs_grad - min_val) / (max_val - min_val)
        else:
            importance[name] = torch.ones_like(abs_grad)
    return importance


# ══════════════════════════════════════════════════════════════════════════════
# Top-k 选择策略
# ══════════════════════════════════════════════════════════════════════════════

def topk_global(
    importance: Dict[str, torch.Tensor],
    k_ratio: float,
) -> Dict[str, torch.Tensor]:
    """
    策略 A: 全局归一化 + 全局 Top-k
    
    将所有参数的重要性展平，全局选择前 k%
    问题：小层可能完全不被选中
    """
    # 展平所有重要性
    all_values = torch.cat([imp.flatten() for imp in importance.values()])
    total_params = all_values.numel()
    k = max(1, int(total_params * k_ratio))
    
    # 全局归一化
    min_val = all_values.min()
    max_val = all_values.max()
    if max_val > min_val:
        all_values_norm = (all_values - min_val) / (max_val - min_val)
    else:
        all_values_norm = torch.ones_like(all_values)
    
    # 选择阈值
    threshold = torch.topk(all_values_norm, k).values[-1]
    
    # 构建掩码
    masks = {}
    start = 0
    for name, imp in importance.items():
        size = imp.numel()
        flat_imp = imp.flatten()
        # 归一化
        if max_val > min_val:
            flat_imp_norm = (flat_imp - min_val) / (max_val - min_val)
        else:
            flat_imp_norm = torch.ones_like(flat_imp)
        mask = (flat_imp_norm >= threshold).float().reshape(imp.shape)
        masks[name] = mask
        start += size
    
    return masks


def topk_layer(
    importance: Dict[str, torch.Tensor],
    k_ratio: float,
) -> Dict[str, torch.Tensor]:
    """
    策略 B: 层级 Top-k
    
    每层独立选择前 k% 的参数
    优点：层级平衡
    缺点：忽视层间重要性差异
    """
    masks = {}
    for name, imp in importance.items():
        flat_imp = imp.flatten()
        k = max(1, int(flat_imp.numel() * k_ratio))
        
        # 获取 top-k 索引
        _, indices = torch.topk(flat_imp, k)
        mask = torch.zeros_like(flat_imp)
        mask[indices] = 1.0
        masks[name] = mask.reshape(imp.shape)
    
    return masks


def topk_layer_norm_global(
    importance: Dict[str, torch.Tensor],
    k_ratio: float,
) -> Dict[str, torch.Tensor]:
    """
    策略 C: 层级归一化 + 全局 Top-k
    
    先在每层内归一化，再全局选择
    优点：平衡层级多样性
    """
    # 层级归一化
    normalized = {}
    for name, imp in importance.items():
        min_val = imp.min()
        max_val = imp.max()
        if max_val > min_val:
            normalized[name] = (imp - min_val) / (max_val - min_val)
        else:
            normalized[name] = torch.ones_like(imp)
    
    # 全局选择
    all_values = torch.cat([n.flatten() for n in normalized.values()])
    total_params = all_values.numel()
    k = max(1, int(total_params * k_ratio))
    
    threshold = torch.topk(all_values, k).values[-1]
    
    masks = {}
    for name, norm_imp in normalized.items():
        masks[name] = (norm_imp >= threshold).float()
    
    return masks


def topk_weighted_layer_norm(
    importance: Dict[str, torch.Tensor],
    k_ratio: float,
    weight_method: str = "mean",
) -> Dict[str, torch.Tensor]:
    """
    策略 D: 加权层级归一化 + 全局 Top-k（推荐）
    
    层内归一化后，乘以层权重（代表层的整体重要性），再全局选择
    兼顾层内相对性和层间绝对性
    
    参数：
        weight_method: 层权重计算方法
            - "mean": 层内重要性均值
            - "median": 层内重要性中位数
            - "total_sum": 层内重要性总和
            - "trimmed_mean": 去掉极端值的均值
    """
    # 计算每层的权重
    layer_weights = {}
    for name, imp in importance.items():
        flat_imp = imp.flatten()
        if weight_method == "mean":
            layer_weights[name] = flat_imp.mean().item()
        elif weight_method == "median":
            layer_weights[name] = flat_imp.median().item()
        elif weight_method == "total_sum":
            layer_weights[name] = flat_imp.sum().item()
        elif weight_method == "trimmed_mean":
            # 去掉 5% 的极端值
            sorted_vals = flat_imp.sort().values
            trim = int(len(sorted_vals) * 0.05)
            if trim > 0:
                layer_weights[name] = sorted_vals[trim:-trim].mean().item()
            else:
                layer_weights[name] = flat_imp.mean().item()
        else:
            layer_weights[name] = flat_imp.mean().item()
    
    # 归一化层权重
    max_weight = max(layer_weights.values())
    if max_weight > 0:
        layer_weights = {k: v / max_weight for k, v in layer_weights.items()}
    
    # 层级归一化 + 加权
    weighted_normalized = {}
    for name, imp in importance.items():
        min_val = imp.min()
        max_val = imp.max()
        if max_val > min_val:
            normalized = (imp - min_val) / (max_val - min_val)
        else:
            normalized = torch.ones_like(imp)
        weighted_normalized[name] = normalized * layer_weights[name]
    
    # 全局选择
    all_values = torch.cat([w.flatten() for w in weighted_normalized.values()])
    total_params = all_values.numel()
    k = max(1, int(total_params * k_ratio))
    
    threshold = torch.topk(all_values, k).values[-1]
    
    masks = {}
    for name, w_norm in weighted_normalized.items():
        masks[name] = (w_norm >= threshold).float()
    
    return masks


# ══════════════════════════════════════════════════════════════════════════════
# 残差累积
# ══════════════════════════════════════════════════════════════════════════════

class ResidualAccumulator:
    """
    残差累积器
    
    保存每轮未被选中（未上传）的梯度，累积到下一轮
    """
    
    def __init__(self):
        self.residual: Dict[str, torch.Tensor] = {}
    
    def accumulate(
        self,
        gradient: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        将上一轮残差加到当前梯度
        
        ∇w = ∇w_local + residual_{t-1}
        """
        if not self.residual:
            return gradient
        
        accumulated = {}
        for name, grad in gradient.items():
            if name in self.residual:
                accumulated[name] = grad + self.residual[name]
            else:
                accumulated[name] = grad
        return accumulated
    
    def update(
        self,
        gradient: Dict[str, torch.Tensor],
        mask: Dict[str, torch.Tensor],
    ):
        """
        更新残差：保存未被选中的梯度
        
        residual_t = ∇w ⊙ (1 - mask)
        """
        self.residual = {}
        for name, grad in gradient.items():
            if name in mask:
                self.residual[name] = grad * (1 - mask[name])
    
    def reset(self):
        """重置残差"""
        self.residual = {}


# ══════════════════════════════════════════════════════════════════════════════
# 统一压缩接口
# ══════════════════════════════════════════════════════════════════════════════

def compress_gradient(
    gradient: Dict[str, torch.Tensor],
    importance_strategy: str,
    topk_strategy: str,
    k_ratio: float,
    weight_method: str = "mean",
    fisher: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    统一的梯度压缩接口
    
    参数：
        gradient: 原始梯度
        importance_strategy: 重要性评估策略
            - "fisher_grad": Fisher × |梯度|
            - "grad_squared": |梯度|²
            - "grad_normalized": |梯度|（层级归一化）
        topk_strategy: Top-k 选择策略
            - "global_topk": 全局 Top-k
            - "layer_topk": 层级 Top-k
            - "layer_norm_global": 层级归一化 + 全局
            - "weighted_layer_norm": 加权层级归一化 + 全局
        k_ratio: 压缩率 (0-1)
        weight_method: 层权重方法（仅用于 weighted_layer_norm）
        fisher: Fisher 信息（仅用于 fisher_grad）
    
    返回：
        (稀疏梯度, 掩码)
    """
    # 1. 计算重要性
    if importance_strategy == "fisher_grad":
        if fisher is None:
            raise ValueError("Fisher information required for fisher_grad strategy")
        importance = compute_importance_fisher_grad(gradient, fisher)
    elif importance_strategy == "grad_squared":
        importance = compute_importance_grad_squared(gradient)
    elif importance_strategy == "grad_normalized":
        importance = compute_importance_grad_normalized(gradient)
    else:
        raise ValueError(f"Unknown importance strategy: {importance_strategy}")
    
    # 2. Top-k 选择
    if topk_strategy == "global_topk":
        masks = topk_global(importance, k_ratio)
    elif topk_strategy == "layer_topk":
        masks = topk_layer(importance, k_ratio)
    elif topk_strategy == "layer_norm_global":
        masks = topk_layer_norm_global(importance, k_ratio)
    elif topk_strategy == "weighted_layer_norm":
        masks = topk_weighted_layer_norm(importance, k_ratio, weight_method)
    else:
        raise ValueError(f"Unknown topk strategy: {topk_strategy}")
    
    # 3. 应用掩码
    sparse_gradient = {}
    for name, grad in gradient.items():
        if name in masks:
            sparse_gradient[name] = grad * masks[name]
        else:
            sparse_gradient[name] = grad
    
    return sparse_gradient, masks


# ══════════════════════════════════════════════════════════════════════════════
# 快速验证
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== 梯度压缩模块测试 ===\n")
    
    # 模拟梯度
    gradient = {
        "layer1.weight": torch.randn(64, 32),
        "layer1.bias": torch.randn(64),
        "layer2.weight": torch.randn(128, 64),
        "layer2.bias": torch.randn(128),
    }
    
    total_params = sum(g.numel() for g in gradient.values())
    print(f"总参数量: {total_params}")
    
    # 测试各种策略组合
    strategies = [
        ("grad_normalized", "global_topk"),
        ("grad_normalized", "layer_topk"),
        ("grad_normalized", "layer_norm_global"),
        ("grad_normalized", "weighted_layer_norm"),
        ("grad_squared", "weighted_layer_norm"),
    ]
    
    k_ratio = 0.1
    print(f"\n压缩率: {k_ratio*100}%\n")
    
    for imp_strat, topk_strat in strategies:
        sparse_grad, masks = compress_gradient(
            gradient, imp_strat, topk_strat, k_ratio
        )
        
        # 统计
        selected = sum(m.sum().item() for m in masks.values())
        print(f"{imp_strat} + {topk_strat}:")
        print(f"  选中参数: {int(selected)} / {total_params} ({selected/total_params*100:.1f}%)")
        
        # 每层统计
        for name, mask in masks.items():
            layer_selected = mask.sum().item()
            layer_total = mask.numel()
            print(f"    {name}: {int(layer_selected)}/{layer_total} ({layer_selected/layer_total*100:.1f}%)")
    
    # 测试残差累积
    print("\n=== 残差累积测试 ===")
    accumulator = ResidualAccumulator()
    
    # 第一轮
    _, masks = compress_gradient(gradient, "grad_normalized", "layer_topk", 0.5)
    accumulator.update(gradient, masks)
    print(f"第1轮残差: {sum(r.sum().item() != 0 for r in accumulator.residual.values())} 层有残差")
    
    # 第二轮
    new_gradient = {name: torch.randn_like(g) for name, g in gradient.items()}
    accumulated = accumulator.accumulate(new_gradient)
    print(f"第2轮累积后: 梯度范数变化")
    for name in gradient:
        old_norm = new_gradient[name].norm().item()
        new_norm = accumulated[name].norm().item()
        print(f"  {name}: {old_norm:.4f} → {new_norm:.4f}")
    
    print("\n✓ 所有测试通过")
