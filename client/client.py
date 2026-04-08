"""
client/client.py
联邦学习客户端
实现：本地训练、梯度压缩、裁剪、加噪、上传
"""

import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass

import sys
sys.path.append('..')

from config import ClientConfig, DPConfig, TrainingStrategy, ImportanceStrategy, TopKStrategy
from compression.topk import (
    compress_gradient, ResidualAccumulator,
    compute_fisher_information, compute_importance_grad_normalized
)
from dp.noise import (
    compute_base_noise_std, add_noise_to_gradient,
    allocate_heterogeneous_epsilon, compute_heterogeneous_noise_std,
    add_heterogeneous_noise
)


@dataclass
class ClientUpdate:
    """客户端上传的内容"""
    gradient: Dict[str, torch.Tensor]    # 加噪后的梯度更新
    data_size: int                        # 本地数据集大小
    stat: float                           # L2范数统计（用于自适应裁剪）
    clipped: bool                         # 是否被裁剪
    noise_sigma: float = 0.0              # 添加的噪声强度（用于日志）


class FLClient:
    """
    联邦学习客户端
    
    职责：
    1. 接收全局模型
    2. 本地训练（可选 Contrastive Loss）
    3. 残差累积
    4. 重要性评估 + Top-k 压缩
    5. 梯度裁剪
    6. 差分隐私加噪
    7. 上传更新
    """
    
    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        dataloader: DataLoader,
        config: ClientConfig,
        dp_config: DPConfig,
        device: torch.device,
    ):
        self.client_id = client_id
        self.model = copy.deepcopy(model).to(device)
        self.dataloader = dataloader
        self.config = config
        self.dp_config = dp_config
        self.device = device
        
        # 本地数据集大小
        self.data_size = len(dataloader.dataset)
        
        # 残差累积器
        self.residual_accumulator = ResidualAccumulator()
        
        # 保存上一轮的本地模型（用于 Contrastive Loss）
        self.old_local_weights: Optional[Dict[str, torch.Tensor]] = None
        
        # Fisher 信息缓存（用于 fisher_grad 策略）
        self.fisher_cache: Optional[Dict[str, torch.Tensor]] = None
        
        # 随机数生成器（保证可复现）
        # 注意：Generator 在 CPU 上更兼容，噪声生成后再移动到目标设备
        self.rng = torch.Generator(device='cpu')
    
    def receive_global_model(self, global_weights: Dict[str, torch.Tensor]):
        """Step 1: 接收全局模型"""
        self.model.load_state_dict(global_weights)
        # 保存全局模型作为参考
        self.global_weights = {k: v.clone() for k, v in global_weights.items()}
    
    def local_train(self) -> Dict[str, torch.Tensor]:
        """
        Step 2: 本地训练
        
        根据配置选择训练策略：
        - STANDARD: 只用交叉熵
        - CONTRASTIVE: L_CE + α||w-w_g||² - β||w-w_old||²
        - FEDPROX: L_CE + μ||w-w_g||²
        """
        self.model.train()
        
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.config.lr,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()
        
        # 保存训练前的模型参数
        init_weights = {k: v.clone() for k, v in self.model.state_dict().items()}
        
        for epoch in range(self.config.local_epochs):
            for data, target in self.dataloader:
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                
                # 基础交叉熵损失
                loss = criterion(output, target)
                
                # 添加正则项
                if self.config.training_strategy == TrainingStrategy.CONTRASTIVE:
                    loss += self._contrastive_regularization()
                elif self.config.training_strategy == TrainingStrategy.FEDPROX:
                    loss += self._fedprox_regularization()
                
                loss.backward()
                optimizer.step()
        
        # 计算模型更新（梯度 = 新参数 - 旧参数）
        gradient = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                gradient[name] = param.data - init_weights[name]
        
        # 更新 old_local_weights（用于下一轮的 Contrastive Loss）
        self.old_local_weights = {k: v.clone() for k, v in self.model.state_dict().items()}
        
        return gradient
    
    def _contrastive_regularization(self) -> torch.Tensor:
        """
        Contrastive Loss 正则项
        
        L_contrastive = α||w - w_g||² - β||w - w_old||²
        
        α 项：拉向全局模型，减少 drift
        β 项：推离上一轮本地模型，保持多样性
        """
        reg = torch.tensor(0.0, device=self.device)
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            # α||w - w_g||²: 全局一致性
            if name in self.global_weights:
                diff_global = param - self.global_weights[name]
                reg += self.config.alpha * diff_global.pow(2).sum()
            
            # -β||w - w_old||²: 局部多样性
            if self.old_local_weights is not None and name in self.old_local_weights:
                diff_old = param - self.old_local_weights[name]
                reg -= self.config.beta * diff_old.pow(2).sum()
        
        return reg
    
    def _fedprox_regularization(self) -> torch.Tensor:
        """
        FedProx 正则项
        
        L_prox = μ||w - w_g||²
        """
        reg = torch.tensor(0.0, device=self.device)
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name in self.global_weights:
                diff = param - self.global_weights[name]
                reg += self.config.mu * diff.pow(2).sum()
        
        return reg
    
    def accumulate_residual(
        self,
        gradient: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Step 3: 残差累积"""
        if self.config.use_residual:
            return self.residual_accumulator.accumulate(gradient)
        return gradient
    
    def compute_importance(
        self,
        gradient: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Step 4: 参数重要性评估"""
        strategy = self.config.importance_strategy
        
        if strategy == ImportanceStrategy.FISHER_GRAD:
            # 计算或使用缓存的 Fisher 信息
            if self.fisher_cache is None:
                self.fisher_cache = compute_fisher_information(
                    self.model, self.dataloader, self.device
                )
            # Fisher × |梯度|
            importance = {}
            for name, grad in gradient.items():
                if name in self.fisher_cache:
                    importance[name] = self.fisher_cache[name] * grad.abs()
                else:
                    importance[name] = grad.abs()
            return importance
        
        elif strategy == ImportanceStrategy.GRAD_SQUARED:
            return {name: grad.pow(2) for name, grad in gradient.items()}
        
        elif strategy == ImportanceStrategy.GRAD_NORMALIZED:
            return compute_importance_grad_normalized(gradient)
        
        else:
            raise ValueError(f"Unknown importance strategy: {strategy}")
    
    def topk_compress(
        self,
        gradient: Dict[str, torch.Tensor],
        importance: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Step 5: Top-k 压缩"""
        strategy = self.config.topk_strategy.value
        k_ratio = self.config.topk_ratio
        weight_method = self.config.layer_weight_method.value
        
        sparse_gradient, masks = compress_gradient(
            gradient=gradient,
            importance_strategy=self.config.importance_strategy.value,
            topk_strategy=strategy,
            k_ratio=k_ratio,
            weight_method=weight_method,
        )
        
        # 更新残差
        if self.config.use_residual:
            self.residual_accumulator.update(gradient, masks)
        
        return sparse_gradient, masks
    
    def clip_gradient(
        self,
        gradient: Dict[str, torch.Tensor],
        clip_norm: float,
    ) -> Tuple[Dict[str, torch.Tensor], float, bool]:
        """
        Step 6: 梯度裁剪
        
        返回：(裁剪后的梯度, L2范数, 是否被裁剪)
        """
        # 计算全局 L2 范数
        total_norm_sq = sum(g.pow(2).sum() for g in gradient.values())
        total_norm = total_norm_sq.sqrt().item()
        
        clipped = total_norm > clip_norm
        
        if clipped:
            scale = clip_norm / total_norm
            clipped_gradient = {name: g * scale for name, g in gradient.items()}
        else:
            clipped_gradient = gradient
        
        return clipped_gradient, total_norm, clipped
    
    def add_dp_noise(
        self,
        gradient: Dict[str, torch.Tensor],
        masks: Dict[str, torch.Tensor],
        importance: Dict[str, torch.Tensor],
        epsilon_round: float,
        clip_norm: float,
        sampling_rate: float,
    ) -> tuple:
        """
        Step 7: 差分隐私加噪
        
        支持同构噪声和异构噪声两种模式
        
        返回：(加噪后的梯度, 噪声强度sigma)
        """
        if not self.dp_config.enabled:
            return gradient, 0.0
        
        # 计算稀疏率
        total_params = sum(g.numel() for g in gradient.values())
        selected_params = sum(m.sum().item() for m in masks.values())
        sparsity_rate = selected_params / total_params if total_params > 0 else 1.0
        
        if self.dp_config.use_heterogeneous_noise:
            # 异构噪声模式
            noisy_grad, avg_sigma = self._add_heterogeneous_noise(
                gradient, masks, importance,
                epsilon_round, clip_norm
            )
            return noisy_grad, avg_sigma
        else:
            # 同构噪声模式
            sigma = compute_base_noise_std(
                epsilon=epsilon_round,
                delta=self.dp_config.delta,
                clip_norm=clip_norm,
                sampling_rate=sampling_rate,
                sparsity_rate=sparsity_rate,
                use_subsampling_amplification=self.dp_config.use_subsampling_amplification,
                use_sparsity_amplification=self.dp_config.use_sparsity_amplification,
            )
            
            noisy_gradient = {}
            for name, grad in gradient.items():
                noisy_gradient[name] = add_noise_to_gradient(
                    grad, sigma, generator=self.rng
                )
            return noisy_gradient, sigma
    
    def _add_heterogeneous_noise(
        self,
        gradient: Dict[str, torch.Tensor],
        masks: Dict[str, torch.Tensor],
        importance: Dict[str, torch.Tensor],
        epsilon_total: float,
        clip_norm: float,
    ) -> tuple:
        """
        异构噪声添加（跨层统一分配隐私预算）
        
        返回：(加噪后的梯度, 平均噪声强度)
        """
        # Step 1: 收集所有层的信息
        all_grads = []
        all_masks = []
        all_imps = []
        layer_info = []  # 记录每层的起始位置和长度
        
        offset = 0
        for name, grad in gradient.items():
            if name not in masks:
                continue
            
            flat_grad = grad.flatten()
            flat_mask = masks[name].flatten()
            flat_imp = importance.get(name, grad.abs()).flatten()
            
            all_grads.append(flat_grad)
            all_masks.append(flat_mask)
            all_imps.append(flat_imp)
            
            layer_info.append({
                "name": name,
                "shape": grad.shape,
                "start": offset,
                "length": flat_grad.numel()
            })
            offset += flat_grad.numel()
        
        if not all_grads:
            return gradient, 0.0
        
        # Step 2: 拼接成全局张量
        global_grad = torch.cat(all_grads)
        global_mask = torch.cat(all_masks)
        global_imp = torch.cat(all_imps)
        
        # 计算选中参数数量
        num_selected = global_mask.sum().item()
        if num_selected == 0:
            return gradient, 0.0
        
        # Step 3: 统一分配隐私预算（所有参数共享 epsilon_total）
        epsilon_i = allocate_heterogeneous_epsilon(
            global_imp, global_mask, epsilon_total
        )
        
        # Step 4: 计算噪声标准差
        sigma_i = compute_heterogeneous_noise_std(
            epsilon_i, self.dp_config.delta, clip_norm
        )
        
        # Step 5: 添加噪声
        global_noisy = add_heterogeneous_noise(
            global_grad, sigma_i, generator=self.rng
        )
        
        # Step 6: 拆分回各层
        noisy_gradient = {}
        for info in layer_info:
            name = info["name"]
            start = info["start"]
            length = info["length"]
            shape = info["shape"]
            
            flat_noisy = global_noisy[start:start + length]
            noisy_gradient[name] = flat_noisy.reshape(shape)
        
        # 对于没有 mask 的层，保持原样
        for name, grad in gradient.items():
            if name not in noisy_gradient:
                noisy_gradient[name] = grad
        
        # 计算平均噪声强度
        selected_sigmas = sigma_i[global_mask > 0]
        avg_sigma = selected_sigmas.mean().item() if selected_sigmas.numel() > 0 else 0.0
        
        return noisy_gradient, avg_sigma

    def train_and_upload(
        self,
        global_weights: Dict[str, torch.Tensor],
        clip_norm: float,
        epsilon_round: float,
        sampling_rate: float,
    ) -> ClientUpdate:
        """
        完整的训练流程
        
        执行 Step 1-8，返回上传内容
        """
        # Step 1: 接收全局模型
        self.receive_global_model(global_weights)
        
        # Step 2: 本地训练
        gradient = self.local_train()
        
        # Step 3: 残差累积
        gradient = self.accumulate_residual(gradient)
        
        # Step 4: 重要性评估
        importance = self.compute_importance(gradient)
        
        # Step 5: Top-k 压缩
        sparse_gradient, masks = self.topk_compress(gradient, importance)
        
        # Step 6: 梯度裁剪
        clipped_gradient, l2_norm, clipped = self.clip_gradient(
            sparse_gradient, clip_norm
        )
        
        # Step 7: 差分隐私加噪
        noisy_gradient, noise_sigma = self.add_dp_noise(
            clipped_gradient, masks, importance,
            epsilon_round, clip_norm, sampling_rate
        )
        
        # Step 8: 构建上传内容
        stat = 1.0 if clipped else 0.0  # 二值统计
        if self.config.stat_type.value == "l2_norm":
            stat = l2_norm  # 实际 L2 范数
        
        return ClientUpdate(
            gradient=noisy_gradient,
            data_size=self.data_size,
            stat=stat,
            clipped=clipped,
            noise_sigma=noise_sigma,
        )


# ══════════════════════════════════════════════════════════════════════════════
# 快速验证
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    
    print("=== 客户端模块测试 ===\n")
    
    # 创建简单模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(32, 16)
            self.fc2 = nn.Linear(16, 10)
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            return self.fc2(x)
    
    device = torch.device("cpu")
    model = SimpleModel()
    
    # 创建假数据
    X = torch.randn(100, 32)
    y = torch.randint(0, 10, (100,))
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    
    # 配置
    config = ClientConfig()
    config.training_strategy = TrainingStrategy.CONTRASTIVE
    config.topk_ratio = 0.3
    
    dp_config = DPConfig()
    dp_config.enabled = True
    dp_config.use_heterogeneous_noise = True
    
    # 创建客户端
    client = FLClient(
        client_id=0,
        model=model,
        dataloader=dataloader,
        config=config,
        dp_config=dp_config,
        device=device,
    )
    
    # 模拟全局模型
    global_weights = model.state_dict()
    
    # 执行训练
    update = client.train_and_upload(
        global_weights=global_weights,
        clip_norm=1.0,
        epsilon_round=0.8,
        sampling_rate=0.1,
    )
    
    print(f"客户端 {client.client_id} 训练完成:")
    print(f"  数据量: {update.data_size}")
    print(f"  是否被裁剪: {update.clipped}")
    print(f"  统计值: {update.stat:.4f}")
    print(f"  梯度层数: {len(update.gradient)}")
    for name, grad in update.gradient.items():
        print(f"    {name}: {grad.shape}, 非零率: {(grad != 0).float().mean():.2%}")
    
    print("\n✓ 客户端测试通过")
