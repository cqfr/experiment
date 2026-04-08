"""
server/server.py
联邦学习服务器
实现：聚合客户端更新、全局模型更新、裁剪阈值调整、模型分发
"""

import copy
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import sys
sys.path.append('..')

from config import ServerConfig, ClipUpdateMethod, DownlinkStrategy


@dataclass
class AggregatedResult:
    """聚合结果"""
    global_gradient: Dict[str, torch.Tensor]    # 聚合后的全局梯度
    stats_aggregated: Dict[str, float]          # 聚合后的统计信息


class FLServer:
    """
    联邦学习服务器
    
    职责：
    1. 维护全局模型
    2. 接收 Edge 聚合的更新
    3. 更新全局模型
    4. 动态调整裁剪阈值
    5. 分发模型到客户端（可选稀疏化）
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: ServerConfig,
        device: torch.device,
    ):
        self.config = config
        self.device = device
        
        # 全局模型
        self.global_model = copy.deepcopy(model).to(device)
        
        # 裁剪阈值
        self.clip_norm = config.initial_clip
        
        # 历史统计（用于 EMA 等）
        self.clip_history: List[float] = []
        self.stats_history: List[Dict[str, float]] = []
        
        # 当前轮次
        self.current_round = 0
    
    def get_global_weights(self) -> Dict[str, torch.Tensor]:
        """获取全局模型参数"""
        return copy.deepcopy(self.global_model.state_dict())
    
    def get_clip_norm(self) -> float:
        """获取当前裁剪阈值"""
        return self.clip_norm
    
    def aggregate(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        data_sizes: List[int],
        stats: List[float],
    ) -> AggregatedResult:
        """
        聚合客户端更新
        
        使用数据量加权：g_global = Σ (n_i / Σn_j) × Δw_i
        
        参数：
            client_updates: 各客户端的梯度更新列表
            data_sizes: 各客户端的数据集大小
            stats: 各客户端的 L2 范数统计（用于自适应裁剪）
        
        返回：
            AggregatedResult: 聚合后的全局梯度和统计信息
        """
        # 加权聚合
        global_gradient = self._weighted_aggregate(client_updates, data_sizes)
        
        # 处理统计信息
        stats_aggregated = self._aggregate_stats(stats)
        
        return AggregatedResult(
            global_gradient=global_gradient,
            stats_aggregated=stats_aggregated,
        )
    
    def _weighted_aggregate(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        data_sizes: List[int],
    ) -> Dict[str, torch.Tensor]:
        """
        数据量加权聚合
        
        g_global = Σ (n_i / Σn_j) × Δw_i
        """
        if len(client_updates) == 0:
            return {}
        
        # 计算总数据量
        total_data = sum(data_sizes)
        
        # 计算权重
        weights = [size / total_data for size in data_sizes]
        
        # 加权聚合
        global_gradient = {}
        param_names = client_updates[0].keys()
        
        for name in param_names:
            weighted_sum = None
            for i, update in enumerate(client_updates):
                if name not in update:
                    continue
                
                weighted_grad = update[name] * weights[i]
                
                if weighted_sum is None:
                    weighted_sum = weighted_grad.clone()
                else:
                    weighted_sum += weighted_grad
            
            if weighted_sum is not None:
                global_gradient[name] = weighted_sum
        
        return global_gradient
    
    def _aggregate_stats(
        self,
        stats: List[float],
    ) -> Dict[str, float]:
        """
        处理统计信息（用于自适应裁剪）
        
        返回：median, q25, q75, fraction_clipped 等
        """
        if len(stats) == 0:
            return {"median": 0.0, "q25": 0.0, "q75": 0.0, "count": 0, "fraction_clipped": 0.0}
        
        stats_array = np.array(stats)
        
        return {
            "median": float(np.quantile(stats_array, 0.5)),
            "q25": float(np.quantile(stats_array, 0.25)),
            "q75": float(np.quantile(stats_array, 0.75)),
            "count": len(stats),
            "fraction_clipped": float(np.mean(stats_array > 0.5)),  # 二值统计时
        }
    
    def update_global_model(
        self,
        global_gradient: Dict[str, torch.Tensor],
        stats_aggregated: Dict[str, float],
    ) -> Dict[str, float]:
        """
        更新全局模型
        
        Step 13: w_global_{t+1} = w_global_t + η × g_global
        Step 14: 更新裁剪阈值
        
        返回：本轮的训练指标
        """
        # Step 13: 更新全局模型
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                if name in global_gradient:
                    param.data += self.config.server_lr * global_gradient[name].to(self.device)
        
        # Step 14: 更新裁剪阈值
        old_clip = self.clip_norm
        self._update_clip_norm(stats_aggregated)
        
        # 记录历史
        self.stats_history.append(stats_aggregated)
        self.clip_history.append(self.clip_norm)
        self.current_round += 1
        
        return {
            "round": self.current_round,
            "old_clip": old_clip,
            "new_clip": self.clip_norm,
            "stats_median": stats_aggregated.get("median", 0.0),
            "fraction_clipped": stats_aggregated.get("fraction_clipped", 0.0),
        }
    
    def _update_clip_norm(self, stats: Dict[str, float]):
        """
        Step 14: 更新裁剪阈值
        
        选项 A: 自适应裁剪（Adaptive Clipping）
        选项 B: EMA 平滑
        选项 C: 分位数跟踪
        """
        method = self.config.clip_update_method
        
        if method == ClipUpdateMethod.ADAPTIVE:
            # 选项 A: 自适应裁剪
            fraction_clipped = stats.get("fraction_clipped", 0.5)
            
            if fraction_clipped > self.config.target_quantile:
                # 太多客户端被裁剪，增大阈值
                self.clip_norm *= (1 + self.config.clip_lr)
            else:
                # 较少客户端被裁剪，减小阈值
                self.clip_norm *= (1 - self.config.clip_lr)
        
        elif method == ClipUpdateMethod.EMA:
            # 选项 B: EMA 平滑
            median = stats.get("median", self.clip_norm)
            alpha = self.config.ema_alpha
            self.clip_norm = alpha * self.clip_norm + (1 - alpha) * median
        
        elif method == ClipUpdateMethod.QUANTILE:
            # 选项 C: 分位数跟踪
            # 直接使用目标分位数对应的值作为裁剪阈值
            # 这需要 Edge 上传打乱后的统计值
            if "shuffled_stats" in stats:
                shuffled = stats["shuffled_stats"]
                self.clip_norm = np.quantile(shuffled, self.config.target_quantile)
            else:
                # 回退到使用中位数
                median = stats.get("median", self.clip_norm)
                self.clip_norm = median
        
        # 限制裁剪阈值的范围
        self.clip_norm = max(0.01, min(100.0, self.clip_norm))
    
    def prepare_broadcast(self) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        """
        Step 15: 准备模型分发
        
        选项 A: 下发完整模型
        选项 B: 下发 Top-k 更新（需要上一轮的模型）
        
        返回：(要分发的模型权重, 可选的稀疏更新掩码)
        """
        if self.config.downlink_strategy == DownlinkStrategy.FULL:
            # 选项 A: 下发完整模型
            return self.get_global_weights(), None
        
        elif self.config.downlink_strategy == DownlinkStrategy.TOPK:
            # 选项 B: 下发 Top-k 更新
            # 注意：这需要计算相对于上一轮的更新
            # 这里简化处理，直接对模型参数做 Top-k
            weights = self.get_global_weights()
            masks = {}
            
            # 计算全局 Top-k
            all_values = torch.cat([w.abs().flatten() for w in weights.values()])
            k = int(all_values.numel() * self.config.downlink_topk_ratio)
            threshold = torch.topk(all_values, k).values[-1]
            
            sparse_weights = {}
            for name, w in weights.items():
                mask = (w.abs() >= threshold).float()
                masks[name] = mask
                sparse_weights[name] = w * mask
            
            return sparse_weights, masks
        
        else:
            return self.get_global_weights(), None
    
    def evaluate(
        self,
        test_loader: torch.utils.data.DataLoader,
    ) -> Dict[str, float]:
        """
        评估全局模型
        
        返回：准确率、损失等指标
        """
        self.global_model.eval()
        criterion = nn.CrossEntropyLoss()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                
                loss = criterion(output, target)
                total_loss += loss.item() * data.size(0)
                
                _, predicted = output.max(1)
                correct += predicted.eq(target).sum().item()
                total += data.size(0)
        
        self.global_model.train()
        
        return {
            "loss": total_loss / total,
            "accuracy": correct / total,
            "correct": correct,
            "total": total,
        }


# ══════════════════════════════════════════════════════════════════════════════
# 快速验证
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== Server 模块测试 ===\n")
    
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
    
    # 测试不同的裁剪更新策略
    strategies = [
        ClipUpdateMethod.ADAPTIVE,
        ClipUpdateMethod.EMA,
        ClipUpdateMethod.QUANTILE,
    ]
    
    for strategy in strategies:
        print(f"\n=== 测试 {strategy.value} ===")
        
        config = ServerConfig(
            initial_clip=1.0,
            clip_update_method=strategy,
        )
        
        server = FLServer(model, config, device)
        
        # 模拟多轮训练
        for round_num in range(5):
            # 创建假的聚合梯度
            global_gradient = {
                "fc1.weight": torch.randn(16, 32) * 0.1,
                "fc1.bias": torch.randn(16) * 0.1,
                "fc2.weight": torch.randn(10, 16) * 0.1,
                "fc2.bias": torch.randn(10) * 0.1,
            }
            
            # 创建假的统计信息
            stats = {
                "median": np.random.uniform(0.5, 2.0),
                "q25": np.random.uniform(0.3, 0.8),
                "q75": np.random.uniform(1.5, 3.0),
                "fraction_clipped": np.random.uniform(0.3, 0.7),
                "count": 10,
                "shuffled_stats": np.random.uniform(0.5, 2.0, 10).tolist(),
            }
            
            # 更新模型
            metrics = server.update_global_model(global_gradient, stats)
            
            print(f"  Round {metrics['round']}: "
                  f"clip {metrics['old_clip']:.3f} → {metrics['new_clip']:.3f}, "
                  f"clipped {metrics['fraction_clipped']:.1%}")
    
    # 测试下行通信策略
    print("\n=== 下行通信策略测试 ===")
    
    # 完整模型
    config_full = ServerConfig(downlink_strategy=DownlinkStrategy.FULL)
    server_full = FLServer(model, config_full, device)
    weights_full, mask_full = server_full.prepare_broadcast()
    print(f"完整模型: {len(weights_full)} 层, 掩码: {mask_full}")
    
    # Top-k 稀疏
    config_topk = ServerConfig(
        downlink_strategy=DownlinkStrategy.TOPK,
        downlink_topk_ratio=0.5,
    )
    server_topk = FLServer(model, config_topk, device)
    weights_sparse, masks = server_topk.prepare_broadcast()
    
    total_params = sum(w.numel() for w in weights_sparse.values())
    non_zero = sum((w != 0).sum().item() for w in weights_sparse.values())
    print(f"Top-k 稀疏: {len(weights_sparse)} 层, 非零率: {non_zero/total_params:.1%}")
    
    # 测试评估
    print("\n=== 评估测试 ===")
    from torch.utils.data import TensorDataset, DataLoader
    
    X_test = torch.randn(100, 32)
    y_test = torch.randint(0, 10, (100,))
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    metrics = server.evaluate(test_loader)
    print(f"测试结果: Loss={metrics['loss']:.4f}, Acc={metrics['accuracy']:.2%}")
    
    print("\n✓ Server 模块测试通过")
