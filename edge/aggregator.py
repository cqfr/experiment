"""
edge/aggregator.py
Edge 聚合器（可信）
实现：收集客户端更新、加权聚合、统计信息处理
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import sys
sys.path.append('..')

from config import EdgeConfig, StatsAggMethod


@dataclass
class EdgeOutput:
    """Edge 上传到 Server 的内容"""
    global_gradient: Dict[str, torch.Tensor]    # 聚合后的全局梯度
    stats_aggregated: Dict[str, float]          # 聚合后的统计信息


class TrustedEdge:
    """
    可信 Edge 聚合器
    
    职责：
    1. 收集所有参与客户端的更新
    2. 使用真实数据集大小进行加权聚合
    3. 处理和聚合统计信息（保护隐私）
    4. 上传聚合结果到 Server
    """
    
    def __init__(self, config: EdgeConfig):
        self.config = config
    
    def aggregate(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        data_sizes: List[int],
        stats: List[float],
    ) -> EdgeOutput:
        """
        完整聚合流程
        
        参数：
            client_updates: 各客户端的梯度更新列表
            data_sizes: 各客户端的数据集大小
            stats: 各客户端的 L2 范数统计
        
        返回：
            EdgeOutput: 聚合后的全局梯度和统计信息
        """
        # Step 9: 收集客户端信息（已完成，作为参数传入）
        
        # Step 10: 加权聚合
        global_gradient = self._weighted_aggregate(client_updates, data_sizes)
        
        # Step 11: 处理统计信息
        stats_aggregated = self._aggregate_stats(stats)
        
        return EdgeOutput(
            global_gradient=global_gradient,
            stats_aggregated=stats_aggregated,
        )
    
    def _weighted_aggregate(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        data_sizes: List[int],
    ) -> Dict[str, torch.Tensor]:
        """
        Step 10: 加权聚合
        
        g_global = Σ (|D_i| / C_total) × ∇w_i
        
        Edge 是可信的，可以使用真实的数据集大小进行加权
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
        Step 11: 处理统计信息
        
        选项 A: 分位数法 - 只上传 {median, q25, q75}
        选项 B: 打乱法 - 打乱后上传所有统计值
        """
        if len(stats) == 0:
            return {"median": 0.0, "q25": 0.0, "q75": 0.0, "count": 0}
        
        stats_array = np.array(stats)
        
        if self.config.stats_agg_method == StatsAggMethod.QUANTILE:
            # 选项 A: 分位数法（隐私保护更强）
            return {
                "median": float(np.quantile(stats_array, 0.5)),
                "q25": float(np.quantile(stats_array, 0.25)),
                "q75": float(np.quantile(stats_array, 0.75)),
                "count": len(stats),
                "fraction_clipped": float(np.mean(stats_array > 0.5)),  # 二值统计时
            }
        
        elif self.config.stats_agg_method == StatsAggMethod.SHUFFLE:
            # 选项 B: 打乱法（保留更多信息）
            shuffled = stats_array.copy()
            np.random.shuffle(shuffled)
            return {
                "median": float(np.median(shuffled)),
                "q25": float(np.quantile(shuffled, 0.25)),
                "q75": float(np.quantile(shuffled, 0.75)),
                "count": len(stats),
                "shuffled_stats": shuffled.tolist(),  # 打乱后的完整列表
                "fraction_clipped": float(np.mean(stats_array > 0.5)),
            }
        
        else:
            raise ValueError(f"Unknown stats aggregation method: {self.config.stats_agg_method}")


class SimpleAggregator:
    """
    简单聚合器（用于 FedAvg 等不使用 Edge 的场景）
    
    直接进行等权重或数据量加权的聚合
    """
    
    @staticmethod
    def fedavg_aggregate(
        client_updates: List[Dict[str, torch.Tensor]],
        data_sizes: Optional[List[int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        FedAvg 聚合
        
        如果提供 data_sizes，使用数据量加权
        否则使用等权重
        """
        if len(client_updates) == 0:
            return {}
        
        num_clients = len(client_updates)
        
        # 确定权重
        if data_sizes is not None:
            total_data = sum(data_sizes)
            weights = [size / total_data for size in data_sizes]
        else:
            weights = [1.0 / num_clients] * num_clients
        
        # 聚合
        aggregated = {}
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
                aggregated[name] = weighted_sum
        
        return aggregated


# ══════════════════════════════════════════════════════════════════════════════
# 快速验证
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== Edge 聚合器测试 ===\n")
    
    # 创建模拟的客户端更新
    num_clients = 5
    client_updates = []
    data_sizes = []
    stats = []
    
    for i in range(num_clients):
        update = {
            "layer1.weight": torch.randn(64, 32) * (i + 1) * 0.1,
            "layer1.bias": torch.randn(64) * (i + 1) * 0.1,
            "layer2.weight": torch.randn(10, 64) * (i + 1) * 0.1,
            "layer2.bias": torch.randn(10) * (i + 1) * 0.1,
        }
        client_updates.append(update)
        data_sizes.append(100 + i * 50)  # 不同的数据量
        stats.append(np.random.random() * 2)  # 随机 L2 范数
    
    print(f"客户端数: {num_clients}")
    print(f"数据量: {data_sizes}")
    print(f"统计值: {[f'{s:.3f}' for s in stats]}")
    
    # 测试分位数聚合
    print("\n=== 分位数聚合 ===")
    config_quantile = EdgeConfig(stats_agg_method=StatsAggMethod.QUANTILE)
    edge_quantile = TrustedEdge(config_quantile)
    
    output = edge_quantile.aggregate(client_updates, data_sizes, stats)
    
    print(f"聚合后的梯度层数: {len(output.global_gradient)}")
    for name, grad in output.global_gradient.items():
        print(f"  {name}: {grad.shape}, 范数: {grad.norm():.4f}")
    
    print(f"\n统计信息:")
    for key, value in output.stats_aggregated.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # 测试打乱聚合
    print("\n=== 打乱聚合 ===")
    config_shuffle = EdgeConfig(stats_agg_method=StatsAggMethod.SHUFFLE)
    edge_shuffle = TrustedEdge(config_shuffle)
    
    output_shuffle = edge_shuffle.aggregate(client_updates, data_sizes, stats)
    print(f"统计信息:")
    for key, value in output_shuffle.stats_aggregated.items():
        if key == "shuffled_stats":
            print(f"  {key}: {[f'{v:.3f}' for v in value]}")
        elif isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # 测试简单聚合器
    print("\n=== FedAvg 聚合 ===")
    fedavg_result = SimpleAggregator.fedavg_aggregate(client_updates, data_sizes)
    print(f"FedAvg 聚合结果层数: {len(fedavg_result)}")
    
    # 验证加权正确性
    print("\n=== 加权验证 ===")
    for name in ["layer1.bias"]:
        # 手动计算加权平均
        total_data = sum(data_sizes)
        manual_avg = sum(
            client_updates[i][name] * (data_sizes[i] / total_data)
            for i in range(num_clients)
        )
        
        # 比较
        diff = (output.global_gradient[name] - manual_avg).abs().max()
        print(f"{name}: 差异 = {diff:.10f} (应该接近 0)")
    
    print("\n✓ Edge 聚合器测试通过")
