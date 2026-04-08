"""
config.py
联邦学习实验配置 - 所有超参数和策略选项集中管理
"""

from dataclasses import dataclass, field
from typing import Literal
from enum import Enum

# ══════════════════════════════════════════════════════════════════════════════
# 策略枚举定义
# ══════════════════════════════════════════════════════════════════════════════

class TrainingStrategy(Enum):
    """本地训练策略"""
    STANDARD = "standard"           # 只用交叉熵损失
    CONTRASTIVE = "contrastive"     # L_CE + α||w-w_g||² - β||w-w_old||²
    FEDPROX = "fedprox"             # L_CE + μ||w-w_g||²


class ImportanceStrategy(Enum):
    """参数重要性评估策略"""
    FISHER_GRAD = "fisher_grad"           # Fisher信息 × |梯度|
    GRAD_SQUARED = "grad_squared"         # |梯度| × |梯度|
    GRAD_NORMALIZED = "grad_normalized"   # |梯度|（层级归一化）


class TopKStrategy(Enum):
    """Top-k 选择策略"""
    GLOBAL_TOPK = "global_topk"                       # 全局归一化 + 全局Top-k
    LAYER_TOPK = "layer_topk"                         # 层级Top-k
    LAYER_NORM_GLOBAL = "layer_norm_global"           # 层级归一化 + 全局Top-k
    WEIGHTED_LAYER_NORM = "weighted_layer_norm"       # 加权层级归一化 + 全局Top-k


class LayerWeightMethod(Enum):
    """层权重计算方法（用于 WEIGHTED_LAYER_NORM）"""
    MEAN = "mean"
    MEDIAN = "median"
    TOTAL_SUM = "total_sum"
    TRIMMED_MEAN = "trimmed_mean"


class StatType(Enum):
    """裁剪统计类型"""
    BINARY = "binary"       # 二值标志：是否被裁剪
    L2_NORM = "l2_norm"     # 实际L2范数值


class StatsAggMethod(Enum):
    """Edge端统计聚合方法"""
    QUANTILE = "quantile"   # 只上传分位数
    SHUFFLE = "shuffle"     # 打乱后上传


class ClipUpdateMethod(Enum):
    """Server端裁剪阈值更新方法"""
    ADAPTIVE = "adaptive"   # 自适应裁剪
    EMA = "ema"             # EMA平滑
    QUANTILE = "quantile"   # 分位数跟踪


class DownlinkStrategy(Enum):
    """下行通信策略"""
    FULL = "full"           # 下发完整模型
    TOPK = "topk"           # 下发Top-k更新


# ══════════════════════════════════════════════════════════════════════════════
# 配置类
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ClientConfig:
    """客户端配置"""
    # 本地训练
    training_strategy: TrainingStrategy = TrainingStrategy.CONTRASTIVE
    local_epochs: int = 5
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 1e-4
    
    # Contrastive Loss 超参数
    alpha: float = 0.005    # 全局一致性权重 ||w - w_g||²
    beta: float = 0.005     # 局部多样性权重 ||w - w_old||²
    
    # FedProx 超参数
    mu: float = 0.01        # 正则项权重
    
    # 重要性评估
    importance_strategy: ImportanceStrategy = ImportanceStrategy.GRAD_NORMALIZED
    
    # Top-k 压缩
    topk_strategy: TopKStrategy = TopKStrategy.WEIGHTED_LAYER_NORM
    topk_ratio: float = 0.1    # 压缩率 k%
    layer_weight_method: LayerWeightMethod = LayerWeightMethod.MEAN
    
    # 梯度裁剪
    stat_type: StatType = StatType.L2_NORM
    
    # 残差累积
    use_residual: bool = True


@dataclass
class EdgeConfig:
    """Edge聚合器配置"""
    stats_agg_method: StatsAggMethod = StatsAggMethod.QUANTILE


@dataclass
class ServerConfig:
    """服务器配置"""
    # 模型更新
    server_lr: float = 1.0   # 全局更新学习率
    
    # 裁剪阈值
    initial_clip: float = 1.0
    clip_update_method: ClipUpdateMethod = ClipUpdateMethod.ADAPTIVE
    target_quantile: float = 0.5    # 自适应裁剪目标分位数
    clip_lr: float = 0.2            # 裁剪阈值调整步长
    ema_alpha: float = 0.8          # EMA 平滑系数
    
    # 下行通信
    downlink_strategy: DownlinkStrategy = DownlinkStrategy.FULL
    downlink_topk_ratio: float = 0.5   # 下行 Top-k 比例


@dataclass
class DPConfig:
    """差分隐私配置"""
    enabled: bool = True
    
    # 隐私预算
    epsilon_total: float = 8.0      # 总隐私预算
    delta: float = 1e-5             # δ 参数
    
    # 噪声计算
    use_subsampling_amplification: bool = True    # 子采样隐私放大
    use_sparsity_amplification: bool = True       # 稀疏化隐私放大
    
    # 异构噪声
    use_heterogeneous_noise: bool = True


@dataclass
class ExperimentConfig:
    """实验整体配置"""
    # 基础设置
    seed: int = 42
    num_clients: int = 100
    clients_per_round: int = 10
    num_rounds: int = 100
    batch_size: int = 32
    
    # 数据集
    dataset: str = "cifar10"
    iid: bool = False
    alpha: float = 0.5    # Dirichlet 分布参数
    
    # 模型
    model: str = "resnet18"
    num_classes: int = 10
    
    # 各组件配置
    client: ClientConfig = field(default_factory=ClientConfig)
    edge: EdgeConfig = field(default_factory=EdgeConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    dp: DPConfig = field(default_factory=DPConfig)
    
    # 日志与保存
    log_interval: int = 1
    save_interval: int = 10
    save_dir: str = "./checkpoints"
    log_dir: str = "./logs"


# ══════════════════════════════════════════════════════════════════════════════
# 预设配置（快速实验用）
# ══════════════════════════════════════════════════════════════════════════════

def get_fedavg_config() -> ExperimentConfig:
    """FedAvg 基线配置"""
    config = ExperimentConfig()
    config.client.training_strategy = TrainingStrategy.STANDARD
    config.client.topk_ratio = 1.0      # 不压缩
    config.client.use_residual = False
    config.dp.enabled = False
    config.server.initial_clip = 1000
    # FedAvg 使用固定裁剪阈值（不自适应）
    config.server.clip_update_method = ClipUpdateMethod.EMA
    config.server.ema_alpha = 1.0  # alpha=1 表示完全保持原值，不更新
    return config


def get_dp_fedavg_config(epsilon: float = 8.0) -> ExperimentConfig:
    """DP-FedAvg 基线配置"""
    config = ExperimentConfig()
    config.client.training_strategy = TrainingStrategy.STANDARD
    config.client.topk_ratio = 1.0
    config.client.use_residual = False
    config.dp.enabled = True
    config.dp.epsilon_total = epsilon
    config.dp.use_heterogeneous_noise = False
    # DP-FedAvg 使用固定裁剪阈值（不自适应）
    config.server.clip_update_method = ClipUpdateMethod.EMA
    config.server.ema_alpha = 1.0  # 固定阈值
    return config


def get_proposed_config(epsilon: float = 8.0) -> ExperimentConfig:
    """我们提出的方法配置"""
    config = ExperimentConfig()
    config.client.training_strategy = TrainingStrategy.CONTRASTIVE
    config.client.importance_strategy = ImportanceStrategy.GRAD_NORMALIZED
    config.client.topk_strategy = TopKStrategy.WEIGHTED_LAYER_NORM
    config.client.topk_ratio = 0.1
    config.client.use_residual = True
    config.dp.enabled = True
    config.dp.epsilon_total = epsilon
    config.dp.use_heterogeneous_noise = True
    # 我们的方法使用自适应裁剪
    config.server.clip_update_method = ClipUpdateMethod.ADAPTIVE
    return config


if __name__ == "__main__":
    # 测试配置
    config = get_proposed_config()
    print(f"实验配置:")
    print(f"  客户端数: {config.num_clients}")
    print(f"  每轮参与: {config.clients_per_round}")
    print(f"  总轮数: {config.num_rounds}")
    print(f"  训练策略: {config.client.training_strategy.value}")
    print(f"  Top-k 策略: {config.client.topk_strategy.value}")
    print(f"  压缩率: {config.client.topk_ratio}")
    print(f"  DP 隐私预算: ε={config.dp.epsilon_total}")
