from __future__ import annotations

"""Configuration objects and experiment presets."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class TrainingStrategy(Enum):
    """Local optimization strategy on each client."""

    STANDARD = "standard"
    CONTRASTIVE = "contrastive"
    FEDPROX = "fedprox"
    DP_FEDSAM = "dp_fedsam"


class ImportanceStrategy(Enum):
    """Importance score used before sparse transmission."""

    FISHER_GRAD = "fisher_grad"
    GRAD_SQUARED = "grad_squared"
    GRAD_NORMALIZED = "grad_normalized"


class TopKStrategy(Enum):
    """Top-k selection strategy."""

    GLOBAL_TOPK = "global_topk"
    LAYER_TOPK = "layer_topk"
    LAYER_NORM_GLOBAL = "layer_norm_global"
    WEIGHTED_LAYER_NORM = "weighted_layer_norm"


class LayerWeightMethod(Enum):
    """Layer scoring method for weighted layer-normalized top-k."""

    MEAN = "mean"
    MEDIAN = "median"
    TOTAL_SUM = "total_sum"
    TRIMMED_MEAN = "trimmed_mean"


class StatType(Enum):
    """Statistics uploaded by clients for clipping control."""

    BINARY = "binary"
    L2_NORM = "l2_norm"


class StatsAggMethod(Enum):
    """How server-side statistics are aggregated."""

    QUANTILE = "quantile"
    ALL = "all"


class ClipUpdateMethod(Enum):
    """How clipping norm is updated round-by-round."""

    ADAPTIVE = "adaptive"
    EMA = "ema"


class DownlinkStrategy(Enum):
    """Server downlink payload type."""

    FULL = "full"
    TOPK = "topk"


@dataclass
class ClientConfig:
    """Client-side optimization and compression options."""

    training_strategy: TrainingStrategy = TrainingStrategy.CONTRASTIVE
    local_epochs: int = 5
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 1e-4

    # Contrastive regularizer: alpha*||w-wg||^2 + beta*max(0,m-||w-w_old||)^2
    alpha: float = 0.005
    beta: float = 0.005
    contrastive_margin: float = 1.0

    # FedProx regularizer weight.
    mu: float = 0.01

    # SAM-related hyper-parameters used by DP-FedSAM baseline.
    sam_rho: float = 0.05
    sam_eps: float = 1e-12

    importance_strategy: ImportanceStrategy = ImportanceStrategy.GRAD_NORMALIZED

    topk_strategy: TopKStrategy = TopKStrategy.WEIGHTED_LAYER_NORM
    topk_ratio: float = 0.1
    layer_weight_method: LayerWeightMethod = LayerWeightMethod.MEAN

    stat_type: StatType = StatType.L2_NORM
    use_residual: bool = True


@dataclass
class EdgeConfig:
    """Edge/Server statistic aggregation behavior."""

    stats_agg_method: StatsAggMethod = StatsAggMethod.QUANTILE


@dataclass
class ServerConfig:
    """Server-side optimization and clipping controls."""

    server_lr: float = 1.0

    initial_clip: float = 1.0
    clip_update_method: ClipUpdateMethod = ClipUpdateMethod.ADAPTIVE
    target_quantile: float = 0.5
    clip_lr: float = 0.2
    ema_alpha: float = 0.8

    downlink_strategy: DownlinkStrategy = DownlinkStrategy.FULL
    downlink_topk_ratio: float = 0.5


@dataclass
class DPConfig:
    """Differential privacy options."""

    enabled: bool = True
    epsilon_total: float = 8.0
    delta: float = 1e-5

    # Use relative heterogenous noise and scale to satisfy the same round target.
    use_heterogeneous_noise: bool = True
    min_relative_noise: float = 0.3
    max_relative_noise: float = 3.0

    # RDP accountant settings.
    rdp_orders: tuple[float, ...] = (
        2,
        3,
        4,
        5,
        8,
        16,
        32,
        64,
        128,
        256,
    )
    rdp_steps_per_round: int = 1

    # Explicit modeling assumption accepted by the user.
    trusted_server_for_stats: bool = True


@dataclass
class ExperimentConfig:
    """End-to-end experiment configuration."""

    seed: int = 42
    num_clients: int = 100
    clients_per_round: int = 10
    num_rounds: int = 100
    batch_size: int = 32

    dataset: Literal["cifar10", "mnist"] = "cifar10"
    iid: bool = False
    alpha: float = 0.5

    model: str = "resnet18"
    num_classes: int = 10

    client: ClientConfig = field(default_factory=ClientConfig)
    edge: EdgeConfig = field(default_factory=EdgeConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    dp: DPConfig = field(default_factory=DPConfig)

    log_interval: int = 1
    save_interval: int = 10
    save_dir: str = "./checkpoints"
    log_dir: str = "./logs"


def get_fedavg_config() -> ExperimentConfig:
    """FedAvg baseline (no DP, no compression)."""

    config = ExperimentConfig()
    config.client.training_strategy = TrainingStrategy.STANDARD
    config.client.topk_ratio = 1.0
    config.client.use_residual = False
    config.dp.enabled = False
    config.server.initial_clip = 1_000.0
    config.server.clip_update_method = ClipUpdateMethod.EMA
    config.server.ema_alpha = 1.0
    return config


def get_dp_fedavg_config(epsilon: float = 8.0) -> ExperimentConfig:
    """DP-FedAvg baseline."""

    config = ExperimentConfig()
    config.client.training_strategy = TrainingStrategy.STANDARD
    config.client.topk_ratio = 1.0
    config.client.use_residual = False
    config.dp.enabled = True
    config.dp.epsilon_total = epsilon
    config.dp.use_heterogeneous_noise = False
    config.server.clip_update_method = ClipUpdateMethod.EMA
    config.server.ema_alpha = 1.0
    return config


def get_dp_fedsam_config(epsilon: float = 8.0) -> ExperimentConfig:
    """DP-FedSAM baseline (same DP setting, local SAM optimizer)."""

    config = get_dp_fedavg_config(epsilon=epsilon)
    config.client.training_strategy = TrainingStrategy.DP_FEDSAM
    config.client.sam_rho = 0.05
    return config


def get_proposed_config(epsilon: float = 8.0) -> ExperimentConfig:
    """Proposed method with all modules enabled."""

    config = ExperimentConfig()
    config.client.training_strategy = TrainingStrategy.CONTRASTIVE
    config.client.importance_strategy = ImportanceStrategy.GRAD_NORMALIZED
    config.client.topk_strategy = TopKStrategy.WEIGHTED_LAYER_NORM
    config.client.topk_ratio = 0.1
    config.client.use_residual = True

    config.dp.enabled = True
    config.dp.epsilon_total = epsilon
    config.dp.use_heterogeneous_noise = True

    config.server.clip_update_method = ClipUpdateMethod.ADAPTIVE
    return config


if __name__ == "__main__":
    cfg = get_proposed_config()
    print("Experiment preset loaded")
    print(f"strategy={cfg.client.training_strategy.value}")
    print(f"topk={cfg.client.topk_ratio}")
    print(f"epsilon={cfg.dp.epsilon_total}")
