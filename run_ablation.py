"""
run_ablation.py
消融实验脚本

实验组：
1. FedAvg（基线，无DP）
2. DP-FedAvg（基线，有DP）
3. 完整方法（三模块全开）
4. 去掉梯度压缩
5. 去掉自适应裁剪
6. 去掉异构噪声
7. 去掉Contrastive Loss
"""

import os
import json
import copy
from datetime import datetime
from typing import Dict, List

from config import (
    ExperimentConfig,
    ClientConfig,
    ServerConfig,
    DPConfig,
    TrainingStrategy,
    ImportanceStrategy,
    TopKStrategy,
    ClipUpdateMethod,
)
from train import FLTrainer


# ══════════════════════════════════════════════════════════════════════════════
# 实验配置定义
# ══════════════════════════════════════════════════════════════════════════════

def get_base_config() -> ExperimentConfig:
    """基础配置（所有实验共享）"""
    config = ExperimentConfig()
    
    # 基础设置
    config.seed = 42
    config.num_clients = 100
    config.clients_per_round = 10
    config.num_rounds = 100
    config.batch_size = 32
    
    # 数据集
    config.dataset = "cifar10"
    config.iid = False
    config.alpha = 0.5  # Non-IID
    
    # 日志
    config.log_interval = 5
    config.save_interval = 20
    
    return config


def get_fedavg_config() -> ExperimentConfig:
    """实验1: FedAvg（无DP基线）"""
    config = get_base_config()
    
    # 无DP
    config.dp.enabled = False
    
    # 标准训练
    config.client.training_strategy = TrainingStrategy.STANDARD
    config.client.topk_ratio = 1.0  # 不压缩
    config.client.use_residual = False
    
    return config


def get_dp_fedavg_config(epsilon: float = 8.0) -> ExperimentConfig:
    """实验2: DP-FedAvg（有DP基线）"""
    config = get_base_config()
    
    # DP配置
    config.dp.enabled = True
    config.dp.epsilon_total = epsilon
    config.dp.use_heterogeneous_noise = False  # 同构噪声
    
    # 标准训练
    config.client.training_strategy = TrainingStrategy.STANDARD
    config.client.topk_ratio = 1.0  # 不压缩
    config.client.use_residual = False
    
    # 固定裁剪阈值
    config.server.clip_update_method = ClipUpdateMethod.EMA
    
    return config


def get_proposed_full_config(epsilon: float = 8.0) -> ExperimentConfig:
    """实验3: 完整方法（三模块全开）"""
    config = get_base_config()
    
    # DP配置
    config.dp.enabled = True
    config.dp.epsilon_total = epsilon
    config.dp.use_heterogeneous_noise = True
    config.dp.use_subsampling_amplification = True
    config.dp.use_sparsity_amplification = True
    
    # Contrastive Loss
    config.client.training_strategy = TrainingStrategy.CONTRASTIVE
    config.client.alpha = 0.005
    config.client.beta = 0.005
    
    # 梯度压缩
    config.client.topk_ratio = 0.1
    config.client.topk_strategy = TopKStrategy.WEIGHTED_LAYER_NORM
    config.client.importance_strategy = ImportanceStrategy.GRAD_NORMALIZED
    config.client.use_residual = True
    
    # 自适应裁剪
    config.server.clip_update_method = ClipUpdateMethod.ADAPTIVE
    config.server.target_quantile = 0.5
    config.server.clip_lr = 0.2
    
    return config


def get_ablation_no_compression(epsilon: float = 8.0) -> ExperimentConfig:
    """实验4: 去掉梯度压缩"""
    config = get_proposed_full_config(epsilon)
    
    # 禁用压缩
    config.client.topk_ratio = 1.0  # 不压缩
    config.client.use_residual = False
    config.dp.use_sparsity_amplification = False
    
    return config


def get_ablation_no_adaptive_clip(epsilon: float = 8.0) -> ExperimentConfig:
    """实验5: 去掉自适应裁剪"""
    config = get_proposed_full_config(epsilon)
    
    # 使用固定裁剪阈值
    config.server.clip_update_method = ClipUpdateMethod.EMA
    config.server.ema_alpha = 1.0  # 相当于固定阈值
    
    return config


def get_ablation_no_hetero_noise(epsilon: float = 8.0) -> ExperimentConfig:
    """实验6: 去掉异构噪声"""
    config = get_proposed_full_config(epsilon)
    
    # 使用同构噪声
    config.dp.use_heterogeneous_noise = False
    
    return config


def get_ablation_no_contrastive(epsilon: float = 8.0) -> ExperimentConfig:
    """实验7: 去掉Contrastive Loss"""
    config = get_proposed_full_config(epsilon)
    
    # 标准训练
    config.client.training_strategy = TrainingStrategy.STANDARD
    
    return config


# ══════════════════════════════════════════════════════════════════════════════
# 实验运行
# ══════════════════════════════════════════════════════════════════════════════

EXPERIMENTS = {
    "fedavg": ("FedAvg (无DP)", get_fedavg_config),
    "dp_fedavg": ("DP-FedAvg", get_dp_fedavg_config),
    "proposed_full": ("完整方法", get_proposed_full_config),
    "ablation_no_compress": ("消融: 无梯度压缩", get_ablation_no_compression),
    "ablation_no_adaptive": ("消融: 无自适应裁剪", get_ablation_no_adaptive_clip),
    "ablation_no_hetero": ("消融: 无异构噪声", get_ablation_no_hetero_noise),
    "ablation_no_contrastive": ("消融: 无Contrastive Loss", get_ablation_no_contrastive),
}


def run_single_experiment(
    exp_name: str,
    config: ExperimentConfig,
    exp_label: str,
) -> Dict:
    """运行单个实验"""
    print(f"\n{'='*60}")
    print(f"实验: {exp_label}")
    print(f"名称: {exp_name}")
    print(f"{'='*60}")
    
    # 设置保存路径
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.save_dir = f"./results/{exp_name}_{timestamp}"
    config.log_dir = f"./results/{exp_name}_{timestamp}/logs"
    
    # 创建训练器并训练
    trainer = FLTrainer(config)
    results = trainer.train()
    
    # 添加实验信息
    results["experiment"] = exp_name
    results["label"] = exp_label
    results["config"] = {
        "dp_enabled": config.dp.enabled,
        "epsilon": config.dp.epsilon_total if config.dp.enabled else None,
        "topk_ratio": config.client.topk_ratio,
        "training_strategy": config.client.training_strategy.value,
        "clip_method": config.server.clip_update_method.value,
        "hetero_noise": config.dp.use_heterogeneous_noise,
    }
    
    return results


def run_ablation_study(
    epsilon: float = 8.0,
    experiments: List[str] = None,
    num_rounds: int = 100,
) -> Dict[str, Dict]:
    """
    运行消融实验
    
    参数:
        epsilon: 隐私预算
        experiments: 要运行的实验列表（默认全部）
        num_rounds: 训练轮数
    """
    if experiments is None:
        experiments = list(EXPERIMENTS.keys())
    
    all_results = {}
    
    for exp_name in experiments:
        if exp_name not in EXPERIMENTS:
            print(f"警告: 未知实验 {exp_name}")
            continue
        
        exp_label, config_fn = EXPERIMENTS[exp_name]
        
        # 获取配置
        if "fedavg" in exp_name and "dp" not in exp_name:
            config = config_fn()
        else:
            config = config_fn(epsilon)
        
        # 更新轮数
        config.num_rounds = num_rounds
        
        # 运行实验
        results = run_single_experiment(exp_name, config, exp_label)
        all_results[exp_name] = results
    
    # 保存汇总结果
    summary_path = f"./results/ablation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs("./results", exist_ok=True)
    
    summary = {
        "experiments": list(all_results.keys()),
        "epsilon": epsilon,
        "num_rounds": num_rounds,
        "results": {
            name: {
                "best_accuracy": r["best_accuracy"],
                "final_accuracy": r["final_accuracy"],
                "config": r["config"],
            }
            for name, r in all_results.items()
        }
    }
    
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print("消融实验完成")
    print(f"{'='*60}")
    print(f"\n结果汇总:")
    print(f"{'实验':<30} {'最佳准确率':>12} {'最终准确率':>12}")
    print("-" * 56)
    for name, r in all_results.items():
        label = EXPERIMENTS[name][0]
        print(f"{label:<30} {r['best_accuracy']:>12.2%} {r['final_accuracy']:>12.2%}")
    print(f"\n汇总已保存到: {summary_path}")
    
    return all_results


# ══════════════════════════════════════════════════════════════════════════════
# 主函数
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="消融实验")
    parser.add_argument("--experiments", type=str, nargs="+", default=None,
                        choices=list(EXPERIMENTS.keys()),
                        help="要运行的实验（默认全部）")
    parser.add_argument("--epsilon", type=float, default=8.0,
                        help="隐私预算 ε")
    parser.add_argument("--num_rounds", type=int, default=100,
                        help="训练轮数")
    parser.add_argument("--quick_test", action="store_true",
                        help="快速测试模式（少量轮次）")
    
    args = parser.parse_args()
    
    if args.quick_test:
        args.num_rounds = 5
        print("快速测试模式: 5轮")
    
    run_ablation_study(
        epsilon=args.epsilon,
        experiments=args.experiments,
        num_rounds=args.num_rounds,
    )
