"""
run_hyperparam.py
超参数敏感性实验脚本

实验内容：
1. 压缩率 k 的影响：10%, 20%, 30%, 50%, 100%
2. 隐私预算 ε 的影响：1, 2, 4, 8, 16
3. Non-IID 程度（α）的影响：0.1, 0.3, 0.5, 1.0, 10.0
"""

import os
import json
import copy
from datetime import datetime
from typing import Dict, List
from itertools import product

from config import (
    ExperimentConfig,
    TrainingStrategy,
    TopKStrategy,
    ClipUpdateMethod,
)
from train import FLTrainer


# ══════════════════════════════════════════════════════════════════════════════
# 基础配置
# ══════════════════════════════════════════════════════════════════════════════

def get_base_proposed_config() -> ExperimentConfig:
    """基础提出方法配置"""
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
    config.alpha = 0.5
    
    # DP配置
    config.dp.enabled = True
    config.dp.epsilon_total = 8.0
    config.dp.use_heterogeneous_noise = True
    
    # 客户端配置
    config.client.training_strategy = TrainingStrategy.CONTRASTIVE
    config.client.topk_ratio = 0.1
    config.client.topk_strategy = TopKStrategy.WEIGHTED_LAYER_NORM
    config.client.use_residual = True
    
    # 服务器配置
    config.server.clip_update_method = ClipUpdateMethod.ADAPTIVE
    
    # 日志
    config.log_interval = 10
    config.save_interval = 50
    
    return config


# ══════════════════════════════════════════════════════════════════════════════
# 实验函数
# ══════════════════════════════════════════════════════════════════════════════

def run_topk_sensitivity(
    k_values: List[float] = [0.1, 0.2, 0.3, 0.5, 1.0],
    num_rounds: int = 100,
) -> Dict:
    """
    实验1: 压缩率 k 的影响
    """
    print("\n" + "="*60)
    print("实验1: 压缩率 k 敏感性分析")
    print("="*60)
    
    results = {}
    
    for k in k_values:
        print(f"\n--- k = {k:.0%} ---")
        
        config = get_base_proposed_config()
        config.num_rounds = num_rounds
        config.client.topk_ratio = k
        
        # 当 k=1.0 时禁用残差累积
        if k >= 1.0:
            config.client.use_residual = False
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config.save_dir = f"./results/hyperparam/topk_{int(k*100)}_{timestamp}"
        config.log_dir = config.save_dir + "/logs"
        
        trainer = FLTrainer(config)
        exp_results = trainer.train()
        
        results[f"k={k:.0%}"] = {
            "k": k,
            "best_accuracy": exp_results["best_accuracy"],
            "final_accuracy": exp_results["final_accuracy"],
        }
    
    return results


def run_epsilon_sensitivity(
    epsilon_values: List[float] = [1.0, 2.0, 4.0, 8.0, 16.0],
    num_rounds: int = 100,
) -> Dict:
    """
    实验2: 隐私预算 ε 的影响
    """
    print("\n" + "="*60)
    print("实验2: 隐私预算 ε 敏感性分析")
    print("="*60)
    
    results = {}
    
    for eps in epsilon_values:
        print(f"\n--- ε = {eps} ---")
        
        config = get_base_proposed_config()
        config.num_rounds = num_rounds
        config.dp.epsilon_total = eps
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config.save_dir = f"./results/hyperparam/eps_{eps}_{timestamp}"
        config.log_dir = config.save_dir + "/logs"
        
        trainer = FLTrainer(config)
        exp_results = trainer.train()
        
        results[f"ε={eps}"] = {
            "epsilon": eps,
            "best_accuracy": exp_results["best_accuracy"],
            "final_accuracy": exp_results["final_accuracy"],
        }
    
    return results


def run_alpha_sensitivity(
    alpha_values: List[float] = [0.1, 0.3, 0.5, 1.0, 10.0],
    num_rounds: int = 100,
) -> Dict:
    """
    实验3: Non-IID 程度（α）的影响
    
    α 越小，数据越异构
    """
    print("\n" + "="*60)
    print("实验3: Non-IID 程度 α 敏感性分析")
    print("="*60)
    
    results = {}
    
    for alpha in alpha_values:
        print(f"\n--- α = {alpha} ---")
        
        config = get_base_proposed_config()
        config.num_rounds = num_rounds
        config.alpha = alpha
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config.save_dir = f"./results/hyperparam/alpha_{alpha}_{timestamp}"
        config.log_dir = config.save_dir + "/logs"
        
        trainer = FLTrainer(config)
        exp_results = trainer.train()
        
        results[f"α={alpha}"] = {
            "alpha": alpha,
            "best_accuracy": exp_results["best_accuracy"],
            "final_accuracy": exp_results["final_accuracy"],
        }
    
    return results


def run_all_hyperparam_experiments(
    num_rounds: int = 100,
    run_topk: bool = True,
    run_epsilon: bool = True,
    run_alpha: bool = True,
) -> Dict:
    """
    运行所有超参数实验
    """
    all_results = {}
    
    if run_topk:
        all_results["topk_sensitivity"] = run_topk_sensitivity(num_rounds=num_rounds)
    
    if run_epsilon:
        all_results["epsilon_sensitivity"] = run_epsilon_sensitivity(num_rounds=num_rounds)
    
    if run_alpha:
        all_results["alpha_sensitivity"] = run_alpha_sensitivity(num_rounds=num_rounds)
    
    # 保存汇总
    os.makedirs("./results/hyperparam", exist_ok=True)
    summary_path = f"./results/hyperparam/summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("超参数实验完成")
    print(f"{'='*60}")
    
    # 打印汇总
    for exp_name, exp_results in all_results.items():
        print(f"\n{exp_name}:")
        print(f"{'参数':<20} {'最佳准确率':>12} {'最终准确率':>12}")
        print("-" * 46)
        for key, val in exp_results.items():
            print(f"{key:<20} {val['best_accuracy']:>12.2%} {val['final_accuracy']:>12.2%}")
    
    print(f"\n汇总已保存到: {summary_path}")
    
    return all_results


# ══════════════════════════════════════════════════════════════════════════════
# 主函数
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="超参数敏感性实验")
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["all", "topk", "epsilon", "alpha"],
                        help="要运行的实验类型")
    parser.add_argument("--num_rounds", type=int, default=100,
                        help="训练轮数")
    parser.add_argument("--quick_test", action="store_true",
                        help="快速测试模式（少量轮次和参数）")
    
    args = parser.parse_args()
    
    if args.quick_test:
        args.num_rounds = 5
        print("快速测试模式")
    
    if args.experiment == "all":
        run_all_hyperparam_experiments(num_rounds=args.num_rounds)
    elif args.experiment == "topk":
        run_topk_sensitivity(num_rounds=args.num_rounds)
    elif args.experiment == "epsilon":
        run_epsilon_sensitivity(num_rounds=args.num_rounds)
    elif args.experiment == "alpha":
        run_alpha_sensitivity(num_rounds=args.num_rounds)
