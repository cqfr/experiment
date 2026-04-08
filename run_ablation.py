from __future__ import annotations

"""Ablation and baseline runner."""

import argparse
import json
import os
from datetime import datetime
from typing import Dict, List

from config import ClipUpdateMethod, TrainingStrategy
from train import FLTrainer
from config import (
    ExperimentConfig,
    get_dp_fedavg_config,
    get_dp_fedsam_config,
    get_fedavg_config,
    get_proposed_config,
)


def _base_overrides(config: ExperimentConfig, num_rounds: int, dataset: str) -> ExperimentConfig:
    config.num_rounds = num_rounds
    config.dataset = dataset
    config.num_clients = 100
    config.clients_per_round = 10
    config.iid = False
    config.alpha = 0.5
    return config


def get_ablation_set(epsilon: float, num_rounds: int, dataset: str) -> Dict[str, ExperimentConfig]:
    exps: Dict[str, ExperimentConfig] = {}

    exps["fedavg"] = _base_overrides(get_fedavg_config(), num_rounds, dataset)
    exps["dp_fedavg"] = _base_overrides(get_dp_fedavg_config(epsilon), num_rounds, dataset)
    exps["dp_fedsam"] = _base_overrides(get_dp_fedsam_config(epsilon), num_rounds, dataset)
    exps["proposed_full"] = _base_overrides(get_proposed_config(epsilon), num_rounds, dataset)

    no_compression = _base_overrides(get_proposed_config(epsilon), num_rounds, dataset)
    no_compression.client.topk_ratio = 1.0
    no_compression.client.use_residual = False
    exps["ablation_no_compression"] = no_compression

    no_adaptive_clip = _base_overrides(get_proposed_config(epsilon), num_rounds, dataset)
    no_adaptive_clip.server.clip_update_method = ClipUpdateMethod.EMA
    no_adaptive_clip.server.ema_alpha = 1.0
    exps["ablation_no_adaptive_clip"] = no_adaptive_clip

    no_hetero_noise = _base_overrides(get_proposed_config(epsilon), num_rounds, dataset)
    no_hetero_noise.dp.use_heterogeneous_noise = False
    exps["ablation_no_hetero_noise"] = no_hetero_noise

    no_contrastive = _base_overrides(get_proposed_config(epsilon), num_rounds, dataset)
    no_contrastive.client.training_strategy = TrainingStrategy.STANDARD
    exps["ablation_no_contrastive"] = no_contrastive

    return exps


def run_one(name: str, config: ExperimentConfig) -> Dict:
    print("=" * 72)
    print(f"Running: {name}")
    print("=" * 72)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.save_dir = f"./results/{name}_{timestamp}"
    config.log_dir = f"./results/{name}_{timestamp}/logs"

    trainer = FLTrainer(config)
    result = trainer.train()
    result["experiment"] = name
    result["config"] = {
        "dataset": config.dataset,
        "strategy": config.client.training_strategy.value,
        "topk_ratio": config.client.topk_ratio,
        "clip_update": config.server.clip_update_method.value,
        "heterogeneous_noise": config.dp.use_heterogeneous_noise,
        "epsilon_total": config.dp.epsilon_total if config.dp.enabled else None,
    }
    return result


def run_ablation_study(
    epsilon: float,
    num_rounds: int,
    dataset: str,
    experiments: List[str] | None,
) -> Dict[str, Dict]:
    all_configs = get_ablation_set(epsilon=epsilon, num_rounds=num_rounds, dataset=dataset)

    if experiments is None:
        selected = list(all_configs.keys())
    else:
        selected = experiments

    results: Dict[str, Dict] = {}
    for name in selected:
        if name not in all_configs:
            print(f"Skip unknown experiment: {name}")
            continue
        results[name] = run_one(name, all_configs[name])

    os.makedirs("./results", exist_ok=True)
    summary_path = f"./results/ablation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\nSummary")
    print(f"{'Experiment':<28} {'Best Acc':>12} {'Final Acc':>12}")
    print("-" * 56)
    for name, r in results.items():
        print(f"{name:<28} {r['best_accuracy']:>12.2%} {r['final_accuracy']:>12.2%}")
    print(f"Saved summary to: {summary_path}")

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline/ablation experiments")
    parser.add_argument("--experiments", nargs="+", default=None)
    parser.add_argument("--epsilon", type=float, default=8.0)
    parser.add_argument("--num_rounds", type=int, default=100)
    parser.add_argument("--dataset", choices=["cifar10", "mnist"], default="cifar10")
    parser.add_argument("--quick_test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick_test:
        args.num_rounds = 5
    run_ablation_study(
        epsilon=args.epsilon,
        num_rounds=args.num_rounds,
        dataset=args.dataset,
        experiments=args.experiments,
    )


if __name__ == "__main__":
    main()
