from __future__ import annotations

"""Parallel multi-process launcher for the current OmegaConf-based train.py."""

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence


@dataclass(frozen=True)
class Experiment:
    name: str
    overrides: tuple[str, ...]


def _csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _bool(value: bool) -> str:
    return "true" if value else "false"


def _fmt(value: float) -> str:
    return f"{value:g}"


def _common(args: argparse.Namespace, seed: int) -> list[str]:
    return [
        f"seed={seed}",
        f"model.name={args.model}",
        f"data.dataset={args.dataset}",
        f"data.num_clients={args.num_clients}",
        f"data.clients_per_round={args.clients_per_round}",
        f"data.batch_size={args.batch_size}",
        f"data.iid={_bool(args.iid)}",
        f"data.alpha={args.alpha}",
        f"trainer.num_rounds={args.rounds}",
        f"trainer.max_workers={args.intra_workers}",
        f"trainer.num_workers={args.data_workers}",
        f"trainer.eval_batch_size={args.eval_batch_size}",
        f"trainer.save_interval={args.save_interval}",
        f"trainer.log_interval={args.log_interval}",
        f"client.local_epochs={args.local_epochs}",
        f"client.lr={args.lr}",
    ]


def _ours_core(args: argparse.Namespace) -> list[str]:
    return [
        "experiment.name=proposed",
        "client.training_strategy=standard",
        "compressor.type=topk",
        "compressor.use_residual=false",
        "compressor.importance_strategy=grad_normalized",
        "compressor.topk_strategy=weighted_layer_norm",
        "compressor.layer_weight_method=mean",
        "server.clip_update_method=adaptive",
        "dp.enabled=true",
        "dp.use_heterogeneous_noise=true",
        "dp.use_global_importance_for_topk=true",
        "dp.global_local_mix_lambda=0.0",
        "dp.enable_importance_freeze=false",
        f"dp.epsilon_total={args.base_epsilon}",
    ]


def _ours_full(args: argparse.Namespace) -> list[str]:
    return [
        "experiment.name=proposed",
        f"client.training_strategy={args.best_regularizer}",
        "compressor.type=topk",
        "compressor.use_residual=true",
        f"compressor.importance_strategy={args.best_importance}",
        f"compressor.topk_strategy={args.best_topk_strategy}",
        "compressor.layer_weight_method=mean",
        f"compressor.topk_ratio={args.best_topk}",
        "server.clip_update_method=adaptive",
        f"server.target_quantile={args.best_target_quantile}",
        f"server.clip_lr={args.best_clip_lr}",
        "dp.enabled=true",
        "dp.use_heterogeneous_noise=true",
        "dp.use_global_importance_for_topk=true",
        f"dp.global_local_mix_lambda={args.best_mix_lambda}",
        "dp.enable_importance_freeze=true",
    ]


def _fedavg() -> list[str]:
    return ["experiment.name=fedavg"]


def _dp_fedavg(epsilon: float, clip: float) -> list[str]:
    return [
        "experiment.name=dp_fedavg",
        f"dp.epsilon_total={_fmt(epsilon)}",
        f"server.initial_clip={_fmt(clip)}",
    ]


def _dp_fedsam(epsilon: float, clip: float) -> list[str]:
    return [
        "experiment.name=dp_fedsam",
        f"dp.epsilon_total={_fmt(epsilon)}",
        f"server.initial_clip={_fmt(clip)}",
    ]


def validation_suite(args: argparse.Namespace) -> list[Experiment]:
    epsilons = _csv_floats(args.validation_epsilons)
    clips = _csv_floats(args.dp_clips)
    seeds = _csv_ints(args.seeds)
    experiments: list[Experiment] = []
    for seed in seeds:
        common = _common(args, seed)
        experiments.append(Experiment(f"val_fedavg_s{seed}", tuple(common + _fedavg())))

        for clip in clips:
            for eps in epsilons:
                name = f"val_dp_fedavg_clip{_fmt(clip)}_eps{_fmt(eps)}_s{seed}"
                experiments.append(Experiment(name, tuple(common + _dp_fedavg(eps, clip))))

        for topk in _csv_floats(args.core_topks):
            for target_q in _csv_floats(args.core_target_quantiles):
                name = f"val_ours_core_k{_fmt(topk)}_q{_fmt(target_q)}_s{seed}"
                overrides = _ours_core(args) + [
                    f"compressor.topk_ratio={_fmt(topk)}",
                    f"server.target_quantile={_fmt(target_q)}",
                ]
                experiments.append(Experiment(name, tuple(common + overrides)))

        for regularizer in ("contrastive", "fedprox"):
            for importance in ("grad_normalized", "fisher_grad"):
                name = f"val_ours_full_{regularizer}_{importance}_s{seed}"
                overrides = _ours_full(args) + [
                    f"client.training_strategy={regularizer}",
                    f"compressor.importance_strategy={importance}",
                    f"dp.epsilon_total={args.base_epsilon}",
                ]
                experiments.append(Experiment(name, tuple(common + overrides)))

        if args.include_fedsam:
            name = f"val_dp_fedsam_eps{_fmt(args.base_epsilon)}_clip{_fmt(args.best_dp_clip)}_s{seed}"
            experiments.append(Experiment(name, tuple(common + _dp_fedsam(args.base_epsilon, args.best_dp_clip))))
    return experiments


def main_suite(args: argparse.Namespace) -> list[Experiment]:
    epsilons = _csv_floats(args.main_epsilons)
    seeds = _csv_ints(args.seeds)
    experiments: list[Experiment] = []
    for seed in seeds:
        common = _common(args, seed)
        experiments.append(Experiment(f"main_fedavg_s{seed}", tuple(common + _fedavg())))
        for eps in epsilons:
            experiments.append(
                Experiment(
                    f"main_dp_fedavg_eps{_fmt(eps)}_s{seed}",
                    tuple(common + _dp_fedavg(eps, args.best_dp_clip)),
                )
            )
            if args.include_fedsam:
                experiments.append(
                    Experiment(
                        f"main_dp_fedsam_eps{_fmt(eps)}_s{seed}",
                        tuple(common + _dp_fedsam(eps, args.best_dp_clip)),
                    )
                )
            experiments.append(
                Experiment(
                    f"main_ours_full_eps{_fmt(eps)}_s{seed}",
                    tuple(common + _ours_full(args) + [f"dp.epsilon_total={_fmt(eps)}"]),
                )
            )
    return experiments


def ablation_macro_suite(args: argparse.Namespace) -> list[Experiment]:
    seeds = _csv_ints(args.seeds)
    experiments: list[Experiment] = []
    for seed in seeds:
        common = _common(args, seed)
        base = _ours_full(args) + [f"dp.epsilon_total={args.base_epsilon}"]
        variants = {
            "full": [],
            "no_adaptive_clip": [
                "server.clip_update_method=ema",
                "server.ema_alpha=1.0",
                f"server.initial_clip={_fmt(args.best_dp_clip)}",
            ],
            "no_compression": [
                "compressor.type=identity",
                "compressor.topk_ratio=1.0",
                "compressor.use_residual=false",
                "dp.use_global_importance_for_topk=false",
                "dp.global_local_mix_lambda=0.0",
            ],
            "no_heterogeneous_noise": [
                "dp.use_heterogeneous_noise=false",
            ],
        }
        for name, overrides in variants.items():
            experiments.append(Experiment(f"abl_macro_{name}_s{seed}", tuple(common + base + overrides)))
    return experiments


def ablation_fine_suite(args: argparse.Namespace) -> list[Experiment]:
    seeds = _csv_ints(args.seeds)
    experiments: list[Experiment] = []
    for seed in seeds:
        common = _common(args, seed)
        base = _ours_full(args) + [f"dp.epsilon_total={args.base_epsilon}"]
        variants = {
            "regularizer_standard": ["client.training_strategy=standard"],
            "regularizer_fedprox": ["client.training_strategy=fedprox"],
            "regularizer_contrastive": ["client.training_strategy=contrastive"],
            "importance_grad_normalized": ["compressor.importance_strategy=grad_normalized"],
            "importance_grad_squared": ["compressor.importance_strategy=grad_squared"],
            "importance_fisher_grad": ["compressor.importance_strategy=fisher_grad"],
            "topk_global": ["compressor.topk_strategy=global_topk"],
            "topk_layer": ["compressor.topk_strategy=layer_topk"],
            "topk_layer_norm_global": ["compressor.topk_strategy=layer_norm_global"],
            "topk_weighted_layer_norm": ["compressor.topk_strategy=weighted_layer_norm"],
            "no_residual": ["compressor.use_residual=false"],
            "no_global_prior": ["dp.use_global_importance_for_topk=false", "dp.global_local_mix_lambda=0.0"],
            "no_importance_freeze": ["dp.enable_importance_freeze=false"],
            "mix_lambda_0": ["dp.global_local_mix_lambda=0.0"],
            "mix_lambda_02": ["dp.global_local_mix_lambda=0.2"],
            "mix_lambda_04": ["dp.global_local_mix_lambda=0.4"],
            "mix_lambda_06": ["dp.global_local_mix_lambda=0.6"],
        }
        for name, overrides in variants.items():
            experiments.append(Experiment(f"abl_fine_{name}_s{seed}", tuple(common + base + overrides)))
    return experiments


def sensitivity_suite(args: argparse.Namespace) -> list[Experiment]:
    seeds = _csv_ints(args.seeds)
    experiments: list[Experiment] = []
    for seed in seeds:
        common = _common(args, seed)
        base = _ours_full(args) + [f"dp.epsilon_total={args.base_epsilon}"]
        for topk in _csv_floats(args.sens_topks):
            experiments.append(
                Experiment(
                    f"sens_topk_{_fmt(topk)}_s{seed}",
                    tuple(common + base + [f"compressor.topk_ratio={_fmt(topk)}"]),
                )
            )
        for alpha in _csv_floats(args.sens_alphas):
            overrides = [item for item in common if not item.startswith("data.alpha=")]
            overrides.append(f"data.alpha={_fmt(alpha)}")
            experiments.append(Experiment(f"sens_alpha_{_fmt(alpha)}_s{seed}", tuple(overrides + base)))
        for lr in _csv_floats(args.sens_lrs):
            overrides = [item for item in common if not item.startswith("client.lr=")]
            overrides.append(f"client.lr={_fmt(lr)}")
            experiments.append(Experiment(f"sens_lr_{_fmt(lr)}_s{seed}", tuple(overrides + base)))
    return experiments


def build_suite(args: argparse.Namespace) -> list[Experiment]:
    suites = {
        "validation": validation_suite,
        "main": main_suite,
        "ablation_macro": ablation_macro_suite,
        "ablation_fine": ablation_fine_suite,
        "sensitivity": sensitivity_suite,
    }
    if args.suite == "all":
        experiments: list[Experiment] = []
        for suite_name in ("validation", "main", "ablation_macro", "ablation_fine", "sensitivity"):
            experiments.extend(suites[suite_name](args))
        return experiments
    return suites[args.suite](args)


def _effective_signature(overrides: Sequence[str]) -> tuple[str, ...]:
    """Normalize OmegaConf dotlist overrides by last-write-wins semantics."""

    effective: dict[str, str] = {}
    positional: list[str] = []
    for item in overrides:
        if "=" not in item:
            positional.append(item)
            continue
        key, value = item.split("=", 1)
        effective[key] = value
    return tuple(positional + [f"{key}={effective[key]}" for key in sorted(effective)])


def validate_experiments(experiments: Sequence[Experiment]) -> None:
    names: dict[str, int] = {}
    signatures: dict[tuple[str, ...], str] = {}
    duplicate_names: list[str] = []
    duplicate_configs: list[tuple[str, str]] = []

    for idx, exp in enumerate(experiments):
        if exp.name in names:
            duplicate_names.append(exp.name)
        names[exp.name] = idx

        signature = _effective_signature(exp.overrides)
        previous_name = signatures.get(signature)
        if previous_name is not None:
            duplicate_configs.append((previous_name, exp.name))
        else:
            signatures[signature] = exp.name

    if duplicate_names or duplicate_configs:
        messages = []
        if duplicate_names:
            messages.append(f"duplicate experiment names: {', '.join(sorted(set(duplicate_names)))}")
        if duplicate_configs:
            pairs = ", ".join(f"{left} == {right}" for left, right in duplicate_configs[:10])
            if len(duplicate_configs) > 10:
                pairs += f", ... and {len(duplicate_configs) - 10} more"
            messages.append(f"duplicate effective configs: {pairs}")
        raise ValueError("; ".join(messages))


def _command_line(parts: Sequence[str]) -> str:
    return subprocess.list2cmdline(list(parts)) if os.name == "nt" else " ".join(parts)


def _run(args: argparse.Namespace, experiments: Sequence[Experiment]) -> int:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    launcher_log_dir = Path(args.launcher_log_dir) / run_id
    launcher_log_dir.mkdir(parents=True, exist_ok=True)

    max_parallel = max(1, int(args.max_parallel))
    pending = list(experiments)
    running: list[tuple[subprocess.Popen, object, Experiment]] = []
    failures = 0

    while pending or running:
        while pending and len(running) < max_parallel:
            exp = pending.pop(0)
            save_dir = str(Path(args.output_root) / "checkpoints" / f"{run_id}_{exp.name}")
            log_dir = str(Path(args.output_root) / "logs" / f"{run_id}_{exp.name}")
            overrides = list(exp.overrides) + [
                f"trainer.save_dir={save_dir}",
                f"trainer.log_dir={log_dir}",
            ]
            cmd = [sys.executable, "train.py", *overrides]
            out_path = launcher_log_dir / f"{exp.name}.out"
            out_handle = out_path.open("w", encoding="utf-8")
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            if args.cuda_visible_devices:
                env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
            print(f"[start] {exp.name}")
            print(f"        {_command_line(cmd)}")
            proc = subprocess.Popen(
                cmd,
                stdout=out_handle,
                stderr=subprocess.STDOUT,
                cwd=args.workdir,
                env=env,
            )
            running.append((proc, out_handle, exp))

        time.sleep(args.poll_interval)
        still_running: list[tuple[subprocess.Popen, object, Experiment]] = []
        for proc, out_handle, exp in running:
            ret = proc.poll()
            if ret is None:
                still_running.append((proc, out_handle, exp))
                continue
            out_handle.close()
            if ret == 0:
                print(f"[done]  {exp.name}")
            else:
                failures += 1
                print(f"[fail]  {exp.name} exit={ret}")
        running = still_running

    print(f"Launcher logs: {launcher_log_dir}")
    return 1 if failures else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run train.py experiment grids with bounded process parallelism.")
    parser.add_argument("--suite", choices=["validation", "main", "ablation_macro", "ablation_fine", "sensitivity", "all"], default="validation")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-parallel", type=int, default=3)
    parser.add_argument("--intra-workers", type=int, default=2)
    parser.add_argument("--data-workers", type=int, default=0)
    parser.add_argument("--cuda-visible-devices", default="")
    parser.add_argument("--workdir", default=".")
    parser.add_argument("--output-root", default="./runs")
    parser.add_argument("--launcher-log-dir", default="./runs/launcher_logs")
    parser.add_argument("--poll-interval", type=float, default=5.0)

    parser.add_argument("--dataset", choices=["cifar10", "mnist"], default="cifar10")
    parser.add_argument("--model", choices=["simple_cnn", "resnet18"], default="simple_cnn")
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--num-clients", type=int, default=500)
    parser.add_argument("--clients-per-round", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--local-epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--iid", action="store_true")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=1000)

    parser.add_argument("--base-epsilon", type=float, default=8.0)
    parser.add_argument("--validation-epsilons", default="1,2,4,6,8,10")
    parser.add_argument("--main-epsilons", default="1,2,4,6,8,10")
    parser.add_argument("--dp-clips", default="0.5,1.0,1.5")
    parser.add_argument("--include-fedsam", action="store_true")

    parser.add_argument("--core-topks", default="0.05,0.1,0.2")
    parser.add_argument("--core-target-quantiles", default="0.5,0.7")
    parser.add_argument("--best-dp-clip", type=float, default=1.0)
    parser.add_argument("--best-topk", type=float, default=0.1)
    parser.add_argument("--best-target-quantile", type=float, default=0.7)
    parser.add_argument("--best-clip-lr", type=float, default=0.2)
    parser.add_argument("--best-mix-lambda", type=float, default=0.4)
    parser.add_argument("--best-regularizer", choices=["standard", "fedprox", "contrastive"], default="contrastive")
    parser.add_argument("--best-importance", choices=["grad_normalized", "grad_squared", "fisher_grad"], default="fisher_grad")
    parser.add_argument("--best-topk-strategy", choices=["global_topk", "layer_topk", "layer_norm_global", "weighted_layer_norm"], default="weighted_layer_norm")

    parser.add_argument("--sens-topks", default="0.05,0.1,0.2,0.3")
    parser.add_argument("--sens-alphas", default="0.1,0.3,0.5,1.0")
    parser.add_argument("--sens-lrs", default="0.003,0.01,0.03")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiments = build_suite(args)
    validate_experiments(experiments)
    print(f"Suite={args.suite} experiments={len(experiments)} max_parallel={args.max_parallel}")
    if args.dry_run:
        for exp in experiments:
            cmd = [sys.executable, "train.py", *exp.overrides]
            print(f"{exp.name}: {_command_line(cmd)}")
        return
    raise SystemExit(_run(args, experiments))


if __name__ == "__main__":
    main()
