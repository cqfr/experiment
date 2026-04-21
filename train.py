from __future__ import annotations

"""Main federated training entry with OmegaConf and threaded client execution."""

import json
import math
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from client.client import FLClient
from components.strategies import PrivacyEngine
from data.utils import get_dataloader
from dp.noise import allocate_client_noise_stds
from models import ResNet18, SimpleCNN
from server.server import FLServer


@dataclass
class ExperimentState:
    rounds: List[int] = field(default_factory=list)
    train_metrics: List[Dict[str, float]] = field(default_factory=list)
    test_metrics: List[Dict[str, float]] = field(default_factory=list)
    clip_history: List[float] = field(default_factory=list)
    privacy_spent: List[float] = field(default_factory=list)
    upload_ratio: List[float] = field(default_factory=list)
    noise_multiplier: List[float] = field(default_factory=list)
    sigma_agg: List[float] = field(default_factory=list)
    clip_snr_proxy: List[float] = field(default_factory=list)
    signal_l2_norm: List[float] = field(default_factory=list)
    expected_noise_l2_norm: List[float] = field(default_factory=list)
    snr_signal_to_noise: List[float] = field(default_factory=list)


class FLTrainer:
    """Orchestrates client sampling, local updates, aggregation, and evaluation."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self._set_seed(int(cfg.seed))

        self.device = self._resolve_device()
        print(f"Device: {self.device}")

        self.train_loaders, self.test_loader = get_dataloader(
            num_clients=int(cfg.data.num_clients),
            batch_size=int(cfg.data.batch_size),
            alpha=float(cfg.data.alpha),
            iid=bool(cfg.data.iid),
            dataset=str(cfg.data.dataset),
            data_dir=cfg.data.data_dir,
            min_samples_per_client=int(cfg.data.min_samples_per_client),
            split_max_attempts=int(cfg.data.split_max_attempts),
            num_workers=int(getattr(cfg.trainer, "num_workers", 0)),
            persistent_workers=bool(getattr(cfg.trainer, "persistent_workers", False)),
        )

        self.model = self._build_model().cpu()
        self.server = FLServer(model=self.model, cfg=cfg, device=self.device)
        self.privacy_engine = PrivacyEngine(cfg.dp, num_rounds=int(cfg.trainer.num_rounds))

        self.clients: List[FLClient] = [
            FLClient(
                client_id=client_id,
                model=self._build_model(),
                dataloader=self.train_loaders[client_id],
                cfg=cfg,
                device=self.device,
            )
            for client_id in range(int(cfg.data.num_clients))
        ]

        self.history = ExperimentState()
        os.makedirs(str(cfg.trainer.save_dir), exist_ok=True)
        os.makedirs(str(cfg.trainer.log_dir), exist_ok=True)

    def _resolve_device(self) -> torch.device:
        requested = str(self.cfg.trainer.device)
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(requested)

    def _build_model(self) -> torch.nn.Module:
        model_name = str(self.cfg.model.name)
        num_classes = int(self.cfg.model.num_classes)
        if model_name == "simple_cnn":
            return SimpleCNN(num_classes=num_classes)
        if model_name == "resnet18":
            return ResNet18(num_classes=num_classes)
        raise ValueError(f"Unsupported model: {model_name}")

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _select_clients(self, count: int) -> List[int]:
        return random.sample(range(int(self.cfg.data.num_clients)), count)

    def _build_client_weights(self, selected_sizes: List[int]) -> List[float]:
        if str(self.cfg.server.aggregation_weight_strategy) == "equal":
            return [1.0 / max(1, len(selected_sizes)) for _ in selected_sizes]

        total_selected = sum(selected_sizes)
        if total_selected <= 0:
            return [1.0 / max(1, len(selected_sizes)) for _ in selected_sizes]
        return [size / total_selected for size in selected_sizes]

    def _build_local_noise_stds(
        self,
        client_weights: Sequence[float],
        privacy_state,
        clip_norm: float,
    ) -> List[float]:
        if not self.privacy_engine.is_enabled():
            return [0.0 for _ in client_weights]

        if str(self.cfg.server.aggregation_weight_strategy) == "equal":
            local_std = PrivacyEngine.compute_local_noise_std(
                clip_norm=clip_norm,
                noise_multiplier=privacy_state.noise_multiplier,
                num_selected_clients=len(client_weights),
            )
            return [local_std for _ in client_weights]

        # [MOD][阶段1] data_size 加权聚合时从目标聚合方差回推每个 client 的本地噪声，避免用等权公式错配。
        return allocate_client_noise_stds(
            client_weights=client_weights,
            sigma_agg=float(privacy_state.sigma_agg),
            strategy="uniform",
            max_sigma_scale=float(getattr(self.cfg.dp, "client_variance_max_scale", 10.0)),
        )

    def _need_viz(self, round_num: int, client_id: int) -> bool:
        return (
            bool(self.cfg.importance_viz.enabled)
            and round_num % max(1, int(self.cfg.importance_viz.interval)) == 0
            and client_id == int(self.cfg.importance_viz.client_id)
        )

    def _make_generator(self, round_num: int, client_id: int) -> torch.Generator:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(self.cfg.seed) * 100_000 + round_num * 1_000 + client_id + 17)
        return generator

    def train_round(self, round_num: int) -> Dict[str, float]:
        clip_norm = self.server.get_clip_norm()
        use_heterogeneous_noise = bool(self.cfg.dp.enabled and self.cfg.dp.use_heterogeneous_noise)
        clip_weight_template = self.server.get_clip_weight_template() if use_heterogeneous_noise else None
        global_importance_template = (
            self.server.get_global_importance_template()
            if bool(getattr(self.cfg.dp, "use_global_importance_for_topk", True))
            else None
        )

        selected_ids = self._select_clients(int(self.cfg.data.clients_per_round))
        selected_sizes = [self.clients[client_id].data_size for client_id in selected_ids]
        client_weights = self._build_client_weights(selected_sizes)

        sampling_rate = float(self.cfg.data.clients_per_round) / float(self.cfg.data.num_clients)
        compressor_ratio = float(self.cfg.compressor.topk_ratio) if str(self.cfg.compressor.type) == "topk" else 1.0
        privacy_state = self.privacy_engine.calibrate_round(
            client_weights=client_weights,
            clip_norm=clip_norm,
            sampling_rate=sampling_rate,
            compressor_ratio=compressor_ratio,
        )
        local_noise_stds = self._build_local_noise_stds(
            client_weights=client_weights,
            privacy_state=privacy_state,
            clip_norm=clip_norm,
        )

        need_importance_upload = self.server.should_request_importance()
        force_refresh_local_importance = self.server.should_refresh_local_importance(round_num)

        global_weights = self.server.get_global_weights()
        client_updates = self._run_clients_parallel(
            selected_ids=selected_ids,
            global_weights=global_weights,
            clip_norm=clip_norm,
            clip_weight_template=clip_weight_template,
            global_importance_template=global_importance_template,
            local_noise_stds=local_noise_stds,
            need_importance_upload=need_importance_upload,
            force_refresh_local_importance=force_refresh_local_importance,
            round_num=round_num,
        )

        aggregated_importance = self.server.aggregate_importance(client_updates) if need_importance_upload else None
        # [MOD][阶段1] client update 已经在本地裁剪并加噪，server.aggregate 只做 weighted sum。
        aggregated = self.server.aggregate(client_updates=client_updates)
        train_metrics = self.server.update_global_model(
            global_update=aggregated.noisy_update,
            stats_aggregated=aggregated.stats_aggregated,
            aggregated_importance=aggregated_importance,
        )

        expected_noise_l2_norm = float(privacy_state.sigma_agg * math.sqrt(max(1, aggregated.total_params)))
        snr_signal_to_noise = (
            float(aggregated.signal_l2_norm / max(expected_noise_l2_norm, 1e-12))
            if privacy_state.sigma_agg > 0
            else 0.0
        )
        avg_upload_ratio = float(np.mean([update.upload_ratio for update in client_updates])) if client_updates else 1.0

        train_metrics.update(
            {
                "avg_upload_ratio": avg_upload_ratio,
                "noise_multiplier": float(privacy_state.noise_multiplier),
                "sigma_agg": float(privacy_state.sigma_agg),
                "sensitivity_l2": float(privacy_state.sensitivity_l2),
                "clip_snr_proxy": float(privacy_state.clip_snr_proxy),
                "signal_l2_norm": float(aggregated.signal_l2_norm),
                "expected_noise_l2_norm": expected_noise_l2_norm,
                "snr_signal_to_noise": float(snr_signal_to_noise),
                "importance_upload": 1.0 if need_importance_upload else 0.0,
                "local_importance_refresh": 1.0 if force_refresh_local_importance else 0.0,
                "avg_local_noise_std": float(np.mean(local_noise_stds)) if local_noise_stds else 0.0,
            }
        )

        if self.privacy_engine.is_enabled():
            eps_spent = self.privacy_engine.consume_round(privacy_state)
            train_metrics["epsilon_spent"] = float(eps_spent)
            train_metrics["epsilon_remaining"] = float(self.privacy_engine.remaining_budget())

        return train_metrics

    def _run_clients_parallel(
        self,
        selected_ids: List[int],
        global_weights,
        clip_norm: float,
        clip_weight_template,
        global_importance_template,
        local_noise_stds: Sequence[float],
        need_importance_upload: bool,
        force_refresh_local_importance: bool,
        round_num: int,
    ):
        client_updates = []
        max_workers = int(self.cfg.trainer.max_workers)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    self.clients[client_id].train_and_upload,
                    global_weights=copy_to_cpu(global_weights),
                    clip_norm=clip_norm,
                    local_noise_std=float(local_noise_stds[idx]),
                    clip_weight_template=copy_to_cpu(clip_weight_template),
                    global_importance_template=copy_to_cpu(global_importance_template),
                    need_importance_upload=need_importance_upload,
                    force_refresh_local_importance=force_refresh_local_importance,
                    round_num=round_num,
                    return_importance_snapshot=self._need_viz(round_num, client_id),
                    importance_max_elements=int(self.cfg.importance_viz.max_elements),
                    generator=self._make_generator(round_num=round_num, client_id=client_id),
                )
                for idx, client_id in enumerate(selected_ids)
            ]

            for future in as_completed(futures):
                client_updates.append(future.result())
        return client_updates

    def evaluate(self) -> Dict[str, float]:
        return self.server.evaluate(self.test_loader)

    def train(self) -> Dict[str, float]:
        print("=" * 72)
        print("Start federated training")
        print("=" * 72)

        best_acc = 0.0
        last_eval = {"accuracy": 0.0, "loss": 0.0}

        for round_num in tqdm(range(1, int(self.cfg.trainer.num_rounds) + 1), desc="Training"):
            if self.privacy_engine.is_exhausted():
                print(f"Privacy budget exhausted at round {round_num}.")
                break

            train_metrics = self.train_round(round_num)
            test_metrics = self.evaluate()
            last_eval = test_metrics

            self.history.rounds.append(round_num)
            self.history.train_metrics.append(train_metrics)
            self.history.test_metrics.append(test_metrics)
            self.history.clip_history.append(train_metrics.get("new_clip", 0.0))
            self.history.upload_ratio.append(train_metrics.get("avg_upload_ratio", 1.0))
            self.history.noise_multiplier.append(train_metrics.get("noise_multiplier", 0.0))
            self.history.sigma_agg.append(train_metrics.get("sigma_agg", 0.0))
            self.history.clip_snr_proxy.append(train_metrics.get("clip_snr_proxy", 0.0))
            self.history.signal_l2_norm.append(train_metrics.get("signal_l2_norm", 0.0))
            self.history.expected_noise_l2_norm.append(train_metrics.get("expected_noise_l2_norm", 0.0))
            self.history.snr_signal_to_noise.append(train_metrics.get("snr_signal_to_noise", 0.0))

            if self.privacy_engine.is_enabled():
                self.history.privacy_spent.append(train_metrics.get("epsilon_spent", 0.0))

            if round_num % int(self.cfg.trainer.log_interval) == 0:
                msg = (
                    f"Round {round_num:03d} | "
                    f"Acc={test_metrics['accuracy']:.2%} | "
                    f"Loss={test_metrics['loss']:.4f} | "
                    f"Clip={train_metrics.get('new_clip', 0.0):.4f} | "
                    f"Upload={train_metrics.get('avg_upload_ratio', 1.0):.2%} | "
                    f"clip_snr_proxy={train_metrics.get('clip_snr_proxy', 0.0):.4f} | "
                    f"signal={train_metrics.get('signal_l2_norm', 0.0):.4f} | "
                    f"expected_noise={train_metrics.get('expected_noise_l2_norm', 0.0):.4f} | "
                    f"SNR={train_metrics.get('snr_signal_to_noise', 0.0):.4f} | "
                    f"ImpFrozen={train_metrics.get('importance_frozen', 0.0):.0f}"
                )
                if self.privacy_engine.is_enabled():
                    msg += f" | eps={train_metrics.get('epsilon_spent', 0.0):.4f}"
                tqdm.write(msg)

            if test_metrics["accuracy"] > best_acc:
                best_acc = test_metrics["accuracy"]
                self._save_checkpoint(round_num, is_best=True)

            if round_num % int(self.cfg.trainer.save_interval) == 0:
                self._save_checkpoint(round_num)

        final_round = self.history.rounds[-1] if self.history.rounds else 0
        self._save_checkpoint(final_round, is_final=True)
        self._save_history()

        print("=" * 72)
        print("Training complete")
        print(f"Best accuracy: {best_acc:.2%}")
        if self.privacy_engine.is_enabled():
            print(f"Final epsilon: {self.privacy_engine.current_epsilon():.6f}")
        print("=" * 72)

        return {
            "best_accuracy": float(best_acc),
            "final_accuracy": float(last_eval["accuracy"]),
            "final_loss": float(last_eval["loss"]),
            "total_rounds": float(len(self.history.rounds)),
        }

    def _save_checkpoint(self, round_num: int, is_best: bool = False, is_final: bool = False) -> None:
        checkpoint = {
            "round": round_num,
            "model_state_dict": self.server.global_model.state_dict(),
            "clip_norm": self.server.clip_norm,
            "global_importance_template": self.server.global_importance_template,
            "clip_weight_template": self.server.clip_weight_template,
            "importance_frozen": self.server.importance_frozen,
            "config": OmegaConf.to_container(self.cfg, resolve=True),
        }

        if is_best:
            path = os.path.join(str(self.cfg.trainer.save_dir), "best_model.pt")
        elif is_final:
            path = os.path.join(str(self.cfg.trainer.save_dir), "final_model.pt")
        else:
            path = os.path.join(str(self.cfg.trainer.save_dir), f"checkpoint_round_{round_num}.pt")
        torch.save(checkpoint, path)

    def _save_history(self) -> None:
        history_data = {
            "rounds": self.history.rounds,
            "test_accuracy": [metric["accuracy"] for metric in self.history.test_metrics],
            "test_loss": [metric["loss"] for metric in self.history.test_metrics],
            "clip_history": self.history.clip_history,
            "privacy_spent": self.history.privacy_spent,
            "upload_ratio": self.history.upload_ratio,
            "noise_multiplier": self.history.noise_multiplier,
            "sigma_agg": self.history.sigma_agg,
            "clip_snr_proxy": self.history.clip_snr_proxy,
            "signal_l2_norm": self.history.signal_l2_norm,
            "expected_noise_l2_norm": self.history.expected_noise_l2_norm,
            "snr_signal_to_noise": self.history.snr_signal_to_noise,
        }

        summary = {
            "best_accuracy": max(history_data["test_accuracy"]) if history_data["test_accuracy"] else 0.0,
            "final_accuracy": history_data["test_accuracy"][-1] if history_data["test_accuracy"] else 0.0,
            "final_loss": history_data["test_loss"][-1] if history_data["test_loss"] else 0.0,
            "total_rounds": len(history_data["rounds"]),
            "final_epsilon": history_data["privacy_spent"][-1] if history_data["privacy_spent"] else 0.0,
            "final_clip_norm": history_data["clip_history"][-1] if history_data["clip_history"] else 0.0,
            "importance_frozen": self.server.importance_frozen,
            "importance_distance": self.server.last_importance_distance,
        }

        full_record = {
            "experiment_info": {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "device": str(self.device),
            },
            "config": OmegaConf.to_container(self.cfg, resolve=True),
            "summary": summary,
            "history": history_data,
        }

        record_path = os.path.join(str(self.cfg.trainer.log_dir), "experiment_record.json")
        with open(record_path, "w", encoding="utf-8") as handle:
            json.dump(full_record, handle, indent=2, ensure_ascii=False)

        history_path = os.path.join(str(self.cfg.trainer.log_dir), "history.json")
        with open(history_path, "w", encoding="utf-8") as handle:
            json.dump(history_data, handle, indent=2, ensure_ascii=False)


def copy_to_cpu(tensors):
    if tensors is None:
        return None
    return {name: tensor.detach().cpu().clone() for name, tensor in tensors.items()}


def load_config(argv: List[str]) -> DictConfig:
    base_cfg = OmegaConf.load(Path("configs") / "base.yaml")
    cli_cfg = OmegaConf.from_dotlist(argv)
    cfg = OmegaConf.merge(base_cfg, cli_cfg)

    experiment_name = str(cfg.experiment.name)
    if experiment_name in {"fedavg", "dp_fedavg", "dp_fedsam"}:
        cfg.compressor.type = "identity"
        cfg.compressor.use_residual = False
        cfg.compressor.topk_ratio = 1.0
        cfg.dp.use_global_importance_for_topk = False
        cfg.dp.enable_importance_freeze = False
        cfg.dp.global_local_mix_lambda = 0.0
    if experiment_name == "fedavg":
        cfg.dp.enabled = False
        cfg.dp.use_heterogeneous_noise = False
        cfg.client.training_strategy = "standard"
        cfg.server.initial_clip = 1_000.0
        cfg.server.clip_update_method = "ema"
        cfg.server.ema_alpha = 1.0
    elif experiment_name == "dp_fedavg":
        cfg.dp.enabled = True
        cfg.dp.use_heterogeneous_noise = False
        cfg.client.training_strategy = "standard"
        cfg.server.clip_update_method = "ema"
        cfg.server.ema_alpha = 1.0
    elif experiment_name == "dp_fedsam":
        cfg.dp.enabled = True
        cfg.dp.use_heterogeneous_noise = False
        cfg.client.training_strategy = "dp_fedsam"
        cfg.server.clip_update_method = "ema"
        cfg.server.ema_alpha = 1.0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg.trainer.save_dir = f"./checkpoints/{experiment_name}_{timestamp}"
    cfg.trainer.log_dir = f"./logs/{experiment_name}_{timestamp}"
    return cfg


def main(argv: Optional[List[str]] = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    cfg = load_config(argv)
    trainer = FLTrainer(cfg)
    results = trainer.train()
    print("Final results:")
    for key, value in results.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
