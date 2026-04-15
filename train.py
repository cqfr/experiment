from __future__ import annotations

"""Main federated training entry with OmegaConf and threaded client execution."""

import json
import math
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from client.client import FLClient
from components.strategies import PrivacyEngine
from data.utils import get_dataloader
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
        )

        self.model = self._build_model().cpu()
        self.server = FLServer(model=self.model, cfg=cfg, device=self.device)
        self.privacy_engine = PrivacyEngine(cfg.dp, num_rounds=int(cfg.trainer.num_rounds))

        self.clients: List[FLClient] = [
            FLClient(
                client_id=i,
                model=self._build_model(),
                dataloader=self.train_loaders[i],
                cfg=cfg,
                device=self.device,
            )
            for i in range(int(cfg.data.num_clients))
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

    def train_round(self, round_num: int) -> Dict[str, float]:
        clip_norm = self.server.get_clip_norm()
        clip_weight_template = None
        if bool(self.cfg.dp.enabled and self.cfg.dp.use_heterogeneous_noise):
            clip_weight_template = self.server.get_clip_weight_template()

        selected_ids = self._select_clients(int(self.cfg.data.clients_per_round))
        selected_sizes = [self.clients[idx].data_size for idx in selected_ids]
        client_weights = self._build_client_weights(selected_sizes)

        sampling_rate = float(self.cfg.data.clients_per_round) / float(self.cfg.data.num_clients)
        compressor_ratio = float(self.cfg.compressor.topk_ratio) if str(self.cfg.compressor.type) == "topk" else 1.0
        privacy_state = self.privacy_engine.calibrate_round(
            client_weights=client_weights,
            clip_norm=clip_norm,
            sampling_rate=sampling_rate,
            compressor_ratio=compressor_ratio,
        )

        global_weights = self.server.get_global_weights()
        client_updates = self._run_clients_parallel(
            selected_ids=selected_ids,
            global_weights=global_weights,
            clip_norm=clip_norm,
            clip_weight_template=clip_weight_template,
            round_num=round_num,
        )

        aggregated_importance = self.server.aggregate_importance(client_updates)
        clean_update = self.server._weighted_aggregate(
            [update.delta_w for update in client_updates],
            [update.data_size for update in client_updates],
        )

        server_generator = self._make_generator(round_num=round_num, client_id=-1)
        noisy_update = self.privacy_engine.add_server_noise(
            global_update=clean_update,
            sigma_agg=privacy_state.sigma_agg,
            use_heterogeneous_noise=bool(self.cfg.dp.use_heterogeneous_noise),
            clip_weight_template=clip_weight_template,
            generator=server_generator,
        )

        aggregated = self.server.aggregate(client_updates=client_updates, noisy_update=noisy_update)
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

        train_metrics.update(
            {
                "avg_upload_ratio": float(np.mean([u.upload_ratio for u in client_updates])) if client_updates else 1.0,
                "noise_multiplier": float(privacy_state.noise_multiplier),
                "sigma_agg": float(privacy_state.sigma_agg),
                "sensitivity_l2": float(privacy_state.sensitivity_l2),
                "clip_snr_proxy": float(privacy_state.clip_snr_proxy),
                "signal_l2_norm": float(aggregated.signal_l2_norm),
                "expected_noise_l2_norm": float(expected_noise_l2_norm),
                "snr_signal_to_noise": float(snr_signal_to_noise),
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
        round_num: int,
    ):
        client_updates = []
        with ThreadPoolExecutor(max_workers=int(self.cfg.trainer.max_workers)) as executor:
            futures = []
            for client_id in selected_ids:
                futures.append(
                    executor.submit(
                        self.clients[client_id].train_and_upload,
                        global_weights=copy_to_cpu(global_weights),
                        clip_norm=clip_norm,
                        clip_weight_template=copy_to_cpu(clip_weight_template),
                        return_importance_snapshot=self._need_viz(round_num, client_id),
                        importance_max_elements=int(self.cfg.importance_viz.max_elements),
                        generator=self._make_generator(round_num=round_num, client_id=client_id),
                    )
                )

            for future in as_completed(futures):
                client_updates.append(future.result())
        return client_updates

    def _need_viz(self, round_num: int, client_id: int) -> bool:
        return (
            bool(self.cfg.importance_viz.enabled)
            and round_num % max(1, int(self.cfg.importance_viz.interval)) == 0
            and client_id == int(self.cfg.importance_viz.client_id)
        )

    def _make_generator(self, round_num: int, client_id: int) -> torch.Generator:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(self.cfg.seed) * 100000 + round_num * 1000 + client_id + 17)
        return generator

    def _build_client_weights(self, selected_sizes: List[int]) -> List[float]:
        if str(self.cfg.server.aggregation_weight_strategy) == "equal":
            return [1.0 / max(1, len(selected_sizes)) for _ in selected_sizes]
        total_selected = sum(selected_sizes)
        if total_selected <= 0:
            return [1.0 / max(1, len(selected_sizes)) for _ in selected_sizes]
        return [size / total_selected for size in selected_sizes]

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
                    f"signal={train_metrics.get('signal_l2_norm', 0.0):.4f} | "
                    f"noise={train_metrics.get('expected_noise_l2_norm', 0.0):.4f} | "
                    f"SNR={train_metrics.get('snr_signal_to_noise', 0.0):.4f}"
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

    def _save_checkpoint(
        self,
        round_num: int,
        is_best: bool = False,
        is_final: bool = False,
    ) -> None:
        checkpoint = {
            "round": round_num,
            "model_state_dict": self.server.global_model.state_dict(),
            "clip_norm": self.server.clip_norm,
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
            "test_accuracy": [m["accuracy"] for m in self.history.test_metrics],
            "test_loss": [m["loss"] for m in self.history.test_metrics],
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

        os.makedirs(str(self.cfg.trainer.log_dir), exist_ok=True)
        record_path = os.path.join(str(self.cfg.trainer.log_dir), "experiment_record.json")
        with open(record_path, "w", encoding="utf-8") as f:
            json.dump(full_record, f, indent=2, ensure_ascii=False)

        history_path = os.path.join(str(self.cfg.trainer.log_dir), "history.json")
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history_data, f, indent=2, ensure_ascii=False)


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
    if experiment_name == "fedavg":
        cfg.dp.enabled = False
    if experiment_name == "dp_fedavg":
        cfg.client.training_strategy = "standard"
        cfg.dp.use_heterogeneous_noise = False
        cfg.server.clip_update_method = "ema"
        cfg.server.ema_alpha = 1.0
    if experiment_name == "dp_fedsam":
        cfg.client.training_strategy = "dp_fedsam"
        cfg.dp.use_heterogeneous_noise = False
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
    for k, v in results.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
