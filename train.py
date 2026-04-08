from __future__ import annotations

"""Main federated training entry."""

import argparse
import json
import math
import os
import random
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from client.client import FLClient
from config import (
    ExperimentConfig,
    get_dp_fedavg_config,
    get_dp_fedsam_config,
    get_fedavg_config,
    get_proposed_config,
)
from data.utils import get_dataloader
from dp.noise import RDPAccountant
from models.resnet import ResNet18
from server.server import FLServer


class FLTrainer:
    """Orchestrates client sampling, local updates, aggregation, and evaluation."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self._set_seed(config.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")

        self.train_loaders, self.test_loader = get_dataloader(
            num_clients=config.num_clients,
            batch_size=config.batch_size,
            alpha=config.alpha,
            iid=config.iid,
            dataset=config.dataset,
        )

        self.model = ResNet18(num_classes=config.num_classes).to(self.device)
        self.server = FLServer(model=self.model, config=config.server, device=self.device)

        self.clients: List[FLClient] = [
            FLClient(
                client_id=i,
                model=self.model,
                dataloader=self.train_loaders[i],
                config=config.client,
                dp_config=config.dp,
                device=self.device,
            )
            for i in range(config.num_clients)
        ]

        if config.dp.enabled:
            self.privacy_accountant = RDPAccountant(
                epsilon_total=config.dp.epsilon_total,
                delta=config.dp.delta,
                num_rounds=config.num_rounds,
                orders=config.dp.rdp_orders,
            )
        else:
            self.privacy_accountant = None

        self.history = {
            "rounds": [],
            "train_metrics": [],
            "test_metrics": [],
            "clip_history": [],
            "privacy_spent": [],
            "upload_ratio": [],
            "sigma_base": [],
        }

        os.makedirs(config.save_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _select_clients(self, count: int) -> List[int]:
        return random.sample(range(self.config.num_clients), count)

    def train_round(self, round_num: int) -> Dict[str, float]:
        clip_norm = self.server.get_clip_norm()
        sampling_rate = self.config.clients_per_round / self.config.num_clients

        if self.privacy_accountant is not None:
            sigma_base = self.privacy_accountant.solve_sigma_for_round(
                q=sampling_rate,
                steps=self.config.dp.rdp_steps_per_round,
            )
        else:
            sigma_base = 0.0

        selected_ids = self._select_clients(self.config.clients_per_round)
        global_weights = self.server.get_global_weights()

        client_deltas: List[Dict[str, torch.Tensor]] = []
        data_sizes: List[int] = []
        stats: List[float] = []
        noise_sigmas: List[float] = []
        upload_ratios: List[float] = []
        viz_client_id: Optional[int] = None
        viz_importance: Optional[torch.Tensor] = None
        viz_mask: Optional[torch.Tensor] = None

        for client_id in selected_ids:
            client = self.clients[client_id]
            need_viz = (
                self.config.importance_viz_enabled
                and (round_num % max(1, self.config.importance_viz_interval) == 0)
                and (client_id == self.config.importance_viz_client)
            )
            update = client.train_and_upload(
                global_weights=global_weights,
                clip_norm=clip_norm,
                sigma_base=sigma_base,
                return_importance_snapshot=need_viz,
                importance_max_elements=self.config.importance_viz_max_elements,
            )
            client_deltas.append(update.delta_w)
            data_sizes.append(update.data_size)
            stats.append(update.stat)
            noise_sigmas.append(update.noise_sigma)
            upload_ratios.append(update.upload_ratio)
            if need_viz and update.importance_vector is not None:
                viz_client_id = client_id
                viz_importance = update.importance_vector
                viz_mask = update.mask_vector

        aggregated = self.server.aggregate(
            client_updates=client_deltas,
            data_sizes=data_sizes,
            stats=stats,
            stats_agg_method=self.config.edge.stats_agg_method,
        )

        train_metrics = self.server.update_global_model(
            global_update=aggregated.global_update,
            stats_aggregated=aggregated.stats_aggregated,
        )

        train_metrics["avg_noise_sigma"] = (
            float(np.mean(noise_sigmas)) if noise_sigmas else 0.0
        )
        train_metrics["avg_upload_ratio"] = (
            float(np.mean(upload_ratios)) if upload_ratios else 1.0
        )
        train_metrics["sigma_base"] = float(sigma_base)
        if viz_importance is not None and viz_client_id is not None:
            self._save_importance_visualization(
                round_num=round_num,
                client_id=viz_client_id,
                importance_vector=viz_importance,
                mask_vector=viz_mask,
            )

        if self.privacy_accountant is not None:
            eps_spent = self.privacy_accountant.consume_round(
                sigma=sigma_base,
                q=sampling_rate,
                steps=self.config.dp.rdp_steps_per_round,
            )
            train_metrics["epsilon_spent"] = float(eps_spent)
            train_metrics["epsilon_remaining"] = float(
                self.privacy_accountant.remaining_budget()
            )

        return train_metrics

    def _save_importance_visualization(
        self,
        round_num: int,
        client_id: int,
        importance_vector: torch.Tensor,
        mask_vector: Optional[torch.Tensor] = None,
    ) -> None:
        """Save heatmap + raw snapshot for importance/mask vectors."""
        viz_dir = os.path.join(self.config.log_dir, "importance_viz")
        os.makedirs(viz_dir, exist_ok=True)

        vec = importance_vector.detach().float().cpu().numpy()
        side = int(math.ceil(math.sqrt(vec.size)))
        padded = np.full((side * side,), np.nan, dtype=np.float32)
        padded[: vec.size] = vec
        imp_mat = padded.reshape(side, side)

        mask_mat = None
        if mask_vector is not None:
            mvec = mask_vector.detach().float().cpu().numpy()
            mpad = np.full((side * side,), np.nan, dtype=np.float32)
            mpad[: min(mvec.size, side * side)] = mvec[: side * side]
            mask_mat = mpad.reshape(side, side)

        npz_path = os.path.join(viz_dir, f"round_{round_num:04d}_client_{client_id}.npz")
        np.savez_compressed(
            npz_path,
            importance_vector=vec,
            mask_vector=(
                mask_vector.detach().float().cpu().numpy()
                if mask_vector is not None
                else np.array([], dtype=np.float32)
            ),
        )

        try:
            import matplotlib.pyplot as plt

            cols = 2 if mask_mat is not None else 1
            fig, axes = plt.subplots(1, cols, figsize=(6 * cols, 5), dpi=120)
            if cols == 1:
                axes = [axes]

            ax0 = axes[0]
            im0 = ax0.imshow(np.log1p(np.nan_to_num(imp_mat, nan=0.0)), cmap="viridis", aspect="auto")
            ax0.set_title(f"Importance Heatmap (R{round_num}, C{client_id})")
            ax0.set_xlabel("Index")
            ax0.set_ylabel("Index")
            fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

            if mask_mat is not None:
                ax1 = axes[1]
                im1 = ax1.imshow(np.nan_to_num(mask_mat, nan=0.0), cmap="gray_r", vmin=0.0, vmax=1.0, aspect="auto")
                ax1.set_title("Top-k Mask Snapshot")
                ax1.set_xlabel("Index")
                ax1.set_ylabel("Index")
                fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

            fig.tight_layout()
            png_path = os.path.join(viz_dir, f"round_{round_num:04d}_client_{client_id}.png")
            fig.savefig(png_path)
            plt.close(fig)
        except Exception as exc:
            print(f"[importance_viz] plot failed at round {round_num}: {exc}")

    def evaluate(self) -> Dict[str, float]:
        return self.server.evaluate(self.test_loader)

    def train(self) -> Dict[str, float]:
        print("=" * 72)
        print("Start federated training")
        print("=" * 72)

        best_acc = 0.0
        last_eval = {"accuracy": 0.0, "loss": 0.0}

        for round_num in tqdm(range(1, self.config.num_rounds + 1), desc="Training"):
            if self.privacy_accountant is not None and self.privacy_accountant.is_exhausted():
                print(f"Privacy budget exhausted at round {round_num}.")
                break

            train_metrics = self.train_round(round_num)
            test_metrics = self.evaluate()
            last_eval = test_metrics

            self.history["rounds"].append(round_num)
            self.history["train_metrics"].append(train_metrics)
            self.history["test_metrics"].append(test_metrics)
            self.history["clip_history"].append(train_metrics.get("new_clip", 0.0))
            self.history["upload_ratio"].append(train_metrics.get("avg_upload_ratio", 1.0))
            self.history["sigma_base"].append(train_metrics.get("sigma_base", 0.0))

            if self.privacy_accountant is not None:
                self.history["privacy_spent"].append(train_metrics.get("epsilon_spent", 0.0))

            if round_num % self.config.log_interval == 0:
                msg = (
                    f"Round {round_num:03d} | "
                    f"Acc={test_metrics['accuracy']:.2%} | "
                    f"Loss={test_metrics['loss']:.4f} | "
                    f"Clip={train_metrics.get('new_clip', 0.0):.4f} | "
                    f"Upload={train_metrics.get('avg_upload_ratio', 1.0):.2%}"
                )
                if self.privacy_accountant is not None:
                    msg += (
                        f" | eps={train_metrics.get('epsilon_spent', 0.0):.4f}"
                        f" | sigma={train_metrics.get('sigma_base', 0.0):.4f}"
                    )
                tqdm.write(msg)

            if test_metrics["accuracy"] > best_acc:
                best_acc = test_metrics["accuracy"]
                self._save_checkpoint(round_num, is_best=True)

            if round_num % self.config.save_interval == 0:
                self._save_checkpoint(round_num)

        final_round = self.history["rounds"][-1] if self.history["rounds"] else 0
        self._save_checkpoint(final_round, is_final=True)
        self._save_history()

        print("=" * 72)
        print("Training complete")
        print(f"Best accuracy: {best_acc:.2%}")
        if self.privacy_accountant is not None:
            print(f"Final epsilon: {self.privacy_accountant.current_epsilon():.6f}")
        print("=" * 72)

        return {
            "best_accuracy": float(best_acc),
            "final_accuracy": float(last_eval["accuracy"]),
            "final_loss": float(last_eval["loss"]),
            "total_rounds": float(len(self.history["rounds"])),
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
            "config": self._config_to_dict(self.config),
        }

        if is_best:
            path = os.path.join(self.config.save_dir, "best_model.pt")
        elif is_final:
            path = os.path.join(self.config.save_dir, "final_model.pt")
        else:
            path = os.path.join(self.config.save_dir, f"checkpoint_round_{round_num}.pt")

        torch.save(checkpoint, path)

    def _save_history(self) -> None:
        history_data = {
            "rounds": self.history["rounds"],
            "test_accuracy": [m["accuracy"] for m in self.history["test_metrics"]],
            "test_loss": [m["loss"] for m in self.history["test_metrics"]],
            "clip_history": self.history["clip_history"],
            "privacy_spent": self.history["privacy_spent"],
            "noise_sigma": [m.get("avg_noise_sigma", 0.0) for m in self.history["train_metrics"]],
            "upload_ratio": self.history["upload_ratio"],
            "sigma_base": self.history["sigma_base"],
        }

        summary = {
            "best_accuracy": max(history_data["test_accuracy"]) if history_data["test_accuracy"] else 0.0,
            "final_accuracy": history_data["test_accuracy"][-1] if history_data["test_accuracy"] else 0.0,
            "final_loss": history_data["test_loss"][-1] if history_data["test_loss"] else 0.0,
            "total_rounds": len(history_data["rounds"]),
            "final_epsilon": history_data["privacy_spent"][-1] if history_data["privacy_spent"] else 0.0,
            "final_clip_norm": history_data["clip_history"][-1] if history_data["clip_history"] else 0.0,
            "final_upload_ratio": history_data["upload_ratio"][-1] if history_data["upload_ratio"] else 1.0,
        }

        full_record = {
            "experiment_info": {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "device": str(self.device),
            },
            "config": self._config_to_dict(self.config),
            "summary": summary,
            "history": history_data,
        }

        os.makedirs(self.config.log_dir, exist_ok=True)
        record_path = os.path.join(self.config.log_dir, "experiment_record.json")
        with open(record_path, "w", encoding="utf-8") as f:
            json.dump(full_record, f, indent=2, ensure_ascii=False)

        history_path = os.path.join(self.config.log_dir, "history.json")
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history_data, f, indent=2, ensure_ascii=False)

        summary_path = os.path.join(self.config.log_dir, "config_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(self._config_to_dict(self.config), indent=2, ensure_ascii=False))

    @staticmethod
    def _config_to_dict(config: ExperimentConfig) -> dict:
        return {
            "seed": config.seed,
            "num_clients": config.num_clients,
            "clients_per_round": config.clients_per_round,
            "num_rounds": config.num_rounds,
            "batch_size": config.batch_size,
            "dataset": config.dataset,
            "iid": config.iid,
            "alpha": config.alpha,
            "model": config.model,
            "num_classes": config.num_classes,
            "client": {
                "training_strategy": config.client.training_strategy.value,
                "local_epochs": config.client.local_epochs,
                "lr": config.client.lr,
                "momentum": config.client.momentum,
                "weight_decay": config.client.weight_decay,
                "alpha": config.client.alpha,
                "beta": config.client.beta,
                "contrastive_margin": config.client.contrastive_margin,
                "mu": config.client.mu,
                "sam_rho": config.client.sam_rho,
                "importance_strategy": config.client.importance_strategy.value,
                "topk_strategy": config.client.topk_strategy.value,
                "topk_ratio": config.client.topk_ratio,
                "layer_weight_method": config.client.layer_weight_method.value,
                "stat_type": config.client.stat_type.value,
                "use_residual": config.client.use_residual,
            },
            "server": {
                "server_lr": config.server.server_lr,
                "initial_clip": config.server.initial_clip,
                "clip_update_method": config.server.clip_update_method.value,
                "target_quantile": config.server.target_quantile,
                "clip_lr": config.server.clip_lr,
                "ema_alpha": config.server.ema_alpha,
                "downlink_strategy": config.server.downlink_strategy.value,
                "downlink_topk_ratio": config.server.downlink_topk_ratio,
            },
            "dp": {
                "enabled": config.dp.enabled,
                "epsilon_total": config.dp.epsilon_total,
                "delta": config.dp.delta,
                "use_heterogeneous_noise": config.dp.use_heterogeneous_noise,
                "min_relative_noise": config.dp.min_relative_noise,
                "max_relative_noise": config.dp.max_relative_noise,
                "rdp_orders": list(config.dp.rdp_orders),
                "rdp_steps_per_round": config.dp.rdp_steps_per_round,
                "trusted_server_for_stats": config.dp.trusted_server_for_stats,
            },
            "importance_viz": {
                "enabled": config.importance_viz_enabled,
                "interval": config.importance_viz_interval,
                "client_id": config.importance_viz_client,
                "max_elements": config.importance_viz_max_elements,
            },
        }


def run_experiment(experiment_name: str, config: ExperimentConfig) -> Dict[str, float]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.save_dir = f"./checkpoints/{experiment_name}_{timestamp}"
    config.log_dir = f"./logs/{experiment_name}_{timestamp}"

    trainer = FLTrainer(config)
    return trainer.train()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Federated learning training")
    parser.add_argument(
        "--experiment",
        type=str,
        default="proposed",
        choices=["fedavg", "dp_fedavg", "dp_fedsam", "proposed"],
    )
    parser.add_argument("--num_rounds", type=int, default=100)
    parser.add_argument("--num_clients", type=int, default=100)
    parser.add_argument("--clients_per_round", type=int, default=10)
    parser.add_argument("--epsilon", type=float, default=8.0)
    parser.add_argument("--topk_ratio", type=float, default=0.1)
    parser.add_argument("--iid", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--dataset", type=str, choices=["cifar10", "mnist"], default="cifar10")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--importance_viz", action="store_true")
    parser.add_argument("--importance_viz_interval", type=int, default=10)
    parser.add_argument("--importance_viz_client", type=int, default=0)
    parser.add_argument("--importance_viz_max_elements", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.experiment == "fedavg":
        config = get_fedavg_config()
    elif args.experiment == "dp_fedavg":
        config = get_dp_fedavg_config(epsilon=args.epsilon)
    elif args.experiment == "dp_fedsam":
        config = get_dp_fedsam_config(epsilon=args.epsilon)
    else:
        config = get_proposed_config(epsilon=args.epsilon)

    config.num_rounds = args.num_rounds
    config.num_clients = args.num_clients
    config.clients_per_round = args.clients_per_round
    config.iid = args.iid
    config.alpha = args.alpha
    config.dataset = args.dataset
    config.seed = args.seed
    config.importance_viz_enabled = args.importance_viz
    config.importance_viz_interval = args.importance_viz_interval
    config.importance_viz_client = args.importance_viz_client
    config.importance_viz_max_elements = args.importance_viz_max_elements

    if args.experiment in {"proposed", "dp_fedavg", "dp_fedsam"}:
        config.client.topk_ratio = args.topk_ratio

    results = run_experiment(args.experiment, config)
    print("Final results:")
    for k, v in results.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
