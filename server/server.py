from __future__ import annotations

"""Federated server implementation."""

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

TensorDict = Dict[str, torch.Tensor]


@dataclass
class AggregatedResult:
    clean_update: TensorDict
    noisy_update: TensorDict
    stats_aggregated: Dict[str, float]
    signal_l2_norm: float
    total_params: int


class FLServer:
    """Server for aggregation, clip-template maintenance, and evaluation."""

    def __init__(self, model: nn.Module, cfg, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.global_model = copy.deepcopy(model).to(device)

        self.clip_norm = float(cfg.server.initial_clip)
        self.clip_history: List[float] = []
        self.stats_history: List[Dict[str, float]] = []
        self.current_round = 0

        self.clip_weight_template: TensorDict = {
            name: torch.ones_like(param.detach()).cpu()
            for name, param in self.global_model.named_parameters()
            if param.requires_grad
        }
        self.importance_ema_template: TensorDict = {
            name: torch.ones_like(param.detach()).cpu()
            for name, param in self.global_model.named_parameters()
            if param.requires_grad
        }

    def get_global_weights(self) -> TensorDict:
        return copy.deepcopy(self.global_model.state_dict())

    def get_clip_norm(self) -> float:
        return self.clip_norm

    def get_clip_weight_template(self) -> TensorDict:
        return copy.deepcopy(self.clip_weight_template)

    def aggregate(self, client_updates, noisy_update: TensorDict) -> AggregatedResult:
        deltas = [update.delta_w for update in client_updates]
        data_sizes = [update.data_size for update in client_updates]

        clean_update = self._weighted_aggregate(deltas, data_sizes)
        signal_l2_norm = self._l2_norm(clean_update)
        total_params = sum(tensor.numel() for tensor in clean_update.values())
        stats_aggregated = self._aggregate_stats(client_updates)

        return AggregatedResult(
            clean_update=clean_update,
            noisy_update=noisy_update,
            stats_aggregated=stats_aggregated,
            signal_l2_norm=signal_l2_norm,
            total_params=total_params,
        )

    def update_global_model(
        self,
        global_update: TensorDict,
        stats_aggregated: Dict[str, float],
        aggregated_importance: Optional[TensorDict] = None,
    ) -> Dict[str, float]:
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                if name in global_update:
                    param.add_(float(self.cfg.server.server_lr) * global_update[name].to(self.device))

        old_clip = self.clip_norm
        self._update_clip_norm(stats_aggregated)
        if aggregated_importance is not None and bool(self.cfg.dp.use_heterogeneous_noise):
            self._update_clip_weight_template(aggregated_importance)

        self.current_round += 1
        self.stats_history.append(stats_aggregated)
        self.clip_history.append(self.clip_norm)

        return {
            "round": float(self.current_round),
            "old_clip": float(old_clip),
            "new_clip": float(self.clip_norm),
            "stats_median": float(stats_aggregated.get("median", 0.0)),
            "fraction_clipped": float(stats_aggregated.get("fraction_clipped", 0.0)),
        }

    def evaluate(self, test_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        self.global_model.eval()
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                logits = self.global_model(data)
                loss = criterion(logits, target)
                total_loss += float(loss.item()) * data.size(0)
                pred = logits.argmax(dim=1)
                correct += int((pred == target).sum().item())
                total += data.size(0)

        self.global_model.train()
        return {
            "loss": total_loss / max(1, total),
            "accuracy": correct / max(1, total),
            "correct": float(correct),
            "total": float(total),
        }

    def aggregate_importance(self, client_updates) -> Optional[TensorDict]:
        importance_updates = [update.importance_dict for update in client_updates if update.importance_dict is not None]
        if not importance_updates:
            return None

        aggregated: TensorDict = {}
        for name in importance_updates[0].keys():
            tensors = [importance[name].float() for importance in importance_updates if name in importance]
            if tensors:
                aggregated[name] = torch.stack(tensors, dim=0).mean(dim=0)
        return aggregated if aggregated else None

    def _weighted_aggregate(self, client_updates: List[TensorDict], data_sizes: List[int]) -> TensorDict:
        if not client_updates:
            return {}

        if str(self.cfg.server.aggregation_weight_strategy) == "equal":
            weights = [1.0 / max(1, len(client_updates)) for _ in client_updates]
        else:
            total_data = sum(data_sizes)
            if total_data <= 0:
                weights = [1.0 / max(1, len(client_updates)) for _ in client_updates]
            else:
                weights = [size / total_data for size in data_sizes]

        aggregated: TensorDict = {}
        for name in client_updates[0].keys():
            acc = None
            for idx, update in enumerate(client_updates):
                if name not in update:
                    continue
                term = update[name] * weights[idx]
                acc = term.clone() if acc is None else acc + term
            if acc is not None:
                aggregated[name] = acc
        return aggregated

    def _update_clip_norm(self, stats: Dict[str, float]) -> None:
        method = str(self.cfg.server.clip_update_method)
        if method == "adaptive":
            fraction_clipped = float(stats.get("fraction_clipped", 0.0))
            if fraction_clipped > float(self.cfg.server.target_quantile):
                self.clip_norm *= 1.0 + float(self.cfg.server.clip_lr)
            else:
                self.clip_norm *= 1.0 - float(self.cfg.server.clip_lr)
        elif method == "ema":
            median = float(stats.get("median", self.clip_norm))
            alpha = float(self.cfg.server.ema_alpha)
            self.clip_norm = alpha * self.clip_norm + (1.0 - alpha) * median

        self.clip_norm = float(max(1e-3, min(1e3, self.clip_norm)))

    def _aggregate_stats(self, client_updates) -> Dict[str, float]:
        if not client_updates:
            return {
                "median": 0.0,
                "q25": 0.0,
                "q75": 0.0,
                "count": 0.0,
                "fraction_clipped": 0.0,
            }

        norms = np.array([float(update.stat) for update in client_updates], dtype=np.float64)
        clipped_flags = np.array([1.0 if update.clipped else 0.0 for update in client_updates], dtype=np.float64)
        return {
            "median": float(np.quantile(norms, 0.5)),
            "q25": float(np.quantile(norms, 0.25)),
            "q75": float(np.quantile(norms, 0.75)),
            "count": float(norms.size),
            "fraction_clipped": float(clipped_flags.mean()),
        }

    def _update_clip_weight_template(self, aggregated_importance: TensorDict) -> None:
        alpha = float(self.cfg.dp.template_ema)
        updated_importance_ema: TensorDict = {}
        updated_clip_template: TensorDict = {}

        for name, current_template in self.clip_weight_template.items():
            public_importance = aggregated_importance.get(name)
            if public_importance is None:
                updated_importance_ema[name] = self.importance_ema_template[name]
                updated_clip_template[name] = current_template
                continue

            sanitized_importance = torch.nan_to_num(public_importance.detach().float().abs(), nan=0.0, posinf=0.0, neginf=0.0).cpu()
            prev_ema = self.importance_ema_template[name]
            next_ema = alpha * prev_ema + (1.0 - alpha) * sanitized_importance
            updated_importance_ema[name] = next_ema
            updated_clip_template[name] = self._importance_to_template(next_ema)

        self.importance_ema_template = updated_importance_ema
        self.clip_weight_template = updated_clip_template

    def _importance_to_template(self, importance: torch.Tensor) -> torch.Tensor:
        if importance.numel() == 0:
            return torch.ones_like(importance)

        mean_importance = importance.mean()
        if mean_importance <= 0:
            normalized = torch.ones_like(importance)
        else:
            normalized = importance / mean_importance

        inverse_relative_scale = normalized.clamp_min(1e-6).reciprocal()
        clipped = inverse_relative_scale.clamp(
            min=float(self.cfg.dp.min_relative_noise),
            max=float(self.cfg.dp.max_relative_noise),
        )
        rms = torch.sqrt(torch.mean(clipped.pow(2))).clamp_min(1e-6)
        return (clipped / rms).cpu()

    @staticmethod
    def _l2_norm(tensors: TensorDict) -> float:
        if not tensors:
            return 0.0
        total = sum(tensor.float().pow(2).sum() for tensor in tensors.values())
        return float(torch.sqrt(total).item())
