from __future__ import annotations

"""Federated server implementation."""

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from config import ClipUpdateMethod, DPConfig, DownlinkStrategy, ServerConfig, StatsAggMethod
from dp.noise import add_heterogeneous_noise, add_noise_to_tensor, allocate_relative_scales

if TYPE_CHECKING:
    from client.client import ClientUpdate


@dataclass
class AggregatedResult:
    """Aggregation output on server side."""

    global_update: Dict[str, torch.Tensor]
    stats_aggregated: Dict[str, float]


class FLServer:
    """Server for weighted aggregation, global update, and adaptive clipping."""

    def __init__(self, model: nn.Module, config: ServerConfig, device: torch.device):
        self.config = config
        self.device = device
        self.global_model = copy.deepcopy(model).to(device)

        self.clip_norm = float(config.initial_clip)
        self.clip_history: List[float] = []
        self.stats_history: List[Dict[str, float]] = []
        self.current_round = 0

    def get_global_weights(self) -> Dict[str, torch.Tensor]:
        return copy.deepcopy(self.global_model.state_dict())

    def get_clip_norm(self) -> float:
        return self.clip_norm

    def aggregate(
        self,
        client_updates: List[ClientUpdate],
        sigma_agg: float,
        dp_config: DPConfig,
        stats_agg_method: StatsAggMethod = StatsAggMethod.QUANTILE,
    ) -> AggregatedResult:
        deltas = [update.delta_w for update in client_updates]
        data_sizes = [update.data_size for update in client_updates]
        stats = [update.stat for update in client_updates]

        # Step A: weighted aggregation of clean client updates.
        global_update = self._weighted_aggregate(deltas, data_sizes)

        # Step B: central DP noise on the aggregated update.
        if dp_config.enabled and sigma_agg > 0 and global_update:
            global_mask = {
                name: (tensor != 0).float()
                for name, tensor in global_update.items()
            }

            if not dp_config.use_heterogeneous_noise:
                noisy_update: Dict[str, torch.Tensor] = {}
                for name, tensor in global_update.items():
                    noise = add_noise_to_tensor(torch.zeros_like(tensor), sigma_agg)
                    noisy_update[name] = tensor + noise * global_mask[name]
                global_update = noisy_update
            else:
                global_update = self._add_central_heterogeneous_noise(
                    global_update=global_update,
                    global_mask=global_mask,
                    client_updates=client_updates,
                    sigma_agg=sigma_agg,
                    dp_config=dp_config,
                )

        # Step C: aggregate statistics for clipping control.
        stats_aggregated = self._aggregate_stats(stats, method=stats_agg_method)
        return AggregatedResult(global_update=global_update, stats_aggregated=stats_aggregated)

    def _add_central_heterogeneous_noise(
        self,
        global_update: Dict[str, torch.Tensor],
        global_mask: Dict[str, torch.Tensor],
        client_updates: List[ClientUpdate],
        sigma_agg: float,
        dp_config: DPConfig,
    ) -> Dict[str, torch.Tensor]:
        importance_sum = {
            name: torch.zeros_like(tensor)
            for name, tensor in global_update.items()
        }
        importance_count = 0

        for update in client_updates:
            if update.importance_dict is None:
                continue
            importance_count += 1
            for name in importance_sum.keys():
                imp = update.importance_dict.get(name)
                if imp is not None:
                    importance_sum[name] = importance_sum[name] + imp.to(importance_sum[name].device)

        if importance_count > 0:
            global_importance = {
                name: tensor / float(importance_count)
                for name, tensor in importance_sum.items()
            }
        else:
            global_importance = {
                name: tensor.abs()
                for name, tensor in global_update.items()
            }

        flat_delta = []
        flat_mask = []
        flat_importance = []
        slices = []

        offset = 0
        for name, tensor in global_update.items():
            delta_flat = tensor.flatten()
            mask_flat = global_mask[name].flatten()
            importance_flat = global_importance[name].flatten()

            flat_delta.append(delta_flat)
            flat_mask.append(mask_flat)
            flat_importance.append(importance_flat)
            slices.append((name, offset, offset + delta_flat.numel(), tensor.shape))
            offset += delta_flat.numel()

        if not flat_delta:
            return global_update

        global_delta_flat = torch.cat(flat_delta)
        global_mask_flat = torch.cat(flat_mask)
        global_importance_flat = torch.cat(flat_importance)

        rel_scales = allocate_relative_scales(
            importance=global_importance_flat,
            mask=global_mask_flat,
            min_scale=dp_config.min_relative_noise,
            max_scale=dp_config.max_relative_noise,
            mode=dp_config.relative_noise_mode,
            alpha=dp_config.relative_noise_alpha,
            eps=dp_config.relative_noise_eps,
        )

        noisy_delta_flat, _ = add_heterogeneous_noise(
            tensor=global_delta_flat,
            sigma_base=sigma_agg,
            relative_scales=rel_scales,
            mask=global_mask_flat,
        )

        noisy_update: Dict[str, torch.Tensor] = {}
        for name, start, end, shape in slices:
            noisy_update[name] = noisy_delta_flat[start:end].reshape(shape)

        return noisy_update

    def _weighted_aggregate(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        data_sizes: List[int],
    ) -> Dict[str, torch.Tensor]:
        if not client_updates:
            return {}

        total_data = max(1, sum(data_sizes))
        weights = [size / total_data for size in data_sizes]

        global_update: Dict[str, torch.Tensor] = {}
        for name in client_updates[0].keys():
            acc = None
            for i, update in enumerate(client_updates):
                if name not in update:
                    continue
                term = update[name] * weights[i]
                acc = term.clone() if acc is None else (acc + term)
            if acc is not None:
                global_update[name] = acc
        return global_update

    def _aggregate_stats(self, stats: List[float], method: StatsAggMethod) -> Dict[str, float]:
        if not stats:
            return {
                "median": 0.0,
                "q25": 0.0,
                "q75": 0.0,
                "count": 0.0,
                "fraction_clipped": 0.0,
            }

        arr = np.array(stats, dtype=np.float64)
        out = {
            "median": float(np.quantile(arr, 0.5)),
            "q25": float(np.quantile(arr, 0.25)),
            "q75": float(np.quantile(arr, 0.75)),
            "count": float(arr.size),
            "fraction_clipped": float(np.mean(arr > self.clip_norm)),
        }
        if method == StatsAggMethod.ALL:
            out["mean"] = float(np.mean(arr))
            out["std"] = float(np.std(arr))
        return out

    def update_global_model(
        self,
        global_update: Dict[str, torch.Tensor],
        stats_aggregated: Dict[str, float],
    ) -> Dict[str, float]:
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                if name in global_update:
                    param.data += self.config.server_lr * global_update[name].to(self.device)

        old_clip = self.clip_norm
        self._update_clip_norm(stats_aggregated)

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

    def _update_clip_norm(self, stats: Dict[str, float]) -> None:
        method = self.config.clip_update_method

        if method == ClipUpdateMethod.ADAPTIVE:
            fraction_clipped = float(stats.get("fraction_clipped", 0.5))
            if fraction_clipped > self.config.target_quantile:
                self.clip_norm *= 1.0 + self.config.clip_lr
            else:
                self.clip_norm *= 1.0 - self.config.clip_lr
        elif method == ClipUpdateMethod.EMA:
            median = float(stats.get("median", self.clip_norm))
            alpha = self.config.ema_alpha
            self.clip_norm = alpha * self.clip_norm + (1.0 - alpha) * median

        self.clip_norm = float(max(1e-2, min(1e3, self.clip_norm)))

    def prepare_broadcast(self) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        if self.config.downlink_strategy == DownlinkStrategy.FULL:
            return self.get_global_weights(), None

        weights = self.get_global_weights()
        all_values = torch.cat([w.abs().flatten() for w in weights.values()])
        k = max(1, int(all_values.numel() * self.config.downlink_topk_ratio))
        threshold = torch.topk(all_values, k).values[-1]

        sparse_weights: Dict[str, torch.Tensor] = {}
        masks: Dict[str, torch.Tensor] = {}
        for name, w in weights.items():
            m = (w.abs() >= threshold).float()
            masks[name] = m
            sparse_weights[name] = w * m

        return sparse_weights, masks

    def evaluate(self, test_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        self.global_model.eval()
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
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


if __name__ == "__main__":
    print("server module ready")
