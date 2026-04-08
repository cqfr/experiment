from __future__ import annotations

"""Federated server implementation."""

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from config import ClipUpdateMethod, DownlinkStrategy, ServerConfig, StatsAggMethod


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
        client_updates: List[Dict[str, torch.Tensor]],
        data_sizes: List[int],
        stats: List[float],
        stats_agg_method: StatsAggMethod = StatsAggMethod.QUANTILE,
    ) -> AggregatedResult:
        global_update = self._weighted_aggregate(client_updates, data_sizes)
        stats_aggregated = self._aggregate_stats(stats, method=stats_agg_method)
        return AggregatedResult(global_update=global_update, stats_aggregated=stats_aggregated)

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
            "fraction_clipped": float(np.mean(arr > 0.5)),
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
