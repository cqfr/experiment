from __future__ import annotations

"""Optional edge-side aggregation helpers."""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch

from config import EdgeConfig, StatsAggMethod


@dataclass
class EdgeOutput:
    global_update: Dict[str, torch.Tensor]
    stats_aggregated: Dict[str, float]


class TrustedEdge:
    """Trusted edge aggregator mirroring server-side aggregation logic."""

    def __init__(self, config: EdgeConfig):
        self.config = config

    def aggregate(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        data_sizes: List[int],
        stats: List[float],
    ) -> EdgeOutput:
        global_update = self._weighted_aggregate(client_updates, data_sizes)
        stats_aggregated = self._aggregate_stats(stats)
        return EdgeOutput(global_update=global_update, stats_aggregated=stats_aggregated)

    @staticmethod
    def _weighted_aggregate(
        client_updates: List[Dict[str, torch.Tensor]],
        data_sizes: List[int],
    ) -> Dict[str, torch.Tensor]:
        if not client_updates:
            return {}

        total_data = max(1, sum(data_sizes))
        weights = [size / total_data for size in data_sizes]

        out: Dict[str, torch.Tensor] = {}
        for name in client_updates[0].keys():
            acc = None
            for i, update in enumerate(client_updates):
                if name not in update:
                    continue
                term = update[name] * weights[i]
                acc = term.clone() if acc is None else (acc + term)
            if acc is not None:
                out[name] = acc
        return out

    def _aggregate_stats(self, stats: List[float]) -> Dict[str, float]:
        if not stats:
            return {
                "median": 0.0,
                "q25": 0.0,
                "q75": 0.0,
                "fraction_clipped": 0.0,
                "count": 0.0,
            }

        arr = np.array(stats, dtype=np.float64)
        out = {
            "median": float(np.quantile(arr, 0.5)),
            "q25": float(np.quantile(arr, 0.25)),
            "q75": float(np.quantile(arr, 0.75)),
            "fraction_clipped": float(np.mean(arr > 0.5)),
            "count": float(arr.size),
        }
        if self.config.stats_agg_method == StatsAggMethod.ALL:
            out["mean"] = float(np.mean(arr))
            out["std"] = float(np.std(arr))
        return out


class SimpleAggregator:
    """Simple FedAvg aggregation helper."""

    @staticmethod
    def fedavg_aggregate(
        client_updates: List[Dict[str, torch.Tensor]],
        data_sizes: Optional[List[int]] = None,
    ) -> Dict[str, torch.Tensor]:
        if not client_updates:
            return {}

        num_clients = len(client_updates)
        if data_sizes is not None:
            total_data = max(1, sum(data_sizes))
            weights = [size / total_data for size in data_sizes]
        else:
            weights = [1.0 / num_clients] * num_clients

        out: Dict[str, torch.Tensor] = {}
        for name in client_updates[0].keys():
            acc = None
            for i, update in enumerate(client_updates):
                if name not in update:
                    continue
                term = update[name] * weights[i]
                acc = term.clone() if acc is None else (acc + term)
            if acc is not None:
                out[name] = acc
        return out


if __name__ == "__main__":
    print("edge module ready")
