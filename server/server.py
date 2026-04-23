from __future__ import annotations

"""Federated server implementation."""

import copy
import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from components.strategies import normalize_importance_dict, resolve_amp_dtype, resolve_amp_enabled

TensorDict = Dict[str, torch.Tensor]


@dataclass
class AggregatedResult:
    clean_update: TensorDict
    noisy_update: TensorDict
    stats_aggregated: Dict[str, float]
    signal_l2_norm: float
    total_params: int


@dataclass
class BroadcastPayload:
    global_weights: TensorDict
    clip_weight_template: Optional[TensorDict]
    global_importance_template: Optional[TensorDict]


class FLServer:
    """Server for aggregation, clip-template maintenance, and evaluation."""

    def __init__(self, model: nn.Module, cfg, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.global_model = copy.deepcopy(model).to(device)
        self.channels_last = bool(getattr(cfg.trainer, "channels_last", False) and device.type == "cuda")
        self.use_amp = resolve_amp_enabled(cfg.trainer, device)
        self.amp_dtype = resolve_amp_dtype(getattr(cfg.trainer, "amp_dtype", "float16"), device)
        if self.channels_last:
            self.global_model = self.global_model.to(memory_format=torch.channels_last)

        self.clip_norm = float(cfg.server.initial_clip)
        self.clip_history: List[float] = []
        self.stats_history: List[Dict[str, float]] = []
        self.current_round = 0

        self.global_importance_template: TensorDict = {
            name: torch.ones_like(param.detach()).cpu()
            for name, param in self.global_model.named_parameters()
            if param.requires_grad
        }
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
        self.importance_frozen = False
        self.freeze_counter = 0
        self.last_importance_distance = 0.0

    def get_global_weights(self) -> TensorDict:
        return copy.deepcopy(self.global_model.state_dict())

    def build_broadcast_payload(
        self,
        include_clip_weight_template: bool,
        include_global_importance_template: bool,
    ) -> BroadcastPayload:
        global_weights = {
            name: tensor.detach().cpu().clone()
            for name, tensor in self.global_model.state_dict().items()
        }
        clip_weight_template = (
            {name: tensor.detach().cpu().clone() for name, tensor in self.clip_weight_template.items()}
            if include_clip_weight_template
            else None
        )
        global_importance_template = (
            {name: tensor.detach().cpu().clone() for name, tensor in self.global_importance_template.items()}
            if include_global_importance_template
            else None
        )
        return BroadcastPayload(
            global_weights=global_weights,
            clip_weight_template=clip_weight_template,
            global_importance_template=global_importance_template,
        )

    def get_clip_norm(self) -> float:
        return self.clip_norm

    def get_clip_weight_template(self) -> TensorDict:
        return copy.deepcopy(self.clip_weight_template)

    def get_global_importance_template(self) -> TensorDict:
        return copy.deepcopy(self.global_importance_template)

    def should_request_importance(self) -> bool:
        # [MOD][阶段3] 冻结后不再强制 client 每轮上传 importance，降低通信与 server 端模板波动。
        needs_global_template = bool(getattr(self.cfg.dp, "use_global_importance_for_topk", True)) or bool(
            self.cfg.dp.use_heterogeneous_noise
        )
        return needs_global_template and not self.importance_frozen

    def should_refresh_local_importance(self, round_num: int) -> bool:
        if not self.importance_frozen:
            return True
        interval = max(1, int(getattr(self.cfg.dp, "local_importance_refresh_interval_after_freeze", 5)))
        return round_num % interval == 0

    def aggregate(self, client_updates) -> AggregatedResult:
        deltas = [update.delta_w for update in client_updates]
        data_sizes = [update.data_size for update in client_updates]

        clean_update = self._weighted_aggregate(deltas, data_sizes)
        signal_l2_norm = self._l2_norm(clean_update)
        total_params = sum(tensor.numel() for tensor in clean_update.values())
        stats_aggregated = self._aggregate_stats(client_updates)

        return AggregatedResult(
            clean_update=clean_update,
            # [MOD][阶段1] server 仅聚合 client 已带噪更新，不再进行第二次中心化加噪。
            noisy_update={name: tensor.detach().clone() for name, tensor in clean_update.items()},
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
        needs_global_template = bool(getattr(self.cfg.dp, "use_global_importance_for_topk", True)) or bool(
            self.cfg.dp.use_heterogeneous_noise
        )
        if aggregated_importance is not None and needs_global_template:
            self._update_importance_templates(aggregated_importance)

        self.current_round += 1
        self.stats_history.append(stats_aggregated)
        self.clip_history.append(self.clip_norm)

        return {
            "round": float(self.current_round),
            "old_clip": float(old_clip),
            "new_clip": float(self.clip_norm),
            "stats_median": float(stats_aggregated.get("median", 0.0)),
            "fraction_clipped": float(stats_aggregated.get("fraction_clipped", 0.0)),
            "fraction_unclipped": float(stats_aggregated.get("fraction_unclipped", 0.0)),
            "noisy_unclipped_quantile": float(stats_aggregated.get("noisy_unclipped_quantile", 0.0)),
            "importance_frozen": 1.0 if self.importance_frozen else 0.0,
            "importance_distance": float(self.last_importance_distance),
        }

    def evaluate(self, test_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        self.global_model.eval()
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.inference_mode():
            for data, target in test_loader:
                data = data.to(self.device, non_blocking=True)
                if self.channels_last and data.ndim == 4:
                    data = data.contiguous(memory_format=torch.channels_last)
                target = target.to(self.device, non_blocking=True)
                if self.use_amp and self.amp_dtype is not None and self.device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=self.amp_dtype):
                        logits = self.global_model(data)
                        loss = criterion(logits, target)
                else:
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
        importance_payloads = [
            (update.importance_dict, update.data_size)
            for update in client_updates
            if update.importance_dict is not None
        ]
        if not importance_payloads:
            return None

        method = str(getattr(self.cfg.dp, "importance_aggregation", "tempered_weighted"))
        beta = float(getattr(self.cfg.dp, "importance_weight_beta", 0.5))
        if method == "mean":
            weights = [1.0 / len(importance_payloads) for _ in importance_payloads]
        else:
            if method == "weighted":
                exponent = 1.0
            elif method == "tempered_weighted":
                exponent = beta
            else:
                raise ValueError(f"Unknown importance aggregation method: {method}")
            raw_weights = [max(float(data_size), 1.0) ** exponent for _, data_size in importance_payloads]
            total_weight = sum(raw_weights)
            weights = [weight / max(total_weight, 1e-12) for weight in raw_weights]

        aggregated: TensorDict = {}
        template_keys = self.global_importance_template.keys()
        for name in template_keys:
            acc = None
            for (importance_dict, _), weight in zip(importance_payloads, weights):
                if name not in importance_dict:
                    continue
                normalized = self._normalize_importance_tensor(importance_dict[name])
                weighted_term = normalized * float(weight)
                acc = weighted_term.clone() if acc is None else acc + weighted_term
            if acc is not None:
                aggregated[name] = acc.cpu()
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
            # [MOD][阶段1] Andrew et al. (2019): C_{t+1} = C_t * exp(-eta_c * (q_hat_t - gamma)).
            noisy_unclipped_quantile = float(stats.get("noisy_unclipped_quantile", 0.0))
            target_quantile = float(self.cfg.server.target_quantile)
            clip_lr = float(self.cfg.server.clip_lr)
            self.clip_norm *= math.exp(-clip_lr * (noisy_unclipped_quantile - target_quantile))
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
                "unclipped_count": 0.0,
                "fraction_unclipped": 0.0,
                "fraction_clipped": 0.0,
                "noisy_unclipped_quantile": 0.0,
            }

        norms = np.array([float(update.stat) for update in client_updates], dtype=np.float64)
        clipped_flags = np.array([1.0 if update.clipped else 0.0 for update in client_updates], dtype=np.float64)
        unclipped_flags = 1.0 - clipped_flags

        count = float(norms.size)
        unclipped_count = float(unclipped_flags.sum())

        clip_count_noise_multiplier = float(getattr(self.cfg.dp, "clip_count_noise_multiplier", 0.1))
        noisy_unclipped_quantile = float(
            np.clip(
                (unclipped_count + np.random.normal(loc=0.0, scale=clip_count_noise_multiplier)) / max(count, 1.0),
                0.0,
                1.0,
            )
        )

        return {
            "median": float(np.quantile(norms, 0.5)),
            "q25": float(np.quantile(norms, 0.25)),
            "q75": float(np.quantile(norms, 0.75)),
            "count": count,
            "unclipped_count": unclipped_count,
            "fraction_unclipped": float(unclipped_flags.mean()),
            "fraction_clipped": float(clipped_flags.mean()),
            "noisy_unclipped_quantile": noisy_unclipped_quantile,
        }

    def _update_importance_templates(self, aggregated_importance: TensorDict) -> None:
        prev_template = copy.deepcopy(self.global_importance_template)
        alpha = float(self.cfg.dp.template_ema)
        updated_importance_ema: TensorDict = {}
        updated_global_template: TensorDict = {}
        updated_clip_template: TensorDict = {}

        for name, current_template in self.global_importance_template.items():
            public_importance = aggregated_importance.get(name)
            if public_importance is None:
                updated_importance_ema[name] = self.importance_ema_template[name]
                updated_global_template[name] = current_template
                updated_clip_template[name] = self.clip_weight_template[name]
                continue

            sanitized_importance = self._normalize_importance_tensor(public_importance).cpu()
            prev_ema = self.importance_ema_template[name]
            next_ema = alpha * prev_ema + (1.0 - alpha) * sanitized_importance
            updated_importance_ema[name] = next_ema
            updated_global_template[name] = next_ema.cpu()
            updated_clip_template[name] = self._importance_to_template(next_ema)

        candidate_distance = self._template_distance(prev_template, updated_global_template)
        self.last_importance_distance = candidate_distance

        if not self.importance_frozen:
            self.global_importance_template = updated_global_template
            self.clip_weight_template = updated_clip_template
            self.importance_ema_template = updated_importance_ema
            self._maybe_freeze(updated_global_template, candidate_distance)

    def _maybe_freeze(self, candidate_template: TensorDict, distance: float) -> None:
        del candidate_template
        if not bool(getattr(self.cfg.dp, "enable_importance_freeze", False)):
            return

        warmup_rounds = int(getattr(self.cfg.dp, "importance_freeze_warmup_rounds", 10))
        threshold = float(getattr(self.cfg.dp, "importance_freeze_threshold", 1e-3))
        patience = int(getattr(self.cfg.dp, "importance_freeze_patience", 3))

        if self.current_round < warmup_rounds:
            self.freeze_counter = 0
            return

        if distance < threshold:
            self.freeze_counter += 1
        else:
            self.freeze_counter = 0

        if self.freeze_counter >= patience:
            # [MOD][阶段3] 一旦全局 importance 模板稳定，就冻结，后续不再每轮更新和索取 importance。
            self.importance_frozen = True

    def _normalize_importance_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        normalized = normalize_importance_dict({"tmp": tensor})["tmp"]
        return normalized

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

    def _template_distance(self, previous: TensorDict, current: TensorDict) -> float:
        numerator_terms = []
        denominator_terms = []
        for name in previous.keys():
            if name not in current:
                continue
            prev = previous[name].float()
            cur = current[name].float()
            numerator_terms.append((cur - prev).pow(2).sum())
            denominator_terms.append(prev.pow(2).sum())
        if not numerator_terms:
            return 0.0
        numerator = torch.sqrt(torch.stack(numerator_terms).sum())
        denominator = torch.sqrt(torch.stack(denominator_terms).sum()).clamp_min(1e-6)
        return float((numerator / denominator).item())

    @staticmethod
    def _l2_norm(tensors: TensorDict) -> float:
        if not tensors:
            return 0.0
        total = sum(tensor.float().pow(2).sum() for tensor in tensors.values())
        return float(torch.sqrt(total).item())
