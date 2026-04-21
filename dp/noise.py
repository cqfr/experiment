from __future__ import annotations

"""Differential privacy accounting and noise utilities."""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch


def _rdp_subsampled_gaussian(q: float, noise_multiplier: float, alpha: float) -> float:
    """Approximate one-step RDP for the subsampled Gaussian mechanism."""

    if noise_multiplier <= 0:
        raise ValueError("noise_multiplier must be > 0")
    if not (0 < q <= 1):
        raise ValueError("q must be in (0, 1]")
    if alpha <= 1:
        raise ValueError("alpha must be > 1")
    return (alpha * (q**2)) / (2.0 * (noise_multiplier**2))


def _epsilon_from_rdp(rdp_values: Dict[float, float], delta: float) -> float:
    """Convert accumulated RDP to epsilon under fixed delta."""

    if not (0 < delta < 1):
        raise ValueError("delta must be in (0, 1)")

    candidates = []
    for alpha, rdp in rdp_values.items():
        if alpha <= 1:
            continue
        candidates.append(rdp + math.log(1.0 / delta) / (alpha - 1.0))
    return min(candidates) if candidates else float("inf")


@dataclass
class RDPAccountant:
    epsilon_total: float
    delta: float
    num_rounds: int
    orders: Tuple[float, ...]

    def __post_init__(self) -> None:
        if self.epsilon_total <= 0:
            raise ValueError("epsilon_total must be > 0")
        if self.num_rounds <= 0:
            raise ValueError("num_rounds must be > 0")
        if not self.orders:
            raise ValueError("orders must not be empty")
        self._rdp_cumulative = {order: 0.0 for order in self.orders}
        self.round_count = 0

    def current_epsilon(self) -> float:
        return _epsilon_from_rdp(self._rdp_cumulative, self.delta)

    def remaining_budget(self) -> float:
        return max(0.0, self.epsilon_total - self.current_epsilon())

    def is_exhausted(self) -> bool:
        return self.current_epsilon() >= self.epsilon_total

    def solve_noise_multiplier_for_round(
        self,
        q: float,
        steps: int = 1,
        z_lo: float = 1e-3,
        z_hi: float = 1e3,
        tol: float = 1e-6,
        max_iter: int = 200,
    ) -> float:
        if steps <= 0:
            raise ValueError("steps must be > 0")

        def total_eps(z: float) -> float:
            total_rdp = {
                alpha: self.num_rounds * steps * _rdp_subsampled_gaussian(q=q, noise_multiplier=z, alpha=alpha)
                for alpha in self.orders
            }
            return _epsilon_from_rdp(total_rdp, self.delta)

        while total_eps(z_hi) > self.epsilon_total and z_hi < 1e7:
            z_hi *= 2.0

        for _ in range(max_iter):
            mid = 0.5 * (z_lo + z_hi)
            eps_mid = total_eps(mid)
            if abs(eps_mid - self.epsilon_total) <= tol:
                return mid
            if eps_mid > self.epsilon_total:
                z_lo = mid
            else:
                z_hi = mid
        return z_hi

    def solve_sigma_for_round(self, *args, **kwargs) -> float:
        return self.solve_noise_multiplier_for_round(*args, **kwargs)

    def consume_round(
        self,
        noise_multiplier: Optional[float] = None,
        q: float = 1.0,
        steps: int = 1,
        sigma: Optional[float] = None,
    ) -> float:
        z = noise_multiplier if noise_multiplier is not None else sigma
        if z is None:
            raise ValueError("noise_multiplier (or legacy sigma) must be provided")

        for alpha in self.orders:
            self._rdp_cumulative[alpha] += steps * _rdp_subsampled_gaussian(q=q, noise_multiplier=z, alpha=alpha)
        self.round_count += 1
        return self.current_epsilon()


def add_noise_to_tensor(
    tensor: torch.Tensor,
    sigma: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    if sigma <= 0:
        return tensor
    if generator is not None:
        noise = torch.randn(tensor.shape, generator=generator, dtype=tensor.dtype)
        noise = noise.to(tensor.device) * sigma
    else:
        noise = torch.randn_like(tensor) * sigma
    return tensor + noise


def compute_weighted_sensitivity(client_weights: Sequence[float], clip_norm: float) -> float:
    """Sensitivity upper bound for weighted aggregation under add/remove adjacency."""

    if clip_norm < 0:
        raise ValueError("clip_norm must be >= 0")
    if not client_weights:
        return 0.0
    return max(abs(float(weight)) for weight in client_weights) * clip_norm


def allocate_client_noise_stds(
    client_weights: Sequence[float],
    sigma_agg: float,
    strategy: str = "uniform",
    client_importance: Optional[Sequence[float]] = None,
    max_sigma_scale: float = 10.0,
) -> list[float]:
    """Back-solve per-client stds from target aggregate std."""

    if sigma_agg <= 0:
        return [0.0 for _ in client_weights]
    if not client_weights:
        return []

    normalized_weights = [max(0.0, float(weight)) for weight in client_weights]
    total_weight = sum(normalized_weights)
    if total_weight <= 0:
        raise ValueError("client_weights must contain positive values")
    normalized_weights = [weight / total_weight for weight in normalized_weights]

    eps = 1e-12
    if strategy == "uniform":
        denom = math.sqrt(sum(weight * weight for weight in normalized_weights))
        sigma_base = sigma_agg / max(denom, eps)
        sigmas = [sigma_base for _ in normalized_weights]
    elif strategy == "heterogeneous":
        if client_importance is None:
            raise ValueError("client_importance is required for heterogeneous strategy")
        if len(client_importance) != len(normalized_weights):
            raise ValueError("client_importance length mismatch")

        raw_scales = [1.0 / max(float(importance), eps) for importance in client_importance]
        total_scale = sum(raw_scales)
        if total_scale <= 0:
            raise ValueError("invalid client_importance values")
        rel = [value / total_scale for value in raw_scales]

        sigmas = [
            sigma_agg * math.sqrt(rel_i) / max(abs(weight_i), eps)
            for weight_i, rel_i in zip(normalized_weights, rel)
        ]

        if max_sigma_scale > 0:
            max_sigma = max_sigma_scale * sigma_agg
            sigmas = [min(sigma, max_sigma) for sigma in sigmas]
    else:
        raise ValueError(f"Unknown client noise allocation strategy: {strategy}")

    target_var = sigma_agg * sigma_agg
    actual_var = sum((weight_i * weight_i) * (sigma_i * sigma_i) for weight_i, sigma_i in zip(normalized_weights, sigmas))
    if actual_var > 0:
        scale = math.sqrt(target_var / actual_var)
        sigmas = [sigma * scale for sigma in sigmas]
    return sigmas


def allocate_relative_scales(
    importance: torch.Tensor,
    mask: torch.Tensor,
    min_scale: float = 0.3,
    max_scale: float = 3.0,
    mode: str = "inverse_power",
    alpha: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Create per-coordinate relative std scales before normalization."""

    if min_scale <= 0 or max_scale <= 0:
        raise ValueError("scales must be positive")
    if max_scale < min_scale:
        raise ValueError("max_scale must be >= min_scale")
    if eps <= 0:
        raise ValueError("eps must be > 0")

    scales = torch.zeros_like(importance)
    selected = mask > 0
    if not bool(selected.any()):
        return scales

    imp = torch.nan_to_num(importance[selected].abs(), nan=0.0, posinf=0.0, neginf=0.0)
    if mode == "linear":
        imp_min = imp.min()
        imp_max = imp.max()
        if imp_max > imp_min:
            imp_norm = (imp - imp_min) / (imp_max - imp_min)
        else:
            imp_norm = torch.ones_like(imp)
        rel = max_scale - imp_norm * (max_scale - min_scale)
        scales[selected] = rel
        return scales

    if mode == "inverse_power":
        if alpha <= 0:
            raise ValueError("alpha must be > 0 for inverse_power")
        inv = imp.clamp_min(eps).pow(-alpha)
        total = inv.sum()
        if total <= 0:
            rel = torch.ones_like(inv)
        else:
            rel = torch.sqrt(inv / total * float(inv.numel()))
        rel = rel.clamp(min=min_scale, max=max_scale)
        scales[selected] = rel
        return scales

    raise ValueError(f"Unknown relative scale mode: {mode}")


def normalize_relative_scales(scales: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Normalize scales so mean_selected(scale^2) == 1."""

    selected = mask > 0
    if not bool(selected.any()):
        return scales

    selected_scales = scales[selected].clamp_min(1e-8)
    gamma = torch.sqrt(torch.mean(selected_scales.pow(2)))
    normalized = scales.clone()
    normalized[selected] = selected_scales / gamma
    return normalized


def add_heterogeneous_noise(
    tensor: torch.Tensor,
    sigma_base: float,
    relative_scales: torch.Tensor,
    mask: torch.Tensor,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, float]:
    """Add diagonal Gaussian noise with normalized heterogeneous scales."""

    if sigma_base <= 0:
        return tensor, 0.0

    normalized_scales = normalize_relative_scales(relative_scales, mask)
    sigma_i = torch.zeros_like(tensor)
    selected = mask > 0
    sigma_i[selected] = sigma_base * normalized_scales[selected]

    if generator is not None:
        noise = torch.randn(tensor.shape, generator=generator, dtype=tensor.dtype).to(tensor.device)
    else:
        noise = torch.randn_like(tensor)
    noisy = tensor + noise * sigma_i

    avg_sigma = float(sigma_i[selected].mean().item()) if bool(selected.any()) else 0.0
    return noisy, avg_sigma


PrivacyAccountant = RDPAccountant
