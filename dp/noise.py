from __future__ import annotations

"""Differential privacy accounting and noise utilities."""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch


def _rdp_subsampled_gaussian(q: float, noise_multiplier: float, alpha: float) -> float:
    """Approximate one-step RDP for Poisson-subsampled Gaussian mechanism.

    This uses the common small-q upper bound:
        RDP(alpha) ~= alpha * q^2 / (2 * noise_multiplier^2)
    It is monotonic in noise multiplier z and sufficient for stable round-wise
    calibration.
    """

    if noise_multiplier <= 0:
        raise ValueError("noise_multiplier must be > 0")
    if not (0 < q <= 1):
        raise ValueError("q must be in (0, 1]")
    if alpha <= 1:
        raise ValueError("alpha must be > 1")
    return (alpha * (q**2)) / (2.0 * (noise_multiplier**2))


def _epsilon_from_rdp(rdp_values: Dict[float, float], delta: float) -> float:
    """Convert accumulated RDP to (epsilon, delta)-DP by best order search."""

    if not (0 < delta < 1):
        raise ValueError("delta must be in (0, 1)")

    eps_candidates = []
    for alpha, rdp in rdp_values.items():
        if alpha <= 1:
            continue
        eps_candidates.append(rdp + math.log(1.0 / delta) / (alpha - 1.0))
    return min(eps_candidates) if eps_candidates else float("inf")


@dataclass
class RDPAccountant:
    """Round-wise RDP accountant with noise-multiplier calibration."""

    epsilon_total: float
    delta: float
    num_rounds: int
    orders: Tuple[float, ...]

    def __post_init__(self) -> None:
        if self.num_rounds <= 0:
            raise ValueError("num_rounds must be > 0")
        if self.epsilon_total <= 0:
            raise ValueError("epsilon_total must be > 0")
        if not self.orders:
            raise ValueError("orders must not be empty")

        self.epsilon_round_target = self.epsilon_total / float(self.num_rounds)
        self._rdp_cumulative = {order: 0.0 for order in self.orders}
        self.round_count = 0

    def current_epsilon(self) -> float:
        """Current composed epsilon under configured delta."""

        return _epsilon_from_rdp(self._rdp_cumulative, self.delta)

    def remaining_budget(self) -> float:
        """Remaining global epsilon budget."""

        return max(0.0, self.epsilon_total - self.current_epsilon())

    def is_exhausted(self) -> bool:
        """Whether global privacy budget is exhausted."""

        return self.current_epsilon() >= self.epsilon_total

    def solve_noise_multiplier_for_round(
        self,
        q: float,
        steps: int = 1,
        epsilon_target: Optional[float] = None,
        z_lo: float = 1e-3,
        z_hi: float = 1e3,
        tol: float = 1e-6,
        max_iter: int = 200,
    ) -> float:
        """Binary-search noise multiplier z for one-round epsilon target."""

        if steps <= 0:
            raise ValueError("steps must be > 0")
        target = self.epsilon_round_target if epsilon_target is None else epsilon_target
        if target <= 0:
            raise ValueError("epsilon target must be > 0")

        def round_eps(z: float) -> float:
            one_round_rdp = {
                alpha: steps
                * _rdp_subsampled_gaussian(
                    q=q,
                    noise_multiplier=z,
                    alpha=alpha,
                )
                for alpha in self.orders
            }
            return _epsilon_from_rdp(one_round_rdp, self.delta)

        # Ensure upper bound is feasible.
        while round_eps(z_hi) > target and z_hi < 1e7:
            z_hi *= 2.0

        for _ in range(max_iter):
            mid = 0.5 * (z_lo + z_hi)
            eps_mid = round_eps(mid)
            if abs(eps_mid - target) <= tol:
                return mid
            if eps_mid > target:
                z_lo = mid
            else:
                z_hi = mid

        return z_hi

    def solve_sigma_for_round(self, *args, **kwargs) -> float:
        """Backward-compatible alias; returns noise multiplier z."""
        return self.solve_noise_multiplier_for_round(*args, **kwargs)

    def consume_round(
        self,
        noise_multiplier: Optional[float] = None,
        q: float = 1.0,
        steps: int = 1,
        sigma: Optional[float] = None,
    ) -> float:
        """Accumulate one round and return new composed epsilon."""

        z = noise_multiplier
        if z is None:
            z = sigma
        if z is None:
            raise ValueError("noise_multiplier (or legacy sigma) must be provided")

        for alpha in self.orders:
            self._rdp_cumulative[alpha] += steps * _rdp_subsampled_gaussian(
                q=q,
                noise_multiplier=z,
                alpha=alpha,
            )
        self.round_count += 1
        return self.current_epsilon()


def add_noise_to_tensor(
    tensor: torch.Tensor,
    sigma: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Add isotropic Gaussian noise with std sigma to a tensor."""

    if sigma <= 0:
        return tensor
    if generator is not None:
        noise = torch.randn(tensor.shape, generator=generator, dtype=tensor.dtype)
        noise = noise.to(tensor.device) * sigma
    else:
        noise = torch.randn_like(tensor) * sigma
    return tensor + noise


def compute_weighted_sensitivity(
    client_weights: Sequence[float],
    clip_norm: float,
) -> float:
    """Sensitivity upper bound for weighted aggregation under add/remove adjacency."""

    if clip_norm < 0:
        raise ValueError("clip_norm must be >= 0")
    if not client_weights:
        return 0.0

    p_max = max(abs(float(w)) for w in client_weights)
    return p_max * clip_norm


def allocate_client_noise_stds(
    client_weights: Sequence[float],
    sigma_agg: float,
    strategy: str = "uniform",
    client_importance: Optional[Sequence[float]] = None,
    max_sigma_scale: float = 10.0,
) -> list[float]:
    """Back-solve per-client base stds from target aggregate std.

    Constraint:
        sum_i p_i^2 * sigma_i^2 == sigma_agg^2
    """

    if sigma_agg <= 0:
        return [0.0 for _ in client_weights]
    if not client_weights:
        return []

    p = [max(0.0, float(w)) for w in client_weights]
    p_sum = sum(p)
    if p_sum <= 0:
        raise ValueError("client_weights must contain positive values")
    p = [w / p_sum for w in p]

    eps = 1e-12

    if strategy == "uniform":
        denom = math.sqrt(sum(w * w for w in p))
        sigma_base = sigma_agg / max(denom, eps)
        sigmas = [sigma_base for _ in p]
    elif strategy == "heterogeneous":
        if client_importance is None:
            raise ValueError("client_importance is required for heterogeneous strategy")
        if len(client_importance) != len(p):
            raise ValueError("client_importance length mismatch")

        raw_r = [1.0 / max(float(imp), eps) for imp in client_importance]
        total_r = sum(raw_r)
        if total_r <= 0:
            raise ValueError("invalid client_importance values")
        r = [v / total_r for v in raw_r]

        sigmas = [
            sigma_agg * math.sqrt(r_i) / max(abs(p_i), eps)
            for p_i, r_i in zip(p, r)
        ]

        if max_sigma_scale > 0:
            max_sigma = max_sigma_scale * sigma_agg
            sigmas = [min(s, max_sigma) for s in sigmas]
    else:
        raise ValueError(f"Unknown client noise allocation strategy: {strategy}")

    # Renormalize to satisfy exact aggregate variance after optional clipping.
    target_var = sigma_agg * sigma_agg
    actual_var = sum((p_i * p_i) * (s_i * s_i) for p_i, s_i in zip(p, sigmas))
    if actual_var > 0:
        scale = math.sqrt(target_var / actual_var)
        sigmas = [s * scale for s in sigmas]

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
    """Create relative std scales: important entries get less noise.

    This returns per-parameter multiplicative factors before normalization.
    - mode="inverse_power": sigma^2 proportional to 1/(importance+eps)^alpha.
    - mode="linear": legacy linear interpolation in [min_scale, max_scale].
    """

    if min_scale <= 0 or max_scale <= 0:
        raise ValueError("scales must be positive")
    if max_scale < min_scale:
        raise ValueError("max_scale must be >= min_scale")
    if eps <= 0:
        raise ValueError("eps must be > 0")

    scales = torch.zeros_like(importance)
    selected = mask > 0
    if selected.sum().item() == 0:
        return scales

    imp = torch.nan_to_num(
        importance[selected].abs(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    if mode == "linear":
        imp_min = imp.min()
        imp_max = imp.max()
        if imp_max > imp_min:
            imp_norm = (imp - imp_min) / (imp_max - imp_min)
        else:
            imp_norm = torch.ones_like(imp)

        # High importance -> smaller noise scale.
        rel = max_scale - imp_norm * (max_scale - min_scale)
        scales[selected] = rel
        return scales

    if mode == "inverse_power":
        if alpha <= 0:
            raise ValueError("alpha must be > 0 for inverse_power")

        # Variance weights follow inverse importance; std uses sqrt(weights).
        inv = imp.clamp_min(eps).pow(-alpha)
        total = inv.sum()
        if total <= 0:
            rel = torch.ones_like(inv)
        else:
            weights = inv / total
            rel = torch.sqrt(weights * float(weights.numel()))

        rel = rel.clamp(min=min_scale, max=max_scale)
        scales[selected] = rel
        return scales

    raise ValueError(f"Unknown relative scale mode: {mode}")


def normalize_relative_scales(
    scales: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Normalize scales so per-client noise variance budget stays unchanged.

    We enforce:
        mean_selected(sigma_i^2) = sigma_base^2
    with sigma_i = sigma_base * scale_i * gamma.
    """

    selected = mask > 0
    if selected.sum().item() == 0:
        return scales

    selected_scales = scales[selected].clamp(min=1e-8)
    gamma = torch.sqrt(torch.mean(selected_scales**2))
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

    norm_scales = normalize_relative_scales(relative_scales, mask)

    sigma_i = torch.zeros_like(tensor)
    selected = mask > 0
    sigma_i[selected] = sigma_base * norm_scales[selected]

    if generator is not None:
        z = torch.randn(tensor.shape, generator=generator, dtype=tensor.dtype)
        z = z.to(tensor.device)
    else:
        z = torch.randn_like(tensor)
    noisy = tensor + z * sigma_i

    avg_sigma = sigma_i[selected].mean().item() if selected.sum().item() > 0 else 0.0
    return noisy, avg_sigma


# Backward-compatible alias used in older scripts.
PrivacyAccountant = RDPAccountant


if __name__ == "__main__":
    accountant = RDPAccountant(
        epsilon_total=8.0,
        delta=1e-5,
        num_rounds=100,
        orders=(2, 4, 8, 16, 32, 64, 128),
    )
    z = accountant.solve_noise_multiplier_for_round(q=0.1, steps=1)
    eps = accountant.consume_round(noise_multiplier=z, q=0.1, steps=1)
    print(f"noise_multiplier={z:.6f}, eps={eps:.6f}")
