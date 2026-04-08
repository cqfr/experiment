from __future__ import annotations

"""Differential privacy accounting and noise utilities."""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch


def _rdp_subsampled_gaussian(q: float, sigma: float, alpha: float) -> float:
    """Approximate one-step RDP for Poisson-subsampled Gaussian mechanism.

    This uses the common small-q upper bound:
        RDP(alpha) ~= alpha * q^2 / (2 * sigma^2)
    It is monotonic in sigma and sufficient for stable round-wise calibration.
    """

    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    if not (0 < q <= 1):
        raise ValueError("q must be in (0, 1]")
    if alpha <= 1:
        raise ValueError("alpha must be > 1")
    return (alpha * (q**2)) / (2.0 * (sigma**2))


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
    """Round-wise RDP accountant with sigma calibration for each round target."""

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

    def solve_sigma_for_round(
        self,
        q: float,
        steps: int = 1,
        epsilon_target: Optional[float] = None,
        sigma_lo: float = 1e-3,
        sigma_hi: float = 1e3,
        tol: float = 1e-6,
        max_iter: int = 80,
    ) -> float:
        """Binary-search sigma that satisfies one-round epsilon target."""

        if steps <= 0:
            raise ValueError("steps must be > 0")
        target = self.epsilon_round_target if epsilon_target is None else epsilon_target
        if target <= 0:
            raise ValueError("epsilon target must be > 0")

        def round_eps(sigma: float) -> float:
            one_round_rdp = {
                alpha: steps * _rdp_subsampled_gaussian(q=q, sigma=sigma, alpha=alpha)
                for alpha in self.orders
            }
            return _epsilon_from_rdp(one_round_rdp, self.delta)

        # Ensure upper bound is feasible.
        while round_eps(sigma_hi) > target and sigma_hi < 1e7:
            sigma_hi *= 2.0

        for _ in range(max_iter):
            mid = 0.5 * (sigma_lo + sigma_hi)
            eps_mid = round_eps(mid)
            if abs(eps_mid - target) <= tol:
                return mid
            if eps_mid > target:
                sigma_lo = mid
            else:
                sigma_hi = mid

        return sigma_hi

    def consume_round(self, sigma: float, q: float, steps: int = 1) -> float:
        """Accumulate one round and return new composed epsilon."""

        for alpha in self.orders:
            self._rdp_cumulative[alpha] += steps * _rdp_subsampled_gaussian(
                q=q,
                sigma=sigma,
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


def allocate_relative_scales(
    importance: torch.Tensor,
    mask: torch.Tensor,
    min_scale: float = 0.3,
    max_scale: float = 3.0,
) -> torch.Tensor:
    """Create relative std scales: important entries get less noise.

    This returns per-parameter multiplicative factors before normalization.
    """

    if min_scale <= 0 or max_scale <= 0:
        raise ValueError("scales must be positive")
    if max_scale < min_scale:
        raise ValueError("max_scale must be >= min_scale")

    scales = torch.zeros_like(importance)
    selected = mask > 0
    if selected.sum().item() == 0:
        return scales

    imp = importance[selected]
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


def normalize_relative_scales(
    scales: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Normalize scales so harmonic privacy-equivalent sigma stays unchanged.

    We enforce:
        mean_selected(1 / sigma_i^2) = 1 / sigma_base^2
    with sigma_i = sigma_base * scale_i * gamma.
    """

    selected = mask > 0
    if selected.sum().item() == 0:
        return scales

    selected_scales = scales[selected].clamp(min=1e-8)
    gamma = torch.sqrt(torch.mean(1.0 / (selected_scales**2)))
    normalized = scales.clone()
    normalized[selected] = selected_scales * gamma
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
    sigma = accountant.solve_sigma_for_round(q=0.1, steps=1)
    eps = accountant.consume_round(sigma=sigma, q=0.1, steps=1)
    print(f"sigma={sigma:.6f}, eps={eps:.6f}")
