from __future__ import annotations

"""Federated client implementation."""

import copy
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import ClientConfig, DPConfig, ImportanceStrategy, TrainingStrategy
from compression.topk import (
    ResidualAccumulator,
    compute_fisher_information,
    compute_importance_grad_normalized,
    compress_gradient,
)
from dp.noise import (
    add_heterogeneous_noise,
    add_noise_to_tensor,
    allocate_relative_scales,
)


@dataclass
class ClientUpdate:
    """Payload uploaded by a client."""

    delta_w: Dict[str, torch.Tensor]
    data_size: int
    stat: float
    clipped: bool
    noise_sigma: float = 0.0
    upload_ratio: float = 1.0
    importance_vector: Optional[torch.Tensor] = None
    mask_vector: Optional[torch.Tensor] = None


class FLClient:
    """Client worker for local training, compression, clipping, and DP noise."""

    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        dataloader: DataLoader,
        config: ClientConfig,
        dp_config: DPConfig,
        device: torch.device,
    ) -> None:
        self.client_id = client_id
        self.model = copy.deepcopy(model).to(device)
        self.dataloader = dataloader
        self.config = config
        self.dp_config = dp_config
        self.device = device

        self.data_size = len(dataloader.dataset)
        self.residual_accumulator = ResidualAccumulator()

        self.old_local_weights: Optional[Dict[str, torch.Tensor]] = None
        self.global_weights: Optional[Dict[str, torch.Tensor]] = None
        self.fisher_cache: Optional[Dict[str, torch.Tensor]] = None

        self.rng = torch.Generator(device="cpu")

    def receive_global_model(self, global_weights: Dict[str, torch.Tensor]) -> None:
        self.model.load_state_dict(global_weights)
        self.global_weights = {k: v.clone() for k, v in global_weights.items()}

    def local_train(self) -> Dict[str, torch.Tensor]:
        """Run local optimization and return model delta: Delta_w = w_local - w_global."""

        self.model.train()
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.config.lr,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()

        init_weights = {k: v.clone() for k, v in self.model.state_dict().items()}

        if self.config.training_strategy == TrainingStrategy.DP_FEDSAM:
            self._train_dp_fedsam(optimizer, criterion)
        else:
            self._train_standard(optimizer, criterion)

        delta_w: Dict[str, torch.Tensor] = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                delta_w[name] = param.data - init_weights[name]

        self.old_local_weights = {k: v.clone() for k, v in self.model.state_dict().items()}
        return delta_w

    def _train_standard(self, optimizer: optim.Optimizer, criterion: nn.Module) -> None:
        for _ in range(self.config.local_epochs):
            for data, target in self.dataloader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                logits = self.model(data)
                loss = criterion(logits, target)

                if self.config.training_strategy in {
                    TrainingStrategy.CONTRASTIVE,
                    TrainingStrategy.FEDPROX,
                }:
                    loss = loss + self._regularization_loss()

                loss.backward()
                optimizer.step()

    def _train_dp_fedsam(self, optimizer: optim.Optimizer, criterion: nn.Module) -> None:
        """Local DP-FedSAM baseline (SAM-style two-step update)."""

        for _ in range(self.config.local_epochs):
            for data, target in self.dataloader:
                data, target = data.to(self.device), target.to(self.device)

                # First step: gradient at current weights.
                optimizer.zero_grad()
                loss = criterion(self.model(data), target)
                loss.backward()

                grad_norm = self._grad_l2_norm()
                scale = self.config.sam_rho / (grad_norm + self.config.sam_eps)

                perturbations = []
                with torch.no_grad():
                    for param in self.model.parameters():
                        if param.grad is None:
                            perturbations.append(None)
                            continue
                        e_w = param.grad * scale
                        param.add_(e_w)
                        perturbations.append(e_w)

                # Second step: gradient at perturbed weights.
                optimizer.zero_grad()
                loss_perturbed = criterion(self.model(data), target)
                loss_perturbed.backward()

                with torch.no_grad():
                    for param, e_w in zip(self.model.parameters(), perturbations):
                        if e_w is not None:
                            param.sub_(e_w)

                optimizer.step()

    def _grad_l2_norm(self) -> torch.Tensor:
        norms = []
        for param in self.model.parameters():
            if param.grad is not None:
                norms.append(torch.norm(param.grad, p=2))
        if not norms:
            return torch.tensor(0.0, device=self.device)
        return torch.norm(torch.stack(norms), p=2)

    def _regularization_loss(self) -> torch.Tensor:
        if self.global_weights is None:
            return torch.tensor(0.0, device=self.device)

        reg = torch.tensor(0.0, device=self.device)

        if self.config.training_strategy == TrainingStrategy.FEDPROX:
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in self.global_weights:
                    diff = param - self.global_weights[name]
                    reg = reg + self.config.mu * diff.pow(2).sum()
            return reg

        # Bounded contrastive term:
        # alpha*||w-wg||^2 + beta*max(0, margin-||w-w_old||)^2
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            if name in self.global_weights:
                diff_global = param - self.global_weights[name]
                reg = reg + self.config.alpha * diff_global.pow(2).sum()

            if self.old_local_weights is not None and name in self.old_local_weights:
                diff_old = param - self.old_local_weights[name]
                distance = torch.norm(diff_old, p=2)
                hinge = torch.relu(self.config.contrastive_margin - distance)
                reg = reg + self.config.beta * hinge.pow(2)

        return reg

    def accumulate_residual(self, delta_w: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.config.use_residual:
            return self.residual_accumulator.accumulate(delta_w)
        return delta_w

    def compute_importance(self, delta_w: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        strategy = self.config.importance_strategy

        if strategy == ImportanceStrategy.FISHER_GRAD:
            if self.fisher_cache is None:
                self.fisher_cache = compute_fisher_information(
                    self.model,
                    self.dataloader,
                    self.device,
                )
            return {
                name: self.fisher_cache.get(name, torch.ones_like(d)).to(d.device) * d.abs()
                for name, d in delta_w.items()
            }

        if strategy == ImportanceStrategy.GRAD_SQUARED:
            return {name: d.abs() * d.abs() for name, d in delta_w.items()}

        return compute_importance_grad_normalized(delta_w)

    def topk_compress(
        self,
        delta_w: Dict[str, torch.Tensor],
        importance: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], float]:
        sparse_delta, masks = compress_gradient(
            gradient=delta_w,
            importance_strategy=self.config.importance_strategy.value,
            topk_strategy=self.config.topk_strategy.value,
            k_ratio=self.config.topk_ratio,
            weight_method=self.config.layer_weight_method.value,
            importance=importance,
        )

        if self.config.use_residual:
            self.residual_accumulator.update(delta_w, masks)

        total_params = sum(v.numel() for v in masks.values())
        kept_params = sum(v.sum().item() for v in masks.values())
        upload_ratio = float(kept_params / total_params) if total_params > 0 else 1.0
        return sparse_delta, masks, upload_ratio

    def clip_delta(
        self,
        delta_w: Dict[str, torch.Tensor],
        clip_norm: float,
    ) -> Tuple[Dict[str, torch.Tensor], float, bool]:
        total_norm_sq = sum(d.pow(2).sum() for d in delta_w.values())
        total_norm = total_norm_sq.sqrt().item()

        clipped = total_norm > clip_norm
        if clipped:
            scale = clip_norm / max(total_norm, 1e-12)
            clipped_delta = {name: d * scale for name, d in delta_w.items()}
        else:
            clipped_delta = delta_w

        return clipped_delta, total_norm, clipped

    def add_dp_noise(
        self,
        delta_w: Dict[str, torch.Tensor],
        masks: Dict[str, torch.Tensor],
        importance: Dict[str, torch.Tensor],
        sigma_base: float,
    ) -> Tuple[Dict[str, torch.Tensor], float]:
        if not self.dp_config.enabled or sigma_base <= 0:
            return delta_w, 0.0

        if not self.dp_config.use_heterogeneous_noise:
            noisy = {
                name: add_noise_to_tensor(d, sigma_base, generator=self.rng)
                for name, d in delta_w.items()
            }
            return noisy, sigma_base

        return self._add_heterogeneous_noise(delta_w, masks, importance, sigma_base)

    def _add_heterogeneous_noise(
        self,
        delta_w: Dict[str, torch.Tensor],
        masks: Dict[str, torch.Tensor],
        importance: Dict[str, torch.Tensor],
        sigma_base: float,
    ) -> Tuple[Dict[str, torch.Tensor], float]:
        flat_delta = []
        flat_mask = []
        flat_importance = []
        slices = []

        offset = 0
        for name, d in delta_w.items():
            d_flat = d.flatten()
            m_flat = masks.get(name, torch.ones_like(d)).flatten()
            i_flat = importance.get(name, d.abs()).flatten()

            flat_delta.append(d_flat)
            flat_mask.append(m_flat)
            flat_importance.append(i_flat)
            slices.append((name, offset, offset + d_flat.numel(), d.shape))
            offset += d_flat.numel()

        if not flat_delta:
            return delta_w, 0.0

        global_delta = torch.cat(flat_delta)
        global_mask = torch.cat(flat_mask)
        global_importance = torch.cat(flat_importance)

        rel_scales = allocate_relative_scales(
            importance=global_importance,
            mask=global_mask,
            min_scale=self.dp_config.min_relative_noise,
            max_scale=self.dp_config.max_relative_noise,
        )

        noisy_global, avg_sigma = add_heterogeneous_noise(
            tensor=global_delta,
            sigma_base=sigma_base,
            relative_scales=rel_scales,
            mask=global_mask,
            generator=self.rng,
        )

        noisy_delta: Dict[str, torch.Tensor] = {}
        for name, start, end, shape in slices:
            noisy_delta[name] = noisy_global[start:end].reshape(shape)

        return noisy_delta, avg_sigma

    def train_and_upload(
        self,
        global_weights: Dict[str, torch.Tensor],
        clip_norm: float,
        sigma_base: float,
        return_importance_snapshot: bool = False,
        importance_max_elements: int = 4096,
    ) -> ClientUpdate:
        self.receive_global_model(global_weights)

        delta_w = self.local_train()
        delta_w = self.accumulate_residual(delta_w)

        importance = self.compute_importance(delta_w)
        sparse_delta, masks, upload_ratio = self.topk_compress(delta_w, importance)

        clipped_delta, l2_norm, clipped = self.clip_delta(sparse_delta, clip_norm)

        noisy_delta, noise_sigma = self.add_dp_noise(
            clipped_delta,
            masks,
            importance,
            sigma_base,
        )

        stat = 1.0 if clipped else 0.0
        if self.config.stat_type.value == "l2_norm":
            stat = l2_norm

        importance_vector = None
        mask_vector = None
        if return_importance_snapshot:
            importance_vector = self._build_vector_snapshot(
                tensors=importance,
                max_elements=importance_max_elements,
            )
            mask_vector = self._build_vector_snapshot(
                tensors=masks,
                max_elements=importance_max_elements,
            )

        return ClientUpdate(
            delta_w=noisy_delta,
            data_size=self.data_size,
            stat=stat,
            clipped=clipped,
            noise_sigma=noise_sigma,
            upload_ratio=upload_ratio,
            importance_vector=importance_vector,
            mask_vector=mask_vector,
        )

    @staticmethod
    def _build_vector_snapshot(
        tensors: Dict[str, torch.Tensor],
        max_elements: int = 4096,
    ) -> torch.Tensor:
        """Flatten dict tensors and keep a bounded-size snapshot for visualization."""
        if not tensors:
            return torch.zeros(1)
        flat = torch.cat([t.detach().float().flatten().cpu() for t in tensors.values()])
        if flat.numel() <= max_elements:
            return flat
        # Uniform subsampling keeps spatial coverage of the full vector.
        idx = torch.linspace(0, flat.numel() - 1, steps=max_elements).long()
        return flat[idx]


if __name__ == "__main__":
    print("client module ready")
