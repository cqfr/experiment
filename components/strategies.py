from __future__ import annotations

"""Training, compression, clipping, and privacy strategies."""

import copy
import math
from dataclasses import dataclass
from typing import Dict, Optional, Protocol, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from compression.topk import (
    ResidualAccumulator,
    compute_fisher_information,
    compute_importance_grad_normalized,
    compress_gradient,
)
from dp.noise import (
    RDPAccountant,
    add_heterogeneous_noise,
    add_noise_to_tensor,
    compute_weighted_sensitivity,
)

TensorDict = Dict[str, torch.Tensor]


@dataclass
class CompressionResult:
    delta_w: TensorDict
    upload_ratio: float
    importance_dict: Optional[TensorDict] = None
    mask_dict: Optional[TensorDict] = None
    importance_vector: Optional[torch.Tensor] = None
    mask_vector: Optional[torch.Tensor] = None


@dataclass
class ClipResult:
    delta_w: TensorDict
    norm_value: float
    clipped: bool


@dataclass
class PrivacyRoundState:
    noise_multiplier: float
    sigma_agg: float
    sensitivity_l2: float
    clip_snr_proxy: float
    q_accounting: float


class LocalTrainer(Protocol):
    def train(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        global_weights: Optional[TensorDict] = None,
        old_local_weights: Optional[TensorDict] = None,
        generator: Optional[torch.Generator] = None,
    ) -> TensorDict:
        ...

    def should_store_old_weights(self) -> bool:
        ...

    def needs_global_weights(self) -> bool:
        ...


class _BaseTrainer:
    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def _optimizer(self, model: nn.Module) -> optim.Optimizer:
        return optim.SGD(
            model.parameters(),
            lr=float(self.cfg.lr),
            momentum=float(self.cfg.momentum),
            weight_decay=float(self.cfg.weight_decay),
        )

    def _criterion(self) -> nn.Module:
        return nn.CrossEntropyLoss()

    def _init_weights(self, model: nn.Module) -> TensorDict:
        return {k: v.detach().clone() for k, v in model.state_dict().items()}

    def _delta_from_init(self, model: nn.Module, init_weights: TensorDict) -> TensorDict:
        delta_w: TensorDict = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                delta_w[name] = param.data - init_weights[name]
        return delta_w

    def should_store_old_weights(self) -> bool:
        return False

    def needs_global_weights(self) -> bool:
        return False


class StandardTrainer(_BaseTrainer):
    def train(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        global_weights: Optional[TensorDict] = None,
        old_local_weights: Optional[TensorDict] = None,
        generator: Optional[torch.Generator] = None,
    ) -> TensorDict:
        del global_weights, old_local_weights, generator
        model.train()
        optimizer = self._optimizer(model)
        criterion = self._criterion()
        init_weights = self._init_weights(model)

        for _ in range(int(self.cfg.local_epochs)):
            for data, target in dataloader:
                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(data), target)
                loss.backward()
                optimizer.step()

        return self._delta_from_init(model, init_weights)


class ContrastiveTrainer(_BaseTrainer):
    def should_store_old_weights(self) -> bool:
        return True

    def needs_global_weights(self) -> bool:
        return True

    def train(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        global_weights: Optional[TensorDict] = None,
        old_local_weights: Optional[TensorDict] = None,
        generator: Optional[torch.Generator] = None,
    ) -> TensorDict:
        del generator
        model.train()
        optimizer = self._optimizer(model)
        criterion = self._criterion()
        init_weights = self._init_weights(model)

        for _ in range(int(self.cfg.local_epochs)):
            for data, target in dataloader:
                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(data), target)
                loss = loss + self._regularization_loss(model, global_weights, old_local_weights, device)
                loss.backward()
                optimizer.step()

        return self._delta_from_init(model, init_weights)

    def _regularization_loss(
        self,
        model: nn.Module,
        global_weights: Optional[TensorDict],
        old_local_weights: Optional[TensorDict],
        device: torch.device,
    ) -> torch.Tensor:
        if global_weights is None:
            return torch.tensor(0.0, device=device)

        reg = torch.tensor(0.0, device=device)
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name in global_weights:
                diff_global = param - global_weights[name]
                reg = reg + float(self.cfg.alpha) * diff_global.pow(2).sum()
            if old_local_weights is not None and name in old_local_weights:
                diff_old = param - old_local_weights[name]
                distance = torch.norm(diff_old, p=2)
                hinge = torch.relu(float(self.cfg.contrastive_margin) - distance)
                reg = reg + float(self.cfg.beta) * hinge.pow(2)
        return reg


class FedProxTrainer(_BaseTrainer):
    def needs_global_weights(self) -> bool:
        return True

    def train(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        global_weights: Optional[TensorDict] = None,
        old_local_weights: Optional[TensorDict] = None,
        generator: Optional[torch.Generator] = None,
    ) -> TensorDict:
        del old_local_weights, generator
        model.train()
        optimizer = self._optimizer(model)
        criterion = self._criterion()
        init_weights = self._init_weights(model)

        for _ in range(int(self.cfg.local_epochs)):
            for data, target in dataloader:
                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(data), target)
                loss = loss + self._prox_loss(model, global_weights, device)
                loss.backward()
                optimizer.step()

        return self._delta_from_init(model, init_weights)

    def _prox_loss(
        self,
        model: nn.Module,
        global_weights: Optional[TensorDict],
        device: torch.device,
    ) -> torch.Tensor:
        if global_weights is None:
            return torch.tensor(0.0, device=device)
        reg = torch.tensor(0.0, device=device)
        for name, param in model.named_parameters():
            if param.requires_grad and name in global_weights:
                diff = param - global_weights[name]
                reg = reg + float(self.cfg.mu) * diff.pow(2).sum()
        return reg


class DPFedSAMTrainer(_BaseTrainer):
    def train(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        global_weights: Optional[TensorDict] = None,
        old_local_weights: Optional[TensorDict] = None,
        generator: Optional[torch.Generator] = None,
    ) -> TensorDict:
        del global_weights, old_local_weights, generator
        model.train()
        optimizer = self._optimizer(model)
        criterion = self._criterion()
        init_weights = self._init_weights(model)

        for _ in range(int(self.cfg.local_epochs)):
            for data, target in dataloader:
                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(data), target)
                loss.backward()

                grad_norm = self._grad_l2_norm(model, device)
                scale = float(self.cfg.sam_rho) / (grad_norm + float(self.cfg.sam_eps))
                perturbations = []
                with torch.no_grad():
                    for param in model.parameters():
                        if param.grad is None:
                            perturbations.append(None)
                            continue
                        e_w = param.grad * scale
                        param.add_(e_w)
                        perturbations.append(e_w)

                optimizer.zero_grad(set_to_none=True)
                loss_perturbed = criterion(model(data), target)
                loss_perturbed.backward()

                with torch.no_grad():
                    for param, e_w in zip(model.parameters(), perturbations):
                        if e_w is not None:
                            param.sub_(e_w)
                optimizer.step()

        return self._delta_from_init(model, init_weights)

    @staticmethod
    def _grad_l2_norm(model: nn.Module, device: torch.device) -> torch.Tensor:
        norms = [torch.norm(param.grad, p=2) for param in model.parameters() if param.grad is not None]
        if not norms:
            return torch.tensor(0.0, device=device)
        return torch.norm(torch.stack(norms), p=2)


class Compressor(Protocol):
    def move_state_to(self, device: torch.device) -> None:
        ...

    def move_state_to_cpu(self) -> None:
        ...

    def compress(
        self,
        delta_w: TensorDict,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        return_importance_snapshot: bool = False,
        importance_max_elements: int = 4096,
    ) -> CompressionResult:
        ...


class IdentityCompressor:
    def move_state_to(self, device: torch.device) -> None:
        del device

    def move_state_to_cpu(self) -> None:
        return None

    def compress(
        self,
        delta_w: TensorDict,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        return_importance_snapshot: bool = False,
        importance_max_elements: int = 4096,
    ) -> CompressionResult:
        del model, dataloader, device, return_importance_snapshot, importance_max_elements
        return CompressionResult(delta_w=delta_w, upload_ratio=1.0)


class TopKCompressor:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.residual_accumulator = ResidualAccumulator()
        self.fisher_cache: Optional[TensorDict] = None

    def move_state_to(self, device: torch.device) -> None:
        if self.residual_accumulator.residual:
            self.residual_accumulator.residual = {
                k: v.to(device) for k, v in self.residual_accumulator.residual.items()
            }
        if self.fisher_cache is not None:
            self.fisher_cache = {k: v.to(device) for k, v in self.fisher_cache.items()}

    def move_state_to_cpu(self) -> None:
        if self.residual_accumulator.residual:
            self.residual_accumulator.residual = {
                k: v.detach().cpu() for k, v in self.residual_accumulator.residual.items()
            }
        if self.fisher_cache is not None:
            self.fisher_cache = {k: v.detach().cpu() for k, v in self.fisher_cache.items()}

    def compress(
        self,
        delta_w: TensorDict,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        return_importance_snapshot: bool = False,
        importance_max_elements: int = 4096,
    ) -> CompressionResult:
        working_delta = delta_w
        if bool(self.cfg.use_residual):
            working_delta = self.residual_accumulator.accumulate(delta_w)

        importance = self._compute_importance(working_delta, model, dataloader, device)
        sparse_delta, masks = compress_gradient(
            gradient=working_delta,
            importance_strategy=str(self.cfg.importance_strategy),
            topk_strategy=str(self.cfg.topk_strategy),
            k_ratio=float(self.cfg.topk_ratio),
            weight_method=str(self.cfg.layer_weight_method),
            fisher=self.fisher_cache,
            importance=importance,
        )

        if bool(self.cfg.use_residual):
            self.residual_accumulator.update(working_delta, masks)

        total_params = sum(v.numel() for v in masks.values())
        kept_params = sum(v.sum().item() for v in masks.values())
        upload_ratio = float(kept_params / total_params) if total_params > 0 else 1.0

        importance_vector = None
        mask_vector = None
        if return_importance_snapshot:
            importance_vector = build_vector_snapshot(importance, max_elements=importance_max_elements)
            mask_vector = build_vector_snapshot(masks, max_elements=importance_max_elements)

        return CompressionResult(
            delta_w=sparse_delta,
            upload_ratio=upload_ratio,
            importance_dict=importance,
            mask_dict=masks,
            importance_vector=importance_vector,
            mask_vector=mask_vector,
        )

    def _compute_importance(
        self,
        delta_w: TensorDict,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
    ) -> TensorDict:
        strategy = str(self.cfg.importance_strategy)
        if strategy == "fisher_grad":
            if self.fisher_cache is None:
                self.fisher_cache = compute_fisher_information(model, dataloader, device)
            return {
                name: self.fisher_cache.get(name, torch.ones_like(d)).to(d.device) * d.abs()
                for name, d in delta_w.items()
            }
        if strategy == "grad_squared":
            return {name: d.abs() * d.abs() for name, d in delta_w.items()}
        return compute_importance_grad_normalized(delta_w)


class StandardL2Clipper:
    def clip(
        self,
        delta_w: TensorDict,
        clip_norm: float,
        clip_weights: Optional[TensorDict] = None,
    ) -> ClipResult:
        del clip_weights
        total_norm_sq = sum(d.pow(2).sum() for d in delta_w.values())
        total_norm = total_norm_sq.sqrt().item()
        clipped = total_norm > clip_norm
        if clipped:
            scale = clip_norm / max(total_norm, 1e-12)
            delta_w = {name: d * scale for name, d in delta_w.items()}
        return ClipResult(delta_w=delta_w, norm_value=total_norm, clipped=clipped)


class WeightedL2Clipper:
    def clip(
        self,
        delta_w: TensorDict,
        clip_norm: float,
        clip_weights: Optional[TensorDict] = None,
    ) -> ClipResult:
        if clip_weights is None:
            raise ValueError("clip_weights are required for weighted clipping")

        norm_terms = []
        for name, tensor in delta_w.items():
            if name not in clip_weights:
                raise KeyError(f"Missing clip weight template for parameter: {name}")

            weights = clip_weights[name].to(tensor.device).clamp_min(1e-6)
            # 中文说明：
            # 1. 带权裁剪计算的是 ||g||_{W^{-1}} = sqrt(sum_i (g_i / w_i)^2)。
            # 2. 这里除以 w_i，等价于先把更新映射到“噪声归一化坐标系”里再算普通 L2 范数。
            # 3. 当后续第 i 维高斯噪声标准差与 w_i 成比例时，这个范数对应的就是与噪声协方差匹配的椭球裁剪。
            # 4. baseline 使用同方差噪声，噪声球是各向同性的，因此不需要这套椭球裁剪，直接标准 L2 即可。
            norm_terms.append(((tensor / weights) ** 2).sum())

        weighted_norm = torch.stack(norm_terms).sum().sqrt().item() if norm_terms else 0.0
        clipped = weighted_norm > clip_norm
        if clipped:
            scale = clip_norm / max(weighted_norm, 1e-12)
            delta_w = {name: d * scale for name, d in delta_w.items()}
        return ClipResult(delta_w=delta_w, norm_value=weighted_norm, clipped=clipped)


class PrivacyEngine:
    def __init__(self, dp_cfg, num_rounds: int) -> None:
        self.dp_cfg = dp_cfg
        self.accountant = None
        if bool(dp_cfg.enabled):
            self.accountant = RDPAccountant(
                epsilon_total=float(dp_cfg.epsilon_total),
                delta=float(dp_cfg.delta),
                num_rounds=int(num_rounds),
                orders=tuple(float(order) for order in dp_cfg.rdp_orders),
            )

    def is_enabled(self) -> bool:
        return self.accountant is not None

    def is_exhausted(self) -> bool:
        return self.accountant.is_exhausted() if self.accountant is not None else False

    def calibrate_round(
        self,
        client_weights,
        clip_norm: float,
        sampling_rate: float,
        compressor_ratio: float = 1.0,
    ) -> PrivacyRoundState:
        if self.accountant is None:
            return PrivacyRoundState(
                noise_multiplier=0.0,
                sigma_agg=0.0,
                sensitivity_l2=0.0,
                clip_snr_proxy=0.0,
                q_accounting=0.0,
            )

        q_accounting = float(sampling_rate)
        if bool(self.dp_cfg.account_for_topk_in_q):
            q_accounting *= float(compressor_ratio)

        noise_multiplier = self.accountant.solve_noise_multiplier_for_round(
            q=q_accounting,
            steps=int(self.dp_cfg.rdp_steps_per_round),
        )
        sensitivity_l2 = compute_weighted_sensitivity(client_weights=client_weights, clip_norm=clip_norm)
        sigma_agg = noise_multiplier * sensitivity_l2
        clip_snr_proxy = float(clip_norm / max(sigma_agg, 1e-12)) if sigma_agg > 0 else 0.0

        return PrivacyRoundState(
            noise_multiplier=float(noise_multiplier),
            sigma_agg=float(sigma_agg),
            sensitivity_l2=float(sensitivity_l2),
            clip_snr_proxy=float(clip_snr_proxy),
            q_accounting=float(q_accounting),
        )

    def consume_round(self, round_state: PrivacyRoundState) -> float:
        if self.accountant is None:
            return 0.0
        return self.accountant.consume_round(
            noise_multiplier=round_state.noise_multiplier,
            q=round_state.q_accounting,
            steps=int(self.dp_cfg.rdp_steps_per_round),
        )

    def current_epsilon(self) -> float:
        return self.accountant.current_epsilon() if self.accountant is not None else 0.0

    def remaining_budget(self) -> float:
        return self.accountant.remaining_budget() if self.accountant is not None else 0.0

    def add_server_noise(
        self,
        global_update: TensorDict,
        sigma_agg: float,
        use_heterogeneous_noise: bool,
        clip_weight_template: Optional[TensorDict],
        generator: Optional[torch.Generator] = None,
    ) -> TensorDict:
        if sigma_agg <= 0:
            return global_update

        if not use_heterogeneous_noise:
            return {
                name: add_noise_to_tensor(tensor, sigma_agg, generator=generator)
                for name, tensor in global_update.items()
            }

        if clip_weight_template is None:
            raise ValueError("clip_weight_template is required for heterogeneous noise")

        flat_update = []
        flat_scales = []
        slices = []
        offset = 0
        for name, tensor in global_update.items():
            flat_tensor = tensor.flatten()
            flat_scale = clip_weight_template[name].to(tensor.device).flatten()
            flat_update.append(flat_tensor)
            flat_scales.append(flat_scale)
            slices.append((name, offset, offset + flat_tensor.numel(), tensor.shape))
            offset += flat_tensor.numel()

        merged_update = torch.cat(flat_update)
        merged_scales = torch.cat(flat_scales)
        mask = torch.ones_like(merged_update)
        noisy_update_flat, _ = add_heterogeneous_noise(
            tensor=merged_update,
            sigma_base=sigma_agg,
            relative_scales=merged_scales,
            mask=mask,
            generator=generator,
        )

        noisy_update: TensorDict = {}
        for name, start, end, shape in slices:
            noisy_update[name] = noisy_update_flat[start:end].reshape(shape)
        return noisy_update


def build_local_trainer(client_cfg) -> LocalTrainer:
    strategy = str(client_cfg.training_strategy)
    if strategy == "standard":
        return StandardTrainer(client_cfg)
    if strategy == "contrastive":
        return ContrastiveTrainer(client_cfg)
    if strategy == "fedprox":
        return FedProxTrainer(client_cfg)
    if strategy == "dp_fedsam":
        return DPFedSAMTrainer(client_cfg)
    raise ValueError(f"Unknown training strategy: {strategy}")


def build_compressor(compressor_cfg) -> Compressor:
    compressor_type = str(compressor_cfg.type)
    if compressor_type == "identity":
        return IdentityCompressor()
    if compressor_type == "topk":
        return TopKCompressor(compressor_cfg)
    raise ValueError(f"Unknown compressor type: {compressor_type}")


def build_clipper(use_weighted: bool):
    return WeightedL2Clipper() if use_weighted else StandardL2Clipper()


def build_vector_snapshot(tensors: TensorDict, max_elements: int = 4096) -> torch.Tensor:
    if not tensors:
        return torch.zeros(1)
    flat = torch.cat([tensor.detach().float().flatten().cpu() for tensor in tensors.values()])
    if flat.numel() <= max_elements:
        return flat
    idx = torch.linspace(0, flat.numel() - 1, steps=max_elements).long()
    return flat[idx]


def clone_tensor_dict(tensors: Optional[TensorDict]) -> Optional[TensorDict]:
    if tensors is None:
        return None
    return {k: v.detach().clone() for k, v in tensors.items()}
