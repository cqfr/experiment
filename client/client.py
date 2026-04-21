from __future__ import annotations

"""Federated client implementation with modular strategies."""

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from components.strategies import build_clipper, build_compressor, build_local_trainer
from dp.noise import add_heterogeneous_noise, add_noise_to_tensor

TensorDict = Dict[str, torch.Tensor]


@dataclass
class ClientUpdate:
    delta_w: TensorDict
    data_size: int
    stat: float
    clipped: bool
    upload_ratio: float = 1.0
    importance_dict: Optional[TensorDict] = None
    mask_dict: Optional[TensorDict] = None
    importance_vector: Optional[torch.Tensor] = None
    mask_vector: Optional[torch.Tensor] = None


class FLClient:
    """Client worker with pluggable local trainer, compressor, and clipper."""

    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        dataloader: DataLoader,
        cfg,
        device: torch.device,
    ) -> None:
        self.client_id = client_id
        # [MOD][阶段1] 默认保持 client 模型常驻，避免每轮强制 CPU/GPU 来回切换。
        self.model = model
        self.dataloader = dataloader
        self.cfg = cfg
        self.device = device

        self.data_size = len(dataloader.dataset)
        self.local_trainer = build_local_trainer(cfg.client)
        self.compressor = build_compressor(cfg.compressor)
        self.clipper = build_clipper(use_weighted=bool(cfg.dp.enabled and cfg.dp.use_heterogeneous_noise))

        self.old_local_weights: Optional[TensorDict] = None
        self.global_weights: Optional[TensorDict] = None
        self.last_importance_refresh_round = 0
        self._runtime_on_device = any(param.device.type == self.device.type for param in self.model.parameters())

    @staticmethod
    def _cfg_get(node, key: str, default):
        try:
            return node.get(key, default)
        except Exception:
            return getattr(node, key, default)

    def _move_runtime_state_to_device(self) -> None:
        # [MOD][阶段1] 仅在必要时迁移到 device，减少同步式 H2D 拷贝。
        if not self._runtime_on_device:
            self.model = self.model.to(self.device, non_blocking=True)
        if self.old_local_weights is not None:
            self.old_local_weights = {
                name: tensor.to(self.device, non_blocking=True)
                for name, tensor in self.old_local_weights.items()
            }
        self.compressor.move_state_to(self.device)
        self._runtime_on_device = True

    def _offload_runtime_state_to_cpu(self) -> None:
        # [MOD][阶段1] 仅在显存紧张或显式关闭驻留时才 offload，区别于原来的 finally 强制卸载。
        self.model = self.model.cpu()
        if self.old_local_weights is not None:
            self.old_local_weights = {
                name: tensor.detach().cpu()
                for name, tensor in self.old_local_weights.items()
            }
        self.compressor.move_state_to_cpu()
        self.global_weights = None
        self._runtime_on_device = False

    def _maybe_empty_cuda_cache(self) -> None:
        should_empty_cache = bool(self._cfg_get(self.cfg.trainer, "cuda_empty_cache_each_round", False))
        if self.device.type == "cuda" and should_empty_cache:
            torch.cuda.empty_cache()

    def _should_offload_after_round(self) -> bool:
        if self.device.type != "cuda":
            return False

        keep_on_device = bool(self._cfg_get(self.cfg.trainer, "keep_client_model_on_device", True))
        if not keep_on_device:
            return True

        min_free_ratio = float(self._cfg_get(self.cfg.trainer, "min_cuda_free_ratio", 0.15))
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
        except Exception:
            return False
        return (free_bytes / max(total_bytes, 1)) < min_free_ratio

    def _receive_global_model(self, global_weights: TensorDict) -> None:
        self.model.load_state_dict(global_weights)
        if self.local_trainer.needs_global_weights():
            self.global_weights = {
                name: tensor.detach().to(self.device, non_blocking=True)
                for name, tensor in global_weights.items()
            }
        else:
            self.global_weights = None

    def train_and_upload(
        self,
        global_weights: TensorDict,
        clip_norm: float,
        local_noise_std: float = 0.0,
        clip_weight_template: Optional[TensorDict] = None,
        global_importance_template: Optional[TensorDict] = None,
        need_importance_upload: bool = True,
        force_refresh_local_importance: bool = True,
        round_num: int = 0,
        return_importance_snapshot: bool = False,
        importance_max_elements: int = 4096,
        generator: Optional[torch.Generator] = None,
    ) -> ClientUpdate:
        self._move_runtime_state_to_device()
        try:
            self._receive_global_model(global_weights)

            delta_w = self.local_trainer.train(
                model=self.model,
                dataloader=self.dataloader,
                device=self.device,
                global_weights=self.global_weights,
                old_local_weights=self.old_local_weights,
                generator=generator,
            )

            if str(self.cfg.compressor.type) == "identity":
                compression_result = None
                clip_result = self.clipper.clip(
                    delta_w=delta_w,
                    clip_norm=clip_norm,
                    clip_weights=clip_weight_template,
                )
                upload_ratio = 1.0
                importance_dict = None
                mask_dict = None
                importance_vector = None
                mask_vector = None
            else:
                reuse_cached_local_importance = bool(not force_refresh_local_importance)
                compression_result = self.compressor.compress(
                    delta_w=delta_w,
                    model=self.model,
                    dataloader=self.dataloader,
                    device=self.device,
                    return_importance_snapshot=return_importance_snapshot,
                    importance_max_elements=importance_max_elements,
                    global_importance_template=global_importance_template,
                    global_local_mix_lambda=float(getattr(self.cfg.dp, "global_local_mix_lambda", 0.0)),
                    reuse_cached_local_importance=reuse_cached_local_importance,
                    force_refresh_local_importance=force_refresh_local_importance,
                )
                clip_result = self.clipper.clip(
                    delta_w=compression_result.delta_w,
                    clip_norm=clip_norm,
                    clip_weights=clip_weight_template,
                )
                upload_ratio = float(compression_result.upload_ratio)
                importance_dict = compression_result.importance_dict if need_importance_upload else None
                mask_dict = compression_result.mask_dict
                importance_vector = compression_result.importance_vector
                mask_vector = compression_result.mask_vector
                if force_refresh_local_importance:
                    self.last_importance_refresh_round = int(round_num)

            noisy_delta_w = self._apply_local_noise(
                delta_w=clip_result.delta_w,
                sigma_base=float(local_noise_std),
                clip_weight_template=clip_weight_template,
                mask_dict=mask_dict,
                generator=generator,
            )
            was_clipped = bool(clip_result.clipped)

            stat = 1.0 if not was_clipped else 0.0
            if str(self.cfg.client.stat_type) == "l2_norm":
                stat = float(clip_result.norm_value)

            if self.local_trainer.should_store_old_weights():
                self.old_local_weights = {
                    name: tensor.detach().clone()
                    for name, tensor in self.model.state_dict().items()
                }
            else:
                self.old_local_weights = None

            payload_delta_w = {name: tensor.detach().cpu() for name, tensor in noisy_delta_w.items()}
            payload_importance_dict = (
                {name: tensor.detach().cpu() for name, tensor in importance_dict.items()}
                if importance_dict is not None
                else None
            )
            payload_mask_dict = (
                {name: tensor.detach().cpu() for name, tensor in mask_dict.items()}
                if mask_dict is not None
                else None
            )
            payload_importance_vector = importance_vector.detach().cpu() if importance_vector is not None else None
            payload_mask_vector = mask_vector.detach().cpu() if mask_vector is not None else None

            # [MOD][阶段1] 显式释放中间张量，减少 GPU 上不必要的生命周期延长。
            del delta_w
            if compression_result is not None:
                del compression_result
            del clip_result
            del noisy_delta_w

            return ClientUpdate(
                delta_w=payload_delta_w,
                data_size=self.data_size,
                stat=stat,
                clipped=was_clipped,
                upload_ratio=upload_ratio,
                importance_dict=payload_importance_dict,
                mask_dict=payload_mask_dict,
                importance_vector=payload_importance_vector,
                mask_vector=payload_mask_vector,
            )
        finally:
            self.global_weights = None
            if self._should_offload_after_round():
                self._offload_runtime_state_to_cpu()
            self._maybe_empty_cuda_cache()

    def _apply_local_noise(
        self,
        delta_w: TensorDict,
        sigma_base: float,
        clip_weight_template: Optional[TensorDict],
        mask_dict: Optional[TensorDict],
        generator: Optional[torch.Generator],
    ) -> TensorDict:
        if sigma_base <= 0:
            return {name: tensor.detach().clone() for name, tensor in delta_w.items()}

        use_heterogeneous_noise = bool(self.cfg.dp.enabled and self.cfg.dp.use_heterogeneous_noise)
        if not use_heterogeneous_noise or clip_weight_template is None:
            # [MOD][阶段1] 均匀本地噪声路径，替代原来的 server-side central noise。
            return {
                name: add_noise_to_tensor(tensor, sigma_base, generator=generator)
                for name, tensor in delta_w.items()
            }

        noisy_delta: TensorDict = {}
        for name, tensor in delta_w.items():
            if name not in clip_weight_template:
                raise KeyError(f"Missing heterogeneous clip/noise template for parameter: {name}")
            relative_scales = clip_weight_template[name].to(tensor.device)
            if mask_dict is not None and name in mask_dict:
                mask = mask_dict[name].to(tensor.device)
            else:
                mask = torch.ones_like(tensor)
            noisy_tensor, _ = add_heterogeneous_noise(
                tensor=tensor,
                sigma_base=sigma_base,
                relative_scales=relative_scales,
                mask=mask,
                generator=generator,
            )
            noisy_delta[name] = noisy_tensor
        return noisy_delta
