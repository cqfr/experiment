from __future__ import annotations

"""Federated client implementation with modular strategies."""

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from components.strategies import (
    build_clipper,
    build_compressor,
    build_local_trainer,
    clone_tensor_dict,
)

TensorDict = Dict[str, torch.Tensor]


@dataclass
class ClientUpdate:
    """Payload uploaded by a client."""

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
        self.model = model.cpu()
        self.dataloader = dataloader
        self.cfg = cfg
        self.device = device

        self.data_size = len(dataloader.dataset)
        self.local_trainer = build_local_trainer(cfg.client)
        self.compressor = build_compressor(cfg.compressor)
        self.clipper = build_clipper(
            use_weighted=bool(cfg.dp.enabled and cfg.dp.use_heterogeneous_noise)
        )

        self.old_local_weights: Optional[TensorDict] = None
        self.global_weights: Optional[TensorDict] = None

    def _move_runtime_state_to_device(self) -> None:
        self.model = self.model.to(self.device)
        if self.old_local_weights is not None:
            self.old_local_weights = {k: v.to(self.device) for k, v in self.old_local_weights.items()}
        self.compressor.move_state_to(self.device)

    def _move_runtime_state_to_cpu(self) -> None:
        self.model = self.model.cpu()
        if self.old_local_weights is not None:
            self.old_local_weights = {k: v.detach().cpu() for k, v in self.old_local_weights.items()}
        self.compressor.move_state_to_cpu()
        self.global_weights = None

    def _receive_global_model(self, global_weights: TensorDict) -> None:
        self.model.load_state_dict(global_weights)
        if self.local_trainer.needs_global_weights():
            self.global_weights = {k: v.detach().clone().to(self.device) for k, v in global_weights.items()}
        else:
            self.global_weights = None

    def train_and_upload(
        self,
        global_weights: TensorDict,
        clip_norm: float,
        clip_weight_template: Optional[TensorDict] = None,
        return_importance_snapshot: bool = False,
        importance_max_elements: int = 4096,
        generator: Optional[torch.Generator] = None,
    ) -> ClientUpdate:
        del generator
        self._move_runtime_state_to_device()
        try:
            self._receive_global_model(global_weights)

            delta_w = self.local_trainer.train(
                model=self.model,
                dataloader=self.dataloader,
                device=self.device,
                global_weights=self.global_weights,
                old_local_weights=self.old_local_weights,
            )

            compression_result = self.compressor.compress(
                delta_w=delta_w,
                model=self.model,
                dataloader=self.dataloader,
                device=self.device,
                return_importance_snapshot=return_importance_snapshot,
                importance_max_elements=importance_max_elements,
            )

            clip_result = self.clipper.clip(
                delta_w=compression_result.delta_w,
                clip_norm=clip_norm,
                clip_weights=clip_weight_template,
            )

            stat = 1.0 if clip_result.clipped else 0.0
            if str(self.cfg.client.stat_type) == "l2_norm":
                stat = float(clip_result.norm_value)

            if self.local_trainer.should_store_old_weights():
                self.old_local_weights = {
                    k: v.detach().clone()
                    for k, v in self.model.state_dict().items()
                }
            else:
                self.old_local_weights = None

            return ClientUpdate(
                delta_w={name: tensor.detach().cpu() for name, tensor in clip_result.delta_w.items()},
                data_size=self.data_size,
                stat=stat,
                clipped=clip_result.clipped,
                upload_ratio=float(compression_result.upload_ratio),
                importance_dict=(
                    {name: tensor.detach().cpu() for name, tensor in compression_result.importance_dict.items()}
                    if compression_result.importance_dict is not None
                    else None
                ),
                mask_dict=(
                    {name: tensor.detach().cpu() for name, tensor in compression_result.mask_dict.items()}
                    if compression_result.mask_dict is not None
                    else None
                ),
                importance_vector=(
                    compression_result.importance_vector.detach().cpu()
                    if compression_result.importance_vector is not None
                    else None
                ),
                mask_vector=(
                    compression_result.mask_vector.detach().cpu()
                    if compression_result.mask_vector is not None
                    else None
                ),
            )
        finally:
            self._move_runtime_state_to_cpu()

    def get_clip_weight_template_copy(self, template: Optional[TensorDict]) -> Optional[TensorDict]:
        return clone_tensor_dict(template)


if __name__ == "__main__":
    print("client module ready")
