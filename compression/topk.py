from __future__ import annotations

"""Gradient importance scoring and Top-k compression utilities."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def compute_fisher_information(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_batches: int = 5,
) -> Dict[str, torch.Tensor]:
    """Estimate diagonal Fisher information from mini-batches."""

    model.eval()
    # 创建一个与模型参数同结构的字典，初始值为全零，用于存储Fisher信息
    fisher: Dict[str, torch.Tensor] = {
        name: torch.zeros_like(param.data)
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    total_samples = 0
    criterion = nn.CrossEntropyLoss(reduction="sum")

# 对角fisher矩阵
    for batch_idx, (data, target) in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
        data, target = data.to(device), target.to(device)
        batch_size = data.size(0)
        total_samples += batch_size

        model.zero_grad()
        loss = criterion(model(data), target)
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None and name in fisher:
                fisher[name] += param.grad.data.pow(2) * batch_size

    if total_samples > 0:
        for name in fisher:
            fisher[name] /= total_samples

    model.train()
    return fisher


def compute_importance_fisher_grad(
    gradient: Dict[str, torch.Tensor],
    fisher: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    return {
        name: fisher.get(name, torch.ones_like(grad)) * grad.abs()
        for name, grad in gradient.items()
    }


def compute_importance_grad_squared(
    gradient: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    return {name: grad.abs() * grad.abs() for name, grad in gradient.items()}


def compute_importance_grad_normalized(
    gradient: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    importance: Dict[str, torch.Tensor] = {}
    for name, grad in gradient.items():
        abs_grad = grad.abs()
        min_val = abs_grad.min()
        max_val = abs_grad.max()
        if max_val > min_val:
            importance[name] = (abs_grad - min_val) / (max_val - min_val)
        else:
            importance[name] = torch.ones_like(abs_grad)
    return importance


def topk_global(importance: Dict[str, torch.Tensor], k_ratio: float) -> Dict[str, torch.Tensor]:
    all_values = torch.cat([imp.flatten() for imp in importance.values()])
    total = all_values.numel() # 计算所有重要性值的总数量
    k = max(1, int(total * k_ratio))

    threshold = torch.topk(all_values, k).values[-1] # 通过阈值来计算的
    return {name: (imp >= threshold).float() for name, imp in importance.items()}


def topk_layer(importance: Dict[str, torch.Tensor], k_ratio: float) -> Dict[str, torch.Tensor]:
    masks: Dict[str, torch.Tensor] = {}
    for name, imp in importance.items():
        flat = imp.flatten()
        k = max(1, int(flat.numel() * k_ratio))
        _, idx = torch.topk(flat, k)
        m = torch.zeros_like(flat)
        m[idx] = 1.0
        masks[name] = m.reshape_as(imp)
    return masks


def topk_layer_norm_global(
    importance: Dict[str, torch.Tensor],
    k_ratio: float,
) -> Dict[str, torch.Tensor]:
    layer_norm = compute_importance_grad_normalized(importance) # 对每一层的重要性进行归一化
    return topk_global(layer_norm, k_ratio)

# 这尼玛是什么神人想出来的，感觉没什么用啊。，意义不明，自相矛盾了，有点
def topk_weighted_layer_norm(
    importance: Dict[str, torch.Tensor],
    k_ratio: float,
    weight_method: str = "mean",
) -> Dict[str, torch.Tensor]:
    layer_norm = compute_importance_grad_normalized(importance)
    layer_weights: Dict[str, float] = {}

    for name, imp in importance.items(): #感觉仅适用于fisher矩阵
        flat = imp.flatten()
        if weight_method == "median":
            layer_weights[name] = flat.median().item()
        elif weight_method == "total_sum":
            layer_weights[name] = flat.sum().item()
        elif weight_method == "trimmed_mean":
            sorted_vals = flat.sort().values
            trim = int(0.05 * sorted_vals.numel())
            if trim > 0 and sorted_vals.numel() > 2 * trim:
                layer_weights[name] = sorted_vals[trim:-trim].mean().item()
            else:
                layer_weights[name] = flat.mean().item()
        else:
            layer_weights[name] = flat.mean().item()

    max_w = max(layer_weights.values()) if layer_weights else 1.0
    if max_w <= 0:
        max_w = 1.0

    weighted = {
        name: layer_norm[name] * (layer_weights[name] / max_w)
        for name in layer_norm
    }
    return topk_global(weighted, k_ratio)


class ResidualAccumulator:
    """Error-feedback residual accumulator."""

    def __init__(self) -> None:
        self.residual: Dict[str, torch.Tensor] = {}

    def accumulate(self, gradient: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if not self.residual:
            return gradient
        return {
            name: grad + self.residual.get(name, torch.zeros_like(grad))# 找得到name就返回name对应的residual，否则返回一个全零的tensor
            for name, grad in gradient.items()
        }

    def update(self, gradient: Dict[str, torch.Tensor], mask: Dict[str, torch.Tensor]) -> None:
        self.residual = {}
        for name, grad in gradient.items():
            if name in mask:
                self.residual[name] = grad * (1.0 - mask[name])

    def reset(self) -> None:
        self.residual = {}


def compress_gradient(
    gradient: Dict[str, torch.Tensor],
    importance_strategy: str,
    topk_strategy: str,
    k_ratio: float,
    weight_method: str = "mean",
    fisher: Optional[Dict[str, torch.Tensor]] = None,
    importance: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Unified API for importance scoring + top-k masking."""

    if importance is None: # 如果没有就重新计算一遍
        if importance_strategy == "fisher_grad":
            if fisher is None:
                raise ValueError("fisher is required when using fisher_grad")
            importance = compute_importance_fisher_grad(gradient, fisher)
        elif importance_strategy == "grad_squared":
            importance = compute_importance_grad_squared(gradient)
        elif importance_strategy == "grad_normalized":
            importance = compute_importance_grad_normalized(gradient)
        else:
            raise ValueError(f"unknown importance strategy: {importance_strategy}")

    if topk_strategy == "global_topk":
        masks = topk_global(importance, k_ratio)
    elif topk_strategy == "layer_topk":
        masks = topk_layer(importance, k_ratio)
    elif topk_strategy == "layer_norm_global":
        masks = topk_layer_norm_global(importance, k_ratio)
    elif topk_strategy == "weighted_layer_norm":
        masks = topk_weighted_layer_norm(importance, k_ratio, weight_method)
    else:
        raise ValueError(f"unknown topk strategy: {topk_strategy}")

    sparse = {
        name: grad * masks.get(name, torch.ones_like(grad))
        for name, grad in gradient.items()
    }
    return sparse, masks


if __name__ == "__main__":
    g = {
        "a": torch.randn(16, 8),
        "b": torch.randn(16),
    }
    sparse, masks = compress_gradient(
        gradient=g,
        importance_strategy="grad_normalized",
        topk_strategy="weighted_layer_norm",
        k_ratio=0.2,
    )
    print(sum(v.sum().item() for v in masks.values()))
