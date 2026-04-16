from __future__ import annotations

"""Lightweight CNN for CIFAR-10 under strict DP training."""

import torch
import torch.nn as nn


def _group_norm(num_channels: int) -> nn.GroupNorm:
    num_groups = min(8, num_channels)
    while num_groups > 1 and num_channels % num_groups != 0:
        num_groups -= 1
    return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            _group_norm(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            _group_norm(out_channels),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SimpleCNN(nn.Module):
    """A compact CIFAR-10 CNN with GroupNorm and fewer than 500k parameters."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 48),
            ConvBlock(48, 96),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 8 * 8, 48),
            nn.ReLU(inplace=False),
            nn.Linear(48, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)
