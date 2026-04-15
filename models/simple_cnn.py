from __future__ import annotations

"""Lightweight CNN for CIFAR-10 under strict DP training."""

import torch
import torch.nn as nn


def _group_norm(num_channels: int) -> nn.GroupNorm:
    num_groups = min(8, num_channels)
    while num_channels % num_groups != 0 and num_groups > 1:
        num_groups -= 1
    return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)


class SimpleCNN(nn.Module):
    """A compact GroupNorm CNN with fewer than 500k parameters."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            _group_norm(32),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            _group_norm(64),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 96),
            nn.ReLU(inplace=False),
            nn.Linear(96, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


if __name__ == "__main__":
    model = SimpleCNN(num_classes=10)
    params = sum(p.numel() for p in model.parameters())
    print(f"params={params}")
