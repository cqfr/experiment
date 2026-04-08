"""
models/resnet.py
ResNet-18，所有 BatchNorm 替换为 GroupNorm，兼容 Opacus。
"""

import torch
import torch.nn as nn

# ──────────────────────────────────────────────
# 工具函数：构建 GroupNorm
# ──────────────────────────────────────────────

def gn(num_channels: int) -> nn.GroupNorm:
    """
    统一的 GroupNorm 构造函数。
    num_groups=32 是标准默认值；当通道数较少时自动降为 num_channels//2。
    """
    num_groups = min(32, num_channels // 2)
    return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)


# ──────────────────────────────────────────────
# 残差块
# ──────────────────────────────────────────────

class BasicBlock(nn.Module):
    """
    ResNet-18 的基本残差块，包含两个 3x3 卷积。

    结构：
        x → Conv → GN → ReLU → Conv → GN → (+x) → ReLU → 输出
                                              ↑
                                          shortcut
    当输入输出维度不一致时（stride=2 或通道数变化），
    shortcut 用 1x1 卷积做维度对齐。
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1 = gn(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = gn(out_channels)

        self.relu = nn.ReLU(inplace=False)

        # shortcut：维度不一致时需要对齐
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                gn(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out = out + self.shortcut(x)   # 残差连接
        out = self.relu(out)
        return out


# ──────────────────────────────────────────────
# ResNet-18
# ──────────────────────────────────────────────

class ResNet18(nn.Module):
    """
    ResNet-18（GroupNorm 版本），适配 CIFAR-10。

    与标准 ImageNet ResNet-18 的区别：
    - 初始卷积改为 3x3（原为 7x7），适配 32x32 的小图
    - 去掉初始 MaxPool，避免特征图过早缩小
    - 所有 BatchNorm 替换为 GroupNorm
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        # 初始层：针对 CIFAR-10 小图优化
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            gn(64),
            nn.ReLU(inplace=False),
        )

        # 4 个 Layer，每个包含 2 个残差块
        self.layer1 = self._make_layer(64,  64,  stride=1)
        self.layer2 = self._make_layer(64,  128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)
        self.layer4 = self._make_layer(256, 512, stride=2)

        # 分类头
        self.pool = nn.AdaptiveAvgPool2d(1)   # 全局平均池化 → (B, 512, 1, 1)
        self.fc   = nn.Linear(512, num_classes)

    @staticmethod
    def _make_layer(in_channels: int, out_channels: int, stride: int) -> nn.Sequential:
        """构建一个 Layer：2 个残差块，只有第一个块可能改变分辨率/通道数。"""
        return nn.Sequential(
            BasicBlock(in_channels, out_channels, stride=stride),
            BasicBlock(out_channels, out_channels, stride=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)     # (B, 64, 32, 32)
        x = self.layer1(x)   # (B, 64, 32, 32)
        x = self.layer2(x)   # (B, 128, 16, 16)
        x = self.layer3(x)   # (B, 256, 8, 8)
        x = self.layer4(x)   # (B, 512, 4, 4)
        x = self.pool(x)     # (B, 512, 1, 1)
        x = x.flatten(1)     # (B, 512)
        x = self.fc(x)       # (B, 10)
        return x
# ──────────────────────────────────────────────
# 快速验证
# ──────────────────────────────────────────────

if __name__ == "__main__":
    model = ResNet18(num_classes=10)

    # 统计参数量
    total = sum(p.numel() for p in model.parameters())
    print(f"参数量：{total / 1e6:.2f}M")

    # 验证前向传播
    x = torch.randn(4, 3, 32, 32)   # batch_size=4, CIFAR-10 图片
    out = model(x)
    print(f"输入 shape：{x.shape}")
    print(f"输出 shape：{out.shape}")  # 期望：(4, 10)
    print("前向传播正常 ✓")