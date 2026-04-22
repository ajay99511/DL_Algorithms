"""
ResNet-18 variant adapted for 32×32 CIFAR-10.

References:
    # Ref: He et al., 2016 — "Deep Residual Learning for Image Recognition"
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class ResidualBlock(nn.Module):
    """
    Basic residual block with two 3×3 convolutions and a skip connection.

    If in_channels != out_channels or stride != 1, a 1×1 conv is used
    to match dimensions in the shortcut path.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut projection when dimensions change
        self.shortcut: nn.Module
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, in_channels, H, W) -> (B, out_channels, H', W')"""
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = self.relu(out)
        return out


class SmallResNet(nn.Module):
    """
    ResNet-18 variant adapted for 32×32 CIFAR-10.

    Architecture:
        - Initial 3×3 conv (no max-pool, to preserve spatial resolution for 32×32)
        - 4 residual stages: [64, 128, 256, 512] channels
        - Global average pooling
        - Linear classifier

    # Ref: He et al., 2016 — "Deep Residual Learning for Image Recognition"
    """

    def __init__(self, n_classes: int = 10) -> None:
        super().__init__()

        # Initial convolution — use 3×3 (not 7×7) for small 32×32 images
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # 4 residual stages
        self.layer1 = self._make_layer(64, 64, stride=1)
        self.layer2 = self._make_layer(64, 128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)
        self.layer4 = self._make_layer(256, 512, stride=2)

        # Global average pooling + classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, n_classes)

        self._init_weights()

    def _make_layer(self, in_channels: int, out_channels: int, stride: int) -> nn.Sequential:
        """Create a residual stage with 2 blocks."""
        return nn.Sequential(
            ResidualBlock(in_channels, out_channels, stride=stride),
            ResidualBlock(out_channels, out_channels, stride=1),
        )

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, 3, 32, 32) image tensor

        Returns:
            (B, n_classes) logits
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
