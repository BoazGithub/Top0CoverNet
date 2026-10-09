"""Basic building blocks."""
from __future__ import annotations

import torch.nn as nn


class ConvBNAct(nn.Sequential):
    """Conv2d -> BatchNorm2d -> ReLU (activation optional)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1,
                 padding: int | None = None, dilation: int = 1, act: bool = True):
        if padding is None:
            padding = dilation * (kernel_size // 2)
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding,
                      dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        if act:
            layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)
