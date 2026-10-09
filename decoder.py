"""Topology-aware decoder emitting a semantic map and a binary NEB map."""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import ConvBNAct


class _UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(ConvBNAct(in_ch + skip_ch, out_ch, 3), ConvBNAct(out_ch, out_ch, 3))

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], 1))


class TopologyAwareDecoder(nn.Module):
    """Decoder stages D1..D5 (1/8 -> full resolution) with E3, E2, E1 skips."""

    def __init__(self, in_channels: int, skip_channels: Sequence[int], num_classes: int,
                 channels: Sequence[int] = (128, 96, 64, 48, 32)):
        super().__init__()
        e1, e2, e3 = skip_channels[:3]
        self.d1 = nn.Sequential(ConvBNAct(in_channels + e3, channels[0], 3), ConvBNAct(channels[0], channels[0], 3))
        self.d2 = _UpBlock(channels[0], e2, channels[1])
        self.d3 = _UpBlock(channels[1], e1, channels[2])
        self.d4 = ConvBNAct(channels[2], channels[3], 3)
        self.d5 = ConvBNAct(channels[3], channels[4], 3)
        self.sem_head = nn.Conv2d(channels[4], num_classes, 1)
        self.neb_head = nn.Conv2d(channels[4], 1, 1)

    def forward(self, x: torch.Tensor, skips: Sequence[torch.Tensor], out_size):
        e1, e2, e3 = skips[:3]
        x = F.interpolate(x, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        x = self.d1(torch.cat([x, e3], 1))
        x = self.d2(x, e2)
        x = self.d3(x, e1)
        x = self.d4(F.interpolate(x, size=out_size, mode="bilinear", align_corners=False))
        x = self.d5(x)
        return self.sem_head(x), self.neb_head(x)
