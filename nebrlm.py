"""Nodal Edge and Boundary Representation Learning Module."""
from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn

from .layers import ConvBNAct


class CoordinateGate(nn.Module):
    """Direction-factorised gate .

    Pooling is performed separately along each spatial axis so that *where* a
    response occurs survives, unlike channel attention, which collapses both axes.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(8, channels // reduction)
        self.f1 = ConvBNAct(channels, mid, 1)
        self.to_h = nn.Conv2d(mid, channels, 1)
        self.to_w = nn.Conv2d(mid, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        z_h = x.mean(dim=3, keepdim=True)                    # B,C,H,1
        z_w = x.mean(dim=2, keepdim=True).transpose(2, 3)    # B,C,W,1
        y = self.f1(torch.cat([z_h, z_w], dim=2))
        y_h, y_w = torch.split(y, [h, w], dim=2)
        g_h = torch.sigmoid(self.to_h(y_h))
        g_w = torch.sigmoid(self.to_w(y_w.transpose(2, 3)))
        return g_h * g_w


class NEBRLM(nn.Module):
    """Gated multi-dilation edge evidence.

    Returns the edge-enhanced features. The coarse
    NEB map defines the valid zones used by GTRM.
    """

    def __init__(self, channels: int, dilations: Sequence[int] = (1, 2, 4), reduction: int = 16):
        super().__init__()
        self.gate = CoordinateGate(channels, reduction)
        self.extractors = nn.ModuleList([ConvBNAct(channels, channels, 3, dilation=d) for d in dilations])
        self.neb_head = nn.Sequential(ConvBNAct(channels, channels // 2, 3), nn.Conv2d(channels // 2, 1, 1))

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        g = self.gate(feat)
        f_edge = sum(g * extractor(feat) for extractor in self.extractors)
        neb_logits = self.neb_head(f_edge)
        return feat + f_edge, f_edge, neb_logits
