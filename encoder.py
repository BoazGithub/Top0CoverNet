"""Dual-branch CNN--ViT encoder and multi-scale fusion (Sec. III-B of the paper)."""
from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import ConvBNAct


class MultiKernelResidualBlock(nn.Module):
    """Dilated multi-kernel residual aggregation..
    """

    KERNELS = (1, 3, 5)

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, dilation: int = 1,
                 proportions: Sequence[float] = (1.0, 0.5, 0.25)):
        super().__init__()
        if len(proportions) != 3:
            raise ValueError("proportions must have three entries (1x1, 3x3, 5x5)")
        if all(p == 0 for p in proportions):
            raise ValueError("at least one kernel proportion must be non-zero")
        self.branches = nn.ModuleList()
        weights = []
        for k, p in zip(self.KERNELS, proportions):
            if p > 0:
                d = dilation if k > 1 else 1
                self.branches.append(ConvBNAct(in_ch, out_ch, k, stride=stride, dilation=d))
                weights.append(float(p))
        self.register_buffer("weights", torch.tensor(weights), persistent=False)
        self.refine = ConvBNAct(out_ch, out_ch, 3, act=False)
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                                          nn.BatchNorm2d(out_ch))
        else:
            self.shortcut = nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = sum(w * branch(x) for w, branch in zip(self.weights, self.branches))
        return self.act(self.refine(y) + self.shortcut(x))


class CNNBranch(nn.Module):
    """Five-stage convolutional branch supplying local texture."""

    def __init__(self, in_channels: int = 3, channels: Sequence[int] = (32, 64, 128, 256, 512),
                 blocks: Sequence[int] = (1, 2, 2, 2, 2),
                 proportions: Sequence[float] = (1.0, 0.5, 0.25)):
        super().__init__()
        self.stages = nn.ModuleList()
        prev = in_channels
        for i, (ch, n) in enumerate(zip(channels, blocks)):
            layers = [MultiKernelResidualBlock(prev, ch, stride=2, proportions=proportions)]
            for _ in range(n - 1):
                # dilation keeps the receptive field growing in the deepest stage
                layers.append(MultiKernelResidualBlock(ch, ch, dilation=2 if i == len(channels) - 1 else 1,
                                                       proportions=proportions))
            self.stages.append(nn.Sequential(*layers))
            prev = ch
        self.out_channels = list(channels)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats = []
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        return feats


class TransformerBranch(nn.Module):
    """Patch-token transformer."""

    def __init__(self, in_channels: int = 3, dim: int = 256, depth: int = 4, heads: int = 8,
                 patch_size: int = 16, mlp_ratio: float = 4.0, base_grid: int = 32, dropout: float = 0.0):
        super().__init__()
        self.patch_size = patch_size
        self.embed = nn.Conv2d(in_channels, dim, patch_size, stride=patch_size)
        self.pos = nn.Parameter(torch.zeros(1, dim, base_grid, base_grid))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(dim, heads, int(dim * mlp_ratio), dropout=dropout,
                                       activation="gelu", batch_first=True, norm_first=True)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.out_channels = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.embed(x)
        b, c, h, w = tokens.shape
        tokens = tokens + F.interpolate(self.pos, size=(h, w), mode="bicubic", align_corners=False)
        z = tokens.flatten(2).transpose(1, 2)
        for layer in self.layers:
            z = layer(z)
        z = self.norm(z)
        return z.transpose(1, 2).reshape(b, c, h, w)


class DualBranchEncoder(nn.Module):
    """CNN and transformer branches fused).
    """

    def __init__(self, in_channels: int = 3, mode: str = "dual",
                 cnn_channels: Sequence[int] = (32, 64, 128, 256, 512),
                 kernel_proportion: Sequence[float] = (1.0, 0.5, 0.25),
                 vit_dim: int = 256, vit_depth: int = 4, vit_heads: int = 8, patch_size: int = 16):
        super().__init__()
        if mode not in {"dual", "cnn", "vit"}:
            raise ValueError(f"unknown encoder mode '{mode}'")
        self.mode = mode
        self.out_channels = list(cnn_channels)
        if mode in {"dual", "cnn"}:
            self.cnn = CNNBranch(in_channels, cnn_channels, proportions=kernel_proportion)
        if mode in {"dual", "vit"}:
            self.vit = TransformerBranch(in_channels, vit_dim, vit_depth, vit_heads, patch_size)
        if mode == "dual":
            self.fuse4 = ConvBNAct(cnn_channels[3] + vit_dim, cnn_channels[3], 1)
            self.fuse5 = ConvBNAct(cnn_channels[4] + vit_dim, cnn_channels[4], 1)
        if mode == "vit":
            self.pyramid = nn.ModuleList([ConvBNAct(vit_dim, ch, 1) for ch in cnn_channels])

    @staticmethod
    def _resize(x: torch.Tensor, size) -> torch.Tensor:
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        if self.mode == "vit":
            t = self.vit(x)
            h, w = x.shape[-2:]
            sizes = [(-(-h // 2 ** (i + 1)), -(-w // 2 ** (i + 1))) for i in range(5)]
            return [proj(self._resize(t, s)) for proj, s in zip(self.pyramid, sizes)]
        feats = self.cnn(x)
        if self.mode == "dual":
            t = self.vit(x)
            e4, e5 = feats[3], feats[4]
            feats[3] = self.fuse4(torch.cat([e4, self._resize(t, e4.shape[-2:])], 1))
            feats[4] = self.fuse5(torch.cat([e5, self._resize(t, e5.shape[-2:])], 1))
        return feats


class MultiScaleFusion(nn.Module):
    """Fuse E2..E5 at 1/8 resolution.
    """

    def __init__(self, in_channels: Sequence[int], dim: int = 128, source: str = "all"):
        super().__init__()
        if source not in {"all", "last"}:
            raise ValueError("source must be 'all' or 'last'")
        self.source = source
        chans = list(in_channels) if source == "all" else [in_channels[-1]]
        self.proj = nn.ModuleList([ConvBNAct(c, dim, 1) for c in chans])
        self.mix = ConvBNAct(dim, dim, 3)
        self.out_channels = dim

    def forward(self, feats: Sequence[torch.Tensor]) -> torch.Tensor:
        """``feats`` is [E2, E3, E4, E5]; output is at the E3 (1/8) resolution."""
        size = feats[1].shape[-2:]
        inputs = feats if self.source == "all" else [feats[-1]]
        fused = sum(F.interpolate(p(f), size=size, mode="bilinear", align_corners=False)
                    for p, f in zip(self.proj, inputs))
        return self.mix(fused)
