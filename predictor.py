"""Sliding-window inference for large scenes with overlap blending."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F


class SlidingWindowPredictor:
    """Predict a full scene by blending overlapping tiles with a Hann window.

    Blending the class probabilities, rather than stitching hard labels, avoids the
    seam artefacts that independent tile predictions otherwise produce.
    """

    def __init__(self, model: torch.nn.Module, num_classes: int, tile: int = 512, overlap: float = 0.5,
                 batch_size: int = 4, device: str | torch.device = "cuda",
                 mean=None, std=None):
        self.model = model.eval().to(device)
        self.num_classes, self.tile = num_classes, tile
        self.stride = max(1, int(tile * (1 - overlap)))
        self.batch_size, self.device = batch_size, torch.device(device)
        self.mean = None if mean is None else torch.tensor(mean).view(1, -1, 1, 1)
        self.std = None if std is None else torch.tensor(std).view(1, -1, 1, 1)
        w = torch.hann_window(tile, periodic=False).clamp(min=1e-3)
        self.weight = (w[:, None] * w[None, :]).to(self.device)

    def _positions(self, h: int, w: int):
        ys = list(range(0, max(h - self.tile, 0) + 1, self.stride))
        xs = list(range(0, max(w - self.tile, 0) + 1, self.stride))
        if ys[-1] + self.tile < h:
            ys.append(h - self.tile)
        if xs[-1] + self.tile < w:
            xs.append(w - self.tile)
        return [(y, x) for y in ys for x in xs]

    @torch.no_grad()
    def __call__(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """``image``: float32 HxWxC in [0, 1]. Returns ``label`` (HxW), ``prob`` (KxHxW), ``neb`` (HxW)."""
        h0, w0 = image.shape[:2]
        x = torch.from_numpy(image.transpose(2, 0, 1)).float()[None]
        ph, pw = max(0, self.tile - h0), max(0, self.tile - w0)
        x = F.pad(x, (0, pw, 0, ph), mode="reflect") if (ph or pw) else x
        if self.mean is not None:
            x = (x - self.mean) / self.std
        _, _, h, w = x.shape
        prob = torch.zeros(self.num_classes, h, w, device=self.device)
        neb = torch.zeros(h, w, device=self.device)
        norm = torch.zeros(h, w, device=self.device)
        pos = self._positions(h, w)
        for i in range(0, len(pos), self.batch_size):
            chunk = pos[i:i + self.batch_size]
            tiles = torch.cat([x[..., y:y + self.tile, xx:xx + self.tile] for y, xx in chunk]).to(self.device)
            out = self.model(tiles)
            p, b = out["sem"].softmax(1), torch.sigmoid(out["neb"][:, 0])
            for k, (y, xx) in enumerate(chunk):
                prob[:, y:y + self.tile, xx:xx + self.tile] += p[k] * self.weight
                neb[y:y + self.tile, xx:xx + self.tile] += b[k] * self.weight
                norm[y:y + self.tile, xx:xx + self.tile] += self.weight
        prob, neb = prob / norm, neb / norm
        prob, neb = prob[:, :h0, :w0], neb[:h0, :w0]
        return {"label": prob.argmax(0).cpu().numpy().astype(np.uint8), "prob": prob.cpu().numpy(),
                "neb": neb.cpu().numpy()}
