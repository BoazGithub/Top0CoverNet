"""Semantic and structural metrics).

Semantic: OA, mIoU, macro F1, FWIoU and Cohen's kappa from a pooled confusion matrix.
Structural:
    * BF1 - NEB F1 within a tolerance band of the reference NEB (pooled counts);
    * CP  - compactness averaged over predicted regions;
    * GR  - gap ratio, the share of reference regions returned as >1 connected component.
"""
from __future__ import annotations

import math
from typing import Dict

import numpy as np
import torch
from scipy import ndimage

_FOUR = ndimage.generate_binary_structure(2, 1)


def _boundary(label: np.ndarray, ignore_index: int) -> np.ndarray:
    b = np.zeros(label.shape, bool)
    for axis in (0, 1):
        a = np.take(label, range(label.shape[axis] - 1), axis=axis)
        c = np.take(label, range(1, label.shape[axis]), axis=axis)
        d = (a != c) & (a != ignore_index) & (c != ignore_index)
        pad_lo = [(0, 0), (0, 0)]
        pad_hi = [(0, 0), (0, 0)]
        pad_lo[axis], pad_hi[axis] = (0, 1), (1, 0)
        b |= np.pad(d, pad_lo) | np.pad(d, pad_hi)
    return b


class SegmentationMetrics:
    """Accumulates a confusion matrix over batches."""

    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self) -> None:
        self.cm = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, pred: torch.Tensor | np.ndarray, target: torch.Tensor | np.ndarray) -> None:
        pred = pred.cpu().numpy() if torch.is_tensor(pred) else pred
        target = target.cpu().numpy() if torch.is_tensor(target) else target
        m = target != self.ignore_index
        idx = self.num_classes * target[m].astype(np.int64) + pred[m].astype(np.int64)
        self.cm += np.bincount(idx, minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)

    def compute(self) -> Dict[str, float]:
        cm = self.cm.astype(np.float64)
        tp = np.diag(cm)
        gt, pr, total = cm.sum(1), cm.sum(0), cm.sum()
        present = gt > 0
        iou = tp / np.maximum(gt + pr - tp, 1)
        f1 = 2 * tp / np.maximum(gt + pr, 1)
        oa = tp.sum() / max(total, 1)
        pe = (gt * pr).sum() / max(total ** 2, 1)
        return {
            "OA": float(100 * oa),
            "mIoU": float(100 * iou[present].mean()) if present.any() else 0.0,
            "F1": float(f1[present].mean()) if present.any() else 0.0,
            "FWIoU": float(100 * (gt / max(total, 1) * iou).sum()),
            "Kappa": float((oa - pe) / max(1 - pe, 1e-12)),
            "IoU_per_class": [float(v) for v in 100 * iou],
        }


class StructuralMetrics:
    """Pooled BF1, CP and GR over a dataset."""

    def __init__(self, num_classes: int, ignore_index: int = 255, tolerance: int = 2, min_region: int = 16):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.tolerance = tolerance
        self.min_region = min_region
        self._struct = ndimage.generate_binary_structure(2, 1)
        self.reset()

    def reset(self) -> None:
        self.bp_hit = self.bp_tot = self.br_hit = self.br_tot = 0
        self.cp_sum, self.cp_n = 0.0, 0
        self.gaps, self.regions = 0, 0

    def _dilate(self, b: np.ndarray) -> np.ndarray:
        return ndimage.binary_dilation(b, self._struct, iterations=self.tolerance) if self.tolerance else b

    @staticmethod
    def _perimeter(region: np.ndarray) -> int:
        padded = np.pad(region, 1)
        return int((padded[1:, :] != padded[:-1, :]).sum() + (padded[:, 1:] != padded[:, :-1]).sum())

    def _update_one(self, pred: np.ndarray, gt: np.ndarray) -> None:
        valid = gt != self.ignore_index
        pred = np.where(valid, pred, self.ignore_index)
        pb, gb = _boundary(pred, self.ignore_index), _boundary(gt, self.ignore_index)
        self.bp_hit += int((pb & self._dilate(gb)).sum()); self.bp_tot += int(pb.sum())
        self.br_hit += int((gb & self._dilate(pb)).sum()); self.br_tot += int(gb.sum())
        for c in range(self.num_classes):
            pc = pred == c
            if pc.any():
                lab, n = ndimage.label(pc, _FOUR)
                areas = np.bincount(lab.ravel())[1:]
                for r in np.nonzero(areas >= self.min_region)[0] + 1:
                    region = lab == r
                    self.cp_sum += 4 * math.pi * areas[r - 1] / self._perimeter(region) ** 2
                    self.cp_n += 1
            gc = gt == c
            if gc.any():
                glab, gn = ndimage.label(gc, _FOUR)
                gareas = np.bincount(glab.ravel())[1:]
                for r in np.nonzero(gareas >= self.min_region)[0] + 1:
                    inside = (glab == r) & pc
                    _, k = ndimage.label(inside, _FOUR)
                    self.regions += 1
                    self.gaps += int(k > 1)

    def update(self, pred, target) -> None:
        pred = pred.cpu().numpy() if torch.is_tensor(pred) else pred
        target = target.cpu().numpy() if torch.is_tensor(target) else target
        if pred.ndim == 2:
            pred, target = pred[None], target[None]
        for p, g in zip(pred, target):
            self._update_one(p, g)

    def compute(self) -> Dict[str, float]:
        precision = self.bp_hit / max(self.bp_tot, 1)
        recall = self.br_hit / max(self.br_tot, 1)
        return {
            "BF1": float(2 * precision * recall / max(precision + recall, 1e-12)),
            "CP": float(self.cp_sum / max(self.cp_n, 1)),
            "GR": float(self.gaps / max(self.regions, 1)),
        }
