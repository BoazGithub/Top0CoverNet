"""Geometry-Aware Optimisation Objective (GAOO).

    L_total = L_CE^sem + lambda_1 L_edge^bin + lambda_2 L_topo + lambda_3 L_compact

with lambda = (0.5, 0.3, 0.2) and positive-class weight w = 8 for the NEB term.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryNEBLoss(nn.Module):
    """Weighted binary cross-entropy over the NEB map, Eq. (9)."""

    def __init__(self, pos_weight: float = 8.0):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor([pos_weight]), persistent=False)

    def forward(self, logits: torch.Tensor, target: torch.Tensor,
                valid: Optional[torch.Tensor] = None) -> torch.Tensor:
        logits = logits[:, 0] if logits.dim() == 4 else logits
        loss = F.binary_cross_entropy_with_logits(logits, target.float(), pos_weight=self.pos_weight,
                                                  reduction="none")
        if valid is not None:
            return (loss * valid).sum() / valid.sum().clamp(min=1)
        return loss.mean()


class CompactnessLoss(nn.Module):
    """Soft isoperimetric penalty, Eq. (11): mean over present classes of 1 - 4 pi A / P^2.

    ``A`` and ``P`` are computed differentiably from the soft class-probability map;
    the perimeter is the summed magnitude of the finite-difference gradient. Note that
    a pixelised square scores pi/4 under this criterion, so the term acts as a prior
    rather than reaching zero.
    """

    def __init__(self, ignore_index: int = 255, eps: float = 1e-6):
        super().__init__()
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        b, k = logits.shape[:2]
        valid = (target != self.ignore_index).unsqueeze(1).to(logits.dtype)
        probs = logits.softmax(1) * valid
        area = probs.sum((2, 3))
        gx = probs[..., :-1, 1:] - probs[..., :-1, :-1]
        gy = probs[..., 1:, :-1] - probs[..., :-1, :-1]
        perim = torch.sqrt(gx ** 2 + gy ** 2 + self.eps).sum((2, 3))
        compact = (4 * math.pi * area / perim.clamp(min=self.eps) ** 2).clamp(max=1.0)
        tgt = target.clone()
        tgt[tgt == self.ignore_index] = k
        present = F.one_hot(tgt, k + 1)[..., :k].sum((1, 2)) > 0         # C^+ per image
        if not present.any():
            return logits.sum() * 0.0
        return (1.0 - compact)[present].mean()


class PersistentHomologyLoss(nn.Module):

    def __init__(self, size: int = 64, dims: Sequence[int] = (0, 1), ignore_index: int = 255,
                 max_classes_per_image: Optional[int] = None):
        super().__init__()
        try:
            import gudhi  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError("PersistentHomologyLoss requires GUDHI: pip install gudhi") from exc
        self.size = size
        self.dims = tuple(dims)
        self.ignore_index = ignore_index
        self.max_classes = max_classes_per_image
        self._order: Optional[str] = None   # flattening order of GUDHI cell indices

    # ------------------------------------------------------------------ diagrams
    def _critical_pairs(self, f: np.ndarray):
        """Return {dim: int array (n, 2)} of (birth_idx, death_idx) in C order; death -1 if essential."""
        import gudhi

        h, w = f.shape
        cc = gudhi.CubicalComplex(top_dimensional_cells=f)
        cc.compute_persistence(homology_coeff_field=2, min_persistence=0.0)
        regular, essential = cc.cofaces_of_persistence_pairs()
        if self._order is None:
            self._order = self._infer_order(cc, f, regular)

        def to_c(idx: np.ndarray) -> np.ndarray:
            if self._order == "C":
                return idx
            return (idx % h) * w + (idx // h)

        pairs = {}
        for d in self.dims:
            reg = regular[d] if d < len(regular) else np.zeros((0, 2), dtype=np.int64)
            ess = essential[d] if d < len(essential) else np.zeros((0,), dtype=np.int64)
            reg = to_c(np.asarray(reg, dtype=np.int64).reshape(-1, 2))
            ess = to_c(np.asarray(ess, dtype=np.int64).reshape(-1))
            pairs[d] = np.concatenate([reg, np.stack([ess, -np.ones_like(ess)], 1)], 0)
        return pairs

    @staticmethod
    def _infer_order(cc, f: np.ndarray, regular) -> str:
        """Determine whether GUDHI cell indices follow C or Fortran flattening.

        Births of finite intervals are compared with the filtration values at the
        reported critical cells under both orders (GUDHI uses Fortran order for
        NumPy input in current releases; the check guards against version changes).
        """
        for d, reg in enumerate(regular):
            reg = np.asarray(reg).reshape(-1, 2)
            if len(reg) == 0:
                continue
            iv = np.asarray(cc.persistence_intervals_in_dimension(d)).reshape(-1, 2)
            births = np.sort(iv[np.isfinite(iv[:, 1]), 0])
            for order in ("F", "C"):
                if np.allclose(np.sort(f.ravel(order)[reg[:, 0]]), births):
                    return order
        return "F"

    def _betti(self, mask: np.ndarray) -> Dict[int, int]:
        """Number of persistent features of the binary reference in each dimension."""
        pairs = self._critical_pairs(1.0 - mask.astype(np.float64))
        flat = (1.0 - mask.astype(np.float64)).ravel()
        out = {}
        for d, p in pairs.items():
            if len(p) == 0:
                out[d] = 0
                continue
            births = flat[p[:, 0]]
            deaths = np.where(p[:, 1] >= 0, flat[np.clip(p[:, 1], 0, None)], 1.0)
            out[d] = int(((deaths - births) > 0.5).sum())
        return out

    # ------------------------------------------------------------------ loss
    def _single(self, prob: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        f = 1.0 - prob
        flat = f.reshape(-1)
        pairs = self._critical_pairs(f.detach().cpu().double().numpy())
        beta = self._betti(mask.cpu().numpy())
        loss = flat.new_zeros(())
        for d in self.dims:
            p = pairs[d]
            if len(p) == 0:
                continue
            bi = torch.as_tensor(p[:, 0], device=flat.device)
            di = torch.as_tensor(p[:, 1], device=flat.device)
            births = flat[bi]
            deaths = torch.where(di >= 0, flat[di.clamp(min=0)], torch.ones_like(births))
            order = torch.argsort((deaths - births).detach(), descending=True)
            k = min(beta[d], len(p))
            m, u = order[:k], order[k:]
            loss = loss + (births[m] ** 2).sum() + ((deaths[m] - 1.0) ** 2).sum()
            if u.numel():
                mid = 0.5 * (births[u] + deaths[u]).detach()
                loss = loss + ((births[u] - mid) ** 2).sum() + ((deaths[u] - mid) ** 2).sum()
        return loss

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = F.adaptive_avg_pool2d(logits.float().softmax(1), self.size)
        tgt = F.interpolate(target.unsqueeze(1).float(), size=(self.size, self.size), mode="nearest")[:, 0].long()
        total, count = probs.new_zeros(()), 0
        for b in range(probs.shape[0]):
            classes = [int(c) for c in torch.unique(tgt[b]) if int(c) != self.ignore_index]
            if self.max_classes is not None:
                classes = classes[: self.max_classes]
            for c in classes:
                total = total + self._single(probs[b, c], (tgt[b] == c).float())
                count += 1
        return total / max(count, 1)


class GAOOLoss(nn.Module):
    
    def __init__(self, num_classes: int, lambda_edge: float = 0.5, lambda_topo: float = 0.3,
                 lambda_compact: float = 0.2, pos_weight: float = 8.0, ignore_index: int = 255,
                 topo_size: int = 64, aux_weight: float = 0.4):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.lambdas = (lambda_edge, lambda_topo, lambda_compact)
        self.aux_weight = aux_weight
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.edge = BinaryNEBLoss(pos_weight)
        self.topo = PersistentHomologyLoss(topo_size, ignore_index=ignore_index) if lambda_topo > 0 else None
        self.compact = CompactnessLoss(ignore_index) if lambda_compact > 0 else None

    @staticmethod
    def _down_mask(mask: torch.Tensor, size) -> torch.Tensor:
        return F.interpolate(mask.unsqueeze(1).float(), size=size, mode="nearest")[:, 0].long()

    @staticmethod
    def _down_neb(neb: torch.Tensor, size) -> torch.Tensor:
        # adaptive max-pooling keeps thin boundaries visible at coarse resolution
        return F.adaptive_max_pool2d(neb.unsqueeze(1).float(), size)[:, 0]

    def forward(self, outputs: Dict[str, torch.Tensor], mask: torch.Tensor, neb: torch.Tensor) -> Dict[str, torch.Tensor]:
        l_edge_w, l_topo_w, l_comp_w = self.lambdas
        valid = (mask != self.ignore_index).float()
        terms = {"ce": self.ce(outputs["sem"], mask)}
        zero = terms["ce"].new_zeros(())
        terms["edge"] = self.edge(outputs["neb"], neb, valid) if l_edge_w > 0 else zero
        terms["topo"] = self.topo(outputs["sem"], mask) if self.topo is not None else zero
        terms["compact"] = self.compact(outputs["sem"], mask) if self.compact is not None else zero
        aux = zero
        if self.aux_weight > 0:
            if outputs.get("sem_coarse") is not None:
                size = outputs["sem_coarse"].shape[-2:]
                aux = aux + self.ce(outputs["sem_coarse"], self._down_mask(mask, size))
            if outputs.get("neb_coarse") is not None:
                size = outputs["neb_coarse"].shape[-2:]
                aux = aux + self.edge(outputs["neb_coarse"], self._down_neb(neb, size))
        terms["aux"] = aux
        terms["total"] = (terms["ce"] + l_edge_w * terms["edge"] + l_topo_w * terms["topo"]
                          + l_comp_w * terms["compact"] + self.aux_weight * aux)
        return terms


def build_loss(cfg, num_classes: int, ignore_index: int = 255) -> GAOOLoss:
    c = dict(cfg)
    return GAOOLoss(num_classes, c.get("lambda_edge", 0.5), c.get("lambda_topo", 0.3),
                    c.get("lambda_compact", 0.2), c.get("pos_weight", 8.0), ignore_index,
                    c.get("topo_size", 64), c.get("aux_weight", 0.4))
