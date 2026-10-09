"""Graph-driven Topological Reasoning Module.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .layers import ConvBNAct

_NEIGHBOURS = ((0, 1), (1, 0), (0, -1), (-1, 0))


def _shift(x: torch.Tensor, dy: int, dx: int, fill) -> torch.Tensor:
    """Return ``out[b, y, x] = x[b, y + dy, x + dx]`` with ``fill`` outside the image."""
    _, h, w = x.shape
    out = torch.full_like(x, fill)
    out[:, max(-dy, 0):h - max(dy, 0), max(-dx, 0):w - max(dx, 0)] = \
        x[:, max(dy, 0):h - max(-dy, 0), max(dx, 0):w - max(-dx, 0)]
    return out


@dataclass
class RegionGraph:
    """Region adjacency graph for a batch.
    """

    node_map: torch.Tensor
    edge_index: torch.Tensor
    num_nodes: int
    node_batch: torch.Tensor

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1]) // 2


class RegionGraphBuilder:
    """Lines 1-4 of Algorithm 1: valid zones, 4-connected regions, geometric edges."""

    def __init__(self, tau: float = 0.3, min_size: int = 16, max_iter: int | None = None):
        self.tau = tau
        self.min_size = min_size
        self.max_iter = max_iter

    @torch.no_grad()
    def __call__(self, sem_logits: torch.Tensor, neb_logits: torch.Tensor) -> RegionGraph:
        cls = sem_logits.argmax(1)
        valid = torch.sigmoid(neb_logits[:, 0]) < self.tau
        node_map = self._components(cls, valid)
        if self.min_size > 1:
            node_map = self._merge_small(node_map)
        n = int(node_map.max().item()) + 1 if (node_map >= 0).any() else 0
        edge_index = self._edges(node_map, n)
        node_batch = torch.zeros(n, dtype=torch.long, device=cls.device)
        if n:
            bidx = torch.arange(cls.shape[0], device=cls.device).view(-1, 1, 1).expand_as(node_map)
            mask = node_map >= 0
            node_batch.scatter_(0, node_map[mask], bidx[mask])
        return RegionGraph(node_map, edge_index, n, node_batch)

    def _components(self, cls: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Connected-component labelling by min-label propagation with pointer jumping."""
        b, h, w = cls.shape
        total = b * h * w
        big = total
        ids = torch.arange(total, device=cls.device).view(b, h, w)
        lab = torch.where(valid, ids, torch.full_like(ids, big))
        for _ in range(self.max_iter or (h * w)):
            new = lab
            for dy, dx in _NEIGHBOURS:
                same = _shift(valid, dy, dx, False) & (_shift(cls, dy, dx, -1) == cls) & valid
                cand = torch.where(same, _shift(lab, dy, dx, big), torch.full_like(lab, big))
                new = torch.minimum(new, cand)
            # pointer jumping: adopt the label of the pixel our label points to
            flat = new.view(-1)
            jumped = torch.where(valid, flat[new.clamp(max=total - 1)].view(b, h, w), new)
            new = torch.minimum(new, jumped)
            if torch.equal(new, lab):
                break
            lab = new
        node_map = torch.full_like(lab, -1)
        if valid.any():
            _, inverse = torch.unique(lab[valid], return_inverse=True)
            node_map[valid] = inverse
        return node_map

    @staticmethod
    def _pairs(node_map: torch.Tensor):
        out_a, out_b = [], []
        for dy, dx in ((0, 1), (1, 0)):
            h, w = node_map.shape[-2:]
            a = node_map[:, :h - dy, :w - dx]
            c = node_map[:, dy:, dx:]
            m = (a >= 0) & (c >= 0) & (a != c)
            out_a += [a[m], c[m]]
            out_b += [c[m], a[m]]
        return torch.cat(out_a), torch.cat(out_b)

    def _merge_small(self, node_map: torch.Tensor) -> torch.Tensor:
        """Merge regions smaller than ``min_size`` into the neighbour with the longest shared boundary."""
        valid = node_map >= 0
        if not valid.any():
            return node_map
        n = int(node_map.max().item()) + 1
        sizes = torch.bincount(node_map[valid], minlength=n)
        a, c = self._pairs(node_map)
        if a.numel() == 0:
            return node_map
        keys, counts = torch.unique(a * n + c, return_counts=True)
        ua, uc = keys // n, keys % n
        # merge only into a strictly larger neighbour (ties broken by id) -> no cycles
        ok = (sizes[ua] < self.min_size) & ((sizes[uc] > sizes[ua]) | ((sizes[uc] == sizes[ua]) & (uc < ua)))
        if not ok.any():
            return node_map
        ua, uc, counts = ua[ok].cpu(), uc[ok].cpu(), counts[ok].cpu()
        order = torch.argsort(counts, stable=True)      # last write = longest shared boundary
        target = torch.arange(n)
        target[ua[order]] = uc[order]
        for _ in range(4):                               # follow short merge chains
            target = target[target]
        target = target.to(node_map.device)
        merged = torch.where(valid, target[node_map.clamp(min=0)], node_map)
        out = torch.full_like(merged, -1)
        _, inverse = torch.unique(merged[valid], return_inverse=True)
        out[valid] = inverse
        return out

    def _edges(self, node_map: torch.Tensor, n: int) -> torch.Tensor:
        if n == 0:
            return torch.zeros(2, 0, dtype=torch.long, device=node_map.device)
        a, c = self._pairs(node_map)
        if a.numel() == 0:
            return torch.zeros(2, 0, dtype=torch.long, device=node_map.device)
        keys = torch.unique(a * n + c)
        return torch.stack([keys // n, keys % n])


class RegionGCNLayer(nn.Module):
    """Symmetric-normalised message passing."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        wh = self.linear(h)
        if edge_index.numel():
            src, dst = edge_index
            deg = torch.bincount(dst, minlength=num_nodes).clamp(min=1).to(wh.dtype)
            norm = (deg[src] * deg[dst]).rsqrt().unsqueeze(1)
            wh = wh + torch.zeros_like(wh).index_add_(0, dst, wh[src] * norm)
        return self.act(wh)


class GTRM(nn.Module):
    """Valid-zone region graph construction, propagation and scatter."""

    def __init__(self, channels: int, num_classes: int, depth: int = 2, tau: float = 0.3,
                 min_region_size: int = 16, pooling: str = "concat"):
        super().__init__()
        if pooling not in {"concat", "avg", "max"}:
            raise ValueError("pooling must be 'concat', 'avg' or 'max'")
        self.pooling = pooling
        self.coarse_head = nn.Sequential(ConvBNAct(channels, channels, 3), nn.Conv2d(channels, num_classes, 1))
        self.builder = RegionGraphBuilder(tau, min_region_size)
        in_dim = 2 * channels if pooling == "concat" else channels
        self.layers = nn.ModuleList([RegionGCNLayer(in_dim if i == 0 else channels, channels) for i in range(depth)])
        self.out_proj = ConvBNAct(channels, channels, 1)

    def _pool(self, feats: torch.Tensor, idx: torch.Tensor, n: int) -> torch.Tensor:
        c = feats.shape[1]
        pooled = []
        if self.pooling in {"concat", "avg"}:
            counts = torch.bincount(idx, minlength=n).clamp(min=1).to(feats.dtype).unsqueeze(1)
            pooled.append(feats.new_zeros(n, c).index_add_(0, idx, feats) / counts)
        if self.pooling in {"concat", "max"}:
            pooled.append(feats.new_zeros(n, c).scatter_reduce(
                0, idx.unsqueeze(1).expand(-1, c), feats, reduce="amax", include_self=False))
        return torch.cat(pooled, dim=1)

    def forward(self, feat: torch.Tensor, neb_logits: torch.Tensor):
        sem_coarse = self.coarse_head(feat)
        graph = self.builder(sem_coarse, neb_logits)
        if graph.num_nodes == 0:
            return feat, sem_coarse, graph
        b, c, h, w = feat.shape
        flat = feat.permute(0, 2, 3, 1).reshape(-1, c)
        node_map = graph.node_map.view(-1)
        valid = node_map >= 0
        idx = node_map[valid]
        x = self._pool(flat[valid], idx, graph.num_nodes)
        for layer in self.layers:
            x = layer(x, graph.edge_index, graph.num_nodes)
        struct = flat.new_zeros(b * h * w, c)
        struct[valid] = x[idx]                                   # scatter node states to pixels
        struct = struct.view(b, h, w, c).permute(0, 3, 1, 2)
        return feat + self.out_proj(struct), sem_coarse, graph
