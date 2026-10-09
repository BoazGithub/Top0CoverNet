"""Evaluation over a data loader with semantic and structural metrics."""
from __future__ import annotations

from typing import Dict

import torch
from torch.utils.data import DataLoader

from ..metrics import SegmentationMetrics, StructuralMetrics


class Evaluator:
    def __init__(self, num_classes: int, ignore_index: int = 255, structural: bool = True,
                 bf1_tolerance: int = 2, min_region: int = 16, device: str | torch.device = "cuda"):
        self.semantic = SegmentationMetrics(num_classes, ignore_index)
        self.structural = StructuralMetrics(num_classes, ignore_index, bf1_tolerance, min_region) if structural else None
        self.device = torch.device(device)

    @torch.no_grad()
    def run(self, model: torch.nn.Module, loader: DataLoader) -> Dict[str, float]:
        model.eval()
        self.semantic.reset()
        if self.structural:
            self.structural.reset()
        for batch in loader:
            pred = model(batch["image"].to(self.device, non_blocking=True))["sem"].argmax(1).cpu()
            self.semantic.update(pred, batch["mask"])
            if self.structural:
                self.structural.update(pred, batch["mask"])
        results = self.semantic.compute()
        if self.structural:
            results.update(self.structural.compute())
        return results
