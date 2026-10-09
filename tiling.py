"""Patch extraction with overlap and leakage-free fold construction."""
from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import cv2
import numpy as np


class PatchTiler:
    """Cut scenes into ``size`` x ``size`` patches with fractional ``overlap``.

    Each patch is tagged with a geographic block id, ``<scene>_<row//block>_<col//block>``,
    so that :class:`SplitBuilder` can keep all patches of one block in the same fold.
    """

    def __init__(self, size: int = 512, overlap: float = 0.5, block: int = 2048):
        if not 0 <= overlap < 1:
            raise ValueError("overlap must be in [0, 1)")
        self.size = size
        self.stride = max(1, int(round(size * (1 - overlap))))
        self.block = block

    def windows(self, h: int, w: int) -> Iterator[Tuple[int, int]]:
        ys = list(range(0, max(h - self.size, 0) + 1, self.stride))
        xs = list(range(0, max(w - self.size, 0) + 1, self.stride))
        if ys[-1] + self.size < h:
            ys.append(h - self.size)
        if xs[-1] + self.size < w:
            xs.append(w - self.size)
        for y in ys:
            for x in xs:
                yield y, x

    def tile_scene(self, image_path: str, label_path: str, out_root: str) -> List[Dict[str, str]]:
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        lab = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
        if img is None or lab is None:
            raise FileNotFoundError(f"{image_path} / {label_path}")
        scene = Path(image_path).stem
        out = Path(out_root)
        (out / "images").mkdir(parents=True, exist_ok=True)
        (out / "labels").mkdir(parents=True, exist_ok=True)
        records = []
        for y, x in self.windows(*lab.shape[:2]):
            stem = f"{scene}_{y:06d}_{x:06d}"
            cv2.imwrite(str(out / "images" / f"{stem}.tif"), img[y:y + self.size, x:x + self.size])
            cv2.imwrite(str(out / "labels" / f"{stem}.png"), lab[y:y + self.size, x:x + self.size])
            records.append({"stem": stem, "block": f"{scene}_{y // self.block}_{x // self.block}"})
        return records


class SplitBuilder:
    """Grouped 70/15/15 splits for k folds: all patches of a geographic block share a split.

    With 50% patch overlap, splitting patches independently would leak overlapping
    content between training and testing; grouping by block prevents this.
    """

    def __init__(self, folds: int = 5, ratios=(0.70, 0.15, 0.15), seed: int = 0):
        if abs(sum(ratios) - 1.0) > 1e-6:
            raise ValueError("ratios must sum to 1")
        self.folds, self.ratios, self.seed = folds, ratios, seed

    def build(self, records: List[Dict[str, str]], out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        blocks = sorted({r["block"] for r in records})
        for k in range(self.folds):
            rng = np.random.default_rng(self.seed + k)
            order = list(rng.permutation(blocks))
            n_train = int(round(self.ratios[0] * len(order)))
            n_val = int(round(self.ratios[1] * len(order)))
            assign = {b: "train" for b in order[:n_train]}
            assign.update({b: "val" for b in order[n_train:n_train + n_val]})
            assign.update({b: "test" for b in order[n_train + n_val:]})
            for split in ("train", "val", "test"):
                with open(os.path.join(out_dir, f"fold{k}_{split}.txt"), "w") as handle:
                    handle.writelines(f"{r['stem']}\n" for r in records if assign[r["block"]] == split)
        with open(os.path.join(out_dir, "patches.csv"), "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["stem", "block"])
            writer.writeheader()
            writer.writerows(records)
