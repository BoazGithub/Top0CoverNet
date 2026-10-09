"""Datasets, NEB target generation and augmentation."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

IMG_EXT = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


def read_image(path: str) -> np.ndarray:
    """Read an image as float32 HxWxC in [0, 1] (multi-band GeoTIFFs via tifffile if available)."""
    img = None
    if path.lower().endswith((".tif", ".tiff")):
        try:
            import tifffile
            img = tifffile.imread(path)
            if img.ndim == 3 and img.shape[0] < img.shape[-1] and img.shape[0] <= 16:
                img = np.transpose(img, (1, 2, 0))          # bands-first -> bands-last
        except Exception:  # tifffile missing, or a codec (e.g. LZW) needing imagecodecs
            img = None
    if img is None:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None and img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 2:
        img = img[..., None]
    scale = np.iinfo(img.dtype).max if np.issubdtype(img.dtype, np.integer) else 1.0
    return (img.astype(np.float32) / scale).clip(0, 1)


def read_mask(path: str) -> np.ndarray:
    mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(path)
    return mask[..., 0] if mask.ndim == 3 else mask


class NEBTargetGenerator:
    """Derive the binary NEB field b from a label map y..
    """

    def __init__(self, width: int = 1, ignore_index: int = 255):
        self.width = width
        self.ignore_index = ignore_index

    def __call__(self, mask: np.ndarray) -> np.ndarray:
        m = mask.astype(np.int32)
        neb = np.zeros(m.shape, dtype=bool)
        for axis in (0, 1):
            a = m
            b = np.roll(m, -1, axis=axis)
            diff = (a != b) & (a != self.ignore_index) & (b != self.ignore_index)
            if axis == 0:
                diff[-1, :] = False
            else:
                diff[:, -1] = False
            neb |= diff
            neb |= np.roll(diff, 1, axis=axis)
        if self.width > 1:
            k = np.ones((2 * self.width - 1, 2 * self.width - 1), np.uint8)
            neb = cv2.dilate(neb.astype(np.uint8), k) > 0
        return neb.astype(np.uint8)


class Augmentor:
    """Augmentations: flips, rotation, elastic deformation, noise,
    brightness/contrast and spectral band shuffling for multispectral inputs."""

    def __init__(self, flip_p: float = 0.5, rotate_deg: float = 45.0, elastic_alpha: float = 120.0,
                 elastic_sigma: float = 6.0, elastic_p: float = 0.3, noise_sigma: float = 0.02,
                 brightness: float = 0.2, contrast: float = 0.2, band_shuffle_p: float = 0.0,
                 ignore_index: int = 255, seed: Optional[int] = None):
        self.flip_p, self.rotate_deg = flip_p, rotate_deg
        self.elastic_alpha, self.elastic_sigma, self.elastic_p = elastic_alpha, elastic_sigma, elastic_p
        self.noise_sigma, self.brightness, self.contrast = noise_sigma, brightness, contrast
        self.band_shuffle_p, self.ignore_index = band_shuffle_p, ignore_index
        self.rng = np.random.default_rng(seed)

    def _warp(self, img, mask, matrix=None, maps=None):
        h, w = mask.shape
        if matrix is not None:
            img = cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            mask = cv2.warpAffine(mask, matrix, (w, h), flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=self.ignore_index)
        else:
            img = cv2.remap(img, *maps, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            mask = cv2.remap(mask, *maps, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                             borderValue=self.ignore_index)
        if img.ndim == 2:
            img = img[..., None]
        return img, mask

    def __call__(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        r = self.rng
        if r.random() < self.flip_p:
            img, mask = img[:, ::-1], mask[:, ::-1]
        if r.random() < self.flip_p:
            img, mask = img[::-1], mask[::-1]
        img, mask = np.ascontiguousarray(img), np.ascontiguousarray(mask)
        h, w = mask.shape
        if self.rotate_deg > 0:
            angle = r.uniform(-self.rotate_deg, self.rotate_deg)
            img, mask = self._warp(img, mask, matrix=cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0))
        if self.elastic_alpha > 0 and r.random() < self.elastic_p:
            dx = cv2.GaussianBlur(r.uniform(-1, 1, (h, w)).astype(np.float32), (0, 0), self.elastic_sigma)
            dy = cv2.GaussianBlur(r.uniform(-1, 1, (h, w)).astype(np.float32), (0, 0), self.elastic_sigma)
            gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
            img, mask = self._warp(img, mask, maps=(gx + self.elastic_alpha * dx, gy + self.elastic_alpha * dy))
        if self.brightness or self.contrast:
            c = 1.0 + r.uniform(-self.contrast, self.contrast)
            b = r.uniform(-self.brightness, self.brightness)
            img = (img - img.mean()) * c + img.mean() + b
        if self.noise_sigma > 0:
            img = img + r.normal(0, self.noise_sigma, img.shape).astype(np.float32)
        if img.shape[-1] > 3 and r.random() < self.band_shuffle_p:
            img = img[..., r.permutation(img.shape[-1])]
        return np.clip(img, 0, 1).astype(np.float32), mask


class LandCoverDataset(Dataset):
    """Image/label patches with on-the-fly NEB targets.
    """

    def __init__(self, root: str, split_file: Optional[str] = None, augmentor: Optional[Augmentor] = None,
                 mean: Optional[Sequence[float]] = None, std: Optional[Sequence[float]] = None,
                 class_map: Optional[Dict[int, int]] = None, ignore_index: int = 255, neb_width: int = 1):
        self.root = Path(root)
        self.augmentor = augmentor
        self.mean = None if mean is None else np.asarray(mean, np.float32)
        self.std = None if std is None else np.asarray(std, np.float32)
        self.ignore_index = ignore_index
        self.neb = NEBTargetGenerator(neb_width, ignore_index)
        self.lut = self._build_lut(class_map, ignore_index) if class_map else None
        self.samples = self._index(split_file)
        if not self.samples:
            raise RuntimeError(f"no samples found under {self.root}")

    @staticmethod
    def _build_lut(class_map: Dict[int, int], ignore_index: int) -> np.ndarray:
        lut = np.full(65536, ignore_index, dtype=np.int64)
        for src, dst in class_map.items():
            lut[int(src)] = int(dst)
        return lut

    def _index(self, split_file: Optional[str]) -> List[Tuple[str, str]]:
        img_dir, lab_dir = self.root / "images", self.root / "labels"
        labels = {p.stem: str(p) for p in lab_dir.iterdir() if p.suffix.lower() in IMG_EXT}
        images = {p.stem: str(p) for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT}
        stems = sorted(set(images) & set(labels))
        if split_file:
            with open(split_file) as handle:
                wanted = {line.strip() for line in handle if line.strip()}
            stems = [s for s in stems if s in wanted]
        return [(images[s], labels[s]) for s in stems]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path, lab_path = self.samples[idx]
        img, mask = read_image(img_path), read_mask(lab_path).astype(np.int64)
        if self.lut is not None:
            mask = self.lut[mask]
        mask = mask.astype(np.uint8 if mask.max() < 256 else np.int32)
        if self.augmentor is not None:
            img, mask = self.augmentor(img, mask)
        neb = self.neb(mask)
        if self.mean is not None:
            img = (img - self.mean) / self.std
        return {
            "image": torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float(),
            "mask": torch.from_numpy(mask.astype(np.int64)),
            "neb": torch.from_numpy(neb.astype(np.float32)),
            "name": os.path.basename(img_path),
        }


def build_dataloader(cfg, split: str, train: bool) -> DataLoader:
    """Create a loader for ``split`` ('train', 'val' or 'test') from the ``data`` config section."""
    d = cfg.data
    aug = Augmentor(**dict(d.get("augmentation", {})), ignore_index=d.get("ignore_index", 255)) if train else None
    split_file = None
    if d.get("split_dir"):
        split_file = os.path.join(d.split_dir, f"fold{d.get('fold', 0)}_{split}.txt")
    dataset = LandCoverDataset(d.root, split_file, aug, d.get("mean"), d.get("std"),
                               d.get("class_map"), d.get("ignore_index", 255), d.get("neb_width", 1))
    return DataLoader(dataset, batch_size=cfg.train.batch_size if train else cfg.eval.get("batch_size", 1),
                      shuffle=train, num_workers=d.get("num_workers", 4), pin_memory=torch.cuda.is_available(),
                      drop_last=train)
