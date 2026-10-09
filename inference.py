"""Large-scene inference with overlapping tiles (regional mapping).ne
"""
import argparse
import os

import cv2
import numpy as np
import torch

from top0covernet import build_model
from top0covernet.data.dataset import read_image
from top0covernet.engine import SlidingWindowPredictor
from top0covernet.utils import Config


def write_raster(path: str, array: np.ndarray, reference: str) -> None:
    try:
        import rasterio
        with rasterio.open(reference) as src:
            profile = src.profile
        profile.update(count=1, dtype=array.dtype.name, compress="lzw")
        with rasterio.open(path + ".tif", "w", **profile) as dst:
            dst.write(array, 1)
    except Exception:  # rasterio missing or non-georeferenced input: plain image fallback
        cv2.imwrite(path + ".png", array if array.dtype == np.uint8 else (array * 255).astype(np.uint8))


def main():
    parser = argparse.ArgumentParser(description="Top0CoverNet+ sliding-window inference")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True, help="output path prefix")
    parser.add_argument("--tile", type=int, default=512)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--opts", nargs="*", default=[], help="config overrides, e.g. model.num_classes=5")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config).merge_overrides(args.opts)
    model = build_model(cfg.model)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state.get("model", state))
    predictor = SlidingWindowPredictor(model, cfg.model.num_classes, args.tile, args.overlap, args.batch_size,
                                       args.device, cfg.data.get("mean"), cfg.data.get("std"))
    result = predictor(read_image(args.input))
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    write_raster(args.output + "_label", result["label"], args.input)
    write_raster(args.output + "_neb", result["neb"].astype(np.float32), args.input)
    print(f"saved {args.output}_label and {args.output}_neb")


if __name__ == "__main__":
    main()
