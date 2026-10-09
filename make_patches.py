"""Tile scenes into 512x512 patches (50% overlap) and build leakage-free folds.

Example:
    python tools/make_patches.py --images data/raw/images --labels data/raw/labels \
        --out data/sKwanda_V2/patches --splits data/sKwanda_V2/splits
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from top0covernet.data import PatchTiler, SplitBuilder  # noqa: E402

EXT = (".tif", ".tiff", ".png", ".jpg")


def main():
    parser = argparse.ArgumentParser(description="Patch extraction and fold construction")
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--block", type=int, default=2048, help="geographic block size in pixels")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    labels = {p.stem: p for p in Path(args.labels).iterdir() if p.suffix.lower() in EXT}
    tiler = PatchTiler(args.size, args.overlap, args.block)
    records = []
    for img in sorted(Path(args.images).iterdir()):
        if img.suffix.lower() in EXT and img.stem in labels:
            records += tiler.tile_scene(str(img), str(labels[img.stem]), args.out)
            print(f"{img.name}: {len(records)} patches so far")
    SplitBuilder(args.folds, seed=args.seed).build(records, args.splits)
    print(f"wrote {len(records)} patches and {args.folds} folds to {args.splits}")


if __name__ == "__main__":
    main()
