"""Evaluate a checkpoint on the test split (semantic + structural metrics).

Cross-dataset transfer: point --config at the *target* dataset config and pass the
source checkpoint; use ``data.class_map`` for label harmonisation.

Example:
    python test.py --config configs/top0covernet_skwanda.yaml --checkpoint work_dirs/top0covernet_skwanda/best.pth
"""
import argparse
import json

import torch

from top0covernet import build_model
from top0covernet.data import build_dataloader
from top0covernet.engine import Evaluator
from top0covernet.utils import Config


def main():
    parser = argparse.ArgumentParser(description="Evaluate Top0CoverNet+")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default=None, help="optional JSON file for the metrics")
    parser.add_argument("--opts", nargs="*", default=[])
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config).merge_overrides(args.opts)
    model = build_model(cfg.model)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state.get("model", state))
    model.to(args.device)
    evaluator = Evaluator(cfg.model.num_classes, cfg.data.ignore_index, cfg.eval.structural,
                          cfg.eval.bf1_tolerance, cfg.eval.min_region, args.device)
    results = evaluator.run(model, build_dataloader(cfg, args.split, train=False))
    for key, value in results.items():
        if key != "IoU_per_class":
            print(f"{key:>6s}: {value:.4f}")
    print("IoU per class:", ", ".join(f"{v:.2f}" for v in results["IoU_per_class"]))
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()
