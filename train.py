"""Train Top0CoverNet+.

Example:
    python train.py --config configs/top0covernet_skwanda.yaml
    python train.py --config configs/top0covernet_skwanda.yaml --opts model.use_gtrm=false loss.lambda_topo=0
"""
import argparse
import os

import torch

from top0covernet import build_loss, build_model
from top0covernet.data import build_dataloader
from top0covernet.engine import Evaluator, Trainer
from top0covernet.utils import Config, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train Top0CoverNet+")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None, help="checkpoint to resume from")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--opts", nargs="*", default=[], help="overrides, e.g. train.epochs=50")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.from_yaml(args.config).merge_overrides(args.opts)
    set_seed(cfg.experiment.seed)
    work_dir = cfg.experiment.work_dir
    os.makedirs(work_dir, exist_ok=True)
    cfg.to_yaml(os.path.join(work_dir, "config.yaml"))

    model = build_model(cfg.model)
    criterion = build_loss(cfg.loss, cfg.model.num_classes, cfg.data.ignore_index)
    evaluator = Evaluator(cfg.model.num_classes, cfg.data.ignore_index, cfg.eval.structural,
                          cfg.eval.bf1_tolerance, cfg.eval.min_region, args.device)
    trainer = Trainer(model, criterion, build_dataloader(cfg, "train", train=True),
                      build_dataloader(cfg, "val", train=False), cfg.train, evaluator, work_dir, args.device)
    if args.resume:
        trainer.resume(args.resume)
    best = trainer.fit()
    trainer.logger.info(f"best validation mIoU: {best:.2f}")


if __name__ == "__main__":
    main()
