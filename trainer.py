"""Training loop: AdamW, step-wise exponential decay, early stopping on val mIoU."""
from __future__ import annotations

import json
import os
import time
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from ..utils import AverageMeter, get_logger
from .evaluator import Evaluator


class Trainer:
    """Owns the optimisation loop, checkpointing and early stopping.

    Args:
        model: network returning the dict documented in :class:`Top0CoverNetPlus`.
        criterion: :class:`GAOOLoss` (returns a dict of terms with key ``total``).
        train_loader / val_loader: data loaders yielding ``image``, ``mask``, ``neb``.
        cfg: the ``train`` config section.
        evaluator: validation evaluator.
        work_dir: where checkpoints and logs are written.
    """

    def __init__(self, model, criterion, train_loader: DataLoader, val_loader: DataLoader, cfg,
                 evaluator: Evaluator, work_dir: str, device: str | torch.device = "cuda"):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)
        self.train_loader, self.val_loader = train_loader, val_loader
        self.cfg = cfg
        self.evaluator = evaluator
        self.work_dir = work_dir
        os.makedirs(work_dir, exist_ok=True)
        self.logger = get_logger(log_file=os.path.join(work_dir, "train.log"))
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=tuple(cfg.get("betas", (0.9, 0.999))),
                                           eps=cfg.get("eps", 1e-8), weight_decay=cfg.get("weight_decay", 0.01))
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=cfg.get("lr_step", 10),
                                                         gamma=cfg.get("lr_gamma", 0.95))
        self.min_lr = cfg.get("min_lr", 1e-5)
        self.use_amp = bool(cfg.get("amp", False)) and self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.start_epoch, self.best, self.bad_epochs = 0, -1.0, 0
        self.history = []

    # ------------------------------------------------------------------ checkpoints
    def save(self, name: str, epoch: int) -> None:
        torch.save({"epoch": epoch, "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(), "best": self.best, "bad_epochs": self.bad_epochs},
                   os.path.join(self.work_dir, name))

    def resume(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.start_epoch = state["epoch"] + 1
        self.best, self.bad_epochs = state.get("best", -1.0), state.get("bad_epochs", 0)
        self.logger.info(f"resumed from {path} at epoch {self.start_epoch}")

    # ------------------------------------------------------------------ loops
    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        meters: Dict[str, AverageMeter] = {}
        max_iters: Optional[int] = self.cfg.get("max_iters_per_epoch")
        for it, batch in enumerate(self.train_loader):
            if max_iters and it >= max_iters:
                break
            image = batch["image"].to(self.device, non_blocking=True)
            mask = batch["mask"].to(self.device, non_blocking=True)
            neb = batch["neb"].to(self.device, non_blocking=True)
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(image)
            outputs = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in outputs.items()}
            terms = self.criterion(outputs, mask, neb)
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(terms["total"]).backward()
            if self.cfg.get("grad_clip"):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            for k, v in terms.items():
                meters.setdefault(k, AverageMeter()).update(v.item(), image.shape[0])
            if it % self.cfg.get("log_interval", 20) == 0:
                self.logger.info(f"epoch {epoch} it {it} " + " ".join(f"{k}={m.avg:.4f}" for k, m in meters.items()))
        return {k: m.avg for k, m in meters.items()}

    def _step_lr(self) -> None:
        self.scheduler.step()
        for group in self.optimizer.param_groups:
            group["lr"] = max(group["lr"], self.min_lr)

    def fit(self) -> float:
        epochs, patience = self.cfg.epochs, self.cfg.get("patience", 20)
        for epoch in range(self.start_epoch, epochs):
            t0 = time.time()
            train_stats = self.train_one_epoch(epoch)
            val = self.evaluator.run(self.model, self.val_loader)
            self._step_lr()
            record = {"epoch": epoch, "lr": self.optimizer.param_groups[0]["lr"], "time_s": time.time() - t0,
                      **{f"train_{k}": v for k, v in train_stats.items()},
                      **{f"val_{k}": v for k, v in val.items() if k != "IoU_per_class"}}
            self.history.append(record)
            self.logger.info(" | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in record.items()))
            if val["mIoU"] > self.best:
                self.best, self.bad_epochs = float(val["mIoU"]), 0
                self.save("best.pth", epoch)
            else:
                self.bad_epochs += 1
            self.save("last.pth", epoch)
            with open(os.path.join(self.work_dir, "history.json"), "w") as handle:
                json.dump(self.history, handle, indent=1)
            if self.bad_epochs >= patience:
                self.logger.info(f"early stopping at epoch {epoch} (best val mIoU {self.best:.2f})")
                break
        best = torch.load(os.path.join(self.work_dir, "best.pth"), map_location=self.device)
        self.model.load_state_dict(best["model"])      # restore best weights
        return self.best
