"""Parameters, FLOPs and latency following the benchmarking protocol.
"""
import argparse
import time

import torch

from top0covernet import build_model
from top0covernet.utils import Config


def main():
    parser = argparse.ArgumentParser(description="Top0CoverNet+ cost benchmark")
    parser.add_argument("--config", required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--opts", nargs="*", default=[])
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config).merge_overrides(args.opts)
    model = build_model(cfg.model).to(args.device).eval()
    x = torch.randn(1, cfg.model.get("in_channels", 3), args.size, args.size, device=args.device)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Params: {params:.2f} M")
    try:
        from thop import profile
        macs, _ = profile(model, inputs=(x,), verbose=False)
        # thop counts multiply-accumulates; the paper reports the thop value as FLOPs
        print(f"FLOPs (thop): {macs / 1e9:.2f} G")
    except ImportError:
        print("thop not installed; skipping FLOPs")
    sync = torch.cuda.synchronize if x.is_cuda else (lambda: None)
    with torch.no_grad():
        for _ in range(args.warmup):
            model(x)
        sync()
        t0 = time.perf_counter()
        for _ in range(args.runs):
            model(x)
        sync()
    print(f"Inference: {(time.perf_counter() - t0) / args.runs * 1000:.1f} ms / image")


if __name__ == "__main__":
    main()
