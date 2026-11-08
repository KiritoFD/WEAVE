"""V8 export pure inference weights (velocity U-Net only)."""
from __future__ import annotations

import argparse
from pathlib import Path
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ema", action="store_true")
    args = ap.parse_args()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ck["ema"] if args.ema else ck["model"]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": state, "config": ck.get("config", {}), "score_free": True}, out)
    print(f"Exported V8 inference weights to {out}")


if __name__ == "__main__":
    main()
