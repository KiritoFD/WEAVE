"""V11-B P0: environment/VRAM/data acceptance.

Checks: DWT roundtrip on real paintings, per-style band sigmas,
train-step VRAM at microbatch 1/2/4, s/step and s/img estimates.
Writes docs/v11b/P0_PROFILE.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image

from .bands import dwt2, idwt2
from .dataset_real import RealWaveletDataset, collate_batch, make_batch
from .model import WaveletFlow
from .train_flow import estimate_sigmas, fm_step_full

REPO = Path(__file__).resolve().parents[2]
DOCS = REPO / "docs" / "v11b"


def check_dwt(n=8):
    ds = RealWaveletDataset("train", crop=512)
    errs = []
    for i in range(n):
        crop, _ = ds[i]
        x = crop.unsqueeze(0)
        d = dwt2(x)
        y = idwt2(d["LL2"], d["H2"], d["H1"])
        errs.append((x - y).abs().max().item())
    return {"max_err": max(errs), "mean_err": sum(errs) / len(errs)}


def bench_step(model, sigmas, device, micro, steps=5):
    """Time fm_step_full + backward at given microbatch."""
    ds = RealWaveletDataset("train", crop=128)
    items = [ds[i] for i in range(micro * 4)]
    torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
    torch.cuda.synchronize() if device == "cuda" else None
    t0 = time.time()
    losses = []
    for s in range(steps):
        batch = items[s * micro:(s + 1) * micro]
        crops = torch.stack([b[0] for b in batch])
        sids = torch.tensor([b[1] for b in batch])
        bb = make_batch(crops, sids, device)
        loss, _ = fm_step_full(model, bb, sigmas, device)
        loss.backward()
        model.zero_grad(set_to_none=True)
    if device == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t0) / steps
    peak = (torch.cuda.max_memory_allocated() / 2**30
            if device == "cuda" else 0.0)
    return {"micro": micro, "s_per_step": round(dt, 3),
            "peak_GB": round(peak, 2),
            "imgs_per_s": round(micro / dt, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=40)
    ap.add_argument("--skip-sigma", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    DOCS.mkdir(parents=True, exist_ok=True)
    out = {"device": device}

    print("[p0] DWT roundtrip on real 512px images...", flush=True)
    out["dwt_roundtrip"] = check_dwt()
    print(" ", out["dwt_roundtrip"], flush=True)

    sigmas = None
    if not args.skip_sigma:
        print("[p0] estimating per-style band sigmas...", flush=True)
        t0 = time.time()
        sigmas = estimate_sigmas(device=device)
        out["sigmas_s"] = round(time.time() - t0, 1)
        out["sigmas"] = sigmas
        print(" ", json.dumps(sigmas)[:300], flush=True)

    print("[p0] building model + VRAM sweep...", flush=True)
    model = WaveletFlow(width=args.width).to(device)
    out["params_M"] = round(model.n_params() / 1e6, 3)
    out["bench"] = []
    if sigmas is None:
        sigmas = estimate_sigmas(n_images=64, device=device)
    for micro in (1, 2, 4, 8):
        try:
            r = bench_step(model, sigmas, device, micro, steps=3)
            out["bench"].append(r)
            print(f"  micro={micro}: {r['s_per_step']}s/step "
                  f"peak={r['peak_GB']}GB {r['imgs_per_s']}img/s", flush=True)
            if r["peak_GB"] > 5.0:
                break
        except torch.cuda.OutOfMemoryError:
            print(f"  micro={micro}: OOM", flush=True)
            torch.cuda.empty_cache()
            out["bench"].append({"micro": micro, "oom": True})
            break

    # est. 512px full-image inference cost: bands at 128/256 res
    (DOCS / "P0_PROFILE.json").write_text(
        json.dumps(out, indent=1), encoding="utf-8")
    print(f"[p0] wrote {DOCS/'P0_PROFILE.json'}", flush=True)


if __name__ == "__main__":
    main()
