"""Measure per-component training cost and peak VRAM for a given DiT size.

Times, over N optimizer updates at the real training objective:
  ot   : endpoint pairing (Sinkhorn + multinomial when ot_ratio>0)
  prep : Brownian bridge sample + EDM preconditioning
  fwd  : forward pass of the shared bidirectional DiT
  bwd  : backward pass
  opt  : clip_grad_norm + AdamW step + EMA update
Also reports peak VRAM and seconds per update.

Usage:
  python -m sbflow.scripts.bench_components --width 384 --depth 12 --heads 6 --microbatch 16 --accum 4
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from tools.sbflow.bridge import bridge_sample
from tools.sbflow.network import BidirDiT
from tools.sbflow.ot import paired_batch
from tools.sbflow.preconditioning import (
    MomentAccumulator, normalized_target, precondition_input,
)


def read_cache(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-cache", default="runs/blog_sbflow/cache/Impressionism.pt")
    ap.add_argument("--target-cache", default="runs/blog_sbflow/cache/Ukiyo_e.pt")
    ap.add_argument("--width", type=int, default=384)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--patch", type=int, default=2)
    ap.add_argument("--microbatch", type=int, default=16)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--updates", type=int, default=100)
    ap.add_argument("--warmup-updates", type=int, default=10)
    ap.add_argument("--eps", type=float, default=0.04)
    ap.add_argument("--ot-ratio", type=float, default=0.5)
    ap.add_argument("--ot-reg", type=float, default=0.05)
    ap.add_argument("--ot-iters", type=int, default=200)
    ap.add_argument("--ot-tol", type=float, default=0.0)
    ap.add_argument("--min-t", type=float, default=0.002)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--tag", default="bench")
    ap.add_argument("--out", default="runs/blog_sbflow/bench")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    c0, c1 = read_cache(args.source_cache), read_cache(args.target_cache)
    raw0, raw1 = c0["latent"].float(), c1["latent"].float()
    both = torch.cat([raw0, raw1], dim=0)
    m = both.mean((0, 2, 3))
    s = both.var((0, 2, 3), unbiased=False).clamp_min(1e-8).sqrt()
    A = {"z": ((raw0 - m[None, :, None, None]) / s[None, :, None, None]).to(device),
         "thumb": c0["thumb"].float().to(device) / 255.0}
    B = {"z": ((raw1 - m[None, :, None, None]) / s[None, :, None, None]).to(device),
         "thumb": c1["thumb"].float().to(device) / 255.0}
    ch, h, _ = raw0.shape[1:]

    gen = torch.Generator(device=device).manual_seed(7)
    acc = MomentAccumulator()
    for _ in range(16):
        ia = torch.randint(len(A["z"]), (args.microbatch,), device=device, generator=gen)
        ib = torch.randint(len(B["z"]), (args.microbatch,), device=device, generator=gen)
        acc.update(A["z"][ia], B["z"][ib])
    mom = acc.finalize()

    mk = {"channels": ch, "image_size": h, "patch": args.patch,
          "width": args.width, "depth": args.depth, "heads": args.heads}
    model = BidirDiT(**mk).to(device)
    ema = BidirDiT(**mk).to(device)
    ema.load_state_dict(model.state_dict())
    ema.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    npars = sum(p.numel() for p in model.parameters())

    def one_update(profile: bool) -> dict:
        t = {}
        model.train()
        opt.zero_grad(set_to_none=True)
        marks = {}

        def tick(name):
            if profile:
                torch.cuda.synchronize()
                marks[name] = time.perf_counter()

        tick("start")
        for _ in range(args.accum):
            ia = torch.randint(len(A["z"]), (args.microbatch,), device=device, generator=gen)
            ib = torch.randint(len(B["z"]), (args.microbatch,), device=device, generator=gen)
            z0, z1, _ = paired_batch(A["z"][ia], B["z"][ib], A["thumb"][ia], B["thumb"][ib],
                                     args.ot_ratio, args.ot_reg, gen, mode="rgb",
                                     ot_iters=args.ot_iters, ot_tol=args.ot_tol)
            tick("ot")
            tt = torch.rand(args.microbatch, device=device, generator=gen) * (1 - 2 * args.min_t) + args.min_t
            xi = torch.randn(z0.shape, device=device, generator=gen, dtype=z0.dtype)
            xt, yplus, yminus = bridge_sample(z0, z1, tt, args.eps, xi)
            half = args.microbatch // 2
            xf, cf = precondition_input(xt[:half], tt[:half], args.eps, mom)
            xb, cb = precondition_input(xt[half:], 1 - tt[half:], args.eps, mom.reversed())
            y_true = torch.cat([normalized_target(xt[:half], yplus[:half], cf),
                                normalized_target(xt[half:], yminus[half:], cb)], 0)
            x = torch.cat([xf, xb], 0)
            tau = torch.cat([tt[:half], 1 - tt[half:]], 0)
            d = torch.cat([torch.ones(half, device=device, dtype=torch.long),
                           torch.zeros(args.microbatch - half, device=device, dtype=torch.long)])
            tick("prep")
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                y_pred = model(x, tau, d).float()
                loss = F.mse_loss(y_pred, y_true.float())
            tick("fwd")
            (loss / args.accum).backward()
            tick("bwd")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for pe, pv in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pv, 1 - 0.999)
        tick("opt")
        if profile:
            seq = ["start", "ot", "prep", "fwd", "bwd", "opt"]
            for a, b in zip(seq[:-1], seq[1:]):
                t[b] = marks[b] - marks[a]
            t["total"] = marks["opt"] - marks["start"]
        return t

    for _ in range(args.warmup_updates):
        one_update(False)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    agg = {"ot": 0.0, "prep": 0.0, "fwd": 0.0, "bwd": 0.0, "opt": 0.0, "total": 0.0}
    t0 = time.perf_counter()
    for i in range(args.updates):
        rec = one_update(True)
        for k in agg:
            agg[k] += rec.get(k, 0.0)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated(0) / 1048576

    n = args.updates
    out = {
        "tag": args.tag,
        "model": mk,
        "n_params": npars,
        "microbatch": args.microbatch,
        "accum": args.accum,
        "effective_batch": args.microbatch * args.accum,
        "updates": n,
        "wall_sec": round(wall, 3),
        "sec_per_update": round(wall / n, 4),
        "peak_vram_mib": round(peak, 1),
        "component_ms_per_update": {k: round(1000 * v / n, 2) for k, v in agg.items()},
        "component_share": {k: round(v / max(agg["total"], 1e-9), 4) for k, v in agg.items() if k != "total"},
        "ot_ratio": args.ot_ratio,
    }
    Path(args.out).mkdir(parents=True, exist_ok=True)
    p = Path(args.out) / f"{args.tag}.json"
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print("saved", p)


if __name__ == "__main__":
    main()
