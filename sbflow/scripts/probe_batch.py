"""Probe peak VRAM and seconds/update vs microbatch for a given DiT size.

Used to calibrate the batch so training actually uses the available GPU memory.
Only forward/backward/optimizer are measured (no Sinkhorn), which matches the
`--ot-ratio 0` training configuration.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from sbflow.bridge import bridge_sample
from sbflow.network import BidirDiT
from sbflow.preconditioning import MomentAccumulator, normalized_target, precondition_input


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--patch", type=int, default=2)
    ap.add_argument("--batches", default="32,64,128,256,512")
    ap.add_argument("--reps", type=int, default=12)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--eps", type=float, default=0.04)
    ap.add_argument("--min-t", type=float, default=0.002)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--tag", default="probe")
    ap.add_argument("--out", default="runs/blog_sbflow/bench")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    c0 = torch.load("runs/blog_sbflow/cache/Impressionism.pt", map_location="cpu", weights_only=False)
    c1 = torch.load("runs/blog_sbflow/cache/Ukiyo_e.pt", map_location="cpu", weights_only=False)
    r0, r1 = c0["latent"].float(), c1["latent"].float()
    both = torch.cat([r0, r1], 0)
    m = both.mean((0, 2, 3))
    s = both.var((0, 2, 3), unbiased=False).clamp_min(1e-8).sqrt()
    A = ((r0 - m[None, :, None, None]) / s[None, :, None, None]).to(device)
    B = ((r1 - m[None, :, None, None]) / s[None, :, None, None]).to(device)
    ch, h, _ = r0.shape[1:]

    gen = torch.Generator(device=device).manual_seed(7)
    acc = MomentAccumulator()
    for _ in range(16):
        ia = torch.randint(len(A), (16,), device=device, generator=gen)
        ib = torch.randint(len(B), (16,), device=device, generator=gen)
        acc.update(A[ia], B[ib])
    mom = acc.finalize()

    mk = {"channels": ch, "image_size": h, "patch": args.patch,
          "width": args.width, "depth": args.depth, "heads": args.heads}
    model = BidirDiT(**mk).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    npars = sum(p.numel() for p in model.parameters())

    def step(mb: int) -> None:
        model.train()
        opt.zero_grad(set_to_none=True)
        ia = torch.randint(len(A), (mb,), device=device, generator=gen)
        ib = torch.randint(len(B), (mb,), device=device, generator=gen)
        z0, z1 = A[ia], B[ib]
        tt = torch.rand(mb, device=device, generator=gen) * (1 - 2 * args.min_t) + args.min_t
        xi = torch.randn(z0.shape, device=device, generator=gen, dtype=z0.dtype)
        xt, yp, ym = bridge_sample(z0, z1, tt, args.eps, xi)
        half = mb // 2
        xf, cf = precondition_input(xt[:half], tt[:half], args.eps, mom)
        xb, cb = precondition_input(xt[half:], 1 - tt[half:], args.eps, mom.reversed())
        yt = torch.cat([normalized_target(xt[:half], yp[:half], cf),
                        normalized_target(xt[half:], ym[half:], cb)], 0)
        x = torch.cat([xf, xb], 0)
        tau = torch.cat([tt[:half], 1 - tt[half:]], 0)
        d = torch.cat([torch.ones(half, device=device, dtype=torch.long),
                       torch.zeros(mb - half, device=device, dtype=torch.long)])
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            loss = F.mse_loss(model(x, tau, d).float(), yt.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    rows = []
    for mb in [int(x) for x in args.batches.split(",") if x.strip()]:
        mb = mb + (mb % 2)
        try:
            for _ in range(args.warmup):
                step(mb)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            for _ in range(args.reps):
                step(mb)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / args.reps
            peak = torch.cuda.max_memory_allocated(0) / 1048576
            rows.append({"microbatch": mb, "sec_per_microbatch": round(dt, 4),
                         "ms_per_sample": round(1000 * dt / mb, 3),
                         "peak_vram_mib": round(peak, 1)})
            print(f"  mb={mb:4d}  {dt*1000:7.1f} ms/microbatch  {1000*dt/mb:6.2f} ms/sample  "
                  f"peak={peak:8.1f} MiB", flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f"  mb={mb:4d}  OOM", flush=True)
            rows.append({"microbatch": mb, "oom": True})
            torch.cuda.empty_cache()
            break

    out = {"tag": args.tag, "model": mk, "n_params": npars, "rows": rows}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    p = Path(args.out) / f"{args.tag}.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("saved", p)


if __name__ == "__main__":
    main()
