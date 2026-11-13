"""Profile the mini-batch Sinkhorn coupling that dominates V9 training time.

Reports, for several batch sizes and iteration counts:
  - cost-matrix construction (cdist on 32x32 RGB thumbnails)
  - Sinkhorn dual iterations (fixed iters vs early-stop tolerance)
  - row-conditional sampling (multinomial)
  - GPU busy time vs CPU wall time, to separate FLOPs from launch/Python overhead
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from tools.sbflow.ot import choose_targets, image_cost, sinkhorn


def timeit(fn, reps: int, warm: int = 3) -> tuple[float, float]:
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) / reps
    # pure GPU busy time
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps):
        fn()
    e.record()
    torch.cuda.synchronize()
    gpu = s.elapsed_time(e) / reps / 1000.0
    return wall, gpu


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", default="16,64,256")
    ap.add_argument("--iters", default="50,100,200")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--reg", type=float, default=0.05)
    ap.add_argument("--out", default="runs/blog_sbflow/bench/ot_profile.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} reg={args.reg}", flush=True)
    rows = []
    for b in [int(x) for x in args.batches.split(",")]:
        t0 = torch.rand(b, 3, 32, 32, device=device)
        t1 = torch.rand(b, 3, 32, 32, device=device)
        w_cost, g_cost = timeit(lambda: image_cost(t0, t1, "rgb"), args.reps)
        cost = image_cost(t0, t1, "rgb")
        rec = {
            "batch": b,
            "cost_matrix_ms": round(1000 * w_cost, 3),
            "cost_matrix_gpu_ms": round(1000 * g_cost, 3),
            "cost_elems": b * b,
        }
        print(f"\n--- batch={b}  cost-matrix {1000*w_cost:.3f} ms "
              f"(gpu {1000*g_cost:.3f} ms, {b*b} entries) ---", flush=True)

        for it in [int(x) for x in args.iters.split(",")]:
            w_fix, g_fix = timeit(lambda: sinkhorn(cost, args.reg, it), args.reps)
            w_tol, g_tol = timeit(lambda: sinkhorn(cost, args.reg, it, tol=1e-6, check_every=10), args.reps)
            _, info_tol = sinkhorn(cost, args.reg, it, tol=1e-6, check_every=10)
            rec[f"iters_{it}"] = {
                "fixed_ms": round(1000 * w_fix, 3),
                "fixed_gpu_ms": round(1000 * g_fix, 3),
                "fixed_ms_per_iter": round(1000 * w_fix / it, 4),
                "earlystop_ms": round(1000 * w_tol, 3),
                "earlystop_gpu_ms": round(1000 * g_tol, 3),
                "earlystop_iters_used": info_tol["iters_used"],
                "row_err": info_tol["row_err"],
            }
            print(f"  iters={it:3d}: fixed {1000*w_fix:7.2f} ms (gpu {1000*g_fix:6.2f}) "
                  f"| early-stop {1000*w_tol:7.2f} ms used {info_tol['iters_used']:3d} "
                  f"row_err={info_tol['row_err']:.1e}", flush=True)

        w_mn, g_mn = timeit(lambda: choose_targets(cost, args.reg)[0], args.reps)
        rec["choose_targets_ms"] = round(1000 * w_mn, 3)
        rec["choose_targets_gpu_ms"] = round(1000 * g_mn, 3)
        print(f"  choose_targets: {1000*w_mn:.2f} ms (gpu {1000*g_mn:.2f})", flush=True)
        rows.append(rec)

    # Python-overhead share at the default setting used in training.
    summary = []
    for r in rows:
        k = r.get("iters_200")
        if not k:
            continue
        summary.append({
            "batch": r["batch"],
            "total_ms": round(r["cost_matrix_ms"] + k["fixed_ms"] + r["choose_targets_ms"], 2),
            "gpu_ms": round(r["cost_matrix_gpu_ms"] + k["fixed_gpu_ms"] + r["choose_targets_gpu_ms"], 2),
        })
        summary[-1]["overhead_ms"] = round(summary[-1]["total_ms"] - summary[-1]["gpu_ms"], 2)
        summary[-1]["overhead_share"] = round(
            summary[-1]["overhead_ms"] / max(summary[-1]["total_ms"], 1e-9), 4)

    out = {"reg": args.reg, "device": device, "rows": rows, "summary_200iters": summary}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\n=== summary at 200 iters ===")
    print(json.dumps(summary, indent=2))
    print("saved", args.out)


if __name__ == "__main__":
    main()
