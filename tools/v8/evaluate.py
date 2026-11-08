"""V8 evaluation on independent validation 200 and/or frozen Test600+Same150.

Runs Euler/Heun at NFE <=16 and writes Evaluator V2-style metrics.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.v8.common import (  # noqa: E402
    DOCS_V8, STYLES, V8_ROOT, aggregate_rows, batch_tensors, build_independent_val_tasks,
    build_source_latents, compute_batch_rows, ensure_dirs, load_eval_bundle, load_frozen_tasks,
    save_json, source_cache_for_frozen,
)
from tools.v8.models import VelocityUNet  # noqa: E402
from tools.v8.solvers import solve  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to V8 checkpoint, or NOOP for zero-velocity VAE no-op baseline")
    ap.add_argument("--name", required=True)
    ap.add_argument("--tasks", choices=["val200", "test750"], default="val200")
    ap.add_argument("--solvers", default="euler4,euler8,euler16,heun4,heun8")
    ap.add_argument("--out-suffix", default="", help="Optional suffix for output JSON to isolate solver shards")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== V8 Eval {args.name} ckpt={args.ckpt} tasks={args.tasks} Device={device} ===", flush=True)

    tasks = build_independent_val_tasks(10) if args.tasks == "val200" else load_frozen_tasks()
    models, ref_feats, vae, trt = load_eval_bundle(device)
    src_latents = build_source_latents(tasks, vae, device)
    source_cache = source_cache_for_frozen() if args.tasks == "test750" else {}

    if args.ckpt == "NOOP":
        class ZeroVelocity(torch.nn.Module):
            def forward(self, z, tau, s, t):
                return torch.zeros_like(z)
        model = ZeroVelocity().to(device).eval()
        ck = {"model": {}, "ema": {}, "config": {"noop": True}}
        print("[V8 Eval] Using deterministic zero-velocity VAE No-op baseline", flush=True)
    else:
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        model = VelocityUNet(base_dim=48, num_styles=len(STYLES)).to(device).eval()
        model.load_state_dict(ck["ema"] if args.name.endswith("_ema") else ck["model"])

    results = {"protocol": {"tasks": args.tasks, "n_tasks": len(tasks), "rgb_reference": "paper_rgb_v1",
                            "checkpoint": str(args.ckpt), "score_free": True}, "methods": {}, "rows": {}}
    solver_specs = []
    for spec in args.solvers.split(","):
        if spec.startswith("euler"):
            solver_specs.append((spec, "euler", int(spec[5:])))
        elif spec.startswith("heun"):
            solver_specs.append((spec, "heun", int(spec[4:])))
        else:
            raise ValueError(spec)

    for spec, method, steps in solver_specs:
        rows = []
        t0 = time.time()
        for i in range(0, len(tasks), args.batch):
            chunk = tasks[i : i + args.batch]
            with torch.inference_mode():
                z0, s, t = batch_tensors(chunk, src_latents, device)
                z_gen, nfe = solve(model, z0, s, t, method=method, n_steps=steps)
                if nfe > 16:
                    raise ValueError(f"NFE {nfe} > 16")
                l2 = (z_gen.float() - z0.float()).flatten(1).norm(p=2, dim=1).cpu().tolist()
                rows.extend(compute_batch_rows(z_gen, chunk, models, ref_feats, vae, trt, source_cache, latent_l2=l2, nfe=nfe))
            if (i // args.batch) % 10 == 0:
                torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        wall = round(time.time() - t0, 1)
        cross = aggregate_rows(rows, only_cross=True)
        same = aggregate_rows([r for r in rows if not r["is_cross"]]) if args.tasks == "test750" else {}
        allv = aggregate_rows(rows)
        for g in (cross, same, allv):
            if g:
                g["NFE"] = nfe
                g["gen_wall_sec"] = wall
        results["methods"][spec] = {"cross": cross, "same": same, "all": allv}
        results["rows"][spec] = rows
        print(f"  [{spec:8s} NFE={nfe:2d}] dT={cross['mean_dT']:+.4f} Top-1={cross['Target_Top1']:4.1f}% "
              f"Margin={cross['Target_Margin']:+.4f} DINO-C={cross['DINO_C']:.4f} LPIPS={cross['LPIPS']:.4f} "
              f"ukiyo={cross['ukiyo_top1']:4.1f}% styles+={cross['styles_improved']}/5 gen={wall}s", flush=True)

    out_json = V8_ROOT / f"{args.name}_{args.tasks}_eval{args.out_suffix}.json"
    save_json(results, out_json)
    print(f"[V8 Eval] Saved {out_json}", flush=True)


if __name__ == "__main__":
    main()
