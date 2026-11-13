"""V11 P3: unconditional generation of the five styles from the shared pivot model.

The pivot formulation trains one conditional flow F_s: N(0,I) -> p_s per style, so
sampling a style is just `solve_transfer(z_gauss, pivot -> s)`. No source image is
involved, which is exactly what separates the two questions V10 conflated:

    (1) can the shared model generate genuine Rococo / Ukiyo-e / Early Renaissance?
    (2) given that it can, can it preserve the structure of an input image?

If (1) already fails, cross-domain transport cannot succeed and no amount of bridge
tuning will help. If (1) succeeds and transport still fails, the problem is the content
constraint (theory section 2 / 4).

Usage
    python -m sbflow.gen_uncond --ckpt runs/blog_sbflow/v11/p3/model_step4000.pt \
        --out runs/blog_sbflow/v11/p3_gen --per-style 100
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image

from .data5 import STYLE_NAMES
from .gen5 import png_bytes, to_u8
from .infer5 import load_model5, solve_transfer
from .vae import decode_latents, load_sdvae


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-style", type=int, default=100)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--method", choices=["euler", "heun"], default="heun")
    ap.add_argument("--t-floor", type=float, default=0.002)
    ap.add_argument("--weights", choices=["raw", "ema"], default="ema")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    out = Path(args.out)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck, model, stats = load_model5(args.ckpt, device, args.weights)
    cfg = ck.get("config", {})
    if not cfg.get("pivot", False):
        raise SystemExit("gen_uncond needs a checkpoint trained with --pivot; the Gaussian "
                         "prior is what makes unconditional sampling possible")
    n_real = int(stats.style_mean.shape[0] - 1)
    pivot_id = n_real
    styles = STYLE_NAMES[:n_real]
    eps = float(cfg["eps"])
    objective = ck.get("objective", "bidir")
    sf = float(ck["ae_scaling_factor"])
    gmean, gstd = stats.global_mean, stats.global_std

    vae, trt = load_sdvae(device)
    mk = ck.get("model_kwargs", {})
    channels = int(mk.get("channels", 4))
    img = int(mk.get("image_size", 64))
    latent_shape = (channels, img, img)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    print(f"[gen-uncond] ckpt={args.ckpt} styles={styles} pivot_id={pivot_id} "
          f"latent={latent_shape} per_style={args.per_style} steps={args.steps} "
          f"{args.method}", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    pool = ThreadPoolExecutor(max_workers=args.workers)
    manifest: List[dict] = []
    t0 = time.perf_counter()

    with torch.inference_mode():
        for j, name in enumerate(styles):
            sdir = out / "generated" / name
            sdir.mkdir(parents=True, exist_ok=True)
            made = 0
            while made < args.per_style:
                k = min(args.batch, args.per_style - made)
                z = torch.randn(k, *latent_shape, device=device, generator=gen,
                                dtype=torch.float32)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                    zout = solve_transfer(model, z, pivot_id, j, stats, eps, args.steps,
                                          args.method, args.t_floor, objective)
                dec = decode_latents(vae, trt,
                                     (zout * gstd[None, :, None, None]
                                      + gmean[None, :, None, None]) / sf,
                                     batch=k)
                u8 = to_u8(dec)
                for i in range(k):
                    rel = f"generated/{name}/{made + i:04d}.png"
                    pool.submit((sdir / f"{made + i:04d}.png").write_bytes, png_bytes(u8[i]))
                    manifest.append({"style": name, "index": made + i, "image": rel,
                                     "latent_l2": float(zout[i].flatten().norm())})
                made += k
                if made % (args.batch * 2) == 0:
                    print(f"[gen-uncond] {name} {made}/{args.per_style}", flush=True)
    pool.shutdown(wait=True)

    wall = time.perf_counter() - t0
    info = {"checkpoint": str(args.ckpt), "pivot": True, "n_styles": n_real,
            "per_style": args.per_style, "steps": args.steps, "solver": args.method,
            "weights": args.weights, "seed": args.seed,
            "total_sec": round(wall, 2),
            "images_per_sec": round(len(manifest) / max(wall, 1e-9), 2)}
    (out / "manifest.json").write_text(json.dumps({"info": info, "tasks": manifest},
                                                  ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    print(f"[gen-uncond] done {len(manifest)} images in {wall:.1f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
