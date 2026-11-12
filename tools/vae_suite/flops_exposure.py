"""Compute (a) model FLOPs per image for the SD-VAE training step, (b) data exposure.

(a) uses torch.utils.flop_counter on CPU: conv/matmul FLOPs are shape-derived,
    so the number is identical to GPU.
(b) uses the actual triplet index to compute per-image exposure.
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/root/Workspace/WEAVE_VAE")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL

A0_DIR = "/root/Workspace/WEAVE_VAE/runs_vae/A0/final_diffusers_vae"
OUT = Path("/root/Workspace/WEAVE_VAE/runs/vae_style_rank20")
SEC_PER_UPDATE = 11.87
ACCUM = 8
BATCH = 6
UPDATES = int(11.0 * 3600 / SEC_PER_UPDATE)


def measure_flops():
    from torch.utils.flop_counter import FlopCounterMode
    vae = AutoencoderKL.from_pretrained(A0_DIR).eval()
    x = torch.randn(1, 3, 512, 512)

    # forward only (encode + decode)
    fc = FlopCounterMode(display=False)
    with torch.no_grad(), fc:
        post = vae.encode(x).latent_dist
        _ = vae.decode(post.mean).sample
    fwd = fc.get_total_flops()

    # backward: use autograd FLOPs estimation = 2x fwd for conv/matmul graphs
    return int(fwd)


def main():
    print("=== FLOPs / exposure analysis ===", flush=True)
    try:
        fwd = measure_flops()
        print(f"[FLOPs] forward (encode+decode) per image: {fwd/1e12:.4f} TFLOPs", flush=True)
    except Exception as e:
        print(f"[FLOPs] measurement failed: {e}", flush=True)
        fwd = None

    # ---- exposure ----
    tri = torch.load(OUT / "triplets.pt", weights_only=False)
    idx = torch.load(OUT / "index.pt", weights_only=False)
    pos_list = idx["pos_list"]
    N_pool = len(pos_list)

    cnt = Counter()
    for key in ("anchor", "positive", "negative"):
        for v in tri[key].tolist():
            cnt[v] += 1
    appearances = np.array([cnt.get(i, 0) for i in range(N_pool)])

    n_triplets_drawn = UPDATES * ACCUM * (BATCH // 3)   # 16 triplets per update
    repeat_per_triplet = n_triplets_drawn / max(tri["n"], 1)
    exposure = appearances * repeat_per_triplet

    used = int((appearances > 0).sum())
    print(f"\n[Exposure] pool={N_pool} images | images appearing in >=1 triplet: {used} "
          f"({used/N_pool*100:.1f}%)", flush=True)
    print(f"  triplets in index: {tri['n']} | draws: {n_triplets_drawn} | "
          f"avg draws per triplet: {repeat_per_triplet:.2f}", flush=True)
    print(f"  per-image exposures: mean={exposure.mean():.2f} median={np.median(exposure[appearances>0]):.2f} "
          f"p10={np.percentile(exposure[appearances>0],10):.2f} "
          f"p90={np.percentile(exposure[appearances>0],90):.2f} "
          f"max={exposure.max():.1f}", flush=True)
    print(f"  total image-exposures: {int(exposure.sum())}", flush=True)
    print(f"  equivalent epochs over pool: {exposure.sum()/N_pool:.2f}", flush=True)

    res = {
        "flops_fwd_per_image_TFLOPs": None if fwd is None else round(fwd / 1e12, 4),
        "sec_per_update": SEC_PER_UPDATE,
        "updates_11h": UPDATES,
        "images_per_update": BATCH * ACCUM,
        "img_per_sec": round(BATCH * ACCUM / SEC_PER_UPDATE, 3),
        "pool_images": N_pool,
        "images_in_triplets": used,
        "frac_pool_used": round(used / N_pool, 4),
        "triplet_draws": n_triplets_drawn,
        "avg_draws_per_triplet": round(repeat_per_triplet, 3),
        "exposure_mean": round(float(exposure.mean()), 3),
        "exposure_median_used": round(float(np.median(exposure[appearances > 0])), 3),
        "exposure_p10": round(float(np.percentile(exposure[appearances > 0], 10)), 3),
        "exposure_p90": round(float(np.percentile(exposure[appearances > 0], 90)), 3),
        "exposure_max": round(float(exposure.max()), 2),
        "total_image_exposures": int(exposure.sum()),
        "equivalent_epochs": round(float(exposure.sum() / N_pool), 3),
    }

    if fwd is not None:
        # training step = fwd + recompute(grad ckpt) + bwd(~2x fwd)  => ~4x fwd
        per_img_train = fwd * 4
        achieved = per_img_train * (BATCH * ACCUM) / SEC_PER_UPDATE
        peak_bf16 = 165.2e12          # RTX 4090 dense BF16 (no sparsity)
        res["flops_train_per_image_TFLOPs"] = round(per_img_train / 1e12, 4)
        res["achieved_TFLOPs"] = round(achieved / 1e12, 2)
        res["peak_TFLOPs_4090_bf16"] = round(peak_bf16 / 1e12, 1)
        res["MFU_pct"] = round(achieved / peak_bf16 * 100, 2)
        print(f"\n[MFU] train FLOPs/image ~ {per_img_train/1e12:.3f} TFLOPs "
              f"(4x fwd, grad-checkpoint recompute)", flush=True)
        print(f"  achieved = {achieved/1e12:.2f} TFLOPs | 4090 BF16 peak = {peak_bf16/1e12:.1f} TFLOPs "
              f"-> MFU = {achieved/peak_bf16*100:.2f}%", flush=True)

    (OUT / "flops_exposure.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\n[Saved] {OUT/'flops_exposure.json'}", flush=True)


if __name__ == "__main__":
    main()
