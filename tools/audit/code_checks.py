"""Code-level correctness checks required before trusting the measurements.

1. `swd.py` merges batch patches with reshape(-1, D): measure the difference
   between a merged-batch SWD and the per-image average, under mixed styles.
2. random-projection resampling: variance of the SWD estimate across projectors.
3. unequal sample counts + searchsorted quantile alignment: bias vs matched n.
4. component mixture in b1_style_operator: per-component gradients measured
   separately (see latent_losses) -- here just report the coefficient weights.

Usage: python3 tools/audit/code_checks.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import audit_core as C
import latent_losses as L


def main() -> None:
    ds = C.load_dataset()
    split = C.make_split(1000, n_ref=800, n_query=200, seed=0)
    z_q = torch.stack([ds.latents[s][split.query[s]][:32] for s in range(5)]).to("cuda")
    desc = C.feat_patch(z_q.reshape(-1, 4, 64, 64))          # [160,N,11]
    D = desc.shape[-1]
    out: dict = {}

    # 1. merged-batch vs per-image SWD (mixed-style batch 32 = 8/style)
    g = torch.Generator().manual_seed(0)
    proj = torch.randn(D, 32, generator=g)
    proj = proj / proj.norm(dim=0, keepdim=True)
    proj = proj.cuda()
    mixed = torch.stack([z_q[s][:8] for s in range(5)]).reshape(-1, 4, 64, 64)
    d_mixed = C.feat_patch(mixed)
    tgt = torch.cat([desc[s * 32:s * 32 + 64] for s in range(5)])  # 320 patches as "target set"
    merged = C.swd2_proj(d_mixed.reshape(-1, D), tgt, proj=proj)
    per_image = torch.stack([C.swd2_proj(d_mixed[i], tgt, proj=proj) for i in range(d_mixed.shape[0])]).mean()
    # grouped-by-style (correct): average over style-conditional descriptor sets
    grouped = torch.stack([
        C.swd2_proj(d_mixed[s * 8:(s + 1) * 8].reshape(-1, D), tgt[s * 64:(s + 1) * 64], proj=proj)
        for s in range(5)]).mean()
    out["swd_batch_merge"] = {
        "merged_value": float(merged), "per_image_mean": float(per_image),
        "style_grouped_mean": float(grouped),
        "merged_vs_per_image_rel": float((merged - per_image) / per_image.abs().clamp_min(1e-8)),
    }

    # 2. projection variance
    vals = []
    for k in range(12):
        gk = torch.Generator().manual_seed(k + 100)
        p = torch.randn(D, 32, generator=gk)
        p = (p / p.norm(dim=0, keepdim=True)).cuda()
        vals.append(float(C.swd2_proj(desc[0], tgt[:64], proj=p)))
    vt = torch.tensor(vals)
    out["projection_variance"] = {
        "mean": float(vt.mean()), "std": float(vt.std()),
        "cv": float(vt.std() / vt.mean().abs().clamp_min(1e-8)),
        "min": float(vt.min()), "max": float(vt.max()),
    }

    # 3. unequal sample counts (searchsorted / quantile alignment)
    rows = {}
    for n_src, n_tgt in ((64, 320), (320, 64), (256, 256), (64, 64)):
        v = float(C.swd2_proj(desc[0][:n_src], tgt[:n_tgt], proj=proj))
        rows[f"{n_src}v{n_tgt}"] = v
    out["sample_count_bias"] = rows

    # 4. b1_style_operator weights (static read of the module defaults)
    import b1_style_operator as b1
    out["b1_defaults"] = {
        k: getattr(getattr(b1, "StyleOperatorConfig", object), k, None)
        for k in ("swd_weight", "moment_weight", "selfsim_weight", "edge_weight")}

    C.save_json("code_checks.json", out)
    print(out)


if __name__ == "__main__":
    main()
