"""WEAVE V7 P2-v3: stable 1-NFE conditional transport prototype.

After two failure modes (pure DMD collapse; hybrid wavelet-style collapse),
this v3 temporarily drops the unstable adversarial/DMD component and trains a
single-step residual transport with:
  - identity on same-style pairs,
  - reconstruction to real target-style *distribution moments* through the existing
    robust in-domain style loss,
  - content transport cost,
  - bounded residual.

This is not the final RDMD objective, but it establishes whether a 1-NFE
pair-conditioned operator can perform usable transport at all on old VAE latents.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import paths  # noqa: E402
from tools.v6.canonicalize import ContentCanonicalizer  # noqa: E402
from tools.v6.train_eval_v6 import build_indomain_style_targets, compute_style_distribution_loss  # noqa: E402
from tools.v7.common import STYLES, V6_ROOT, V7_ROOT, ensure_v7_dirs, save_json  # noqa: E402
from tools.v7.models_v7 import SingleStepTransportNet  # noqa: E402
from tools.v7.p2_train_rdmd import apply_transport, content_transport_cost  # noqa: E402


def main() -> None:
    ensure_v7_dirs()
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== WEAVE V7 P2-v3 Stable 1-NFE Transport (Device: {device}) ===", flush=True)

    asset_root = paths.asset_root()
    orig_dir = asset_root / "pseudo_gt_24h" / "orig_mean_fp16"
    train_by_style = {}
    for sid, s in enumerate(STYLES):
        d = torch.load(orig_dir / f"{sid:02d}_{s}.pt", map_location="cpu", weights_only=False)
        train_by_style[s] = d["latents"].float()[100:]
    gpu_by_style = {s: train_by_style[s].to(device).float() for s in STYLES}
    style_targets = build_indomain_style_targets(train_by_style, asset_root, device=device, top_k=64)

    global_stats = torch.load(V6_ROOT / "cache" / "canonicalizer_stats.pt", map_location="cpu", weights_only=False)
    canon = ContentCanonicalizer(global_stats).to(device).eval()
    transport = SingleStepTransportNet(base_dim=48, num_styles=len(STYLES)).to(device)
    opt = torch.optim.AdamW(transport.parameters(), lr=2e-4, weight_decay=1e-4)

    steps = 1200
    per_style = 8
    B = per_style * len(STYLES)
    lambda_content = 0.25
    lambda_style = 0.6
    lambda_rec = 1.0
    residual_clip = 0.25

    hist = {"style": [], "content": [], "rec": [], "total": []}
    t0 = time.time()
    for step in range(1, steps + 1):
        z_list, src_list = [], []
        for sid, s in enumerate(STYLES):
            idx = torch.randint(0, gpu_by_style[s].shape[0], (per_style,), device=device)
            z_list.append(gpu_by_style[s][idx])
            src_list.append(torch.full((per_style,), sid, device=device, dtype=torch.long))
        z = torch.cat(z_list, dim=0)
        src_ids = torch.cat(src_list, dim=0)
        shift = torch.randint(1, len(STYLES), (B,), device=device)
        tgt_ids = (src_ids + shift) % len(STYLES)

        y = apply_transport(transport, z, src_ids, tgt_ids, residual_clip)
        loss_style = compute_style_distribution_loss(y, tgt_ids, style_targets)
        loss_content = content_transport_cost(z, y, canon)
        # s==t identity is exact by model construction; no invalid identity loss.
        loss_rec = y.new_tensor(0.0)

        loss = lambda_style * loss_style + lambda_content * loss_content
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(transport.parameters(), 1.0)
        opt.step()

        hist["style"].append(float(loss_style.item()))
        hist["content"].append(float(loss_content.item()))
        hist["rec"].append(float(loss_rec.item()))
        hist["total"].append(float(loss.item()))

        if step == 1 or step % 100 == 0 or step == steps:
            print(f"[V7 P2-v3] step {step:4d}/{steps} style={np.mean(hist['style'][-50:]):.4f} "
                  f"content={np.mean(hist['content'][-50:]):.4f} rec={np.mean(hist['rec'][-50:]):.6f} "
                  f"total={np.mean(hist['total'][-50:]):.4f} elapsed={time.time()-t0:.1f}s", flush=True)

    out_ckpt = V7_ROOT / "v7_stable_1nfe_transport.pt"
    torch.save({
        "transport": transport.state_dict(),
        "config": {
            "base_dim": 48,
            "steps": steps,
            "batch": B,
            "lambda_style": lambda_style,
            "lambda_content": lambda_content,
            "lambda_rec": lambda_rec,
            "residual_clip": residual_clip,
        },
    }, out_ckpt)
    save_json({
        "steps": steps,
        "train_wall_sec": round(time.time() - t0, 1),
        "peak_vram_mib": round(torch.cuda.max_memory_allocated(0) / (1024**2), 1),
        "final_losses": {k: round(float(np.mean(v[-50:])), 5) for k, v in hist.items()},
        "checkpoint": str(out_ckpt.relative_to(_REPO_ROOT)),
    }, V7_ROOT / "p2_v3_stable_train_summary.json")
    print(f"[V7 P2-v3] Saved checkpoint to {out_ckpt}", flush=True)


if __name__ == "__main__":
    main()
