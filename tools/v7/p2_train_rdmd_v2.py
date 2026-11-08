"""WEAVE V7 P2-v2: hybrid single-step distribution transport.

The pure RDMD/DMD prototype collapsed (DINO-C ~0.08) with random fake ScoreNet,
and became near-identity after over-stabilization. This v2 keeps the 1-NFE
philosophy but uses a robust hybrid objective that is actually trainable on a
single 3060 within a short prototype budget:

  L = λ_content * C_low/edge(z, T(z))
    + λ_style  * in-domain style distribution matching on T(z)
    + λ_dmd    * bounded real/fake score distribution matching
    + λ_rec    * light identity reconstruction on same-style pairs

No GAN, no RGB reconstruction, no paired pseudo-GT. s==t exact identity by
construction. Inference remains ONE forward pass of SingleStepTransportNet.
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
from tools.v6.models_v6 import ConditionalScoreNet  # noqa: E402
from tools.v6.train_eval_v6 import build_indomain_style_targets, compute_style_distribution_loss  # noqa: E402
from tools.v7.common import STYLES, V7_ROOT, ensure_v7_dirs, save_json  # noqa: E402
from tools.v7.models_v7 import SingleStepTransportNet  # noqa: E402
from tools.v7.p2_train_rdmd import apply_transport, content_transport_cost  # noqa: E402
from tools.v6.canonicalize import ContentCanonicalizer  # noqa: E402
from tools.v7.common import V6_ROOT  # noqa: E402


def main() -> None:
    ensure_v7_dirs()
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== WEAVE V7 P2-v2 Hybrid 1-NFE Transport (Device: {device}) ===", flush=True)

    asset_root = paths.asset_root()
    orig_dir = asset_root / "pseudo_gt_24h" / "orig_mean_fp16"
    train_by_style = {}
    for sid, s in enumerate(STYLES):
        d = torch.load(orig_dir / f"{sid:02d}_{s}.pt", map_location="cpu", weights_only=False)
        train_by_style[s] = d["latents"].float()[100:]
    gpu_by_style = {s: train_by_style[s].to(device).float() for s in STYLES}
    style_targets = build_indomain_style_targets(train_by_style, asset_root, device=device, top_k=64)

    real_score = ConditionalScoreNet(base_dim=64, num_styles=len(STYLES)).to(device).eval()
    real_ck = torch.load(V7_ROOT / "scorenet_s1_pure_mse.pt", map_location="cpu", weights_only=False)
    real_score.load_state_dict(real_ck["model"])
    for p in real_score.parameters():
        p.requires_grad_(False)

    fake_score = ConditionalScoreNet(base_dim=64, num_styles=len(STYLES)).to(device)
    fake_score.load_state_dict(real_ck["model"])
    transport = SingleStepTransportNet(base_dim=48, num_styles=len(STYLES)).to(device)

    global_stats = torch.load(V6_ROOT / "cache" / "canonicalizer_stats.pt", map_location="cpu", weights_only=False)
    canon = ContentCanonicalizer(global_stats).to(device).eval()

    opt_g = torch.optim.AdamW(transport.parameters(), lr=2e-4, weight_decay=1e-4)
    opt_f = torch.optim.AdamW(fake_score.parameters(), lr=1e-4, weight_decay=1e-4)

    steps = 1200
    microbatch_per_style = 8
    B = microbatch_per_style * len(STYLES)
    lambda_content = 0.20
    lambda_style = 1.0
    lambda_dmd = 0.05
    lambda_rec = 0.10
    residual_clip = 0.35
    sigma_dmd = 0.80

    hist = {"fake": [], "dmd": [], "style": [], "content": [], "rec": [], "total": []}
    t0 = time.time()
    for step in range(1, steps + 1):
        z_list, src_list = [], []
        for sid, s in enumerate(STYLES):
            pool = gpu_by_style[s]
            idx = torch.randint(0, pool.shape[0], (microbatch_per_style,), device=device)
            z_list.append(pool[idx])
            src_list.append(torch.full((microbatch_per_style,), sid, device=device, dtype=torch.long))
        z = torch.cat(z_list, dim=0)
        src_ids = torch.cat(src_list, dim=0)
        shift = torch.randint(1, len(STYLES), (B,), device=device)
        tgt_ids = (src_ids + shift) % len(STYLES)

        y = apply_transport(transport, z, src_ids, tgt_ids, residual_clip)

        # 1) Fake score update on detached generated distribution.
        eps_f = torch.randn_like(y)
        sigma_f = torch.full((B,), sigma_dmd, device=device, dtype=torch.float32)
        y_sig = y.detach() + sigma_dmd * eps_f
        eps_fake = fake_score(y_sig, sigma_f, tgt_ids)
        loss_fake = F.mse_loss(eps_fake.float(), eps_f)
        opt_f.zero_grad(set_to_none=True)
        loss_fake.backward()
        torch.nn.utils.clip_grad_norm_(fake_score.parameters(), 1.0)
        opt_f.step()

        # 2) Transport update: robust hybrid objective.
        y = apply_transport(transport, z, src_ids, tgt_ids, residual_clip)
        with torch.no_grad():
            eps_d = torch.randn_like(y)
            y_sig2 = y.detach() + sigma_dmd * eps_d
            sig_vec = torch.full((B,), sigma_dmd, device=device, dtype=torch.float32)
            eps_real = real_score(y_sig2, sig_vec, tgt_ids)
            eps_fake_det = fake_score(y_sig2, sig_vec, tgt_ids)
            target = y.detach() + lambda_dmd * (eps_real - eps_fake_det)
        loss_dmd = F.mse_loss(y.float(), target.float())

        loss_content = content_transport_cost(z, y, canon)
        loss_style = compute_style_distribution_loss(y, tgt_ids, style_targets)

        # Same-style exact identity is enforced by construction; this term only
        # stabilizes early training against runaway residual drift.
        same_ids = torch.randint(0, len(STYLES), (B,), device=device)
        z_same = torch.cat([
            gpu_by_style[s][torch.randint(0, gpu_by_style[s].shape[0], (1,), device=device)]
            for s in STYLES for _ in range(microbatch_per_style)
        ], dim=0)
        # Re-sample correctly per style rather than relying on flattened list order.
        z_same_list = []
        for sid, s in enumerate(STYLES):
            idx = torch.randint(0, gpu_by_style[s].shape[0], (microbatch_per_style,), device=device)
            z_same_list.append(gpu_by_style[s][idx])
        z_same = torch.cat(z_same_list, dim=0)
        same_labels = torch.cat([
            torch.full((microbatch_per_style,), sid, device=device, dtype=torch.long)
            for sid in range(len(STYLES))
        ], dim=0)
        y_same = transport(z_same, same_labels, same_labels)
        loss_rec = F.mse_loss(y_same, z_same)

        loss_g = (
            lambda_content * loss_content
            + lambda_style * loss_style
            + loss_dmd
            + lambda_rec * loss_rec
        )
        opt_g.zero_grad(set_to_none=True)
        loss_g.backward()
        torch.nn.utils.clip_grad_norm_(transport.parameters(), 1.0)
        opt_g.step()

        hist["fake"].append(float(loss_fake.item()))
        hist["dmd"].append(float(loss_dmd.item()))
        hist["style"].append(float(loss_style.item()))
        hist["content"].append(float(loss_content.item()))
        hist["rec"].append(float(loss_rec.item()))
        hist["total"].append(float(loss_g.item()))

        if step == 1 or step % 100 == 0 or step == steps:
            peak = torch.cuda.max_memory_allocated(0) / (1024**2)
            print(f"[V7 P2-v2] step {step:4d}/{steps} fake={np.mean(hist['fake'][-50:]):.4f} "
                  f"dmd={np.mean(hist['dmd'][-50:]):.4f} style={np.mean(hist['style'][-50:]):.4f} "
                  f"content={np.mean(hist['content'][-50:]):.4f} rec={np.mean(hist['rec'][-50:]):.4f} "
                  f"total={np.mean(hist['total'][-50:]):.4f} peak={peak:.0f}MiB "
                  f"elapsed={time.time()-t0:.1f}s", flush=True)

    out_ckpt = V7_ROOT / "rdmd_single_step_hybrid.pt"
    torch.save({
        "transport": transport.state_dict(),
        "fake_score": fake_score.state_dict(),
        "config": {
            "base_dim": 48,
            "steps": steps,
            "batch": B,
            "lambda_content": lambda_content,
            "lambda_style": lambda_style,
            "lambda_dmd": lambda_dmd,
            "lambda_rec": lambda_rec,
            "residual_clip": residual_clip,
            "sigma_dmd": sigma_dmd,
            "real_score": "S1 pure denoising MSE",
        },
    }, out_ckpt)
    save_json({
        "steps": steps,
        "batch": B,
        "train_wall_sec": round(time.time() - t0, 1),
        "peak_vram_mib": round(torch.cuda.max_memory_allocated(0) / (1024**2), 1),
        "final_losses": {k: round(float(np.mean(v[-50:])), 5) for k, v in hist.items()},
        "checkpoint": str(out_ckpt.relative_to(_REPO_ROOT)),
    }, V7_ROOT / "p2_rdmd_hybrid_train_summary.json")
    print(f"[V7 P2-v2] Saved checkpoint to {out_ckpt}", flush=True)


if __name__ == "__main__":
    main()
