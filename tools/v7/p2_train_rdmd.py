"""WEAVE V7 P2: single-step regularized distribution matching transport prototype.

Implements a practical RDMD/DMD-style objective on frozen SD1.5 VAE latents:

    T_theta(z,s,t) = z + R_theta(z, e_t-e_s),  exact identity when s==t.

Training alternates:
  1. Fake conditional ScoreNet D_fake on T_theta(z,s,t) outputs (detached):
         L_fake = || eps_fake(y+sigma*eps, sigma, t) - eps ||^2
  2. Generator distribution-matching gradient:
         L_DMD = || alpha * (eps_real - eps_fake) - (y - y_stop_grad) ||^2
     where eps_real comes from the frozen S1 pure-MSE conditional ScoreNet.
     This is the SDS/DMD surrogate: gradient wrt theta equals (fake_score - real_score).
  3. Content transport cost:
         low-frequency C1 LL structure + Haar edge consistency.

No GAN, no RGB reconstruction, no paired pseudo-GT. Inference is 1 network forward.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import paths  # noqa: E402
from wavelet import dwt2_haar  # noqa: E402
from tools.v6.canonicalize import ContentCanonicalizer  # noqa: E402
from tools.v6.models_v6 import ConditionalScoreNet  # noqa: E402
from tools.v7.common import STYLES, V6_ROOT, V7_ROOT, ensure_v7_dirs, save_json  # noqa: E402
from tools.v7.models_v7 import SingleStepTransportNet  # noqa: E402


def sample_sigma(B: int, device: str) -> torch.Tensor:
    sigmas = torch.tensor([0.15, 0.30, 0.50, 0.80], device=device, dtype=torch.float32)
    return sigmas[torch.randint(0, len(sigmas), (B,), device=device)]


def content_transport_cost(z: torch.Tensor, y: torch.Tensor, canon: ContentCanonicalizer) -> torch.Tensor:
    """Soft structure cost. C1 is only a cost, never the sole network input."""
    ll_z, lh_z, hl_z, _ = dwt2_haar(z.float())
    ll_y, lh_y, hl_y, _ = dwt2_haar(y.float())

    def norm_ll(ll):
        m = ll.mean(dim=(2, 3), keepdim=True)
        s = ll.std(dim=(2, 3), keepdim=True).clamp(min=1e-4)
        return (ll - m) / s

    def norm_edge(a, b):
        return torch.cat([a, b], dim=1)

    l_low = F.mse_loss(norm_ll(ll_y), norm_ll(ll_z))
    # Edge layout is normalized globally to tolerate stylistic stroke replacement.
    e_z = norm_edge(lh_z, hl_z)
    e_y = norm_edge(lh_y, hl_y)
    e_z = F.normalize(e_z.flatten(1), dim=-1)
    e_y = F.normalize(e_y.flatten(1), dim=-1)
    l_edge = (1.0 - (e_z * e_y).sum(dim=-1)).mean()
    return l_low + 0.35 * l_edge


class _NoOp:
    """Deprecated marker: old Straight-Through clamp is intentionally removed."""


def apply_transport(transport, z, src_ids, tgt_ids, cap: float | None = None):
    """Unified train/eval output: the model owns the smooth tanh residual bound.

    ``cap`` is kept only for backward-compatible call signatures; it is ignored.
    This prevents the previous training/evaluation mismatch where training used
    a clamp but evaluation used an unbounded residual.
    """
    return transport(z, src_ids, tgt_ids)


def main() -> None:
    ensure_v7_dirs()
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== WEAVE V7 P2 RDMD Single-Step Distribution Transport (Device: {device}) ===", flush=True)

    asset_root = paths.asset_root()
    orig_dir = asset_root / "pseudo_gt_24h" / "orig_mean_fp16"
    train_by_style = {}
    for sid, s in enumerate(STYLES):
        d = torch.load(orig_dir / f"{sid:02d}_{s}.pt", map_location="cpu", weights_only=False)
        train_by_style[s] = d["latents"].float()[100:]
    gpu_by_style = {s: train_by_style[s].to(device).float() for s in STYLES}

    # Frozen real Score = S1 pure denoising MSE estimator from P1.
    real_score = ConditionalScoreNet(base_dim=64, num_styles=len(STYLES)).to(device).eval()
    s1_ckpt = V7_ROOT / "scorenet_s1_pure_mse.pt"
    real_ck = torch.load(s1_ckpt, map_location="cpu", weights_only=False)
    real_score.load_state_dict(real_ck["model"])
    for p in real_score.parameters():
        p.requires_grad_(False)

    fake_score = ConditionalScoreNet(base_dim=64, num_styles=len(STYLES)).to(device)
    # DMD/RDMD practice: initialize the fake score near the real score, then adapt
    # it online to the generator output distribution. Random init makes the first
    # distribution-matching gradient too noisy and tends to collapse content.
    fake_score.load_state_dict(real_ck["model"])
    transport = SingleStepTransportNet(base_dim=48, num_styles=len(STYLES)).to(device)

    n_transport = sum(p.numel() for p in transport.parameters())
    print(f"[V7] TransportNet params={n_transport/1e6:.2f}M; real_score=frozen S1; fake_score trainable", flush=True)

    opt_g = torch.optim.AdamW(transport.parameters(), lr=2e-4, weight_decay=1e-4)
    opt_f = torch.optim.AdamW(fake_score.parameters(), lr=2e-4, weight_decay=1e-4)

    global_stats = torch.load(V6_ROOT / "cache" / "canonicalizer_stats.pt", map_location="cpu", weights_only=False)
    canon = ContentCanonicalizer(global_stats).to(device).eval()

    steps = 1200
    microbatch_per_style = 8
    B = microbatch_per_style * len(STYLES)
    # One-sided score-difference gradient was too aggressive with random fake score.
    # Keep the distribution signal but strongly cap the first-stage update.
    dmd_alpha = 0.10
    lambda_content = 1.5
    residual_clip = 0.20
    sigma_dmd = 0.80

    hist = {"fake": [], "dmd": [], "content": [], "total": []}
    t0 = time.time()
    for step in range(1, steps + 1):
        # Sample balanced source latents and random shifted targets.
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

        sigma = sample_sigma(B, device)
        eps = torch.randn_like(z)
        y = apply_transport(transport, z, src_ids, tgt_ids, residual_clip)

        # 1) Update fake ScoreNet on detached generator outputs.
        y_fake = y.detach()
        y_sig = y_fake + sigma.view(-1, 1, 1, 1) * eps
        eps_fake = fake_score(y_sig, sigma, tgt_ids)
        loss_fake = F.mse_loss(eps_fake.float(), eps)
        opt_f.zero_grad(set_to_none=True)
        loss_fake.backward()
        torch.nn.utils.clip_grad_norm_(fake_score.parameters(), 1.0)
        opt_f.step()

        # 2) Update transport with bounded DMD distribution-matching surrogate.
        y = apply_transport(transport, z, src_ids, tgt_ids, residual_clip)
        with torch.no_grad():
            # Large noise scale gives a smoother, lower-variance score difference.
            eps_dmd = torch.randn_like(y)
            y_sig2 = y.detach() + sigma_dmd * eps_dmd
            sig_vec = torch.full((B,), sigma_dmd, device=device, dtype=torch.float32)
            eps_real = real_score(y_sig2, sig_vec, tgt_ids)
            eps_fake_det = fake_score(y_sig2, sig_vec, tgt_ids)
            # Corrected sign: KL(q||p) descent direction is (eps_real-eps_fake)/sigma.
            direction = (eps_real - eps_fake_det) / sigma_dmd
            target = y.detach() - dmd_alpha * direction
        loss_dmd = 0.5 * F.mse_loss(y.float(), target.float())

        loss_content = content_transport_cost(z, y, canon)
        loss_g = loss_dmd + lambda_content * loss_content
        opt_g.zero_grad(set_to_none=True)
        loss_g.backward()
        torch.nn.utils.clip_grad_norm_(transport.parameters(), 1.0)
        opt_g.step()

        hist["fake"].append(float(loss_fake.item()))
        hist["dmd"].append(float(loss_dmd.item()))
        hist["content"].append(float(loss_content.item()))
        hist["total"].append(float(loss_g.item()))

        if step == 1 or step % 100 == 0 or step == steps:
            peak = torch.cuda.max_memory_allocated(0) / (1024**2)
            print(f"[V7 RDMD] step {step:4d}/{steps} fake={np.mean(hist['fake'][-50:]):.4f} "
                  f"dmd={np.mean(hist['dmd'][-50:]):.4f} content={np.mean(hist['content'][-50:]):.4f} "
                  f"total={np.mean(hist['total'][-50:]):.4f} peak={peak:.0f}MiB "
                  f"elapsed={time.time()-t0:.1f}s", flush=True)

    out_ckpt = V7_ROOT / "rdmd_single_step.pt"
    torch.save({
        "transport": transport.state_dict(),
        "fake_score": fake_score.state_dict(),
        "config": {
            "base_dim": 48,
            "steps": steps,
            "batch": B,
            "dmd_alpha": dmd_alpha,
            "lambda_content": lambda_content,
            "real_score": "S1 pure denoising MSE",
        },
    }, out_ckpt)
    save_json({
        "transport_params": n_transport,
        "steps": steps,
        "batch": B,
        "train_wall_sec": round(time.time() - t0, 1),
        "peak_vram_mib": round(torch.cuda.max_memory_allocated(0) / (1024**2), 1),
        "final_losses": {k: round(float(np.mean(v[-50:])), 5) for k, v in hist.items()},
        "checkpoint": str(out_ckpt.relative_to(_REPO_ROOT)),
    }, V7_ROOT / "p2_rdmd_train_summary.json")
    print(f"[V7 P2] Saved checkpoint to {out_ckpt}", flush=True)


if __name__ == "__main__":
    main()
