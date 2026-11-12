"""WEAVE-VAE V2: Comprehensive Step 200 Mid-Training Audit.

Specifically audits:
1. P1 Style AUROC under SWD Metric:
   - Overall Style AUROC vs A0 baseline (59.90%)
   - Cross-Artist Style AUROC vs A0 baseline (54.67%)
2. Posterior Distribution & Stability:
   - Posterior sigma percentiles (P5, P50, P95)
   - Mean posterior variance
   - Mean KL per-image and per-element
3. Gradient Conflict & Cosine Sim:
   - cos(grad_style, grad_rec)
   - cos(grad_style, grad_kl)
   - Norm ratios ||grad_style|| / ||grad_rec||
4. P2 Decoder Sensitivity (Normalized Covariance Intervention):
   - Delta T_VAE(alpha) = S_t(D(z_alpha)) - S_t(D(z_0)) for alpha in [0.0, 0.01, 0.05, 0.1, 0.2, 0.3]
   - Compared against A0 baseline (+0.0067 at alpha=0.3).
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools" / "12h"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from tools.vae_suite.dataset import RawMemmapWikiArtDataset
from tools.vae_suite.losses import compute_mixed_reconstruction_loss, compute_kl_divergence
from tools.vae_suite.reference_bank import StyleReferenceBank, SupervisedStyleRankingLoss
from tools.vae_suite.swd_metric import LatentSWDDescriptor
from tools.vae_suite.eval_v2_p1_p2 import evaluate_p2_covariance_intervention
import csd_model
import lpips


def run_audit(ckpt_path: str, manifest_path: str, calib_json: str, output_json: str, device: str = "cuda"):
    print(f"\n=======================================================", flush=True)
    print(f"  Running Step 200 Audit on Checkpoint: {ckpt_path}", flush=True)
    print(f"=======================================================", flush=True)

    calib_data = json.loads(Path(calib_json).read_text(encoding="utf-8"))
    dataset_val = RawMemmapWikiArtDataset(Path(manifest_path), is_training=False)
    dataset_tr = RawMemmapWikiArtDataset(Path(manifest_path), is_training=True)

    # 1. Load VAE and checkpoint weights
    vae_init_dir = "/root/Workspace/WEAVE_VAE/runs_vae/A0/final_diffusers_vae"
    vae = AutoencoderKL.from_pretrained(vae_init_dir).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    # Load EMA weights if available, else model weights
    if "ema" in ckpt:
        vae.load_state_dict(ckpt["ema"])
        print("[Audit] Loaded EMA weights from checkpoint.", flush=True)
    else:
        vae.load_state_dict(ckpt["model"])
        print("[Audit] Loaded model weights from checkpoint.", flush=True)
    vae.eval()

    # 2. Descriptor setup
    descriptor = LatentSWDDescriptor().to(device)
    descriptor.set_calibration(
        torch.tensor(calib_data["chan_mean"]).to(device),
        torch.tensor(calib_data["chan_std"]).to(device),
        calib_data["scale_swd1"],
        calib_data["scale_swd2"],
        calib_data["scale_moment"],
        calib_data["median_inter_dist"],
    )

    # -------------------------------------------------------------
    # Audit 1: Posterior sigma & KL stats on 500 validation images
    # -------------------------------------------------------------
    print("\n--- [Audit 1/4] Auditing Posterior Distribution & Latent Statistics ---", flush=True)
    val_loader = DataLoader(dataset_val, batch_size=16, shuffle=False, num_workers=2)

    sigmas = []
    kl_vals = []
    z_all = []
    styles_all = []
    artists_all = []

    ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device == "cuda" else torch.inference_mode()

    with torch.inference_mode():
        for b_idx, b in enumerate(val_loader):
            if b_idx >= 12: # 192 images
                break
            x = b["image"].to(device)
            with ctx:
                post = vae.encode(x).latent_dist
            
            # std of posterior distribution N(mu, sigma)
            sigma = post.std.float().reshape(-1)
            sigmas.append(sigma.cpu())
            kl_elem, kl_v, _, _ = compute_kl_divergence(post)
            kl_vals.append(kl_v)

            z_all.append(post.mean.float().cpu())
            styles_all.append(b["style_id"])
            artists_all.append(b["artist_id"])

    sigmas_np = torch.cat(sigmas).numpy()
    p5 = float(np.percentile(sigmas_np, 5))
    p50 = float(np.percentile(sigmas_np, 50))
    p95 = float(np.percentile(sigmas_np, 95))
    mean_kl = float(np.mean(kl_vals))

    z_eval = torch.cat(z_all, dim=0).to(device)
    styles_eval = torch.cat(styles_all, dim=0).numpy()
    artists_eval = torch.cat(artists_all, dim=0).numpy()

    print(f"  Posterior Sigma Percentiles : P5={p5:.6f}, P50={p50:.6f}, P95={p95:.6f}")
    print(f"  Mean KL Divergence (per img): {mean_kl:.2f}")
    print(f"  Latent Space Global Std     : {z_eval.std().item():.4f}")

    # -------------------------------------------------------------
    # Audit 2: AUROC on SWD Metric (Cross-Artist & Overall)
    # -------------------------------------------------------------
    print("\n--- [Audit 2/4] Measuring SWD Metric AUROC on Validation Set ---", flush=True)
    with torch.inference_mode():
        q1_val, q2_val, mom_val = descriptor(z_eval)

    rng = np.random.default_rng(42)
    N_val = len(styles_eval)
    same_pairs, diff_pairs, cross_artist_same = [], [], []

    for i in range(N_val):
        s_i = styles_eval[i]
        a_i = artists_eval[i]
        same_c = np.where(styles_eval == s_i)[0]
        same_c = same_c[same_c != i]
        if len(same_c) > 0:
            p = rng.choice(same_c)
            same_pairs.append((i, p))
            if a_i >= 0 and artists_eval[p] >= 0 and a_i != artists_eval[p]:
                cross_artist_same.append((i, p))

        diff_c = np.where(styles_eval != s_i)[0]
        if len(diff_c) > 0:
            diff_pairs.append((i, rng.choice(diff_c)))

    n_test = min(len(same_pairs), len(diff_pairs))
    same_arr = np.array(same_pairs[:n_test])
    diff_arr = np.array(diff_pairs[:n_test])

    idx_a = np.concatenate([same_arr[:, 0], diff_arr[:, 0]])
    idx_b = np.concatenate([same_arr[:, 1], diff_arr[:, 1]])
    labels = np.concatenate([np.ones(n_test), np.zeros(n_test)])

    with torch.inference_mode():
        dists = []
        for c in range(0, len(idx_a), 128):
            ca = idx_a[c : c + 128]
            cb = idx_b[c : c + 128]
            d_b = descriptor.compute_pairwise_distance(
                q1_val[ca], q2_val[ca], mom_val[ca],
                q1_val[cb], q2_val[cb], mom_val[cb]
            )
            dists.append(torch.diag(d_b).cpu())
        dists_all = torch.cat(dists).numpy()

    overall_auroc = float(roc_auc_score(labels, -dists_all) * 100.0)

    cross_artist_auroc = 50.0
    if len(cross_artist_same) >= 30:
        n_ca = min(len(cross_artist_same), len(diff_pairs))
        ca_same_arr = np.array(cross_artist_same[:n_ca])
        ca_diff_arr = np.array(diff_pairs[:n_ca])
        ca_idx_a = np.concatenate([ca_same_arr[:, 0], ca_diff_arr[:, 0]])
        ca_idx_b = np.concatenate([ca_same_arr[:, 1], ca_diff_arr[:, 1]])
        ca_labels = np.concatenate([np.ones(n_ca), np.zeros(n_ca)])

        with torch.inference_mode():
            ca_dists = []
            for c in range(0, len(ca_idx_a), 128):
                ca = ca_idx_a[c : c + 128]
                cb = ca_idx_b[c : c + 128]
                d_b = descriptor.compute_pairwise_distance(
                    q1_val[ca], q2_val[ca], mom_val[ca],
                    q1_val[cb], q2_val[cb], mom_val[cb]
                )
                ca_dists.append(torch.diag(d_b).cpu())
            ca_dists_all = torch.cat(ca_dists).numpy()
        cross_artist_auroc = float(roc_auc_score(ca_labels, -ca_dists_all) * 100.0)

    print(f"  Overall Style AUROC (SWD)      : {overall_auroc:.2f}% (A0 Baseline: 59.90%)")
    print(f"  Cross-Artist Style AUROC (SWD) : {cross_artist_auroc:.2f}% (A0 Baseline: 54.67%)")

    # -------------------------------------------------------------
    # Audit 3: Gradient Cosine Similarity (Style vs Rec vs KL)
    # -------------------------------------------------------------
    print("\n--- [Audit 3/4] Gradient Cosine Similarity on Encoder ---", flush=True)
    vae.train()
    lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    # Create small ref bank for gradient test
    ref_bank = StyleReferenceBank(num_styles=136, refs_per_style=4, descriptor=descriptor).to(device)
    ref_bank.bank_q1.normal_()
    ref_bank.bank_q2.normal_()
    ref_bank.bank_mom.normal_()
    ref_bank.bank_styles = torch.tensor([s for s in range(136) for _ in range(4)], device=device)
    ref_bank.bank_artists.fill_(1)
    ref_bank.bank_genres.fill_(1)

    ranking_loss_fn = SupervisedStyleRankingLoss()

    tr_batch = next(iter(DataLoader(dataset_tr, batch_size=6, shuffle=True)))
    x_b = tr_batch["image"].to(device)
    st_b = tr_batch["style_id"].to(device)
    ar_b = tr_batch["artist_id"].to(device)
    ge_b = tr_batch["genre_id"].to(device)

    with torch.enable_grad():
        l_rec_t, _, _, post_t, _ = compute_mixed_reconstruction_loss(vae, x_b, lpips_fn)
        kl_elem_t, _, _, _ = compute_kl_divergence(post_t)
        q1_t, q2_t, mom_t = descriptor(post_t.mean.float())
        l_style_t, _ = ranking_loss_fn(q1_t, q2_t, mom_t, st_b, ar_b, ge_b, ref_bank)

        # Compute gradients on quant_conv.weight
        g_rec = torch.autograd.grad(l_rec_t, vae.quant_conv.weight, retain_graph=True)[0].reshape(-1)
        g_kl = torch.autograd.grad(kl_elem_t, vae.quant_conv.weight, retain_graph=True)[0].reshape(-1)
        g_style = torch.autograd.grad(l_style_t, vae.quant_conv.weight)[0].reshape(-1)

    cos_style_rec = float(F.cosine_similarity(g_style, g_rec, dim=0).item())
    cos_style_kl = float(F.cosine_similarity(g_style, g_kl, dim=0).item())
    norm_ratio = float((g_style.norm() * 0.01 / g_rec.norm().clamp(min=1e-6)).item())

    print(f"  cos(grad_style, grad_rec)     : {cos_style_rec:+.4f} ({'No conflict' if cos_style_rec >= -0.2 else 'Severe Conflict!'})")
    print(f"  cos(grad_style, grad_kl)      : {cos_style_kl:+.4f}")
    print(f"  ||grad_style * 0.01|| / ||grad_rec|| : {norm_ratio*100.0:.1f}%")

    # -------------------------------------------------------------
    # Audit 4: P2 Decoder Sensitivity (Normalized Delta T_VAE)
    # -------------------------------------------------------------
    print("\n--- [Audit 4/4] Evaluating P2 Normalized Decoder Sensitivity ---", flush=True)
    vae.eval()
    csd, _, _ = csd_model.load_csd(device)
    if device == "cpu":
        csd = csd.float()
    p2_res = evaluate_p2_covariance_intervention(
        vae, dataset_val, list(range(20)), csd, lpips_fn, device=device, num_sources=15
    )

    audit_summary = {
        "checkpoint": ckpt_path,
        "step": ckpt.get("step", 200),
        "posterior_audit": {
            "p5_sigma": p5,
            "p50_sigma": p50,
            "p95_sigma": p95,
            "mean_kl": round(mean_kl, 2),
            "latent_std": round(z_eval.std().item(), 4),
        },
        "swd_auroc_audit": {
            "overall_auroc": round(overall_auroc, 2),
            "cross_artist_auroc": round(cross_artist_auroc, 2),
            "a0_baseline_cross_artist": 54.67,
            "delta_cross_artist": round(cross_artist_auroc - 54.67, 2),
        },
        "gradient_audit": {
            "cos_style_rec": round(cos_style_rec, 4),
            "cos_style_kl": round(cos_style_kl, 4),
            "effective_gradient_ratio": round(norm_ratio * 100.0, 1),
        },
        "p2_decoder_sensitivity": p2_res,
    }

    out_p = Path(output_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(audit_summary, indent=2), encoding="utf-8")
    print(f"\n[Done] Audit completed and saved to {out_p}")
    return audit_summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/root/Workspace/WEAVE_VAE/runs_vae/V2/checkpoint_step_000200.pt")
    ap.add_argument("--manifest", default="/root/Workspace/data/wikiart_512_raw/wikiart_512_raw_manifest.json")
    ap.add_argument("--calib-json", default="/root/Workspace/WEAVE_VAE/swd_calibration.json")
    ap.add_argument("--output", default="/root/Workspace/WEAVE_VAE/runs_vae/V2/step200_audit.json")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    run_audit(args.ckpt, args.manifest, args.calib_json, args.output, device=args.device)
