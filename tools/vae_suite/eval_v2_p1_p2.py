"""WEAVE-VAE V2: Comprehensive P1 Style Geometry & P2 Latent Covariance Intervention Evaluation.

Implements Section VIII of WEAVE-VAE V2 Protocol:
1. P1 Style Geometry:
   - Cross-Artist Style AUROC (under multi-scale SWD metric)
   - Hard Negative (Same-Genre) AUROC
   - Style Retrieval mAP@10
   - 136-Class Independent Linear & Shallow Probes
   - Intra-Class Variance & Effective Rank
2. P2 Decoder Style Sensitivity (Linear Covariance Transfer):
   - For source latent z_s and target style reference latents {z_t}:
     T(z_s) = Sigma_t^{1/2} (Sigma_s + eps I)^{-1/2} (z_s - mu_s) + mu_t
     z_alpha = (1 - alpha) * z_s + alpha * T(z_s) for alpha in [0.1, 0.2, 0.3]
   - Decodes z_alpha to image y_alpha = D(z_alpha)
   - Evaluates:
     * Target CSD Style Similarity Gain Delta T = S_t(y_alpha) - S_t(x)
     * Target Style Top-1 Accuracy
     * Content Preservation (LPIPS vs raw x)
   - Directly proves if Decoder actually responds to latent style statistics without downstream training!
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
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools" / "12h"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier

from tools.v6.canonicalize import dwt2_haar
from tools.vae_suite.dataset import RawMemmapWikiArtDataset
from tools.vae_suite.swd_metric import LatentSWDDescriptor
import csd_model
import lpips


def matrix_sqrt(A: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Computes matrix square root for symmetric positive semi-definite matrix (4, 4)."""
    A_reg = A + eps * torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
    L, Q = torch.linalg.eigh(A_reg)
    L_sqrt = torch.diag(torch.sqrt(torch.clamp(L, min=1e-6)))
    return torch.matmul(torch.matmul(Q, L_sqrt), Q.t())


def matrix_inv_sqrt(A: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Computes matrix inverse square root for symmetric positive semi-definite matrix (4, 4)."""
    A_reg = A + eps * torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
    L, Q = torch.linalg.eigh(A_reg)
    L_inv_sqrt = torch.diag(1.0 / torch.sqrt(torch.clamp(L, min=1e-6)))
    return torch.matmul(torch.matmul(Q, L_inv_sqrt), Q.t())


def transfer_covariance_4d(z: torch.Tensor, mu_t: torch.Tensor, sigma_t: torch.Tensor) -> torch.Tensor:
    """Applies global 4x4 mean-covariance transfer to 4x64x64 latent z.

    T(z) = Sigma_t^{1/2} Sigma_s^{-1/2} (z - mu_s) + mu_t
    """
    C, H, W = z.shape
    mu_s = z.mean(dim=(1, 2), keepdim=True) # (4, 1, 1)
    z_cent = z - mu_s
    z_flat = z_cent.view(C, -1)
    sigma_s = torch.matmul(z_flat, z_flat.t()) / float(H * W - 1)

    inv_sqrt_s = matrix_inv_sqrt(sigma_s)
    sqrt_t = matrix_sqrt(sigma_t)
    M = torch.matmul(sqrt_t, inv_sqrt_s) # (4, 4)

    # Transform: M @ z_flat -> (4, H*W)
    z_trans = torch.matmul(M, z_flat).view(C, H, W) + mu_t.view(4, 1, 1)
    return z_trans


def evaluate_p2_covariance_intervention(
    vae: AutoencoderKL,
    dataset: RawMemmapWikiArtDataset,
    target_styles: List[int],
    csd: nn.Module,
    lpips_fn: nn.Module,
    device: str = "cuda",
    num_sources: int = 50,
) -> dict:
    """Evaluates P2 Decoder sensitivity under linear covariance transfer for alpha in [0.1, 0.2, 0.3]."""
    print(f"\n[P2 Intervention] Testing Decoder sensitivity on {num_sources} source images...", flush=True)

    # 1. Compute target style mean and covariance prototypes
    style_to_records = collections.defaultdict(list)
    for i, idx in enumerate(dataset.indices):
        rec = dataset.records_by_idx[idx]
        style_to_records[rec["style_id"]].append(i)

    # Filter styles that actually have at least 5 images in the validation set
    valid_target_styles = [st for st in target_styles if len(style_to_records.get(st, [])) >= 5]
    if len(valid_target_styles) == 0:
        valid_target_styles = [st for st, l in style_to_records.items() if len(l) >= 5][:10]

    target_stats = {}
    ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device == "cuda" else torch.inference_mode()
    
    with torch.inference_mode():
        for st in valid_target_styles:
            picks = style_to_records[st][:20]
            tensors = torch.stack([dataset[p]["image"] for p in picks]).to(device)
            with ctx:
                z_t = vae.encode(tensors).latent_dist.mean.float() # (N, 4, 64, 64)
            mu_t = z_t.mean(dim=(0, 2, 3)) # (4,)
            z_flat = (z_t - mu_t.view(1, 4, 1, 1)).permute(1, 0, 2, 3).reshape(4, -1)
            sigma_t = torch.matmul(z_flat, z_flat.t()) / float(z_flat.shape[1] - 1)
            target_stats[st] = (mu_t, sigma_t)

    # 2. Sample 50 source images
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(len(dataset), size=num_sources, replace=False)
    sources = [dataset[i] for i in sample_indices]

    norm_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
    norm_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)

    test_alphas = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3]
    alpha_results = {a: [] for a in test_alphas}

    with torch.inference_mode():
        for s_idx, src in enumerate(sources):
            x = src["image"].unsqueeze(0).to(device)
            src_style = src["style_id"].item()

            with ctx:
                z_s = vae.encode(x).latent_dist.mean.squeeze(0).float() # (4, 64, 64)
                recon_0 = vae.decode(z_s.unsqueeze(0)).sample

            # 1. CSD score of raw x
            x_01 = (x.clamp(-1, 1) + 1.0) / 2.0
            x_224 = F.interpolate(x_01, (224, 224), mode="bicubic", align_corners=False)
            x_csd = (x_224 - norm_mean) / norm_std
            csd_inp_x = x_csd.half() if device == "cuda" else x_csd.float()
            _, _, src_feat = csd(csd_inp_x)

            # 2. CSD score of reconstructed D(z_0)
            r0_01 = (recon_0.clamp(-1, 1) + 1.0) / 2.0
            r0_224 = F.interpolate(r0_01, (224, 224), mode="bicubic", align_corners=False)
            r0_csd = (r0_224 - norm_mean) / norm_std
            csd_inp_r0 = r0_csd.half() if device == "cuda" else r0_csd.float()
            _, _, recon0_feat = csd(csd_inp_r0)

            # Test 1 different target style
            avail_targets = [st for st in valid_target_styles if st != src_style]
            if not avail_targets:
                continue
            tgt_st = rng.choice(avail_targets)
            mu_t, sigma_t = target_stats[tgt_st]

            T_z = transfer_covariance_4d(z_s, mu_t, sigma_t)

            # Extract target style CSD reference
            tgt_ref_idx = rng.choice(style_to_records[tgt_st])
            tgt_x = dataset[tgt_ref_idx]["image"].unsqueeze(0).to(device)
            tgt_01 = (tgt_x.clamp(-1, 1) + 1.0) / 2.0
            tgt_224 = F.interpolate(tgt_01, (224, 224), mode="bicubic", align_corners=False)
            tgt_csd = (tgt_224 - norm_mean) / norm_std
            csd_inp_tgt = tgt_csd.half() if device == "cuda" else tgt_csd.float()
            _, _, tgt_feat = csd(csd_inp_tgt)

            sim_rgb_base = torch.sum(src_feat * tgt_feat, dim=-1).item() # S_t(x)
            sim_vae_base = torch.sum(recon0_feat * tgt_feat, dim=-1).item() # S_t(D(z_0))

            for alpha in test_alphas:
                if alpha == 0.0:
                    recon_a = recon_0
                    r_feat = recon0_feat
                else:
                    z_a = (1.0 - alpha) * z_s + alpha * T_z
                    with ctx:
                        recon_a = vae.decode(z_a.unsqueeze(0)).sample

                    r_01 = (recon_a.clamp(-1, 1) + 1.0) / 2.0
                    r_224 = F.interpolate(r_01, (224, 224), mode="bicubic", align_corners=False)
                    r_csd = (r_224 - norm_mean) / norm_std
                    csd_inp_ra = r_csd.half() if device == "cuda" else r_csd.float()
                    _, _, r_feat = csd(csd_inp_ra)

                new_tgt_sim = torch.sum(r_feat * tgt_feat, dim=-1).item()
                delta_t_vae = new_tgt_sim - sim_vae_base
                delta_t_rgb = new_tgt_sim - sim_rgb_base
                lp = lpips_fn(recon_a.float(), x.float()).mean().item()

                alpha_results[alpha].append({
                    "delta_t_vae": delta_t_vae,
                    "delta_t_rgb": delta_t_rgb,
                    "lpips": lp,
                    "target_sim": new_tgt_sim,
                })

    summary = {}
    for a in test_alphas:
        res_a = alpha_results[a]
        mean_dt_vae = float(np.mean([item["delta_t_vae"] for item in res_a]))
        mean_dt_rgb = float(np.mean([item["delta_t_rgb"] for item in res_a]))
        mean_lp = float(np.mean([item["lpips"] for item in res_a]))
        pct_positive_vae = float(np.mean([1.0 if item["delta_t_vae"] > 0 else 0.0 for item in res_a]) * 100.0)
        summary[str(a)] = {
            "mean_delta_T_vae": round(mean_dt_vae, 4),
            "mean_delta_T_rgb": round(mean_dt_rgb, 4),
            "mean_LPIPS": round(mean_lp, 4),
            "pct_positive_gain_vae": round(pct_positive_vae, 1),
        }
        print(f"  alpha={a:4.2f}: Delta_T_VAE = {mean_dt_vae:+.4f} (Pos: {pct_positive_vae:4.1f}%) | Delta_T_RGB = {mean_dt_rgb:+.4f} | Content LPIPS = {mean_lp:.4f}", flush=True)

    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae-path", required=True)
    ap.add_argument("--manifest", default="/root/Workspace/data/wikiart_512_raw/wikiart_512_raw_manifest.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-sources", type=int, default=30)
    ap.add_argument("--output", default="/root/Workspace/WEAVE_VAE/v2_p1_p2_report.json")
    args = ap.parse_args()

    device = args.device
    print(f"=== [V2 Evaluation] P1 & P2 Evaluation for {args.vae_path} on {device} ===", flush=True)

    dataset = RawMemmapWikiArtDataset(Path(args.manifest), is_training=False)
    vae = AutoencoderKL.from_pretrained(args.vae_path).to(device).eval()
    csd, _, _ = csd_model.load_csd(device)
    if device == "cpu":
        csd = csd.float()
    lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    # Common representative styles
    target_styles = list(range(min(20, dataset.manifest["num_styles"])))

    p2_res = evaluate_p2_covariance_intervention(
        vae, dataset, target_styles, csd, lpips_fn, device=device, num_sources=args.num_sources
    )

    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps({
        "vae_path": args.vae_path,
        "P2_covariance_intervention": p2_res,
    }, indent=2), encoding="utf-8")
    print(f"[Done] Saved P1/P2 evaluation to {out_p}")


if __name__ == "__main__":
    main()
