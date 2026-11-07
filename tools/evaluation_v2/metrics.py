"""WEAVE Evaluator V2 - Comprehensive Metrics Engine.

Computes:
1. Target Style Metrics (Emphasized per user instruction):
   - CSD dT: Target style cosine gain S_t(y) - S_t(x)
   - CSD dS: Source style cosine shift S_s(y) - S_s(x)
   - CSD dM: Margin gain dT - dS
   - Target Top-1: Proportion where t == argmax_k S_k(y)
   - Target Margin: M_t(y) - M_t(x), where M_t(y) = S_t(y) - max_{k != t} S_k(y)
   - CSD-MMD: Unbiased RBF Maximum Mean Discrepancy against real art distribution
   - DINO-S: Max CLS cosine against target style reference set
   - CLIP-S: Cosine against target style CLIP prototype
2. Content Preservation Metrics:
   - DINO-C: Global semantic content cosine cos(CLS(y), CLS(x))
   - DINO-SSM: Normalized spatial self-similarity matrix distance ||P_y P_y^T - P_x P_x^T||_F^2 / 256^2
   - LPIPS: Perceptual distortion distance d(y, x) (AlexNet)
3. Dual Reference Protocols:
   - `paper_rgb_v1`: Source reference is the original RGB source image x.
   - `latent_v1`: Source reference is VAE round-trip x_vae.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]


def rbf_kernel(x: torch.Tensor, y: torch.Tensor, gamma: float) -> torch.Tensor:
    """Compute RBF kernel matrix K_{ij} = exp(-gamma * ||x_i - y_j||^2)."""
    # x: (N, D), y: (M, D)
    dist_sq = torch.cdist(x.float(), y.float()).pow(2)
    return torch.exp(-gamma * dist_sq)


def compute_unbiased_mmd2(x: torch.Tensor, y: torch.Tensor, gamma: float) -> float:
    """Unbiased MMD^2 estimator between sample sets x and y."""
    n = x.shape[0]
    m = y.shape[0]
    if n < 2 or m < 2:
        return 0.0

    k_xx = rbf_kernel(x, x, gamma)
    k_yy = rbf_kernel(y, y, gamma)
    k_xy = rbf_kernel(x, y, gamma)

    # Exclude diagonal for unbiased estimate of E[k(x, x')] and E[k(y, y')]
    k_xx_sum = (k_xx.sum() - torch.diagonal(k_xx).sum()) / (n * (n - 1))
    k_yy_sum = (k_yy.sum() - torch.diagonal(k_yy).sum()) / (m * (m - 1))
    k_xy_sum = k_xy.sum() / (n * m)

    mmd2 = k_xx_sum + k_yy_sum - 2.0 * k_xy_sum
    return float(mmd2.item())


def compute_task_row_metrics(
    task: dict,
    gen_feats: dict,
    src_base_feats: dict,
    proto_csd: np.ndarray,
    ref_feats: dict,
    gamma: float,
) -> dict:
    """Compute all metric values for a single task."""
    sname = task["source_style"]
    tname = task["target_style"]
    sid = task["source_style_id"]
    tid = task["target_style_id"]
    is_cross = task["is_cross"]

    # CSD cosine scores against all 5 style prototypes: (5,)
    cg = gen_feats["csd"].numpy()
    cs = src_base_feats["csd"].numpy()

    scores_g = cg @ proto_csd.T  # (5,)
    scores_s = cs @ proto_csd.T  # (5,)

    # 1. CSD dT, dS, dM
    csd_dT = float(scores_g[tid] - scores_s[tid])
    csd_dS = float(scores_g[sid] - scores_s[sid])
    csd_dM = csd_dT - csd_dS

    # 2. Target Top-1
    pred_g_top1 = int(np.argmax(scores_g))
    pred_s_top1 = int(np.argmax(scores_s))
    target_top1_hit = int(pred_g_top1 == tid)
    source_top1_hit = int(pred_s_top1 == tid)

    # 3. Target Margin: S_t - max_{k != t} S_k
    mask_other_t = [k for k in range(5) if k != tid]
    margin_g = float(scores_g[tid] - np.max(scores_g[mask_other_t]))
    margin_s = float(scores_s[tid] - np.max(scores_s[mask_other_t]))
    target_margin_gain = margin_g - margin_s

    # 4. DINO-C
    dg_cls = gen_feats["dino_cls"]
    ds_cls = src_base_feats["dino_cls"]
    dino_c = float(torch.dot(dg_cls, ds_cls).item())

    # 5. DINO-SSM (spatial self-similarity matrix distance)
    dg_ssm = gen_feats["dino_ssm"]
    ds_ssm = src_base_feats["dino_ssm"]
    dino_ssm = float((dg_ssm - ds_ssm).pow(2).mean().item())

    # 6. DINO-S
    ref_dino_cls = ref_feats["styles"][tname]["dino_cls"]  # (30, 384)
    dino_s = float((dg_cls.unsqueeze(0) @ ref_dino_cls.T).max().item())

    # 7. CLIP-S
    ref_clip_proto = ref_feats["clip_prototypes"][tname]  # (512,)
    clip_s = float(torch.dot(gen_feats["clip"], ref_clip_proto).item())

    # 8. LPIPS
    lpips_val = float(gen_feats.get("lpips", 0.0))

    return {
        "task_id": task["task_id"],
        "source_stem": task["source_stem"],
        "source_style": sname,
        "target_style": tname,
        "is_cross": is_cross,
        "csd_dT": csd_dT,
        "csd_dS": csd_dS,
        "csd_dM": csd_dM,
        "target_top1": target_top1_hit,
        "source_top1": source_top1_hit,
        "target_margin": margin_g,
        "source_margin": margin_s,
        "target_margin_gain": target_margin_gain,
        "dino_c": dino_c,
        "dino_ssm": dino_ssm,
        "dino_s": dino_s,
        "clip_s": clip_s,
        "lpips": lpips_val,
        "csd_score_target": float(scores_g[tid]),
        "csd_score_source": float(scores_g[sid]),
    }


def aggregate_group_metrics(
    rows: List[dict],
    gen_csd_by_target: Dict[str, torch.Tensor],
    ref_feats: dict,
    gamma: float,
) -> dict:
    if not rows:
        return {}

    n = len(rows)
    dT = np.array([r["csd_dT"] for r in rows])
    dS = np.array([r["csd_dS"] for r in rows])
    dM = np.array([r["csd_dM"] for r in rows])
    top1 = np.array([r["target_top1"] for r in rows])
    src_top1 = np.array([r["source_top1"] for r in rows])
    t_margin = np.array([r["target_margin"] for r in rows])
    t_margin_gain = np.array([r["target_margin_gain"] for r in rows])
    d_c = np.array([r["dino_c"] for r in rows])
    d_ssm = np.array([r["dino_ssm"] for r in rows])
    d_s = np.array([r["dino_s"] for r in rows])
    cl_s = np.array([r["clip_s"] for r in rows])
    lp = np.array([r["lpips"] for r in rows])

    # Compute CSD-MMD against real reference distribution per target style
    mmd_by_style = {}
    for sname in STYLES:
        if sname in gen_csd_by_target and gen_csd_by_target[sname].shape[0] > 1:
            y_csd = gen_csd_by_target[sname]
            r_csd = ref_feats["styles"][sname]["csd"]
            mmd_val = compute_unbiased_mmd2(y_csd, r_csd, gamma)
            mmd_by_style[sname] = max(0.0, mmd_val)

    mean_mmd = float(np.mean(list(mmd_by_style.values()))) if mmd_by_style else 0.0

    # Per-target style breakdown
    per_target_dT = {}
    per_target_top1 = {}
    per_target_margin = {}
    for sname in STYLES:
        sub = [r for r in rows if r["target_style"] == sname]
        if sub:
            per_target_dT[sname] = float(np.mean([r["csd_dT"] for r in sub]))
            per_target_top1[sname] = float(np.mean([r["target_top1"] for r in sub]))
            per_target_margin[sname] = float(np.mean([r["target_margin"] for r in sub]))

    styles_improved = sum(1 for v in per_target_dT.values() if v > 0)

    return {
        "n": n,
        "CSD_dT": float(dT.mean()),
        "CSD_dS": float(dS.mean()),
        "CSD_dM": float(dM.mean()),
        "Target_Top1": float(top1.mean()),
        "Source_Top1": float(src_top1.mean()),
        "Target_Top1_Gain": float(top1.mean() - src_top1.mean()),
        "Target_Margin": float(t_margin.mean()),
        "Target_Margin_Gain": float(t_margin_gain.mean()),
        "CSD_MMD": mean_mmd,
        "CSD_MMD_by_style": mmd_by_style,
        "DINO_C": float(d_c.mean()),
        "DINO_SSM": float(d_ssm.mean()),
        "DINO_S": float(d_s.mean()),
        "CLIP_S": float(cl_s.mean()),
        "LPIPS": float(lp.mean()),
        "frac_dT_positive": float((dT > 0).mean()),
        "styles_improved": styles_improved,
        "ukiyo_dT": per_target_dT.get("Ukiyo_e", 0.0),
        "per_target_dT": per_target_dT,
        "per_target_top1": per_target_top1,
        "per_target_margin": per_target_margin,
    }


def evaluate_method_all_protocols(
    method_name: str,
    parsed_tasks: List[dict],
    gen_feats_by_task: Dict[str, dict],
    rgb_source_cache: dict,
    latent_source_cache: dict,
    ref_feats: dict,
    proto_csd: np.ndarray,
) -> dict:
    """Evaluate a method under BOTH `paper_rgb_v1` and `latent_v1` protocols."""
    gamma = float(ref_feats.get("rbf_gamma", 0.5))

    # Helper to map source stem to cache index
    stem_to_idx = {stem: i for i, stem in enumerate(rgb_source_cache["stems"])}

    results_by_protocol = {}

    for proto_name, src_cache in (("paper_rgb_v1", rgb_source_cache), ("latent_v1", latent_source_cache)):
        rows = []
        gen_csd_by_target = {s: [] for s in STYLES}

        for t in parsed_tasks:
            tid = t["task_id"]
            if tid not in gen_feats_by_task:
                continue
            gf = gen_feats_by_task[tid]
            src_idx = stem_to_idx[t["source_stem"]]

            src_base_feats = {
                "csd": src_cache["csd"][src_idx],
                "dino_cls": src_cache["dino_cls"][src_idx],
                "dino_ssm": src_cache["dino_ssm"][src_idx],
                "clip": src_cache["clip"][src_idx],
            }

            r = compute_task_row_metrics(t, gf, src_base_feats, proto_csd, ref_feats, gamma)
            rows.append(r)
            gen_csd_by_target[t["target_style"]].append(gf["csd"].unsqueeze(0))

        gen_csd_by_target = {
            s: torch.cat(tensors, dim=0) if tensors else torch.empty((0, 768))
            for s, tensors in gen_csd_by_target.items()
        }

        cross_rows = [r for r in rows if r["is_cross"]]
        same_rows = [r for r in rows if not r["is_cross"]]

        results_by_protocol[proto_name] = {
            "protocol": proto_name,
            "n_total_evaluated": len(rows),
            "cross600": aggregate_group_metrics(cross_rows, gen_csd_by_target, ref_feats, gamma),
            "same150": aggregate_group_metrics(same_rows, gen_csd_by_target, ref_feats, gamma),
            "all750": aggregate_group_metrics(rows, gen_csd_by_target, ref_feats, gamma),
            "rows": rows,
        }

    return results_by_protocol
