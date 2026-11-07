"""WEAVE V6: Inference Strength & Freedom Exploration Sweep on Validation 200 Tasks.

Tests the style transfer capacity ceiling of existing V6 models WITHOUT re-training:
1. D1 Direct Operator (D1_c1_gentle and D1_c1_pure):
   z' = z + alpha * (G_phi(c, t) - z)
   for alpha in [1.0, 1.4, 1.8, 2.2]
2. Conditional Score Difference (g_{s -> t}):
   Grid sweep across:
   - step_size: [0.055, 0.10, 0.16]
   - num_steps: [12, 24]
   - struct_anchor_weight: [0.08, 0.00] (turning off spatial layout anchoring)

Monitors the key diagnostic thresholds specified in Section V of user's proposal:
  - Target Top-1: target 15% - 25% (currently ~5%)
  - Target Margin: target >= -0.20 (currently -0.33)
  - CSD dT: target >= +0.12 (currently +0.055)
  - DINO-C: allowed to drop to 0.60 - 0.75 for strong artistic transformation
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools" / "12h"))
sys.path.insert(0, str(REPO_ROOT / "tools" / "trt"))

import paths
import csd_model
from tools.pseudo_gt.matching import encode_dino, load_dino_model
from utils.run_evaluation import _load_eval_image_tensor, to_lpips_input
from tools.v5.route_c import csd_prototypes, decode_trt, load_trt, load_vae
from tools.v6.canonicalize import ContentCanonicalizer
from tools.v6.models_v6 import C3UNet, ConditionalScoreNet
from tools.v6.score_diff import edit_with_score_difference
from tools.v6.train_eval_v6 import boot_ci, generate_v6

STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
V5_ROOT = REPO_ROOT / "runs" / "v5"
V6_ROOT = REPO_ROOT / "runs" / "v6"
DOCS_V6 = REPO_ROOT / "docs" / "v6"


def evaluate_latents_on_val200(
    gen_fn,
    tasks: List[dict],
    z_all: torch.Tensor,
    src_idx: torch.Tensor,
    src_ids_200: torch.Tensor,
    tgt_ids_200: torch.Tensor,
    src_tensors: Dict[str, torch.Tensor],
    csd_src_cache: Dict[str, np.ndarray],
    dino_src_cache: Dict[str, np.ndarray],
    csd: torch.nn.Module,
    proto: np.ndarray,
    dino: torch.nn.Module,
    lpips_fn: torch.nn.Module,
    vae: torch.nn.Module,
    trt,
    device: str = "cuda",
) -> dict:
    """Vectorized evaluation of any generator on the 200 cross-style validation tasks."""
    with torch.inference_mode():
        z_gen = gen_fn(z_all[src_idx], src_ids_200, tgt_ids_200)
        img = decode_trt(trt, vae, z_gen, batch=8)
        u8 = img.mul(255.0).round().clamp(0, 255).to(torch.uint8).cpu()
        gen01 = u8.float().div_(255.0)
        arr_gen = u8.permute(0, 2, 3, 1).numpy()
        pils = [Image.fromarray(arr_gen[i]) for i in range(len(tasks))]

        fg = csd_model.feats_pils(csd, pils, batch=16, device=device)  # (200, 768)
        dg, _ = encode_dino(dino, pils, batch=16, device=device)
        dg = dg.float().cpu().numpy()

        lps = []
        for s in range(0, len(tasks), 16):
            chunk = gen01[s : s + 16].to(device)
            src = torch.cat([src_tensors[tasks[idx]["src_stem"]] for idx in range(s, min(s + 16, len(tasks)))], 0)
            lps.append(lpips_fn(to_lpips_input(chunk), to_lpips_input(src)).flatten().cpu())
        lps = torch.cat(lps, 0).numpy()

    # CSD 5-way cosine matrix: (200, 5)
    scores = fg @ proto.T
    pred_top1 = np.argmax(scores, axis=1)
    tgt_numpy = tgt_ids_200.cpu().numpy()
    src_numpy = src_ids_200.cpu().numpy()

    target_top1_hits = (pred_top1 == tgt_numpy)
    top1_rate = float(target_top1_hits.mean())

    # Target margin: S_t - max_{k != t} S_k
    target_margins = []
    for i in range(len(tasks)):
        tid = tgt_numpy[i]
        mask_other = [k for k in range(5) if k != tid]
        m_i = scores[i, tid] - np.max(scores[i, mask_other])
        target_margins.append(m_i)
    target_margin_arr = np.array(target_margins)
    mean_target_margin = float(target_margin_arr.mean())

    dTs, dSs, dMs, dCs = [], [], [], []
    per_style_dT = {s: [] for s in STYLES}
    for i in range(len(tasks)):
        tid = tgt_numpy[i]
        sid = src_numpy[i]
        stem = tasks[i]["src_stem"]
        fs = csd_src_cache[stem]
        dt = float(scores[i, tid] - (fs @ proto[tid]))
        ds = float(scores[i, sid] - (fs @ proto[sid]))
        dTs.append(dt)
        dSs.append(ds)
        dMs.append(dt - ds)
        dCs.append(float(dg[i] @ dino_src_cache[stem]))
        per_style_dT[tasks[i]["tgt_style"]].append(dt)

    dT_arr = np.array(dTs)
    per_tgt_mean = {s: float(np.mean(per_style_dT[s])) for s in STYLES}
    styles_improved = sum(1 for v in per_tgt_mean.values() if v > 0)

    return {
        "mean_dT": round(float(dT_arr.mean()), 4),
        "dT_ci": [round(x, 4) for x in boot_ci(dT_arr)],
        "mean_dS": round(float(np.mean(dSs)), 4),
        "mean_dM": round(float(np.mean(dMs)), 4),
        "Target_Top1": round(top1_rate * 100.0, 1),
        "Target_Margin": round(mean_target_margin, 4),
        "DINO_C": round(float(np.mean(dCs)), 4),
        "LPIPS": round(float(lps.mean()), 4),
        "styles_improved": styles_improved,
        "ukiyo_dT": round(per_tgt_mean["Ukiyo_e"], 4),
        "per_target_dT": {k: round(v, 4) for k, v in per_tgt_mean.items()},
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== WEAVE V6 Inference Strength & Style Ceiling Sweep (Device: {device}) ===")

    # 1. Load Validation 200 tasks & source features
    tasks = json.loads((V5_ROOT / "v5_eval200_tasks.json").read_text(encoding="utf-8"))
    lat_dict = torch.load(V5_ROOT / "cache" / "eval_src_latents.pt", map_location="cpu")
    stems = sorted({t["src_stem"] for t in tasks})
    z_all = torch.stack([lat_dict[s] for s in stems]).to(device)
    stem_idx = {s: i for i, s in enumerate(stems)}
    src_idx = torch.tensor([stem_idx[t["src_stem"]] for t in tasks], device=device)
    src_ids_200 = torch.tensor([t["src_style_id"] for t in tasks], device=device, dtype=torch.long)
    tgt_ids_200 = torch.tensor([t["tgt_style_id"] for t in tasks], device=device, dtype=torch.long)

    csd_src50 = torch.load(V5_ROOT / "cache" / "eval_src_csd_feats.pt", map_location="cpu", weights_only=False)
    csd_src_cache = {s: np.asarray(csd_src50[s]) for s in stems}

    src_tensors = {
        t["src_stem"]: _load_eval_image_tensor(Path(t["src_path"]), size=512).unsqueeze(0).to(device)
        for t in tasks
    }

    # 2. Load Evaluation Models
    t0 = time.time()
    proto = csd_prototypes()
    csd, _, _ = csd_model.load_csd(device)
    dino = load_dino_model(device=device)
    vae = load_vae(device)
    trt = load_trt(vae, device)
    import lpips
    lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    # Precompute source DINO CLS features
    dino_src_cache = {}
    with torch.inference_mode():
        img_src = decode_trt(trt, vae, z_all, batch=8)
        u8_src = img_src.mul(255.0).round().clamp(0, 255).to(torch.uint8).cpu().permute(0, 2, 3, 1).numpy()
        pils_src = [Image.fromarray(u8_src[i]) for i in range(len(stems))]
        dcls, _ = encode_dino(dino, pils_src, batch=16, device=device)
        for i, s in enumerate(stems):
            dino_src_cache[s] = dcls[i].float().cpu().numpy()
    print(f"[Setup] Evaluation models & source cache ready in {time.time()-t0:.1f}s", flush=True)

    # 3. Canonicalizer
    global_stats = torch.load(V6_ROOT / "cache" / "canonicalizer_stats.pt", map_location="cpu", weights_only=False)
    canon = ContentCanonicalizer(global_stats).to(device)

    results = {
        "D1_c1_gentle_alpha_sweep": {},
        "D1_c1_pure_alpha_sweep": {},
        "ScoreDiff_inference_grid": {},
    }

    eval_kwargs = dict(
        tasks=tasks, z_all=z_all, src_idx=src_idx,
        src_ids_200=src_ids_200, tgt_ids_200=tgt_ids_200,
        src_tensors=src_tensors, csd_src_cache=csd_src_cache,
        dino_src_cache=dino_src_cache, csd=csd, proto=proto,
        dino=dino, lpips_fn=lpips_fn, vae=vae, trt=trt, device=device,
    )

    # =========================================================================
    # Part 1: D1 Direct Operator Residual Extrapolation: z' = z + alpha * (G(c, t) - z)
    # =========================================================================
    for d1_tag, canon_mode, ckpt_name in (
        ("D1_c1_gentle", "c1_gentle", "D1_c1_gentle.pt"),
        ("D1_c1_pure", "c1", "D1_c1_pure.pt"),
    ):
        ckpt_path = V6_ROOT / d1_tag / ckpt_name
        if not ckpt_path.exists():
            print(f"[Sweep] Skipping {d1_tag} (checkpoint not found)")
            continue
        print(f"\n--- Sweeping Residual Alpha for {d1_tag} (canon={canon_mode}) ---", flush=True)
        d1_ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        d1_model = C3UNet(base_dim=48, num_styles=5, mode="direct").to(device).eval()
        d1_model.load_state_dict(d1_ck["model"])

        sweep_target = results["D1_c1_gentle_alpha_sweep"] if d1_tag == "D1_c1_gentle" else results["D1_c1_pure_alpha_sweep"]

        for alpha in [1.0, 1.4, 1.8, 2.2]:
            name = f"alpha_{alpha:.1f}"
            t_sw = time.time()
            def d1_gen(z_batch, s_ids, t_ids, al=alpha, cm=canon_mode):
                g_out = generate_v6(d1_model, canon, z_batch, t_ids, canon_mode=cm)
                delta = g_out - z_batch
                return z_batch + al * delta

            res = evaluate_latents_on_val200(d1_gen, **eval_kwargs)
            res["alpha"] = alpha
            res["wall_sec"] = round(time.time() - t_sw, 1)
            sweep_target[name] = res
            print(f"  [{d1_tag} alpha={alpha:.1f}] CSD_dT={res['mean_dT']:+.4f} Top-1={res['Target_Top1']}% "
                  f"Margin={res['Target_Margin']:+.4f} DINO-C={res['DINO_C']:.4f} LPIPS={res['LPIPS']:.4f} "
                  f"ukiyo={res['ukiyo_dT']:+.4f} ({res['wall_sec']}s)", flush=True)

    # =========================================================================
    # Part 2: Score Difference Grid: step_size x num_steps x anchor
    # =========================================================================
    sd_ckpt_path = V6_ROOT / "score_diff" / "score_diff_net.pt"
    if sd_ckpt_path.exists():
        print(f"\n--- Sweeping Score Difference Inference Parameters ---", flush=True)
        sd_ck = torch.load(sd_ckpt_path, map_location="cpu", weights_only=False)
        sd_model = ConditionalScoreNet(base_dim=64, num_styles=5).to(device).eval()
        sd_model.load_state_dict(sd_ck["model"])

        step_sizes = [0.055, 0.10, 0.16]
        num_steps_list = [12, 24]
        anchors = [0.08, 0.00]

        for n_steps in num_steps_list:
            for lr_step in step_sizes:
                for anc in anchors:
                    name = f"steps{n_steps}_lr{lr_step:.3f}_anc{anc:.2f}"
                    t_sw = time.time()
                    def sd_gen(z_batch, s_ids, t_ids, st=n_steps, lr=lr_step, a=anc):
                        return edit_with_score_difference(
                            sd_model, z_batch, s_ids, t_ids,
                            num_steps=st, step_size=lr, struct_anchor_weight=a, seed=42,
                        )

                    res = evaluate_latents_on_val200(sd_gen, **eval_kwargs)
                    res["num_steps"] = n_steps
                    res["step_size"] = lr_step
                    res["struct_anchor_weight"] = anc
                    res["wall_sec"] = round(time.time() - t_sw, 1)
                    results["ScoreDiff_inference_grid"][name] = res
                    print(f"  [ScoreDiff {name}] CSD_dT={res['mean_dT']:+.4f} Top-1={res['Target_Top1']}% "
                          f"Margin={res['Target_Margin']:+.4f} DINO-C={res['DINO_C']:.4f} LPIPS={res['LPIPS']:.4f} "
                          f"ukiyo={res['ukiyo_dT']:+.4f} ({res['wall_sec']}s)", flush=True)

    # Save summary json
    out_file = V6_ROOT / "sweep_inference_strength.json"
    out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[Sweep] Saved complete sweep results to {out_file}", flush=True)

    # Markdown Summary
    lines = []
    lines.append("# WEAVE V6 推理强度与风格上限扫描报告 (Validation 200 Tasks)\n")
    lines.append("> 目标：在不重新训练的前提下，通过调整推理参数（D1 残差外推倍率 $\\alpha$，ScoreDiff 积分步长/步数/解除结构锚定），")
    lines.append("> 探索模型是否已学到足够强的画风方向，明确瓶颈是“推理过于保守”还是“未学到强画风”。\n")

    lines.append("## 一、D1 直接算子残差外推扫描 ($z' = z + \\alpha(G(c, t) - z)$)\n")
    lines.append("| 配置 / 模型 | 倍率 $\\alpha$ | CSD $\\Delta T \\uparrow$ | Target Top-1 $\\uparrow$ | Target Margin $\\uparrow$ | DINO-C $\\uparrow$ | LPIPS $\\downarrow$ | 5类改善 | Ukiyo-e $\\Delta T$ |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|:---:|---:|")
    for k, v in results["D1_c1_gentle_alpha_sweep"].items():
        lines.append(f"| **D1_gentle ({k})** | {v['alpha']:.1f} | **{v['mean_dT']:+.4f}** | **{v['Target_Top1']}%** | **{v['Target_Margin']:+.4f}** | {v['DINO_C']:.4f} | {v['LPIPS']:.4f} | {v['styles_improved']}/5 | {v['ukiyo_dT']:+.4f} |")
    for k, v in results["D1_c1_pure_alpha_sweep"].items():
        lines.append(f"| **D1_pure ({k})** | {v['alpha']:.1f} | **{v['mean_dT']:+.4f}** | **{v['Target_Top1']}%** | **{v['Target_Margin']:+.4f}** | {v['DINO_C']:.4f} | {v['LPIPS']:.4f} | {v['styles_improved']}/5 | {v['ukiyo_dT']:+.4f} |")

    lines.append("\n## 二、Score Difference 差分积分强度与无锚定扫描 ($g_{s \\to t}$)\n")
    lines.append("| 配置 (步数_步长_锚定) | 步数 | 步长 | 结构锚定 | CSD $\\Delta T \\uparrow$ | Target Top-1 $\\uparrow$ | Target Margin $\\uparrow$ | DINO-C $\\uparrow$ | LPIPS $\\downarrow$ | 5类改善 | Ukiyo-e $\\Delta T$ |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|")
    for k, v in results["ScoreDiff_inference_grid"].items():
        lines.append(f"| **{k}** | {v['num_steps']} | {v['step_size']:.3f} | {v['struct_anchor_weight']:.2f} | **{v['mean_dT']:+.4f}** | **{v['Target_Top1']}%** | **{v['Target_Margin']:+.4f}** | {v['DINO_C']:.4f} | {v['LPIPS']:.4f} | {v['styles_improved']}/5 | {v['ukiyo_dT']:+.4f} |")

    md_out = DOCS_V6 / "SWEEP_INFERENCE_STRENGTH_REPORT.md"
    md_out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[Sweep] Saved Markdown summary to {md_out}")


if __name__ == "__main__":
    main()
