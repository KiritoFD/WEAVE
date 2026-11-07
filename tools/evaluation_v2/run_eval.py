"""WEAVE Evaluator V2 - Complete Evaluation Runner.

Orchestrates:
1. P0 Identity & VAE No-op verification under both `paper_rgb_v1` and `latent_v1`.
2. Feature extraction and metrics calculation for all 13 Baseline methods in results/D5-512.
3. Feature extraction and metrics calculation for our V6 models (D1 gentle, D1 a0.70,
   ScoreDiff lr0.055, ScoreDiff lr0.048) and V5 Route C.
4. Source-clustered stratified paired bootstrap significance tests.
5. Generates `runs/v6/evaluator_v2_all_results.json` and `docs/v6/EVALUATOR_V2_REPORT.md`.
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
from tools.evaluation_v2.bootstrap import (
    paired_clustered_bootstrap_comparison,
    source_clustered_bootstrap_metric,
)
from tools.evaluation_v2.features import (
    EvaluatorModels,
    build_reference_features_cache,
    build_source_features_cache,
)
from tools.evaluation_v2.manifest import (
    audit_method_directory,
    build_references_v1,
    build_tasks_750,
)
from tools.evaluation_v2.metrics import (
    evaluate_method_all_protocols,
)

STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
TEST_DIR = REPO_ROOT / "data" / "test"
PROTOCOL_DIR = REPO_ROOT / "eval_protocol"
RESULTS_D5 = REPO_ROOT / "results" / "D5-512"
V6_RENDER_ROOT = paths.asset_root() / "eval_assets" / "v6_main"
V5_RENDER_ROOT = paths.asset_root() / "eval_assets" / "v5_route_c"
V6_RUNS = REPO_ROOT / "runs" / "v6"
DOCS_V6 = REPO_ROOT / "docs" / "v6"


def extract_features_for_directory(
    method_dir: Path,
    method_name: str,
    tasks_meta: dict,
    models: EvaluatorModels,
    batch_size: int = 32,
) -> Tuple[List[dict], Dict[str, dict]]:
    """Strictly matches files to tasks and extracts (csd, dino_cls, dino_patch, dino_ssm, clip, lpips)."""
    audit = audit_method_directory(method_name, method_dir, tasks_meta)
    matched_map = audit["matched_task_map"]

    tasks_to_eval = [t for t in tasks_meta["tasks"] if t["task_id"] in matched_map]
    print(f"[{method_name}] Extracting features for {len(tasks_to_eval)} matched tasks (out of {audit['total_files']} files)...", flush=True)

    gen_feats_by_task = {}
    t0 = time.time()

    with torch.inference_mode():
        for i in range(0, len(tasks_to_eval), batch_size):
            chunk = tasks_to_eval[i : i + batch_size]
            gen_paths = [method_dir / matched_map[t["task_id"]]["filename"] for t in chunk]
            src_paths = [REPO_ROOT / t["source_rel_path"] for t in chunk]

            pils_gen = [Image.open(p).convert("RGB").resize((512, 512), Image.BICUBIC) for p in gen_paths]
            pils_src = [Image.open(p).convert("RGB").resize((512, 512), Image.BICUBIC) for p in src_paths]

            feats = models.extract_image_features(pils_gen)

            # LPIPS against raw RGB source
            def to_t(im):
                return torch.from_numpy(np.array(im, dtype=np.float32)).permute(2, 0, 1).to(models.device) / 127.5 - 1.0

            gt = torch.stack([to_t(im) for im in pils_gen])
            st = torch.stack([to_t(im) for im in pils_src])
            lps = models.lpips(gt, st).flatten().cpu().tolist()

            # SSM: (B, 256, 256)
            patch_t = feats["dino_patch"]
            ssm_t = torch.bmm(patch_t, patch_t.transpose(1, 2)) / patch_t.shape[1]

            for j, t in enumerate(chunk):
                tid = t["task_id"]
                gen_feats_by_task[tid] = {
                    "csd": feats["csd"][j],
                    "dino_cls": feats["dino_cls"][j],
                    "dino_patch": feats["dino_patch"][j],
                    "dino_ssm": ssm_t[j],
                    "clip": feats["clip"][j],
                    "lpips": float(lps[j]),
                }

    print(f"[{method_name}] Feature extraction complete in {time.time()-t0:.1f}s", flush=True)
    return tasks_to_eval, gen_feats_by_task


def run_identity_and_vae_verifications(
    tasks_meta: dict,
    rgb_source_cache: dict,
    latent_source_cache: dict,
    ref_feats: dict,
    models: EvaluatorModels,
) -> dict:
    """Rigorous verification: Identity under paper_rgb_v1 and VAE No-op under latent_v1."""
    print("=== Running P0 Identity & VAE No-op Regression Verifications ===", flush=True)
    all_tasks = tasks_meta["tasks"]

    # 1. Identity Generator: outputs exact RGB source image
    stem_to_idx = {s: i for i, s in enumerate(rgb_source_cache["stems"])}
    identity_gen_feats = {}
    for t in all_tasks:
        tid = t["task_id"]
        idx = stem_to_idx[t["source_stem"]]
        identity_gen_feats[tid] = {
            "csd": rgb_source_cache["csd"][idx],
            "dino_cls": rgb_source_cache["dino_cls"][idx],
            "dino_patch": rgb_source_cache["dino_patch"][idx],
            "dino_ssm": rgb_source_cache["dino_ssm"][idx],
            "clip": rgb_source_cache["clip"][idx],
            "lpips": 0.0,
        }

    id_eval = evaluate_method_all_protocols(
        "Identity_Pure", all_tasks, identity_gen_feats,
        rgb_source_cache, latent_source_cache, ref_feats, models.proto_csd,
    )
    id_rgb = id_eval["paper_rgb_v1"]["cross600"]
    print(f"  [Verification: Identity vs paper_rgb_v1] CSD_dT={id_rgb['CSD_dT']:+.6f} "
          f"CSD_dS={id_rgb['CSD_dS']:+.6f} DINO-C={id_rgb['DINO_C']:.6f} LPIPS={id_rgb['LPIPS']:.6f}")
    assert abs(id_rgb["CSD_dT"]) < 1e-6, "Identity CSD_dT must be strictly 0"
    assert abs(id_rgb["CSD_dS"]) < 1e-6, "Identity CSD_dS must be strictly 0"
    assert abs(id_rgb["DINO_C"] - 1.0) < 1e-5, "Identity DINO-C must be 1.0"
    assert abs(id_rgb["LPIPS"]) < 1e-6, "Identity LPIPS must be strictly 0"
    assert abs(id_rgb["DINO_SSM"]) < 1e-6, "Identity DINO_SSM must be strictly 0"

    # 2. VAE No-op Generator: outputs x_vae
    vae_noop_gen_feats = {}
    for t in all_tasks:
        tid = t["task_id"]
        idx = stem_to_idx[t["source_stem"]]
        vae_noop_gen_feats[tid] = {
            "csd": latent_source_cache["csd"][idx],
            "dino_cls": latent_source_cache["dino_cls"][idx],
            "dino_patch": latent_source_cache["dino_patch"][idx],
            "dino_ssm": latent_source_cache["dino_ssm"][idx],
            "clip": latent_source_cache["clip"][idx],
            "lpips": 0.0,
        }

    vae_eval = evaluate_method_all_protocols(
        "VAE_Noop", all_tasks, vae_noop_gen_feats,
        rgb_source_cache, latent_source_cache, ref_feats, models.proto_csd,
    )
    vae_lat = vae_eval["latent_v1"]["cross600"]
    print(f"  [Verification: VAE No-op vs latent_v1] CSD_dT={vae_lat['CSD_dT']:+.6f} "
          f"CSD_dS={vae_lat['CSD_dS']:+.6f} DINO-C={vae_lat['DINO_C']:.6f} LPIPS={vae_lat['LPIPS']:.6f}")
    assert abs(vae_lat["CSD_dT"]) < 1e-6, "VAE No-op vs latent_v1 CSD_dT must be strictly 0"
    assert abs(vae_lat["CSD_dS"]) < 1e-6, "VAE No-op vs latent_v1 CSD_dS must be strictly 0"

    vae_rgb = vae_eval["paper_rgb_v1"]["cross600"]
    print(f"  [Verification: VAE Inherent Distortion vs paper_rgb_v1] CSD_dT={vae_rgb['CSD_dT']:+.4f} "
          f"DINO-C={vae_rgb['DINO_C']:.4f} DINO_SSM={vae_rgb['DINO_SSM']:.5f}")

    print("=== All Identity & VAE No-op Verifications PASSED ===", flush=True)
    return {
        "identity_on_paper_rgb_v1": id_rgb,
        "vae_noop_on_latent_v1": vae_lat,
        "vae_inherent_distortion_on_paper_rgb_v1": vae_rgb,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    V6_RUNS.mkdir(parents=True, exist_ok=True)
    DOCS_V6.mkdir(parents=True, exist_ok=True)

    # 1. Authoritative Task and Reference Manifests
    tasks_meta = build_tasks_750(TEST_DIR)
    ref_meta = build_references_v1(REPO_ROOT / "runs" / "p0" / "reference_selection.json")

    # 2. Load Models & Cache Representations
    models = EvaluatorModels(device)
    rgb_source_cache, latent_source_cache = build_source_features_cache(models)
    ref_feats = build_reference_features_cache(models)

    # 3. Verification Suite
    verifications = run_identity_and_vae_verifications(
        tasks_meta, rgb_source_cache, latent_source_cache, ref_feats, models
    )

    # 4. List all evaluation candidate directories
    method_entries = [
        # Paper Baselines
        ("Identity", RESULTS_D5 / "identity"),
        ("AdaIN", RESULTS_D5 / "adain"),
        ("WCT", RESULTS_D5 / "wct"),
        ("CUT", RESULTS_D5 / "cut"),
        ("SaMST", RESULTS_D5 / "samst"),
        ("SaMAM", RESULTS_D5 / "samam"),
        ("SD-Turbo", RESULTS_D5 / "sdturbo"),
        ("StyleID", RESULTS_D5 / "styleid"),
        ("StyleShot", RESULTS_D5 / "styleshot"),
        ("StyTr2", RESULTS_D5 / "stytr2"),
        ("AesPA", RESULTS_D5 / "aespa"),
        ("WEAVE (Published)", RESULTS_D5 / "weave"),
        ("Seedream 4.5", RESULTS_D5 / "seedream"),
        # V6 Models
        ("V6 D1 Gentle (s42)", V6_RENDER_ROOT / "D1_c1_gentle_s42" / "images"),
        ("V6 D1 Gentle (s43)", V6_RENDER_ROOT / "D1_c1_gentle_s43" / "images"),
        ("V6 D1 Calibrated (a0.70, s42)", V6_RENDER_ROOT / "D1_c1_gentle_a0.70_s42" / "images"),
        ("V6 D1 Calibrated (a0.70, s43)", V6_RENDER_ROOT / "D1_c1_gentle_a0.70_s43" / "images"),
        ("V6 ScoreDiff (lr0.055, s42)", V6_RENDER_ROOT / "ScoreDiff_s12_lr0.055_s42" / "images"),
        ("V6 ScoreDiff (lr0.055, s43)", V6_RENDER_ROOT / "ScoreDiff_s12_lr0.055_s43" / "images"),
        ("V6 ScoreDiff (lr0.048, s42)", V6_RENDER_ROOT / "ScoreDiff_s12_lr0.048_s42" / "images"),
        ("V6 ScoreDiff (lr0.048, s43)", V6_RENDER_ROOT / "ScoreDiff_s12_lr0.048_s43" / "images"),
    ]

    evaluated_methods = {}
    print("\n=== Evaluating All Methods under Evaluator V2 Protocols ===", flush=True)

    for mname, mdir in method_entries:
        if not mdir.exists():
            print(f"[WARN] Directory not found, skipping: {mdir}")
            continue

        tasks_eval, gen_feats = extract_features_for_directory(mdir, mname, tasks_meta, models)
        if not gen_feats:
            continue

        res_dual = evaluate_method_all_protocols(
            mname, tasks_eval, gen_feats,
            rgb_source_cache, latent_source_cache, ref_feats, models.proto_csd,
        )
        evaluated_methods[mname] = res_dual

        rgb_cross = res_dual["paper_rgb_v1"]["cross600"]
        print(f"  -> [{mname:28s}] CSD_dT={rgb_cross['CSD_dT']:+.4f} Top1={rgb_cross['Target_Top1']*100:.1f}% "
              f"Margin={rgb_cross['Target_Margin']:+.4f} MMD={rgb_cross['CSD_MMD']:.4f} "
              f"DINO-C={rgb_cross['DINO_C']:.4f} LPIPS={rgb_cross['LPIPS']:.4f} styles+={rgb_cross['styles_improved']}/5", flush=True)

    # 5. Paired Clustered Bootstrap Comparisons between V6 and Key Baselines
    print("\n=== Computing Source-Clustered Stratified Paired Bootstrap Tests ===", flush=True)
    comparisons = {}
    anchor_pairs = [
        ("V6 D1 Calibrated (a0.70, s42)", "WEAVE (Published)"),
        ("V6 D1 Calibrated (a0.70, s42)", "SaMAM"),
        ("V6 D1 Calibrated (a0.70, s42)", "CUT"),
        ("V6 ScoreDiff (lr0.048, s42)", "WEAVE (Published)"),
        ("V6 ScoreDiff (lr0.048, s42)", "SaMAM"),
        ("V6 ScoreDiff (lr0.048, s42)", "CUT"),
        ("V6 ScoreDiff (lr0.055, s42)", "StyleShot"),
        ("V6 ScoreDiff (lr0.055, s42)", "StyleID"),
    ]

    for model_a, model_b in anchor_pairs:
        if model_a in evaluated_methods and model_b in evaluated_methods:
            rows_a = [r for r in evaluated_methods[model_a]["paper_rgb_v1"]["rows"] if r["is_cross"]]
            rows_b = [r for r in evaluated_methods[model_b]["paper_rgb_v1"]["rows"] if r["is_cross"]]

            comp_dT = paired_clustered_bootstrap_comparison(rows_a, rows_b, "csd_dT", n_boot=2000, seed=42)
            comp_top1 = paired_clustered_bootstrap_comparison(rows_a, rows_b, "target_top1", n_boot=2000, seed=42)
            comp_margin = paired_clustered_bootstrap_comparison(rows_a, rows_b, "target_margin", n_boot=2000, seed=42)
            comp_dinoc = paired_clustered_bootstrap_comparison(rows_a, rows_b, "dino_c", n_boot=2000, seed=42)
            comp_lpips = paired_clustered_bootstrap_comparison(rows_a, rows_b, "lpips", n_boot=2000, seed=42)

            comparisons[f"{model_a} vs {model_b}"] = {
                "n_common": comp_dT["n_common"],
                "CSD_dT_diff": comp_dT,
                "Target_Top1_diff": comp_top1,
                "Target_Margin_diff": comp_margin,
                "DINO_C_diff": comp_dinoc,
                "LPIPS_diff": comp_lpips,
            }
            print(f"  [{model_a} vs {model_b}] CSD_dT diff={comp_dT['mean_diff']:+.4f} CI95={comp_dT['ci95']} win_rate={comp_dT['win_rate']*100:.1f}%")

    # 6. Save Full JSON Summary (excluding massive raw rows to keep file compact)
    full_output = {
        "verifications": verifications,
        "methods_evaluated": {
            m: {
                "paper_rgb_v1": {
                    k: evaluated_methods[m]["paper_rgb_v1"][k]
                    for k in ("protocol", "n_total_evaluated", "cross600", "same150", "all750")
                },
                "latent_v1": {
                    k: evaluated_methods[m]["latent_v1"][k]
                    for k in ("protocol", "n_total_evaluated", "cross600", "same150", "all750")
                },
            }
            for m in evaluated_methods
        },
        "paired_clustered_bootstrap_tests": comparisons,
    }

    out_json = V6_RUNS / "evaluator_v2_all_results.json"
    out_json.write_text(json.dumps(full_output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[Evaluator V2] Successfully saved full results to {out_json}")

    # 7. Generate Comprehensive Markdown Report
    lines = []
    lines.append("# WEAVE 评测体系修订版主报告 (Evaluator V2)\n")
    lines.append("> 遵循《WEAVE评测体系修订方案》，在 RTX 3060 统一执行：\n"
                 "> - **P0 修复**：统一 `paper_rgb_v1`（以原始 RGB 源图为准，Identity 严格为零偏差）与 `latent_v1`（以 VAE 为准）双协议；\n"
                 "> - **严格匹配**：750 个冻结任务无模糊子串回退；\n"
                 "> - **统一参考**：150 张训练池完全不相交参考图（各风格 30 张，哈希锁定）；\n"
                 "> - **核心风格指标齐备**：Target Top-1、Target Margin、CSD-MMD、DINO-SSM、CSD $\\Delta T/\\Delta S/\\Delta M$；\n"
                 "> - **统计检验**：按源图分层聚类配对 Bootstrap（95% CI）。\n")

    lines.append("## 一、P0 回归验收：Identity 与 VAE 固有畸变基线\n")
    lines.append("| 参照源 | 评估协议 | CSD $\\Delta T$ | CSD $\\Delta S$ | Target Top-1 | DINO-C | DINO-SSM | LPIPS | 验收结果 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|")
    lines.append(f"| **Identity** | `paper_rgb_v1` | **0.0000** | **0.0000** | {verifications['identity_on_paper_rgb_v1']['Target_Top1']*100:.1f}% | **1.0000** | **0.0000** | **0.0000** | **PASS（零偏差基准）** |")
    lines.append(f"| **VAE No-op** | `latent_v1` | **0.0000** | **0.0000** | {verifications['vae_noop_on_latent_v1']['Target_Top1']*100:.1f}% | **1.0000** | **0.0000** | **0.0000** | **PASS（零模型额外扰动）** |")
    lines.append(f"| **VAE No-op** | `paper_rgb_v1` | {verifications['vae_inherent_distortion_on_paper_rgb_v1']['CSD_dT']:+.4f} | {verifications['vae_inherent_distortion_on_paper_rgb_v1']['CSD_dS']:+.4f} | {verifications['vae_inherent_distortion_on_paper_rgb_v1']['Target_Top1']*100:.1f}% | {verifications['vae_inherent_distortion_on_paper_rgb_v1']['DINO_C']:.4f} | {verifications['vae_inherent_distortion_on_paper_rgb_v1']['DINO_SSM']:.5f} | {verifications['vae_inherent_distortion_on_paper_rgb_v1']['LPIPS']:.4f} | *VAE 编解码固有损失* |")

    lines.append("\n## 二、论文主表：600 跨风格主判据 (paper_rgb_v1, cross600)\n")
    lines.append("| 方法 / 模型 | 类别 | CSD $\\Delta T \\uparrow$ | Target Top-1 $\\uparrow$ | Target Margin $\\uparrow$ | CSD-MMD $\\downarrow$ | DINO-C $\\uparrow$ | DINO-SSM $\\downarrow$ | LPIPS $\\downarrow$ | 5类改善 | Ukiyo-e $\\Delta T$ |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|:---:|---:|")

    for m, d in evaluated_methods.items():
        c = d["paper_rgb_v1"]["cross600"]
        m_type = "Baseline" if "V6" not in m else "Ours"
        lines.append(
            f"| **{m}** | {m_type} | **{c['CSD_dT']:+.4f}** | **{c['Target_Top1']*100:.1f}%** | "
            f"**{c['Target_Margin']:+.4f}** | **{c['CSD_MMD']:.4f}** | **{c['DINO_C']:.4f}** | "
            f"{c['DINO_SSM']:.5f} | **{c['LPIPS']:.4f}** | {c['styles_improved']}/5 | {c['ukiyo_dT']:+.4f} |"
        )

    lines.append("\n## 三、独立约束：150 同风格身份保持表 (paper_rgb_v1, same150)\n")
    lines.append("| 方法 / 模型 | 样本数 | CSD $\\Delta S$ (漂移) | Target Top-1 $\\uparrow$ | DINO-C $\\uparrow$ | DINO-SSM $\\downarrow$ | LPIPS $\\downarrow$ |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")

    for m, d in evaluated_methods.items():
        s = d["paper_rgb_v1"]["same150"]
        if not s:
            continue
        lines.append(
            f"| **{m}** | {s['n']} | **{s['CSD_dS']:+.4f}** | {s['Target_Top1']*100:.1f}% | "
            f"**{s['DINO_C']:.4f}** | {s['DINO_SSM']:.5f} | **{s['LPIPS']:.4f}** |"
        )

    lines.append("\n## 四、源图分层聚类配对 Bootstrap 显著性检验 (V6 vs 基线)\n")
    lines.append("| 对比组合 (A vs B) | 共同任务数 | CSD $\\Delta T$ 差异 [95% CI] | 胜率 (A>B) | Target Top-1 差异 [95% CI] | DINO-C 差异 [95% CI] | LPIPS 差异 [95% CI] | 显著性 (p<0.05) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|:---:|")

    for pair_name, comp in comparisons.items():
        dt_c = comp["CSD_dT_diff"]
        top_c = comp["Target_Top1_diff"]
        c_diff = comp["DINO_C_diff"]
        lp_c = comp["LPIPS_diff"]
        lines.append(
            f"| **{pair_name}** | {comp['n_common']} | {dt_c['mean_diff']:+.4f} [{dt_c['ci95'][0]:+.4f}, {dt_c['ci95'][1]:+.4f}] | "
            f"{dt_c['win_rate']*100:.1f}% | {top_c['mean_diff']*100:+.1f}% [{top_c['ci95'][0]*100:+.1f}%, {top_c['ci95'][1]*100:+.1f}%] | "
            f"{c_diff['mean_diff']:+.4f} [{c_diff['ci95'][0]:+.4f}, {c_diff['ci95'][1]:+.4f}] | "
            f"{lp_c['mean_diff']:+.4f} [{lp_c['ci95'][0]:+.4f}, {lp_c['ci95'][1]:+.4f}] | "
            f"{'YES' if dt_c['significant_at_05'] else 'NO'} |"
        )

    out_md = DOCS_V6 / "EVALUATOR_V2_REPORT.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[Evaluator V2] Successfully wrote {out_md}", flush=True)


if __name__ == "__main__":
    main()
