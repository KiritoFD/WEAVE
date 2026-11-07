"""WEAVE V6: Final 750 Evaluation for Winning Strong Style Recipes (D1_T1_strong & ScoreDiff_s12_lr0.100_anc0.00).

Orchestrates:
1. Full 750 (600 cross + 150 same) evaluation of D1_T1_strong (Recipe T1).
2. Full 750 evaluation of ScoreDiff under unanchored strong editing (s12, lr=0.100, anc=0.00).
3. Evaluates under both paper_rgb_v1 and latent_v1 protocols.
4. Generates comprehensive markdown report: docs/v6/STYLE_STRENGTH_CEILING_REPORT.md.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools" / "12h"))
sys.path.insert(0, str(REPO_ROOT / "tools" / "trt"))

import paths
from tools.evaluation_v2.bootstrap import paired_clustered_bootstrap_comparison
from tools.evaluation_v2.features import (
    EvaluatorModels,
    build_reference_features_cache,
    build_source_features_cache,
)
from tools.evaluation_v2.manifest import build_references_v1, build_tasks_750
from tools.evaluation_v2.metrics import evaluate_method_all_protocols
from tools.evaluation_v2.run_eval import extract_features_for_directory
from tools.v6.canonicalize import ContentCanonicalizer
from tools.v6.models_v6 import C3UNet, ConditionalScoreNet
from tools.v6.pareto_and_full_eval import full_eval750_fn
from tools.v6.score_diff import edit_with_score_difference
from tools.v6.train_eval_v6 import generate_v6

STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
TEST_DIR = REPO_ROOT / "data" / "test"
V6_ROOT = REPO_ROOT / "runs" / "v6"
RENDER_ROOT = paths.asset_root() / "eval_assets" / "v6_main"
DOCS_V6 = REPO_ROOT / "docs" / "v6"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Running Final 750 Evaluation for Strong Style Recipes (Device: {device}) ===")

    tasks_meta = build_tasks_750(TEST_DIR)
    ref_meta = build_references_v1(REPO_ROOT / "runs" / "p0" / "reference_selection.json")
    models = EvaluatorModels(device)
    rgb_source_cache, latent_source_cache = build_source_features_cache(models)
    ref_feats = build_reference_features_cache(models)

    global_stats = torch.load(V6_ROOT / "cache" / "canonicalizer_stats.pt", map_location="cpu", weights_only=False)
    canon = ContentCanonicalizer(global_stats).to(device)

    # 1. Render & Evaluate D1_T1_strong (600 cross + 150 same)
    d1_t1_ck = torch.load(V6_ROOT / "D1_T1_strong" / "D1_T1_strong.pt", map_location="cpu", weights_only=False)
    d1_t1 = C3UNet(base_dim=48, num_styles=5, mode="direct").to(device).eval()
    d1_t1.load_state_dict(d1_t1_ck["model"])

    print("\n[Render] Rendering 750 images for D1_T1_strong...", flush=True)
    full_eval750_fn(
        lambda z, s_id, t_id: generate_v6(d1_t1, canon, z, t_id, canon_mode="c1_gentle"),
        cell="D1_T1_strong_s42",
        device=device,
    )

    # 2. Render & Evaluate ScoreDiff strong (s12, lr0.100, anc0.00)
    sd_ck = torch.load(V6_ROOT / "score_diff" / "score_diff_net.pt", map_location="cpu", weights_only=False)
    sd_model = ConditionalScoreNet(base_dim=64, num_styles=5).to(device).eval()
    sd_model.load_state_dict(sd_ck["model"])

    print("\n[Render] Rendering 750 images for ScoreDiff_s12_lr0.100_anc0.00...", flush=True)
    full_eval750_fn(
        lambda z, s_id, t_id: edit_with_score_difference(
            sd_model, z, s_id, t_id, num_steps=12, step_size=0.100, struct_anchor_weight=0.00, seed=42
        ),
        cell="ScoreDiff_s12_lr0.100_anc0.00_s42",
        device=device,
    )

    # 3. Extract features and evaluate under Evaluator V2 dual protocols
    cells_to_eval = [
        ("D1_T1_strong (Ours Strong)", RENDER_ROOT / "D1_T1_strong_s42" / "images"),
        ("ScoreDiff_strong (Ours Strong)", RENDER_ROOT / "ScoreDiff_s12_lr0.100_anc0.00_s42" / "images"),
    ]

    results = {}
    for mname, mdir in cells_to_eval:
        tasks_eval, gen_feats = extract_features_for_directory(mdir, mname, tasks_meta, models)
        res_dual = evaluate_method_all_protocols(
            mname, tasks_eval, gen_feats,
            rgb_source_cache, latent_source_cache, ref_feats, models.proto_csd,
        )
        results[mname] = res_dual
        rgb_c = res_dual["paper_rgb_v1"]["cross600"]
        print(f"  -> [{mname}] CSD_dT={rgb_c['CSD_dT']:+.4f} Top-1={rgb_c['Target_Top1']*100:.1f}% "
              f"Margin={rgb_c['Target_Margin']:+.4f} DINO-C={rgb_c['DINO_C']:.4f} LPIPS={rgb_c['LPIPS']:.4f} "
              f"styles+={rgb_c['styles_improved']}/5 ukiyo={rgb_c['ukiyo_dT']:+.4f}", flush=True)

    out_json = V6_ROOT / "strong_style_final750_results.json"
    out_json.write_text(json.dumps({
        m: {
            "paper_rgb_v1": {
                k: results[m]["paper_rgb_v1"][k]
                for k in ("protocol", "n_total_evaluated", "cross600", "same150", "all750")
            },
            "latent_v1": {
                k: results[m]["latent_v1"][k]
                for k in ("protocol", "n_total_evaluated", "cross600", "same150", "all750")
            },
        }
        for m in results
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[Done] Saved strong style results to {out_json}")


if __name__ == "__main__":
    main()
