"""WEAVE V7 P3: full 600 cross + 150 same-style evaluation.

Evaluates:
- V7 RDMD single-step transport (1 NFE)
- V7 RDMD with inference residual scale sweep alpha in [1.0, 1.4, 1.8]
- Calib_Balanced_s16 ScoreDiff reference (96 NFE), on the same frozen 750 tasks
- Optional pure S1 Tweedie 2-NFE diagnostic

Output:
- runs/v7/p3_v7_eval750_results.json
- docs/v7/V7_RDMD_EVALUATION_REPORT.md
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.evaluation_v2.manifest import build_tasks_750  # noqa: E402
from tools.v6.score_diff import edit_with_score_difference  # noqa: E402
from tools.v7.common import (
    DOCS_V7,
    STYLES,
    V7_ROOT,
    aggregate_rows,
    batch_tensors,
    build_source_latents,
    compute_batch_rows,
    ensure_v7_dirs,
    load_eval_bundle,
    load_score_model,
    save_json,
)
from tools.v7.models_v7 import SingleStepTransportNet  # noqa: E402

CALIB_SCALES = {
    "Minimalism": 0.50,
    "Ukiyo_e": 1.60,
    "Rococo": 1.40,
    "Impressionism": 1.30,
    "Early_Renaissance": 1.20,
}
BATCH = 16


def source_cache_for_test(models) -> Dict[str, dict]:
    """Load Evaluator V2 cached RGB source features for the 150 frozen test sources."""
    cache = torch.load(_REPO_ROOT / "eval_protocol" / "cache" / "sources_paper_rgb_v1.pt", map_location="cpu", weights_only=False)
    out = {}
    from PIL import Image
    for i, stem in enumerate(cache["stems"]):
        # rel path from tasks manifest is recovered lazily in compute_batch_rows if absent.
        out[stem] = {
            "csd": cache["csd"][i].float(),
            "dino_cls": cache["dino_cls"][i].float(),
            "clip": cache["clip"][i].float(),
        }
    return out


def main() -> None:
    ensure_v7_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== WEAVE V7 P3 Full 750 Evaluation (Device: {device}) ===", flush=True)

    manifest = build_tasks_750()
    tasks = manifest["tasks"]
    for t in tasks:
        if "source_rel_path" not in t:
            prefix = t["source_style"]
            t["source_rel_path"] = f"data/test/{prefix}/{t['source_filename']}"

    models, ref_feats, vae, trt = load_eval_bundle(device)
    # Source cache needs PILs for LPIPS; compute_batch_rows will lazily fill from rel_path.
    source_cache = source_cache_for_test(models)
    src_latents = build_source_latents(tasks, vae, device)

    v7_ck = torch.load(V7_ROOT / "rdmd_single_step_hybrid.pt", map_location="cpu", weights_only=False)
    transport = SingleStepTransportNet(base_dim=48, num_styles=len(STYLES)).to(device).eval()
    transport.load_state_dict(v7_ck["transport"])

    sd_model = load_score_model(device=device)

    results: dict = {
        "protocol": {
            "tasks": "eval_protocol/tasks_750.json (600 cross + 150 same)",
            "rgb_reference": "paper_rgb_v1",
            "reference_pool": "eval_protocol/references_v1.json",
            "v7_inference": "SingleStepTransportNet residual scaling; s==t exact identity by construction",
        },
        "methods": {},
    }

    def eval_gen(name: str, gen_fn, nfe: int, note: str = "") -> None:
        rows = []
        t0 = time.time()
        for i in range(0, len(tasks), BATCH):
            chunk = tasks[i : i + BATCH]
            z, s, t, ids = batch_tensors(chunk, src_latents, device)
            z_gen = gen_fn(z, s, t, ids)
            l2 = (z_gen.float() - z.float()).flatten(1).norm(p=2, dim=1).cpu().tolist()
            rows.extend(compute_batch_rows(z_gen, chunk, models, ref_feats, vae, trt, source_cache, latent_l2=l2))
        wall = time.time() - t0

        cross = aggregate_rows(rows, only_cross=True)
        same = aggregate_rows([r for r in rows if not r.get("is_cross", True)])
        allv = aggregate_rows(rows)
        for g in (cross, same, allv):
            if g:
                g["NFE"] = nfe
                g["gen_wall_sec"] = round(wall, 1)
                g["note"] = note
        results["methods"][name] = {"cross600": cross, "same150": same, "all750": allv}
        print(f"  [{name:32s}] cross Top-1={cross['Target_Top1']:4.1f}% dT={cross['mean_dT']:+.4f} "
              f"Margin={cross['Target_Margin']:+.4f} DINO-C={cross['DINO_C']:.4f} "
              f"LPIPS={cross['LPIPS']:.4f} ukiyo={cross['ukiyo_top1']:4.1f}% "
              f"styles+={cross['styles_improved']}/5 same_dT={same['mean_dT'] if same else 0:+.4f} "
              f"gen={wall:.1f}s", flush=True)

    # 1. V7 one-step operator
    def v7_gen(alpha=1.0):
        def _fn(z, s, t, ids, a=alpha):
            base = z
            y = transport(z, s, t)
            return base + a * (y - base)
        return _fn

    for alpha in [1.0, 1.4, 1.8]:
        eval_gen(f"V7_Hybrid_alpha{alpha:.1f}", v7_gen(alpha), nfe=1, note="single-step hybrid distribution transport")

    # 2. ScoreDiff reference on same 750 tasks
    def calib_gen(z, s, t, ids):
        return edit_with_score_difference(
            sd_model, z, s, t,
            num_steps=16, step_size=0.120, struct_anchor_weight=0.08,
            seed=42, task_ids=ids, style_step_scales=CALIB_SCALES,
        )
    eval_gen("Calib_Balanced_s16_ref", calib_gen, nfe=96, note="V6.1 ScoreDiff calibrated baseline")

    # 3. Select best V7 by Target Top-1 among DINO-C >= 0.60 (guardrail, diagnostic)
    v7_methods = {k: v for k, v in results["methods"].items() if k.startswith("V7_")}
    eligible = [(k, v) for k, v in v7_methods.items() if v["cross600"]["DINO_C"] >= 0.60]
    best_name, best = max(
        eligible or list(v7_methods.items()),
        key=lambda kv: kv[1]["cross600"]["Target_Top1"],
    )
    best_cross = best["cross600"]
    base_cross = results["methods"]["Calib_Balanced_s16_ref"]["cross600"]

    results["acceptance"] = {
        "best_v7": best_name,
        "criteria": {
            "NFE_eq_1": best_cross["NFE"] == 1,
            "Top1_ge_27.7": best_cross["Target_Top1"] >= 27.7,
            "dT_ge_0.08": best_cross["mean_dT"] >= 0.08,
            "DINO_C_ge_0.68": best_cross["DINO_C"] >= 0.68,
            "Ukiyo_Top1_ge_15": best_cross["ukiyo_top1"] >= 15.0,
            "Ukiyo_dT_gt_VAE_Noop": best_cross["per_target_dT"]["Ukiyo_e"] > 0.0389,
            "Top1_ge_Calib_Balanced_on_same_tasks": best_cross["Target_Top1"] >= base_cross["Target_Top1"],
        },
        "comparison": {
            "V7_Top1": best_cross["Target_Top1"],
            "Calib_Balanced_Top1": base_cross["Target_Top1"],
            "V7_dT": best_cross["mean_dT"],
            "Calib_Balanced_dT": base_cross["mean_dT"],
            "V7_DINO_C": best_cross["DINO_C"],
            "Calib_Balanced_DINO_C": base_cross["DINO_C"],
        },
    }

    out_json = V7_ROOT / "p3_v7_hybrid_eval750_results.json"
    save_json(results, out_json)

    lines = []
    lines.append("# WEAVE V7 P3：单步 RDMD 分布传输 750 全量评测报告\n")
    lines.append("> 评测：冻结 `eval_protocol/tasks_750.json`，600 跨风格 + 150 同风格；")
    lines.append("> `paper_rgb_v1` 原始 RGB 源图协议；统一 `references_v1` 风格参考池；")
    lines.append("> ScoreDiff 使用 task_id 独立哈希噪声，V7 为确定性 1-NFE 前向。\n")

    lines.append("## 一、600 跨风格主表\n")
    lines.append("| 模型 | NFE | CSD $\\Delta T \\uparrow$ | Target Top-1 $\\uparrow$ | Target Margin $\\uparrow$ | DINO-C $\\uparrow$ | LPIPS $\\downarrow$ | 5类改善 | Ukiyo-e Top-1 | Ukiyo-e $\\Delta T$ | 生成耗时 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|:---:|---:|---:|---:|")
    for name, d in results["methods"].items():
        c = d["cross600"]
        lines.append(f"| **{name}** | **{c['NFE']}** | **{c['mean_dT']:+.4f}** | **{c['Target_Top1']}%** | "
                     f"**{c['Target_Margin']:+.4f}** | {c['DINO_C']:.4f} | {c['LPIPS']:.4f} | "
                     f"{c['styles_improved']}/5 | **{c['ukiyo_top1']}%** | {c['ukiyo_dT']:+.4f} | {c['gen_wall_sec']}s |")

    lines.append("\n## 二、150 同风格身份保持\n")
    lines.append("| 模型 | NFE | CSD $\\Delta S$ | DINO-C | LPIPS | 潜空间位移 L2 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name, d in results["methods"].items():
        s = d["same150"]
        lines.append(f"| **{name}** | {s['NFE']} | **{s['mean_dS']:+.4f}** | **{s['DINO_C']:.4f}** | "
                     f"{s['LPIPS']:.4f} | {s['mean_latent_l2']:.4f} |")

    lines.append("\n## 三、与验收目标的对照\n")
    lines.append("| 指标 | V7 最佳 | 目标 | 是否达成 |")
    lines.append("|---|---:|---:|:---:|")
    lines.append(f"| 潜空间 NFE | **1** | 1 | **YES** |")
    lines.append(f"| Target Top-1 | **{best_cross['Target_Top1']}%** | ≥ 27.7% | {'YES' if best_cross['Target_Top1'] >= 27.7 else 'NO'} |")
    lines.append(f"| CSD $\\Delta T$ | **{best_cross['mean_dT']:+.4f}** | ≥ +0.08 | {'YES' if best_cross['mean_dT'] >= 0.08 else 'NO'} |")
    lines.append(f"| DINO-C | **{best_cross['DINO_C']:.4f}** | ≥ 0.68 | {'YES' if best_cross['DINO_C'] >= 0.68 else 'NO'} |")
    lines.append(f"| Ukiyo-e Top-1 | **{best_cross['ukiyo_top1']}%** | ≥ 15% | {'YES' if best_cross['ukiyo_top1'] >= 15 else 'NO'} |")
    lines.append(f"| Ukiyo-e $\\Delta T$ | **{best_cross['ukiyo_dT']:+.4f}** | > VAE No-op (+0.0389) | {'YES' if best_cross['per_target_dT']['Ukiyo_e'] > 0.0389 else 'NO'} |")

    lines.append("\n## 四、结论\n")
    lines.append(f"- V7 最佳配置为 `{best_name}`：1 次前向完成迁移，Top-1 = **{best_cross['Target_Top1']}%**，$\\Delta T$ = **{best_cross['mean_dT']:+.4f}**，DINO-C = **{best_cross['DINO_C']:.4f}**，Ukiyo-e Top-1 = **{best_cross['ukiyo_top1']}%**。")
    lines.append(f"- 同批任务上 Calib_Balanced_s16 (96 NFE)：Top-1 = **{base_cross['Target_Top1']}%**，$\\Delta T$ = **{base_cross['mean_dT']:+.4f}**，DINO-C = **{base_cross['DINO_C']:.4f}**，Ukiyo-e Top-1 = **{base_cross['ukiyo_top1']}%**。")
    lines.append("- 若 V7 尚未全部达标，应把结论限定为“验证 1-NFE 分布传输可行性与失败模式”，")
    lines.append("  而不是宣称全面替代 ScoreDiff；下一步应优先改进真实 Score / 风格原型 / 新 VAE 潜空间，而非回到多步积分。\n")

    out_md = DOCS_V7 / "V7_HYBRID_EVALUATION_REPORT.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[V7 P3] Saved {out_json} and {out_md}", flush=True)


if __name__ == "__main__":
    main()
