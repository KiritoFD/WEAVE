"""P3 evaluation for V7 corrected-DMD 1-NFE transport checkpoints.

Uses the exact same output function as training: SingleStepTransportNet.forward,
which includes c*tanh(R/c) and exact s==t identity. Evaluates all checkpoints
matching ``p2_fix_*.pt`` on frozen 600 cross + 150 same tasks.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.evaluation_v2.manifest import build_tasks_750  # noqa: E402
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
    save_json,
)
from tools.v7.models_v7 import SingleStepTransportNet  # noqa: E402

BATCH = 16


def source_cache_for_test() -> Dict[str, dict]:
    cache = torch.load(_REPO_ROOT / "eval_protocol" / "cache" / "sources_paper_rgb_v1.pt", map_location="cpu", weights_only=False)
    return {
        stem: {
            "csd": cache["csd"][i].float(),
            "dino_cls": cache["dino_cls"][i].float(),
            "clip": cache["clip"][i].float(),
        }
        for i, stem in enumerate(cache["stems"])
    }


def main() -> None:
    ensure_v7_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== WEAVE V7 Corrected-DMD P3 Evaluation (Device: {device}) ===", flush=True)

    manifest = build_tasks_750()
    tasks = manifest["tasks"]
    for t in tasks:
        if "source_rel_path" not in t:
            t["source_rel_path"] = f"data/test/{t['source_style']}/{t['source_filename']}"

    models, ref_feats, vae, trt = load_eval_bundle(device)
    source_cache = source_cache_for_test()
    src_latents = build_source_latents(tasks, vae, device)

    ckpts = sorted(V7_ROOT.glob("p2_fix_*.pt"))
    if not ckpts:
        raise FileNotFoundError("No p2_fix_*.pt checkpoints found")

    results = {
        "protocol": {
            "tasks": "eval_protocol/tasks_750.json (600 cross + 150 same)",
            "rgb_reference": "paper_rgb_v1",
            "reference_pool": "eval_protocol/references_v1.json",
            "output": "SingleStepTransportNet.forward with smooth tanh residual, identical to training",
            "NFE": 1,
        },
        "methods": {},
    }

    for ckpt_path in ckpts:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cap = float(ck["config"].get("residual_cap", 0.35))
        model = SingleStepTransportNet(base_dim=48, num_styles=len(STYLES), residual_cap=cap).to(device).eval()
        model.load_state_dict(ck["transport"])
        name = ckpt_path.stem.replace("p2_fix_", "V7_Fix_")

        rows = []
        t0 = time.time()
        for i in range(0, len(tasks), BATCH):
            chunk = tasks[i : i + BATCH]
            z, s, t, ids = batch_tensors(chunk, src_latents, device)
            # Exact training/inference function: no external clamp.
            z_gen = model(z, s, t)
            l2 = (z_gen.float() - z.float()).flatten(1).norm(p=2, dim=1).cpu().tolist()
            rows.extend(compute_batch_rows(z_gen, chunk, models, ref_feats, vae, trt, source_cache, latent_l2=l2))
        wall = round(time.time() - t0, 1)

        cross = aggregate_rows(rows, only_cross=True)
        same = aggregate_rows([r for r in rows if not r.get("is_cross", True)])
        allv = aggregate_rows(rows)
        for g in (cross, same, allv):
            if g:
                g["NFE"] = 1
                g["gen_wall_sec"] = wall
                g["residual_cap"] = cap
        results["methods"][name] = {"cross600": cross, "same150": same, "all750": allv}
        print(f"  [{name:22s}] Top-1={cross['Target_Top1']:4.1f}% dT={cross['mean_dT']:+.4f} "
              f"Margin={cross['Target_Margin']:+.4f} DINO-C={cross['DINO_C']:.4f} "
              f"LPIPS={cross['LPIPS']:.4f} ukiyo={cross['ukiyo_top1']:4.1f}% "
              f"styles+={cross['styles_improved']}/5 same_dS={same['mean_dS']:+.4f} gen={wall}s", flush=True)

    out_json = V7_ROOT / "p3_v7_fix_eval750_results.json"
    save_json(results, out_json)

    lines = [
        "# WEAVE V7 修正 DMD 符号后 750 全量评测报告\n",
        "> 修复：DMD target = y - alpha*(eps_real-eps_fake)/sigma；",
        "tanh 平滑残差；训练/推理完全统一；删除无效 Identity Loss。\n",
        "## 600 跨风格主表\n",
        "| 模型 | NFE | CSD ΔT↑ | Target Top-1↑ | Target Margin↑ | DINO-C↑ | LPIPS↓ | 5类改善 | Ukiyo Top-1 | Ukiyo ΔT |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|---:|---:|",
    ]
    for name, d in results["methods"].items():
        c = d["cross600"]
        lines.append(f"| **{name}** | **1** | **{c['mean_dT']:+.4f}** | **{c['Target_Top1']}%** | "
                     f"**{c['Target_Margin']:+.4f}** | {c['DINO_C']:.4f} | {c['LPIPS']:.4f} | "
                     f"{c['styles_improved']}/5 | {c['ukiyo_top1']}% | {c['ukiyo_dT']:+.4f} |")
    lines.append("\n## 150 同风格身份保持\n")
    lines.append("| 模型 | CSD ΔS | DINO-C | LPIPS | 潜空间位移 L2 |")
    lines.append("|---|---:|---:|---:|---:|")
    for name, d in results["methods"].items():
        s = d["same150"]
        lines.append(f"| **{name}** | {s['mean_dS']:+.4f} | {s['DINO_C']:.4f} | {s['LPIPS']:.4f} | {s['mean_latent_l2']:.4f} |")
    lines.append("\n## 结论\n")
    lines.append("报告仅陈述修正实现后的实测结果；是否替代 Calib_Balanced_s16 仍需按 Top-1/ΔT/DINO-C/Ukiyo-e 四项共同判断。\n")
    out_md = DOCS_V7 / "V7_DMD_FIX_EVALUATION_REPORT.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[V7 Fix] Saved {out_json} and {out_md}", flush=True)


if __name__ == "__main__":
    main()
