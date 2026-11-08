"""Generate V8_RESULTS.md and paired clustered bootstrap summary from Stage1 eval JSONs."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.evaluation_v2.bootstrap import paired_clustered_bootstrap_comparison  # noqa: E402
from tools.v8.common import DOCS_V8, STYLES, V8_ROOT  # noqa: E402


def load(name: str, tasks: str) -> dict:
    p = V8_ROOT / f"{name}_{tasks}_eval.json"
    if not p.exists():
        raise FileNotFoundError(p)
    return json.loads(p.read_text(encoding="utf-8"))


def main(tasks: str = "val200") -> None:
    names = ["VAE_Noop", "B0_ema", "B1_ema", "B2_ema", "B3_ema"]
    data = {}
    for n in names:
        try:
            data[n] = load(n, tasks)
        except FileNotFoundError:
            print(f"WARN missing result for {n}, skipping")
    names = [n for n in names if n in data]
    solvers = [s for s in ["euler4", "euler8", "euler16", "heun4", "heun8"] if s in data["B0_ema"]["methods"]]
    comparisons = {}
    noop_solver = next(iter(data["VAE_Noop"]["rows"].keys()))
    anchor = data["VAE_Noop"]["rows"][noop_solver]
    for n in names:
        if n == "VAE_Noop":
            continue
        comparisons[n] = {}
        for s in solvers:
            if s in data[n].get("rows", {}):
                rows = data[n]["rows"][s]
                comparisons[n][s] = {
                    "dT_vs_VAE_Noop": paired_clustered_bootstrap_comparison(rows, anchor, "csd_dT", n_boot=2000, seed=42),
                    "Top1_vs_VAE_Noop": paired_clustered_bootstrap_comparison(rows, anchor, "target_top1", n_boot=2000, seed=42),
                    "DINO_C_vs_VAE_Noop": paired_clustered_bootstrap_comparison(rows, anchor, "dino_c", n_boot=2000, seed=42),
                }
    out_json = V8_ROOT / f"stage1_{tasks}_bootstrap.json"
    out_json.write_text(json.dumps(comparisons, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = []
    lines.append(f"# WEAVE V8 Stage1 结果报告（{tasks}）\n")
    lines.append("> 完全去 Score：B0=Random+CFM, B1=Random+Brownian(σmid=0.10), B2=Content-OT+CFM, B3=Content-OT+Brownian。")
    lines.append("> 所有模型统一 800 optimizer updates、effective batch 120、相同 seed / LR schedule / 数据划分。\n")
    lines.append("## 主表\n")
    lines.append("| 模型 | Solver | NFE | ΔT↑ | ΔT_edit vs VAE No-op | Top-1↑ | Margin↑ | DINO-C↑ | LPIPS↓ | Ukiyo Top-1 | Ukiyo ΔT | 5类改善 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|")
    noop_methods = data["VAE_Noop"]["methods"]
    noop_dT = noop_methods[noop_solver]["cross"]["mean_dT"] if noop_methods else 0.0
    for n in names:
        for s in data[n]["methods"]:
            c = data[n]["methods"][s]["cross"]
            edit = c["mean_dT"] - noop_dT
            lines.append(f"| **{n}** | {s} | {c['NFE']} | **{c['mean_dT']:+.4f}** | {edit:+.4f} | **{c['Target_Top1']}%** | "
                         f"{c['Target_Margin']:+.4f} | {c['DINO_C']:.4f} | {c['LPIPS']:.4f} | "
                         f"{c['ukiyo_top1']}% | {c['ukiyo_dT']:+.4f} | {c['styles_improved']}/5 |")
    lines.append("\n## 配对聚类 Bootstrap：相对 VAE No-op 的 ΔT 增益\n")
    lines.append("| 模型 | Solver | mean_diff | 95% CI | win_rate | p_boot | 显著 |")
    lines.append("|---|---|---:|---:|---:|---:|:---:|")
    for n, d in comparisons.items():
        for s, c in d.items():
            x = c["dT_vs_VAE_Noop"]
            lines.append(f"| {n} | {s} | {x['mean_diff']:+.4f} | [{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}] | "
                         f"{x['win_rate']*100:.1f}% | {x['p_boot']:.4f} | {'YES' if x['significant_at_05'] else 'NO'} |")
    lines.append("\n## Go/No-Go\n")
    best = []
    for n, d in comparisons.items():
        for s, c in d.items():
            x = c["dT_vs_VAE_Noop"]
            if x["significant_at_05"] and x["mean_diff"] > 0:
                best.append((n, s, x["mean_diff"]))
    if best:
        lines.append(f"存在至少一个独立验证集 Pareto 候选：{sorted(best, key=lambda z: -z[2])[:5]}。\n")
    else:
        lines.append("在独立验证集上，没有模型相对 VAE No-op 取得显著正向 ΔT 增益；按文档第一阶段 Go/No-Go，暂不进入 B4/B5。\n")
    if tasks == "test750":
        lines.append("## 同风格 150 恒等兼容性\n")
        for n in names:
            for s, m in data[n]["methods"].items():
                same = m.get("same", {})
                if same:
                    lines.append(f"- {n}/{s}: CSD ΔS {same['mean_dS']:+.4f}, DINO-C {same['DINO_C']:.4f}, LPIPS {same['LPIPS']:.4f}, latent L2 {same['mean_latent_l2']:.4f}")
    out_md = DOCS_V8 / ("V8_RESULTS.md" if tasks == "test750" else f"V8_RESULTS_{tasks}.md")
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {out_json} and {out_md}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "val200")
