"""Generate V8 results report with cross-style-only paired bootstrap.

P0 review fixes implemented here:
  * bootstrap is computed on CROSS-STYLE rows only (600 of the 750 tasks); the
    150 same-style tasks are solver-identity pass-throughs and must not dilute
    the mean difference.
  * direct paired contrast B2 vs B0 is reported (does content-OT actually beat
    random coupling on style gain?), alongside every model vs VAE No-op.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.evaluation_v2.bootstrap import paired_clustered_bootstrap_comparison  # noqa: E402
from tools.v8.common import DOCS_V8, V8_ROOT  # noqa: E402

METRICS = ["csd_dT", "target_top1", "dino_c", "lpips"]


def load(name: str, tasks: str) -> dict | None:
    p = V8_ROOT / f"{name}_{tasks}_eval.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def cross_rows(entry: dict, solver: str):
    rows = entry["rows"].get(solver, [])
    return [r for r in rows if r.get("is_cross", True)]


def compare(a_rows, b_rows, tag: str) -> dict:
    return {m: paired_clustered_bootstrap_comparison(a_rows, b_rows, m, n_boot=2000, seed=42) for m in METRICS}


def main(tasks: str = "test750") -> None:
    names = ["VAE_Noop", "B0_raw", "B0_ema", "B1_ema", "B2_raw", "B2_ema", "B3_ema"]
    data = {}
    for n in names:
        d = load(n, tasks)
        if d is None:
            print(f"WARN missing {n}")
            continue
        data[n] = d
    if not data:
        raise SystemExit("no results")

    available = {n: sorted(d["rows"].keys()) for n, d in data.items()}
    print("available solvers:", available)

    ref_name = "VAE_Noop"
    ref_solver = available[ref_name][0]
    ref = cross_rows(data[ref_name], ref_solver)

    comparisons: dict = {}
    for n, d in data.items():
        if n == ref_name:
            continue
        comparisons[n] = {}
        for s in available[n]:
            rows = cross_rows(d, s)
            comparisons[n][s] = compare(rows, ref, f"{n}/{s} vs NoOp")

    # Direct contrasts that the review asked for.
    direct = {}
    for a, b in [("B2_raw", "B0_raw"), ("B2_ema", "B0_ema"), ("B2_raw", "B2_ema"), ("B0_raw", "B0_ema")]:
        if a in data and b in data:
            sa, sb = available[a][0], available[b][0]
            direct[f"{a} vs {b}"] = compare(cross_rows(data[a], sa), cross_rows(data[b], sb), f"{a} vs {b}")

    out = {
        "protocol": {
            "tasks": tasks,
            "n_cross_rows": len(ref),
            "solver_reference": ref_solver,
            "note": "paired source-clustered bootstrap on CROSS-STYLE tasks only (same-style rows excluded)",
        },
        "vs_vae_noop": comparisons,
        "direct_contrasts": direct,
    }
    (V8_ROOT / f"p0_stage1_{tasks}_bootstrap_cross.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # ---------------- markdown ----------------
    L = []
    L.append(f"# WEAVE V8 结果报告（{tasks}，跨风格口径统计修正版）\n")
    L.append("> 完全去 Score 的概率流路线。")
    L.append("> 统计口径修正：配对聚类 Bootstrap **仅使用 600 条跨风格任务**；")
    L.append("> 150 条同风格任务是求解器硬恒等直通，已从统计中排除，避免把净增益稀释为 600/750。")
    L.append("> 同时给出 B2（Content-OT）对 B0（Random）的直接配对检验。\n")

    L.append("## 一、方法主表（600 跨风格）\n")
    L.append("| 模型 | 权重 | Solver | NFE | ΔT↑ | 净增益 vs No-op | Top-1↑ | Margin↑ | DINO-C↑ | LPIPS↓ | Ukiyo ΔT | 5类改善 |")
    L.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|")
    if ref_name in data:
        c = data[ref_name]["methods"][ref_solver]["cross"]
        L.append(f"| **VAE No-op** | — | {ref_solver} | {c['NFE']} | {c['mean_dT']:+.4f} | +0.0000 | "
                 f"{c['Target_Top1']}% | {c['Target_Margin']:+.4f} | {c['DINO_C']:.4f} | {c['LPIPS']:.4f} | "
                 f"{c['ukiyo_dT']:+.4f} | {c['styles_improved']}/5 |")
    noop_dT = data[ref_name]["methods"][ref_solver]["cross"]["mean_dT"] if ref_name in data else 0.0
    for n, d in data.items():
        if n == ref_name:
            continue
        weight = "raw" if n.endswith("_raw") else "ema"
        for s in available[n]:
            c = d["methods"][s]["cross"]
            nfe = c.get("NFE", d.get("protocol", {}).get("nfe", "?"))
            L.append(f"| **{n}** | {weight} | {s} | {nfe} | **{c['mean_dT']:+.4f}** | {c['mean_dT']-noop_dT:+.4f} | "
                     f"**{c['Target_Top1']}%** | {c['Target_Margin']:+.4f} | {c['DINO_C']:.4f} | {c['LPIPS']:.4f} | "
                     f"{c['ukiyo_dT']:+.4f} | {c['styles_improved']}/5 |")

    L.append("\n## 二、相对 VAE No-op 的配对聚类 Bootstrap（仅 600 跨风格）\n")
    L.append("| 模型 | Solver | ΔT 差值 | 95% CI | 胜率 | p | 显著 | DINO-C 差值 [95% CI] | LPIPS 差值 [95% CI] |")
    L.append("|---|---|---:|---:|---:|---:|:---:|---|---|")
    for n, solvers in comparisons.items():
        for s, c in solvers.items():
            x = c["csd_dT"]; dc = c["dino_c"]; lp = c["lpips"]
            L.append(f"| {n} | {s} | {x['mean_diff']:+.4f} | [{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}] | "
                     f"{x['win_rate']*100:.1f}% | {x['p_boot']:.4f} | {'YES' if x['significant_at_05'] else 'NO'} | "
                     f"{dc['mean_diff']:+.4f} [{dc['ci95'][0]:+.4f}, {dc['ci95'][1]:+.4f}] | "
                     f"{lp['mean_diff']:+.4f} [{lp['ci95'][0]:+.4f}, {lp['ci95'][1]:+.4f}] |")

    L.append("\n## 三、直接配对对照（关键问题：OT 是否真的优于 Random？Raw 是否优于 EMA？）\n")
    L.append("| 对照 | 指标 | 差值 | 95% CI | 胜率 | p | 显著 |")
    L.append("|---|---|---:|---:|---:|---:|:---:|")
    for name, c in direct.items():
        for m, x in c.items():
            L.append(f"| {name} | {m} | {x['mean_diff']:+.4f} | [{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}] | "
                     f"{x['win_rate']*100:.1f}% | {x['p_boot']:.4f} | {'YES' if x['significant_at_05'] else 'NO'} |")

    L.append("\n## 四、同风格 150 恒等兼容性（不参与统计）\n")
    for n, d in data.items():
        s = d["methods"][ref_solver].get("same") if ref_solver in d["methods"] else None
        if s:
            L.append(f"- {n}: CSD ΔS {s['mean_dS']:+.4f}, DINO-C {s['DINO_C']:.4f}, LPIPS {s['LPIPS']:.4f}, latent L2 {s['mean_latent_l2']:.4f}")

    DOCS_V8.mkdir(parents=True, exist_ok=True)
    (DOCS_V8 / f"V8_RESULTS_{tasks}_cross.md").write_text("\n".join(L), encoding="utf-8")
    print(f"saved runs/v8/p0_stage1_{tasks}_bootstrap_cross.json and docs/v8/V8_RESULTS_{tasks}_cross.md")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "test750")
