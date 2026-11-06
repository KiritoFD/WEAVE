"""补充测量分析：分层、配对差 CI、五类拆分、共同可行集与 5 层筛选。"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence

import numpy as np

import paths

METRICS = [
    "dT_F", "dS_F", "dM_F", "dT_R", "dS_R", "dM_R",
    "coverage", "boundary_length", "component_count", "mean_patch_distance",
    "spatial_periodicity", "q_mean", "salient_region_fraction", "alpha_sum",
    "dino_c", "dino_patch_cos", "structure_corr", "lpips", "seam_ratio",
    "outside_change", "R_active", "band_inside_frac", "band_outside_frac",
    "band_LL", "band_LH", "band_HL", "band_HH",
    "argmax_source", "argmax_donor", "csd_margin",
    "quad_tl", "quad_tr", "quad_bl", "quad_br",
]

PATTERNS = ["R64", "G64", "C64", "S64", "F64", "R128", "G128", "C128"]
POSITION_ABL = ["L1", "L2", "L3", "L4"]
COVERAGE_ABL = ["C12", "C37", "C50"]
FEATHER_ABL = ["FEATHER_4", "FEATHER_8", "FEATHER_16", "FEATHER_EFF"]
LL_ABL = ["LL_0", "LL_50", "LL_100"]
CONTROLS = ["CTRL_none", "CTRL_rand", "CTRL_same", "CTRL_destyle", "CTRL_seed"]
ORDER = PATTERNS + POSITION_ABL + COVERAGE_ABL + FEATHER_ABL + LL_ABL + ["A_P5"] + CONTROLS


def derive(row: dict) -> dict:
    dT = row["csd_T"] - row["csd_T_base"]
    dS = row["csd_S"] - row["csd_S_base"]
    out = {
        "dT_F": dT, "dS_F": dS, "dM_F": dT - dS,
        "dT_R": -dS, "dS_R": -dT, "dM_R": dT - dS,
        "coverage": row["mask"]["coverage_actual"],
        "boundary_length": row["mask"]["boundary_length"],
        "component_count": row["mask"]["component_count"],
        "mean_patch_distance": row["mask"]["mean_patch_distance"],
        "spatial_periodicity": row["mask"]["spatial_periodicity"],
        "q_mean": row["mask"]["match_q_mean"],
        "salient_region_fraction": row["mask"]["salient_region_fraction"],
        "alpha_sum": row.get("alpha_sum", 1.0),
        "dino_c": row["dino_c"],
        "dino_patch_cos": row["dino_patch_cos"],
        "structure_corr": row["structure_corr"],
        "lpips": row["lpips"],
        "seam_ratio": row["seam_ratio"],
        "outside_change": row["outside_change"],
        "R_active": row["R_active"],
        "band_inside_frac": row["band_inside_frac"],
        "band_outside_frac": row["band_outside_frac"],
        "band_LL": row["bands"]["LL"],
        "band_LH": row["bands"]["LH"],
        "band_HL": row["bands"]["HL"],
        "band_HH": row["bands"]["HH"],
        "argmax_source": float(row["argmax_style"] == row["source_style"]),
        "argmax_donor": float(row["argmax_style"] == row["target_style"]),
        "csd_margin": row["csd_T"] - row["csd_S"],
        "argmax_base_source": float(row["argmax_base"] == row["source_style"]),
    }
    quad = row["mask"].get("quadrant_coverage", {}) or {}
    for k, src in (("quad_tl", "q_tl"), ("quad_tr", "q_tr"), ("quad_bl", "q_bl"), ("quad_br", "q_br")):
        out[k] = float(quad.get(src, float("nan")))
    return out


def boot_mean(values: Sequence[float], strata: Sequence[str], rng: np.random.Generator, n_boot: int = 2000):
    vals = np.asarray(values, dtype=np.float64)
    st = np.asarray(strata)
    if len(vals) == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = []
        for s in np.unique(st):
            cur = np.where(st == s)[0]
            idx.append(rng.choice(cur, size=len(cur), replace=True))
        sel = np.concatenate(idx)
        out[b] = vals[sel].mean()
    return {
        "mean": float(vals.mean()),
        "lo": float(np.percentile(out, 2.5)),
        "hi": float(np.percentile(out, 97.5)),
        "n": int(len(vals)),
    }


def paired_diff(rows_a: Dict[str, dict], rows_b: Dict[str, dict], key: str,
                rng: np.random.Generator, n_boot: int = 2000):
    common = sorted(set(rows_a) & set(rows_b))
    if not common:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    a = np.array([rows_a[i][key] for i in common])
    b = np.array([rows_b[i][key] for i in common])
    d = a - b
    strata = np.array([rows_a[i]["source_style"] for i in common])
    out = np.empty(n_boot)
    for k in range(n_boot):
        idx = []
        for s in np.unique(strata):
            cur = np.where(strata == s)[0]
            idx.append(rng.choice(cur, size=len(cur), replace=True))
        sel = np.concatenate(idx)
        out[k] = d[sel].mean()
    return {
        "mean": float(d.mean()),
        "lo": float(np.percentile(out, 2.5)),
        "hi": float(np.percentile(out, 97.5)),
        "n": len(common),
    }


def load(paths_list: Iterable[str]) -> tuple[Dict[str, Dict[str, dict]], Dict[str, dict]]:
    by_cfg: Dict[str, Dict[str, dict]] = defaultdict(dict)
    raw = defaultdict(list)
    for p in paths_list:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            sid = r["source_id"]
            by_cfg[r["config"]][sid] = {**derive(r), "source_style": r["source_style"],
                                        "target_style": r["target_style"], "donor_id": r["donor_id"]}
            raw[r["config"]].append(r)
    return by_cfg, raw


def fmt(stat: dict, nd: int = 3) -> str:
    return f"{stat['mean']:.{nd}f} [{stat['lo']:.{nd}f}, {stat['hi']:.{nd}f}] (n={stat['n']})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True, nargs="+")
    ap.add_argument("--out", default=str(Path(paths.project_root()) / "runs/p2/supp_summary.md"))
    ap.add_argument("--json", default=str(Path(paths.project_root()) / "runs/p2/supp_summary.json"))
    args = ap.parse_args()
    by_cfg, raw = load(args.records)
    rng = np.random.default_rng(20261106)
    styles = sorted({r["source_style"] for rs in by_cfg.values() for r in rs.values()})

    summary: Dict[str, dict] = {}
    for cfg in ORDER:
        rows = by_cfg.get(cfg, {})
        if not rows:
            continue
        cur = {k: boot_mean([r[k] for r in rows.values()], [r["source_style"] for r in rows.values()], rng)
               for k in METRICS}
        per_style = {}
        for s in styles:
            sl = [r for r in rows.values() if r["source_style"] == s]
            if sl:
                per_style[s] = {k: float(np.mean([r[k] for r in sl]))
                                for k in ("dT_F", "dS_F", "dM_F", "dT_R", "dS_R", "dM_R", "csd_margin",
                                          "dino_c", "lpips", "dino_patch_cos", "seam_ratio")}
        summary[cfg] = {"stats": cur, "per_style": per_style,
                        "n": len(rows), "ids": sorted(rows)}

    # 共同可行集（分 64px 五种 / 128px 三种）
    feasible = {c: set(by_cfg.get(c, {})) for c in PATTERNS}
    c64 = set.intersection(*[feasible[c] for c in PATTERNS[:5]]) if all(feasible[c] for c in PATTERNS[:5]) else set()
    c128 = set.intersection(*[feasible[c] for c in PATTERNS[5:]]) if all(feasible[c] for c in PATTERNS[5:]) else set()
    common = c64
    summary["_feasibility"] = {
        "per_pattern_n": {c: len(feasible[c]) for c in PATTERNS},
        "common64_n": len(c64),
        "common128_n": len(c128),
        "common64_ids": sorted(c64),
        "coverage_actual_mean": {c: float(np.mean([by_cfg[c][i]["coverage"] for i in feasible[c]]))
                                 if feasible[c] else 0.0 for c in PATTERNS},
    }

    # 配对差
    pairs = []
    paid_keys = ("dT_F", "dM_F", "dino_patch_cos", "structure_corr", "seam_ratio", "dino_c")
    bases = {"R64": by_cfg.get("R64", {}), "A_P5": by_cfg.get("A_P5", {})}
    for cfg in ORDER:
        if cfg in bases:
            continue
        for bname, bmap in bases.items():
            if not bmap or not by_cfg.get(cfg):
                continue
            for key in paid_keys:
                st = paired_diff(by_cfg[cfg], bmap, key, rng)
                pairs.append({"config": cfg, "vs": bname, "metric": key, **st})
    summary["_paired"] = pairs

    # 分层：按接缝长度 / 匹配质量四分位（各 Pattern 独立可行集内）
    strata_tables = {}
    if "R64" in by_cfg:
        for cfg in PATTERNS:
            rows = by_cfg.get(cfg, {})
            if not rows:
                continue
            bl = np.array([rows[i]["boundary_length"] for i in sorted(rows)])
            qm = np.array([rows[i]["q_mean"] for i in sorted(rows)])
            bl_q = np.quantile(bl, [0.25, 0.5, 0.75])
            qm_q = np.quantile(qm, [0.25, 0.5, 0.75])
            ids = sorted(rows)
            cells = []
            for lo, hi in [(0, bl_q[0]), (bl_q[0], bl_q[1]), (bl_q[1], bl_q[2]), (bl_q[2], np.inf)]:
                sel = [i for i in ids if lo <= rows[i]["boundary_length"] <= hi]
                cells.append({"boundary_band": f"[{lo:.0f},{hi:.0f}]", "n": len(sel),
                              "dM_F": float(np.mean([rows[i]["dM_F"] for i in sel])) if sel else None,
                              "seam_ratio": float(np.mean([rows[i]["seam_ratio"] for i in sel])) if sel else None})
            strata_tables[cfg] = {"seam_bands": cells,
                                  "q_bands": [
                                      {"q_band": f"[{lo:.2f},{hi:.2f}]",
                                       "n": len([i for i in ids if lo <= rows[i]["q_mean"] <= hi]),
                                       "dM_F": float(np.mean([rows[i]["dM_F"] for i in ids if lo <= rows[i]["q_mean"] <= hi])) if any(lo <= rows[i]["q_mean"] <= hi for i in ids) else None}
                                      for lo, hi in [(0, qm_q[0]), (qm_q[0], qm_q[1]), (qm_q[1], qm_q[2]), (qm_q[2], np.inf)]]}
    summary["_strata"] = strata_tables

    # 5 层筛选（只做数据可判定的 1–4 层；第 5 层来自 P4 训练结果）
    layers = {}
    if "A_P5" in by_cfg:
        ref = by_cfg["A_P5"]
        ref_seam = float(np.mean([r["seam_ratio"] for r in ref.values()]))
        ref_dino = float(np.mean([r["dino_patch_cos"] for r in ref.values()]))
    else:
        ref_seam, ref_dino = float("inf"), -float("inf")
    for cfg in ORDER:
        rows = by_cfg.get(cfg, {})
        if not rows:
            continue
        dM = summary[cfg]["stats"]["dM_F"]
        dT = summary[cfg]["stats"]["dT_F"]
        pos_styles = sum(1 for s, v in summary[cfg]["per_style"].items() if v["dT_F"] > 0)
        layers[cfg] = {
            "L1_correctness": bool(0.9 <= summary[cfg]["stats"]["coverage"]["mean"] <= 1.1
                                   and summary[cfg]["stats"]["alpha_sum"]["mean"] > 0),
            "L2_realism": bool(summary[cfg]["stats"]["dino_patch_cos"]["mean"] >= ref_dino * 0.95
                               and summary[cfg]["stats"]["seam_ratio"]["mean"] <= max(ref_seam * 1.3, 1e-6)),
            "L3_style": bool(dT["lo"] > 0 and dM["lo"] > 0 and pos_styles >= 4),
            "L4_supervision": bool(summary[cfg]["stats"]["R_active"]["mean"] >= 0.6
                                   and summary[cfg]["stats"]["band_inside_frac"]["mean"] >= 0.5
                                   and summary[cfg]["stats"]["q_mean"]["mean"] >= 0.45),
            "positive_styles": pos_styles,
        }
    summary["_layers"] = layers

    Path(args.json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    md = ["# 补充测量汇总（自动生成）", ""]
    md.append(f"记录数：{sum(len(v) for v in by_cfg.values())}；配置数：{len(by_cfg)}")
    md.append("")
    md.append("## 可行性")
    md.append("| 配置 | 可行图数 | 实际覆盖率 |")
    md.append("|---|---|---|")
    for c, n in summary["_feasibility"]["per_pattern_n"].items():
        md.append(f"| {c} | {n} | {summary['_feasibility']['coverage_actual_mean'][c]:.4f} |")
    md.append(f"\n共同可行集（5 种 64px）：{summary['_feasibility']['common64_n']}；"
              f"共同可行集（3 种 128px）：{summary['_feasibility']['common128_n']}")
    md.append("")
    md.append("## 逐配置指标（95% CI，分风格分层 bootstrap）")
    md.append("| 配置 | n | ΔT_F | ΔS_F | ΔM | ΔT_R | ΔS_R | R_seam | R_active | inside | argmax=源 | argmax=供体 |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for cfg in ORDER:
        if cfg not in summary:
            continue
        s = summary[cfg]["stats"]
        md.append(
            f"| {cfg} | {summary[cfg]['n']} | {fmt(s['dT_F'])} | {fmt(s['dS_F'])} | {fmt(s['dM_F'])} | "
            f"{fmt(s['dT_R'])} | {fmt(s['dS_R'])} | {fmt(s['seam_ratio'], 2)} | {fmt(s['R_active'], 3)} | "
            f"{fmt(s['band_inside_frac'], 3)} | {s['argmax_source']['mean']:.2f} | {s['argmax_donor']['mean']:.2f} |"
        )
    md.append("")
    md.append("## 8 项 mask 测量")
    md.append("| 配置 | coverage | components | boundary_len | mean_dist | periodicity | q_mean | salient_frac |")
    md.append("|---|---|---|---|---|---|---|---|")
    for cfg in ORDER:
        if cfg not in summary:
            continue
        s = summary[cfg]["stats"]
        md.append(
            f"| {cfg} | {s['coverage']['mean']:.4f} | {s['component_count']['mean']:.2f} | "
            f"{s['boundary_length']['mean']:.0f} | {s['mean_patch_distance']['mean']:.3f} | "
            f"{s['spatial_periodicity']['mean']:.3f} | {s['q_mean']['mean']:.3f} | {s['salient_region_fraction']['mean']:.3f} |"
        )
    md.append("")
    md.append("## 五类逐风格 ΔT_F")
    md.append("| 配置 | " + " | ".join(styles) + " |")
    md.append("|---|" + "---|" * len(styles))
    for cfg in ORDER:
        if cfg not in summary:
            continue
        ps = summary[cfg]["per_style"]
        md.append(f"| {cfg} | " + " | ".join(f"{ps.get(s, {}).get('dT_F', float('nan')):.3f}" for s in styles) + " |")
    md.append("")
    md.append("## 五类逐风格 ΔT_R（反向，目标=真实图自身风格）")
    md.append("| 配置 | " + " | ".join(styles) + " |")
    md.append("|---|" + "---|" * len(styles))
    for cfg in ORDER:
        if cfg not in summary:
            continue
        ps = summary[cfg]["per_style"]
        md.append(f"| {cfg} | " + " | ".join(f"{ps.get(s, {}).get('dT_R', float('nan')):.3f}" for s in styles) + " |")
    md.append("")
    md.append("## 四象限覆盖")
    md.append("| 配置 | tl | tr | bl | br |")
    md.append("|---|---|---|---|---|")
    for cfg in ORDER:
        if cfg not in summary:
            continue
        s = summary[cfg]["stats"]
        md.append(f"| {cfg} | {s['quad_tl']['mean']:.3f} | {s['quad_tr']['mean']:.3f} | "
                  f"{s['quad_bl']['mean']:.3f} | {s['quad_br']['mean']:.3f} |")
    md.append("")
    md.append("## 配对差（同一图像）")
    md.append("| 配置 | vs | 指标 | 均值差 | 95% CI | n |")
    md.append("|---|---|---|---|---|---|")
    for p in pairs:
        md.append(f"| {p['config']} | {p['vs']} | {p['metric']} | {p['mean']:.4f} | "
                  f"[{p['lo']:.4f}, {p['hi']:.4f}] | {p['n']} |")
    md.append("")
    md.append("## 分层：接缝长度 / 匹配质量四分位（Pattern 独立可行集内）")
    md.append("| 配置 | 接缝Q1 ΔM | 接缝Q4 ΔM | qQ1 ΔM | qQ4 ΔM | 接缝Q4 平均 R_seam |")
    md.append("|---|---|---|---|---|---|")
    for cfg, t in strata_tables.items():
        sb, qb = t["seam_bands"], t["q_bands"]
        md.append(f"| {cfg} | {sb[0]['dM_F']:.3f} | {sb[3]['dM_F']:.3f} | {qb[0]['dM_F']:.3f} | "
                  f"{qb[3]['dM_F']:.3f} | {sb[3]['seam_ratio']:.2f} |")
    md.append("")
    md.append("## 共同可行集 vs 各 Pattern 独立可行集")
    md.append("| 配置 | 独立集 n | 独立集 ΔM | 共同集 n | 共同集 ΔM | 差异 |")
    md.append("|---|---|---|---|---|---|")
    for cfg in PATTERNS:
        rows = by_cfg.get(cfg, {})
        if not rows:
            continue
        own = float(np.mean([r["dM_F"] for r in rows.values()]))
        sel = [rows[i] for i in common if i in rows]
        cmean = float(np.mean([r["dM_F"] for r in sel])) if sel else float("nan")
        md.append(f"| {cfg} | {len(rows)} | {own:.4f} | {len(sel)} | {cmean:.4f} | {cmean - own:+.4f} |")
    md.append("")
    md.append("## 5 层筛选")
    md.append("| 配置 | L1 正确性 | L2 真实性 | L3 风格 | L4 监督 | 正风格数 |")
    md.append("|---|---|---|---|---|---|")
    for cfg, v in layers.items():
        md.append(f"| {cfg} | {v['L1_correctness']} | {v['L2_realism']} | {v['L3_style']} | "
                  f"{v['L4_supervision']} | {v['positive_styles']}/5 |")
    Path(args.out).write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[analyze] -> {args.out}")


if __name__ == "__main__":
    main()
