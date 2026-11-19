"""A04: 模型强度旋钮与 Pareto 曲线分析（V25 Slot A，CPU-only，无 GPU/ssh/git）。

任务来源: docs/19_pull_together/agent2.md §四 A04。

数据源（全部本地只读）:
- docs/performance/metrics.csv          19 条实测臂的唯一数值来源（读数 cross600, n=600）
- runs/v25/r1_inference_ablation.json   曲线 B tau_records（τ_min 实测配置与分布）
- runs/v25/r1b_unlock_arms.json         B_d6_t070 / B_d6_t072 tau_records
- runs/v25/r3_oneshot_minimal.json      O3 tau_curve（τ0 + wj）与 V24 复用臂读数

输出（本任务文件所有权）:
- docs/v25/audits/a04_strength_curves.json   每臂读数 / 分段斜率 / 单调性 / 饱和区 / strength 映射 / Pareto
- docs/v25/audits/a04_strength_pareto.svg/.png  双面板: (DINO-C, DINO-S) 与 (DINO-C, Top-1) 局部前沿
- docs/v25/audits/a04_strength_report.md     分析报告

运行:
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= \
        python -m tools.v25_cpu.a04_strength_analysis

硬约束: 无绝对路径（Path(__file__).resolve().parents[2]）; matplotlib Agg; 线程数 2;
禁止外推虚构配对（A01 验收第 5 条）——本脚本只使用 metrics.csv 实测臂，插值一律标注
descriptive only，映射表外推区标注 UNMEASURED。
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
METRICS_CSV = REPO_ROOT / "docs" / "performance" / "metrics.csv"
R1_JSON = REPO_ROOT / "runs" / "v25" / "r1_inference_ablation.json"
R1B_JSON = REPO_ROOT / "runs" / "v25" / "r1b_unlock_arms.json"
R3_JSON = REPO_ROOT / "runs" / "v25" / "r3_oneshot_minimal.json"
OUT_DIR = REPO_ROOT / "docs" / "v25" / "audits"
JSON_OUT = OUT_DIR / "a04_strength_curves.json"
SVG_OUT = OUT_DIR / "a04_strength_pareto.svg"
PNG_OUT = OUT_DIR / "a04_strength_pareto.png"
MD_OUT = OUT_DIR / "a04_strength_report.md"

METRICS = ("dino_c", "dino_s", "lpips", "target_top1")
# 饱和判停阈值（本审计预注册）: 连续两点 |Δ| < 阈值 → 该段标注饱和。
# Top-1 阈值 0.005 = 3 个任务（1/600 ≈ 0.00167/任务），低于此差分在计数噪声内。
SAT_THRESHOLDS = {"dino_c": 0.002, "dino_s": 0.002, "lpips": 0.002, "target_top1": 0.005}
TOP1_TASK_RESOLUTION = 1.0 / 600.0
LPIPS_MAX = 0.51  # 任务指定的 LPIPS 约束线（≈ PRISM-1 λ=1.0 锚臂 0.5057 的操作包络）
NAIVE_S_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)

FAMILIES: list[dict[str, Any]] = [
    {
        "key": "prism1_lambda",
        "engine": "PRISM-1 (1.5M 单步前馈, 17 ms/img)",
        "interface": "PRISMCore.transfer(strength)",
        "knob": "lambda",
        "knob_symbol": "lambda",
        "direction": "direct",  # 旋钮↑ = 编辑更强 (DINO-C↓)
        "color": "#1f77b4",
        "short": "l",
        "arms": [
            ("M0_prism1_l075", 0.75, "V24-P0-M0M2 复用臂"),
            ("M0_prism1_l100", 1.00, "V24-P0-M0M2 复用臂"),
            ("A_p1_l130", 1.30, "V25-R1 实跑"),
            ("A_p1_l160", 1.60, "V25-R1 实跑"),
        ],
        "notes": "λ 为加法风格注入系数；实测区间 [0.75, 1.60]。",
    },
    {
        "key": "d6_tau_min",
        "engine": "V20 D6 SLFUNet 分段视界流 (22M, T4 锚 + full CFG, tau_hi=0.78)",
        "interface": "PRISMCore.transfer(strength)",
        "knob": "tau_min",
        "knob_symbol": "tau_min",
        "direction": "inverse",  # 旋钮↑ = 编辑更弱 (DINO-C↑) —— 与 λ/γ 方向相反
        "color": "#d62728",
        "short": "tmin",
        "arms": [
            ("M0_v20_d6", 0.40, "V24-P0-M0M2 复用臂; 04 文档记录 τ∈[0.40,0.74], 无 tau_record"),
            ("B_d6_t050", 0.50, "V25-R1 实跑; tau_records"),
            ("B_d6_t057", 0.58, "V25-R1 实跑; tau_records (臂名 057, tau_cfg=0.58, 以记录为准)"),
            ("B_d6_t066", 0.66, "V25-R1 实跑; tau_records"),
            ("B_d6_t070", 0.70, "V25-R1b 实跑; tau_records"),
            ("B_d6_t072", 0.72, "V25-R1b 实跑; tau_records"),
        ],
        "notes": "τ_min 为采样带下界（tau_hi 固定 0.78）；实测 τ_min↑ → DINO-C↑（更保守）。",
    },
    {
        "key": "hybrid_gamma",
        "engine": "Hybrid 线性混合 (PRISM-1 λ=1.0 + V20 D6 双模型串行)",
        "interface": "PRISMCore.transfer(strength)",
        "knob": "gamma",
        "knob_symbol": "gamma",
        "direction": "direct",
        "color": "#2ca02c",
        "short": "g",
        "arms": [
            ("M1_lin_g25", 0.25, "V24-P0-M0M2 复用臂"),
            ("M1_lin_g40", 0.40, "V24-P0-M0M2 复用臂"),
            ("M1_lin_g55", 0.55, "V24-P0-M0M2 复用臂"),
            ("C_hyb_g70", 0.70, "V25-R1 实跑"),
            ("C_hyb_g85", 0.85, "V25-R1 实跑"),
        ],
        "notes": "γ 为 D6 分支线性混合权重；实测区间 [0.25, 0.85]。V24 复用臂与 V25 实跑臂同族定义（04 文档曲线 C）。",
    },
    {
        "key": "o3_tau0",
        "engine": "O3 One-Shot Minimal (RGB+ConvNeXt gated, 参考图条件)",
        "interface": "PRISMRef.transfer(strength)",
        "knob": "tau0",
        "knob_symbol": "tau0",
        "direction": "inverse",  # 旋钮↑ = 编辑更弱 (DINO-C↑)
        "color": "#9467bd",
        "short": "t0",
        "arms": [
            ("O3_rgb_cnx_gated_t40", 0.40, "V25-R3 tau_curve 实测 (wj=2.3)"),
            ("O3_rgb_cnx_gated_t45", 0.45, "runs/v24 复用臂 (r3 JSON reused; JSON 无 tau0/wj 记录)"),
            ("O3_rgb_cnx_gated_t55", 0.55, "runs/v24 复用臂 (r3 JSON reused; JSON 无 tau0/wj 记录)"),
            ("O3_rgb_cnx_gated_t60", 0.60, "V25-R3 tau_curve 实测 (wj=1.9)"),
        ],
        "notes": "τ0↑ → DINO-C↑（更保守）。混杂: t40/t60 的内容权重 wj 不同（2.3 vs 1.9），曲线非纯 τ0 单变量。",
    },
]


def r5(x: float | None) -> float | None:
    return None if x is None else round(float(x), 5)


def load_metrics_rows() -> dict[str, dict[str, str]]:
    with open(METRICS_CSV, encoding="utf-8-sig", newline="") as f:
        rows = {}
        for row in csv.DictReader(f):
            arm = row.get("arm", "").strip()
            if arm:
                rows[arm] = row
    return rows


def load_tau_records() -> tuple[dict[str, Any], dict[str, Any]]:
    with open(R1_JSON, encoding="utf-8") as f:
        r1 = json.load(f)
    with open(R1B_JSON, encoding="utf-8") as f:
        r1b = json.load(f)
    merged = {}
    merged.update(r1.get("tau_records", {}))
    merged.update(r1b.get("tau_records", {}))
    return merged, {"r1": r1.get("tau_records", {}), "r1b": r1b.get("tau_records", {})}


def load_r3() -> dict[str, Any]:
    with open(R3_JSON, encoding="utf-8") as f:
        return json.load(f)


def build_arm_records() -> tuple[dict[str, dict[str, Any]], list[str]]:
    """从 metrics.csv + tau_records + r3 tau_curve 组装 19 条臂的统一读数与元数据。"""
    rows = load_metrics_rows()
    tau_recs, tau_sources = load_tau_records()
    r3 = load_r3()

    tau_curve = {e["arm"]: e for e in r3.get("tau_curve", [])}
    reused = {e["arm"]: e for e in (r3.get("reused") or {}).values()}

    records: dict[str, dict[str, Any]] = {}
    problems: list[str] = []
    for fam in FAMILIES:
        for arm, knob, provenance in fam["arms"]:
            row = rows.get(arm)
            if row is None:
                problems.append(f"{fam['key']}/{arm}: metrics.csv 缺失")
                continue
            if row.get("n_cross", "").strip() != "600":
                problems.append(f"{fam['key']}/{arm}: n_cross={row.get('n_cross')!r} != 600")
            try:
                vals = {m: float(row[m]) for m in METRICS}
            except (KeyError, ValueError) as exc:
                problems.append(f"{fam['key']}/{arm}: 指标缺失/非数值 ({exc})")
                continue
            if any(math.isnan(v) for v in vals.values()):
                problems.append(f"{fam['key']}/{arm}: 指标含 NaN")
                continue
            rec: dict[str, Any] = {
                "arm": arm,
                "family": fam["key"],
                "knob": fam["knob"],
                "knob_value": knob,
                "knob_direction": fam["direction"],
                "group": row.get("group", ""),
                "provenance": provenance,
                **{m: r5(v) for m, v in vals.items()},
                "style_affinity": r5(float(row["style_affinity"])) if row.get("style_affinity") else None,
                "clip_s": r5(float(row["clip_s"])) if row.get("clip_s") else None,
                "eval_source": row.get("eval_source", ""),
                "tau_meta": None,
            }
            # tau 元数据 + 旋钮值交叉校验
            if arm in tau_recs:
                tr = tau_recs[arm]
                cfg_lo = tr.get("tau_cfg", [None])[0]
                if cfg_lo is None or abs(float(cfg_lo) - knob) > 1e-9:
                    problems.append(
                        f"{fam['key']}/{arm}: 旋钮值 {knob} 与 tau_cfg {cfg_lo} 不一致"
                    )
                rec["tau_meta"] = {
                    "source": "runs/v25 tau_records",
                    "tau_cfg": tr.get("tau_cfg"),
                    "tau_mean": r5(tr.get("tau_mean")),
                    "tau_p50": r5(tr.get("tau_p50")),
                    "tau_max": r5(tr.get("tau_max")),
                    "frac_at_cap": tr.get("frac_at_cap"),
                    "branches": tr.get("branches"),
                    "gate": tr.get("gate"),
                }
            if arm in tau_curve:
                tc = tau_curve[arm]
                if abs(float(tc["tau0"]) - knob) > 1e-9:
                    problems.append(
                        f"{fam['key']}/{arm}: 旋钮值 {knob} 与 r3 tau_curve tau0 {tc['tau0']} 不一致"
                    )
                rec["tau_meta"] = {
                    "source": "runs/v25/r3_oneshot_minimal.json tau_curve",
                    "tau0": r5(tc["tau0"]),
                    "wj": tc.get("wj"),
                }
            if arm in reused:
                ru = reused[arm]
                for k_json, k_csv in (
                    ("dino_c", "dino_c"),
                    ("dino_s", "dino_s"),
                    ("lpips", "lpips"),
                    ("style_affinity", "style_affinity"),
                    ("csd_top1", "target_top1"),
                ):
                    if k_json in ru and ru[k_json] is not None:
                        if abs(float(ru[k_json]) - vals.get(k_csv, float("nan"))) > 1e-6:
                            problems.append(
                                f"{fam['key']}/{arm}: r3 reused {k_json}={ru[k_json]} 与 csv {k_csv} 不一致"
                            )
            records[arm] = rec
    if problems:
        raise ValueError("A04 数据校验失败:\n  " + "\n  ".join(problems))
    if len(records) != sum(len(f["arms"]) for f in FAMILIES):
        raise ValueError("A04: 组装的臂数与定义不符")
    return records, list(tau_sources.keys())


# ---------------------------------------------------------------- 曲线分析


def analyse_family(fam: dict[str, Any], records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    arms = [records[a] for a, _, _ in fam["arms"]]
    arms_by_knob = sorted(arms, key=lambda r: r["knob_value"])
    direction = fam["direction"]

    # ---- 自然旋钮序分段（knob 升序）
    segments = []
    for a, b in zip(arms_by_knob, arms_by_knob[1:]):
        seg = {
            "from_arm": a["arm"],
            "to_arm": b["arm"],
            "knob_range": [a["knob_value"], b["knob_value"]],
            "d_knob": r5(b["knob_value"] - a["knob_value"]),
        }
        for m in METRICS:
            seg["d_" + m] = r5(b[m] - a[m])
        dc = seg["d_dino_c"]
        ds = seg["d_dino_s"]
        seg["slope_deltaS_over_deltaC"] = r5(ds / dc) if abs(dc) > 1e-12 else None
        # 强化方向效率: ΔS / |ΔC|（ΔC 取强化方向的绝对内容代价）
        sign = 1.0 if direction == "direct" else -1.0
        d_c_strength = sign * dc  # 强化方向上应为负（内容下降）
        d_s_strength = sign * ds
        d_t_strength = sign * seg["d_target_top1"]
        d_l_strength = sign * seg["d_lpips"]
        if d_c_strength < -1e-12:
            seg["strength_direction"] = "strengthening"
            seg["efficiency_dS_per_content"] = r5(d_s_strength / (-d_c_strength))
            seg["efficiency_dTop1_per_content"] = r5(d_t_strength / (-d_c_strength))
            seg["lpips_cost_per_content"] = r5(d_l_strength / (-d_c_strength))
        else:
            seg["strength_direction"] = "anomalous (内容未按强化方向下降)"
            seg["efficiency_dS_per_content"] = None
        saturated = {}
        for m in METRICS:
            saturated[m] = abs(seg["d_" + m]) < SAT_THRESHOLDS[m]
        seg["saturated"] = saturated
        segments.append(seg)

    # ---- 单调性（自然旋钮序逐段差分方向）
    monotonicity = {}
    for m in METRICS:
        diffs = [s["d_" + m] for s in segments]
        eps = 1e-12
        inc = all(d >= -eps for d in diffs)
        dec = all(d <= eps for d in diffs)
        violations = []
        if inc:
            overall = "increasing"
        elif dec:
            overall = "decreasing"
        else:
            overall = "non-monotone"
            ref = 1.0 if diffs[0] >= 0 else -1.0
            for s, d in zip(segments, diffs):
                if d * ref < -eps:
                    violations.append(
                        {
                            "segment": [s["from_arm"], s["to_arm"]],
                            "d": r5(d),
                            "note": f"与整体方向({'up' if ref > 0 else 'down'})相反",
                        }
                    )
        monotonicity[m] = {
            "overall_direction": overall,
            "is_monotone": overall != "non-monotone",
            "diffs": [r5(d) for d in diffs],
            "violations": violations,
            "allow_equal": True,
        }

    # ---- 饱和区（连续两点 |Δ| < 阈值）
    sat_zones = []
    for s in segments:
        for m in METRICS:
            if s["saturated"][m]:
                sat_zones.append(
                    {
                        "segment": [s["from_arm"], s["to_arm"]],
                        "knob_range": s["knob_range"],
                        "metric": m,
                        "abs_delta": r5(abs(s["d_" + m])),
                        "threshold": SAT_THRESHOLDS[m],
                    }
                )

    # ---- 强化方向排序（weakest edit → strongest edit, 按 DINO-C 降序）
    strength_arms = sorted(arms, key=lambda r: -r["dino_c"])
    # 映射前置条件: DINO-C 对旋钮严格单调（否则内容锚定映射不可逆）
    c_seq = [r["dino_c"] for r in arms_by_knob]
    c_strict_mono = all(b > a for a, b in zip(c_seq, c_seq[1:])) or all(
        b < a for a, b in zip(c_seq, c_seq[1:])
    )
    expected_knob_order = (
        [r["knob_value"] for r in strength_arms]
        if direction == "direct"
        else [r["knob_value"] for r in strength_arms][::-1]
    )
    knob_sorted_strength = sorted(expected_knob_order)
    knob_order_consistent = expected_knob_order == knob_sorted_strength
    if not c_strict_mono:
        raise ValueError(f"{fam['key']}: DINO-C 对旋钮非严格单调，内容锚定 strength 映射不可逆")
    if not knob_order_consistent:
        raise ValueError(f"{fam['key']}: 强化方向与旋钮方向不一致，映射失效")

    # ---- 最有效工作区间（强化方向 efficiency_dS_per_content 最大段）+ 内容代价无风格收益段
    # 只统计 |ΔC| ≥ 饱和阈值的段：双饱和段（ΔC、ΔS 都低于阈值）的效率比值是噪声, 不参与评选。
    eff_segs = [
        s for s in segments if s.get("strength_direction") == "strengthening"
        and s.get("efficiency_dS_per_content") is not None
        and abs(s["d_dino_c"]) >= SAT_THRESHOLDS["dino_c"]
    ]
    most_effective = max(eff_segs, key=lambda s: s["efficiency_dS_per_content"]) if eff_segs else None
    dead_segments = [
        {
            "segment": [s["from_arm"], s["to_arm"]],
            "knob_range": s["knob_range"],
            "d_dino_s_abs": r5(abs(s["d_dino_s"])),
            "d_dino_c_abs": r5(abs(s["d_dino_c"])),
            "note": "风格增量低于饱和阈值而内容代价高于阈值: 纯内容代价、无风格收益",
        }
        for s in segments
        if s.get("strength_direction") == "strengthening"
        and s["saturated"]["dino_s"]
        and abs(s["d_dino_c"]) >= SAT_THRESHOLDS["dino_c"]
    ]

    # ---- strength 映射（内容锚定, 推荐）: s = (C_weakest − C(knob)) / (C_weakest − C_strongest)
    c_weakest = strength_arms[0]["dino_c"]
    c_strongest = strength_arms[-1]["dino_c"]
    span = c_weakest - c_strongest
    knots = []
    for r in strength_arms:
        knots.append(
            {
                "arm": r["arm"],
                "knob_value": r["knob_value"],
                "strength": r5((c_weakest - r["dino_c"]) / span),
                "dino_c": r["dino_c"],
                "dino_s": r["dino_s"],
                "lpips": r["lpips"],
                "target_top1": r["target_top1"],
            }
        )
    map_segments = []
    for k0, k1 in zip(knots, knots[1:]):
        map_segments.append(
            {
                "strength_range": [k0["strength"], k1["strength"]],
                "knob_range": [k0["knob_value"], k1["knob_value"]],
                "evidence_arms": [k0["arm"], k1["arm"]],
                "kind": "interpolation_measured",
                "descriptive_only": False,
                "endpoint_metrics": {"from": k0, "to": k1},
            }
        )
    # 外推占位段（UNMEASURED, 不推荐）
    last0, last1 = knots[-2], knots[-1]
    slope = (last1["knob_value"] - last0["knob_value"]) / max(
        last1["strength"] - last0["strength"], 1e-12
    )
    first0, first1 = knots[0], knots[1]
    slope_lo = (first1["knob_value"] - first0["knob_value"]) / max(
        first1["strength"] - first0["strength"], 1e-12
    )
    extension = [
        {
            "strength_range": [1.0, 1.2],
            "knob_at_1.2": r5(last1["knob_value"] + 0.2 * slope),
            "kind": "UNMEASURED_EXTRAPOLATION",
            "recommended": False,
            "note": "超出实测旋钮区间，无任何实测臂支持；默认行为应为 clamp 到 s=1 端点",
        },
        {
            "strength_range": [-0.2, 0.0],
            "knob_at_-0.2": r5(first0["knob_value"] - 0.2 * slope_lo),
            "kind": "UNMEASURED_EXTRAPOLATION",
            "recommended": False,
            "note": "超出实测旋钮区间，无任何实测臂支持；默认行为应为 clamp 到 s=0 端点",
        },
    ]
    # 朴素均匀映射（备选）: knob = kmin + s*(kmax-kmin), 指标为实测曲线分段线性插值
    kmin = min(r["knob_value"] for r in arms)
    kmax = max(r["knob_value"] for r in arms)
    naive = []
    for s in NAIVE_S_GRID:
        k = kmin + s * (kmax - kmin)
        naive.append(
            {
                "strength": s,
                "knob_value": r5(k),
                **interp_metrics(arms_by_knob, k),
            }
        )
    # 默认档 s=0.5 的内容锚定映射读数
    mid = strength_at(knots, 0.5)

    return {
        "key": fam["key"],
        "engine": fam["engine"],
        "interface": fam["interface"],
        "knob": fam["knob"],
        "direction": direction,
        "direction_note": (
            "旋钮值越大编辑越强"
            if direction == "direct"
            else "旋钮值越大编辑越弱（与 λ/γ 方向相反; 用户 strength 与旋钮反向）"
        ),
        "measured_knob_range": [kmin, kmax],
        "arms": [
            {
                k: rec[k]
                for k in (
                    "arm",
                    "knob",
                    "knob_value",
                    "group",
                    "provenance",
                    "dino_c",
                    "dino_s",
                    "lpips",
                    "target_top1",
                    "style_affinity",
                    "clip_s",
                    "eval_source",
                    "tau_meta",
                )
            }
            for rec in arms_by_knob
        ],
        "segments_natural_knob_order": segments,
        "monotonicity": monotonicity,
        "saturation_zones": sat_zones,
        "saturation_thresholds": SAT_THRESHOLDS,
        "most_effective_interval": {
            "segment": [most_effective["from_arm"], most_effective["to_arm"]],
            "knob_range": most_effective["knob_range"],
            "efficiency_dS_per_content": most_effective["efficiency_dS_per_content"],
            "criterion": "强化方向 ΔDINO-S / |ΔDINO-C| 最大的实测段; 仅统计 |ΔC| ≥ 0.002 的段（双饱和段的效率比值属噪声, 不参与评选）",
        }
        if most_effective
        else None,
        "dead_segments": dead_segments,
        "strength_order_arms": [r["arm"] for r in strength_arms],
        "strength_mapping": {
            "convention": "strength s∈[0,1]: 0=最弱编辑(内容保持最高实测臂), 1=最强编辑(风格最高实测臂); 内容锚定: s=(C_weakest−C)/span",
            "anchor_metric": "dino_c",
            "knots_measured": knots,
            "segments": map_segments,
            "beyond_measured": extension,
            "clamp_policy": "s<0 → clamp 至最弱实测端点; s>1 → clamp 至最强实测端点（超出区间 UNMEASURED）",
            "naive_uniform_alternative": {
                "formula": "knob = kmin + s*(kmax-kmin)",
                "table": naive,
                "descriptive_only": True,
                "note": "旋钮均匀刻度; 指标为实测曲线分段线性插值, 仅描述性",
            },
            "strength_at_0.5": mid,
            "notes": fam["notes"],
        },
    }


def interp_metrics(arms_by_knob: list[dict[str, Any]], knob: float) -> dict[str, Any]:
    """在 (knob → metric) 实测曲线上做分段线性插值（descriptive only）。"""
    out: dict[str, Any] = {"descriptive_only": True, "bracket_arms": None}
    for a, b in zip(arms_by_knob, arms_by_knob[1:]):
        if a["knob_value"] <= knob <= b["knob_value"]:
            t = (knob - a["knob_value"]) / (b["knob_value"] - a["knob_value"])
            out["bracket_arms"] = [a["arm"], b["arm"]]
            for m in METRICS:
                out[m] = r5(a[m] + t * (b[m] - a[m]))
            return out
    # 越界: 不插值, 标 UNMEASURED
    out["value"] = "UNMEASURED (knob 超出实测区间)"
    return out


def strength_at(knots: list[dict[str, Any]], s: float) -> dict[str, Any]:
    """内容锚定映射在给定 s 处的旋钮值（ knots 的 strength 已升序）。"""
    for k in knots:
        if abs(k["strength"] - s) < 1e-9:
            return {"strength": s, "knob_value": k["knob_value"], "arm": k["arm"], "measured": True}
    for k0, k1 in zip(knots, knots[1:]):
        if k0["strength"] < s < k1["strength"]:
            t = (s - k0["strength"]) / (k1["strength"] - k0["strength"])
            knob = k0["knob_value"] + t * (k1["knob_value"] - k0["knob_value"])
            return {
                "strength": s,
                "knob_value": r5(knob),
                "bracket_arms": [k0["arm"], k1["arm"]],
                "measured": False,
                "descriptive_only": True,
            }
    return {"strength": s, "value": "UNMEASURED", "descriptive_only": False}


# ---------------------------------------------------------------- Pareto


def dominates(a: dict[str, Any], b: dict[str, Any], xa: str, ya: str) -> bool:
    """a 支配 b（两维均最大化）: a >= b 且至少一维严格 >。"""
    ge = a[xa] >= b[xa] and a[ya] >= b[ya]
    strict = a[xa] > b[xa] or a[ya] > b[ya]
    return ge and strict


# 亚噪声边距阈值: 支配边距低于此值时, 该维度的"严格更优"属测量分辨率内的贴平,
# 前沿成员资格对测量噪声敏感（仅作信息标注, 不改变严格支配定义）。
NOISE_EPS = {"dino_c": 0.0005, "dino_s": 0.0005, "target_top1": 0.005, "lpips": 0.0005}


def frontier(
    pts: list[dict[str, Any]], xa: str, ya: str
) -> tuple[list[str], list[str], dict[str, list[dict[str, Any]]]]:
    front, dom = [], []
    details: dict[str, list[dict[str, Any]]] = {}
    ex, ey = NOISE_EPS[xa], NOISE_EPS[ya]
    for p in pts:
        dominators = []
        for q in pts:
            if q is p:
                continue
            if dominates(q, p, xa, ya):
                dx = q[xa] - p[xa]
                dy = q[ya] - p[ya]
                dominators.append(
                    {
                        "arm": q["arm"],
                        "d_x": r5(dx),
                        "d_y": r5(dy),
                        "sub_noise_margin": (dx < ex) or (dy < ey),
                    }
                )
        if dominators:
            dom.append(p["arm"])
            details[p["arm"]] = dominators
        else:
            front.append(p["arm"])
    return front, dom, details


def compute_pareto(records: dict[str, Any]) -> dict[str, Any]:
    all_arms = list(records.values())
    out: dict[str, Any] = {}
    for tag, xa, ya in (
        ("cs", "dino_c", "dino_s"),
        ("ctop1", "dino_c", "target_top1"),
    ):
        per_family = {}
        for fam in FAMILIES:
            pts = [records[a] for a, _, _ in fam["arms"]]
            front, dom, det = frontier(pts, xa, ya)
            per_family[fam["key"]] = {
                "frontier": front,
                "dominated": dom,
                "dominated_details": det,
                "frontier_lpips_feasible": [
                    a for a in front if records[a]["lpips"] <= LPIPS_MAX
                ],
            }
        gf, gd, gdet = frontier(all_arms, xa, ya)
        feas = [p for p in all_arms if p["lpips"] <= LPIPS_MAX]
        ff, _, _ = frontier(feas, xa, ya)
        out[tag] = {
            "axes": {"x": xa, "y": ya, "direction": "both maximize"},
            "noise_eps": {"x": NOISE_EPS[xa], "y": NOISE_EPS[ya]},
            "domination_rule": "严格支配（点估计, 两维 >= 且至少一维严格 >）; dominated_details 中 sub_noise_margin=true 表示该支配关系的某一维边距低于噪声阈值（信息标注, 不改变前沿定义）",
            "per_family": per_family,
            "global": {
                "frontier": sorted(gf),
                "dominated": sorted(gd),
                "dominated_details": gdet,
                "frontier_members_violating_lpips": sorted(
                    a for a in gf if records[a]["lpips"] > LPIPS_MAX
                ),
            },
            "global_lpips_constraint": {
                "lpips_max": LPIPS_MAX,
                "feasible_arms": sorted(p["arm"] for p in feas),
                "frontier_of_feasible_subset": sorted(ff),
            },
        }
    return out


# ---------------------------------------------------------------- 图


def make_figure(analyses: dict[str, dict[str, Any]], records: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_arms = list(records.values())
    pareto_res = compute_pareto(records)
    global_front = {
        tag: set(pareto_res[tag]["global"]["frontier"]) for tag in ("cs", "ctop1")
    }
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.8))
    panels = [
        ("cs", "dino_c", "dino_s", "DINO-C (content, higher = closer to source)", "DINO-S (style, higher = closer to reference)",
         "V25 A04: Content-Style knob curves & Pareto (measured arms, cross600)"),
        ("ctop1", "dino_c", "target_top1", "DINO-C (content, higher = closer to source)", "CSD Target Top-1 (%)",
         "V25 A04: Content vs CSD Top-1 (measured arms, cross600)"),
    ]
    label_offsets = {
        "cs": {
            "prism1_lambda": (5, -11),
            "d6_tau_min": (7, 3),
            "hybrid_gamma": (7, 5),
            "o3_tau0": (7, 2),
        },
        "ctop1": {
            "prism1_lambda": (6, -10),
            "d6_tau_min": (-52, 2),
            "hybrid_gamma": (7, 5),
            "o3_tau0": (7, 2),
        },
    }
    for ax, (tag, xa, ya, xlabel, ylabel, title) in zip(axes, panels):
        for fam in FAMILIES:
            pts = [records[a] for a, _, _ in fam["arms"]]
            pts_sorted = sorted(pts, key=lambda r: r["dino_c"])
            # 连接线: 实测点之间的插值参考 → 虚线 + descriptive only
            ax.plot(
                [p[xa] for p in pts_sorted],
                [p[ya] * 100 if ya == "target_top1" else p[ya] for p in pts_sorted],
                ls="--", lw=0.9, color=fam["color"], alpha=0.65, zorder=1,
            )
            for i, p in enumerate(pts):
                feas = p["lpips"] <= LPIPS_MAX
                on_front = p["arm"] in global_front[tag]
                x, y = p[xa], p[ya] * 100 if ya == "target_top1" else p[ya]
                ax.scatter(
                    [x], [y],
                    s=72 if on_front else 46,
                    marker="o",
                    facecolor=fam["color"] if feas else "none",
                    edgecolor="black" if on_front else fam["color"],
                    linewidths=1.6 if on_front else 1.0,
                    zorder=3,
                )
                dx0, dy0 = label_offsets[tag][fam["key"]]
                label = f"{fam['short']}{p['knob_value']:g}"
                ax.annotate(
                    label, (x, y), textcoords="offset points",
                    xytext=(dx0, dy0 + (i % 2) * 12),
                    fontsize=6.3, color=fam["color"] if on_front else "#777777",
                    fontweight="bold" if on_front else "normal", zorder=4,
                )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10.5)
        ax.grid(True, ls=":", lw=0.5, alpha=0.5)
    handles = []
    labels = []
    for fam in FAMILIES:
        h = axes[0].scatter([], [], s=46, marker="o", facecolor=fam["color"], edgecolor=fam["color"])
        handles.append(h)
        labels.append(f"{fam['key']} ({fam['knob']})")
    h = axes[0].scatter([], [], s=46, marker="o", facecolor="none", edgecolor="#555555")
    handles.append(h)
    labels.append(f"open marker: LPIPS > {LPIPS_MAX} (constraint violated)")
    h = axes[0].scatter([], [], s=72, marker="o", facecolor="#bbbbbb", edgecolor="black", linewidths=1.6)
    handles.append(h)
    labels.append("black edge: global non-dominated (this panel)")
    h = axes[0].plot([], [], ls="--", lw=0.9, color="#555555")[0]
    handles.append(h)
    labels.append("dashed connector: descriptive only (measured points linked)")
    axes[0].legend(handles, labels, loc="upper left", fontsize=6.6, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(SVG_OUT, format="svg")
    fig.savefig(PNG_OUT, format="png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 报告


def pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def make_report(analyses: dict[str, dict[str, Any]], pareto: dict[str, Any], records: dict[str, Any], now_utc: str) -> str:
    L: list[str] = []
    ap = L.append
    ap("# A04 模型强度旋钮与 Pareto 曲线分析（2026-11-20）")
    ap("")
    ap(f"> 生成: {now_utc}（`tools/v25_cpu/a04_strength_analysis.py`, CPU-only）。")
    ap("> 任务来源: [agent2.md §四 A04](../../19_pull_together/agent2.md)。数据: [metrics.csv](../../performance/metrics.csv) 19 条实测臂（读数 cross600, n=600）+ [r1_inference_ablation.json](../../../runs/v25/r1_inference_ablation.json) / [r1b_unlock_arms.json](../../../runs/v25/r1b_unlock_arms.json) tau_records + [r3_oneshot_minimal.json](../../../runs/v25/r3_oneshot_minimal.json) tau_curve。全部数值为实测臂读数；任何插值均标注 **descriptive only**，映射表外推区标注 **UNMEASURED**（A01 验收第 5 条口径，见 [a01_round1_ci.md](./a01_round1_ci.md)）。")
    ap("> 数值输出: [a04_strength_curves.json](./a04_strength_curves.json)；图: [a04_strength_pareto.svg](./a04_strength_pareto.svg) / [a04_strength_pareto.png](./a04_strength_pareto.png)。")
    ap("")
    ap("## 一、协议与阈值（预注册）")
    ap("")
    ap("- 四条旋钮 = 四个引擎族的全部实测臂（19 条），禁止虚构配对；每臂 `n_cross=600`。")
    ap(f"- 饱和判停: 连续两点 |Δ| < 阈值 → 标注饱和。阈值: DINO-C/DINO-S/LPIPS `< 0.002`，Top-1 `< 0.005`（= 3 个任务；1 任务 = 1/600 ≈ {TOP1_TASK_RESOLUTION:.5f} = 0.167pp，Top-1 差分低于 3 任务在计数噪声内）。")
    ap(f"- Pareto LPIPS 约束线: `LPIPS ≤ {LPIPS_MAX}`（任务指定，≈ PRISM-1 λ=1.0 锚臂 0.5057 的操作包络，仅作描述性筛）。")
    ap("- 方向约定: 用户 `strength s∈[0,1]`，0 = 最弱编辑（DINO-C 最高实测臂），1 = 最强编辑（DINO-S 最高实测臂）。**D6 τ_min 与 O3 τ0 是反向旋钮**（旋钮↑ = 编辑更弱）。")
    ap("")
    for fam in FAMILIES:
        a = analyses[fam["key"]]
        ap(f"## 二.{list(analyses).index(fam['key']) + 1} {fam['key']}（{fam['engine']}）")
        ap("")
        ap(f"旋钮 `{fam['knob']}`，实测区间 [{a['measured_knob_range'][0]:g}, {a['measured_knob_range'][1]:g}]。{a['direction_note']}。")
        ap("")
        ap("| 臂 | 旋钮值 | DINO-C↑ | DINO-S↑ | LPIPS↓ | Top-1↑ | Aff↑ | 来源 |")
        ap("|---|---:|---:|---:|---:|---:|---:|---|")
        for arm_rec in a["arms"]:
            ap(
                f"| `{arm_rec['arm']}` | {arm_rec['knob_value']:g} | {arm_rec['dino_c']:.5f} | {arm_rec['dino_s']:.5f} | "
                f"{arm_rec['lpips']:.5f} | {pct(arm_rec['target_top1'])} | {arm_rec['style_affinity']:.5f} | {arm_rec['group']} |"
            )
        ap("")
        ap("分段（旋钮自然序）:")
        ap("")
        ap("| 段 | ΔC | ΔS | ΔLPIPS | ΔTop1 | ΔS/ΔC | 效率 ΔS/|ΔC| | 饱和段 |")
        ap("|---|---:|---:|---:|---:|---:|---:|---|")
        for s in a["segments_natural_knob_order"]:
            sat = ",".join(m for m in METRICS if s["saturated"][m]) or "-"
            eff = s.get("efficiency_dS_per_content")
            ap(
                f"| {s['from_arm']}→{s['to_arm']} | {s['d_dino_c']:+.5f} | {s['d_dino_s']:+.5f} | {s['d_lpips']:+.5f} | "
                f"{s['d_target_top1']:+.5f} | {s['slope_deltaS_over_deltaC'] if s['slope_deltaS_over_deltaC'] is not None else 'n/a'} | {eff if eff is not None else 'n/a'} | {sat} |"
            )
        ap("")
        mono = a["monotonicity"]
        ap(
            "单调性: "
            + "; ".join(
                f"{m}={'单调' + ('增' if mono[m]['overall_direction'] == 'increasing' else '减' if mono[m]['overall_direction'] == 'decreasing' else '**非单调**')}"
                + (f"（违例段: {'; '.join(v['segment'][0] + '→' + v['segment'][1] for v in mono[m]['violations'])}）" if mono[m]["violations"] else "")
                for m in METRICS
            )
        )
        ap("")
        me = a["most_effective_interval"]
        ap(
            f"**最有效工作区间**: {me['segment'][0]}→{me['segment'][1]}（{fam['knob']} ∈ [{me['knob_range'][0]:g}, {me['knob_range'][1]:g}]，效率 ΔS/|ΔC| = {me['efficiency_dS_per_content']:.3f}）。"
        )
        if a["dead_segments"]:
            for d in a["dead_segments"]:
                ap(
                    f"**死区**: {d['segment'][0]}→{d['segment'][1]}（{fam['knob']} ∈ [{d['knob_range'][0]:g}, {d['knob_range'][1]:g}]）: {d['note']}（|ΔS|={d['d_dino_s_abs']}, |ΔC|={d['d_dino_c_abs']}）。"
                )
        ap("")
    ap("## 三、strength 映射（内容锚定，分段线性）")
    ap("")
    ap("约定: `s=0` ↔ 各引擎最弱编辑实测臂（DINO-C 最高），`s=1` ↔ 最强编辑实测臂；`s` 以 DINO-C 水位归一，故同 `s` 跨引擎内容水位一致（见水位表）。每段实测依据臂名如下；`s∈[0,1]` 全部为两个实测臂之间的插值，`s<0`/`s>1` 无实测臂支持（**UNMEASURED**），默认 clamp。")
    ap("")
    for iface in ("PRISMCore.transfer(strength)", "PRISMRef.transfer(strength)"):
        ap(f"### `{iface}`")
        ap("")
        for fam in FAMILIES:
            if fam["interface"] != iface:
                continue
            a = analyses[fam["key"]]
            sm = a["strength_mapping"]
            ap(f"**引擎: {fam['key']}（{fam['engine']}）→ 旋钮 `{fam['knob']}`（{a['direction_note']}）**")
            ap("")
            ap("| strength 段 | 旋钮区间 | 实测依据臂 | 端点 DINO-C |")
            ap("|---|---|---|---|")
            for seg in sm["segments"]:
                ep = seg["endpoint_metrics"]
                ap(
                    f"| [{seg['strength_range'][0]:.4f}, {seg['strength_range'][1]:.4f}] | {seg['knob_range'][0]:g} → {seg['knob_range'][1]:g} | "
                    f"`{seg['evidence_arms'][0]}` , `{seg['evidence_arms'][1]}` | {ep['from']['dino_c']:.5f} → {ep['to']['dino_c']:.5f} |"
                )
            ap(f"| [1.0, 1.2] | {sm['beyond_measured'][0]['knob_at_1.2']} (外推) | — | **UNMEASURED** |")
            ap(f"| [-0.2, 0.0] | {sm['beyond_measured'][1]['knob_at_-0.2']} (外推) | — | **UNMEASURED** |")
            m05 = sm["strength_at_0.5"]
            ap("")
            ap(
                f"- 默认档 `strength=0.5` → `{fam['knob']}` = {m05['knob_value']}"
                + (f"（bracket: `{m05['bracket_arms'][0]}`, `{m05['bracket_arms'][1]}`，descriptive only）" if not m05.get("measured") else f"（= 实测臂 `{m05['arm']}`）")
                + "。"
            )
            if fam["key"] == "d6_tau_min":
                ap(f"- 注意: τ_min=0.40 端点来自 M0_v20_d6（tau_hi=0.74），其余 B 臂 tau_hi=0.78 —— 端点臂存在 tau_hi 配置混杂。")
            if fam["key"] == "o3_tau0":
                ap(f"- 注意: O3 曲线存在 **wj 混杂**（t40 wj=2.3 / t60 wj=1.9），映射结点内嵌了 wj 差异，非纯 τ0 单变量。")
            if fam["key"] in ("d6_tau_min", "o3_tau0"):
                ap(f"- 注意: 反向旋钮 —— 接口内部应实现 `knob = f(strength)` 时按上表反向取值（s=1 ↔ 最小旋钮值）。")
            ap("")
    ap("### 跨引擎内容水位对齐表（s 网格, descriptive only）")
    ap("")
    header = "| strength | " + " | ".join(f["key"] + " (" + f["knob"] + ")" for f in FAMILIES) + " |"
    ap(header)
    ap("|" + "---|" * (len(FAMILIES) + 1))
    for s in NAIVE_S_GRID:
        cells = []
        for fam in FAMILIES:
            knots = analyses[fam["key"]]["strength_mapping"]["knots_measured"]
            v = strength_at(knots, s)
            cells.append(f"{v['knob_value']:g}" if "knob_value" in v else "UNMEASURED")
        ap(f"| {s:g} | " + " | ".join(cells) + " |")
    ap("")
    ap("同 `s` 下各引擎的 DINO-C 水位并不相同（各引擎归一化基准不同），跨引擎切换时按上表对齐内容水位（插值，descriptive only）。")
    ap("")
    for idx, (tag, title) in enumerate((("cs", "(DINO-C↑, DINO-S↑)"), ("ctop1", "(DINO-C↑, Top-1↑)"))):
        p = pareto[tag]
        ap(f"## {'四五'[idx]}、局部 Pareto 前沿 {title}")
        ap("")
        ap("| 族 | 前沿成员 | 被支配 | 前沿∩LPIPS≤0.51 |")
        ap("|---|---|---|---|")
        for fam in FAMILIES:
            pf = p["per_family"][fam["key"]]
            ap(
                f"| {fam['key']} | {', '.join('`' + a + '`' for a in pf['frontier'])} | "
                f"{', '.join('`' + a + '`' for a in pf['dominated']) or '-'} | "
                f"{', '.join('`' + a + '`' for a in pf['frontier_lpips_feasible']) or '-（全族 LPIPS>0.51）'} |"
            )
        g = p["global"]
        gl = p["global_lpips_constraint"]
        ap("")
        ap(f"- **全局前沿**: {', '.join('`' + a + '`' for a in g['frontier'])}。")
        ap(f"- 全局前沿中被支配: {', '.join('`' + a + '`' for a in g['dominated'])}。")
        ap(f"- 全局前沿中违反 LPIPS≤{LPIPS_MAX}: {', '.join('`' + a + '`' for a in g['frontier_members_violating_lpips']) or '无'}。")
        ap(f"- LPIPS≤{LPIPS_MAX} 可行子集的前沿（在可行臂内重新计算支配）: {', '.join('`' + a + '`' for a in gl['frontier_of_feasible_subset'])}。")
        ap("")
    ap("图: [a04_strength_pareto.svg](./a04_strength_pareto.svg) 双面板（C-S 前沿 / C-Top1 前沿），家族分色，实测点实心（LPIPS 违约为空心），黑边 = 该面板全局非支配成员，虚线连接线 **descriptive only**。")
    ap("")
    ap("## 六、科学结论")
    ap("")
    ap("### confirmed（实测支持）")
    ap("")
    ap("1. **四条旋钮在 (DINO-C, DINO-S) 平面均严格单调**，各自族内全部臂都在本族 C-S 前沿上 —— 每个旋钮都是合格的强度拨盘（无回摆）。唯一的单调性异常在 **D6 Top-1**（τ_min=0.66→0.70 段 +0.00167 反向），幅度仅 1 个任务（0.167pp），在计数噪声内。")
    ap("2. **D6 τ_min 响应在保守端饱和，0.73 内容水位不可达**。τ_min 0.66→0.70→0.72 的 ΔC 分别 +0.00345 / +0.00132（后者低于 0.002 阈值，饱和段），D6 曲线渐近 DINO-C≈0.7055。`B_d6_t072` 正是 A01 要求的 'τ_min≈0.72 补臂'，实测只到 0.7055 —— **A01 未决项 'D6 vs Hybrid @0.73' 无法靠 τ_min 平移解决**（tau_hi=0.78 固定下），需换杠杆（Hybrid γ≈0.55 水位 0.7321 / PRISM-1 λ≈1.0 水位 0.7440 均已有实测臂覆盖 0.73）。")
    ap("3. **D6 存在死区 τ_min∈[0.50,0.58]**: ΔS=−0.0008（低于 0.002 饱和阈值）而 ΔC=+0.0084 —— 纯内容代价、无风格收益；调 τ_min 应避开该带。最有效区间 τ_min∈[0.66,0.70]（效率 0.606）。")
    ap(f"4. **全局 C-S 前沿 13/19 臂**: D6 全族占据中段（DINO-C 0.669–0.706），Hybrid {chr(123)}g25, g40, g55{chr(125)} 占 0.733–0.753，PRISM-1 仅 l075（0.7715 最高内容），O3 三臂（t40/t45/t60）占 0.546–0.727 低内容带；被支配 6 臂: l100（g25）、l130/l160（g40）、C_hyb_g85 与 o3_t55（t066/t057）、**C_hyb_g70（被 o3_t60 双维度反超：C +0.0145 实在、S +0.00009 属亚噪声贴平**，见 JSON dominated_details 的 sub_noise_margin 标注——点估计口径下 O3 保守端已覆盖 Hybrid g70 的生态位）。")
    ap(f"5. **LPIPS≤{LPIPS_MAX} 约束下格局反转**: D6 全族 LPIPS 0.518–0.537 **全部违约**，可行前沿 = {{l075, g25, g40, g55, o3_t40/t45/t55/t60}}。D6 的 C-S 支配性以 LPIPS≥0.518 为代价 —— 04 文档 'B≳C、可删 Hybrid' 的初步读法（点估计+插值）在 LPIPS 约束口径下不成立，且 A01 第 2 组 CI 已证明密度门控必须保留；删除 Hybrid 的决策必须等 τ 对齐的配对 CI，且需明确是否接受 LPIPS 包络外扩。")
    ap("6. **C-Top1 前沿由 Hybrid 主导**: 全局前沿 = {l075, g25, g40, g55, g70, g85, t070, t072}（8 臂）；PRISM-1 族仅 l075（最大内容端）存活，O3 全族被支配（Top-1 天花板 40.83% 且随编辑增强反而下降）。D6 仅在最保守端（t070/t072）存活 —— 其 Top-1 平台期（≈0.55）使中段被同水位 Hybrid 臂支配。注意 t070/t072 与 g70 之间的 Top-1 差分（≤0.0067 = 4 任务）接近计数噪声，该段前沿成员资格对噪声敏感（top1 噪声阈值 0.005）。")
    ap("7. **各旋钮最有效区间（ΔS/|ΔC| 最大）**: PRISM-1 λ∈[0.75,1.00]（0.216）；D6 τ_min∈[0.66,0.70]（0.606，反向旋钮）；Hybrid γ∈[0.25,0.40]（1.97）；O3 τ0∈[0.55,0.60]（0.290，反向旋钮）。四条曲线共同模式：**保守端效率最高、激进端递减/饱和**，即所有引擎的边际风格收益在弱编辑侧最大。")
    ap("8. **strength 映射已交付**: PRISMCore 三引擎（prism1/λ、d6/τ_min、hybrid/γ）与 PRISMRef 一引擎（o3/τ0）的内容锚定分段线性映射表（每段标注实测依据臂），默认档 s=0.5 → λ≈1.218 / τ_min≈0.528 / γ≈0.630 / τ0≈0.506（descriptive 插值）。")
    ap("")
    ap("### rejected（被本轮证据否定）")
    ap("")
    ap("- 'D6 τ_min 继续上移可实测 DINO-C≈0.73 水位'（A01 §四.1 的解决路径）：B_d6_t070/t072 实测饱和于 0.7055，该路径关闭。")
    ap("- 'O3 τ0 曲线是纯 τ0 单变量'：t40/t60 的 wj 不同（2.3 vs 1.9），曲线混杂，不能作为纯强度响应引用。")
    ap("")
    ap("### inconclusive（未决，不做外推）")
    ap("")
    ap("- D6 vs Hybrid @0.73 配对判定：D6 在 0.73 无任何实测臂（饱和），维持 A01 UNDECIDED；若必须对齐 0.73，需改 D6 其它参数（tau_hi 上调 / 分支结构）补臂后用 a01 工具出 CI。")
    ap("- O3 保守端 s=0（τ0=0.60）仍离恒等映射很远（C=0.7274）；O3 缺少近恒等实测锚，PRISMRef 的弱编辑端行为 UNMEASURED。")
    ap("- D6 τ_min=0.40 端点臂（M0_v20_d6）tau_hi=0.74 与 B 臂 0.78 不一致，τ_min=0.40 端点的映射结点带协议混杂。")
    ap("- Top-1 分辨率：600 任务下单任务 = 0.167pp，所有 <0.5pp 的 Top-1 差分（含 D6 平台期内部差异）不作结论。")
    ap("")
    ap("## 七、风险与局限")
    ap("")
    ap("1. 全部结论为 **cross600 点估计**，无配对 CI；曲线臂之间未做源聚类 bootstrap（A01 工具可对任意两臂补算）。")
    ap("2. 效率/最有效区间按分段差分计算，段间密度不均（λ 段宽 0.25–0.30 vs τ_min 段宽 0.02–0.08），跨旋钮比较效率绝对值时注意分母口径（内容水位跨度不同：D6 全程仅 0.0367，O3 0.1818）。")
    ap("3. strength 映射的内容锚定基于 DINO-C；若下游更关心 Top-1 或 LPIPS 锚定，同表结构可用其它锚指标重算（脚本改 `anchor_metric` 一处）。")
    ap("4. Hybrid 族跨 V24 复用臂（g25/g40/g55）与 V25 实跑臂（g70/g85）两批来源，假定同族定义一致（04 文档曲线 C 口径）；PRISM-1 λ 与 Hybrid γ 的联合空间（λ≠1.0 的 Hybrid）未实测。")
    ap("5. SVG/PNG 中的连接线是实测点连线，仅描述性；面板不呈现任何插值臂读数点。")
    ap("")
    ap("## 八、复算命令")
    ap("")
    ap("```bash")
    ap("OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= \\")
    ap("  python -m tools.v25_cpu.a04_strength_analysis")
    ap("```")
    ap("")
    ap("CPU-only（stdlib + numpy-free 数值 + matplotlib Agg）；无 CUDA / ssh / git 调用；metrics.csv 只读。")
    ap("")
    return "\n".join(L)


def main() -> int:
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    records, tau_sources = build_arm_records()
    analyses = {fam["key"]: analyse_family(fam, records) for fam in FAMILIES}
    pareto = compute_pareto(records)

    payload = {
        "task": "A04 模型强度旋钮与 Pareto 曲线分析",
        "generated_utc": now_utc,
        "tool": "tools/v25_cpu/a04_strength_analysis.py",
        "inputs": {
            "metrics_csv": "docs/performance/metrics.csv",
            "runs_v25_json": [
                "runs/v25/r1_inference_ablation.json",
                "runs/v25/r1b_unlock_arms.json",
                "runs/v25/r3_oneshot_minimal.json",
            ],
            "tau_record_sources": tau_sources,
            "n_arms": len(records),
            "protocol": "test750 实测臂读数 cross600 (n=600)；无插值配对、无外推结论",
        },
        "thresholds": {
            "saturation": SAT_THRESHOLDS,
            "saturation_rule": "连续两点 |Δ| < 阈值 → 该段标注饱和",
            "top1_task_resolution": r5(TOP1_TASK_RESOLUTION),
            "pareto_lpips_max": LPIPS_MAX,
        },
        "families": analyses,
        "pareto": pareto,
        "figure_files": ["docs/v25/audits/a04_strength_pareto.svg", "docs/v25/audits/a04_strength_pareto.png"],
        "report_file": "docs/v25/audits/a04_strength_report.md",
        "descriptive_only_policy": "实测点之间的任何插值读数一律标 descriptive_only=true；strength 映射 s∈[0,1] 为实测臂端点间插值；s<0/s>1 为 UNMEASURED（clamp）。",
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    make_figure(analyses, records)
    report = make_report(analyses, pareto, records, now_utc)
    with open(MD_OUT, "w", encoding="utf-8") as f:
        f.write(report)

    # 控制台摘要
    print(f"[A04] {now_utc} — {len(records)} arms, 4 knob families, outputs written:")
    for p in (JSON_OUT, SVG_OUT, PNG_OUT, MD_OUT):
        print(f"  {p.relative_to(REPO_ROOT).as_posix()}")
    for fam in FAMILIES:
        a = analyses[fam["key"]]
        me = a["most_effective_interval"]
        mono_flags = "".join(
            "1" if a["monotonicity"][m]["is_monotone"] else "0" for m in METRICS
        )
        print(
            f"  {fam['key']:14s} mono(C,S,L,T)={mono_flags} "
            f"best=[{me['knob_range'][0]:g},{me['knob_range'][1]:g}] eff={me['efficiency_dS_per_content']} "
            f"sat_zones={len(a['saturation_zones'])}"
        )
    for tag in ("cs", "ctop1"):
        p = pareto[tag]
        print(
            f"  pareto[{tag}] global_frontier={len(p['global']['frontier'])} "
            f"feasible_frontier={len(p['global_lpips_constraint']['frontier_of_feasible_subset'])}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
