#!/usr/bin/env python3
"""V4 P1 报告：Pattern 三态可行性 + 实测覆盖率分层 + 共同可行集。

不做任何训练决策，只统计 supp 全量逐样本记录。
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path("/mnt/i/Latent_Style/WEAVE")
P1 = ROOT / "runs/p2/supp_records.jsonl"
M1 = ROOT / "runs/p2/supp_meta.json"
P2 = ROOT / "runs/p2/supp_pool2.jsonl"
P3 = ROOT / "runs/p2/supp_pool3.jsonl"
OUT_JSON = ROOT / "runs/p2/v4_pattern_report.json"
OUT_MD = ROOT / "runs/p2/v4_pattern_report.md"

PATTERNS = ["R64", "G64", "C64", "S64", "F64", "R128", "G128", "C128"]
PAT64 = ["R64", "G64", "C64", "S64", "F64"]
PAT128 = ["R128", "G128", "C128"]
NOMINAL = {c: 0.25 for c in PATTERNS + ["A_P5", "FEATHER_4", "FEATHER_8", "FEATHER_16", "FEATHER_EFF",
                                       "L1", "L2", "L3", "L4", "CTRL_rand", "CTRL_same", "CTRL_destyle", "CTRL_seed"]}
NOMINAL["C12"] = 0.125
NOMINAL["C37"] = 0.375
NOMINAL["C50"] = 0.5
# 工程分组阈值（不是风格质量阈值）
T_PASS, T_LOW = 0.22, 0.18


def load(path):
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def summarize(rows, meta):
    per_cfg = defaultdict(list)
    for r in rows:
        per_cfg[r["config"]].append(r)
    out = {}
    for cfg, rs in per_cfg.items():
        covs = [float(r["mask"]["coverage_actual"]) for r in rs]
        n_src = len({r["source_id"] for r in rs})
        nominal = NOMINAL.get(cfg)
        layering = None
        if nominal == 0.25:
            layering = {
                "pass_ge22": sum(1 for c in covs if c >= T_PASS),
                "low_18_22": sum(1 for c in covs if T_LOW <= c < T_PASS),
                "drop_lt18": sum(1 for c in covs if c < T_LOW),
            }
        out[cfg] = {
            "n_rows": len(rs),
            "n_sources_layout_feasible": n_src,
            "layout_infeasible_sources": meta.get("infeasible", {}).get(cfg, 0),
            "coverage_nominal": nominal,
            "coverage_actual_mean": round(sum(covs) / len(covs), 4) if covs else None,
            "coverage_actual_min": round(min(covs), 4) if covs else None,
            "coverage_actual_p05": round(sorted(covs)[max(0, int(0.05 * len(covs)) - 1)], 4) if covs else None,
            "q_mean": round(sum(float(r["q_mean"]) for r in rs) / len(rs), 4) if rs else None,
            "layering_25pct": layering,
        }
    return out


def common_counts(rows):
    src_by_cfg = defaultdict(set)
    for r in rows:
        src_by_cfg[r["config"]].add(r["source_id"])
    def inter(cfgs):
        if not all(src_by_cfg[c] for c in cfgs):
            return 0
        return len(set.intersection(*[src_by_cfg[c] for c in cfgs]))
    return {"all8": inter(PATTERNS), "p64": inter(PAT64), "p128": inter(PAT128),
            "A_P5_and_R128": inter(["A_P5", "R128"])}


def main():
    m1 = json.loads(M1.read_text()) if M1.exists() else {}
    r1 = load(P1)
    # 逐配置 source_id 集合
    src_by_cfg = defaultdict(set)
    for r in r1:
        src_by_cfg[r["config"]].add(r["source_id"])
    summ1 = summarize(r1, m1)

    common64 = set.intersection(*[src_by_cfg[c] for c in PAT64]) if PAT64 and all(src_by_cfg[c] for c in PAT64) else set()
    common128 = set.intersection(*[src_by_cfg[c] for c in PAT128]) if all(src_by_cfg[c] for c in PAT128) else set()
    common_all8 = set.intersection(*[src_by_cfg[c] for c in PATTERNS]) if all(src_by_cfg[c] for c in PATTERNS) else set()
    common_A_E7 = src_by_cfg["A_P5"] & src_by_cfg["R128"]

    rep = {
        "pool1": {"n_records": len(r1), "n_sources": m1.get("n_sources"), "seconds": m1.get("total_seconds"),
                  "configs": summ1},
        "common_feasible": {
            "all8_patterns": sorted(common_all8),
            "n_all8": len(common_all8),
            "patterns64_5": sorted(common64), "n_patterns64_5": len(common64),
            "patterns128_3": sorted(common128), "n_patterns128_3": len(common128),
            "A_P5_and_R128": sorted(common_A_E7), "n_A_P5_and_R128": len(common_A_E7),
        },
        "thresholds": {"pass_ge": T_PASS, "low_range": [T_LOW, T_PASS]},
    }
    # 每个 Pattern 的 25% 主池可用张数（≥22%）
    main_pool = {}
    for c in PATTERNS:
        s = {r["source_id"] for r in r1 if r["config"] == c and float(r["mask"]["coverage_actual"]) >= T_PASS}
        main_pool[c] = len(s)
    rep["main_pool_ge22_n"] = main_pool
    rep["cross_pool_common"] = {"pool1": common_counts(r1), "pool2": common_counts(load(P2)),
                                "pool3": common_counts(load(P3))}

    OUT_JSON.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")

    lines = ["# V4 P1 Pattern 可行性报告（pool1 seed2024, 300 图）", "",
             f"- 记录数 {len(r1)}，meta n_sources={m1.get('n_sources')}，总耗时 {m1.get('total_seconds')}s", "",
             "## 1) 共同可行集", "",
             "| 集合 | 张数 |", "|---|---|",
             f"| 8 个 Pattern 全部可行 | **{len(common_all8)}** |",
             f"| 5 个 64px Pattern 全部可行 | {len(common64)} |",
             f"| 3 个 128px Pattern 全部可行 | {len(common128)} |",
             f"| A_P5 与 R128 同时可行 | {len(common_A_E7)} |", "",
             "跨池（同口径，300 图/池）：",
             "",
             "| 池 | 8-Pattern 共同 | 5×64px 共同 | 3×128px 共同 | A_P5∩R128 |",
             "|---|---|---|---|---|"]
    for pname, rows in (("pool1(2024)", r1), ("pool2(2025)", load(P2)), ("pool3(2026)", load(P3))):
        cc = common_counts(rows)
        lines.append(f"| {pname} | **{cc['all8']}** | {cc['p64']} | {cc['p128']} | {cc['A_P5_and_R128']} |")
    lines.append("")
    lines += ["## 2) 各 Pattern 可行张数与实测覆盖率", "",
              "| Pattern | 名义 | layout_feasible | 不可行 | 实测覆盖 mean | min | p05 | q_mean | ≥22% 张数 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for c in PATTERNS + ["A_P5", "CTRL_rand", "CTRL_same", "CTRL_destyle", "CTRL_seed", "L1", "L3", "C37"]:
        s = summ1.get(c)
        if not s:
            continue
        lay = s["layering_25pct"]
        passn = lay["pass_ge22"] if lay else ("n/a" if s["coverage_nominal"] != 0.25 else 0)
        lines.append(f"| {c} | {s['coverage_nominal'] or '-'} | {s['n_sources_layout_feasible']} | "
                     f"{s['layout_infeasible_sources']} | {s['coverage_actual_mean']} | {s['coverage_actual_min']} | "
                     f"{s['coverage_actual_p05']} | {s['q_mean']} | {passn} |")
    lines += ["", "## 3) 128px 名义 vs 实测（必答项：两者必须同时给出）", "",
              "| Pattern | 名义 | 实测 mean | 实测 min | 25% 主池(≥22%) | 低覆盖(18–22%) | 丢弃(<18%) |",
              "|---|---|---|---|---|---|---|"]
    for c in PAT128 + ["A_P5"]:
        s = summ1.get(c)
        if not s:
            continue
        lay = s["layering_25pct"] or {"pass_ge22": 0, "low_18_22": 0, "drop_lt18": 0}
        lines.append(f"| {c} | 0.25 | {s['coverage_actual_mean']} | {s['coverage_actual_min']} | "
                     f"{lay['pass_ge22']} | {lay['low_18_22']} | {lay['drop_lt18']} |")

    # 跨池一致性
    for name, path in (("pool2", P2), ("pool3", P3)):
        rows = load(path)
        if not rows:
            lines += ["", f"## 4) {name}（seed 2025/2026）", "", "未产出。"]
            continue
        s2 = summarize(rows, {})
        lines += ["", f"## 4) {name}：{len(rows)} 记录", "",
                  "| Pattern | layout_feasible | 实测覆盖 mean |", "|---|---|---|"]
        for c in PATTERNS + ["A_P5"]:
            if c in s2:
                lines.append(f"| {c} | {s2[c]['n_sources_layout_feasible']} | {s2[c]['coverage_actual_mean']} |")

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("\nWROTE", OUT_JSON, OUT_MD)


if __name__ == "__main__":
    main()
