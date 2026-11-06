"""P4 汇总：T0–T5 的 600 跨风格 / 150 同风格 / 五类目标风格 + FM 探针（留出集、错误 ID、跨 Pattern）。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import paths  # noqa: E402

CELLS = ["S0_base_b256", "T1_R64_positive", "T2_F64_positive", "T3_R64_reverse", "T4_F64_reverse",
         "T5_R64_reverse_nostyle"]
LABELS = {
    "S0_base_b256": "T0 原 FM 无伪GT（b256，内部早停 412 步）",
    "T1_R64_positive": "T1 A 正向 R64",
    "T2_F64_positive": "T2 A 正向 F64",
    "T3_R64_reverse": "T3 E 反向 R64",
    "T4_F64_reverse": "T4 E 反向 F64",
    "T5_R64_reverse_nostyle": "T5 E 反向 R64（无 Style-ID）",
}
PAIRS = {"T1_R64_positive": ("R64_pos", "F64_pos"), "T2_F64_positive": ("F64_pos", "R64_pos"),
         "T3_R64_reverse": ("R64_rev", "F64_rev"), "T4_F64_reverse": ("F64_rev", "R64_rev"),
         "T5_R64_reverse_nostyle": ("R64_rev", "F64_rev")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO_ROOT / "runs/p2/pgt_p4_summary.md"))
    ap.add_argument("--json", default=str(REPO_ROOT / "runs/p2/pgt_p4_summary.json"))
    args = ap.parse_args()

    rows = {}
    for cell in CELLS:
        p = REPO_ROOT / "runs/p2" / f"pgt_{cell}.json"
        if not p.exists():
            rows[cell] = {"missing": True}
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        m, s = d["metrics"], d["same_style_150"]["metrics"]
        fm = REPO_ROOT / "runs/p2" / f"fm_{cell}.json"
        fm_rows = json.loads(fm.read_text(encoding="utf-8")) if fm.exists() else []
        fm_tab = {}
        for r in fm_rows:
            fm_tab.setdefault(r["probe"], {})[r["sid_mode"]] = r["total"]
        rows[cell] = {
            "cross600": {k: {kk: m[k][kk] for kk in ("mean", "ci95_low", "ci95_high", "frac_positive", "n")}
                         for k in ("csd_dT", "csd_dS", "csd_dM", "dino_dT", "dino_dM", "swd_gain")},
            "same150": {k: s[k]["mean"] for k in s},
            "per_target": {t: {k: v[k] for k in ("csd_dT", "csd_dM", "dino_dM")} for t, v in
                           d["per_target_style"].items()},
            "fm": fm_tab,
            "checkpoint": d.get("generated_cell"),
            "timing_sec": d.get("timing_sec"),
        }

    Path(args.json).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    md = ["# P4 短训练结果（600 步上限 / 有效 batch 256）", "",
          "| cell | 说明 | CSD ΔT | ΔS | ΔM [95% CI] | DINO ΔM | SWD gain | 150同风格 CSD ΔStyle | LPIPS(same) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for cell in CELLS:
        r = rows[cell]
        if r.get("missing"):
            md.append(f"| {cell} | {LABELS[cell]} | 缺失 | | | | | | |")
            continue
        c, s = r["cross600"], r["same150"]
        md.append(f"| {cell} | {LABELS[cell]} | {c['csd_dT']['mean']:.4f} | {c['csd_dS']['mean']:.4f} | "
                  f"{c['csd_dM']['mean']:.4f} [{c['csd_dM']['ci95_low']:.4f},{c['csd_dM']['ci95_high']:.4f}] | "
                  f"{c['dino_dM']['mean']:.4f} | {c['swd_gain']['mean']:.4f} | "
                  f"{s['csd_dStyle']:.4f} | {s['lpips_gen']:.4f} |")
    md += ["", "## 五类目标风格（600 跨风格）", "",
           "| cell | " + " | ".join(["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]) + " |",
           "|---|" + "---|" * 5]
    for cell in CELLS:
        r = rows[cell]
        if r.get("missing"):
            continue
        md.append(f"| {cell} | " + " | ".join(
            f"{r['per_target'][t]['csd_dT']:.3f}/{r['per_target'][t]['csd_dM']:.3f}"
            for t in ("Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e")) + " |")
    md += ["", "（格式：CSD ΔT / ΔM）", "", "## FM 探针（固定 t=0.5，留出集 400 图）", "",
           "| cell | 训练 Pattern 留出集 | 另一 Pattern 留出集 | 跨 Pattern 差 | 错误 Style-ID Δ |",
           "|---|---|---|---|---|"]
    for cell in CELLS:
        r = rows[cell]
        if r.get("missing") or not r.get("fm"):
            continue
        seen, unseen = PAIRS.get(cell, (None, None))
        if not seen:
            continue
        a = r["fm"].get(seen, {}).get("correct")
        b = r["fm"].get(unseen, {}).get("correct")
        d1 = r["fm"].get(seen, {}).get("1", float("nan"))
        d2 = r["fm"].get(seen, {}).get("2", float("nan"))
        md.append(f"| {cell} | {seen}={a:.5f} | {unseen}={b:.5f} | {b - a:+.5f} | "
                  f"+1:{d1 - a:+.5f} +2:{d2 - a:+.5f} |")
    Path(args.out).write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[p4] -> {args.out}")


if __name__ == "__main__":
    main()
