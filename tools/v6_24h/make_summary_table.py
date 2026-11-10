"""Generate the summary markdown table from the actual reports.

The hand-maintained table at the top of V6_24H_RESULTS.md drifted as arms were added, which is
exactly the kind of transcription error a table generator removes. Group order and the one-line
verdicts are curated; every number is read from runs/v6/<arm>/report.json.

Usage
    python3 -m tools.v6_24h.make_summary_table            # print
    python3 -m tools.v6_24h.make_summary_table --write    # splice into the results doc
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DOC = ROOT / "docs/v6/V6_24H_RESULTS.md"
START = "| 臂 | 组 | ΔT | ΔT>0 | DINO-C | LPIPS | 风格+ | Top-1 | 自一致 | 训练min | 结论 |"
END_MARK = "\n\n**三条全局结论**"

# curated: arm -> (group, one-line verdict)
CURATED = {
    "L2_highS": ("L", "ΔT 领先，内容双跌破、自一致为负"),
    "L11_k1024_dim16": ("L", "**最终基线**：参数 −88.4%、七项无一项变差"),
    "L8_k1024_dim32": ("L", "ΔT + 自一致最好（+0.0076）"),
    "L10_k1024_dim24": ("L", "与 L8 打平但参数少 43%"),
    "L9_k1024_dim32_lc075": ("L", "内容最优（DINO-C / LPIPS 全场最好）"),
    "S6_style_keys": ("S", "Top-1 最高（0.040）；唯一改机制"),
    "L7_highC_dim32": ("L", "5/5（Ukiyo 余量在噪声内）"),
    "N7_dim24": ("N", "参数 −74.5%、ΔT 更好（宽度曲线）"),
    "L4_ref_full": ("L", "首个严格支配 A0（LPIPS −0.0253）"),
    "N3_dim32": ("N", "参数 −55%"),
    "N7_dim40": ("N", "宽度曲线第 4 点"),
    "S2_no_inject_32": ("S", "关 32×32 反而提升 ΔT"),
    "L6_highC": ("L", "λ_c 饱和（+0.0012）"),
    "S6b_style_keys48": ("S", "机制收益不随宽度复现"),
    "A0": ("—", "参考基准"),
    "L3_core_off": ("L", "Core64 采样为空操作"),
    "S1_no_inject_64": ("S", "64×64 层几乎中性"),
    "S3_no_inject_16": ("S", "16×16 层几乎中性"),
    "N4_steps400": ("N", "余量 < 噪声，仅速度收益"),
    "L5_same_off": ("L", "真实取舍点（内容更优）"),
    "L1_lowC": ("L", "被多数臂支配"),
    "N2_no_canon": ("N", "内容较好但 ΔT 弱"),
    "N5_inv_off": ("N", "λ_c 必要（双跌破）"),
    "N1_no_style": ("N", "Style-ID 必要（退化成恒等）"),
}


def load(arm: str) -> dict | None:
    p = ROOT / "runs/v6/D1_c1_gentle/report.json" if arm == "A0" else \
        ROOT / f"runs/v6/{arm}/report.json"
    if not p.exists():
        return None
    r = json.loads(p.read_text())
    e = r.get("eval200") or {}
    if not e.get("mean_dT"):
        return None
    return {"e": e, "params": r.get("n_params"),
            "wall": (r.get("train_wall_sec") or 0) / 60.0}


def build() -> str:
    st = json.loads((ROOT / "runs/v6_24h/state.json").read_text())
    arms = ["A0"] + [k for k, v in st.items() if v.get("status") == "done"]
    rows = []
    for a in arms:
        d = load(a)
        if not d:
            continue
        e = d["e"]
        pt = e.get("per_target_top1") or {}
        top1 = sum(pt.values()) / len(pt) if pt else None
        rows.append((a, d, e, top1))
    # sort by dT descending, A0 last among equals
    rows.sort(key=lambda x: (-x[2]["mean_dT"], x[0] != "A0"))

    out = [START, "|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|"]
    for a, d, e, top1 in rows:
        grp, note = CURATED.get(a, ("?", ""))
        dT = e["mean_dT"]
        dTs = f"**{dT:+.4f}**" if dT == max(r[2]["mean_dT"] for r in rows) else f"{dT:+.4f}"
        if a == "N1_no_style":
            dTs = f"**{dT:+.4f}**"
        sds = e.get("same50_mean_dS")
        ok = "✅" if (sds is None or abs(sds) < 0.02) else "❌"
        sty = e.get("styles_improved")
        sty_s = f"**{sty}/5**" if sty == 5 else f"{sty}/5"
        t1 = f"{top1:.3f}" if top1 is not None else "—"
        wall = f"{d['wall']:.1f}" if d["wall"] else "—"
        out.append(
            f"| {a} | {grp} | {dTs} | {e.get('frac_dT_positive', 0):.2f} | "
            f"{e['mean_dino_c']:.4f} | {e['mean_lpips']:.4f} | {sty_s} | {t1} | {ok} | "
            f"{wall} | {note} |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    table = build()
    if not args.write:
        print(table)
        return
    text = DOC.read_text(encoding="utf-8")
    i = text.index(START)
    j = text.index(END_MARK)
    DOC.write_text(text[:i] + table + text[j:], encoding="utf-8")
    n = table.count("\n") - 1
    print(f"spliced {n} rows into {DOC.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
