"""Style-content Pareto plot for the V6 next-phase delivery.

x = DINO-C (content fidelity, higher is better), y = Cross600 Top-1 (style identity, higher is
better). Points on the upper-left frontier are non-dominated. Every point is annotated with its
provenance so a reader cannot mistake a quoted number for a measured one.

Reads the compact JSONs shipped in deliver/compact, so it can be re-run anywhere the bundle lands.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
COMPACT = ROOT / "deliver" / "full"   # need per-task rows for cross-only metrics
STY = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
MIN = STY.index("Minimalism")


def cross_metrics(path: Path):
    d = json.loads(path.read_text())
    rows = d.get("rows")
    if not rows:
        return None
    cross = [r for r in rows if r["tgt_style"] != r["src_style"]]
    if not cross:
        return None
    return (sum(r["dino_c"] for r in cross) / len(cross),
            sum(r["target_top1"] for r in cross) / len(cross),
            len(cross))


def p0_variants():
    p = ROOT / "deliver" / "full" / "p0_variants.json"
    if not p.exists():
        p = COMPACT / "p0_variants.json"
    if not p.exists():
        return {}
    v = json.loads(p.read_text()).get("variants") or {}
    return {k: (x.get("dino_c"), x.get("top1")) for k, x in v.items()
            if x.get("dino_c") is not None}


LABEL_OFFSET = {
    "B4 distill λ=50": (7, -13),
    "Oracle 4-step dyn": (7, 6),
    "ScoreDiff N=24": (7, 5),
    "B0 control (cont.)": (-4, 11),
    "Q2a Direct": (7, -12),
}


def main() -> None:
    pts = []
    p0 = p0_variants()
    labels_p0 = {"A0": "Q2a Direct", "A1": "Oracle 1-step", "A2": "Oracle 4-step dyn",
                 "A3": "Oracle 1-step (=A1)", "A4": "ScoreDiff N=24"}
    for k, lab in labels_p0.items():
        if k in p0 and p0[k][0] is not None:
            pts.append((p0[k][0], p0[k][1], lab, "P0"))

    for name, lab in (("eval_750_P1_B0_ctrl218", "B0 control (cont.)"),
                      ("eval_750_P1_B3_distill20", "B3 distill λ=20"),
                      ("eval_750_P1_B4_distill50", "B4 distill λ=50"),
                      ("eval_750_scorediff_N12", "ScoreDiff N=12")):
        m = cross_metrics(COMPACT / f"{name}.json")
        if m:
            pts.append((m[0], m[1], lab, "750"))

    if not pts:
        print("no data points found; is the deliver/ bundle present?")
        return

    fig, ax = plt.subplots(figsize=(9.0, 6.2), dpi=160)
    for dino, t1, lab, src in pts:
        marker = "o" if src == "P0" else "s"
        color = "#1f77b4" if src == "P0" else "#d62728"
        ax.scatter(dino, t1 * 100, s=90, marker=marker, color=color,
                   edgecolor="white", linewidth=0.8, zorder=3)
        # two points sit almost on top of each other at (0.796, 24.2), so the label offset is
        # chosen per point instead of uniformly
        off = LABEL_OFFSET.get(lab, (7, 5))
        ax.annotate(lab, (dino, t1 * 100), textcoords="offset points",
                    xytext=off, fontsize=8.5, color="#222222")

    # frontier: keep points not dominated in (dino, top1)
    front = []
    for i, (d1, t1, _, _) in enumerate(pts):
        dominated = any(d2 >= d1 and t2 >= t1 and (d2 > d1 or t2 > t1)
                        for j, (d2, t2, _, _) in enumerate(pts) if j != i)
        if not dominated and (d1, t1) not in front:
            front.append((d1, t1))
    front.sort()
    if len(front) > 1:
        ax.plot([f[0] for f in front], [f[1] * 100 for f in front],
                "--", color="#999999", lw=1.2, zorder=2, label="non-dominated frontier")

    ax.set_xlabel("DINO-C  (content fidelity, higher is better)", fontsize=10)
    ax.set_ylabel("Cross600 Top-1  (target style identity, %)", fontsize=10)
    ax.set_title("V6 next phase · style-content Pareto (600 cross tasks, paper_rgb_v1)",
                 fontsize=11)
    ax.grid(alpha=0.25, linestyle=":")
    ax.legend(handles=[
        plt.Line2D([], [], marker="o", ls="", color="#1f77b4", label="P0 inference variants"),
        plt.Line2D([], [], marker="s", ls="", color="#d62728", label="750-protocol checkpoints"),
        plt.Line2D([], [], ls="--", color="#999999", label="non-dominated frontier"),
    ], loc="lower right", fontsize=8.5, framealpha=0.95)
    ax.set_ylim(bottom=-2)
    fig.tight_layout()

    out = HERE / "pareto_style_content.png"
    fig.savefig(out)
    print(f"wrote {out}")
    print(f"  {len(pts)} points, frontier = {[(round(a,3), round(b*100,1)) for a, b in front]}")


if __name__ == "__main__":
    main()
