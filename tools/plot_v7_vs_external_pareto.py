"""V7 / V7-R vs external Pareto frontier, with q2a TRT speed.

Based on E:/plot_style_content_speed_standalone.py's visual language.

Key rules (per user corrections):
  * The 2-D frontier is computed from EXTERNAL baselines ONLY. WEAVE,
    Q2a+R1 Oracle, B3/B4 and all V7/V7-R candidates are OURS and never
    participate in the frontier line.
  * Y = DINO-S, the one style metric shared by V7/V7-R evals and the external
    leaderboard (V7 evals carry no CLIP-S, so Style Affinity is unavailable).
  * All V7/V7-R candidates inherit the Q2a TRT speed 45.78 ms / img
    (docs/v6/INFERENCE_SPEED_REPORT.md); they share the same frontend.
  * The visible progress is the move UP-RIGHT from the B4 anchor: E3/E4 local
    patch energy raises DINO-C AND DINO-S simultaneously.

Outputs: deliver/v7_vs_external_pareto.{png,svg,pdf}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import patheffects as pe

ROOT = Path(__file__).resolve().parents[1]
DELIVER = ROOT / "deliver"

# ---- EXTERNAL baselines only: (name, DINO-C, DINO-S, ms/img) ----
EXTERNAL = [
    ("StyleShot",    0.3569, 0.5750, 24480.0),
    ("StyleID",      0.3885, 0.5143,  5040.0),
    ("AesPA",        0.6075, 0.4661,  1840.0),
    ("StyTr2",       0.7023, 0.4074,  3040.0),
    ("Seedream 4.5", 0.7343, 0.3885,  2000.0),
    ("SaMAM",        0.8239, 0.3832,  1408.0),
    ("CUT",          0.8435, 0.3755,   400.0),
    ("SD-Turbo",     0.9998, 0.3633,   400.0),
    ("SaMST",        0.1628, 0.2738,   800.0),
    ("WCT",          0.0232, 0.1542,   293.6),
]

# ---- OURS reference points (NEVER in the frontier) ----
Q2A_MS = 45.78          # Q2a TRT stitched estimate, 21.85 img/s
WEAVE_PUB = ("WEAVE (published)", 0.8079, 0.4003, 97.60)
Q2A_ORACLE = ("Q2a + R1 Oracle", 0.7345, 0.4360, Q2A_MS)
OURS_S10 = ("Q2a+Oracle + new S10", 0.7324, 0.4323, Q2A_MS)
B3 = ("B3 anchor", 0.8325, 0.4183, Q2A_MS)
B4 = ("B4 anchor", 0.8199, 0.4214, Q2A_MS)

EXTERNAL_LAB = {
    "StyleShot": (0.40, 0.585, "left"), "StyleID": (0.43, 0.525, "left"),
    "AesPA": (0.62, 0.478, "left"), "StyTr2": (0.71, 0.415, "left"),
    "SaMAM": (0.85, 0.395, "left"), "CUT": (0.87, 0.375, "left"),
    "SD-Turbo": (0.99, 0.352, "right"), "Seedream 4.5": (0.76, 0.375, "left"),
    "SaMST": (0.20, 0.285, "left"), "WCT": (0.07, 0.17, "left"),
}


def pareto(points):
    return sorted([
        a for a in points
        if not any(b[1] >= a[1] and b[2] >= a[2]
                   and (b[1] > a[1] or b[2] > a[2])
                   for b in points if b is not a)
    ], key=lambda p: p[1])


def short_name(path: str) -> str:
    return (path.replace("runs/v7/", "V7/").replace("runs/v7r/", "V7R/")
            .replace("runs/v6_r2/eval_", "").replace("/eval.json", "")
            .replace("p2_predictor/", "").replace("p3_corrector/", "")
            .replace("p2_eval/", "").replace("p3_eval/", "")
            .replace("eval_step_", "s"))


def main():
    d = json.loads((DELIVER / "v7_all_metrics.json").read_text(encoding="utf-8"))
    vpts, probes = [], []
    for path, m in d.items():
        if "dino_s" not in m or "dino_c" not in m:
            continue
        rec = (short_name(path), m["dino_c"], m["dino_s"], Q2A_MS, m)
        (probes if "/p1a_" in path else vpts).append(rec)

    # ---- quantitative progress vs B4 anchor ----
    up_style = [p for p in vpts if p[2] > B4[2]]
    up_right = [p for p in vpts if p[1] > B4[1] and p[2] > B4[2]]
    best_up_right = sorted(up_right, key=lambda z: -(z[1] + z[2]))[:6]
    ext_front = pareto(EXTERNAL)
    # candidates strictly inside/above the external frontier (2-D)
    def ext_dominates(p):
        return any(e[1] >= p[1] and e[2] >= p[2] for e in EXTERNAL)
    beyond = [p for p in vpts if not ext_dominates(p)]
    print(f"real candidates {len(vpts)} | probes {len(probes)}")
    print(f"vs B4: DINO-S up {len(up_style)} | both axes up {len(up_right)}")
    for p in best_up_right:
        print("   up-right:", p[0], "C", round(p[1], 4), "S", round(p[2], 4))
    print(f"beyond external frontier (2-D, no external dominates): {len(beyond)}")

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "svg.fonttype": "none", "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    ink, muted = "#182A42", "#62758B"
    blue, grey = "#356C8F", "#B9C3CE"
    orange, orange_dark, purple = "#ED6547", "#AE3A29", "#8E44AD"
    bg = "#FCFDFE"

    speed = lambda p: 1000.0 / p[3]
    K = 2450.0 / speed(OURS_S10)
    area = lambda p: K * speed(p)

    fig = plt.figure(figsize=(14.6, 9.2), facecolor=bg)
    ax = fig.add_axes([.085, .275, .83, .52], facecolor=bg)
    ax.set_xlim(0.34, 1.06)          # crop from StyleShot (~0.357) leftwards
    ax.set_ylim(.34, .61)
    ax.set_xticks([0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticks([.35 + .025 * i for i in range(11)])
    ax.set_axisbelow(True)
    ax.grid(color="#E8EEF4", lw=1)
    for edge in ("bottom", "left"):
        ax.spines[edge].set_color("#B9C5D0")
    ax.tick_params(colors=muted, length=0, pad=8)
    ax.set_xlabel("Content preservation / DINO-C   → Higher is better",
                  fontsize=13, weight="semibold", color=ink, labelpad=18)
    ax.set_ylabel("Style / DINO-S   ↑ Higher is better",
                  fontsize=13, weight="semibold", color=ink, labelpad=16)

    # EXTERNAL-only Pareto frontier
    fx = [p[1] for p in ext_front]
    fy = [p[2] for p in ext_front]
    ax.plot(fx, fy, color=blue, lw=2.1, ls=(0, (4, 3)), zorder=2)
    ax.fill_between(fx, fy, .34, color=blue, alpha=.035, zorder=1)

    front_names = {p[0] for p in ext_front}
    for p in EXTERNAL:
        hit = p[0] in front_names
        ax.scatter(p[1], p[2], s=area(p), marker="o",
                   fc=blue if hit else grey,
                   ec=blue if area(p) < 25 else "white",
                   lw=.6 if area(p) < 25 else 1.3,
                   alpha=.94 if hit else .72, zorder=6 if hit else 3)

    # diagnostic probes: hollow rings, excluded from verdict
    if probes:
        ax.scatter([p[1] for p in probes], [p[2] for p in probes],
                   s=80, marker="o", fc="none", ec="#B39DDB", lw=1.0,
                   alpha=.7, zorder=4)

    # real V7/V7-R candidates
    ax.scatter([p[1] for p in vpts], [p[2] for p in vpts],
               s=area(Q2A_MS if False else Q2A_ORACLE) * .40, marker="o",
               fc=purple, ec="white", lw=.8, alpha=.62, zorder=5)

    # OURS reference points on top
    ax.scatter(WEAVE_PUB[1], WEAVE_PUB[2], s=area(WEAVE_PUB) * .42, marker="o",
               fc="#E2E8EF", ec="#5A6F88", lw=1.6, alpha=.55, zorder=7)
    ax.scatter(OURS_S10[1], OURS_S10[2], s=area(OURS_S10), marker="o",
               fc=orange, ec="white", lw=2.4, alpha=.98, zorder=9)
    ax.scatter(Q2A_ORACLE[1], Q2A_ORACLE[2], s=area(Q2A_ORACLE) * .50,
               marker="o", fc="#2E6F9E", ec="white", lw=2.2, alpha=.98, zorder=11)
    for nm, p in (("B3", B3), ("B4", B4)):
        ax.scatter(p[1], p[2], s=120, marker="s", fc="#F4C430",
                   ec="#7A5C00", lw=1.2, zorder=12)

    # ---- arrow: the visible UP-RIGHT progress from B4 to E3/E4 ----
    e3 = next((p for p in vpts if p[0].endswith("p1c_E3")), None)
    if e3:
        ax.annotate("UP-RIGHT from B4:\nE3 local patch energy\n"
                    f"C {B4[1]:.3f}→{e3[1]:.3f}, S {B4[2]:.3f}→{e3[2]:.3f}",
                    xy=(e3[1], e3[2]), xytext=(0.60, 0.575),
                    ha="left", va="center", fontsize=9.6, color=purple,
                    weight="bold", zorder=14,
                    bbox=dict(boxstyle="round,pad=0.35", fc="#F6F0FA",
                              ec="#C9B6D8", lw=0.9),
                    arrowprops=dict(arrowstyle="-|>", color=purple, lw=2.0,
                                    connectionstyle="arc3,rad=-0.18"))

    # external labels
    for p in EXTERNAL:
        if p[1] < 0.34:            # WCT / SaMST cropped out
            continue
        tx, ty, ha = EXTERNAL_LAB.get(p[0], (p[1] + .01, p[2] + .008, "left"))
        ann = ax.annotate(p[0], xy=(p[1], p[2]), xytext=(tx, ty),
                          ha=ha, va="center", fontsize=9.2,
                          color=ink if p[0] in front_names else muted,
                          weight="semibold" if p[0] in front_names else "normal",
                          arrowprops=dict(arrowstyle="-", lw=.7,
                                          color="#AAB9C8", shrinkA=3, shrinkB=4),
                          zorder=12)
        ann.set_path_effects([pe.withStroke(linewidth=3, foreground=bg)])

    # ours labels
    ax.annotate("WEAVE (published, ours)", xy=(WEAVE_PUB[1], WEAVE_PUB[2]),
                xytext=(.86, .455), ha="left", fontsize=8.8, color="#5A6F88",
                arrowprops=dict(arrowstyle="-", color="#5A6F88", lw=.8), zorder=12)
    ax.annotate("B3 / B4 anchors", xy=(B4[1], B4[2]),
                xytext=(.90, .445), ha="left", fontsize=8.8, color="#7A5C00",
                arrowprops=dict(arrowstyle="-", color="#7A5C00", lw=.8), zorder=12)
    ax.annotate("Q2a + R1 Oracle\n(strongest ours)",
                xy=(Q2A_ORACLE[1], Q2A_ORACLE[2]), xytext=(.615, .475),
                ha="left", va="center", fontsize=9.2, color="#163A55",
                weight="bold",
                arrowprops=dict(arrowstyle="-|>", color="#2E6F9E", lw=1.3,
                                connectionstyle="arc3,rad=0.12"), zorder=13)
    ax.annotate("OURS: Q2a+Oracle\n+ new S10 decoder",
                xy=(OURS_S10[1], OURS_S10[2]), xytext=(.50, .545),
                ha="left", va="center", fontsize=10, color=orange_dark,
                weight="bold",
                arrowprops=dict(arrowstyle="-|>", color=orange, lw=1.5,
                                connectionstyle="arc3,rad=-0.12"), zorder=13)

    fig.text(.09, .932, "V7 / V7-R PUSH THE OURS CLOUD UP-RIGHT OF B4",
             fontsize=23, weight="heavy", color=ink, va="top")
    fig.text(.09, .867,
             "Frontier = EXTERNAL baselines only (WEAVE and every Q2a/B4/V7 point are ours).  "
             f"Y = DINO-S, shared by all evals.  {len(up_right)} candidates exceed B4 on BOTH axes; "
             "V7/V7-R inherit Q2a TRT speed 45.78 ms/img.",
             fontsize=11.2, color=muted, va="top")
    fig.text(.928, .934,
             f"{len(beyond)} / {len(vpts)} candidates\nnot external-dominated",
             ha="right", va="top", fontsize=11.5, color=purple, weight="bold")

    cat = [
        Line2D([], [], marker="o", ls="none", ms=9, mfc=orange, mec="white",
               label="Ours: Q2a+Oracle+new S10"),
        Line2D([], [], marker="o", ls="none", ms=9, mfc="#2E6F9E", mec="white",
               label="Q2a+R1 Oracle (strongest ours, not in frontier)"),
        Line2D([], [], marker="s", ls="none", ms=8, mfc="#F4C430", mec="#7A5C00",
               label="B3 / B4 anchors"),
        Line2D([], [], marker="o", ls="none", ms=9, mfc=purple, mec="white",
               label=f"V7 / V7-R candidates (n={len(vpts)})"),
        Line2D([], [], marker="o", ls="none", ms=8, mfc="none", mec="#B39DDB",
               label=f"diagnostic probes (n={len(probes)}, excluded)"),
        Line2D([], [], marker="o", ls="none", ms=9, mfc="#E2E8EF", mec="#5A6F88",
               label="WEAVE published (ours)"),
        Line2D([], [], marker="o", ls="none", ms=9, mfc=blue, mec="white",
               label="External baselines"),
        Line2D([], [], color=blue, ls=(0, (4, 3)), lw=2,
               label="EXTERNAL-only 2-D Pareto"),
    ]
    fig.legend(handles=cat, bbox_to_anchor=(.09, .075), loc="lower left",
               ncol=4, frameon=False, fontsize=8.6, labelspacing=0.6,
               handletextpad=0.7, columnspacing=1.4)

    fig.text(.09, .038,
             "Speed: docs/v6/INFERENCE_SPEED_REPORT.md — Q2a TRT stitched 18.7 (enc+WEAVE) + 27.08 "
             "(new S10 decoder) = 45.78 ms/img, NOT end-to-end remeasured; external speeds from paper.tex "
             "main table (min / 750 imgs, RTX 3060).", fontsize=8.5, color=muted)
    fig.text(.09, .018,
             "Quality: paper_rgb_v1 / cross600, DINO-S is the common style axis (V7 evals have no CLIP-S). "
             f"{len(up_right)} real candidates exceed B4 on BOTH DINO-C and DINO-S; "
             f"{len(beyond)} candidates are not 2-D dominated by any external baseline.",
             fontsize=8.5, color=muted)

    out = DELIVER / "v7_vs_external_pareto"
    for ext in ("png", "svg", "pdf"):
        fig.savefig(out.with_suffix("." + ext), bbox_inches="tight",
                    pad_inches=.22, dpi=125 if ext == "png" else None)
    plt.close(fig)
    print("Saved:", out.with_suffix(".png"))
    print("External Pareto:", ", ".join(p[0] for p in ext_front))


if __name__ == "__main__":
    main()
