#!/usr/bin/env python3
"""Master Publication-Grade Visualization Suite for WEAVE (docs/performance & ICME 2027).

Editorial visual language inspired by V8's bubble-Pareto design, strictly fixed-canvas
with zero off-canvas stretching and zero label/legend collisions.
"""
from __future__ import annotations

import shutil
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib import patheffects as pe
import numpy as np

PERF_DIR = Path(__file__).resolve().parent
REPO = PERF_DIR.parents[1]
FIG_OUT = PERF_DIR / "figures"
ICME_FIG = REPO / "docs/icme2027/figures"

INK = "#182A42"
MUTED = "#62758B"
BLUE = "#2E6B9E"
BLUE_DARK = "#1B4965"
GREY = "#B0BCC9"
CORAL = "#ED553B"
CORAL_DARK = "#9E2A1B"
CORAL_SOFT = "#F4B8AC"
CRIMSON = "#C81D25"
PURPLE = "#6C4AB6"
TEAL = "#1F8A70"
BG = "#FCFDFE"
GRID = "#E6EDF5"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10.5,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

EXTERNAL_PTS = [
    ("StyleShot",     0.3569, 0.6949, 92.60, 0.7697, 24480.0),
    ("StyleID",       0.3885, 0.6469, 73.20, 0.5554,  5040.0),
    ("AesPA-Net",     0.6075, 0.5784, 31.00, 0.6247,  1840.0),
    ("StyTr2",        0.7023, 0.5317, 39.50, 0.7386,  3040.0),
    ("Seedream 4.5",  0.7343, 0.5314, 37.30, 0.5202,  2000.0),
    ("SaMam",         0.8239, 0.5199,  2.50, 0.3882,  1408.0),
    ("CUT",           0.8435, 0.5093,  5.70, 0.6193,   400.0),
    ("Hist2Style",    0.9083, 0.5069, 12.00, 0.4350,   200.0),
    ("SD-Turbo",      0.9998, 0.4894,  0.50, 0.0048,   400.0),
]

WEAVE_PTS = [
    ("WEAVE-Pro (Ours)",         0.7428, 0.5674, 38.00, 0.5018, 115.0, "v18_flagship"),
    ("Q2A+Oracle1+S10",          0.7345, 0.5587, 36.30, 0.5014,  68.0, "weave_front"),
    ("V9 LD4",                   0.7718, 0.5615, 35.17, 0.4872, 105.0, "weave_front"),
    ("V10 E4 (KDE)",             0.7713, 0.5616, 34.00, 0.4907, 145.0, "weave_front"),
    ("V8 N2C (2-step)",          0.7715, 0.5605, 32.83, 0.4904,  74.0, "weave_sub"),
    ("WEAVE-1Step (Ours)",       0.7793, 0.5606, 33.83, 0.4806,  28.5, "v18_flagship"),
    ("B4 Distill50 (1-step)",    0.8152, 0.5483, 22.20, 0.4396,  21.0, "weave_front"),
    ("B3 Distill20 (1-step)",    0.8270, 0.5454, 18.80, 0.4282,  21.0, "weave_front"),
    ("De-Styled Flow (τ0=0)",    0.7222, 0.5668, 18.67, 0.4549,  44.8, "weave_sub"),
    ("WEAVE (orig)",             0.8079, 0.5367, 16.70, 0.4974,  97.6, "weave_sub"),
    ("V6 D1 Gentle",             0.8994, 0.5152,  3.20, 0.3639,  22.0, "weave_sub"),
    ("V6 ScoreDiff",             0.9188, 0.5115,  4.00, 0.2079,  48.0, "weave_sub"),
    ("SDEdit+DINO Stu (τ0=0.7)", 0.7386, 0.5766, 13.33, 0.2845,  88.0, "weave_front"),
    ("V15 W96 (dim96)",          0.7506, 0.5692, 10.67, 0.2700,  72.0, "v15_scale"),
    ("V15 S1 (dim128)",          0.7767, 0.5633,  9.50, 0.2421,  86.0, "v15_scale"),
    ("V15 S2 (dim160)",          0.7862, 0.5607,  9.50, 0.2348, 102.0, "v15_scale"),
    ("V15 S3 (dim192)",          0.7920, 0.5584,  8.00, 0.2314, 122.0, "v15_scale"),
    ("V15 S4 (dim256)",          0.7962, 0.5571,  8.33, 0.2285, 165.0, "v15_scale"),
    ("V15 S6 (dim384)",          0.8083, 0.5541,  6.67, 0.2210, 260.0, "v15_scale"),
]

IDT_AFFINITY = 0.4902
TGT_CONTENT = 0.2190


def pareto_2d(pts, x_idx: int, y_idx: int):
    out = []
    for a in pts:
        dom = any(
            b[x_idx] >= a[x_idx] and b[y_idx] >= a[y_idx]
            and (b[x_idx] > a[x_idx] or b[y_idx] > a[y_idx])
            for b in pts if b is not a
        )
        if not dom:
            out.append(a)
    return sorted(out, key=lambda p: p[x_idx])


def _save_multi(fig, stem: str):
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    ICME_FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg", "pdf"):
        p = FIG_OUT / f"{stem}.{ext}"
        fig.savefig(p, dpi=300 if ext == "png" else None)
        shutil.copy2(p, ICME_FIG / f"{stem}.{ext}")
    plt.close(fig)
    print(f"[Plot] Saved {stem}.{{png,svg,pdf}}")


def _draw_throughput_key(ax, K: float):
    """Draw a clean, zero-overlap bubble throughput key in upper-right axes coords."""
    box_x, box_y, box_w, box_h = 0.575, 0.805, 0.410, 0.178
    rect = FancyBboxPatch(
        (box_x, box_y), box_w, box_h, transform=ax.transAxes,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        fc=BG, ec="#DCE4EC", lw=1.0, zorder=16,
    )
    ax.add_patch(rect)
    ax.text(box_x + box_w / 2, box_y + box_h - 0.018,
            "Bubble area ∝ GPU throughput (RTX 3060)",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=8.9, weight="bold", color=INK, zorder=17)
    xs_ax = [0.620, 0.705, 0.805, 0.922]
    sp_vals = [1, 5, 15, 35]
    for x_pos, v in zip(xs_ax, sp_vals):
        ax.scatter([x_pos], [box_y + 0.082], s=min(820.0, K * v),
                   transform=ax.transAxes, marker="o",
                   fc="#DCE4EC", ec="#7F95AB", lw=1.1, zorder=17)
        ax.text(x_pos, box_y + 0.008, f"{v} img/s",
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=8.1, color=MUTED, zorder=18)


def plot_bubble_frontier(*, zoom: bool = False):
    x0 = 0.57 if zoom else 0.20
    ext = [p for p in EXTERNAL_PTS if p[1] >= x0 - 0.02]
    wv = [p for p in WEAVE_PTS if p[1] >= x0 - 0.02]

    K = 44.0
    area_fn = lambda ms: max(22.0, min(1650.0, K * (1000.0 / ms)))

    ext_front = pareto_2d(EXTERNAL_PTS, 1, 2)
    ext_front_names = {p[0] for p in ext_front}
    comb_front = pareto_2d(EXTERNAL_PTS + WEAVE_PTS, 1, 2)
    comb_names = {p[0] for p in comb_front}

    fig = plt.figure(figsize=(14.4, 8.9), facecolor=BG)
    ax = fig.add_axes([0.075, 0.220, 0.885, 0.630], facecolor=BG)

    ax.set_xlim(x0, 1.035)
    ymin = 0.478 if zoom else 0.475
    ymax = 0.668 if zoom else 0.720
    ax.set_ylim(ymin, ymax)

    ax.set_axisbelow(True)
    ax.grid(color=GRID, lw=0.95)
    for edge in ("bottom", "left"):
        ax.spines[edge].set_color("#B9C5D0")
        ax.spines[edge].set_linewidth(1.1)
    ax.tick_params(colors=MUTED, length=4, color="#B9C5D0", pad=6, labelsize=10)
    ax.set_xlabel("Content Preservation / DINO-C   → Higher is better",
                  fontsize=12.0, weight="bold", color=INK, labelpad=10)
    ax.set_ylabel("Style Fidelity / Style Affinity   ↑ Higher is better",
                  fontsize=12.0, weight="bold", color=INK, labelpad=10)

    ax.axhline(IDT_AFFINITY, color="#C0392B", lw=1.3, ls=(0, (6, 4)), alpha=0.82, zorder=2)
    ax.text(1.030, IDT_AFFINITY + 0.0028, "Identity style floor (0.490)",
            fontsize=8.8, color="#C0392B", va="bottom", ha="right", style="italic")
    if not zoom:
        ax.axvline(TGT_CONTENT, color="#C0392B", lw=1.3, ls=(0, (6, 4)), alpha=0.82, zorder=2)
        ax.text(TGT_CONTENT + 0.008, ymax - 0.008, "Target-copy content floor",
                fontsize=8.8, color="#C0392B", va="top", ha="left", style="italic")

    fx = [p[1] for p in ext_front]
    fy = [p[2] for p in ext_front]
    ax.plot(fx, fy, color=BLUE, lw=2.0, ls=(0, (5, 3)), zorder=3)
    ax.fill_between(fx, fy, ymin, color=BLUE, alpha=0.045, zorder=1)

    cf_vis = [p for p in comb_front if p[1] >= 0.60]
    if len(cf_vis) >= 2:
        ax.plot([p[1] for p in cf_vis], [p[2] for p in cf_vis],
                color=CORAL, lw=2.2, ls="-", alpha=0.75, zorder=4)

    for p in ext:
        hit = p[0] in ext_front_names
        s_val = area_fn(p[5])
        ax.scatter(p[1], p[2], s=s_val, marker="o",
                   fc=BLUE if hit else GREY,
                   ec="white" if s_val > 45 else BLUE,
                   lw=1.2 if s_val > 45 else 0.8,
                   alpha=0.92 if hit else 0.72,
                   zorder=6 if hit else 3)

    for p in wv:
        hit = p[0] in comb_names or p[6] == "v18_flagship"
        is_v18 = p[6] == "v18_flagship"
        is_scale = p[6] == "v15_scale"
        s_val = area_fn(p[5])
        if is_v18:
            fc, ec, lw, al, zo = CRIMSON, "#59080C", 2.0, 0.96, 10
        elif hit:
            fc, ec, lw, al, zo = CORAL, CORAL_DARK, 1.4, 0.90, 8
        elif is_scale:
            fc, ec, lw, al, zo = "#B69CE8", PURPLE, 1.0, 0.62, 5
        else:
            fc, ec, lw, al, zo = CORAL_SOFT, "#D98F80", 0.9, 0.58, 5
        ax.scatter(p[1], p[2], s=s_val, marker="o", fc=fc, ec=ec, lw=lw, alpha=al, zorder=zo)

    sc_pts = [p for p in wv if p[6] == "v15_scale"]
    if len(sc_pts) >= 2:
        ax.plot([p[1] for p in sc_pts], [p[2] for p in sc_pts],
                color=PURPLE, lw=1.5, ls=":", zorder=6)

    if not zoom:
        ext_lbls = {
            "StyleShot":    (0.405, 0.703, "left"),
            "StyleID":      (0.440, 0.656, "left"),
            "AesPA-Net":    (0.560, 0.590, "right"),
            "StyTr2":       (0.655, 0.520, "right"),
            "Seedream 4.5": (0.705, 0.512, "right"),
            "SaMam":        (0.815, 0.502, "right"),
            "CUT":          (0.858, 0.525, "left"),
            "Hist2Style":   (0.935, 0.525, "left"),
            "SD-Turbo":     (0.985, 0.481, "right"),
        }
        wv_lbls = {
            "SDEdit+DINO Stu (τ0=0.7)": (0.692, 0.606, "right"),
            "WEAVE-Pro (Ours)":         (0.765, 0.596, "left"),
            "Q2A+Oracle1+S10":          (0.680, 0.556, "right"),
            "V9 LD4":                   (0.798, 0.583, "left"),
            "WEAVE-1Step (Ours)":       (0.832, 0.570, "left"),
            "B4 Distill50 (1-step)":    (0.862, 0.554, "left"),
            "V15 S6 (dim384)":          (0.848, 0.538, "left"),
        }
    else:
        ext_lbls = {
            "AesPA-Net":    (0.618, 0.600, "left"),
            "StyTr2":       (0.672, 0.521, "right"),
            "Seedream 4.5": (0.715, 0.514, "right"),
            "SaMam":        (0.818, 0.503, "right"),
            "CUT":          (0.858, 0.524, "left"),
            "Hist2Style":   (0.922, 0.524, "left"),
            "SD-Turbo":     (0.985, 0.482, "right"),
        }
        wv_lbls = {
            "SDEdit+DINO Stu (τ0=0.7)": (0.706, 0.606, "right"),
            "De-Styled Flow (τ0=0)":    (0.688, 0.580, "right"),
            "WEAVE-Pro (Ours)":         (0.752, 0.595, "left"),
            "Q2A+Oracle1+S10":          (0.695, 0.556, "right"),
            "V9 LD4":                   (0.788, 0.581, "left"),
            "WEAVE-1Step (Ours)":       (0.822, 0.568, "left"),
            "B4 Distill50 (1-step)":    (0.856, 0.553, "left"),
            "B3 Distill20 (1-step)":    (0.868, 0.540, "left"),
            "V15 W96 (dim96)":          (0.730, 0.542, "right"),
            "V15 S6 (dim384)":          (0.835, 0.531, "left"),
        }

    for p in ext:
        if p[0] in ext_lbls:
            tx, ty, ha = ext_lbls[p[0]]
            ann = ax.annotate(
                p[0], xy=(p[1], p[2]), xytext=(tx, ty), ha=ha, va="center",
                fontsize=9.4, color=INK if p[0] in ext_front_names else MUTED,
                weight="bold" if p[0] in ext_front_names else "normal",
                arrowprops=dict(arrowstyle="-", lw=0.85, color="#9EB0C2", shrinkA=2, shrinkB=4),
                zorder=14,
            )
            ann.set_path_effects([pe.withStroke(linewidth=3.0, foreground=BG)])

    for p in wv:
        if p[0] in wv_lbls:
            tx, ty, ha = wv_lbls[p[0]]
            is_v18 = p[6] == "v18_flagship"
            col = CRIMSON if is_v18 else (PURPLE if p[6] == "v15_scale" else CORAL_DARK)
            ann = ax.annotate(
                p[0], xy=(p[1], p[2]), xytext=(tx, ty), ha=ha, va="center",
                fontsize=9.6 if is_v18 else 9.0, color=col,
                weight="bold",
                arrowprops=dict(arrowstyle="-", lw=1.0, color=col, alpha=0.65, shrinkA=2, shrinkB=4),
                zorder=15,
            )
            ann.set_path_effects([pe.withStroke(linewidth=3.0, foreground=BG)])

    _draw_throughput_key(ax, K)

    title_str = "WEAVE++ × STYLE × CONTENT × SPEED" + ("  [ZOOM]" if zoom else "")
    fig.text(0.075, 0.945, title_str, fontsize=19.5, weight="bold", color=INK, va="top")
    fig.text(0.075, 0.896,
             "Cross600 (5 Styles, 750-Task Board) · Unified S10 Referee · Bubble Area ∝ GPU Throughput",
             fontsize=10.8, color=MUTED, va="top")
    fig.text(0.960, 0.948, "35.1 img/s", ha="right", va="top", fontsize=22, color=CRIMSON, weight="bold")
    fig.text(0.960, 0.896, "WEAVE-1Step (0.66M, 28.5 ms/img)", ha="right", va="top", fontsize=9.5, color=MUTED)

    cat_handles = [
        Line2D([], [], marker="o", ls="none", ms=9.0, mfc=CRIMSON, mec="#59080C", label="WEAVE++ (WEAVE-Pro / WEAVE-1Step)"),
        Line2D([], [], marker="o", ls="none", ms=8.5, mfc=CORAL, mec=CORAL_DARK, label="WEAVE Prior Lineage"),
        Line2D([], [], marker="o", ls="none", ms=8.0, mfc="#B69CE8", mec=PURPLE, label="Self-Paired Scaling (96→384)"),
        Line2D([], [], marker="o", ls="none", ms=8.0, mfc=BLUE, mec="white", label="External Pareto Frontier"),
        Line2D([], [], color=CORAL, ls="-", lw=2.2, label="New Combined Frontier"),
        Line2D([], [], color=BLUE, ls=(0, (5, 3)), lw=2.0, label="External-Only Frontier"),
    ]
    fig.legend(handles=cat_handles, bbox_to_anchor=(0.075, 0.088), loc="lower left",
               ncol=3, frameon=False, fontsize=9.3, columnspacing=1.8)

    foot = textwrap.fill(
        "Note: WEAVE-Pro (38.0% CSD Top-1, 0.5674 Affinity) and WEAVE-1Step (33.8% Top-1, 0.7793 DINO-C, 28.5 ms/img) "
        "strictly dominate StyTr2, Seedream 4.5, SaMam, CUT, and Hist2Style across style, content, and throughput.",
        width=138,
    )
    fig.text(0.075, 0.024, foot, fontsize=8.6, color=MUTED, va="bottom")

    stem = "weave_style_content_speed_zoom" if zoom else "weave_style_content_speed"
    _save_multi(fig, stem)


def plot_csd_specificity_frontier():
    x0 = 0.56
    ext = [p for p in EXTERNAL_PTS if p[1] >= x0]
    wv = [p for p in WEAVE_PTS if p[1] >= x0]

    K = 44.0
    area_fn = lambda ms: max(24.0, min(1650.0, K * (1000.0 / ms)))

    ext_front = pareto_2d(ext, 1, 3)
    ext_front_names = {p[0] for p in ext_front}
    comb_front = pareto_2d(ext + wv, 1, 3)
    comb_names = {p[0] for p in comb_front}

    fig = plt.figure(figsize=(14.4, 8.9), facecolor=BG)
    ax = fig.add_axes([0.075, 0.220, 0.885, 0.630], facecolor=BG)
    ax.set_xlim(x0, 1.02)
    ax.set_ylim(-2.0, 58.0)

    ax.set_axisbelow(True)
    ax.grid(color=GRID, lw=0.95)
    for edge in ("bottom", "left"):
        ax.spines[edge].set_color("#B9C5D0")
        ax.spines[edge].set_linewidth(1.1)
    ax.tick_params(colors=MUTED, length=4, color="#B9C5D0", pad=6, labelsize=10)
    ax.set_xlabel("Content Preservation / DINO-C   → Higher is better",
                  fontsize=12.0, weight="bold", color=INK, labelpad=10)
    ax.set_ylabel("5-Way Target Style Classification / CSD Top-1 (%)   ↑ Higher is better",
                  fontsize=12.0, weight="bold", color=INK, labelpad=10)

    ax.axhline(20.0, color="#7F8C8D", lw=1.1, ls=":", alpha=0.75, zorder=2)
    ax.text(1.015, 20.7, "5-class uniform random (20%)", fontsize=8.8, color="#7F8C8D", ha="right", style="italic")

    ax.plot([p[1] for p in ext_front], [p[3] for p in ext_front],
            color=BLUE, lw=2.0, ls=(0, (5, 3)), zorder=3)
    ax.fill_between([p[1] for p in ext_front], [p[3] for p in ext_front], -2.0,
                    color=BLUE, alpha=0.04, zorder=1)
    ax.plot([p[1] for p in comb_front], [p[3] for p in comb_front],
            color=CRIMSON, lw=2.2, ls="-", alpha=0.72, zorder=4)

    for p in ext:
        hit = p[0] in ext_front_names
        s_val = area_fn(p[5])
        ax.scatter(p[1], p[3], s=s_val, marker="o",
                   fc=BLUE if hit else GREY, ec="white", lw=1.1,
                   alpha=0.92 if hit else 0.72, zorder=6 if hit else 3)

    for p in wv:
        is_v18 = p[6] == "v18_flagship"
        is_scale = p[6] == "v15_scale"
        hit = p[0] in comb_names
        s_val = area_fn(p[5])
        if is_v18:
            fc, ec, lw, zo = CRIMSON, "#59080C", 2.1, 11
        elif is_scale:
            fc, ec, lw, zo = "#B69CE8", PURPLE, 1.1, 6
        elif hit:
            fc, ec, lw, zo = CORAL, CORAL_DARK, 1.4, 9
        else:
            fc, ec, lw, zo = CORAL_SOFT, "#D98F80", 1.0, 5
        ax.scatter(p[1], p[3], s=s_val, marker="o", fc=fc, ec=ec, lw=lw, alpha=0.92, zorder=zo)

    ax.annotate(
        "", xy=(0.8083, 6.67), xytext=(0.7506, 10.67),
        arrowprops=dict(arrowstyle="->,head_width=0.35,head_length=0.5", lw=2.2, color=PURPLE, ls="--"),
        zorder=12,
    )
    t_p = ax.text(0.680, 4.8, "Paradox 1: Self-Paired Scaling (3.2M→34.2M)\nincreases source-lock (10.7% → 6.7%)",
                  fontsize=8.7, color=PURPLE, weight="bold", ha="left", va="top")
    t_p.set_path_effects([pe.withStroke(linewidth=3, foreground=BG)])

    ax.annotate(
        "", xy=(0.7815, 33.4), xytext=(0.8135, 22.8),
        arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.55", lw=2.6, color=CRIMSON),
        zorder=12,
    )
    t_s = ax.text(0.825, 28.5, "+11.63pp Single-Step Leap!\n(Whitened DINO + u_tgt−u_src,\nwrong-label = 1.33%)",
                  fontsize=8.9, color=CRIMSON, weight="bold", ha="left", va="center")
    t_s.set_path_effects([pe.withStroke(linewidth=3, foreground=BG)])

    ax.annotate(
        "", xy=(0.744, 37.8), xytext=(0.770, 35.3),
        arrowprops=dict(arrowstyle="->,head_width=0.38,head_length=0.5", lw=2.2, color=CRIMSON),
        zorder=12,
    )

    lbls = {
        "StyTr2":                   (0.672, 42.5, "right", INK),
        "AesPA-Net":                (0.605, 34.5, "left", INK),
        "Seedream 4.5":             (0.692, 35.0, "right", INK),
        "SaMam":                    (0.824, -0.3, "center", MUTED),
        "CUT":                      (0.858,  8.0, "left", MUTED),
        "Hist2Style":               (0.922, 14.8, "left", INK),
        "SD-Turbo":                 (0.985,  3.4, "right", MUTED),
        "WEAVE-Pro (Ours)":         (0.742, 44.0, "center", CRIMSON),
        "Q2A+Oracle1+S10":          (0.682, 39.2, "right", CORAL_DARK),
        "V9 LD4":                   (0.792, 38.6, "left", CORAL_DARK),
        "WEAVE-1Step (Ours)":       (0.822, 35.0, "left", CRIMSON),
        "B4 Distill50 (1-step)":    (0.848, 22.0, "left", CORAL_DARK),
        "De-Styled Flow (τ0=0)":    (0.682, 21.5, "right", CORAL_DARK),
        "SDEdit+DINO Stu (τ0=0.7)": (0.695, 14.5, "right", CORAL_DARK),
        "V15 W96 (dim96)":          (0.715,  9.8, "right", PURPLE),
    }
    for p in ext + wv:
        if p[0] in lbls:
            tx, ty, ha, col = lbls[p[0]]
            ann = ax.annotate(
                p[0], xy=(p[1], p[3]), xytext=(tx, ty), ha=ha, va="center",
                fontsize=9.4, color=col, weight="bold",
                arrowprops=dict(arrowstyle="-", lw=0.9, color=col, alpha=0.6, shrinkA=2, shrinkB=4),
                zorder=15,
            )
            ann.set_path_effects([pe.withStroke(linewidth=3.0, foreground=BG)])

    _draw_throughput_key(ax, K)

    fig.text(0.075, 0.945, "WEAVE++ × SPECIFICITY (CSD TOP-1) × CONTENT × SPEED",
             fontsize=19.5, weight="bold", color=INK, va="top")
    fig.text(0.075, 0.896,
             "Cross600 5-Way Target Classification: Breaking Past Seedream 4.5 & Amortizing into 1 Step",
             fontsize=10.8, color=MUTED, va="top")
    fig.text(0.960, 0.948, "38.0% Top-1", ha="right", va="top", fontsize=22, color=CRIMSON, weight="bold")
    fig.text(0.960, 0.896, "WEAVE-Pro beats Seedream 4.5 (37.3%) & 17× faster", ha="right", va="top", fontsize=9.5, color=MUTED)

    cat_handles = [
        Line2D([], [], marker="o", ls="none", ms=9.0, mfc=CRIMSON, mec="#59080C", label="WEAVE++ (WEAVE-Pro 38.0% / 1Step 33.8%)"),
        Line2D([], [], marker="o", ls="none", ms=8.5, mfc=CORAL, mec=CORAL_DARK, label="WEAVE Prior / Ablation Lineage"),
        Line2D([], [], marker="o", ls="none", ms=8.0, mfc="#B69CE8", mec=PURPLE, label="Self-Paired Scaling (96→384)"),
        Line2D([], [], marker="o", ls="none", ms=8.0, mfc=BLUE, mec="white", label="External Baselines"),
        Line2D([], [], color=CRIMSON, ls="-", lw=2.2, label="WEAVE++ Combined Pareto Front"),
        Line2D([], [], color=BLUE, ls=(0, (5, 3)), lw=2.0, label="External-Only Pareto Front"),
    ]
    fig.legend(handles=cat_handles, bbox_to_anchor=(0.075, 0.088), loc="lower left",
               ncol=3, frameon=False, fontsize=9.3, columnspacing=1.8)

    foot = textwrap.fill(
        "Note: CSD-ViT-L 5-way target classification on 600 cross-style pairs. Self-paired flow scaling degrades transfer via source-lock, "
        "whereas decoupled canonical structure + whitened DINO layout enables +11.63pp single-step amortization.",
        width=138,
    )
    fig.text(0.075, 0.024, foot, fontsize=8.6, color=MUTED, va="bottom")

    _save_multi(fig, "weave_specificity_frontier")


def plot_icme_three_paradoxes():
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6), facecolor=BG)
    for ax in axes:
        ax.set_facecolor(BG)
        ax.grid(color=GRID, lw=0.85, axis="y")
        for edge in ("bottom", "left"):
            ax.spines[edge].set_color("#B9C5D0")

    # (a) S10 Decoder Paradox
    ax = axes[0]
    models_a = ["SD-VAE\n(49.5M, 1.0×)", "Pruned S10\n(10.7M, 2.86×)"]
    top1_a = [34.50, 37.67]
    lpips_a = [51.49, 50.34]
    x = np.arange(2)
    w = 0.34
    b1 = ax.bar(x - w/2, top1_a, w, color=[GREY, CRIMSON], edgecolor=INK, lw=0.9, label="CSD Top-1 (%) ↑")
    b2 = ax.bar(x + w/2, lpips_a, w, color=["#D0D7DE", TEAL], edgecolor=INK, lw=0.9, label="LPIPS ×100 ↓")
    for bar, val in zip(b1, top1_a):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.9, f"{val:.2f}%", ha="center", va="bottom", fontsize=9, weight="bold", color=INK)
    for bar, val in zip(b2, lpips_a):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.9, f"{val/100:.3f}", ha="center", va="bottom", fontsize=9, weight="bold", color=TEAL)
    ax.set_xticks(x)
    ax.set_xticklabels(models_a, fontsize=9.5, weight="bold", color=INK)
    ax.set_ylim(0, 68)
    ax.set_title("(a) Pruned-Decoder Paradox (+3.17pp, p<0.01)", fontsize=10.6, weight="bold", color=INK, pad=10)
    ax.legend(frameon=False, fontsize=8.8, loc="upper left")

    # (b) Capacity Scaling Paradox
    ax = axes[1]
    params_m = [0.13, 0.37, 1.17, 3.16, 4.13, 8.92, 15.52, 34.19]
    gen_t0 = [20.0, 20.5, 28.0, 36.33, 42.67, 45.00, 45.00, 49.33]
    edit_t07 = [8.83, 9.00, 8.83, 10.67, 9.50, 8.00, 8.33, 6.67]
    ax.grid(color=GRID, lw=0.85, which="both")
    ax.plot(params_m, gen_t0, marker="o", lw=2.2, color=BLUE, label=r"Pure Gen ($\tau_0=0$) Top-1 ↑")
    ax.plot(params_m, edit_t07, marker="s", lw=2.4, color=CRIMSON, label=r"SDEdit ($\tau_0=0.7$) Top-1 ↑")
    ax.set_xscale("log")
    ax.set_ylim(0, 58)
    ax.set_xlabel("U-Net Parameters (Millions, log scale)", fontsize=9.6, weight="bold", color=INK)
    ax.set_ylabel("Cross600 CSD Top-1 (%)", fontsize=9.6, weight="bold", color=INK)
    ax.annotate("49.3%", xy=(34.19, 49.33), xytext=(12.5, 51.5), fontsize=9, weight="bold", color=BLUE)
    ax.annotate("6.67% (Source-Lock!)", xy=(34.19, 6.67), xytext=(4.0, 15.5),
                fontsize=8.8, weight="bold", color=CRIMSON,
                arrowprops=dict(arrowstyle="->", color=CRIMSON, lw=1.3))
    ax.set_title("(b) Self-Paired Capacity Paradox (260× Scale)", fontsize=10.6, weight="bold", color=INK, pad=10)
    ax.legend(frameon=False, fontsize=8.6, loc="upper left")

    # (c) Causal Target-Style Control
    ax = axes[2]
    arms_c = ["Post-Hoc\nRGB-H2", "SDEdit Stu\n(τ0=0.7)", "De-Styled\nFlow (τ0=0)", "WEAVE-1Step\n(Ours)"]
    corr_v = [12.33, 11.67, 15.00, 33.83]
    wrong_v = [13.00, 10.67, 5.17, 1.33]
    xc = np.arange(4)
    w_c = 0.34
    b_c = ax.bar(xc - w_c/2, corr_v, w_c, color=CRIMSON, edgecolor=INK, lw=0.9, label="Correct Target Label")
    b_w = ax.bar(xc + w_c/2, wrong_v, w_c, color=GREY, edgecolor=INK, lw=0.9, label="Shifted Wrong Label")
    for bar, val in zip(b_c, corr_v):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.8, f"{val:.1f}%", ha="center", va="bottom", fontsize=8.4, weight="bold", color=CRIMSON)
    for bar, val in zip(b_w, wrong_v):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.8, f"{val:.1f}%", ha="center", va="bottom", fontsize=8.4, weight="bold", color=MUTED)
    ax.annotate("+32.5pp Causal\nSpecificity!", xy=(3 - w_c/2 - 0.04, 31.0), xytext=(1.35, 27.5),
                fontsize=8.8, weight="bold", color=CRIMSON, ha="center",
                arrowprops=dict(arrowstyle="->", color=CRIMSON, lw=1.4))
    ax.set_xticks(xc)
    ax.set_xticklabels(arms_c, fontsize=8.6, weight="bold", color=INK)
    ax.set_ylim(0, 43)
    ax.set_title("(c) Causal Specificity Ladder (Correct vs Wrong ID)", fontsize=10.6, weight="bold", color=INK, pad=10)
    ax.legend(frameon=False, fontsize=8.6, loc="upper left")

    fig.tight_layout(pad=1.3)
    _save_multi(fig, "icme_three_paradoxes")


def plot_icme_architecture():
    fig = plt.figure(figsize=(14.2, 4.5), facecolor="white")
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], facecolor="white")
    ax.set_xlim(0, 14.2)
    ax.set_ylim(0, 4.5)
    ax.axis("off")

    def box(x, y, w, h, title, sub, fc="#F4F7FA", ec=INK, tc=INK, lw=1.5, title_fs=10.0):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12",
                           fc=fc, ec=ec, lw=lw, zorder=3)
        ax.add_patch(p)
        ax.text(x + w/2, y + h*0.67, title, ha="center", va="center",
                fontsize=title_fs, weight="bold", color=tc, zorder=4)
        ax.text(x + w/2, y + h*0.28, sub, ha="center", va="center",
                fontsize=8.2, color=MUTED, zorder=4)

    def arrow(x1, y1, x2, y2, col=INK, ls="-", lw=1.6, label="", lx=None, ly=None):
        arr = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->,head_width=5,head_length=6",
                              color=col, lw=lw, linestyle=ls, zorder=5)
        ax.add_patch(arr)
        if label:
            tx = (x1 + x2)/2 if lx is None else lx
            ty = (y1 + y2)/2 + 0.18 if ly is None else ly
            t = ax.text(tx, ty, label, ha="center", va="center",
                        fontsize=8.2, weight="bold", color=col, zorder=8)
            t.set_path_effects([pe.withStroke(linewidth=3.5, foreground="white")])

    ax.text(2.35, 4.22, "Stage I: Style-Stripped Structural Bottleneck", ha="center", fontsize=10.6, weight="bold", color=BLUE_DARK)
    ax.text(7.60, 4.22, "Stage II: Dual-Mode Latent Transport & Amortization", ha="center", fontsize=10.6, weight="bold", color=CRIMSON)
    ax.text(12.45, 4.22, "Stage III: Pruned Decoding", ha="center", fontsize=10.6, weight="bold", color=TEAL)

    box(0.20, 1.55, 1.35, 1.25, "Source Image\n$x_{\\mathrm{src}}$ / $z_{\\mathrm{src}}$", "4×64×64 Latent", fc="#EEF3F8", ec=BLUE)
    box(2.00, 2.60, 2.35, 1.15, "Haar Canonicalizer $C_1$", "Band Norm + HH Atten.\nBase Anchor $c_{\\mathrm{canon}}$", fc="#EBF3FA", ec=BLUE)
    box(2.00, 0.60, 2.35, 1.48, "Gate-0 Structural Maps", "BT.601 Multi-Scale LCN $C_{64}$\n+ Whitened Gray DINO $\\tilde{P}_{16}$\n(Style Leak: 74% → 18.3%!)", fc="#FFF4F2", ec=CRIMSON)

    arrow(1.60, 2.45, 1.96, 3.08, BLUE)
    arrow(1.60, 1.85, 1.96, 1.38, CRIMSON)

    box(4.90, 2.60, 2.35, 1.15, "Base Operator $B_4$", "0.22M C3UNet One-Shot\n$z_0 = B_4(c_{\\mathrm{canon}}, s_{\\mathrm{tgt}})$", fc="#F4F7FA", ec=INK)
    box(7.90, 2.60, 2.65, 1.15, "WEAVE-Max (LD4+ Teacher)", "Prior-Calibrated $T - \\alpha S$ R1\n+ SPD + Top-30% Risk Mask\n(38.00% Top-1, Beats Seedream)", fc="#FFF0ED", ec=CRIMSON, tc=CRIMSON, lw=1.8)

    box(5.25, 0.60, 4.95, 1.38, "WEAVE-Fast: Zero-Identity DINO Corrector (0.44M)",
        "$\\Delta z = G_\\theta(z_0, c_{\\mathrm{canon}}, C_{64}, \\tilde{P}_{16}, e_{s_{\\mathrm{tgt}}} - e_{s_{\\mathrm{src}}}) \\cdot \\mathbf{1}_{[s_{\\mathrm{tgt}} \\neq s_{\\mathrm{src}}]}$\n"
        "1-Step Pure Feed-Forward · cos = 0.744 · 33.83% Top-1 (vs 1.33% Wrong-ID!)",
        fc="#FFEAE5", ec=CRIMSON, tc=CRIMSON, lw=2.0)

    arrow(4.40, 3.18, 4.86, 3.18, BLUE)
    arrow(7.30, 3.18, 7.86, 3.18, CRIMSON, label="3-step", lx=7.58, ly=3.38)
    arrow(6.08, 2.56, 6.38, 2.02, INK, label="$z_0$", lx=6.02, ly=2.28)
    arrow(4.40, 1.30, 5.20, 1.30, CRIMSON, label="$(C_{64}, \\tilde{P}_{16})$", lx=4.80, ly=1.52)
    arrow(9.15, 2.56, 8.85, 2.02, PURPLE, ls="--", lw=1.5, label="Distill 2.5k Pairs", lx=9.55, ly=2.28)

    box(11.20, 1.45, 2.75, 1.55, "Pruned Decoder S10", "10.66M (−78.5% vs SD-VAE)\n4-Stream Taylor Pruning\n2.86× Faster · +3.17pp Top-1", fc="#E8F6F3", ec=TEAL, tc=TEAL, lw=1.8)

    arrow(10.60, 3.18, 11.50, 3.04, CRIMSON, label="Max Mode", lx=11.02, ly=3.34)
    arrow(10.25, 1.30, 11.16, 1.95, CRIMSON, lw=2.0, label="$z_0 + \\Delta z$ (28.5 ms)", lx=10.68, ly=1.36)

    _save_multi(fig, "icme_architecture")


if __name__ == "__main__":
    plot_bubble_frontier(zoom=False)
    plot_bubble_frontier(zoom=True)
    plot_csd_specificity_frontier()
    plot_icme_three_paradoxes()
    plot_icme_architecture()
