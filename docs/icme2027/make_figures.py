#!/usr/bin/env python3
"""Generate standalone ICME 2027 quantitative figures for PRISM (V19-V20 Quotient-Fiber & Coarse-16 Core-SLF)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
from matplotlib import patheffects as pe
import numpy as np

ICME_DIR = Path(__file__).resolve().parent
FIG_DIR = ICME_DIR / "figures"

METHOD = "CLEAVE"
FLOW_NAME = "CLEAVE P-Core (Ours)"
FAST_NAME = "CLEAVE E-Core (Ours)"
DEC_NAME = "CLEAVE-Dec (Pruned)"

INK = "#182A42"
MUTED = "#62758B"
BLUE = "#2E6B9E"
GREY = "#B0BCC9"
CORAL = "#ED553B"
CORAL_DARK = "#9E2A1B"
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

# (name, dino_c, affinity, top1, lpips, ms_per_img)
EXTERNAL_PTS = [
    ("StyleShot",     0.3569, 0.6949, 92.60, 0.7697, 24480.0),
    ("StyleID",       0.3885, 0.6469, 73.20, 0.5554,  5040.0),
    ("StyleGallery",  0.6113, 0.6432, 52.50, 0.5576, 30400.0),
    ("AesPA-Net",     0.6075, 0.5784, 31.00, 0.6247,  1840.0),
    ("StyTr2",        0.7023, 0.5317, 39.50, 0.7386,  3040.0),
    ("Seedream 4.5",  0.7343, 0.5314, 37.30, 0.5202,  2000.0),
    ("SaMam",         0.8239, 0.5199,  2.50, 0.3882,  1408.0),
    ("CUT",           0.8435, 0.5093,  5.70, 0.6193,   400.0),
    ("Hist2Style",    0.9083, 0.5069, 12.00, 0.4350,   200.0),
    ("SD-Turbo",      0.9998, 0.4894,  0.50, 0.0048,   400.0),
]

# Only the two representative flagship regimes of PRISM (Unified Narrative, RTX 3060)
OURS_PTS = [
    (FLOW_NAME,                   0.5335, 0.6584, 60.50, 0.6022, 541.0, "flagship"),
    (FAST_NAME,                   0.7440, 0.5655, 44.33, 0.5057,  58.5, "flagship"),
]

IDT_AFFINITY = 0.4902


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


def _save(fig, stem: str):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(FIG_DIR / f"{stem}.{ext}", dpi=300 if ext == "png" else None)
    plt.close(fig)
    print(f"[ICME Fig] Saved {stem}.{{pdf,png,svg}}")


def _draw_throughput_key(ax, K: float):
    box_x, box_y, box_w, box_h = 0.565, 0.800, 0.420, 0.182
    rect = FancyBboxPatch(
        (box_x, box_y), box_w, box_h, transform=ax.transAxes,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        fc=BG, ec="#DCE4EC", lw=1.0, zorder=16,
    )
    ax.add_patch(rect)
    ax.text(box_x + box_w / 2, box_y + box_h - 0.018,
            "Bubble area ∝ GPU throughput (img/s)",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=8.9, weight="bold", color=INK, zorder=17)
    xs_ax = [0.612, 0.698, 0.798, 0.918]
    sp_vals = [1, 8, 25, 58]
    for x_pos, v in zip(xs_ax, sp_vals):
        ax.scatter([x_pos], [box_y + 0.084], s=min(820.0, K * v),
                   transform=ax.transAxes, marker="o",
                   fc="#DCE4EC", ec="#7F95AB", lw=1.1, zorder=17)
        ax.text(x_pos, box_y + 0.008, f"{v} img/s",
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=8.0, color=MUTED, zorder=18)


def plot_fig1_pareto_zoom():
    x0 = 0.47
    ext = [p for p in EXTERNAL_PTS if p[1] >= x0]
    ours = [p for p in OURS_PTS if p[1] >= x0]
    K = 15.0
    area_fn = lambda ms: max(22.0, min(920.0, K * (1000.0 / ms)))

    ext_front = pareto_2d(ext, 1, 2)
    ext_front_names = {p[0] for p in ext_front}
    comb_front = pareto_2d(ext + [p for p in ours if p[6] != "scale"], 1, 2)

    fig = plt.figure(figsize=(11.5, 7.6), facecolor=BG)
    ax = fig.add_axes([0.085, 0.215, 0.880, 0.635], facecolor=BG)
    ax.set_xlim(x0, 1.03)
    ax.set_ylim(0.478, 0.692)

    ax.set_axisbelow(True)
    ax.grid(color=GRID, lw=0.95)
    for edge in ("bottom", "left"):
        ax.spines[edge].set_color("#B9C5D0")
        ax.spines[edge].set_linewidth(1.1)
    ax.tick_params(colors=MUTED, length=4, color="#B9C5D0", pad=5, labelsize=10)
    ax.set_xlabel("Content Preservation / DINO-C   → Higher is better",
                  fontsize=11.8, weight="bold", color=INK, labelpad=8)
    ax.set_ylabel("Style Fidelity / Style Affinity   ↑ Higher is better",
                  fontsize=11.8, weight="bold", color=INK, labelpad=8)

    ax.axhline(IDT_AFFINITY, color="#C0392B", lw=1.3, ls=(0, (6, 4)), alpha=0.82, zorder=2)
    ax.text(1.025, IDT_AFFINITY + 0.0028, "Identity style floor (0.490)",
            fontsize=8.8, color="#C0392B", va="bottom", ha="right", style="italic")

    ax.plot([p[1] for p in ext_front], [p[2] for p in ext_front],
            color=BLUE, lw=2.0, ls=(0, (5, 3)), zorder=3)
    ax.fill_between([p[1] for p in ext_front], [p[2] for p in ext_front], 0.478,
                    color=BLUE, alpha=0.045, zorder=1)
    ax.plot([p[1] for p in comb_front], [p[2] for p in comb_front],
            color=CRIMSON, lw=2.4, ls="-", alpha=0.84, zorder=4)

    for p in ext:
        hit = p[0] in ext_front_names
        s_val = area_fn(p[5])
        ax.scatter(p[1], p[2], s=s_val, marker="o",
                   fc=BLUE if hit else GREY, ec="white", lw=1.1,
                   alpha=0.92 if hit else 0.72, zorder=6)

    for p in ours:
        is_flag = p[6] == "flagship"
        is_scale = p[6] == "scale"
        s_val = area_fn(p[5])
        if is_flag:
            fc, ec, lw, zo = CRIMSON, "#59080C", 2.0, 11
        elif is_scale:
            fc, ec, lw, zo = "#B69CE8", PURPLE, 1.0, 5
        else:
            fc, ec, lw, zo = CORAL, CORAL_DARK, 1.3, 8
        ax.scatter(p[1], p[2], s=s_val, marker="o", fc=fc, ec=ec, lw=lw, alpha=0.92, zorder=zo)

    sc_pts = [p for p in ours if p[6] == "scale"]
    ax.plot([p[1] for p in sc_pts], [p[2] for p in sc_pts], color=PURPLE, lw=1.5, ls=":", zorder=6)

    lbls = {
        FLOW_NAME:                (0.535, 0.680, "center", CRIMSON),
        FAST_NAME:                (0.745, 0.590, "center", CRIMSON),
        "StyleGallery":           (0.615, 0.620, "left", BLUE),
        "AesPA-Net":              (0.585, 0.562, "right", MUTED),
        "StyTr2":                 (0.672, 0.517, "right", INK),
        "Seedream 4.5":           (0.722, 0.512, "right", INK),
        "SaMam":                  (0.818, 0.503, "right", MUTED),
        "CUT":                    (0.858, 0.524, "left", MUTED),
        "Hist2Style":             (0.916, 0.523, "left", INK),
        "SD-Turbo":               (0.985, 0.483, "right", MUTED),
    }
    for p in ext + ours:
        if p[0] in lbls:
            tx, ty, ha, col = lbls[p[0]]
            ann = ax.annotate(
                p[0], xy=(p[1], p[2]), xytext=(tx, ty), ha=ha, va="center",
                fontsize=9.2, color=col, weight="bold",
                arrowprops=dict(arrowstyle="-", lw=0.95, color=col, alpha=0.65, shrinkA=2, shrinkB=4),
                zorder=15,
            )
            ann.set_path_effects([pe.withStroke(linewidth=3.0, foreground=BG)])

    _draw_throughput_key(ax, K)

    fig.text(0.085, 0.952, f"{METHOD} × STYLE × CONTENT × SPEED",
             fontsize=17.5, weight="bold", color=INK, va="top")
    fig.text(0.085, 0.900,
             "600 Cross-Style Pairs · PRISM-Flow Beats StyleGallery (0.658 vs 0.643 Aff) at 226× Higher Throughput",
             fontsize=9.3, color=MUTED, va="top")
    fig.text(0.965, 0.954, "5.97 img/s", ha="right", va="top", fontsize=19.5, color=CRIMSON, weight="bold")
    fig.text(0.965, 0.900, f"{METHOD} (1-Step, 58.5 ms cached) · {METHOD}-Flow (541 ms)", ha="right", va="top", fontsize=8.6, color=MUTED)

    cat_handles = [
        Line2D([], [], marker="o", ls="none", ms=9.0, mfc=CRIMSON, mec="#59080C", label=f"{METHOD} (RTX 3060: 1-Step 168ms / Flow 680ms E2E)"),
        Line2D([], [], marker="o", ls="none", ms=8.0, mfc=BLUE, mec="white", label="Published / Commercial Baselines (incl. StyleGallery)"),
        Line2D([], [], color=CRIMSON, ls="-", lw=2.3, label=f"{METHOD} Pareto Frontier"),
        Line2D([], [], color=BLUE, ls=(0, (5, 3)), lw=2.0, label="External-Only Frontier"),
    ]
    fig.legend(handles=cat_handles, bbox_to_anchor=(0.085, 0.045), loc="lower left",
               ncol=2, frameon=False, fontsize=8.8, columnspacing=1.4)

    _save(fig, "fig1_prism_pareto")


def plot_fig2_three_paradoxes():
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.15), facecolor=BG)
    for ax in axes:
        ax.set_facecolor(BG)
        ax.grid(color=GRID, lw=0.85, axis="y")
        for edge in ("bottom", "left"):
            ax.spines[edge].set_color("#B9C5D0")

    # (a) Channel 1 — Conditioning Space: Source-Style Leakage & Stiefel Quotient
    ax = axes[0]
    cats_a = ["Raw\nCNX", "Stiefel\nQ32", "Struct\nΔC×100", "Style\nΔS×100"]
    vals_a = [91.33, 20.67, 74.2, 1.2]
    lbls_a = ["91.3%", "20.7%", "+7.4", "+0.1"]
    cols_a = [CRIMSON, TEAL, BLUE, GREY]
    xa = np.arange(4)
    bars_a = ax.bar(xa, vals_a, 0.54, color=cols_a, edgecolor=INK, lw=0.9, zorder=4)
    ax.axhline(20.0, color="#64748B", lw=1.1, ls="--", zorder=2)
    ax.text(3.45, 23.0, "20% Chance", ha="right", fontsize=8.4, color="#475569", style="italic", weight="bold")
    ax.axvline(1.5, color="#CBD5E1", lw=1.0, ls=":", zorder=2)
    for bar, lbl, col in zip(bars_a, lbls_a, cols_a):
        t_col = CRIMSON if col == CRIMSON else (TEAL if col == TEAL else (BLUE if col == BLUE else MUTED))
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2.2, lbl,
                ha="center", va="bottom", fontsize=8.8, weight="bold", color=t_col)
    ax.set_xticks(xa)
    ax.set_xticklabels(cats_a, fontsize=8.6, weight="bold", color=INK)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Probe (%) / Δ×100", fontsize=9.2, weight="bold", color=INK)
    ax.set_title("(a) Conditioning Leak", fontsize=10.5, weight="bold", color=INK, pad=6)

    # (b) Channel 2 — Trajectory Space: 260× Self-Paired Scaling Paradox
    ax = axes[1]
    params_m = [0.13, 0.37, 1.17, 3.16, 4.13, 8.92, 15.52, 34.19]
    gen_t0 = [20.0, 20.5, 28.0, 36.33, 42.67, 45.00, 45.00, 49.33]
    edit_t07 = [8.83, 9.00, 8.83, 10.67, 9.50, 8.00, 8.33, 6.67]
    ax.grid(color=GRID, lw=0.85, which="both")
    ax.plot(params_m, gen_t0, marker="o", ms=4.5, lw=2.0, color=BLUE, label=r"Gen ($\tau_0\!=\!0$)")
    ax.plot(params_m, edit_t07, marker="s", ms=4.5, lw=2.2, color=CRIMSON, label=r"Edit ($\tau_0\!=\!0.7$)")
    ax.set_xscale("log")
    ax.set_ylim(0, 58)
    ax.set_xlabel("Params (M, log)", fontsize=9.2, weight="bold", color=INK)
    ax.set_ylabel("CSD Top-1 (%)", fontsize=9.2, weight="bold", color=INK)
    ax.annotate("49.3%", xy=(34.19, 49.33), xytext=(8.5, 50.5), fontsize=8.8, weight="bold", color=BLUE)
    ax.annotate("6.67% (Lock!)", xy=(34.19, 6.67), xytext=(1.1, 17.5),
                fontsize=8.6, weight="bold", color=CRIMSON,
                arrowprops=dict(arrowstyle="->", color=CRIMSON, lw=1.2))
    ax.set_title("(b) 260× Scaling Lock", fontsize=10.5, weight="bold", color=INK, pad=6)
    ax.legend(frameon=False, fontsize=8.2, loc="upper left")

    # (c) Channel 3 — Decoding Space: Texture Retraction Paradox
    ax = axes[2]
    models_b = ["Frozen\n49.5M", "Wiki-FT\n49.5M", "CLEAVE-Dec\n10.7M"]
    top1_b = [53.50, 55.83, 60.33]
    lpips_b = [60.27, 66.61, 60.22]
    x = np.arange(3)
    w = 0.35
    b1 = ax.bar(x - w/2, top1_b, w, color=[GREY, "#8FA2B2", CRIMSON], edgecolor=INK, lw=0.9, label="Top-1 (%) ↑")
    b2 = ax.bar(x + w/2, lpips_b, w, color=["#D0D7DE", "#B8C4CE", TEAL], edgecolor=INK, lw=0.9, label="LPIPS×100 ↓")
    for bar, val in zip(b1, top1_b):
        ax.text(bar.get_x() + bar.get_width()/2, val + 1.0, f"{val:.1f}", ha="center", va="bottom", fontsize=8.4, weight="bold", color=INK)
    for bar, val in zip(b2, lpips_b):
        ax.text(bar.get_x() + bar.get_width()/2, val + 1.0, f".{int(round(val*10)):03d}", ha="center", va="bottom", fontsize=8.2, weight="bold", color=TEAL)
    ax.set_xticks(x)
    ax.set_xticklabels(models_b, fontsize=8.5, weight="bold", color=INK)
    ax.set_ylim(0, 79)
    ax.set_title("(c) Decoder Retraction", fontsize=10.5, weight="bold", color=INK, pad=6)
    ax.legend(frameon=False, fontsize=8.1, loc="upper left")

    fig.tight_layout(pad=0.65)
    _save(fig, "fig2_three_paradoxes")


def plot_fig4_specificity_frontier():
    x0 = 0.47
    ext = [p for p in EXTERNAL_PTS if p[1] >= x0]
    ours = [p for p in OURS_PTS if p[1] >= x0]
    K = 15.0
    area_fn = lambda ms: max(22.0, min(920.0, K * (1000.0 / ms)))

    ext_front = pareto_2d(ext, 1, 3)
    ext_front_names = {p[0] for p in ext_front}
    comb_front = pareto_2d(ext + ours, 1, 3)
    comb_names = {p[0] for p in comb_front}

    fig = plt.figure(figsize=(11.5, 7.6), facecolor=BG)
    ax = fig.add_axes([0.085, 0.215, 0.880, 0.635], facecolor=BG)
    ax.set_xlim(x0, 1.02)
    ax.set_ylim(-2.0, 70.0)

    ax.set_axisbelow(True)
    ax.grid(color=GRID, lw=0.95)
    for edge in ("bottom", "left"):
        ax.spines[edge].set_color("#B9C5D0")
        ax.spines[edge].set_linewidth(1.1)
    ax.tick_params(colors=MUTED, length=4, color="#B9C5D0", pad=5, labelsize=10)
    ax.set_xlabel("Content Preservation / DINO-C   → Higher is better",
                  fontsize=11.8, weight="bold", color=INK, labelpad=8)
    ax.set_ylabel("5-Way Target Style Classification / CSD Top-1 (%)   ↑",
                  fontsize=11.8, weight="bold", color=INK, labelpad=8)

    ax.axhline(20.0, color="#7F8C8D", lw=1.1, ls=":", alpha=0.75, zorder=2)
    ax.text(1.015, 21.0, "5-class uniform chance (20%)", fontsize=8.6, color="#7F8C8D", ha="right", style="italic")

    ax.plot([p[1] for p in ext_front], [p[3] for p in ext_front],
            color=BLUE, lw=2.0, ls=(0, (5, 3)), zorder=3)
    ax.fill_between([p[1] for p in ext_front], [p[3] for p in ext_front], -2.0,
                    color=BLUE, alpha=0.04, zorder=1)
    ax.plot([p[1] for p in comb_front], [p[3] for p in comb_front],
            color=CRIMSON, lw=2.4, ls="-", alpha=0.82, zorder=4)

    for p in ext:
        hit = p[0] in ext_front_names
        s_val = area_fn(p[5])
        ax.scatter(p[1], p[3], s=s_val, marker="o",
                   fc=BLUE if hit else GREY, ec="white", lw=1.1,
                   alpha=0.92 if hit else 0.72, zorder=6)

    for p in ours:
        is_flag = p[6] == "flagship"
        is_scale = p[6] == "scale"
        hit = p[0] in comb_names
        s_val = area_fn(p[5])
        if is_flag:
            fc, ec, lw, zo = CRIMSON, "#59080C", 2.1, 11
        elif is_scale:
            fc, ec, lw, zo = "#B69CE8", PURPLE, 1.1, 6
        elif hit:
            fc, ec, lw, zo = CORAL, CORAL_DARK, 1.4, 9
        else:
            fc, ec, lw, zo = CORAL_SOFT, "#D98F80", 1.0, 5
        ax.scatter(p[1], p[3], s=s_val, marker="o", fc=fc, ec=ec, lw=lw, alpha=0.92, zorder=zo)

    ax.annotate("", xy=(0.746, 43.8), xytext=(0.812, 23.0),
                arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.55", lw=2.3, color=CRIMSON), zorder=12)
    t_s = ax.text(0.835, 33.0, "+22.1pp Single-Step Leap!\n(Universal Chord + Router,\nwrong-ID = 0.83%)",
                  fontsize=8.3, color=CRIMSON, weight="bold", ha="left", va="center")
    t_s.set_path_effects([pe.withStroke(linewidth=3, foreground=BG)])

    lbls = {
        FLOW_NAME:                    (0.535, 63.5, "center", CRIMSON),
        FAST_NAME:                    (0.745, 48.5, "center", CRIMSON),
        "StyleGallery":               (0.615, 48.0, "left", BLUE),
        "StyTr2":                     (0.675, 38.0, "right", INK),
        "AesPA-Net":                  (0.592, 28.0, "right", MUTED),
        "Seedream 4.5":               (0.718, 32.0, "right", INK),
        "SaMam":                      (0.824, -0.3, "center", MUTED),
        "CUT":                        (0.858,  8.2, "left", MUTED),
        "Hist2Style":                 (0.920, 15.2, "left", INK),
    }
    for p in ext + ours:
        if p[0] in lbls:
            tx, ty, ha, col = lbls[p[0]]
            ann = ax.annotate(
                p[0], xy=(p[1], p[3]), xytext=(tx, ty), ha=ha, va="center",
                fontsize=9.0, color=col, weight="bold",
                arrowprops=dict(arrowstyle="-", lw=0.9, color=col, alpha=0.6, shrinkA=2, shrinkB=4),
                zorder=15,
            )
            ann.set_path_effects([pe.withStroke(linewidth=3.0, foreground=BG)])

    _draw_throughput_key(ax, K)

    fig.text(0.085, 0.952, f"{METHOD} × SPECIFICITY (CSD TOP-1) × CONTENT × SPEED",
             fontsize=17.0, weight="bold", color=INK, va="top")
    fig.text(0.085, 0.900,
             "5-Way Target Classification · Surpasses StyleGallery (60.5% vs 52.5%), StyTr2 & Seedream 4.5",
             fontsize=9.6, color=MUTED, va="top")
    fig.text(0.965, 0.954, "60.5% Top-1", ha="right", va="top", fontsize=19.5, color=CRIMSON, weight="bold")
    fig.text(0.965, 0.900, f"{METHOD}-Flow (+8.0pp over StyleGallery)", ha="right", va="top", fontsize=8.6, color=MUTED)

    cat_handles = [
        Line2D([], [], marker="o", ls="none", ms=9.0, mfc=CRIMSON, mec="#59080C", label=f"{METHOD} (RTX 3060: 1-Step 168ms / Flow 680ms E2E)"),
        Line2D([], [], marker="o", ls="none", ms=8.0, mfc=BLUE, mec="white", label="Published / Commercial Baselines (incl. StyleGallery)"),
        Line2D([], [], color=CRIMSON, ls="-", lw=2.3, label=f"{METHOD} Pareto Frontier"),
        Line2D([], [], color=BLUE, ls=(0, (5, 3)), lw=2.0, label="External-Only Frontier"),
    ]
    fig.legend(handles=cat_handles, bbox_to_anchor=(0.085, 0.045), loc="lower left",
               ncol=2, frameon=False, fontsize=8.8, columnspacing=1.4)

    _save(fig, "fig4_specificity_frontier")


if __name__ == "__main__":
    import sys
    repo = Path(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from exp.v32_nodino.plot_v32_pareto import plot_v32_pareto
    plot_v32_pareto()
    plot_fig2_three_paradoxes()
    plot_fig4_specificity_frontier()
