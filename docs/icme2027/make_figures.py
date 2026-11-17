#!/usr/bin/env python3
"""Generate standalone ICME 2027 figures for PRISM (centralized modular naming)."""
from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
from matplotlib import patheffects as pe
import numpy as np

ICME_DIR = Path(__file__).resolve().parent
FIG_DIR = ICME_DIR / "figures"

METHOD = "PRISM"
PRO_NAME = "PRISM-Pro (Ours)"
FAST_NAME = "PRISM-1 (Ours)"
DEC_NAME = "PRISM-Dec (Pruned)"

INK = "#182A42"
MUTED = "#62758B"
BLUE = "#2E6B9E"
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

# (name, dino_c, affinity, top1, lpips, ms_per_img)
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

OURS_PTS = [
    (PRO_NAME,                    0.7428, 0.5674, 38.00, 0.5018, 115.0, "flagship"),
    (FAST_NAME,                   0.7793, 0.5606, 33.83, 0.4806,  28.5, "flagship"),
    ("Unbiased Critic (w/o Src)", 0.7718, 0.5615, 35.17, 0.4872, 105.0, "ablation"),
    ("2-Step SPD Critic",         0.7715, 0.5605, 32.83, 0.4904,  74.0, "ablation"),
    ("1-Step Canonical Base",     0.8152, 0.5483, 22.20, 0.4396,  21.0, "ablation"),
    ("De-Styled Flow (τ0=0)",     0.7222, 0.5668, 18.67, 0.4549,  44.8, "ablation"),
    ("SDEdit+DINO Stu (τ0=0.7)",  0.7386, 0.5766, 13.33, 0.2845,  88.0, "ablation"),
    ("Self-Paired 3.2M",          0.7506, 0.5692, 10.67, 0.2700,  72.0, "scale"),
    ("Self-Paired 4.1M",          0.7767, 0.5633,  9.50, 0.2421,  86.0, "scale"),
    ("Self-Paired 8.9M",          0.7920, 0.5584,  8.00, 0.2314, 122.0, "scale"),
    ("Self-Paired 15.5M",         0.7962, 0.5571,  8.33, 0.2285, 165.0, "scale"),
    ("Self-Paired 34.2M",         0.8083, 0.5541,  6.67, 0.2210, 260.0, "scale"),
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
            "Bubble area ∝ GPU throughput (RTX 3060)",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=8.9, weight="bold", color=INK, zorder=17)
    xs_ax = [0.612, 0.698, 0.798, 0.918]
    sp_vals = [1, 5, 15, 35]
    for x_pos, v in zip(xs_ax, sp_vals):
        ax.scatter([x_pos], [box_y + 0.084], s=min(800.0, K * v),
                   transform=ax.transAxes, marker="o",
                   fc="#DCE4EC", ec="#7F95AB", lw=1.1, zorder=17)
        ax.text(x_pos, box_y + 0.008, f"{v} img/s",
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=8.1, color=MUTED, zorder=18)


def plot_fig1_pareto_zoom():
    x0 = 0.58
    ext = [p for p in EXTERNAL_PTS if p[1] >= x0]
    ours = [p for p in OURS_PTS if p[1] >= x0]
    K = 44.0
    area_fn = lambda ms: max(24.0, min(1650.0, K * (1000.0 / ms)))

    ext_front = pareto_2d(ext, 1, 2)
    ext_front_names = {p[0] for p in ext_front}
    comb_front = pareto_2d(ext + [p for p in ours if p[6] != "scale" and "SDEdit" not in p[0]], 1, 2)
    comb_names = {p[0] for p in comb_front}

    fig = plt.figure(figsize=(11.5, 7.6), facecolor=BG)
    ax = fig.add_axes([0.085, 0.215, 0.880, 0.635], facecolor=BG)
    ax.set_xlim(x0, 1.03)
    ax.set_ylim(0.480, 0.625)

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
    ax.text(1.025, IDT_AFFINITY + 0.0022, "Identity style floor (0.490)",
            fontsize=8.8, color="#C0392B", va="bottom", ha="right", style="italic")

    ax.plot([p[1] for p in ext_front], [p[2] for p in ext_front],
            color=BLUE, lw=2.0, ls=(0, (5, 3)), zorder=3)
    ax.fill_between([p[1] for p in ext_front], [p[2] for p in ext_front], 0.480,
                    color=BLUE, alpha=0.045, zorder=1)
    ax.plot([p[1] for p in comb_front], [p[2] for p in comb_front],
            color=CRIMSON, lw=2.3, ls="-", alpha=0.78, zorder=4)

    for p in ext:
        hit = p[0] in ext_front_names
        s_val = area_fn(p[5])
        ax.scatter(p[1], p[2], s=s_val, marker="o",
                   fc=BLUE if hit else GREY, ec="white", lw=1.1,
                   alpha=0.92 if hit else 0.72, zorder=6)

    for p in ours:
        if "SDEdit" in p[0]:
            continue
        is_flag = p[6] == "flagship"
        is_scale = p[6] == "scale"
        s_val = area_fn(p[5])
        if is_flag:
            fc, ec, lw, zo = CRIMSON, "#59080C", 2.1, 11
        elif is_scale:
            fc, ec, lw, zo = "#B69CE8", PURPLE, 1.0, 5
        else:
            fc, ec, lw, zo = CORAL, CORAL_DARK, 1.3, 8
        ax.scatter(p[1], p[2], s=s_val, marker="o", fc=fc, ec=ec, lw=lw, alpha=0.92, zorder=zo)

    sc_pts = [p for p in ours if p[6] == "scale"]
    ax.plot([p[1] for p in sc_pts], [p[2] for p in sc_pts], color=PURPLE, lw=1.5, ls=":", zorder=6)

    lbls = {
        "AesPA-Net":              (0.618, 0.595, "left", INK),
        "StyTr2":                 (0.668, 0.519, "right", INK),
        "Seedream 4.5":           (0.718, 0.512, "right", INK),
        "SaMam":                  (0.818, 0.503, "right", INK),
        "CUT":                    (0.858, 0.523, "left", INK),
        "Hist2Style":             (0.916, 0.522, "left", INK),
        "SD-Turbo":               (0.985, 0.483, "right", MUTED),
        PRO_NAME:                 (0.735, 0.594, "center", CRIMSON),
        FAST_NAME:                (0.815, 0.576, "left", CRIMSON),
        "1-Step Canonical Base":  (0.852, 0.556, "left", CORAL_DARK),
        "De-Styled Flow (τ0=0)":  (0.688, 0.580, "right", CORAL_DARK),
        "Self-Paired 34.2M":      (0.835, 0.538, "left", PURPLE),
    }
    for p in ext + ours:
        if p[0] in lbls:
            tx, ty, ha, col = lbls[p[0]]
            ann = ax.annotate(
                p[0], xy=(p[1], p[2]), xytext=(tx, ty), ha=ha, va="center",
                fontsize=9.4, color=col, weight="bold",
                arrowprops=dict(arrowstyle="-", lw=0.95, color=col, alpha=0.65, shrinkA=2, shrinkB=4),
                zorder=15,
            )
            ann.set_path_effects([pe.withStroke(linewidth=3.0, foreground=BG)])

    _draw_throughput_key(ax, K)

    fig.text(0.085, 0.952, f"{METHOD} × STYLE × CONTENT × SPEED",
             fontsize=17.5, weight="bold", color=INK, va="top")
    fig.text(0.085, 0.900,
             "600 Cross-Style Pairs · Unified Pruned-Decoder Referee · Bubble Area ∝ Throughput",
             fontsize=9.8, color=MUTED, va="top")
    fig.text(0.965, 0.954, "35.1 img/s", ha="right", va="top", fontsize=19.5, color=CRIMSON, weight="bold")
    fig.text(0.965, 0.900, f"{METHOD}-1 (0.66M, 28.5 ms)", ha="right", va="top", fontsize=9.0, color=MUTED)

    cat_handles = [
        Line2D([], [], marker="o", ls="none", ms=8.8, mfc=CRIMSON, mec="#59080C", label=f"{METHOD}-Pro / {METHOD}-1 (Ours)"),
        Line2D([], [], marker="o", ls="none", ms=8.2, mfc=CORAL, mec=CORAL_DARK, label="Controlled Transport Ablations"),
        Line2D([], [], marker="o", ls="none", ms=8.0, mfc="#B69CE8", mec=PURPLE, label="Self-Paired Scaling (3.2M→34.2M)"),
        Line2D([], [], marker="o", ls="none", ms=8.0, mfc=BLUE, mec="white", label="Published / Commercial Baselines"),
        Line2D([], [], color=CRIMSON, ls="-", lw=2.2, label=f"{METHOD} Pareto Frontier"),
        Line2D([], [], color=BLUE, ls=(0, (5, 3)), lw=2.0, label="External-Only Frontier"),
    ]
    fig.legend(handles=cat_handles, bbox_to_anchor=(0.085, 0.045), loc="lower left",
               ncol=2, frameon=False, fontsize=8.9, columnspacing=1.5)

    _save(fig, "fig1_prism_pareto")


def plot_fig2_three_paradoxes():
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.45), facecolor=BG)
    for ax in axes:
        ax.set_facecolor(BG)
        ax.grid(color=GRID, lw=0.85, axis="y")
        for edge in ("bottom", "left"):
            ax.spines[edge].set_color("#B9C5D0")

    # (a) Self-Paired Scaling Paradox
    ax = axes[0]
    params_m = [0.13, 0.37, 1.17, 3.16, 4.13, 8.92, 15.52, 34.19]
    gen_t0 = [20.0, 20.5, 28.0, 36.33, 42.67, 45.00, 45.00, 49.33]
    edit_t07 = [8.83, 9.00, 8.83, 10.67, 9.50, 8.00, 8.33, 6.67]
    ax.grid(color=GRID, lw=0.85, which="both")
    ax.plot(params_m, gen_t0, marker="o", lw=2.2, color=BLUE, label=r"Unconditional Gen ($\tau_0=0$) ↑")
    ax.plot(params_m, edit_t07, marker="s", lw=2.4, color=CRIMSON, label=r"Cross-Style SDEdit ($\tau_0=0.7$) ↑")
    ax.set_xscale("log")
    ax.set_ylim(0, 58)
    ax.set_xlabel("Flow U-Net Parameters (Millions, log)", fontsize=9.6, weight="bold", color=INK)
    ax.set_ylabel("Cross600 CSD Top-1 (%)", fontsize=9.6, weight="bold", color=INK)
    ax.annotate("49.3%", xy=(34.19, 49.33), xytext=(12.0, 51.5), fontsize=9, weight="bold", color=BLUE)
    ax.annotate("6.67% (59% Source-Lock!)", xy=(34.19, 6.67), xytext=(2.2, 16.0),
                fontsize=8.7, weight="bold", color=CRIMSON,
                arrowprops=dict(arrowstyle="->", color=CRIMSON, lw=1.3))
    ax.set_title("(a) 260× Self-Paired Scaling Paradox", fontsize=10.6, weight="bold", color=INK, pad=10)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")

    # (b) Pruned Decoder Regularization Paradox
    ax = axes[1]
    models_b = ["Full SD-VAE Dec\n(49.5M, 1.0×)", f"{DEC_NAME}\n(10.7M, 2.86×)"]
    top1_b = [34.50, 37.67]
    lpips_b = [51.49, 50.34]
    x = np.arange(2)
    w = 0.34
    b1 = ax.bar(x - w/2, top1_b, w, color=[GREY, CRIMSON], edgecolor=INK, lw=0.9, label="CSD Top-1 (%) ↑")
    b2 = ax.bar(x + w/2, lpips_b, w, color=["#D0D7DE", TEAL], edgecolor=INK, lw=0.9, label="LPIPS ×100 ↓")
    for bar, val in zip(b1, top1_b):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.9, f"{val:.2f}%", ha="center", va="bottom", fontsize=9, weight="bold", color=INK)
    for bar, val in zip(b2, lpips_b):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.9, f"{val/100:.3f}", ha="center", va="bottom", fontsize=9, weight="bold", color=TEAL)
    ax.set_xticks(x)
    ax.set_xticklabels(models_b, fontsize=9.3, weight="bold", color=INK)
    ax.set_ylim(0, 68)
    ax.set_title("(b) Spectral Bottleneck Decoder (+3.17pp, p<0.01)", fontsize=10.6, weight="bold", color=INK, pad=10)
    ax.legend(frameon=False, fontsize=8.6, loc="upper left")

    # (c) Counterfactual Causal Specificity Ladder
    ax = axes[2]
    arms_c = ["Post-Hoc\nSubband", "SDEdit Stu\n(τ0=0.7)", "De-Styled\nFlow (τ0=0)", f"{METHOD}-1\n(Ours)"]
    corr_v = [12.33, 11.67, 15.00, 33.83]
    wrong_v = [13.00, 10.67, 5.17, 1.33]
    xc = np.arange(4)
    w_c = 0.34
    b_c = ax.bar(xc - w_c/2, corr_v, w_c, color=CRIMSON, edgecolor=INK, lw=0.9, label="True Target Label")
    b_w = ax.bar(xc + w_c/2, wrong_v, w_c, color=GREY, edgecolor=INK, lw=0.9, label="Counterfactual Wrong ID")
    for bar, val in zip(b_c, corr_v):
        ax.text(bar.get_x() + bar.get_width()*0.40, val + 0.85, f"{val:.1f}%", ha="center", va="bottom", fontsize=7.9, weight="bold", color=CRIMSON)
    for bar, val in zip(b_w, wrong_v):
        ax.text(bar.get_x() + bar.get_width()*0.62, val + 0.85, f"{val:.1f}%", ha="center", va="bottom", fontsize=7.9, weight="bold", color=MUTED)
    ax.annotate("+32.5pp Causal\nSpecificity!", xy=(3 - w_c/2 - 0.04, 31.0), xytext=(1.35, 27.5),
                fontsize=8.8, weight="bold", color=CRIMSON, ha="center",
                arrowprops=dict(arrowstyle="->", color=CRIMSON, lw=1.4))
    ax.set_xticks(xc)
    ax.set_xticklabels(arms_c, fontsize=8.6, weight="bold", color=INK)
    ax.set_ylim(0, 43)
    ax.set_title("(c) Counterfactual Specificity Audit", fontsize=10.6, weight="bold", color=INK, pad=10)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")

    fig.tight_layout(pad=1.2)
    _save(fig, "fig2_three_paradoxes")


def plot_fig4_specificity_frontier():
    x0 = 0.58
    ext = [p for p in EXTERNAL_PTS if p[1] >= x0]
    ours = [p for p in OURS_PTS if p[1] >= x0]
    K = 44.0
    area_fn = lambda ms: max(24.0, min(1650.0, K * (1000.0 / ms)))

    ext_front = pareto_2d(ext, 1, 3)
    ext_front_names = {p[0] for p in ext_front}
    comb_front = pareto_2d(ext + ours, 1, 3)
    comb_names = {p[0] for p in comb_front}

    fig = plt.figure(figsize=(11.5, 7.6), facecolor=BG)
    ax = fig.add_axes([0.085, 0.215, 0.880, 0.635], facecolor=BG)
    ax.set_xlim(x0, 1.02)
    ax.set_ylim(-2.0, 48.0)

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
    ax.text(1.015, 20.7, "5-class uniform chance (20%)", fontsize=8.6, color="#7F8C8D", ha="right", style="italic")

    ax.plot([p[1] for p in ext_front], [p[3] for p in ext_front],
            color=BLUE, lw=2.0, ls=(0, (5, 3)), zorder=3)
    ax.fill_between([p[1] for p in ext_front], [p[3] for p in ext_front], -2.0,
                    color=BLUE, alpha=0.04, zorder=1)
    ax.plot([p[1] for p in comb_front], [p[3] for p in comb_front],
            color=CRIMSON, lw=2.3, ls="-", alpha=0.75, zorder=4)

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

    ax.annotate("", xy=(0.7815, 33.4), xytext=(0.8135, 22.8),
                arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.55", lw=2.5, color=CRIMSON), zorder=12)
    t_s = ax.text(0.832, 27.6, "+11.63pp Single-Step Leap!\n(Whitened DINO + u_t−u_s,\nwrong-ID = 1.33%)",
                  fontsize=8.6, color=CRIMSON, weight="bold", ha="left", va="center")
    t_s.set_path_effects([pe.withStroke(linewidth=3, foreground=BG)])

    ax.annotate("", xy=(0.8083, 6.67), xytext=(0.7506, 10.67),
                arrowprops=dict(arrowstyle="->,head_width=0.35,head_length=0.5", lw=2.0, color=PURPLE, ls="--"), zorder=12)
    t_p = ax.text(0.665, 4.8, "Self-Paired Scaling (3.2M→34.2M)\nworsens source-lock (10.7%→6.7%)",
                  fontsize=8.4, color=PURPLE, weight="bold", ha="left", va="top")
    t_p.set_path_effects([pe.withStroke(linewidth=3, foreground=BG)])

    lbls = {
        "StyTr2":                     (0.675, 41.8, "right", INK),
        "AesPA-Net":                  (0.608, 34.2, "left", INK),
        "Seedream 4.5":               (0.702, 34.8, "right", INK),
        "SaMam":                      (0.824, -0.3, "center", MUTED),
        "CUT":                        (0.858,  7.8, "left", MUTED),
        "Hist2Style":                 (0.920, 14.5, "left", INK),
        "SD-Turbo":                   (0.985,  3.2, "right", MUTED),
        PRO_NAME:                     (0.742, 42.8, "center", CRIMSON),
        "Unbiased Critic (w/o Src)":  (0.718, 29.5, "right", CORAL_DARK),
        FAST_NAME:                    (0.822, 34.4, "left", CRIMSON),
        "1-Step Canonical Base":      (0.848, 21.5, "left", CORAL_DARK),
        "De-Styled Flow (τ0=0)":      (0.692, 20.8, "right", CORAL_DARK),
        "SDEdit+DINO Stu (τ0=0.7)":   (0.702, 14.2, "right", CORAL_DARK),
    }
    for p in ext + ours:
        if p[0] in lbls:
            tx, ty, ha, col = lbls[p[0]]
            ann = ax.annotate(
                p[0], xy=(p[1], p[3]), xytext=(tx, ty), ha=ha, va="center",
                fontsize=9.1, color=col, weight="bold",
                arrowprops=dict(arrowstyle="-", lw=0.9, color=col, alpha=0.6, shrinkA=2, shrinkB=4),
                zorder=15,
            )
            ann.set_path_effects([pe.withStroke(linewidth=3.0, foreground=BG)])

    _draw_throughput_key(ax, K)

    fig.text(0.085, 0.952, f"{METHOD} × SPECIFICITY (CSD TOP-1) × CONTENT × SPEED",
             fontsize=17.0, weight="bold", color=INK, va="top")
    fig.text(0.085, 0.900,
             "5-Way Target Classification vs. Content Preservation · Breaking Past Seedream 4.5 & StyTr2",
             fontsize=9.8, color=MUTED, va="top")
    fig.text(0.965, 0.954, "38.0% Top-1", ha="right", va="top", fontsize=19.5, color=CRIMSON, weight="bold")
    fig.text(0.965, 0.900, f"{METHOD}-Pro beats Seedream 4.5 & 17× faster", ha="right", va="top", fontsize=9.0, color=MUTED)

    cat_handles = [
        Line2D([], [], marker="o", ls="none", ms=8.8, mfc=CRIMSON, mec="#59080C", label=f"{METHOD}-Pro (38.0%) / {METHOD}-1 (33.8%)"),
        Line2D([], [], marker="o", ls="none", ms=8.2, mfc=CORAL, mec=CORAL_DARK, label="Controlled Transport Ablations"),
        Line2D([], [], marker="o", ls="none", ms=8.0, mfc="#B69CE8", mec=PURPLE, label="Self-Paired Scaling (3.2M→34.2M)"),
        Line2D([], [], marker="o", ls="none", ms=8.0, mfc=BLUE, mec="white", label="Published / Commercial Baselines"),
        Line2D([], [], color=CRIMSON, ls="-", lw=2.2, label=f"{METHOD} Combined Frontier"),
        Line2D([], [], color=BLUE, ls=(0, (5, 3)), lw=2.0, label="External-Only Frontier"),
    ]
    fig.legend(handles=cat_handles, bbox_to_anchor=(0.085, 0.045), loc="lower left",
               ncol=2, frameon=False, fontsize=8.9, columnspacing=1.5)

    _save(fig, "fig4_specificity_frontier")


if __name__ == "__main__":
    plot_fig1_pareto_zoom()
    plot_fig2_three_paradoxes()
    plot_fig4_specificity_frontier()
