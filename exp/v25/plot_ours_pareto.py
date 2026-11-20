"""绘制 V25 最新全局 Pareto 图（双面板: DINO-C vs DINO-S 与 DINO-C vs CSD Top-1）。

- 其余约 400 个历史与消融实验臂以高透明度小散点云展示背景探索空间；
- 我们最新的合法非支配前沿点（含 V25 PRISM-HighContent-R12、D6 tau-shift、V22/V23 冠军点）
  统一用橙红色高亮并连成 `Ours-Pareto` 曲线；
- **气泡半径按 RTX 3060 最大安全 Batch 实测端到端吞吐量（img/s）线性映射**（r = 3.2 + 2.0·img/s）；
- 另以橙红菱形标出 V25 C2nt 零教师 One-Shot 新点与 PRISM-1 极速档，以蓝星/灰方标出外部基线包络。

输出:
  docs/performance/figures/ours_pareto.png
  docs/performance/figures/ours_pareto.svg
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "docs/performance/metrics.csv"
BENCH_3060_PATH = ROOT / "runs/v25/r15_3060_maxbatch_bench.json"
OUT_DIR = ROOT / "docs/performance/figures"

ORANGE_RED = "#ff4500"
ORANGE_EDGE = "#9a3412"
ORANGE_FILL = "#fed7aa"

# 左面板 (DINO-C vs DINO-S) 严格非支配 Ours-Pareto 序列（arm, label, offset, 3060 bench engine key）
OURS_PARETO_DS = [
    ("T4Pg45_wc10",               "V22 Peak1500",          (12, 10, "left"),    "d5_full_10s"),
    ("T4Pg65_wc10",               "V22 Gate65",            (-10, -20, "right"), "d5_full_10s"),
    ("F3_dual_fixed_orig",        "V23 Dual-F3",           (14, -8, "left"),    "d5_csj_10s"),
    ("HiC_JPk_csj65_t4",          "V23 HiC-JPk",           (10, 14, "left"),    "d5_csj_10s"),
    ("P14b_d6h_fixed_g69",        "V25 D6H-g69",           (16, 6, "left"),     "d6_csj_b4"),
    ("B_d6_t066",                 "V25 D6-τ0.66",          (16, 7, "left"),     "d6_full_b4"),
    ("HC_R12_g100",               "V25 R12 Single",        (16, 8, "left"),     "d6_csj_b4"),
    ("HC_R12_g85",                "V25 R12 (g85)",         (16, 7, "left"),     "prism_hc_r12"),
    ("HC_R12_g70",                "V25 R12 (g70)",         (16, 6, "left"),     "prism_hc_r12"),
    ("MechA_B4anc_A0_global_g55", "V25 R12 (g55)",         (16, 2, "left"),     "prism_hc_r12"),
]

# 右面板 (DINO-C vs CSD Top-1) 严格非支配 Ours-Pareto 序列
OURS_PARETO_T1 = [
    ("T4Pg45_wc10",               "V22 Peak1500",          (12, 6, "left"),     "d5_full_10s"),
    ("T4Pk_wc15",                 "V22 Peak1500 (wc15)",   (14, 5, "left"),     "d5_full_10s"),
    ("V20.A1_t4_patchswd",        "V20 PRISM-Pro+",        (14, 6, "left"),     "d6_full_b4"),
    ("B_d6_t070",                 "V25 D6-τ0.70",          (-14, -16, "right"), "d6_full_b4"),
    ("M1_lin_g55",                "V24 Hybrid (g55)",      (14, 9, "left"),     "prism_hc_r12"),
    ("M1_lin_g40",                "V24 Hybrid (g40)",      (14, 6, "left"),     "prism_hc_r12"),
    ("HC_B4_full_D6_g55",         "V25 R13 (B4+full)",     (16, 12, "left"),    "prism_hc_r12"),
    ("HC_B4_full_D6H_g55",        "V25 R13 (B4+D6H)",      (16, -14, "left"),   "prism_hc_r12"),
]

# 额外标出的 V25 One-Shot 零教师点 & PRISM-1 极速代表点
EXTRA_HIGHLIGHT = [
    ("C2nt_sw6_lc0.8_t45", "V25 C2nt (t45, 0-teacher)", "D", (-10, 12, "right"), (-14, -16, "right"), "o3_ref_fresh"),
    ("C2nt_sw6_lc0.8_t55", "V25 C2nt (t55, 0-teacher)", "D", (-14, -22, "right"), (-14, -16, "right"), "o3_ref_fresh"),
    ("M0_prism1_l100",     "PRISM-1 (5.97 img/s)",       "o", (-14, -22, "right"), (-28, -4, "right"),  "prism1"),
]

# 外部基线 (名称: (显示名, 左偏移, 右偏移, 固定 ms/img 若 metrics.csv 缺失))
KEY_BASELINES = {
    "StyleShot":         ("StyleShot",             (8, -12, "left"),    (10, -4, "left"),    24480.0),
    "StyleID":           ("StyleID",               (8, 10, "left"),     (10, -4, "left"),    5040.0),
    "AesPA":             ("AesPA",                 (-8, -11, "right"),  (-8, -10, "right"),  1840.0),
    "StyleGallery":      ("StyleGallery (CVPR26)", (12, 16, "left"),    (-22, 14, "right"),  30400.0),
    "StyTr2":            ("StyTr2",                (-8, -12, "right"),  (-10, -10, "right"), 3040.0),
    "Seedream 4.5":      ("Seedream 4.5",          (-8, -13, "right"),  (-12, -14, "right"), 2000.0),
    "WEAVE (Published)": ("WEAVE (Orig)",          (8, 7, "left"),      (10, 6, "left"),     2500.0),
    "SaMAM":             ("SaMAM",                 (-10, -12, "right"), (10, -4, "left"),    1408.0),
    "CUT":               ("CUT",                   (8, 10, "left"),     (-10, -6, "right"),  400.0),
    "Hist2Style":        ("Hist2Style",            (-10, -10, "right"), (-6, 10, "right"),   1500.0),
}


def img_s_to_area(img_s: float) -> float:
    """半径 r ∝ img/s (r_pt = 3.2 + 2.0 * min(img_s, 7.0)); matplotlib scatter s = pi * r^2."""
    r_pt = 3.2 + 2.0 * min(max(0.0, float(img_s)), 7.0)
    return math.pi * (r_pt ** 2)


def load_data():
    rows = []
    with open(CSV_PATH, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r.get("protocol") != "q2a_oracle_cross600":
                continue
            if not (r.get("dino_c") and r.get("dino_s") and r.get("target_top1")):
                continue
            rows.append({
                "arm": r["arm"],
                "group": r["group"],
                "dc": float(r["dino_c"]),
                "ds": float(r["dino_s"]),
                "top1": float(r["target_top1"]) * 100.0,
                "ms": float(r["ms_per_img"]) if r.get("ms_per_img") else None,
            })
    return rows, {r["arm"]: r for r in rows}


def load_3060_speeds() -> dict[str, float]:
    d = json.load(open(BENCH_3060_PATH, encoding="utf-8"))
    return {eng: float(info["best_e2e"]["e2e_img_per_s"]) for eng, info in d["results"].items()}


def pareto_envelope(pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    s = sorted(pts, key=lambda p: (-p[0], -p[1]))
    out, best_y = [], -1e9
    for x, y in s:
        if y > best_y + 1e-6:
            out.append((x, y))
            best_y = y
    return sorted(out, key=lambda p: p[0])


def draw_radius_legend(ax, loc="upper right"):
    """绘制与图上完全同尺度的吞吐量半径图例."""
    if loc == "lower left":
        bx, by, bw, bh = 0.015, 0.23, 0.355, 0.165
    elif loc == "bottom left":
        bx, by, bw, bh = 0.015, 0.025, 0.365, 0.165
    else:
        bx, by, bw, bh = 0.625, 0.48, 0.360, 0.165
    rect = mpatches.FancyBboxPatch(
        (bx, by), bw, bh, transform=ax.transAxes,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        fc="white", ec="#cbd5e1", lw=0.95, alpha=0.94, zorder=15,
    )
    ax.add_patch(rect)
    ax.text(bx + bw / 2, by + bh - 0.018, "Radius ∝ RTX 3060 Max-Batch E2E (img/s)",
            transform=ax.transAxes, ha="center", va="top", fontsize=7.9, fontweight="bold", color="#0f172a", zorder=16)
    vals = [0.03, 0.5, 2.5, 3.7, 6.0]
    xs = [bx + 0.055 + i * (bw - 0.10) / (len(vals) - 1) for i in range(len(vals))]
    for xp, v in zip(xs, vals):
        ax.scatter([xp], [by + 0.078], s=img_s_to_area(v), transform=ax.transAxes,
                   fc="#fed7aa", ec=ORANGE_RED, lw=1.1, zorder=16)
        ax.text(xp, by + 0.012, f"{v:g}", transform=ax.transAxes, ha="center", va="bottom",
                fontsize=7.6, color="#334155", zorder=16)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, by_arm = load_data()
    spd = load_3060_speeds()

    hi_set = {s[0] for s in OURS_PARETO_DS} | {s[0] for s in OURS_PARETO_T1} | {s[0] for s in EXTRA_HIGHLIGHT}
    base_set = set(KEY_BASELINES.keys())
    cloud = [
        r for r in rows
        if r["arm"] not in hi_set
        and r["arm"] not in base_set
        and "Baseline" not in r["group"]
        and 0.30 <= r["dc"] <= 0.94
        and 0.34 <= r["ds"] <= 0.62
    ]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#475569",
        "axes.linewidth": 0.95,
        "grid.color": "#cbd5e1",
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.65,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.8, 7.2), dpi=240)
    fig.patch.set_facecolor("white")

    target_box = mpatches.FancyBboxPatch(
        (0.73, 0.49), 0.08, 0.035,
        boxstyle="round,pad=0.002", facecolor="#dcfce7", edgecolor="#16a34a",
        linestyle="--", linewidth=1.2, alpha=0.45, zorder=1,
    )
    ax1.add_patch(target_box)
    ax1.text(0.77, 0.528, "High-Content Target\n(DINO-C ≥ 0.73, DINO-S ≥ 0.49)",
             color="#15803d", fontsize=8.2, fontweight="bold", ha="center", va="bottom", zorder=4)

    # 背景散点云
    ax1.scatter([r["dc"] for r in cloud], [r["ds"] for r in cloud],
                s=11, c="#64748b", alpha=0.14, edgecolors="none",
                label=f"Other explored arms (n={len(cloud)})", zorder=2)
    ax2.scatter([r["dc"] for r in cloud], [r["top1"] for r in cloud],
                s=11, c="#64748b", alpha=0.14, edgecolors="none",
                label=f"Other explored arms (n={len(cloud)})", zorder=2)

    # 外部基线包络 + 吞吐半径散点
    base_rows = [by_arm[k] for k in KEY_BASELINES if k in by_arm]
    env_b_ds = pareto_envelope([(r["dc"], r["ds"]) for r in base_rows])
    env_b_t1 = pareto_envelope([(r["dc"], r["top1"]) for r in base_rows])
    ax1.plot([p[0] for p in env_b_ds], [p[1] for p in env_b_ds],
             color="#475569", linestyle="--", linewidth=1.3, alpha=0.65,
             label="External Baselines Pareto", zorder=3)
    ax2.plot([p[0] for p in env_b_t1], [p[1] for p in env_b_t1],
             color="#475569", linestyle="--", linewidth=1.3, alpha=0.65,
             label="External Baselines Pareto", zorder=3)

    for k, (lbl, off1, off2, fallback_ms) in KEY_BASELINES.items():
        if k not in by_arm:
            continue
        r = by_arm[k]
        ms = r["ms"] if r["ms"] else fallback_ms
        img_s = 1000.0 / ms
        sz = img_s_to_area(img_s)
        is_sg = (k == "StyleGallery")
        mk = "*" if is_sg else "s"
        sz_use = max(sz * 3.0, 120.0) if is_sg else sz
        col = "#2563eb" if is_sg else "#334155"
        ec = "#1e3a8a" if is_sg else "white"
        kw_arr = dict(arrowstyle="-", color="#2563eb", lw=0.75, alpha=0.7) if is_sg else None
        ax1.scatter([r["dc"]], [r["ds"]], marker=mk, s=sz_use, c=col, edgecolors=ec, linewidths=0.9, zorder=7)
        ax1.annotate(lbl, (r["dc"], r["ds"]), xytext=(off1[0], off1[1]), textcoords="offset points",
                     fontsize=8.5 if is_sg else 7.9, fontweight="bold" if is_sg else "normal",
                     color="#1d4ed8" if is_sg else "#334155", ha=off1[2], arrowprops=kw_arr, zorder=9)
        ax2.scatter([r["dc"]], [r["top1"]], marker=mk, s=sz_use, c=col, edgecolors=ec, linewidths=0.9, zorder=7)
        ax2.annotate(lbl, (r["dc"], r["top1"]), xytext=(off2[0], off2[1]), textcoords="offset points",
                     fontsize=8.5 if is_sg else 7.9, fontweight="bold" if is_sg else "normal",
                     color="#1d4ed8" if is_sg else "#334155", ha=off2[2], arrowprops=kw_arr, zorder=9)

    ax1.scatter([], [], marker="*", s=130, c="#2563eb", edgecolors="#1e3a8a", label="StyleGallery (0.03 img/s)")
    ax1.scatter([], [], marker="s", s=45, c="#334155", edgecolors="white", label="Other external baselines")
    ax2.scatter([], [], marker="*", s=130, c="#2563eb", edgecolors="#1e3a8a", label="StyleGallery (0.03 img/s)")
    ax2.scatter([], [], marker="s", s=45, c="#334155", edgecolors="white", label="Other external baselines")

    # ---- Ours-Pareto (橙红色 #ff4500, 半径 ∝ 3060 max-batch img/s) ----
    ds_pts = [(by_arm[a], lbl, off, spd.get(eng, 3.0)) for a, lbl, off, eng in OURS_PARETO_DS if a in by_arm]
    ax1.plot([p[0]["dc"] for p in ds_pts], [p[0]["ds"] for p in ds_pts],
             color=ORANGE_RED, linewidth=2.8, alpha=0.92, zorder=8, label="Ours-Pareto (V19–V25)")
    ax1.fill_between([p[0]["dc"] for p in ds_pts], [p[0]["ds"] for p in ds_pts], 0.35,
                     color=ORANGE_FILL, alpha=0.16, zorder=1)
    for r, lbl, off, isp in ds_pts:
        ax1.scatter([r["dc"]], [r["ds"]], s=img_s_to_area(isp), c=ORANGE_RED,
                    edgecolors="white", linewidths=1.4, alpha=0.92, zorder=10)
        ax1.annotate(lbl, (r["dc"], r["ds"]), xytext=(off[0], off[1]), textcoords="offset points",
                     fontsize=8.1, fontweight="bold", color=ORANGE_EDGE, ha=off[2], zorder=11,
                     arrowprops=dict(arrowstyle="-", color=ORANGE_RED, lw=0.8, alpha=0.75))

    t1_pts = [(by_arm[a], lbl, off, spd.get(eng, 3.0)) for a, lbl, off, eng in OURS_PARETO_T1 if a in by_arm]
    ax2.plot([p[0]["dc"] for p in t1_pts], [p[0]["top1"] for p in t1_pts],
             color=ORANGE_RED, linewidth=2.8, alpha=0.92, zorder=8, label="Ours-Pareto (V19–V25)")
    ax2.fill_between([p[0]["dc"] for p in t1_pts], [p[0]["top1"] for p in t1_pts], 0.0,
                     color=ORANGE_FILL, alpha=0.14, zorder=1)
    for r, lbl, off, isp in t1_pts:
        ax2.scatter([r["dc"]], [r["top1"]], s=img_s_to_area(isp), c=ORANGE_RED,
                    edgecolors="white", linewidths=1.4, alpha=0.92, zorder=10)
        ax2.annotate(lbl, (r["dc"], r["top1"]), xytext=(off[0], off[1]), textcoords="offset points",
                     fontsize=8.1, fontweight="bold", color=ORANGE_EDGE, ha=off[2], zorder=11,
                     arrowprops=dict(arrowstyle="-", color=ORANGE_RED, lw=0.8, alpha=0.75))

    # V25 C2nt 零教师 One-Shot & PRISM-1 极速点
    for a, lbl, mk, off1, off2, eng in EXTRA_HIGHLIGHT:
        if a not in by_arm:
            continue
        r = by_arm[a]
        sz = img_s_to_area(spd.get(eng, 3.0))
        fc = "#fb923c" if mk == "D" else "#ff6b35"
        ax1.scatter([r["dc"]], [r["ds"]], marker=mk, s=sz, c=fc, edgecolors=ORANGE_EDGE, linewidths=1.3, alpha=0.92, zorder=10)
        ax1.annotate(lbl, (r["dc"], r["ds"]), xytext=(off1[0], off1[1]), textcoords="offset points",
                     fontsize=7.8, fontweight="bold", color="#c2410c", ha=off1[2], zorder=11,
                     arrowprops=dict(arrowstyle="-", color=fc, lw=0.8, alpha=0.85))
        ax2.scatter([r["dc"]], [r["top1"]], marker=mk, s=sz, c=fc, edgecolors=ORANGE_EDGE, linewidths=1.3, alpha=0.92, zorder=10)
        ax2.annotate(lbl, (r["dc"], r["top1"]), xytext=(off2[0], off2[1]), textcoords="offset points",
                     fontsize=7.8, fontweight="bold", color="#c2410c", ha=off2[2], zorder=11,
                     arrowprops=dict(arrowstyle="-", color=fc, lw=0.8, alpha=0.85))

    ax1.scatter([], [], marker="D", s=80, c="#fb923c", edgecolors=ORANGE_EDGE, label="V25 C2nt (0-teacher One-Shot)")
    ax2.scatter([], [], marker="D", s=80, c="#fb923c", edgecolors=ORANGE_EDGE, label="V25 C2nt (0-teacher One-Shot)")

    draw_radius_legend(ax1, loc="lower left")
    draw_radius_legend(ax2, loc="bottom left")

    ax1.set_title("Content vs. Style Similarity (DINO-C ↑ vs. DINO-S ↑)", fontsize=12.5, fontweight="bold", pad=10, color="#0f172a")
    ax1.set_xlabel("Content Preservation — DINO-C ↑ (higher = more faithful)", fontsize=10.5, color="#1e293b")
    ax1.set_ylabel("Style Similarity — DINO-S ↑ (higher = stronger style)", fontsize=10.5, color="#1e293b")
    ax1.set_xlim(0.32, 0.94)
    ax1.set_ylim(0.352, 0.615)
    ax1.grid(True)
    ax1.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=8.2, framealpha=0.95)

    ax2.set_title("Content vs. Style Classification (DINO-C ↑ vs. CSD Top-1 ↑)", fontsize=12.5, fontweight="bold", pad=10, color="#0f172a")
    ax2.set_xlabel("Content Preservation — DINO-C ↑ (higher = more faithful)", fontsize=10.5, color="#1e293b")
    ax2.set_ylabel("Target Style Recognition — CSD Top-1 (%) ↑", fontsize=10.5, color="#1e293b")
    ax2.set_xlim(0.32, 0.94)
    ax2.set_ylim(0, 96)
    ax2.grid(True)
    ax2.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=8.2, framealpha=0.95)

    fig.suptitle(
        "WEAVE / PRISM Global Pareto Frontier (Cross600, S10) — Ours-Pareto (Orange-Red, Radius ∝ RTX 3060 img/s)",
        fontsize=13.6, fontweight="bold", color="#0f172a", y=0.985,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.945])

    png_p = OUT_DIR / "ours_pareto.png"
    svg_p = OUT_DIR / "ours_pareto.svg"
    fig.savefig(png_p, dpi=240)
    fig.savefig(svg_p)
    plt.close(fig)
    print(f"Saved: {png_p} & {svg_p} (ds_pts={len(ds_pts)}, t1_pts={len(t1_pts)}, cloud={len(cloud)})")


if __name__ == "__main__":
    main()
