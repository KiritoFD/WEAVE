"""绘制 WEAVE 总体 Our-Frontier 在整个帕累托前沿上的表现（双面板: DINO-C vs DINO-S 与 DINO-C vs CSD Top-1）。

- 其余 ~325 个历史/消融实验臂以高透明度小散点云展示背景探索空间；
- 精选 8 个间距匀称、经审计的合法代表点构成高亮 Our-Frontier 折线；
- 对比全部主流外部基线及其包络，精调全部标签与细导线确保零重叠。

输出:
  docs/performance/figures/our_frontier_pareto.png
  docs/performance/figures/our_frontier_pareto.svg
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "docs/performance/metrics.csv"
OUT_DIR = ROOT / "docs/performance/figures"

# 8 个精选合法 Our-Frontier 代表点（间距匀称，覆盖高风格 → 均衡 → 极高内容全谱）
# 格式: (arm_id, label, (dx1, dy1, ha1) for left panel, (dx2, dy2, ha2) for right panel)
OUR_FRONTIER_SPEC = [
    ("T4Pg45_wc10",        "V22 Peak1500 (g0.45)", (8, 12, "left"),      (14, 4, "left")),
    ("T4Pk_wc15",          "V22 Peak1500 (wc1.5)", (-18, -8, "right"),   (14, 4, "left")),
    ("F3_dual_fixed_orig", "V23 Dual-F3 (20 NFE)", (-16, -24, "right"),  (-18, -16, "right")),
    ("HiC_JPk_csj65_t4",   "V23 HiC-JPk (20 NFE)", (-16, -12, "right"),  (14, 12, "left")),
    ("M0_v20_d6",          "V20 D6",               (-16, -8, "right"),   (14, 9, "left")),
    ("M1_p1_dswd_g50",     "V24 M1-DualSWD",       (-16, -12, "right"),  (-4, -22, "center")),
    ("M1_lin_g55",         "V24 M1-Lin (g55)",     (16, 4, "left"),      (16, 8, "left")),
    ("M0_prism1_l075",     "V19 PRISM-1 (λ0.75)",  (14, 2, "left"),      (16, 4, "left")),
]

# 外部基线精调偏移，避免任何字-字/字-线碰撞
# 格式: arm -> (label, (dx1, dy1, ha1), (dx2, dy2, ha2))
KEY_BASELINES = {
    "StyleShot":         ("StyleShot",             (8, -13, "left"),   (10, -4, "left")),
    "StyleID":           ("StyleID",               (8, -4, "left"),    (10, -4, "left")),
    "AesPA":             ("AesPA",                 (-8, -12, "right"), (-8, -10, "right")),
    "StyleGallery":      ("StyleGallery (CVPR26)", (12, 14, "left"),   (-24, 14, "right")),
    "StyTr2":            ("StyTr2",                (-8, -12, "right"), (-10, -10, "right")),
    "Seedream 4.5":      ("Seedream 4.5",          (-8, -13, "right"), (-12, -14, "right")),
    "WEAVE (Published)": ("WEAVE (Orig)",          (8, 7, "left"),     (10, 6, "left")),
    "SaMAM":             ("SaMAM",                 (-10, -13, "right"), (10, -4, "left")),
    "CUT":               ("CUT",                   (8, 10, "left"),    (10, 4, "left")),
    "Hist2Style":        ("Hist2Style",            (-10, -10, "right"), (-8, 10, "right")),
}


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
                "lpips": float(r["lpips"]) if r.get("lpips") else np.nan,
            })
    by_arm = {r["arm"]: r for r in rows}
    return rows, by_arm


def compute_pareto_envelope(pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    s = sorted(pts, key=lambda p: (-p[0], -p[1]))
    out, best_y = [], -1e9
    for x, y in s:
        if y > best_y + 1e-6:
            out.append((x, y))
            best_y = y
    return sorted(out, key=lambda p: p[0])


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, by_arm = load_data()

    frontier_set = {spec[0] for spec in OUR_FRONTIER_SPEC}
    baseline_set = set(KEY_BASELINES.keys())

    # 其余所有非基线点作为高透明度背景散点云
    cloud = [
        r for r in rows
        if r["arm"] not in frontier_set
        and r["arm"] not in baseline_set
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.2, 6.9), dpi=240)
    fig.patch.set_facecolor("white")

    # V24 主目标区间框（放于左图右上方空白处，不与任何文字碰撞）
    target_box = mpatches.FancyBboxPatch(
        (0.73, 0.49), 0.08, 0.035,
        boxstyle="round,pad=0.002", facecolor="#dcfce7", edgecolor="#16a34a",
        linestyle="--", linewidth=1.25, alpha=0.55, zorder=1,
    )
    ax1.add_patch(target_box)
    ax1.text(0.77, 0.529, "V24 Main Target\n(DINO-C ≥ 0.73, DINO-S ≥ 0.49)",
             color="#15803d", fontsize=8.5, fontweight="bold", ha="center", va="bottom", zorder=4)

    # 外部基线包络
    base_rows = [by_arm[k] for k in KEY_BASELINES if k in by_arm]
    base_env_ds = compute_pareto_envelope([(r["dc"], r["ds"]) for r in base_rows])
    base_env_t1 = compute_pareto_envelope([(r["dc"], r["top1"]) for r in base_rows])

    # ---- 背景散点云（高透明度小点）----
    ax1.scatter([r["dc"] for r in cloud], [r["ds"] for r in cloud],
                s=13, c="#64748b", alpha=0.16, edgecolors="none",
                label=f"Other explored arms (n={len(cloud)})", zorder=2)
    ax2.scatter([r["dc"] for r in cloud], [r["top1"] for r in cloud],
                s=13, c="#64748b", alpha=0.16, edgecolors="none",
                label=f"Other explored arms (n={len(cloud)})", zorder=2)

    # ---- 外部基线包络 + 散点 + 标签 ----
    ax1.plot([p[0] for p in base_env_ds], [p[1] for p in base_env_ds],
             color="#475569", linestyle="--", linewidth=1.3, alpha=0.65,
             label="External Baselines Pareto", zorder=3)
    ax2.plot([p[0] for p in base_env_t1], [p[1] for p in base_env_t1],
             color="#475569", linestyle="--", linewidth=1.3, alpha=0.65,
             label="External Baselines Pareto", zorder=3)

    for k, (lbl, off1, off2) in KEY_BASELINES.items():
        if k not in by_arm:
            continue
        r = by_arm[k]
        is_sg = (k == "StyleGallery")
        mk, sz = ("*", 175) if is_sg else ("s", 44)
        col = "#2563eb" if is_sg else "#334155"
        ec = "#1e3a8a" if is_sg else "white"
        kw_arr = dict(arrowstyle="-", color="#2563eb", lw=0.75, alpha=0.7) if is_sg else None

        ax1.scatter([r["dc"]], [r["ds"]], marker=mk, s=sz, c=col, edgecolors=ec, linewidths=0.9, zorder=7)
        ax1.annotate(lbl, (r["dc"], r["ds"]), xytext=(off1[0], off1[1]), textcoords="offset points",
                     fontsize=8.8 if is_sg else 8.1, fontweight="bold" if is_sg else "normal",
                     color="#1d4ed8" if is_sg else "#334155", ha=off1[2], arrowprops=kw_arr, zorder=9)

        ax2.scatter([r["dc"]], [r["top1"]], marker=mk, s=sz, c=col, edgecolors=ec, linewidths=0.9, zorder=7)
        ax2.annotate(lbl, (r["dc"], r["top1"]), xytext=(off2[0], off2[1]), textcoords="offset points",
                     fontsize=8.8 if is_sg else 8.1, fontweight="bold" if is_sg else "normal",
                     color="#1d4ed8" if is_sg else "#334155", ha=off2[2], arrowprops=kw_arr, zorder=9)

    ax1.scatter([], [], marker="*", s=140, c="#2563eb", edgecolors="#1e3a8a", label="StyleGallery (strongest external)")
    ax1.scatter([], [], marker="s", s=40, c="#334155", edgecolors="white", label="Other external baselines")
    ax2.scatter([], [], marker="*", s=140, c="#2563eb", edgecolors="#1e3a8a", label="StyleGallery (strongest external)")
    ax2.scatter([], [], marker="s", s=40, c="#334155", edgecolors="white", label="Other external baselines")

    # ---- Our-Frontier 主折线与精选高亮节点 ----
    f_rows = [by_arm[spec[0]] for spec in OUR_FRONTIER_SPEC if spec[0] in by_arm]
    f_dc = [r["dc"] for r in f_rows]
    f_ds = [r["ds"] for r in f_rows]
    f_t1 = [r["top1"] for r in f_rows]

    ax1.plot(f_dc, f_ds, color="#dc2626", linewidth=2.8, alpha=0.94, zorder=8,
             label="Our-Frontier (PRISM V19–V24)")
    ax1.fill_between(f_dc, f_ds, 0.35, color="#fecaca", alpha=0.14, zorder=1)

    ax2.plot(f_dc, f_t1, color="#dc2626", linewidth=2.5, alpha=0.90, zorder=8,
             label="Our-Frontier (same selected points)")
    our_t1_env = compute_pareto_envelope(list(zip(f_dc, f_t1)))
    ax2.plot([p[0] for p in our_t1_env], [p[1] for p in our_t1_env],
             color="#991b1b", linestyle=":", linewidth=1.7, alpha=0.82, zorder=7)

    for arm_id, short_lbl, off1, off2 in OUR_FRONTIER_SPEC:
        if arm_id not in by_arm:
            continue
        r = by_arm[arm_id]
        pt_col = "#ef4444" if arm_id.startswith("M1_") else ("#ea580c" if arm_id in ("F3_dual_fixed_orig", "HiC_JPk_csj65_t4") else "#b91c1c")

        ax1.scatter([r["dc"]], [r["ds"]], s=84, c=pt_col, edgecolors="white", linewidths=1.4, zorder=10)
        ax1.annotate(
            short_lbl, (r["dc"], r["ds"]), xytext=(off1[0], off1[1]), textcoords="offset points",
            fontsize=8.4, fontweight="bold", color="#7f1d1d", ha=off1[2], zorder=11,
            arrowprops=dict(arrowstyle="-", color="#dc2626", lw=0.75, alpha=0.65),
        )

        ax2.scatter([r["dc"]], [r["top1"]], s=84, c=pt_col, edgecolors="white", linewidths=1.4, zorder=10)
        ax2.annotate(
            short_lbl, (r["dc"], r["top1"]), xytext=(off2[0], off2[1]), textcoords="offset points",
            fontsize=8.3, fontweight="bold", color="#7f1d1d", ha=off2[2], zorder=11,
            arrowprops=dict(arrowstyle="-", color="#dc2626", lw=0.75, alpha=0.65),
        )

    ax1.set_title("Content vs. Style Similarity (DINO-C ↑ vs. DINO-S ↑)", fontsize=12.5, fontweight="bold", pad=10, color="#0f172a")
    ax1.set_xlabel("Content Preservation — DINO-C ↑ (higher = more faithful)", fontsize=10.5, color="#1e293b")
    ax1.set_ylabel("Style Similarity — DINO-S ↑ (higher = stronger style)", fontsize=10.5, color="#1e293b")
    ax1.set_xlim(0.32, 0.945)
    ax1.set_ylim(0.352, 0.612)
    ax1.grid(True)
    ax1.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=8.6, framealpha=0.94)

    ax2.set_title("Content vs. Style Classification (DINO-C ↑ vs. CSD Top-1 ↑)", fontsize=12.5, fontweight="bold", pad=10, color="#0f172a")
    ax2.set_xlabel("Content Preservation — DINO-C ↑ (higher = more faithful)", fontsize=10.5, color="#1e293b")
    ax2.set_ylabel("Target Style Recognition — CSD Top-1 (%) ↑", fontsize=10.5, color="#1e293b")
    ax2.set_xlim(0.32, 0.945)
    ax2.set_ylim(0, 99)
    ax2.grid(True)
    ax2.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=8.6, framealpha=0.94)

    fig.suptitle(
        "WEAVE / PRISM Global Pareto Frontier (Cross600, S10 Protocol) — Our-Frontier vs. Explored Cloud & External Baselines",
        fontsize=14.0, fontweight="bold", color="#0f172a", y=0.985,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.945])

    png_p = OUT_DIR / "our_frontier_pareto.png"
    svg_p = OUT_DIR / "our_frontier_pareto.svg"
    fig.savefig(png_p, dpi=240)
    fig.savefig(svg_p)
    plt.close(fig)
    print(f"Saved: {png_p} & {svg_p} (frontier={len(f_rows)} pts, cloud={len(cloud)} pts, baselines={len(base_rows)} pts)")


if __name__ == "__main__":
    main()
