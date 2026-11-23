"""绘制简化版全局 Pareto 图（计算范式对抗版：Diffusion-Heavy vs. Conventional Lightweight vs. PRISM）。

严格落实 ICME 2027 论文视觉叙事要求：
1. 每张子图仅保留一条橙红色全局 Pareto 线（PRISM Global Pareto Frontier），突出 6 个代表点；
   不再画多条 Ours 曲线，不标注 V22/V23/V25/R12/R13 等内部版本名。
2. 全部速度口径严格统一为 RTX 3060 实测（气泡半径 ∝ RTX 3060 Max-Batch E2E img/s），
   慢速基线显式标注 s/img。
3. 移除绿色未达标 Target 框，聚焦完整的 Content–Style–Compute 三维前沿对比。

输出:
  docs/performance/figures/ours_pareto.{png,svg}
  docs/icme2027/figures/fig1_prism_pareto.{pdf,png,svg}
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

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "docs/performance/metrics.csv"
BENCH_3060_PATH = ROOT / "runs/v25/r15_3060_maxbatch_bench.json"
R16_3060_PATH = ROOT / "runs/v25/r16_3060_stage_and_dec_bench.json"
OUT_DIR = ROOT / "docs/performance/figures"

ORANGE_RED = "#ff4500"
ORANGE_DARK = "#9a3412"
ORANGE_FILL = "#fed7aa"

# 保留全量真实帕累托最优点 (先不砍优胜点: DINO-S=0.5789, Top1=76.83%, PatchSWD 64.33%, 6步JPk 59.0%@0.631, D6 55.2%@0.704, HC 0.772)
OURS_PARETO_DS = [
    ("V20.D1_core_slf_hiS",       "PRISM-Flow (Max-Style)", (-6, 14, "left"),    "prism_flow_d2"),
    ("V20.D2_core_slf_bal",       "PRISM-Flow (Bal-16s)",   (-10, -20, "right"), "prism_flow_d2"),
    ("ICME_JPk_s6_csj",           "PRISM-Flow (6-Step)",    (14, 10, "left"),    "d6_csj_s6"),
    ("V25_Tr_T1_d6hcsj",          "PRISM-Flow (Hi-C)",      (14, 10, "left"),    "d6_csj_b4"),
    ("B_d6_t070",                 "PRISM-Flow (D6-τ0.7)",   (12, 10, "left"),    "d6_full_b4"),
    ("HC_R12_g100",               "PRISM-HC (γ=1.0)",       (14, 12, "left"),    "d6_csj_b4"),
    ("MechA_B4anc_A0_global_g55", "PRISM-HC (γ=0.55)",      (14, 6, "left"),     "prism_hc_r12"),
]

# 右面板 (DINO-C vs CSD Top-1): 完整真实帕累托包络
OURS_PARETO_T1 = [
    ("T4Pg45_wc10",               "PRISM-Flow (76.8%)",     (12, 6, "left"),     "d5_full_10s"),
    ("V20.A1_t4_patchswd",        "PRISM-Pro+ (64.3%)",     (14, 10, "left"),    "d6_full_b4"),
    ("B_d6_t070",                 "PRISM-Flow (D6-τ0.7)",   (12, 12, "left"),    "d6_full_b4"),
    ("M1_lin_g55",                "PRISM-Hybrid (γ=0.55)",  (14, 6, "left"),     "prism_hc_r12"),
    ("MechA_B4anc_A0_global_g55", "PRISM-HC (γ=0.55)",      (14, -4, "left"),    "prism_hc_r12"),
]

# 突出代表点: PRISM-1 (消费级 3060 最高吞吐 5.97 img/s)
PRISM1_HIGHLIGHT = ("M0_prism1_l100", "PRISM-1 (5.97 img/s)", (-14, -22, "right"), (-24, -10, "right"), "prism1")

DIFFUSION_BASELINES = {
    "StyleShot":    ("StyleShot\n(24.5 s/img)",            (8, 6, "left"),     (10, -10, "left"),  24480.0),
    "StyleID":      ("StyleID (5.0 s/img)",                (8, 10, "left"),    (10, -4, "left"),    5040.0),
    "StyleGallery": ("StyleGallery\n(CVPR26, 30.4 s/img)", (14, 10, "left"),   (-16, -16, "right"), 30400.0),
}

CONVENTIONAL_BASELINES = {
    "AesPA":        ("AesPA (1.8 s/img)",         (8, -4, "left"),     (-10, -8, "right"),  1840.0),
    "StyTr2":       ("StyTr2 (3.0 s/img)",        (-10, -12, "right"), (-10, -10, "right"), 3040.0),
    "Seedream 4.5": ("Seedream 4.5 (2.0 s/img)",  (-10, -14, "right"), (-12, -14, "right"), 2000.0),
    "SaMAM":        ("SaMAM (1.4 s/img)",         (-10, -12, "right"), (10, -4, "left"),    1408.0),
    "CUT":          ("CUT (0.4 s/img)",           (8, 10, "left"),     (-10, -6, "right"),   400.0),
    "Hist2Style":   ("Hist2Style (0.2 s/img)",    (-16, -2, "right"),  (-16, -6, "right"),   200.0),
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
    by_arm = {r["arm"]: r for r in rows}
    step2b_p = ROOT / "docs/v26/4090/data/icme_step2b_3060.json"
    if step2b_p.exists():
        d2b = json.load(open(step2b_p, encoding="utf-8"))
        for k, v in d2b.get("arms", {}).items():
            rec = {"arm": k, "group": "ICME_Step2b", "dc": float(v["dino_c"]),
                   "ds": float(v["dino_s"]), "top1": float(v["csd_top1"]) * 100.0, "ms": None}
            rows.append(rec)
            by_arm[k] = rec
    return rows, by_arm


def load_3060_speeds() -> dict[str, float]:
    spd = {}
    if BENCH_3060_PATH.exists():
        d = json.load(open(BENCH_3060_PATH, encoding="utf-8"))
        for eng, info in d["results"].items():
            spd[eng] = float(info["best_e2e"]["e2e_img_per_s"])
    if R16_3060_PATH.exists():
        d16 = json.load(open(R16_3060_PATH, encoding="utf-8"))
        d2_speeds = [
            1000.0 / float(v["total_rgb2rgb"]["p50_ms_per_img"])
            for k, v in d16.get("stages", {}).items()
            if k.startswith("prism_flow_d2_")
        ]
        if d2_speeds:
            spd["prism_flow_d2"] = max(d2_speeds)
    step2b_p = ROOT / "docs/v26/4090/data/icme_step2b_3060.json"
    if step2b_p.exists():
        b16 = json.load(open(step2b_p, encoding="utf-8")).get("bench_3060", {}).get("b16", {})
        if b16.get("e2e_s6_csj_with_t4anc_ms"):
            spd["d6_csj_s6"] = 1000.0 / float(b16["e2e_s6_csj_with_t4anc_ms"])
    spd.setdefault("d6_csj_s6", 1.85)
    spd.setdefault("prism_flow_d2", 1.47)
    return spd


def pareto_envelope(pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    s = sorted(pts, key=lambda p: (-p[0], -p[1]))
    out, best_y = [], -1e9
    for x, y in s:
        if y > best_y + 1e-6:
            out.append((x, y))
            best_y = y
    return sorted(out, key=lambda p: p[0])


def draw_radius_legend(ax, loc="lower left"):
    if loc == "lower left":
        bx, by, bw, bh = 0.015, 0.215, 0.365, 0.165
    else:
        bx, by, bw, bh = 0.015, 0.025, 0.365, 0.165
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

    all_baselines = {**DIFFUSION_BASELINES, **CONVENTIONAL_BASELINES}
    hi_set = {s[0] for s in OURS_PARETO_DS} | {s[0] for s in OURS_PARETO_T1} | {PRISM1_HIGHLIGHT[0]}
    base_set = set(all_baselines.keys()) | {"WEAVE (Published)"}
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

    # 背景探索点云
    ax1.scatter([r["dc"] for r in cloud], [r["ds"] for r in cloud],
                s=10, c="#64748b", alpha=0.11, edgecolors="none",
                label=f"Explored configuration cloud (n={len(cloud)})", zorder=2)
    ax2.scatter([r["dc"] for r in cloud], [r["top1"] for r in cloud],
                s=10, c="#64748b", alpha=0.11, edgecolors="none",
                label=f"Explored configuration cloud (n={len(cloud)})", zorder=2)

    # 外部基线包络
    base_rows = [by_arm[k] for k in all_baselines if k in by_arm]
    env_b_ds = pareto_envelope([(r["dc"], r["ds"]) for r in base_rows])
    env_b_t1 = pareto_envelope([(r["dc"], r["top1"]) for r in base_rows])
    ax1.plot([p[0] for p in env_b_ds], [p[1] for p in env_b_ds],
             color="#475569", linestyle="--", linewidth=1.35, alpha=0.68,
             label="External Baselines Envelope", zorder=3)
    ax2.plot([p[0] for p in env_b_t1], [p[1] for p in env_b_t1],
             color="#475569", linestyle="--", linewidth=1.35, alpha=0.68,
             label="External Baselines Envelope", zorder=3)

    # 1) Diffusion-Heavy Baselines (蓝色)
    for k, (lbl, off1, off2, fallback_ms) in DIFFUSION_BASELINES.items():
        if k not in by_arm:
            continue
        r = by_arm[k]
        ms = r["ms"] if r["ms"] else fallback_ms
        sz = img_s_to_area(1000.0 / ms)
        is_sg = (k == "StyleGallery")
        mk = "*" if is_sg else "^"
        sz_use = max(sz * 1.4, 54.0) if is_sg else max(sz, 38.0)
        kw_arr = dict(arrowstyle="-", color="#2563eb", lw=0.85, alpha=0.8) if is_sg else None
        ax1.scatter([r["dc"]], [r["ds"]], marker=mk, s=sz_use, c="#2563eb", edgecolors="#1e3a8a", linewidths=0.9, zorder=7)
        ax1.annotate(lbl, (r["dc"], r["ds"]), xytext=(off1[0], off1[1]), textcoords="offset points",
                     fontsize=8.3 if is_sg else 7.7, fontweight="bold" if is_sg else "normal",
                     color="#1d4ed8", ha=off1[2], arrowprops=kw_arr, zorder=9)
        ax2.scatter([r["dc"]], [r["top1"]], marker=mk, s=sz_use, c="#2563eb", edgecolors="#1e3a8a", linewidths=0.9, zorder=7)
        ax2.annotate(lbl, (r["dc"], r["top1"]), xytext=(off2[0], off2[1]), textcoords="offset points",
                     fontsize=8.3 if is_sg else 7.7, fontweight="bold" if is_sg else "normal",
                     color="#1d4ed8", ha=off2[2], arrowprops=kw_arr, zorder=9)

    # 2) Conventional Lightweight / Feed-Forward Baselines (深灰方块)
    for k, (lbl, off1, off2, fallback_ms) in CONVENTIONAL_BASELINES.items():
        if k not in by_arm:
            continue
        r = by_arm[k]
        ms = r["ms"] if r["ms"] else fallback_ms
        sz = img_s_to_area(1000.0 / ms)
        ax1.scatter([r["dc"]], [r["ds"]], marker="s", s=sz, c="#334155", edgecolors="white", linewidths=0.9, zorder=7)
        ax1.annotate(lbl, (r["dc"], r["ds"]), xytext=(off1[0], off1[1]), textcoords="offset points",
                     fontsize=7.5, color="#334155", ha=off1[2], zorder=9)
        ax2.scatter([r["dc"]], [r["top1"]], marker="s", s=sz, c="#334155", edgecolors="white", linewidths=0.9, zorder=7)
        ax2.annotate(lbl, (r["dc"], r["top1"]), xytext=(off2[0], off2[1]), textcoords="offset points",
                     fontsize=7.5, color="#334155", ha=off2[2], zorder=9)

    ax1.scatter([], [], marker="*", s=75, c="#2563eb", edgecolors="#1e3a8a", label="Arbitrary-Ref Diffusion (5.0–30.4 s/img)")
    ax1.scatter([], [], marker="s", s=45, c="#334155", edgecolors="white", label="Arbitrary-Ref Lightweight (0.2–3.0 s/img)")
    ax2.scatter([], [], marker="*", s=75, c="#2563eb", edgecolors="#1e3a8a", label="Arbitrary-Ref Diffusion (5.0–30.4 s/img)")
    ax2.scatter([], [], marker="s", s=45, c="#334155", edgecolors="white", label="Arbitrary-Ref Lightweight (0.2–3.0 s/img)")

    # 3) 单一橙红色全局 PRISM Pareto 线 + 收束代表点
    ds_pts = [(by_arm[a], lbl, off, spd.get(eng, 3.0)) for a, lbl, off, eng in OURS_PARETO_DS if a in by_arm]
    ax1.plot([p[0]["dc"] for p in ds_pts], [p[0]["ds"] for p in ds_pts],
             color=ORANGE_RED, linestyle="-", linewidth=2.9, alpha=0.94, zorder=8,
             label="PRISM (5-Style Amortized Frontier)")
    ax1.fill_between([p[0]["dc"] for p in ds_pts], [p[0]["ds"] for p in ds_pts], 0.35,
                     color=ORANGE_FILL, alpha=0.14, zorder=1)
    for r, lbl, off, isp in ds_pts:
        ax1.scatter([r["dc"]], [r["ds"]], s=img_s_to_area(isp), c=ORANGE_RED,
                    edgecolors="white", linewidths=1.5, alpha=0.95, zorder=10)
        ax1.annotate(lbl, (r["dc"], r["ds"]), xytext=(off[0], off[1]), textcoords="offset points",
                     fontsize=8.3, fontweight="bold", color=ORANGE_DARK, ha=off[2], zorder=11,
                     arrowprops=dict(arrowstyle="-", color=ORANGE_RED, lw=0.85, alpha=0.8))

    t1_pts = [(by_arm[a], lbl, off, spd.get(eng, 3.0)) for a, lbl, off, eng in OURS_PARETO_T1 if a in by_arm]
    ax2.plot([p[0]["dc"] for p in t1_pts], [p[0]["top1"] for p in t1_pts],
             color=ORANGE_RED, linestyle="-", linewidth=2.9, alpha=0.94, zorder=8,
             label="PRISM (5-Style Amortized Frontier)")
    ax2.fill_between([p[0]["dc"] for p in t1_pts], [p[0]["top1"] for p in t1_pts], 0.0,
                     color=ORANGE_FILL, alpha=0.13, zorder=1)
    for r, lbl, off, isp in t1_pts:
        ax2.scatter([r["dc"]], [r["top1"]], s=img_s_to_area(isp), c=ORANGE_RED,
                    edgecolors="white", linewidths=1.5, alpha=0.95, zorder=10)
        ax2.annotate(lbl, (r["dc"], r["top1"]), xytext=(off[0], off[1]), textcoords="offset points",
                     fontsize=8.3, fontweight="bold", color=ORANGE_DARK, ha=off[2], zorder=11,
                     arrowprops=dict(arrowstyle="-", color=ORANGE_RED, lw=0.85, alpha=0.8))

    # PRISM-1 第 6 个突出代表点
    a1, lbl1, off1, off2, eng1 = PRISM1_HIGHLIGHT
    if a1 in by_arm:
        r = by_arm[a1]
        sz1 = img_s_to_area(spd.get(eng1, 5.97))
        ax1.scatter([r["dc"]], [r["ds"]], s=sz1, c="#ff6b35", edgecolors="white", linewidths=1.5, alpha=0.95, zorder=10)
        ax1.annotate(lbl1, (r["dc"], r["ds"]), xytext=(off1[0], off1[1]), textcoords="offset points",
                     fontsize=8.3, fontweight="bold", color=ORANGE_DARK, ha=off1[2], zorder=11,
                     arrowprops=dict(arrowstyle="-", color="#ff6b35", lw=0.85, alpha=0.8))
        ax2.scatter([r["dc"]], [r["top1"]], s=sz1, c="#ff6b35", edgecolors="white", linewidths=1.5, alpha=0.95, zorder=10)
        ax2.annotate(lbl1, (r["dc"], r["top1"]), xytext=(off2[0], off2[1]), textcoords="offset points",
                     fontsize=8.3, fontweight="bold", color=ORANGE_DARK, ha=off2[2], zorder=11,
                     arrowprops=dict(arrowstyle="-", color="#ff6b35", lw=0.85, alpha=0.8))

    draw_radius_legend(ax1, loc="lower left")
    draw_radius_legend(ax2, loc="bottom left")

    ax1.set_title("(a) Content vs. Style Similarity (DINO-C ↑ vs. DINO-S ↑)", fontsize=12.2, fontweight="bold", pad=10, color="#0f172a")
    ax1.set_xlabel("Content Preservation — DINO-C ↑ (higher = more faithful)", fontsize=10.5, color="#1e293b")
    ax1.set_ylabel("Style Similarity — DINO-S ↑ (higher = stronger style)", fontsize=10.5, color="#1e293b")
    ax1.set_xlim(0.32, 0.94)
    ax1.set_ylim(0.352, 0.615)
    ax1.grid(True)
    ax1.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=8.0, framealpha=0.95)

    ax2.set_title("(b) Content vs. Target Style Recognition (DINO-C ↑ vs. CSD Top-1 ↑)", fontsize=12.2, fontweight="bold", pad=10, color="#0f172a")
    ax2.set_xlabel("Content Preservation — DINO-C ↑ (higher = more faithful)", fontsize=10.5, color="#1e293b")
    ax2.set_ylabel("Target Style Recognition — CSD Top-1 (%) ↑", fontsize=10.5, color="#1e293b")
    ax2.set_xlim(0.32, 0.94)
    ax2.set_ylim(0, 96)
    ax2.grid(True)
    ax2.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=8.0, framealpha=0.95)

    fig.suptitle(
        "5-Style Amortized PRISM vs. Arbitrary-Ref Baselines (Cross600, RTX 3060) — StyleGallery 750-img Infer: 6.33h vs. PRISM Train+Infer: <0.25h",
        fontsize=12.4, fontweight="bold", color="#0f172a", y=0.985,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.945])

    png_p = OUT_DIR / "ours_pareto.png"
    svg_p = OUT_DIR / "ours_pareto.svg"
    fig.savefig(png_p, dpi=240)
    fig.savefig(svg_p)
    icme_fig_dir = ROOT / "docs/icme2027/figures"
    if icme_fig_dir.exists():
        for ext in ("pdf", "png", "svg"):
            fig.savefig(icme_fig_dir / f"fig1_prism_pareto.{ext}", dpi=240 if ext == "png" else None)
    plt.close(fig)
    print(f"Saved: {png_p} & {svg_p} (+ fig1_prism_pareto)")


if __name__ == "__main__":
    main()
