"""Plot Inference Speed vs Negative-Log DINO-C/DINO-S Composite Score."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src.eval.pareto import composite_dino_lse, frontier_indices, is_public_literature

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "docs/performance/metrics.csv"
UNIFIED_S4000 = ROOT / "docs/exp1123/single_q32/unified/results_s4000.json"
OUT_DIR = ROOT / "docs/performance/figures"

EXT_LATENCY_MS = {
    "StyleGallery": 30400.0,
    "StyleShot": 24480.0,
    "StyleID": 5040.0,
    "StyTr2": 3040.0,
    "Seedream 4.5": 2000.0,
    "AesPA": 1840.0,
    "SaMAM": 1408.0,
    "SaMST": 800.0,
    "CUT": 400.0,
    "WCT": 293.6,
    "Hist2Style": 200.0,
}


def load_points() -> tuple[list[dict], list[dict], list[dict]]:
    with LEDGER.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    by_arm = {
        r["arm"]: r for r in rows
        if r["protocol"] == "q2a_oracle_cross600"
        and r["n_cross"] == "600"
        and r["is_diagnostic"].lower() == "false"
        and r["dino_c"] and r["dino_s"]
    }

    ext_pts = []
    for name, ms in EXT_LATENCY_MS.items():
        if name not in by_arm:
            continue
        r = by_arm[name]
        dc, ds = float(r["dino_c"]), float(r["dino_s"])
        ms_val = float(r["ms_per_img"]) if r.get("ms_per_img") else ms
        ext_pts.append({
            "name": name,
            "dc": dc,
            "ds": ds,
            "ms": ms_val,
            "img_s": 1000.0 / ms_val,
            "linear": dc + ds,
            "score": composite_dino_lse(dc, ds),
            "is_public": is_public_literature(name),
        })

    weave_speed_arms = [
        ("V19.S2_prism1_v2", "PRISM-1 (λ=1.0)"),
        ("V19.S2_prism1_l075", "PRISM-1 (λ=0.75)"),
        ("V19.S2_prism1_l130", "PRISM-1 (λ=1.30)"),
        ("V20.D6_core_slf_c69", "PRISM-Flow (D6)"),
        ("V20.D5_core_slf_c64", "PRISM-Flow (D5)"),
        ("V20.D3_core_slf_hiC", "PRISM-Flow (D3)"),
        ("V20.D2_core_slf_bal", "PRISM-Flow (D2)"),
        ("V20.D1_core_slf_hiS", "PRISM-Flow (D1)"),
        ("V19.T4_sweet_vgg_nosid", "PRISM-Pro (T4)"),
        ("V20.A1_t4_patchswd", "PRISM-Pro+ (A1)"),
    ]
    weave_pts = []
    for arm, label in weave_speed_arms:
        if arm not in by_arm or not by_arm[arm].get("ms_per_img"):
            continue
        r = by_arm[arm]
        dc, ds, ms_val = float(r["dino_c"]), float(r["dino_s"]), float(r["ms_per_img"])
        weave_pts.append({
            "name": label,
            "arm": arm,
            "dc": dc,
            "ds": ds,
            "ms": ms_val,
            "img_s": 1000.0 / ms_val,
            "linear": dc + ds,
            "score": composite_dino_lse(dc, ds),
        })

    unified_curve = []
    if UNIFIED_S4000.exists():
        arms = json.loads(UNIFIED_S4000.read_text(encoding="utf-8"))["arms"]
        for k, a in arms.items():
            if a.get("wrong"):
                continue
            dc, ds = float(a["dino_c"]), float(a["dino_s"])
            unified_curve.append({
                "tier": f"{a['strength']:g}",
                "strength": float(a["strength"]),
                "dc": dc,
                "ds": ds,
                "linear": dc + ds,
                "score": composite_dino_lse(dc, ds),
            })
        unified_curve.sort(key=lambda x: x["strength"])

    return ext_pts, weave_pts, unified_curve


def render_chart() -> tuple[Path, Path, dict]:
    ext_pts, weave_pts, unified_curve = load_points()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
    })
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14.4, 5.8), dpi=220, layout="constrained")
    fig.patch.set_facecolor("white")

    # ---- Left panel: Why Linear penalizes balanced tiers vs Neg-Log LSE recovers them ----
    xs = [u["strength"] for u in unified_curve]
    lin_vals = [u["linear"] for u in unified_curve]
    lse_vals = [u["score"] for u in unified_curve]
    lin_min, lin_max = min(lin_vals), max(lin_vals)
    lse_min, lse_max = min(lse_vals), max(lse_vals)
    lin_norm = [(v - lin_min) / max(1e-6, lin_max - lin_min) for v in lin_vals]
    lse_norm = [(v - lse_min) / max(1e-6, lse_max - lse_min) for v in lse_vals]

    ax_l.plot(xs, lin_norm, marker="o", ls="--", lw=1.7, color="#64748b",
              label="Naive Linear (DINO-C + DINO-S): peaks at λ=0 (no edit!)")
    ax_l.plot(xs, lse_norm, marker="D", lw=2.2, color="#0f766e",
              label="Neg-Log LSE Score: peaks in balanced zone (λ=0.45–0.85)")
    for x, y_lse, s_raw in zip(xs, lse_norm, lse_vals):
        dy = -13 if x == 0.7 else (10 if x == 0.0 else 7)
        dx = 0
        ax_l.annotate(f"{s_raw:.2f}", (x, y_lse), xytext=(dx, dy), textcoords="offset points",
                      ha="center", fontsize=8, color="#0f766e", fontweight="bold")
    seed_pt = next(p for p in ext_pts if p["name"] == "Seedream 4.5")
    seed_lse_norm = (seed_pt["score"] - lse_min) / max(1e-6, lse_max - lse_min)
    ax_l.axhline(seed_lse_norm, color="#d97706", ls=":", lw=1.4,
                 label=f"Seedream 4.5 S_LSE ({seed_pt['score']:.2f})")
    ax_l.axvspan(0.40, 1.02, color="#ccfbf1", alpha=0.35, zorder=0, label="Balanced working zone")
    ax_l.set(xlabel="Editing Strength λ (PRISM-1-Unified ema_s4000)",
             ylabel="Normalized Score within Sweep [0, 1]",
             title="A. Balanced Tiers Under Linear vs Neg-Log LSE Score")
    ax_l.grid(alpha=0.22)
    ax_l.legend(loc="lower center", bbox_to_anchor=(0.52, 0.01), frameon=True, facecolor="white", framealpha=0.94, fontsize=7.6)
    ax_l.set_ylim(-0.18, 1.15)

    # ---- Right panel: Inference Speed (img/s, log) vs Neg-Log LSE Composite Score ----
    pub = [p for p in ext_pts if p["is_public"]]
    pub_xy = [(p["img_s"], p["score"]) for p in pub]
    f_idx = frontier_indices(pub_xy)
    if f_idx:
        ax_r.plot(*zip(*(pub_xy[i] for i in f_idx)), color="#475569", ls="--", lw=1.3,
                  label="External Public Literature Pareto", zorder=2)

    ext_offsets = {
        "StyleGallery": (8, 5, "left"),
        "StyleShot": (8, 4, "left"),
        "StyleID": (8, 4, "left"),
        "StyTr2": (-8, 8, "right"),
        "Seedream 4.5": (-10, -14, "right"),
        "AesPA": (8, 8, "left"),
        "SaMAM": (8, -4, "left"),
        "CUT": (-8, -11, "right"),
        "Hist2Style": (-8, -12, "right"),
    }
    for p in ext_pts:
        if p["score"] < 3.2:
            continue
        is_seed = p["name"] == "Seedream 4.5"
        if is_seed:
            ax_r.scatter([p["img_s"]], [p["score"]], marker="*", s=260, c="#f59e0b",
                         edgecolors="#92400e", linewidths=1.0, zorder=8,
                         label="Seedream 4.5 (Commercial, 2.0 s/img)")
        else:
            ax_r.scatter([p["img_s"]], [p["score"]], marker="o", s=52, c="#94a3b8",
                         edgecolors="#334155", linewidths=0.8, zorder=4)
        dx, dy, ha = ext_offsets.get(p["name"], (6, 4, "left"))
        ax_r.annotate(f"{p['name']} ({p['score']:.2f})", (p["img_s"], p["score"]),
                      xytext=(dx, dy), textcoords="offset points", ha=ha,
                      fontsize=7.6 if not is_seed else 8.4,
                      fontweight="bold" if is_seed else "normal",
                      color="#92400e" if is_seed else "#334155", zorder=9)

    w_xy = [(p["img_s"], p["score"]) for p in weave_pts]
    w_front = frontier_indices(w_xy)
    if w_front:
        ax_r.plot(*zip(*(w_xy[i] for i in w_front)), color="#dc2626", lw=1.9,
                  label="WEAVE / PRISM Measured 4090 Frontier", zorder=6)

    w_offsets = {
        "PRISM-1 (λ=0.75)": (-10, 9, "right"),
        "PRISM-1 (λ=1.0)": (-10, -3, "right"),
        "PRISM-1 (λ=1.30)": (-10, -16, "right"),
        "PRISM-Pro (T4)": (-10, 9, "right"),
        "PRISM-Flow (D6)": (0, 11, "center"),
        "PRISM-Flow (D5)": (-8, -13, "right"),
        "PRISM-Flow (D2)": (8, 4, "left"),
        "PRISM-Flow (D1)": (8, -4, "left"),
        "PRISM-Pro+ (A1)": (0, -14, "center"),
    }
    for p in weave_pts:
        is_p1 = "PRISM-1 (λ=1.0)" in p["name"]
        ax_r.scatter([p["img_s"]], [p["score"]],
                     marker="D" if is_p1 else "s",
                     s=95 if is_p1 else 46,
                     c="#dc2626" if is_p1 else "#f97316",
                     edgecolors="#7f1d1d", linewidths=0.9, zorder=7)
        if p["name"] in w_offsets:
            dx, dy, ha = w_offsets[p["name"]]
            ax_r.annotate(f"{p['name']} ({p['img_s']:.1f} img/s)", (p["img_s"], p["score"]),
                          xytext=(dx, dy), textcoords="offset points", ha=ha,
                          fontsize=7.4, fontweight="bold" if is_p1 else "normal",
                          color="#991b1b" if is_p1 else "#9a3412", zorder=9)

    ax_r.set_xscale("log")
    ax_r.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax_r.set(xlabel="Inference Throughput (img/s ↑, log scale; 4090 measured / API)",
             ylabel="Neg-Log LSE Score −log(e^−10C + e^−15S) ↑",
             title="B. Inference Speed vs Neg-Log Balanced Score (Cross600)")
    ax_r.set_ylim(3.2, 6.72)
    ax_r.set_xlim(0.022, 95.0)
    ax_r.text(0.02, 0.02, "Off-scale (< 3.2): SaMST (1.55 @ 1.3 img/s), WCT (0.11 @ 3.4 img/s)",
              transform=ax_r.transAxes, fontsize=7.2, color="#64748b")
    ax_r.grid(alpha=0.22, which="both")
    ax_r.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.92, fontsize=8)

    fig.suptitle(
        r"S_LSE(C, S) = −log(exp(−10C) + exp(−15S))  ·  Balanced Quality × Speed",
        fontsize=11.5, fontweight="bold",
    )

    png_p = OUT_DIR / "speed_composite_score.png"
    svg_p = OUT_DIR / "speed_composite_score.svg"
    fig.savefig(png_p, dpi=220, facecolor="white")
    fig.savefig(svg_p, facecolor="white")
    lines = svg_p.read_text(encoding="utf-8").splitlines()
    svg_p.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")
    plt.close(fig)

    summary = {
        "ext": sorted(ext_pts, key=lambda x: x["score"], reverse=True),
        "weave": sorted(weave_pts, key=lambda x: x["score"], reverse=True),
        "unified": unified_curve,
    }
    return png_p, svg_p, summary


if __name__ == "__main__":
    png_p, svg_p, summary = render_chart()
    print(f"Saved -> {png_p} and {svg_p}")
    print("\n--- External ranking (Neg-Log LSE) ---")
    for i, p in enumerate(summary["ext"], 1):
        print(f"{i:2d}. {p['name']:15s} img/s={p['img_s']:6.2f} dc={p['dc']:.4f} ds={p['ds']:.4f} lin={p['linear']:.4f} S_LSE={p['score']:+.3f}")
    print("\n--- Top WEAVE measured ---")
    for p in summary["weave"][:6]:
        print(f"    {p['name']:20s} img/s={p['img_s']:6.2f} dc={p['dc']:.4f} ds={p['ds']:.4f} S_LSE={p['score']:+.3f}")
