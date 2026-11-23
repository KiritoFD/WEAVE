"""Plot Inference Speed vs Composite DINO-C/DINO-S Scores (Neg-Log LSE zoom-in + Power Mean M_-4)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src.eval.pareto import (
    composite_dino_lse,
    composite_dino_power_mean,
    frontier_indices,
    is_public_literature,
)

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


def load_points(score_fn) -> tuple[list[dict], list[dict], list[dict]]:
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
            "score": score_fn(dc, ds),
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
            "score": score_fn(dc, ds),
        })

    unified_curve = []
    if UNIFIED_S4000.exists():
        arms = json.loads(UNIFIED_S4000.read_text(encoding="utf-8"))["arms"]
        for _, a in arms.items():
            if a.get("wrong"):
                continue
            dc, ds = float(a["dino_c"]), float(a["dino_s"])
            unified_curve.append({
                "tier": f"{a['strength']:g}",
                "strength": float(a["strength"]),
                "dc": dc,
                "ds": ds,
                "linear": dc + ds,
                "score": score_fn(dc, ds),
            })
        unified_curve.sort(key=lambda x: x["strength"])

    return ext_pts, weave_pts, unified_curve


def render_pair(
    *,
    stem: str,
    score_fn,
    score_name: str,
    suptitle: str,
    y_label: str,
    y_lim: tuple[float, float],
    exclude_ext: set[str],
    footnote: str,
    ext_offsets: dict,
    w_offsets: dict,
) -> tuple[Path, Path, dict]:
    ext_pts, weave_pts, unified_curve = load_points(score_fn)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
    })
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14.6, 5.8), dpi=220, layout="constrained")
    fig.patch.set_facecolor("white")

    # ---- Left panel: Linear vs Composite across Unified lambda tiers ----
    xs = [u["strength"] for u in unified_curve]
    lin_vals = [u["linear"] for u in unified_curve]
    sc_vals = [u["score"] for u in unified_curve]
    lin_norm = [(v - min(lin_vals)) / max(1e-6, max(lin_vals) - min(lin_vals)) for v in lin_vals]
    sc_norm = [(v - min(sc_vals)) / max(1e-6, max(sc_vals) - min(sc_vals)) for v in sc_vals]

    ax_l.plot(xs, lin_norm, marker="o", ls="--", lw=1.7, color="#64748b",
              label="Naive Linear (DINO-C + DINO-S): peaks at λ=0 (no edit!)")
    ax_l.plot(xs, sc_norm, marker="D", lw=2.2, color="#0f766e",
              label=f"{score_name}: peaks in balanced working zone")
    for x, y_sc, s_raw in zip(xs, sc_norm, sc_vals):
        dy = -13 if x == 0.7 else (10 if x == 0.0 else 7)
        ax_l.annotate(f"{s_raw:.3f}" if abs(s_raw) < 1.5 else f"{s_raw:.2f}",
                      (x, y_sc), xytext=(0, dy), textcoords="offset points",
                      ha="center", fontsize=7.8, color="#0f766e", fontweight="bold")
    seed_pt = next(p for p in ext_pts if p["name"] == "Seedream 4.5")
    seed_norm = (seed_pt["score"] - min(sc_vals)) / max(1e-6, max(sc_vals) - min(sc_vals))
    ax_l.axhline(seed_norm, color="#d97706", ls=":", lw=1.4,
                 label=f"Seedream 4.5 {score_name} ({seed_pt['score']:.3f})")
    ax_l.axvspan(0.40, 1.02, color="#ccfbf1", alpha=0.35, zorder=0, label="Balanced working zone")
    ax_l.set(xlabel="Editing Strength λ (PRISM-1-Unified ema_s4000)",
             ylabel="Normalized Score within Sweep [0, 1]",
             title=f"A. Balanced Tiers Under Linear vs {score_name}")
    ax_l.grid(alpha=0.22)
    ax_l.legend(loc="lower center", bbox_to_anchor=(0.52, 0.01), frameon=True, facecolor="white", framealpha=0.94, fontsize=7.6)
    ax_l.set_ylim(-0.18, 1.15)

    # ---- Right panel: Zoomed Inference Speed vs Composite Score ----
    kept_ext = [p for p in ext_pts if p["name"] not in exclude_ext and y_lim[0] <= p["score"] <= y_lim[1]]
    pub = [p for p in kept_ext if p["is_public"]]
    pub_xy = [(p["img_s"], p["score"]) for p in pub]
    f_idx = frontier_indices(pub_xy)
    if f_idx:
        ax_r.plot(*zip(*(pub_xy[i] for i in f_idx)), color="#475569", ls="--", lw=1.3,
                  label="External Public Literature Pareto", zorder=2)

    for p in kept_ext:
        is_seed = p["name"] == "Seedream 4.5"
        if is_seed:
            ax_r.scatter([p["img_s"]], [p["score"]], marker="*", s=260, c="#f59e0b",
                         edgecolors="#92400e", linewidths=1.0, zorder=8,
                         label="Seedream 4.5 (Commercial, 2.0 s/img)")
        else:
            ax_r.scatter([p["img_s"]], [p["score"]], marker="o", s=54, c="#94a3b8",
                         edgecolors="#334155", linewidths=0.8, zorder=4)
        dx, dy, ha = ext_offsets.get(p["name"], (6, 4, "left"))
        val_txt = f"{p['score']:.3f}" if abs(p["score"]) < 1.5 else f"{p['score']:.2f}"
        ax_r.annotate(f"{p['name']} ({val_txt})", (p["img_s"], p["score"]),
                      xytext=(dx, dy), textcoords="offset points", ha=ha,
                      fontsize=7.8 if not is_seed else 8.5,
                      fontweight="bold" if is_seed else "normal",
                      color="#92400e" if is_seed else "#334155", zorder=9)

    kept_w = [p for p in weave_pts if p["score"] >= y_lim[0]]
    w_xy = [(p["img_s"], p["score"]) for p in kept_w]
    w_front = frontier_indices(w_xy)
    if w_front:
        ax_r.plot(*zip(*(w_xy[i] for i in w_front)), color="#dc2626", lw=1.9,
                  label="WEAVE / PRISM Measured 4090 Frontier", zorder=6)

    for p in kept_w:
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
                          fontsize=7.6, fontweight="bold" if is_p1 else "normal",
                          color="#991b1b" if is_p1 else "#9a3412", zorder=9)

    ax_r.set_xscale("log")
    ax_r.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax_r.set(xlabel="Inference Throughput (img/s ↑, log scale; 4090 measured / API)",
             ylabel=y_label,
             title=f"B. Inference Speed vs {score_name} (Cross600)")
    ax_r.set_ylim(*y_lim)
    ax_r.set_xlim(0.022, 95.0)
    if footnote:
        ax_r.text(0.02, 0.015, footnote, transform=ax_r.transAxes, fontsize=7.0, color="#64748b")
    ax_r.grid(alpha=0.22, which="both")
    ax_r.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.92, fontsize=8)

    fig.suptitle(suptitle, fontsize=11.5, fontweight="bold")
    png_p = OUT_DIR / f"{stem}.png"
    svg_p = OUT_DIR / f"{stem}.svg"
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


def render_all() -> None:
    # 1. Zoomed Neg-Log LSE (top cluster separated, bottom outliers removed from plot)
    p1, s1, sum1 = render_pair(
        stem="speed_composite_score",
        score_fn=composite_dino_lse,
        score_name="Neg-Log LSE",
        suptitle=r"S_LSE(C, S) = −log(exp(−10C) + exp(−15S))  ·  Zoomed Quality × Speed",
        y_label="Neg-Log LSE Score −log(e^−10C + e^−15S) ↑",
        y_lim=(5.12, 6.50),
        exclude_ext={"StyleID", "StyleShot", "SaMST", "WCT"},
        footnote="Below y-min: StyleID (3.86), StyleShot (3.56), SaMST (1.55), WCT (0.11)",
        ext_offsets={
            "StyleGallery": (8, 6, "left"),
            "StyTr2": (-8, 8, "right"),
            "Seedream 4.5": (-10, -14, "right"),
            "AesPA": (8, 8, "left"),
            "SaMAM": (8, -6, "left"),
            "CUT": (-8, -11, "right"),
            "Hist2Style": (-8, -12, "right"),
        },
        w_offsets={
            "PRISM-1 (λ=0.75)": (-10, 9, "right"),
            "PRISM-1 (λ=1.0)": (-10, -3, "right"),
            "PRISM-1 (λ=1.30)": (-10, -16, "right"),
            "PRISM-Pro (T4)": (-10, 9, "right"),
            "PRISM-Flow (D6)": (0, 11, "center"),
            "PRISM-Flow (D5)": (-10, -10, "right"),
            "PRISM-Flow (D3)": (8, -4, "left"),
            "PRISM-Flow (D2)": (8, 4, "left"),
            "PRISM-Pro+ (A1)": (-10, -10, "right"),
        },
    )

    # 2. Parameter-free non-log Holder Generalized Mean M_-4(C, S)
    p2, s2, sum2 = render_pair(
        stem="speed_power_mean",
        score_fn=composite_dino_power_mean,
        score_name="Power Mean M_-4",
        suptitle=r"M_−4(C, S) = ((C^−4 + S^−4) / 2)^(−1/4)  ·  Parameter-Free Non-Log Quality × Speed",
        y_label="Generalized Power Mean M_−4(DINO-C, DINO-S) ↑",
        y_lim=(0.395, 0.598),
        exclude_ext={"SaMST", "WCT"},
        footnote="No params/logs. Off-scale: SaMST (0.190), WCT (0.028)",
        ext_offsets={
            "StyleGallery": (8, 5, "left"),
            "StyTr2": (-8, 8, "right"),
            "AesPA": (-8, 8, "right"),
            "Seedream 4.5": (-10, -12, "right"),
            "SaMAM": (8, 5, "left"),
            "CUT": (-8, -11, "right"),
            "Hist2Style": (-8, -12, "right"),
            "StyleID": (8, 4, "left"),
            "StyleShot": (8, 4, "left"),
        },
        w_offsets={
            "PRISM-Flow (D6)": (8, 4, "left"),
            "PRISM-Flow (D2)": (8, -4, "left"),
            "PRISM-Flow (D1)": (8, -10, "left"),
            "PRISM-Pro (T4)": (-10, 7, "right"),
            "PRISM-Pro+ (A1)": (0, -14, "center"),
            "PRISM-1 (λ=1.30)": (-10, 10, "right"),
            "PRISM-1 (λ=1.0)": (-10, -2, "right"),
            "PRISM-1 (λ=0.75)": (-10, -14, "right"),
        },
    )

    print(f"Saved Zoomed LSE -> {p1} & {s1}")
    print(f"Saved Power Mean M_-4 -> {p2} & {s2}")
    print("\n--- External ranking (M_-4 non-log, parameter-free) ---")
    for i, p in enumerate(sum2["ext"], 1):
        print(f"{i:2d}. {p['name']:15s} img/s={p['img_s']:6.2f} dc={p['dc']:.4f} ds={p['ds']:.4f} M_-4={p['score']:.4f} S_LSE={composite_dino_lse(p['dc'], p['ds']):.3f}")
    print("\n--- Unified sweep M_-4 ---")
    for u in sum2["unified"]:
        print(f"  λ={u['strength']:<4g} dc={u['dc']:.4f} ds={u['ds']:.4f} M_-4={u['score']:.4f}")


if __name__ == "__main__":
    render_all()
