"""Generate consolidated 1x3 Paper-Style Pareto Triptych (SVG/PNG + Interactive HTML) for CLEAVE V32-V34+.

Replicates the exact 1x3 layout of ICME Fig. 1 (docs/icme2027/figures/fig1_prism_pareto):
  (a) Full-Spectrum Quality Control: DINO-C vs DINO-S
  (b) Stronger Style Specificity: DINO-C vs CSD Top-1 (%)
  (c) Real-Time Performance at Scale: Cold-Start RGB->RGB Throughput (img/s, log) vs CSD Top-1 (%)

Overlays verified V34 (44M C2 + A3 Start, 14.5 img/s @ b16) against V32 (22M E-Core 33.1 img/s + P-Core 18.2-18.5 img/s)
and stratified external baselines under identical Cross600 / S10 / Zero-DINO conditions.
Also preserves V33 / V34 Stage A-D & Holdout-750 interactive views.

Usage:
  python docs/performance/plot_cleave_pareto.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from exp.v32_nodino.plot_v32_pareto import (
    CANONICAL_TRAJECTORY,
    EXTERNAL_BASELINES,
    LEDGER_JSON,
    STANDALONE_ECORE_MAX,
)
from src.eval.pareto import frontier_indices
from src.utils.runtime import read_json

FIG_DIR = REPO / "docs/performance/figures"

# Fallback canonical V32 trajectory if runs/v32_nodino/v32_final_ledger.json is remote
V32_FALLBACK_CANONICAL = [
    {"knob": "0.40", "show_lbl": True,  "ips": 33.1, "seg": "ecore", "dc": 0.8560, "ds": 0.4077, "t1": 20.17},
    {"knob": "0.60", "show_lbl": True,  "ips": 33.1, "seg": "ecore", "dc": 0.7781, "ds": 0.4407, "t1": 38.83},
    {"knob": "0.75", "show_lbl": True,  "ips": 33.1, "seg": "ecore", "dc": 0.7204, "ds": 0.4671, "t1": 45.17},
    {"knob": "0.80", "show_lbl": False, "ips": 18.2, "seg": "pcore", "dc": 0.6585, "ds": 0.5087, "t1": 52.33},
    {"knob": "0.85", "show_lbl": True,  "ips": 18.2, "seg": "pcore", "dc": 0.6010, "ds": 0.5266, "t1": 62.33},
    {"knob": "0.93", "show_lbl": False, "ips": 18.2, "seg": "pcore", "dc": 0.5524, "ds": 0.5446, "t1": 65.67},
    {"knob": "1.00", "show_lbl": True,  "ips": 18.2, "seg": "pcore", "dc": 0.4896, "ds": 0.5628, "t1": 75.17},
    {"knob": "1.25", "show_lbl": True,  "ips": 18.5, "seg": "pcore", "dc": 0.4171, "ds": 0.5814, "t1": 77.33},
    {"knob": "1.35", "show_lbl": False, "ips": 18.5, "seg": "pcore", "dc": 0.3772, "ds": 0.5919, "t1": 77.33},
    {"knob": "1.50", "show_lbl": True,  "ips": 18.5, "seg": "pcore", "dc": 0.2805, "ds": 0.6054, "t1": 80.17},
]
V32_FALLBACK_EC_MAX = {"lbl": "E-Core (1.00)", "ips": 33.1, "dc": 0.5674, "ds": 0.5077, "t1": 67.17}


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_curve(levels: dict[str, dict[str, Any]], *, ips: float | None = None) -> list[dict[str, Any]]:
    pts = []
    for k, m in levels.items():
        if "wrong" in k:
            continue
        pts.append({
            "knob": k,
            "dc": round(float(m["dino_c"]), 4),
            "ds": round(float(m["dino_s"]), 4),
            "t1": round(float(m["csd_top1"]) * 100.0, 2),
            "lpips": round(float(m.get("lpips", 0.0)), 4),
            "same": round(float(m.get("same150_top1", 0.0)) * 100.0, 1),
            "ips": ips,
        })
    return sorted(pts, key=lambda p: p["dc"])


def collect_triptych_data() -> dict[str, Any]:
    """Assemble verified external baselines, canonical V32 P/E-Core, V33, and V34 datasets."""
    if LEDGER_JSON.exists():
        ledger = read_json(LEDGER_JSON)["spectrum"]
        v32_traj = []
        for key, knob, show_lbl, ips, seg in CANONICAL_TRAJECTORY:
            m = ledger[key]
            v32_traj.append({
                "knob": knob,
                "show_lbl": show_lbl,
                "ips": ips,
                "seg": seg,
                "dc": round(float(m["dino_c"]), 4),
                "ds": round(float(m["dino_s"]), 4),
                "t1": round(float(m["csd_top1"]) * 100.0, 2),
            })
        ek, elbl, eips = STANDALONE_ECORE_MAX
        em = ledger[ek]
        v32_ec_max = {
            "lbl": elbl,
            "ips": eips,
            "dc": round(float(em["dino_c"]), 4),
            "ds": round(float(em["dino_s"]), 4),
            "t1": round(float(em["csd_top1"]) * 100.0, 2),
        }
    else:
        v32_traj = V32_FALLBACK_CANONICAL
        v32_ec_max = V32_FALLBACK_EC_MAX

    ext_list = []
    for name, dc, ds, t1, sec, tier in EXTERNAL_BASELINES:
        ext_list.append({
            "name": name,
            "short": name.split(" (")[0],
            "dc": dc,
            "ds": ds,
            "t1": t1,
            "sec": sec,
            "ips": round(1.0 / sec, 4),
            "tier": tier,
        })

    fa_idx = frontier_indices([(b["dc"], b["ds"]) for b in ext_list])
    fb_idx = frontier_indices([(b["dc"], b["t1"]) for b in ext_list])
    fc_idx = frontier_indices([(b["ips"], b["t1"]) for b in ext_list])

    sd = _load_json(REPO / "docs/v34/data/stage_d_report.json")
    c2a3 = _load_json(REPO / "docs/v34/data/c2_a3_control_report.json")
    sc = _load_json(REPO / "docs/v34/data/stage_c_report.json")
    sa = _load_json(REPO / "docs/v34/data/stage_a_report.json")
    sb = _load_json(REPO / "docs/v34/data/stage_b_report.json")
    p2 = _load_json(REPO / "docs/v33/data/p2_report.json")

    v34_ips = float(sd["cold_start_rgb_to_rgb_speed"]["b16_ips"]) if sd else 14.5
    v34_winner = _extract_curve(c2a3["levels"], ips=v34_ips) if c2a3 else []
    v34_c2 = _extract_curve(sc["arms"]["C2_44m_q32"]["levels"], ips=v34_ips) if sc else []
    v34_d_merged = _extract_curve(sd["test750_levels"], ips=v34_ips) if sd else []
    v33_s2 = _extract_curve(p2["arms"]["P2_S2_hi32_pert30"]["levels"], ips=15.0) if p2 else []

    holdout_pts = []
    if sd and "holdout_750" in sd:
        for k, v in sd["holdout_750"].items():
            c = v["cross600"]
            holdout_pts.append({
                "label": k, "dc": round(c["dino_c"], 4), "ds": round(c["dino_s"], 4),
                "t1": round(c["csd_top1"] * 100.0, 2), "lpips": round(c["lpips"], 4),
                "same": round(v["same150"]["csd_top1"] * 100.0, 1),
            })
    if c2a3 and "holdout" in c2a3:
        for k, v in c2a3["holdout"].items():
            c = v["cross600"]
            holdout_pts.append({
                "label": k, "dc": round(c["dino_c"], 4), "ds": round(c["dino_s"], 4),
                "t1": round(c["csd_top1"] * 100.0, 2), "lpips": round(c["lpips"], 4),
                "same": round(v["same150"]["csd_top1"] * 100.0, 1),
            })

    stage_a_pts = []
    if sa:
        for mode in ("A0", "A1", "A2", "A3"):
            for kl in ("s085", "s100"):
                m = sa["arms"].get(f"StageA_{mode}_{kl}")
                if m:
                    stage_a_pts.append({
                        "label": f"{mode}_{kl}", "mode": mode, "knot": kl,
                        "dc": round(float(m["dino_c"]), 4), "ds": round(float(m["dino_s"]), 4),
                        "t1": round(float(m["csd_top1"]) * 100.0, 2),
                        "min_s": round(float(m["per_tgt_dino_s"]["Min"]), 4),
                    })

    stage_b_curves = {}
    if sb:
        for k in ("B0_k8_discrete", "B1_inst_417", "B2_inst_tex1101"):
            if k in sb["arms"]:
                stage_b_curves[k] = _extract_curve(sb["arms"][k]["levels"], ips=18.2)

    return {
        "external": ext_list,
        "ext_front_a": [ext_list[i] for i in fa_idx],
        "ext_front_b": [ext_list[i] for i in fb_idx],
        "ext_front_c": [ext_list[i] for i in fc_idx],
        "v32_ecore": [p for p in v32_traj if p["seg"] == "ecore"],
        "v32_pcore": [p for p in v32_traj if p["seg"] == "pcore"],
        "v32_ec_max": v32_ec_max,
        "v34_winner": v34_winner,
        "v34_c2": v34_c2,
        "v34_d_merged": v34_d_merged,
        "v33_s2": v33_s2,
        "v34_ips": v34_ips,
        "stage_a": stage_a_pts,
        "stage_b": stage_b_curves,
        "holdout": holdout_pts,
    }


def save_matplotlib_triptych(data: dict[str, Any]) -> None:
    """Save crisp 1x3 publication SVG/PNG matching ICME Fig. 1 (docs/v32 + docs/performance)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    TEAL, TEAL_DARK = "#0d9488", "#115e59"
    BLUE_V32, BLUE_DARK = "#2563eb", "#1e3a8a"
    WIN_RED, WIN_DARK = "#e11d48", "#881337"
    HEAVY_PURPLE = "#7c3aed"
    LIGHT_BLUE = "#0284c7"
    SLATE_FRONT = "#64748b"
    INK = "#0f172a"

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial"],
        "font.size": 8.2,
        "axes.labelsize": 8.8,
        "axes.titlesize": 9.3,
        "xtick.labelsize": 7.8,
        "ytick.labelsize": 7.8,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })

    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(13.9, 4.45), dpi=240)
    ec = data["v32_ecore"]
    pc32 = data["v32_pcore"]
    ec_m = data["v32_ec_max"]
    v34 = data["v34_winner"]
    ext = data["external"]

    def _draw_ext(ax, x_key: str, y_key: str, front_pts: list[dict], *, show_sg_star: bool = True):
        ax.plot([p[x_key] for p in front_pts], [p[y_key] for p in front_pts],
                "--", color=SLATE_FRONT, lw=1.35, alpha=0.85, zorder=2)
        for b in ext:
            if b["short"] == "StyleGallery" and show_sg_star:
                ax.scatter([b[x_key]], [b[y_key]], marker="*", s=115, c=HEAVY_PURPLE, edgecolors="#3b0764", lw=0.7, zorder=7)
            elif b["tier"] == "heavy":
                ax.scatter([b[x_key]], [b[y_key]], marker="^", s=34, c="#8b5cf6", alpha=0.68, edgecolors="#4c1d95", lw=0.5, zorder=4)
            else:
                ax.scatter([b[x_key]], [b[y_key]], marker="s", s=26, c=LIGHT_BLUE, alpha=0.65, edgecolors="#075985", lw=0.5, zorder=4)

    def _ann(ax, txt: str, xy, off, *, ha="left", col=INK, bold=False, fs=7.3, leader=False):
        arrow = dict(arrowstyle="-", color="#94a3b8", lw=0.65, shrinkA=0, shrinkB=3) if leader else None
        ax.annotate(txt, xy=xy, xytext=off, textcoords="offset points", ha=ha, va="center",
                    fontsize=fs, color=col, fontweight="bold" if bold else "normal", arrowprops=arrow, zorder=10)

    # Panel (a)
    ax_a.axvspan(0.65, 0.94, color="#ecfdf5", alpha=0.45, zorder=0)
    _draw_ext(ax_a, "dc", "ds", data["ext_front_a"])
    ax_a.plot([p["dc"] for p in ec], [p["ds"] for p in ec], "-", color=TEAL, lw=2.4, zorder=5)
    ax_a.plot([ec[-1]["dc"], pc32[0]["dc"]], [ec[-1]["ds"], pc32[0]["ds"]], ":", color="#64748b", lw=1.4, zorder=4)
    ax_a.plot([p["dc"] for p in pc32], [p["ds"] for p in pc32], "--", color=BLUE_V32, lw=1.9, alpha=0.85, zorder=5)
    ax_a.plot([p["dc"] for p in v34], [p["ds"] for p in v34], "-", color=WIN_RED, lw=3.0, zorder=8)
    ax_a.scatter([p["dc"] for p in ec], [p["ds"] for p in ec], marker="D", s=40, c=TEAL, edgecolors=TEAL_DARK, lw=0.9, zorder=6)
    ax_a.scatter([p["dc"] for p in pc32], [p["ds"] for p in pc32], marker="o", s=34, c=BLUE_V32, edgecolors=BLUE_DARK, lw=0.8, alpha=0.85, zorder=6)
    ax_a.scatter([p["dc"] for p in v34], [p["ds"] for p in v34], marker="o", s=54, c=WIN_RED, edgecolors=WIN_DARK, lw=1.1, zorder=9)
    ax_a.scatter([ec_m["dc"]], [ec_m["ds"]], marker="D", s=42, facecolors="none", edgecolors=TEAL, lw=1.5, zorder=6)

    for b in ext:
        offs = {
            "StyleGallery": ((6, 1), "left", True),
            "StyleShot": ((5, -8), "left", False),
            "StyleID": ((5, 0), "left", False),
            "Qwen-Img-2.1T": ((-5, -3), "right", True),
            "AesPA-Net": ((-4, -10), "right", False),
            "StyTr2": ((-5, 1), "right", False),
            "Seedream 4.5": ((-5, -7), "right", True),
            "SaMAM": ((-6, 0), "right", False),
            "CUT": ((6, 4), "left", False),
            "Hist2Style": ((-4, -10), "center", False),
        }
        if b["short"] in offs:
            o, ha, ldr = offs[b["short"]]
            _ann(ax_a, b["name"] if b["short"] == "StyleGallery" else b["short"], (b["dc"], b["ds"]), o,
                 ha=ha, col="#4c1d95" if b["short"] == "StyleGallery" else "#475569", bold=(b["short"] == "StyleGallery"), leader=ldr)

    for p in v34:
        if p["knob"] in ("s075", "s085", "s100", "s125", "s150"):
            lbl_s = "s=" + p["knob"][1] + "." + p["knob"][2:]
            off_map = {"s075": (5, 6), "s085": (6, 7), "s100": (-5, -10), "s125": (5, 9), "s150": (4, 10)}
            ha_m = "right" if p["knob"] == "s100" else "left"
            _ann(ax_a, lbl_s, (p["dc"], p["ds"]), off_map[p["knob"]], ha=ha_m, col=WIN_DARK, bold=True, fs=7.5)
    _ann(ax_a, "V32 22M", (0.2805, 0.6054), (4, 8), ha="left", col=BLUE_DARK, bold=True, fs=7.1)
    _ann(ax_a, "E-Core s=0.75", (ec[-1]["dc"], ec[-1]["ds"]), (5, 7), ha="left", col=TEAL_DARK, bold=True, fs=7.2)
    ax_a.set_xlim(0.24, 0.94)
    ax_a.set_ylim(0.34, 0.63)
    ax_a.set_title("(a) Full-Spectrum Quality Control", fontweight="bold", pad=8)
    ax_a.set_xlabel("DINO-C (Content Preservation) ↑", fontweight="bold")
    ax_a.set_ylabel("DINO-S (Style Similarity) ↑", fontweight="bold")
    ax_a.grid(True, ls="--", lw=0.45, alpha=0.45)

    # Panel (b)
    ax_b.axvspan(0.65, 0.94, color="#ecfdf5", alpha=0.45, zorder=0)
    _draw_ext(ax_b, "dc", "t1", data["ext_front_b"])
    ax_b.plot([p["dc"] for p in ec], [p["t1"] for p in ec], "-", color=TEAL, lw=2.4, zorder=5)
    ax_b.plot([ec[-1]["dc"], pc32[0]["dc"]], [ec[-1]["t1"], pc32[0]["t1"]], ":", color="#64748b", lw=1.4, zorder=4)
    ax_b.plot([p["dc"] for p in pc32], [p["t1"] for p in pc32], "--", color=BLUE_V32, lw=1.9, alpha=0.85, zorder=5)
    ax_b.plot([p["dc"] for p in v34], [p["t1"] for p in v34], "-", color=WIN_RED, lw=3.0, zorder=8)
    ax_b.scatter([p["dc"] for p in ec], [p["t1"] for p in ec], marker="D", s=40, c=TEAL, edgecolors=TEAL_DARK, lw=0.9, zorder=6)
    ax_b.scatter([p["dc"] for p in pc32], [p["t1"] for p in pc32], marker="o", s=34, c=BLUE_V32, edgecolors=BLUE_DARK, lw=0.8, alpha=0.85, zorder=6)
    ax_b.scatter([p["dc"] for p in v34], [p["t1"] for p in v34], marker="o", s=54, c=WIN_RED, edgecolors=WIN_DARK, lw=1.1, zorder=9)
    ax_b.scatter([ec_m["dc"]], [ec_m["t1"]], marker="D", s=42, facecolors="none", edgecolors=TEAL, lw=1.5, zorder=6)

    for b in ext:
        offs_b = {
            "StyleGallery": ((-6, -4), "right", True),
            "StyleShot": ((6, 3), "left", False),
            "StyleID": ((-6, -3), "right", False),
            "AesPA-Net": ((-5, 0), "right", False),
            "StyTr2": ((-6, -5), "right", True),
            "Seedream 4.5": ((5, -4), "left", True),
            "Hist2Style": ((4, 6), "left", False),
        }
        if b["short"] in offs_b:
            o, ha, ldr = offs_b[b["short"]]
            _ann(ax_b, b["name"] if b["short"] == "StyleGallery" else b["short"], (b["dc"], b["t1"]), o,
                 ha=ha, col="#4c1d95" if b["short"] == "StyleGallery" else "#475569", bold=(b["short"] == "StyleGallery"), leader=ldr)
    for p in v34:
        if p["knob"] in ("s075", "s085", "s100", "s125", "s150"):
            lbl_s = "s=" + p["knob"][1] + "." + p["knob"][2:]
            off_b = {"s075": (5, -7), "s085": (5, 7), "s100": (6, 7), "s125": (6, 6), "s150": (-5, -10)}
            ha_b = "right" if p["knob"] == "s150" else "left"
            _ann(ax_b, lbl_s, (p["dc"], p["t1"]), off_b[p["knob"]], ha=ha_b, col=WIN_DARK, bold=True, fs=7.5)
    ax_b.set_xlim(0.24, 0.94)
    ax_b.set_ylim(-1.0, 99.0)
    ax_b.set_title("(b) Stronger Style Specificity", fontweight="bold", pad=8)
    ax_b.set_xlabel("DINO-C (Content Preservation) ↑", fontweight="bold")
    ax_b.set_ylabel("CSD Top-1 Accuracy (%) ↑", fontweight="bold")
    ax_b.grid(True, ls="--", lw=0.45, alpha=0.45)

    # Panel (c): Speed vs CSD Top-1
    ax_c.axvspan(10.0, 180.0, color="#fff1f2", alpha=0.48, zorder=0)
    ax_c.axvline(10.0, color=WIN_RED, ls=":", lw=1.0, alpha=0.45, zorder=1)
    _draw_ext(ax_c, "ips", "t1", data["ext_front_c"])
    ax_c.plot([p["ips"] for p in ec], [p["t1"] for p in ec], "-", color=TEAL, lw=2.4, zorder=5)
    ax_c.plot([ec[-1]["ips"], pc32[0]["ips"]], [ec[-1]["t1"], pc32[0]["t1"]], ":", color="#64748b", lw=1.3, zorder=4)
    ax_c.plot([p["ips"] for p in pc32], [p["t1"] for p in pc32], "--", color=BLUE_V32, lw=1.9, alpha=0.85, zorder=5)
    ax_c.plot([p["ips"] for p in v34], [p["t1"] for p in v34], "-", color=WIN_RED, lw=3.0, zorder=8)
    ax_c.scatter([p["ips"] for p in ec], [p["t1"] for p in ec], marker="D", s=40, c=TEAL, edgecolors=TEAL_DARK, lw=0.9, zorder=6)
    ax_c.scatter([p["ips"] for p in pc32], [p["t1"] for p in pc32], marker="o", s=34, c=BLUE_V32, edgecolors=BLUE_DARK, lw=0.8, alpha=0.85, zorder=6)
    ax_c.scatter([p["ips"] for p in v34], [p["t1"] for p in v34], marker="o", s=54, c=WIN_RED, edgecolors=WIN_DARK, lw=1.1, zorder=9)
    ax_c.scatter([ec_m["ips"]], [ec_m["t1"]], marker="D", s=42, facecolors="none", edgecolors=TEAL, lw=1.5, zorder=6)

    for b in ext:
        offs_c = {
            "StyleGallery": ((5, 1), "left"),
            "StyleShot": ((5, 0), "left"),
            "StyleID": ((5, 0), "left"),
            "StyTr2": ((5, 1), "left"),
            "Seedream 4.5": ((5, -4), "left"),
            "AesPA-Net": ((5, -2), "left"),
            "Hist2Style": ((-5, 6), "right"),
        }
        if b["short"] in offs_c:
            o, ha = offs_c[b["short"]]
            _ann(ax_c, b["short"], (b["ips"], b["t1"]), o, ha=ha,
                 col="#4c1d95" if b["short"] == "StyleGallery" else "#475569", bold=(b["short"] == "StyleGallery"), fs=7.0)
    _ann(ax_c, f"V34 s=1.50 (92.0%)", (v34[0]["ips"], v34[0]["t1"]), (-6, 5), ha="right", col=WIN_DARK, bold=True, fs=7.1)
    _ann(ax_c, f"V34 s=1.00 (68.3%)", (v34[2]["ips"], v34[2]["t1"]), (-6, -2), ha="right", col=WIN_DARK, bold=True, fs=7.1)
    _ann(ax_c, f"E-Core 1.00\n67.2%", (ec_m["ips"], ec_m["t1"]), (6, 2), ha="left", col=TEAL_DARK, bold=True, fs=7.0)
    ax_c.text(0.45, 0.85, f"V34 44M: {data['v34_ips']:.1f} img/s\nV32 22M: 18.2-33.1 img/s",
              transform=ax_c.transAxes, ha="center", va="top", fontsize=6.8, fontweight="bold", color=WIN_DARK,
              bbox=dict(boxstyle="round,pad=0.25", fc="#fff1f2", ec=WIN_RED, lw=0.8, alpha=0.92))

    ax_c.set_xscale("log")
    ax_c.set_xlim(0.02, 180.0)
    ax_c.set_ylim(-1.0, 99.0)
    ax_c.set_title("(c) Real-Time Performance at Scale", fontweight="bold", pad=8)
    ax_c.set_xlabel("Cold-Start RGB->RGB Throughput (img/s, log) ↑", fontweight="bold")
    ax_c.set_ylabel("CSD Top-1 Accuracy (%) ↑", fontweight="bold")
    ax_c.grid(True, ls="--", lw=0.45, alpha=0.45)

    handles = [
        Line2D([0], [0], color=WIN_RED, marker="o", ms=6.2, lw=2.8, mfc=WIN_RED, mec=WIN_DARK,
               label=f"CLEAVE V34 Winner: 44M C2 + A3 Start ({data['v34_ips']:.1f} img/s @ b16)"),
        Line2D([0], [0], color=BLUE_V32, marker="o", ms=5.2, lw=1.9, ls="--", mfc=BLUE_V32, mec=BLUE_DARK,
               label="CLEAVE V32 Baseline: 22M P-Core (18.2-18.5 img/s)"),
        Line2D([0], [0], color=TEAL, marker="D", ms=5.6, lw=2.4, mfc=TEAL, mec=TEAL_DARK,
               label="CLEAVE E-Core: 1-NFE (33.1 img/s)"),
        Line2D([0], [0], color="none", marker="*", ms=8.5, mfc=HEAVY_PURPLE, mec="#3b0764", label="StyleGallery (30.4s)"),
        Line2D([0], [0], color="none", marker="^", ms=5.8, mfc="#8b5cf6", mec="#4c1d95", label="Heavy Baselines (>1B, 5-25s)"),
        Line2D([0], [0], color="none", marker="s", ms=5.2, mfc=LIGHT_BLUE, mec="#075985", label="Lightweight Baselines (0.2-3.0s)"),
        Line2D([0], [0], color=SLATE_FRONT, ls="--", lw=1.35, label="External Pareto Frontier"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.005), ncol=4,
               frameon=True, facecolor="#f8fafc", edgecolor="#cbd5e1", fontsize=7.6)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.77, bottom=0.14, wspace=0.25)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for stem in ("cleave_pareto_triptych", "cleave_pareto_latest"):
        fig.savefig(FIG_DIR / f"{stem}.svg")
        fig.savefig(FIG_DIR / f"{stem}.png", dpi=240)
    plt.close(fig)
    print(f"Saved 1x3 Triptych SVG+PNG to {FIG_DIR}")


def build_interactive_html(data: dict[str, Any]) -> str:
    """Self-contained 1x3 Paper Triptych + Ablation/Holdout interactive HTML for html_render."""
    payload_json = json.dumps(data, ensure_ascii=False)
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  font-family: var(--font-sans, system-ui, sans-serif);
  color: var(--foreground, #e2e8f0);
  padding: 0;
  margin: 0;
}}
.wrap {{ width: 100%; display: flex; flex-direction: column; gap: 8px; }}
.toolbar {{ display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 6px; }}
.tabs {{ display: flex; gap: 6px; flex-wrap: wrap; }}
.btn {{
  border: 1px solid var(--border, #334155);
  background: var(--secondary, #1e293b);
  color: var(--foreground, #e2e8f0);
  border-radius: var(--radius, 6px);
  padding: 4px 9px;
  font-size: 11px;
  cursor: pointer;
}}
.btn.active {{
  background: var(--primary, #3b82f6);
  color: var(--primary-foreground, #ffffff);
  border-color: var(--primary, #3b82f6);
  font-weight: 600;
}}
.card {{
  background: var(--card, #111827);
  border: 1px solid var(--border, #334155);
  border-radius: var(--radius, 8px);
  padding: 10px 10px 8px 10px;
}}
.legend {{
  display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 6px; font-size: 10.5px; color: var(--muted-foreground, #cbd5e1);
}}
.lg-item {{ display: inline-flex; align-items: center; gap: 4px; }}
.swatch {{ width: 14px; height: 3px; display: inline-block; border-radius: 2px; }}
.info-bar {{
  font-family: var(--font-mono, monospace);
  font-size: 10.5px;
  color: var(--muted-foreground, #94a3b8);
  padding: 5px 8px;
  background: var(--muted, #0f172a);
  border-radius: var(--radius, 6px);
  margin-top: 6px;
  min-height: 24px;
}}
svg {{ width: 100%; height: 300px; display: block; }}
.grid {{ stroke: var(--border, #334155); stroke-opacity: 0.45; stroke-width: 0.8; }}
.ax-lbl {{ fill: var(--muted-foreground, #94a3b8); font-size: 9px; }}
.ax-title {{ fill: var(--foreground, #e2e8f0); font-size: 9.5px; font-weight: 700; }}
.pt-lbl {{ fill: var(--muted-foreground, #cbd5e1); font-size: 8px; paint-order: stroke; stroke: var(--card, #111827); stroke-width: 2.5px; stroke-linejoin: round; }}
</style>
</head>
<body>
<div class="wrap">
  <div class="toolbar">
    <div class="tabs">
      <button class="btn active" data-view="triptych">论文三联图 (V34 vs V32 vs 外部基线)</button>
      <button class="btn" data-view="ablations">V34 各阶段消融 (Stage A / B / C / D)</button>
      <button class="btn" data-view="holdout">独立盲测 Holdout-750</button>
    </div>
  </div>
  <div class="card">
    <div class="legend" id="legend"></div>
    <svg id="chart" viewBox="0 0 960 300" preserveAspectRatio="xMidYMid meet"></svg>
    <div class="info-bar" id="info">悬停任意散点查看精确 DINO-C / DINO-S / CSD Top-1 / Cold Speed 数值。</div>
  </div>
</div>
<script>
const DATA = {payload_json};
const svg = document.getElementById("chart");
const info = document.getElementById("info");
const legendEl = document.getElementById("legend");
let curView = "triptych";

function el(tag, attrs, txt) {{
  const n = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs || {{}})) n.setAttribute(k, v);
  if (txt !== undefined) n.textContent = txt;
  return n;
}}

function drawPanel(ox, oy, pw, ph, spec) {{
  const {{x0, x1, y0, y1, xTicks, yTicks, xFmt, yFmt, logX, title, xLabel, yLabel, shade}} = spec;
  const sx = v => ox + ((logX ? (Math.log10(v) - Math.log10(x0)) / (Math.log10(x1) - Math.log10(x0)) : (v - x0) / (x1 - x0))) * pw;
  const sy = v => oy + ph - ((v - y0) / (y1 - y0)) * ph;

  if (shade) {{
    const xA = Math.max(ox, sx(shade[0])), xB = Math.min(ox + pw, sx(shade[1]));
    svg.appendChild(el("rect", {{x: xA, y: oy, width: Math.max(0, xB - xA), height: ph, fill: shade[2], opacity: 0.08}}));
  }}
  svg.appendChild(el("rect", {{x: ox, y: oy, width: pw, height: ph, fill: "none", stroke: "var(--border, #334155)"}}));
  svg.appendChild(el("text", {{x: ox + pw / 2, y: oy - 6, "text-anchor": "middle", class: "ax-title"}}, title));

  for (const t of xTicks) {{
    const px = sx(t);
    svg.appendChild(el("line", {{x1: px, y1: oy, x2: px, y2: oy + ph, class: "grid"}}));
    svg.appendChild(el("text", {{x: px, y: oy + ph + 12, "text-anchor": "middle", class: "ax-lbl"}}, xFmt ? xFmt(t) : t.toFixed(1)));
  }}
  for (const t of yTicks) {{
    const py = sy(t);
    svg.appendChild(el("line", {{x1: ox, y1: py, x2: ox + pw, y2: py, class: "grid"}}));
    svg.appendChild(el("text", {{x: ox - 4, y: py + 3, "text-anchor": "end", class: "ax-lbl"}}, yFmt ? yFmt(t) : t.toFixed(2)));
  }}
  svg.appendChild(el("text", {{x: ox + pw / 2, y: oy + ph + 26, "text-anchor": "middle", class: "ax-lbl"}}, xLabel));
  svg.appendChild(el("text", {{x: ox - 30, y: oy + ph / 2, "text-anchor": "middle", class: "ax-lbl", transform: `rotate(-90 ${{ox - 30}} ${{oy + ph / 2}})`}}, yLabel));
  return {{sx, sy, ox, oy, pw, ph}};
}}

function renderTriptych() {{
  legendEl.innerHTML = `
    <span class="lg-item"><span class="swatch" style="background:#f43f5e"></span><b>CLEAVE V34 (44M C2+A3, 14.5 img/s)</b></span>
    <span class="lg-item"><span class="swatch" style="background:#3b82f6"></span>CLEAVE V32 (22M, 18.2-18.5 img/s)</span>
    <span class="lg-item"><span class="swatch" style="background:#14b8a6"></span>E-Core 1-NFE (33.1 img/s)</span>
    <span class="lg-item"><span class="swatch" style="background:#a78bfa"></span>StyleGallery (30.4s) / Heavy (&gt;1B)</span>
    <span class="lg-item"><span class="swatch" style="background:#94a3b8"></span>External Pareto Frontier</span>
  `;
  const panels = [
    {{ox: 44,  oy: 22, pw: 264, ph: 242, x0: 0.24, x1: 0.94, y0: 0.34, y1: 0.63, xTicks: [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], yTicks: [0.35, 0.40, 0.45, 0.50, 0.55, 0.60], title: "(a) Full-Spectrum Quality Control", xLabel: "DINO-C ↑", yLabel: "DINO-S ↑", shade: [0.65, 0.94, "#10b981"], xKey: "dc", yKey: "ds", extFront: DATA.ext_front_a}},
    {{ox: 368, oy: 22, pw: 264, ph: 242, x0: 0.24, x1: 0.94, y0: 0.0,  y1: 99.0, xTicks: [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], yTicks: [0, 20, 40, 60, 80], yFmt: v => v + "%", title: "(b) Stronger Style Specificity", xLabel: "DINO-C ↑", yLabel: "CSD Top-1 (%) ↑", shade: [0.65, 0.94, "#10b981"], xKey: "dc", yKey: "t1", extFront: DATA.ext_front_b}},
    {{ox: 690, oy: 22, pw: 256, ph: 242, x0: 0.02, x1: 180.0, y0: 0.0, y1: 99.0, logX: true, xTicks: [0.1, 1, 10, 100], xFmt: v => v < 1 ? v.toFixed(1) : v.toFixed(0), yTicks: [0, 20, 40, 60, 80], yFmt: v => v + "%", title: "(c) Real-Time Performance at Scale", xLabel: "Cold RGB->RGB Throughput (img/s, log) ↑", yLabel: "CSD Top-1 (%) ↑", shade: [10.0, 180.0, "#f43f5e"], xKey: "ips", yKey: "t1", extFront: DATA.ext_front_c}},
  ];

  for (const sp of panels) {{
    const p = drawPanel(sp.ox, sp.oy, sp.pw, sp.ph, sp);
    const {{sx, sy}} = p;
    const {{xKey, yKey}} = sp;

    // External frontier dashed
    svg.appendChild(el("path", {{d: sp.extFront.map((b, i) => `${{i ? "L" : "M"}}${{sx(b[xKey]).toFixed(1)}},${{sy(b[yKey]).toFixed(1)}}`).join(" "), fill: "none", stroke: "#94a3b8", "stroke-width": 1.3, "stroke-dasharray": "4,3"}}));
    for (const b of DATA.external) {{
      const cx = sx(b[xKey]), cy = sy(b[yKey]);
      const col = b.short === "StyleGallery" ? "#c084fc" : (b.tier === "heavy" ? "#8b5cf6" : "#38bdf8");
      const r = b.short === "StyleGallery" ? 5 : 3.2;
      const dot = el("circle", {{cx, cy, r, fill: col, stroke: "#0f172a", "stroke-width": 0.8}});
      dot.onmouseenter = () => {{ info.textContent = `[External] ${{b.name}} | C=${{b.dc.toFixed(4)}} S=${{b.ds.toFixed(4)}} Top1=${{b.t1.toFixed(2)}}% Speed=${{b.ips}} img/s`; }};
      svg.appendChild(dot);
      if (["StyleGallery", "StyleShot", "StyleID", "StyTr2", "Seedream 4.5"].includes(b.short)) {{
        const isSGb = b.short === "StyleGallery" && xKey === "dc" && yKey === "t1";
        const dy = (b.short === "StyleShot" || isSGb) ? 10 : -5;
        const dx = isSGb ? -6 : 4;
        const anc = isSGb ? "end" : "start";
        svg.appendChild(el("text", {{x: cx + dx, y: cy + dy, "text-anchor": anc, class: "pt-lbl"}}, b.short));
      }}
    }}

    // E-Core + V32 P-Core
    const ec = DATA.v32_ecore, pc = DATA.v32_pcore, ecm = DATA.v32_ec_max;
    svg.appendChild(el("path", {{d: ec.map((d, i) => `${{i ? "L" : "M"}}${{sx(d[xKey]).toFixed(1)}},${{sy(d[yKey]).toFixed(1)}}`).join(" "), fill: "none", stroke: "#14b8a6", "stroke-width": 2.2}}));
    svg.appendChild(el("line", {{x1: sx(ec[ec.length-1][xKey]), y1: sy(ec[ec.length-1][yKey]), x2: sx(pc[0][xKey]), y2: sy(pc[0][yKey]), stroke: "#64748b", "stroke-width": 1.3, "stroke-dasharray": "2,2"}}));
    svg.appendChild(el("path", {{d: pc.map((d, i) => `${{i ? "L" : "M"}}${{sx(d[xKey]).toFixed(1)}},${{sy(d[yKey]).toFixed(1)}}`).join(" "), fill: "none", stroke: "#3b82f6", "stroke-width": 1.8, "stroke-dasharray": "4,2"}}));
    for (const d of ec) {{
      const cx = sx(d[xKey]), cy = sy(d[yKey]);
      const pt = el("rect", {{x: cx - 3.5, y: cy - 3.5, width: 7, height: 7, fill: "#14b8a6", transform: `rotate(45 ${{cx}} ${{cy}})`}});
      pt.onmouseenter = () => {{ info.textContent = `[V32 E-Core s=${{d.knob}}] C=${{d.dc.toFixed(4)}} S=${{d.ds.toFixed(4)}} Top1=${{d.t1.toFixed(2)}}% Speed=${{d.ips}} img/s`; }};
      svg.appendChild(pt);
    }}
    for (const d of pc) {{
      const dot = el("circle", {{cx: sx(d[xKey]), cy: sy(d[yKey]), r: 3.1, fill: "#3b82f6", opacity: 0.85}});
      dot.onmouseenter = () => {{ info.textContent = `[V32 22M P-Core s=${{d.knob}}] C=${{d.dc.toFixed(4)}} S=${{d.ds.toFixed(4)}} Top1=${{d.t1.toFixed(2)}}% Speed=${{d.ips}} img/s`; }};
      svg.appendChild(dot);
    }}
    const ecmPt = el("rect", {{x: sx(ecm[xKey]) - 3.5, y: sy(ecm[yKey]) - 3.5, width: 7, height: 7, fill: "none", stroke: "#14b8a6", "stroke-width": 1.6, transform: `rotate(45 ${{sx(ecm[xKey])}} ${{sy(ecm[yKey])}})`}});
    ecmPt.onmouseenter = () => {{ info.textContent = `[V32 E-Core standalone s=1.00] C=${{ecm.dc.toFixed(4)}} S=${{ecm.ds.toFixed(4)}} Top1=${{ecm.t1.toFixed(2)}}% Speed=${{ecm.ips}} img/s`; }};
    svg.appendChild(ecmPt);

    // V34 Winner (44M C2 + A3 Start)
    const v34 = DATA.v34_winner;
    svg.appendChild(el("path", {{d: v34.map((d, i) => `${{i ? "L" : "M"}}${{sx(d[xKey]).toFixed(1)}},${{sy(d[yKey]).toFixed(1)}}`).join(" "), fill: "none", stroke: "#f43f5e", "stroke-width": 2.9}}));
    for (const d of v34) {{
      const cx = sx(d[xKey]), cy = sy(d[yKey]);
      const dot = el("circle", {{cx, cy, r: 4.3, fill: "#f43f5e", stroke: "#fff", "stroke-width": 0.9}});
      dot.onmouseenter = () => {{ info.textContent = `[V34 Winner 44M C2+A3 ${{d.knob}}] C=${{d.dc.toFixed(4)}} S=${{d.ds.toFixed(4)}} Top1=${{d.t1.toFixed(2)}}% LP=${{d.lpips}} Same=${{d.same}}% Speed=${{d.ips}} img/s`; }};
      svg.appendChild(dot);
      if (xKey === "dc" && ["s085", "s100", "s150"].includes(d.knob)) {{
        const lbl = "s=" + d.knob[1] + "." + d.knob.slice(2);
        const isA100 = (yKey === "ds" && d.knob === "s100");
        svg.appendChild(el("text", {{x: cx + (isA100 ? -6 : 5), y: cy + (isA100 ? 11 : -6), "text-anchor": isA100 ? "end" : "start", class: "pt-lbl", style: "fill:#fda4af;font-weight:700"}}, lbl));
      }}
    }}
  }}
}}

function renderAblations() {{
  legendEl.innerHTML = `
    <span class="lg-item"><span class="swatch" style="background:#f43f5e"></span>V34 Winner (44M C2+A3)</span>
    <span class="lg-item"><span class="swatch" style="background:#eab308"></span>V34 D_merged (44M+A3 Bridge Retrain)</span>
    <span class="lg-item"><span class="swatch" style="background:#10b981"></span>V34 C2 (44M+Q32)</span>
    <span class="lg-item"><span class="swatch" style="background:#a855f7"></span>V33/V34 S2 (32x32 Joint)</span>
    <span class="lg-item"><span class="swatch" style="background:#f97316"></span>V34 B2 (Inst 1101-D+Tex)</span>
    <span class="lg-item"><span class="swatch" style="background:#ec4899"></span>V34 Stage A (A3 Zero-Train)</span>
  `;
  const spL = drawPanel(48, 22, 415, 242, {{x0: 0.28, x1: 0.78, y0: 0.44, y1: 0.61, xTicks: [0.3, 0.4, 0.5, 0.6, 0.7], yTicks: [0.45, 0.50, 0.55, 0.60], title: "(a) Stage A-D Curves: DINO-C vs DINO-S", xLabel: "DINO-C ↑", yLabel: "DINO-S ↑"}});
  const spR = drawPanel(520, 22, 415, 242, {{x0: 0.28, x1: 0.78, y0: 30, y1: 96, xTicks: [0.3, 0.4, 0.5, 0.6, 0.7], yTicks: [40, 60, 80], yFmt: v => v + "%", title: "(b) Stage A-D Curves: DINO-C vs CSD Top-1 (%)", xLabel: "DINO-C ↑", yLabel: "CSD Top-1 (%) ↑"}});
  const curves = [
    ["V34 B2 (Inst 1101-D+Tex)", DATA.stage_b.B2_inst_tex1101 || [], "#f97316", ""],
    ["V33 P2_S2 (Frozen 22M+S2)", DATA.v33_s2 || [], "#a855f7", "4,3"],
    ["V34 C2 (44M + Q32)", DATA.v34_c2 || [], "#10b981", ""],
    ["V34 D_merged (44M+A3 Bridge)", DATA.v34_d_merged || [], "#eab308", "4,3"],
    ["V34 Winner (44M C2 + A3 Start)", DATA.v34_winner || [], "#f43f5e", ""],
  ];
  for (const [name, pts, col, dash] of curves) {{
    if (!pts.length) continue;
    for (const [sp, yK] of [[spL, "ds"], [spR, "t1"]]) {{
      const p = pts.filter(d => d.dc >= 0.28 && d.dc <= 0.78);
      svg.appendChild(el("path", {{d: p.map((d, i) => `${{i ? "L" : "M"}}${{sp.sx(d.dc).toFixed(1)}},${{sp.sy(d[yK]).toFixed(1)}}`).join(" "), fill: "none", stroke: col, "stroke-width": name.includes("Winner") ? 2.8 : 1.9, ...(dash ? {{"stroke-dasharray": dash}} : {{}})}}));
      for (const d of p) {{
        const c = el("circle", {{cx: sp.sx(d.dc), cy: sp.sy(d[yK]), r: 3.6, fill: col}});
        c.onmouseenter = () => {{ info.textContent = `${{name}} [${{d.knob}}] | C=${{d.dc.toFixed(4)}} S=${{d.ds.toFixed(4)}} Top1=${{d.t1.toFixed(2)}}%`; }};
        svg.appendChild(c);
      }}
    }}
  }}
  for (const a of DATA.stage_a) {{
    if (a.mode === "A3" || a.mode === "A0") {{
      const col = a.mode === "A3" ? "#ec4899" : "#94a3b8";
      for (const [sp, yK] of [[spL, "ds"], [spR, "t1"]]) {{
        const c = el("rect", {{x: sp.sx(a.dc) - 3.5, y: sp.sy(a[yK]) - 3.5, width: 7, height: 7, fill: col, transform: `rotate(45 ${{sp.sx(a.dc)}} ${{sp.sy(a[yK])}})`}});
        c.onmouseenter = () => {{ info.textContent = `Stage A (${{a.label}}) | C=${{a.dc.toFixed(4)}} S=${{a.ds.toFixed(4)}} Top1=${{a.t1.toFixed(2)}}% Min_S=${{a.min_s}}`; }};
        svg.appendChild(c);
      }}
    }}
  }}
}}

function renderHoldout() {{
  legendEl.innerHTML = `
    <span class="lg-item"><span class="swatch" style="background:#f43f5e"></span>V34 Winner (44M C2+A3)</span>
    <span class="lg-item"><span class="swatch" style="background:#3b82f6"></span>V32 Baseline (22M)</span>
    <span class="lg-item"><span class="swatch" style="background:#eab308"></span>V34 D_merged (Bridge Retrain)</span>
    <span class="lg-item">150 Disjoint Unseen Holdout Sources x 5 Targets vs 150 Unseen References</span>
  `;
  const spL = drawPanel(52, 22, 410, 242, {{x0: 0.53, x1: 0.73, y0: 0.50, y1: 0.58, xTicks: [0.55, 0.60, 0.65, 0.70], yTicks: [0.51, 0.53, 0.55, 0.57], title: "(a) Holdout-750 Blind Split: DINO-C vs DINO-S", xLabel: "Holdout DINO-C ↑", yLabel: "Holdout DINO-S ↑"}});
  const spR = drawPanel(520, 22, 415, 242, {{x0: 0.53, x1: 0.73, y0: 38, y1: 72, xTicks: [0.55, 0.60, 0.65, 0.70], yTicks: [40, 50, 60, 70], yFmt: v => v + "%", title: "(b) Holdout-750 Blind Split: DINO-C vs CSD Top-1 (%)", xLabel: "Holdout DINO-C ↑", yLabel: "Holdout CSD Top-1 (%) ↑"}});
  const cols = {{V32_s085: "#3b82f6", V32_s100: "#3b82f6", D_merged_s085: "#eab308", D_merged_s100: "#eab308", C2_A3_s085: "#f43f5e", C2_A3_s100: "#f43f5e"}};
  for (const [a, b, col] of [["V32_s100", "V32_s085", "#3b82f6"], ["D_merged_s100", "D_merged_s085", "#eab308"], ["C2_A3_s100", "C2_A3_s085", "#f43f5e"]]) {{
    const p0 = DATA.holdout.find(x => x.label === a), p1 = DATA.holdout.find(x => x.label === b);
    if (p0 && p1) {{
      for (const [sp, yK] of [[spL, "ds"], [spR, "t1"]]) {{
        svg.appendChild(el("line", {{x1: sp.sx(p0.dc), y1: sp.sy(p0[yK]), x2: sp.sx(p1.dc), y2: sp.sy(p1[yK]), stroke: col, "stroke-width": 2.4}}));
      }}
    }}
  }}
  for (const p of DATA.holdout) {{
    const col = cols[p.label] || "#f43f5e";
    for (const [sp, yK] of [[spL, "ds"], [spR, "t1"]]) {{
      const cx = sp.sx(p.dc), cy = sp.sy(p[yK]);
      const c = el("circle", {{cx, cy, r: 5.5, fill: col, stroke: "#fff", "stroke-width": 1}});
      c.onmouseenter = () => {{ info.textContent = `[Holdout-750] ${{p.label}} | C=${{p.dc.toFixed(4)}} S=${{p.ds.toFixed(4)}} Top1=${{p.t1.toFixed(2)}}% LP=${{p.lpips}} Same=${{p.same}}%`; }};
      svg.appendChild(c);
      svg.appendChild(el("text", {{x: cx + 7, y: cy - 5, class: "pt-lbl", style: `fill:${{col}};font-weight:700`}}, p.label));
    }}
  }}
}}

function draw() {{
  svg.innerHTML = "";
  if (curView === "triptych") renderTriptych();
  else if (curView === "ablations") renderAblations();
  else renderHoldout();
}}

document.querySelectorAll("[data-view]").forEach(b => b.onclick = () => {{
  document.querySelectorAll("[data-view]").forEach(x => x.classList.remove("active"));
  b.classList.add("active");
  curView = b.dataset.view;
  draw();
}});
draw();
</script>
</body>
</html>"""


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = collect_triptych_data()
    save_matplotlib_triptych(data)
    (FIG_DIR / "cleave_pareto_latest.html").write_text(build_interactive_html(data), encoding="utf-8")
    print(f"Wrote 1x3 Triptych HTML+SVG+PNG ({len(data['v34_winner'])} V34 pts, {len(data['holdout'])} holdout pts)")


if __name__ == "__main__":
    main()
