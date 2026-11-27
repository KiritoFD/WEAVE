"""Generate clean 1x3 Paper-Style Pareto Triptych (SVG/PNG + Interactive HTML) for CLEAVE.

Replicates the exact 1x3 layout & palette of ICME Fig. 1 (docs/icme2027/figures/fig1_prism_pareto):
  (a) Full-Spectrum Quality Control: DINO-C vs DINO-S
  (b) Stronger Style Specificity: DINO-C vs CSD Top-1 (%)
  (c) Real-Time Performance at Scale: Cold-Start RGB->RGB Throughput (img/s, log) vs CSD Top-1 (%)

Overlays the two clean CLEAVE branches without One-Shot clutter:
  - CLEAVE P-Core Final (V35 calibrated 44M, 14.3 img/s @ 4090, s=0.60..1.50)
  - CLEAVE E-Core Original (1-NFE, 33.1 img/s @ 4090, l=0.25..1.00)
plus their unified non-dominated Ours Pareto Frontier against prominent external baselines
under identical Cross600 / S10B / Zero-DINO conditions.

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

V35_FALLBACK_REPORTS = [
    {"arm": "O0_t50",     "group": "O0", "tau0": 0.50, "steps": 6, "dc": 0.6586, "ds": 0.4964, "t1": 14.50, "causal": 5.33,  "art": 66.67, "mv": 80.83, "inst": 69.17, "sg_aff": 0.3946, "ips": 1.35, "b8_ips": None},
    {"arm": "O1_t50",     "group": "O1", "tau0": 0.50, "steps": 6, "dc": 0.6598, "ds": 0.4954, "t1": 14.83, "causal": 6.17,  "art": 67.50, "mv": 75.00, "inst": 72.50, "sg_aff": 0.3949, "ips": 1.48, "b8_ips": None},
    {"arm": "O1_dpm_t25", "group": "O1", "tau0": 0.25, "steps": 6, "dc": 0.5332, "ds": 0.5388, "t1": 24.50, "causal": 16.33, "art": 68.33, "mv": 85.83, "inst": 75.83, "sg_aff": 0.4496, "ips": 3.52, "b8_ips": 3.31},
    {"arm": "O1_t20_w35", "group": "O1", "tau0": 0.20, "steps": 6, "dc": 0.4785, "ds": 0.5510, "t1": 29.50, "causal": 21.17, "art": 77.50, "mv": 86.67, "inst": 72.50, "sg_aff": 0.4667, "ips": 2.58, "b8_ips": 3.10},
    {"arm": "O1_t00_w35", "group": "O1", "tau0": 0.00, "steps": 6, "dc": 0.3705, "ds": 0.5658, "t1": 52.50, "causal": 40.67, "art": 82.50, "mv": 93.33, "inst": 81.67, "sg_aff": 0.4912, "ips": 1.76, "b8_ips": 2.42},
    {"arm": "O2_t30",     "group": "O2", "tau0": 0.30, "steps": 6, "dc": 0.4941, "ds": 0.5491, "t1": 34.50, "causal": 25.00, "art": 74.17, "mv": 84.17, "inst": 75.00, "sg_aff": 0.4572, "ips": 3.40, "b8_ips": 2.56},
    {"arm": "O3_t00",     "group": "O3", "tau0": 0.00, "steps": 6, "dc": 0.3221, "ds": 0.5470, "t1": 48.50, "causal": 36.00, "art": 89.17, "mv": 91.67, "inst": 79.17, "sg_aff": 0.4943, "ips": 3.30, "b8_ips": 2.39},
    {"arm": "O3_t30",     "group": "O3", "tau0": 0.30, "steps": 6, "dc": 0.4177, "ds": 0.5583, "t1": 42.67, "causal": 30.83, "art": 85.83, "mv": 92.50, "inst": 83.33, "sg_aff": 0.4907, "ips": 3.17, "b8_ips": 2.42},
    {"arm": "O3_t35",     "group": "O3", "tau0": 0.35, "steps": 6, "dc": 0.4361, "ds": 0.5540, "t1": 40.50, "causal": 29.00, "art": 78.33, "mv": 93.33, "inst": 85.00, "sg_aff": 0.4833, "ips": 3.46, "b8_ips": 4.41},
]


def _collect_v35_reports() -> list[dict[str, Any]]:
    """Load completed V35 CLEAVE-One full reports from disk if present, else embedded snapshot."""
    v35_dir = REPO / "runs/v35_cleave_one/measured"
    loaded = []
    if v35_dir.exists():
        for fn in sorted(v35_dir.glob("*_full_report.json")):
            d = read_json(fn)
            c = d["test750_cross600"]
            sg = d.get("sg25") or {}
            arm = d["arm"]
            loaded.append({
                "arm": arm,
                "group": arm.split("_")[0],
                "tau0": float(d["tau0"]),
                "steps": int(d["steps"]),
                "dc": round(float(c["dino_c"]), 4),
                "ds": round(float(c["dino_s"]), 4),
                "t1": round(float(c["csd_top1"]) * 100.0, 2),
                "causal": round(float(d["test750_wrong_ctrl"]["causal_delta_pp"]), 2),
                "art": round(float(d["artist_ood"]["follow_2way"]) * 100.0, 2),
                "mv": round(float(d["movement_ood"]["follow_2way"]) * 100.0, 2),
                "inst": round(float(d["instance_swap"]["follow_2way"]) * 100.0, 2),
                "sg_aff": round(float(sg.get("style_affinity", 0.0)), 4),
                "ips": float(sg.get("e2e_img_per_sec", 0.0)),
                "b8_ips": sg.get("e2e_batch8_img_per_sec"),
            })
    return loaded if len(loaded) >= len(V35_FALLBACK_REPORTS) else V35_FALLBACK_REPORTS


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_curve(levels: dict[str, dict[str, Any]], *, ips: float | None = None, branch: str = "") -> list[dict[str, Any]]:
    pts = []
    for k, m in levels.items():
        if "wrong" in k:
            continue
        pts.append({
            "knob": k,
            "branch": branch,
            "dc": round(float(m["dino_c"]), 4),
            "ds": round(float(m["dino_s"]), 4),
            "t1": round(float(m["csd_top1"]) * 100.0, 2),
            "clip": round(float(m.get("clip_s", 0.0)), 4),
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
        short = name.split(" (")[0]
        ext_list.append({
            "name": name,
            "short": short,
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

    sd = _load_json(REPO / "docs/archive/historical/v34/data/stage_d_report.json")
    c2a3 = _load_json(REPO / "docs/archive/historical/v34/data/c2_a3_control_report.json")
    sc = _load_json(REPO / "docs/archive/historical/v34/data/stage_c_report.json")
    sa = _load_json(REPO / "docs/archive/historical/v34/data/stage_a_report.json")
    sb = _load_json(REPO / "docs/archive/historical/v34/data/stage_b_report.json")
    p2 = _load_json(REPO / "docs/archive/historical/v33/data/p2_report.json")

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

    p8 = _load_json(REPO / "docs/archive/historical/v35/data/p8_final_report.json")
    p9 = _load_json(REPO / "docs/archive/historical/v35/data/p9_paper_audit_report.json")
    p7 = _load_json(REPO / "docs/archive/historical/v35/data/p7_biglittle_report.json")
    p3 = _load_json(REPO / "docs/archive/historical/v35/data/p3_stiefel_poolc_report.json")
    p4 = _load_json(REPO / "docs/archive/historical/v35/data/p4_seed2_report.json")
    p5 = _load_json(REPO / "docs/archive/historical/v35/data/p5_duration_report.json")

    v35_final_ips = float(p8["perf_4090"]["speed"]["b16_ips"]) if p8 else 14.3
    if p9 and "p9a_high_content_and_handoff" in p9:
        v35_final = _extract_curve(p9["p9a_high_content_and_handoff"]["pcore_full_levels"], ips=v35_final_ips, branch="pcore")
        v35_eonly = _extract_curve(p9["p9a_high_content_and_handoff"]["ecore_a0_sweep"], ips=33.1, branch="ecore")
        v35_ea3 = _extract_curve(p9["p9a_high_content_and_handoff"]["ecore_a3_sweep"], ips=32.5, branch="ecore_a3")
    else:
        v35_final = _extract_curve(p8["test750_levels"], ips=v35_final_ips, branch="pcore") if p8 else []
        v35_eonly = _extract_curve(p7["e_only"], ips=33.1, branch="ecore") if p7 else []
        v35_ea3 = []
    v35_ponly = _extract_curve(p7["p_only_src_chroma"], ips=None) if p7 else []
    p9_ponly_a3 = _extract_curve(p9["p9c_unified_ep_necessity"]["P9_Ponly_full_a3"]["levels"], ips=None) if (p9 and "p9c_unified_ep_necessity" in p9) else []
    p9_u58_self = _extract_curve(p9["p9e_iso_param_shared_backbone"]["self_ep_levels"], ips=13.1) if (p9 and "p9e_iso_param_shared_backbone" in p9) else []

    # Clean combined Ours Pareto across V35 P-Core (s=0.60..1.50) + Original E-Core (l=0.25..1.00)
    ours_pcore = [p for p in v35_final if p["knob"] != "tau0"]
    ours_ecore = [p for p in v35_eonly if p["knob"] != "l000"]
    ours_union = ours_pcore + ours_ecore
    ours_fa_idx = frontier_indices([(p["dc"], p["ds"]) for p in ours_union]) if ours_union else []
    ours_fb_idx = frontier_indices([(p["dc"], p["t1"]) for p in ours_union]) if ours_union else []
    ours_fc_idx = frontier_indices([(p["ips"], p["t1"]) for p in ours_union if p["ips"]]) if ours_union else []

    if p9 and "p9d_artist_and_work_holdout" in p9:
        ho_lv = p9["p9d_artist_and_work_holdout"]["holdout_levels"]
        for hk, hlabel in [("final_s085", "V35_Final_s085"), ("final_s100", "V35_Final_s100"), ("seed2_s085", "V35_Seed2_s085"), ("seed2_s100", "V35_Seed2_s100")]:
            if hk in ho_lv:
                c = ho_lv[hk]["cross600"]
                holdout_pts.append({
                    "label": hlabel, "dc": round(c["dino_c"], 4), "ds": round(c["dino_s"], 4),
                    "t1": round(c["csd_top1"] * 100.0, 2), "lpips": round(c["lpips"], 4),
                    "same": round(ho_lv[hk]["same150"]["csd_top1"] * 100.0, 1),
                })
    elif p8 and "holdout_s085" in p8:
        for hk, hlabel in [("final", "V35_Final_s085"), ("seed2_r0", "V35_Seed2_s085")]:
            if hk in p8["holdout_s085"]:
                c = p8["holdout_s085"][hk]["cross600"]
                holdout_pts.append({
                    "label": hlabel, "dc": round(c["dino_c"], 4), "ds": round(c["dino_s"], 4),
                    "t1": round(c["csd_top1"] * 100.0, 2), "lpips": round(c["lpips"], 4),
                    "same": round(p8["holdout_s085"][hk]["same150"]["csd_top1"] * 100.0, 1),
                })

    p3_curves = {}
    if p3 and "arms" in p3:
        for k, a in p3["arms"].items():
            p3_curves[k] = _extract_curve(a["levels"], ips=v35_final_ips)
    p4_curves = {}
    if p4 and "arms" in p4:
        for k, a in p4["arms"].items():
            p4_curves[k] = _extract_curve(a["levels"], ips=v35_final_ips)
    p5_points = []
    if p5 and "curve" in p5:
        for k, a in p5["curve"].items():
            m = a["s100"]
            p5_points.append({
                "label": k, "step": int(a["step"]), "weight": a["weight"],
                "dc": round(float(m["dino_c"]), 4), "ds": round(float(m["dino_s"]), 4),
                "t1": round(float(m["csd_top1"]) * 100.0, 2),
                "same": round(float(m["same150_top1"]) * 100.0, 1),
            })

    v35_all = _collect_v35_reports()
    v35_o3 = sorted([r for r in v35_all if r["group"] == "O3"], key=lambda p: p["dc"])
    v35_o1 = sorted([r for r in v35_all if r["group"] == "O1"], key=lambda p: p["dc"])

    return {
        "external": ext_list,
        "ext_front_a": [ext_list[i] for i in fa_idx],
        "ext_front_b": [ext_list[i] for i in fb_idx],
        "ext_front_c": [ext_list[i] for i in fc_idx],
        "ours_pcore": ours_pcore,
        "ours_ecore": ours_ecore,
        "ours_front_a": [ours_union[i] for i in ours_fa_idx],
        "ours_front_b": [ours_union[i] for i in ours_fb_idx],
        "ours_front_c": [ours_union[i] for i in ours_fc_idx],
        "v32_ecore": [p for p in v32_traj if p["seg"] == "ecore"],
        "v32_pcore": [p for p in v32_traj if p["seg"] == "pcore"],
        "v32_ec_max": v32_ec_max,
        "v34_winner": v34_winner,
        "v34_c2": v34_c2,
        "v34_d_merged": v34_d_merged,
        "v33_s2": v33_s2,
        "v34_ips": v34_ips,
        "v35_final": v35_final,
        "v35_final_ips": v35_final_ips,
        "v35_eonly": v35_eonly,
        "v35_ea3": v35_ea3,
        "v35_ponly": v35_ponly,
        "p9_ponly_a3": p9_ponly_a3,
        "p9_u58_self": p9_u58_self,
        "p3_curves": p3_curves,
        "p4_curves": p4_curves,
        "p5_points": p5_points,
        "v35_all": v35_all,
        "v35_o3": v35_o3,
        "v35_o1": v35_o1,
        "stage_a": stage_a_pts,
        "stage_b": stage_b_curves,
        "holdout": holdout_pts,
    }


def save_matplotlib_triptych(data: dict[str, Any]) -> None:
    """Save clean 1x3 paper-style Pareto Triptych SVG/PNG matching original palette & aesthetic."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    TEAL, TEAL_DARK = "#0d9488", "#115e59"
    FINAL_GRN, FINAL_DARK = "#059669", "#064e3b"
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
    pc = data["ours_pcore"]
    ec = data["ours_ecore"]
    ext = data["external"]
    stroke = [pe.withStroke(linewidth=2.4, foreground="white")]

    def _ann(ax, txt: str, xy, off, *, ha="left", col=INK, bold=False, fs=7.3, leader=False):
        arrow = dict(arrowstyle="-", color="#94a3b8", lw=0.7, shrinkA=0, shrinkB=3.5) if leader else None
        ax.annotate(
            txt, xy=xy, xytext=off, textcoords="offset points", ha=ha, va="center",
            fontsize=fs, color=col, fontweight="bold" if bold else "normal",
            arrowprops=arrow, path_effects=stroke, zorder=12,
        )

    def _draw_ext(ax, x_key: str, y_key: str, front_pts: list[dict], offs: dict[str, tuple[tuple[int, int], str, bool]]):
        ax.plot([p[x_key] for p in front_pts], [p[y_key] for p in front_pts],
                "--", color=SLATE_FRONT, lw=1.6, alpha=0.88, zorder=3)
        for b in ext:
            if b["short"] == "StyleGallery":
                ax.scatter([b[x_key]], [b[y_key]], marker="*", s=185, c=HEAVY_PURPLE, edgecolors="#3b0764", lw=0.9, zorder=8)
            elif b["tier"] == "heavy":
                ax.scatter([b[x_key]], [b[y_key]], marker="^", s=62, c="#8b5cf6", alpha=0.92, edgecolors="#3b0764", lw=0.75, zorder=6)
            else:
                ax.scatter([b[x_key]], [b[y_key]], marker="s", s=52, c=LIGHT_BLUE, alpha=0.92, edgecolors="#075985", lw=0.75, zorder=6)
            if b["short"] in offs:
                off, ha, ldr = offs[b["short"]]
                col = "#4c1d95" if (b["short"] == "StyleGallery" or b["tier"] == "heavy") else "#075985"
                _ann(ax, b["name"] if b["short"] == "StyleGallery" else b["short"], (b[x_key], b[y_key]), off,
                     ha=ha, col=col, bold=True, fs=7.2, leader=ldr)

    def _draw_ours(ax, x_key: str, y_key: str, ours_front: list[dict], *, speed_panel: bool = False):
        if ours_front:
            fx = [p[x_key] for p in ours_front]
            fy = [p[y_key] for p in ours_front]
            ax.plot(fx, fy, "-", color="#10b981", lw=5.4, alpha=0.22, solid_capstyle="round", zorder=5)
            ax.plot(fx, fy, "--", color=FINAL_DARK, lw=1.65, alpha=0.92, zorder=10)

        if not speed_panel:
            ax.plot([p[x_key] for p in ec], [p[y_key] for p in ec], "-", color=TEAL, lw=2.4, zorder=6)
            ax.plot([p[x_key] for p in pc], [p[y_key] for p in pc], "-", color=FINAL_GRN, lw=2.9, zorder=8)
        else:
            ax.plot([ec[0][x_key], ec[-1][x_key]], [min(p[y_key] for p in ec), max(p[y_key] for p in ec)],
                    "-", color=TEAL, lw=2.4, alpha=0.85, zorder=6)
            ax.plot([pc[0][x_key], pc[-1][x_key]], [min(p[y_key] for p in pc), max(p[y_key] for p in pc)],
                    "-", color=FINAL_GRN, lw=2.9, alpha=0.90, zorder=8)

        ax.scatter([p[x_key] for p in ec], [p[y_key] for p in ec],
                   marker="D", s=44, c=TEAL, edgecolors=TEAL_DARK, lw=0.95, zorder=7)
        ax.scatter([p[x_key] for p in pc], [p[y_key] for p in pc],
                   marker="o", s=54, c=FINAL_GRN, edgecolors=FINAL_DARK, lw=1.0, zorder=9)

    # Panel (a): DINO-C vs DINO-S
    ax_a.axvspan(0.65, 0.95, color="#ecfdf5", alpha=0.45, zorder=0)
    _draw_ext(ax_a, "dc", "ds", data["ext_front_a"], {
        "StyleGallery": ((6, 2), "left", True),
        "StyleShot": ((6, -6), "left", False),
        "StyleID": ((6, 1), "left", False),
        "Qwen-Img-2.1T": ((-6, -8), "right", True),
        "AesPA-Net": ((-6, 6), "right", False),
        "StyTr2": ((-6, -1), "right", False),
        "Seedream 4.5": ((-6, -8), "right", True),
        "SaMAM": ((-6, 3), "right", False),
        "CUT": ((6, 5), "left", False),
        "Hist2Style": ((-7, -5), "right", False),
    })
    _draw_ours(ax_a, "dc", "ds", data["ours_front_a"])
    for p in pc:
        if p["knob"] in ("s060", "s075", "s085", "s100", "s125", "s150"):
            lbl_s = "s=" + p["knob"][1] + "." + p["knob"][2:]
            off_m = {"s060": (-5, 8), "s075": (5, 7), "s085": (5, 8), "s100": (-5, -10), "s125": (5, 9), "s150": (4, 10)}
            ha_m = "right" if p["knob"] in ("s060", "s100") else "left"
            _ann(ax_a, lbl_s, (p["dc"], p["ds"]), off_m[p["knob"]], ha=ha_m, col=FINAL_DARK, bold=True, fs=7.3)
    for p in ec:
        if p["knob"] in ("l025", "l040", "l055", "l100"):
            lbl_e = "E " + p["knob"][1] + "." + p["knob"][2:]
            off_e = {"l025": (0, 10), "l040": (6, 8), "l055": (-7, -7), "l100": (-6, -9)}
            ha_e = "center" if p["knob"] == "l025" else ("left" if p["knob"] == "l040" else "right")
            _ann(ax_a, lbl_e, (p["dc"], p["ds"]), off_e[p["knob"]], ha=ha_e, col=TEAL_DARK, bold=True, fs=7.1)
    ax_a.set_xlim(0.28, 0.95)
    ax_a.set_ylim(0.35, 0.618)
    ax_a.set_title("(a) Full-Spectrum Quality Control", fontweight="bold", pad=8)
    ax_a.set_xlabel("DINO-C (Content Preservation) ↑", fontweight="bold")
    ax_a.set_ylabel("DINO-S (Style Similarity) ↑", fontweight="bold")
    ax_a.grid(True, ls="--", lw=0.45, alpha=0.45)

    # Panel (b): DINO-C vs CSD Top-1 (%)
    ax_b.axvspan(0.65, 0.95, color="#ecfdf5", alpha=0.45, zorder=0)
    _draw_ext(ax_b, "dc", "t1", data["ext_front_b"], {
        "StyleGallery": ((-6, -5), "right", True),
        "StyleShot": ((6, 2), "left", False),
        "StyleID": ((6, 1), "left", False),
        "Qwen-Img-2.1T": ((-6, 7), "right", True),
        "AesPA-Net": ((-6, -3), "right", False),
        "StyTr2": ((-7, -4), "right", False),
        "Seedream 4.5": ((0, -11), "center", True),
        "SaMAM": ((-6, 5), "right", False),
        "CUT": ((6, 4), "left", False),
        "Hist2Style": ((-6, 6), "right", False),
    })
    _draw_ours(ax_b, "dc", "t1", data["ours_front_b"])
    for p in pc:
        if p["knob"] in ("s060", "s075", "s085", "s100", "s125", "s150"):
            lbl_s = "s=" + p["knob"][1] + "." + p["knob"][2:]
            off_b = {"s060": (-5, 8), "s075": (5, 7), "s085": (5, 8), "s100": (6, 7), "s125": (6, 6), "s150": (0, -12)}
            ha_b = "right" if p["knob"] == "s060" else ("center" if p["knob"] == "s150" else "left")
            _ann(ax_b, lbl_s, (p["dc"], p["t1"]), off_b[p["knob"]], ha=ha_b, col=FINAL_DARK, bold=True, fs=7.3)
    for p in ec:
        if p["knob"] in ("l025", "l040", "l055", "l100"):
            lbl_e = "E " + p["knob"][1] + "." + p["knob"][2:]
            off_eb = {"l025": (-7, 4), "l040": (6, 8), "l055": (6, -5), "l100": (-6, -6)}
            ha_eb = "right" if p["knob"] in ("l025", "l100") else "left"
            _ann(ax_b, lbl_e, (p["dc"], p["t1"]), off_eb[p["knob"]], ha=ha_eb, col=TEAL_DARK, bold=True, fs=7.1)
    ax_b.set_xlim(0.28, 0.95)
    ax_b.set_ylim(-2.0, 99.0)
    ax_b.set_title("(b) Stronger Style Specificity", fontweight="bold", pad=8)
    ax_b.set_xlabel("DINO-C (Content Preservation) ↑", fontweight="bold")
    ax_b.set_ylabel("CSD Top-1 Accuracy (%) ↑", fontweight="bold")
    ax_b.grid(True, ls="--", lw=0.45, alpha=0.45)

    # Panel (c): Speed vs CSD Top-1 (%)
    ax_c.axvspan(10.0, 175.0, color="#ecfdf5", alpha=0.52, zorder=0)
    ax_c.axvline(10.0, color=FINAL_GRN, ls=":", lw=1.0, alpha=0.45, zorder=1)
    _draw_ext(ax_c, "ips", "t1", data["ext_front_c"], {
        "StyleGallery": ((6, 2), "left", False),
        "StyleShot": ((6, 2), "left", False),
        "StyleID": ((6, 1), "left", False),
        "Qwen-Img-2.1T": ((6, 5), "left", False),
        "Seedream 4.5": ((6, -5), "left", False),
        "StyTr2": ((6, 2), "left", False),
        "AesPA-Net": ((6, -2), "left", False),
        "SaMAM": ((-6, 2), "right", False),
        "CUT": ((-6, -5), "right", False),
        "Hist2Style": ((-6, 5), "right", False),
    })
    _draw_ours(ax_c, "ips", "t1", data["ours_front_c"], speed_panel=True)
    pc_150 = [x for x in pc if x["knob"] == "s150"][0]
    pc_100 = [x for x in pc if x["knob"] == "s100"][0]
    pc_085 = [x for x in pc if x["knob"] == "s085"][0]
    ec_100 = [x for x in ec if x["knob"] == "l100"][0]
    ec_065 = [x for x in ec if x["knob"] == "l065"][0]
    _ann(ax_c, f"P-Core s=1.50 ({pc_150['t1']:.1f}%)", (pc_150["ips"], pc_150["t1"]), (-7, 2), ha="right", col=FINAL_DARK, bold=True, fs=7.2)
    _ann(ax_c, f"s=1.00 ({pc_100['t1']:.1f}%)", (pc_100["ips"], pc_100["t1"]), (-7, 1), ha="right", col=FINAL_DARK, bold=True, fs=7.1)
    _ann(ax_c, f"s=0.85 ({pc_085['t1']:.1f}%)", (pc_085["ips"], pc_085["t1"]), (-7, -4), ha="right", col=FINAL_DARK, bold=True, fs=7.1)
    _ann(ax_c, f"E-Core 1.00\n({ec_100['t1']:.1f}%)", (ec_100["ips"], ec_100["t1"]), (7, 5), ha="left", col=TEAL_DARK, bold=True, fs=7.1)
    _ann(ax_c, f"E 0.65 ({ec_065['t1']:.1f}%)", (ec_065["ips"], ec_065["t1"]), (7, -1), ha="left", col=TEAL_DARK, bold=True, fs=7.0)
    ax_c.text(0.39, 0.85, f"Ours P-Core (44M): {data.get('v35_final_ips', 14.3):.1f} img/s\nOurs E-Core (1-NFE): 33.1 img/s",
              transform=ax_c.transAxes, ha="center", va="center", fontsize=6.9, fontweight="bold", color=FINAL_DARK,
              bbox=dict(boxstyle="round,pad=0.25", fc="#ecfdf5", ec=FINAL_GRN, lw=0.85, alpha=0.94))

    ax_c.set_xscale("log")
    ax_c.set_xlim(0.02, 175.0)
    ax_c.set_ylim(-2.0, 99.0)
    ax_c.set_title("(c) Real-Time Performance at Scale", fontweight="bold", pad=8)
    ax_c.set_xlabel("Cold-Start RGB->RGB Throughput (img/s, log) ↑", fontweight="bold")
    ax_c.set_ylabel("CSD Top-1 Accuracy (%) ↑", fontweight="bold")
    ax_c.grid(True, ls="--", lw=0.45, alpha=0.45)

    handles = [
        Line2D([0], [0], color=FINAL_DARK, ls="--", lw=1.7,
               label="Ours Unified Pareto Frontier (P-Core ∪ E-Core Non-Dominated)"),
        Line2D([0], [0], color=FINAL_GRN, marker="o", ms=6.0, lw=2.8, mfc=FINAL_GRN, mec=FINAL_DARK,
               label=f"Ours P-Core Final (44M, {data.get('v35_final_ips', 14.3):.1f} img/s @ 4090, s=0.60–1.50)"),
        Line2D([0], [0], color=TEAL, marker="D", ms=5.6, lw=2.4, mfc=TEAL, mec=TEAL_DARK,
               label="Ours E-Core Original (1-NFE, 33.1 img/s @ 4090, l=0.25–1.00)"),
        Line2D([0], [0], color="none", marker="*", ms=9.2, mfc=HEAVY_PURPLE, mec="#3b0764", label="StyleGallery (30.4s)"),
        Line2D([0], [0], color="none", marker="^", ms=6.2, mfc="#8b5cf6", mec="#3b0764", label="Heavy Baselines (>1B, 5–25s)"),
        Line2D([0], [0], color="none", marker="s", ms=5.6, mfc=LIGHT_BLUE, mec="#075985", label="Lightweight Baselines (0.2–3.0s)"),
        Line2D([0], [0], color=SLATE_FRONT, ls="--", lw=1.6, label="External Pareto Frontier"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=3,
               frameon=True, facecolor="#f8fafc", edgecolor="#cbd5e1", fontsize=7.2)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.73, bottom=0.14, wspace=0.25)

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
  display: flex; flex-wrap: wrap; gap: 9px; margin-bottom: 6px; font-size: 10.5px; color: var(--muted-foreground, #cbd5e1);
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
svg {{ width: 100%; height: auto; display: block; }}
.grid {{ stroke: var(--border, #334155); stroke-opacity: 0.45; stroke-width: 0.8; }}
.ax-lbl {{ fill: var(--muted-foreground, #94a3b8); font-size: 9.2px; }}
.ax-title {{ fill: var(--foreground, #e2e8f0); font-size: 9.8px; font-weight: 700; }}
.pt-lbl {{ fill: var(--foreground, #e2e8f0); font-size: 8.2px; font-weight: 600; paint-order: stroke; stroke: var(--card, #111827); stroke-width: 2.6px; stroke-linejoin: round; }}
.pt-sg {{ fill: #7c3aed; }}
.pt-ours {{ fill: #059669; }}
.pt-ec {{ fill: #0d9488; }}
</style>
</head>
<body>
<div class="wrap">
  <div class="toolbar">
    <div class="tabs">
      <button class="btn active" data-view="triptych">论文三联图 (Ours 交错前沿 vs 外部基线)</button>
      <button class="btn" data-view="v35_final">V35 P0–P9 定型消融 (Stiefel / E+P)</button>
      <button class="btn" data-view="ablations">V34 各阶段 (Stage A–D)</button>
      <button class="btn" data-view="holdout">盲测 Holdout-750</button>
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

function starPoints(cx, cy, rOut, rIn) {{
  const pts = [];
  for (let i = 0; i < 10; i++) {{
    const r = i % 2 === 0 ? rOut : rIn;
    const ang = -Math.PI / 2 + (i * Math.PI) / 5;
    pts.push(`${{(cx + r * Math.cos(ang)).toFixed(1)}},${{(cy + r * Math.sin(ang)).toFixed(1)}}`);
  }}
  return pts.join(" ");
}}

function triPoints(cx, cy, r) {{
  return `${{cx.toFixed(1)}},${{(cy - r).toFixed(1)}} ${{(cx - r * 0.9).toFixed(1)}},${{(cy + r * 0.75).toFixed(1)}} ${{(cx + r * 0.9).toFixed(1)}},${{(cy + r * 0.75).toFixed(1)}}`;
}}

function renderTriptych() {{
  svg.setAttribute("viewBox", "0 0 960 300");
  legendEl.innerHTML = `
    <span class="lg-item"><span class="swatch" style="background:#064e3b;border-top:1.5px dashed #10b981"></span><b>Ours Unified Pareto (P-Core ∪ E-Core)</b></span>
    <span class="lg-item"><span class="swatch" style="background:#059669"></span><b>Ours P-Core Final (44M, ${{DATA.v35_final_ips || 14.3}} img/s @ 4090)</b></span>
    <span class="lg-item"><span class="swatch" style="background:#0d9488"></span><b>Ours E-Core Original (1-NFE, 33.1 img/s @ 4090)</b></span>
    <span class="lg-item"><span class="swatch" style="background:#7c3aed"></span><b>StyleGallery / Heavy (&gt;1B)</b></span>
    <span class="lg-item"><span class="swatch" style="background:#0284c7"></span><b>Light Baselines (0.2–3.0s)</b></span>
    <span class="lg-item"><span class="swatch" style="background:#64748b"></span>External Pareto</span>
  `;
  const panels = [
    {{ox: 44,  oy: 22, pw: 264, ph: 242, x0: 0.28, x1: 0.95, y0: 0.35, y1: 0.618, xTicks: [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], yTicks: [0.35, 0.40, 0.45, 0.50, 0.55, 0.60], title: "(a) Full-Spectrum Quality Control", xLabel: "DINO-C ↑", yLabel: "DINO-S ↑", shade: [0.65, 0.95, "#10b981"], xKey: "dc", yKey: "ds", extFront: DATA.ext_front_a, oursFront: DATA.ours_front_a}},
    {{ox: 368, oy: 22, pw: 264, ph: 242, x0: 0.28, x1: 0.95, y0: 0.0,  y1: 99.0,  xTicks: [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], yTicks: [0, 20, 40, 60, 80], yFmt: v => v + "%", title: "(b) Stronger Style Specificity", xLabel: "DINO-C ↑", yLabel: "CSD Top-1 (%) ↑", shade: [0.65, 0.95, "#10b981"], xKey: "dc", yKey: "t1", extFront: DATA.ext_front_b, oursFront: DATA.ours_front_b}},
    {{ox: 690, oy: 22, pw: 256, ph: 242, x0: 0.02, x1: 175.0, y0: 0.0, y1: 99.0, logX: true, xTicks: [0.1, 1, 10, 100], xFmt: v => v < 1 ? v.toFixed(1) : v.toFixed(0), yTicks: [0, 20, 40, 60, 80], yFmt: v => v + "%", title: "(c) Real-Time Performance at Scale", xLabel: "Cold RGB->RGB Throughput (img/s, log) ↑", yLabel: "CSD Top-1 (%) ↑", shade: [10.0, 175.0, "#10b981"], xKey: "ips", yKey: "t1", extFront: DATA.ext_front_c, oursFront: DATA.ours_front_c, isSpeed: true}},
  ];

  const pc = DATA.ours_pcore || [];
  const ec = DATA.ours_ecore || [];
  const extOffs = {{
    "ds": {{"StyleGallery": [6, 2, "start"], "StyleShot": [6, 7, "start"], "StyleID": [6, 2, "start"], "Qwen-Img-2.1T": [-6, 10, "end"], "AesPA-Net": [-6, -2, "end"], "StyTr2": [-6, 2, "end"], "Seedream 4.5": [-6, 10, "end"], "SaMAM": [-6, -2, "end"], "CUT": [7, -4, "start"], "Hist2Style": [-7, 7, "end"]}},
    "t1": {{"StyleGallery": [-6, 7, "end"], "StyleShot": [6, 2, "start"], "StyleID": [6, 2, "start"], "Qwen-Img-2.1T": [-6, -6, "end"], "AesPA-Net": [-6, 3, "end"], "StyTr2": [-6, 5, "end"], "Seedream 4.5": [0, 11, "middle"], "SaMAM": [-6, -3, "end"], "CUT": [7, 2, "start"], "Hist2Style": [-7, -5, "end"]}},
    "spd": {{"StyleGallery": [6, 2, "start"], "StyleShot": [6, 2, "start"], "StyleID": [6, 2, "start"], "Qwen-Img-2.1T": [6, -4, "start"], "Seedream 4.5": [6, 7, "start"], "StyTr2": [6, -1, "start"], "AesPA-Net": [6, 3, "start"], "SaMAM": [-6, -1, "end"], "CUT": [-6, 5, "end"], "Hist2Style": [-6, -3, "end"]}},
  }};

  for (const sp of panels) {{
    const p = drawPanel(sp.ox, sp.oy, sp.pw, sp.ph, sp);
    const {{sx, sy}} = p;
    const {{xKey, yKey}} = sp;
    const offMap = sp.isSpeed ? extOffs.spd : extOffs[yKey];

    // External frontier dashed
    svg.appendChild(el("path", {{d: sp.extFront.map((b, i) => `${{i ? "L" : "M"}}${{sx(b[xKey]).toFixed(1)}},${{sy(b[yKey]).toFixed(1)}}`).join(" "), fill: "none", stroke: "#64748b", "stroke-width": 1.5, "stroke-dasharray": "4,3", opacity: 0.88}}));

    // Prominent External Baselines (original colors)
    for (const b of DATA.external) {{
      const cx = sx(b[xKey]), cy = sy(b[yKey]);
      let mark;
      if (b.short === "StyleGallery") {{
        mark = el("polygon", {{points: starPoints(cx, cy, 7.2, 3.2), fill: "#7c3aed", stroke: "#ffffff", "stroke-width": 0.9}});
      }} else if (b.tier === "heavy") {{
        mark = el("polygon", {{points: triPoints(cx, cy, 4.8), fill: "#8b5cf6", stroke: "#ffffff", "stroke-width": 0.75}});
      }} else {{
        mark = el("rect", {{x: cx - 3.9, y: cy - 3.9, width: 7.8, height: 7.8, fill: "#0284c7", stroke: "#ffffff", "stroke-width": 0.75}});
      }}
      mark.onmouseenter = () => {{ info.textContent = `[External · ${{b.tier}}] ${{b.name}} | C=${{b.dc.toFixed(4)}} S=${{b.ds.toFixed(4)}} Top1=${{b.t1.toFixed(2)}}% Speed=${{b.ips}} img/s`; }};
      svg.appendChild(mark);
      const showLbl = !sp.isSpeed || ["StyleGallery", "StyleShot", "StyleID", "StyTr2", "Seedream 4.5", "AesPA-Net", "Hist2Style"].includes(b.short);
      if (showLbl && offMap && offMap[b.short]) {{
        const [dx, dy, anc] = offMap[b.short];
        const cls = b.short === "StyleGallery" ? "pt-lbl pt-sg" : "pt-lbl";
        svg.appendChild(el("text", {{x: cx + dx, y: cy + dy, "text-anchor": anc, class: cls}}, b.short));
      }}
    }}

    // Combined Ours Pareto Envelope
    if (sp.oursFront && sp.oursFront.length) {{
      const fd = sp.oursFront.map((d, i) => `${{i ? "L" : "M"}}${{sx(d[xKey]).toFixed(1)}},${{sy(d[yKey]).toFixed(1)}}`).join(" ");
      svg.appendChild(el("path", {{d: fd, fill: "none", stroke: "#10b981", "stroke-width": 5.0, opacity: 0.22, "stroke-linecap": "round"}}));
      svg.appendChild(el("path", {{d: fd, fill: "none", stroke: "#064e3b", "stroke-width": 1.6, "stroke-dasharray": "4,3"}}));
    }}

    // E-Core curve + diamonds
    if (ec.length) {{
      if (!sp.isSpeed) {{
        svg.appendChild(el("path", {{d: ec.map((d, i) => `${{i ? "L" : "M"}}${{sx(d[xKey]).toFixed(1)}},${{sy(d[yKey]).toFixed(1)}}`).join(" "), fill: "none", stroke: "#0d9488", "stroke-width": 2.3}}));
      }} else {{
        svg.appendChild(el("line", {{x1: sx(ec[0][xKey]), y1: sy(Math.min(...ec.map(x => x[yKey]))), x2: sx(ec[0][xKey]), y2: sy(Math.max(...ec.map(x => x[yKey]))), stroke: "#0d9488", "stroke-width": 2.3}}));
      }}
      for (const d of ec) {{
        const cx = sx(d[xKey]), cy = sy(d[yKey]);
        const pt = el("rect", {{x: cx - 3.5, y: cy - 3.5, width: 7, height: 7, fill: "#0d9488", stroke: "#ffffff", "stroke-width": 0.8, transform: `rotate(45 ${{cx}} ${{cy}})`}});
        pt.onmouseenter = () => {{ info.textContent = `[Ours E-Core 1-NFE ${{d.knob}}] C=${{d.dc.toFixed(4)}} S=${{d.ds.toFixed(4)}} Top1=${{d.t1.toFixed(2)}}% LP=${{d.lpips}} Same=${{d.same}}% Speed=${{d.ips}} img/s`; }};
        svg.appendChild(pt);
        if (sp.isSpeed && d.knob === "l100") {{
          svg.appendChild(el("text", {{x: cx + 6, y: cy - 2, "text-anchor": "start", class: "pt-lbl pt-ec"}}, `E 1.00 (${{d.t1.toFixed(1)}}%)`));
        }}
      }}
    }}

    // P-Core curve + circles
    if (pc.length) {{
      if (!sp.isSpeed) {{
        svg.appendChild(el("path", {{d: pc.map((d, i) => `${{i ? "L" : "M"}}${{sx(d[xKey]).toFixed(1)}},${{sy(d[yKey]).toFixed(1)}}`).join(" "), fill: "none", stroke: "#059669", "stroke-width": 2.7}}));
      }} else {{
        svg.appendChild(el("line", {{x1: sx(pc[0][xKey]), y1: sy(Math.min(...pc.map(x => x[yKey]))), x2: sx(pc[0][xKey]), y2: sy(Math.max(...pc.map(x => x[yKey]))), stroke: "#059669", "stroke-width": 2.7}}));
      }}
      for (const d of pc) {{
        const cx = sx(d[xKey]), cy = sy(d[yKey]);
        const dot = el("circle", {{cx, cy, r: 4.1, fill: "#10b981", stroke: "#064e3b", "stroke-width": 0.9}});
        dot.onmouseenter = () => {{ info.textContent = `[Ours P-Core Final ${{d.knob}}] C=${{d.dc.toFixed(4)}} S=${{d.ds.toFixed(4)}} Top1=${{d.t1.toFixed(2)}}% LP=${{d.lpips}} Same=${{d.same}}% Speed=${{d.ips}} img/s (4090)`; }};
        svg.appendChild(dot);
        if (!sp.isSpeed && ["s075", "s100", "s150"].includes(d.knob)) {{
          const lbl = "s=" + d.knob[1] + "." + d.knob.slice(2);
          let dx = 5, dy = -6, anc = "start";
          if (d.knob === "s150" && yKey === "t1") {{ dx = 0; dy = 12; anc = "middle"; }}
          else if (d.knob === "s100" && yKey === "ds") {{ dx = -6; dy = 10; anc = "end"; }}
          svg.appendChild(el("text", {{x: cx + dx, y: cy + dy, "text-anchor": anc, class: "pt-lbl pt-ours"}}, lbl));
        }} else if (sp.isSpeed && ["s150", "s100"].includes(d.knob)) {{
          const lbl = d.knob === "s150" ? `P 1.50 (${{d.t1.toFixed(1)}}%)` : `s=1.00 (${{d.t1.toFixed(1)}}%)`;
          svg.appendChild(el("text", {{x: cx - 6, y: cy + 3, "text-anchor": "end", class: "pt-lbl pt-ours"}}, lbl));
        }}
      }}
    }}
  }}
}}

function renderV35Final() {{
  svg.setAttribute("viewBox", "0 0 960 300");
  legendEl.innerHTML = `
    <span class="lg-item"><span class="swatch" style="background:#10b981"></span><b>CLEAVE V35 Final (Stiefel+E+A3+P+S10B, s0.60–1.50)</b></span>
    <span class="lg-item"><span class="swatch" style="background:#f97316"></span>P9 Unified-58M (Shared Single Backbone, Self-EP)</span>
    <span class="lg-item"><span class="swatch" style="background:#a855f7"></span>Q3 Stiefel+0%PC</span>
    <span class="lg-item"><span class="swatch" style="background:#3b82f6"></span>Q1 Whiten-32</span>
    <span class="lg-item"><span class="swatch" style="background:#14b8a6"></span>E-only (1-NFE A0)</span>
    <span class="lg-item"><span class="swatch" style="background:#f43f5e"></span>P-only (Src+A3)</span>
  `;
  const spL = drawPanel(50, 22, 410, 242, {{x0: 0.32, x1: 0.94, y0: 0.38, y1: 0.61, xTicks: [0.4, 0.5, 0.6, 0.7, 0.8, 0.9], yTicks: [0.40, 0.45, 0.50, 0.55, 0.60], title: "(a) P3/P4/P7/P9 Attribution: DINO-C vs DINO-S", xLabel: "DINO-C ↑", yLabel: "DINO-S ↑"}});
  const spR = drawPanel(520, 22, 415, 242, {{x0: 0.32, x1: 0.94, y0: 0.0, y1: 96.0, xTicks: [0.4, 0.5, 0.6, 0.7, 0.8, 0.9], yTicks: [0, 20, 40, 60, 80], yFmt: v => v + "%", title: "(b) P3/P4/P7/P9 Attribution: DINO-C vs CSD Top-1 (%)", xLabel: "DINO-C ↑", yLabel: "CSD Top-1 (%) ↑"}});
  const p3 = DATA.p3_curves || {{}}, p4 = DATA.p4_curves || {{}};
  const curves = [
    ["P9 P-only (Src+A3)", DATA.p9_ponly_a3 && DATA.p9_ponly_a3.length ? DATA.p9_ponly_a3 : (DATA.v35_ponly || []), "#f43f5e", "3,2"],
    ["P9 E-only (1-NFE A0)", DATA.v35_eonly || [], "#14b8a6", "4,2"],
    ["P9 Unified-58M (Shared 58.04M Self-EP)", (DATA.p9_u58_self || []).filter(d => d.knob !== "tau0"), "#f97316", ""],
    ["P3 Q1 (Whiten-32 + 15%PC)", p3.P3_Q1_whiten_poolc || [], "#3b82f6", "4,3"],
    ["P3 Q3 (Stiefel + 0%PC)", p3.P3_Q3_stiefel_nopc || [], "#a855f7", ""],
    ["P4 R0 (Seed-2 Stiefel)", p4.P4_R0_stiefel_seed2 || [], "#f59e0b", "2,2"],
    ["V35 Final (CLEAVE E+P+Calib s0.60-1.50)", (DATA.v35_final || []).filter(d => d.knob !== "tau0"), "#10b981", ""],
  ];
  for (const [name, pts, col, dash] of curves) {{
    if (!pts.length) continue;
    for (const [sp, yK] of [[spL, "ds"], [spR, "t1"]]) {{
      const p = pts.filter(d => d.dc >= 0.32 && d.dc <= 0.94);
      svg.appendChild(el("path", {{d: p.map((d, i) => `${{i ? "L" : "M"}}${{sp.sx(d.dc).toFixed(1)}},${{sp.sy(d[yK]).toFixed(1)}}`).join(" "), fill: "none", stroke: col, "stroke-width": name.includes("Final") ? 2.9 : 1.9, ...(dash ? {{"stroke-dasharray": dash}} : {{}})}}));
      for (const d of p) {{
        const c = el("circle", {{cx: sp.sx(d.dc), cy: sp.sy(d[yK]), r: name.includes("Final") ? 4.2 : 3.2, fill: col, stroke: "#0f172a", "stroke-width": 0.7}});
        c.onmouseenter = () => {{ info.textContent = `[${{name}}] ${{d.knob}} | C=${{d.dc.toFixed(4)}} S=${{d.ds.toFixed(4)}} Top1=${{d.t1.toFixed(2)}}% CLIP-S=${{(d.clip||0).toFixed(4)}} Same=${{d.same}}%`; }};
        svg.appendChild(c);
      }}
    }}
  }}
}}

function renderAblations() {{
  svg.setAttribute("viewBox", "0 0 960 300");
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
  svg.setAttribute("viewBox", "0 0 960 300");
  legendEl.innerHTML = `
    <span class="lg-item"><span class="swatch" style="background:#10b981"></span><b>V35 Final (Seed-1) &amp; Seed-2 R0</b></span>
    <span class="lg-item"><span class="swatch" style="background:#f43f5e"></span>V34 Uncalibrated (44M C2+A3)</span>
    <span class="lg-item"><span class="swatch" style="background:#3b82f6"></span>V32 Baseline (22M)</span>
    <span class="lg-item"><span class="swatch" style="background:#eab308"></span>V34 D_merged</span>
    <span class="lg-item">150 Holdout Sources x 5 Targets (0 painting overlap)</span>
  `;
  const spL = drawPanel(52, 22, 410, 242, {{x0: 0.53, x1: 0.73, y0: 0.50, y1: 0.58, xTicks: [0.55, 0.60, 0.65, 0.70], yTicks: [0.51, 0.53, 0.55, 0.57], title: "(a) Holdout-750 Blind Split: DINO-C vs DINO-S", xLabel: "Holdout DINO-C ↑", yLabel: "Holdout DINO-S ↑"}});
  const spR = drawPanel(520, 22, 415, 242, {{x0: 0.53, x1: 0.73, y0: 38, y1: 72, xTicks: [0.55, 0.60, 0.65, 0.70], yTicks: [40, 50, 60, 70], yFmt: v => v + "%", title: "(b) Holdout-750 Blind Split: DINO-C vs CSD Top-1 (%)", xLabel: "Holdout DINO-C ↑", yLabel: "Holdout CSD Top-1 (%) ↑"}});
  const cols = {{V32_s085: "#3b82f6", V32_s100: "#3b82f6", D_merged_s085: "#eab308", D_merged_s100: "#eab308", C2_A3_s085: "#f43f5e", C2_A3_s100: "#f43f5e", V35_Final_s085: "#10b981", V35_Final_s100: "#10b981", V35_Seed2_s085: "#059669", V35_Seed2_s100: "#059669"}};
  for (const [a, b, col] of [["V32_s100", "V32_s085", "#3b82f6"], ["D_merged_s100", "D_merged_s085", "#eab308"], ["C2_A3_s100", "C2_A3_s085", "#f43f5e"], ["V35_Final_s100", "V35_Final_s085", "#10b981"], ["V35_Seed2_s100", "V35_Seed2_s085", "#059669"]]) {{
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
      const c = el("circle", {{cx, cy, r: p.label.startsWith("V35_") ? 6.0 : 5.0, fill: col, stroke: "#fff", "stroke-width": 1}});
      c.onmouseenter = () => {{ info.textContent = `[Holdout-750] ${{p.label}} | C=${{p.dc.toFixed(4)}} S=${{p.ds.toFixed(4)}} Top1=${{p.t1.toFixed(2)}}% LP=${{p.lpips}} Same=${{p.same}}%`; }};
      svg.appendChild(c);
      const dy = p.label === "V35_Seed2_s085" ? 12 : -5;
      svg.appendChild(el("text", {{x: cx + 7, y: cy + dy, class: "pt-lbl", style: `fill:${{col}};font-weight:700`}}, p.label));
    }}
  }}
}}

function draw() {{
  svg.innerHTML = "";
  if (curView === "triptych") renderTriptych();
  else if (curView === "v35_final") renderV35Final();
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
    print(f"Wrote 1x3 Triptych HTML+SVG+PNG ({len(data['ours_pcore'])} P-Core pts, {len(data['ours_ecore'])} E-Core pts, {len(data['holdout'])} holdout pts)")


if __name__ == "__main__":
    main()
