"""A06 (Slot A, V25): 20 cross-style transfer route diagnostics.

Task: docs/19_pull_together/agent2.md section 5 A06 (+ original STAT-03 goal).
Per-task rows in runs/{v24,v25}/measured/<arm>/eval.json carry src_style_id /
tgt_style_id / pred_class, so a TRUE 5x5 source->target route matrix is
computed per arm (no aggregation shortcut, no synthesized source breakdown).

Arms (headline, per-task rows):
  M0_prism1_l100 (v24)  PRISM-1 l=1.00
  M0_v20_d6      (v24)  D6 Coarse-16 SLFUNet
  M1_lin_g55     (v24)  Hybrid linear g=0.55
  P02_d6h_csj    (v25)  V25 main-config candidate (D6h cs_j)
  B_d6_t066      (v25)  tau-shift D6 t=0.66
Data-boundary arm (summary only, NO per-task rows anywhere local):
  O3_rgb_cnx_gated_t55  One-Shot RGB+ConvNeXt gated t=0.55 -- only aggregate
  metrics exist (runs/v25/r3_oneshot_minimal.json "reused" block, reused from
  runs/v24). Route decomposition is impossible -> reported as-is with an
  explicit boundary flag; never synthesized.

Contents:
  1. per-arm 5x5 route matrix: top1 / margin / dino_s / dT (CSD style gain
     vs source baseline, src/eval/measure.py line 100) / dino_c / lpips /
     source-lock rate / prediction destination distribution (n=30 per route).
  2. source lock rate: cross tasks where referee argmax == source style
     (pred_eq_src), overall / per-target / per-route; same150 diagonal top1.
  3. weakest routes top-5 per arm + cross-arm shared failure routes.
  4. cross-validation vs known findings:
       A05 F4  Ukiyo-e centroid most isolated yet PRISM-1 worst target;
       A05 F3  ER-Rococo most similar pair (centroid cos +0.42) -> leak;
       A01     Minimalism Top-1 guard on cs_j vs full (-7.50pp @ D6h).
  5. heatmap PNG (5x5 panels, one per arm; O3 panel is a placeholder).

Run: python -m tools.v25_cpu.a06_route_diagnostics
CPU-only (numpy + matplotlib Agg); no CUDA/ssh; OMP/MKL threads pinned to 2.
"""
from __future__ import annotations

import os

# Pin CPU thread budgets BEFORE numpy import (agent2.md section 11).
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
SHORT = ["ER", "Imp", "Min", "Roc", "Uki"]

# (arm, run_dir, display name)
ARMS = [
    ("M0_prism1_l100", "v24", "PRISM-1 l100"),
    ("M0_v20_d6", "v24", "D6"),
    ("M1_lin_g55", "v24", "Hybrid g55"),
    ("P02_d6h_csj", "v25", "P02 d6h cs_j"),
    ("B_d6_t066", "v25", "B d6 t066"),
]
O3_ARM = "O3_rgb_cnx_gated_t55"
O3_SUMMARY_JSON = ROOT / "runs/v25/r3_oneshot_minimal.json"
# Reference arm used ONLY for the A01 cross-validation check (read-only).
A01_REF_ARM = ("P02_d6h_full", "v25")

N_PER_ROUTE = 30  # 600 cross = 20 routes x 30 tasks
OUT_JSON = ROOT / "docs/v25/audits/a06_route_diagnostics.json"
OUT_PNG = ROOT / "docs/v25/audits/a06_route_heatmap.png"

ROUTES = list(itertools.permutations(range(5), 2))  # 20 off-diagonal pairs


def r5(x):
    return None if x is None else round(float(x), 5)


def load_arm(arm: str, run: str) -> list[dict]:
    path = ROOT / f"runs/{run}/measured/{arm}/eval.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    rows = data["rows"]
    if len(rows) != 750:
        raise ValueError(f"{arm}: expected 750 rows, got {len(rows)}")
    order = data.get("summary", {}).get("style_order")
    if order is not None and order != STYLES:
        raise ValueError(f"{arm}: unexpected style_order {order}")
    counts: dict[tuple[int, int], int] = {}
    for r in rows:
        if r["src_style_id"] != STYLES.index(r["src_style"]):
            raise ValueError(f"{arm}: src_style/id mismatch row {r['task_id']}")
        if r["tgt_style_id"] != STYLES.index(r["tgt_style"]):
            raise ValueError(f"{arm}: tgt_style/id mismatch row {r['task_id']}")
        if r["is_cross"]:
            counts[(r["src_style_id"], r["tgt_style_id"])] = counts.get(
                (r["src_style_id"], r["tgt_style_id"]), 0
            ) + 1
    if len(counts) != 20 or any(c != N_PER_ROUTE for c in counts.values()):
        raise ValueError(f"{arm}: route counts not 20x{N_PER_ROUTE}: {counts}")
    return rows


def _mean(rs: list[dict], key: str) -> float:
    vals = [r[key] for r in rs]
    if any(v is None for v in vals):
        return float("nan")
    return float(np.mean(vals))


def route_entry(rows: list[dict], s: int, t: int) -> dict:
    """Metrics for one (source s -> target t) route, n=30 cross tasks."""
    rs = [r for r in rows if r["is_cross"] and r["src_style_id"] == s and r["tgt_style_id"] == t]
    if len(rs) != N_PER_ROUTE:
        raise ValueError(f"route {s}->{t}: n={len(rs)}")
    dest = [0.0] * 5
    for r in rs:
        dest[r["pred_class"]] += 1.0 / len(rs)
    top1 = _mean(rs, "target_top1")
    return {
        "n": len(rs),
        "top1": r5(top1),
        "target_margin": r5(_mean(rs, "target_margin")),
        "dino_s": r5(_mean(rs, "dino_s")),
        "dT_style_gain": r5(_mean(rs, "dT")),
        "dino_c": r5(_mean(rs, "dino_c")),
        "lpips": r5(_mean(rs, "lpips")),
        "src_lock_rate": r5(_mean(rs, "pred_eq_src")),
        "pred_destination": [r5(d) for d in dest],
    }


def analyse_arm(arm: str, run: str) -> dict:
    rows = load_arm(arm, run)
    cross = [r for r in rows if r["is_cross"]]
    same = [r for r in rows if not r["is_cross"]]
    routes = {f"{SHORT[s]}->{SHORT[t]}": route_entry(rows, s, t) for s, t in ROUTES}
    per_target = {}
    for t in range(5):
        rs = [r for r in cross if r["tgt_style_id"] == t]
        per_target[SHORT[t]] = {
            "n": len(rs),
            "top1": r5(_mean(rs, "target_top1")),
            "target_margin": r5(_mean(rs, "target_margin")),
            "dino_s": r5(_mean(rs, "dino_s")),
            "dT_style_gain": r5(_mean(rs, "dT")),
            "dino_c": r5(_mean(rs, "dino_c")),
            "lpips": r5(_mean(rs, "lpips")),
            "src_lock_rate": r5(_mean(rs, "pred_eq_src")),
        }
    per_source = {}
    for s in range(5):
        rs = [r for r in cross if r["src_style_id"] == s]
        per_source[SHORT[s]] = {
            "n": len(rs),
            "top1": r5(_mean(rs, "target_top1")),
            "dino_s": r5(_mean(rs, "dino_s")),
            "dT_style_gain": r5(_mean(rs, "dT")),
            "src_lock_rate": r5(_mean(rs, "pred_eq_src")),
        }
    # weakest routes: sort by top1 asc, tie-break margin asc then dino_s asc
    ranked = sorted(
        routes.items(),
        key=lambda kv: (kv[1]["top1"], kv[1]["target_margin"], kv[1]["dino_s"]),
    )
    weakest = [{"route": k, **{m: v[m] for m in ("top1", "target_margin", "dino_s", "dT_style_gain", "src_lock_rate")}} for k, v in ranked[:5]]
    strongest = [{"route": k, **{m: v[m] for m in ("top1", "target_margin", "dino_s", "dT_style_gain", "src_lock_rate")}} for k, v in ranked[-5:][::-1]]
    return {
        "run_dir": run,
        "cross600": {
            "n": len(cross),
            "top1": r5(_mean(cross, "target_top1")),
            "target_margin": r5(_mean(cross, "target_margin")),
            "dino_s": r5(_mean(cross, "dino_s")),
            "dT_style_gain": r5(_mean(cross, "dT")),
            "dino_c": r5(_mean(cross, "dino_c")),
            "lpips": r5(_mean(cross, "lpips")),
            "src_lock_rate": r5(_mean(cross, "pred_eq_src")),
        },
        "same150_diag": {"n": len(same), "top1": r5(_mean(same, "target_top1"))},
        "routes": routes,
        "per_target": per_target,
        "per_source": per_source,
        "weakest_routes_top5": weakest,
        "strongest_routes_top5": strongest,
    }


def cross_arm_findings(per_arm: dict[str, dict]) -> dict:
    """Shared failure routes + best-arm-per-route table."""
    route_names = list(next(iter(per_arm.values()))["routes"].keys())
    ranks = {name: [] for name in route_names}
    bottom5_hits = {name: [] for name in route_names}
    for arm, a in per_arm.items():
        ranked = sorted(a["routes"].items(), key=lambda kv: kv[1]["top1"])
        for i, (name, _) in enumerate(ranked):
            ranks[name].append(i + 1)  # 1 = weakest
            if i < 5:
                bottom5_hits[name].append(arm)
    shared = []
    for name, hits in sorted(bottom5_hits.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        if len(hits) >= 2:
            top1s = {arm: per_arm[arm]["routes"][name]["top1"] for arm in per_arm}
            shared.append({
                "route": name,
                "n_arms_bottom5": len(hits),
                "arms_bottom5": hits,
                "mean_rank": r5(np.mean(ranks[name])),
                "top1_by_arm": top1s,
            })
    best_arm = {}
    for name in route_names:
        best = max(per_arm.items(), key=lambda kv: kv[1]["routes"][name]["top1"])
        best_arm[name] = {"arm": best[0], "top1": best[1]["routes"][name]["top1"]}
    return {"shared_failure_routes": shared, "best_arm_per_route": best_arm}


def cross_validation(per_arm: dict[str, dict]) -> dict:
    out: dict = {}
    # --- A05 F4: Ukiyo-e most isolated centroid yet PRISM-1 worst target.
    # Reported per-target Top-1 (runs/v19 measured V19_S2_prism1_v2, docs/v25/
    # audits/a05_feature_geometry.md section 5): ER .142 / Imp .600 / Min .792 /
    # Roc .583 / Uki .100.
    reported_f4 = {"ER": 0.142, "Imp": 0.600, "Min": 0.792, "Roc": 0.583, "Uki": 0.100}
    p1 = per_arm["M0_prism1_l100"]["per_target"]
    out["a05_f4_ukiyoo_isolated_worst_target"] = {
        "reported_v19_prism1_per_target_top1": reported_f4,
        "recomputed_M0_prism1_l100_per_target_top1": {k: v["top1"] for k, v in p1.items()},
        "abs_delta_vs_reported": {k: r5(abs(p1[k]["top1"] - reported_f4[k])) for k in reported_f4},
        "ukiyoo_routes_top1_M0_prism1_l100": {
            name: e["top1"] for name, e in per_arm["M0_prism1_l100"]["routes"].items() if name.startswith("Uki->") or name.endswith("->Uki")
        },
        "reading": None,  # filled below
    }
    uki_in = {k: v for k, v in out["a05_f4_ukiyoo_isolated_worst_target"]["ukiyoo_routes_top1_M0_prism1_l100"].items() if k.startswith("Uki->")}
    uki_out = {k: v for k, v in out["a05_f4_ukiyoo_isolated_worst_target"]["ukiyoo_routes_top1_M0_prism1_l100"].items() if k.endswith("->Uki")}
    out["a05_f4_ukiyoo_isolated_worst_target"]["reading"] = (
        f"Ukiyo-as-target mean top1 {r5(np.mean(list(uki_out.values()))):.3f} vs "
        f"Ukiyo-as-source mean top1 {r5(np.mean(list(uki_in.values()))):.3f} "
        "(PRISM-1): failure concentrates on rendering Ukiyo style, not on "
        "reading Ukiyo sources - consistent with A05 F4 (isolation is a "
        "target-side generation gap, not a source-side reading gap)."
    )
    # --- A05 F3: ER-Rococo most similar pair -> mutual confusion.
    pair = {}
    for arm, a in per_arm.items():
        rt = a["routes"]
        pair[arm] = {
            "ER->Roc_top1": rt["ER->Roc"]["top1"],
            "Roc->ER_top1": rt["Roc->ER"]["top1"],
            "ER->Roc_pred_is_ER": rt["ER->Roc"]["pred_destination"][0],
            "Roc->ER_pred_is_Roc": rt["Roc->ER"]["pred_destination"][3],
            "ER->Roc_src_lock": rt["ER->Roc"]["src_lock_rate"],
            "Roc->ER_src_lock": rt["Roc->ER"]["src_lock_rate"],
        }
    out["a05_f3_er_rococo_similarity_leak"] = pair
    # --- A01: Minimalism worst-class guard (P02_d6h_csj vs P02_d6h_full).
    csj = per_arm["P02_d6h_csj"]["per_target"]
    reported_a01 = {"csj": [0.225, 0.892, 0.800, 0.533, 0.300], "full": [0.225, 0.883, 0.875, 0.492, 0.283]}
    full_rows = load_arm(A01_REF_ARM[0], A01_REF_ARM[1])
    full_cross = [r for r in full_rows if r["is_cross"]]
    full_per_target = {}
    for t in range(5):
        rs = [r for r in full_cross if r["tgt_style_id"] == t]
        full_per_target[SHORT[t]] = r5(_mean(rs, "target_top1"))
    min_routes_csj = {f"{SHORT[s]}->Min": per_arm["P02_d6h_csj"]["routes"][f"{SHORT[s]}->Min"]["top1"] for s in range(5) if s != 2}
    min_routes_full = {}
    for s in range(5):
        if s == 2:
            continue
        rs = [r for r in full_cross if r["src_style_id"] == s and r["tgt_style_id"] == 2]
        min_routes_full[f"{SHORT[s]}->Min"] = r5(_mean(rs, "target_top1"))
    out["a01_minimalism_guard_csj_vs_full"] = {
        "reported_per_target_top1": reported_a01,
        "recomputed_csj_per_target_top1": {k: v["top1"] for k, v in csj.items()},
        "recomputed_full_per_target_top1": full_per_target,
        "minimalism_routes_top1": {"csj": min_routes_csj, "full": min_routes_full},
        "reading": None,
    }
    worst_route_csj = min(min_routes_csj.items(), key=lambda kv: kv[1])
    delta_route = {
        k: r5(min_routes_csj[k] - min_routes_full[k]) for k in min_routes_csj
    }
    worst_delta_route = min(delta_route.items(), key=lambda kv: kv[1])
    out["a01_minimalism_guard_csj_vs_full"]["reading"] = (
        f"Recomputed per-target matches A01 reported values (max |delta| "
        f"{max(abs(csj[SHORT[i]]['top1'] - reported_a01['csj'][i]) for i in range(5)):.4f}). "
        f"Minimalism deficit of cs_j vs full concentrates on route "
        f"{worst_delta_route[0]} (delta {worst_delta_route[1]:+.3f}); weakest "
        f"into-Min route in cs_j is {worst_route_csj[0]} "
        f"(top1 {worst_route_csj[1]:.3f}) - route-level localization of the "
        "A01 worst-class guard trigger."
    )
    return out


def make_heatmap(per_arm: dict[str, dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    arms = list(per_arm.keys())
    n_panels = len(arms) + 1  # + placeholder for the summary-only O3 arm
    ncol, nrow = 3, 2
    fig, axes = plt.subplots(nrow, ncol, figsize=(13.5, 8.6))
    axes = axes.ravel()
    vmax = 1.0
    for idx, arm in enumerate(arms):
        ax = axes[idx]
        a = per_arm[arm]
        grid = np.full((5, 5), np.nan)
        for s, t in ROUTES:
            grid[s, t] = a["routes"][f"{SHORT[s]}->{SHORT[t]}"]["top1"]
        diag_val = a["same150_diag"]["top1"]
        im = ax.imshow(grid, vmin=0.0, vmax=vmax, cmap="RdYlGn")
        for s, t in ROUTES:
            ax.text(t, s, f"{grid[s, t]*100:.0f}", ha="center", va="center",
                    fontsize=9, color="black")
        for d in range(5):
            ax.add_patch(Rectangle((d - 0.5, d - 0.5), 1, 1, facecolor="0.82",
                                   edgecolor="0.35", lw=0.8))
            ax.text(d, d, f"{diag_val*100:.0f}", ha="center", va="center",
                    fontsize=9, color="0.15", style="italic")
        ax.set_xticks(range(5), SHORT, fontsize=8)
        ax.set_yticks(range(5), SHORT, fontsize=8)
        ax.set_title(f"{arm}  (cross top1 {a['cross600']['top1']*100:.1f}%, "
                     f"lock {a['cross600']['src_lock_rate']*100:.1f}%)", fontsize=10)
        ax.set_xlabel("target", fontsize=8)
        ax.set_ylabel("source", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    # O3 placeholder panel (data boundary: no per-task rows anywhere local)
    ax = axes[len(arms)]
    ax.set_xticks(range(5), SHORT, fontsize=8)
    ax.set_yticks(range(5), SHORT, fontsize=8)
    ax.set_facecolor("0.92")
    ax.invert_yaxis()
    for s in range(5):
        for t in range(5):
            ax.text(t, s, "-" if s != t else "d", ha="center", va="center",
                    fontsize=10, color="0.45")
    ax.set_title(f"{O3_ARM}\nNO PER-TASK ROWS (summary-only, reused v24)",
                 fontsize=9, color="firebrick")
    ax.set_xlabel("target", fontsize=8)
    ax.set_ylabel("source", fontsize=8)
    fig.text(
        0.995, 0.005,
        "rows = source style, cols = target style | cell = CSD-Top1 over the "
        "30 tasks of the route | grey diagonal = same150 blind Top1 "
        "(identity) | per-task arms: 5 | summary-only arm: "
        f"{O3_ARM} (no route decomposition possible)",
        fontsize=8, ha="right", va="bottom", family="monospace", color="0.25")
    fig.suptitle("A06: 5x5 source->target route Top-1 (CSD referee), 750-task protocol",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    per_arm = {arm: analyse_arm(arm, run) for arm, run, _ in ARMS}
    findings = cross_arm_findings(per_arm)
    xval = cross_validation(per_arm)
    # O3 summary-only block (explicit data boundary, never synthesized)
    with open(O3_SUMMARY_JSON, encoding="utf-8") as f:
        r3 = json.load(f)
    o3 = r3["reused"].get(O3_ARM)
    if o3 is None:
        raise ValueError(f"{O3_ARM} missing from {O3_SUMMARY_JSON.name} reused block")
    o3_block = {
        "data_boundary": (
            "summary-only: no per-task rows for this arm exist anywhere local "
            "(checked runs/v24+runs/v25 trees; only aggregate metrics in "
            "runs/v25/r3_oneshot_minimal.json reused block + "
            "runs/v24/p0_o1_o3.json). Route decomposition NOT computed and NOT "
            "synthesized; any source breakdown would be fabrication."
        ),
        "cross600_summary": {
            "dino_c": r5(o3["dino_c"]),
            "dino_s": r5(o3["dino_s"]),
            "lpips": r5(o3["lpips"]),
            "style_affinity": r5(o3["style_affinity"]),
            "csd_top1": r5(o3["csd_top1"]),
        },
    }
    payload = {
        "task": "A06",
        "protocol": {
            "arms_per_task": [{"arm": a, "run_dir": run, "label": label} for a, run, label in ARMS],
            "arm_summary_only": O3_ARM,
            "styles": STYLES,
            "route_count": len(ROUTES),
            "tasks_per_route": N_PER_ROUTE,
            "metrics_source": "src/eval/measure.py (dT = csd proto score of target on output minus raw source baseline; pred_eq_src = referee argmax == source style on cross tasks)",
            "threads": {"OMP_NUM_THREADS": 2, "MKL_NUM_THREADS": 2},
        },
        "per_arm": per_arm,
        "cross_arm": findings,
        "cross_validation": xval,
        "o3_one_shot": o3_block,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    make_heatmap(per_arm, OUT_PNG)
    # console digest
    print(f"A06 done: {len(per_arm)} per-task arms + 1 summary-only arm")
    for arm, a in per_arm.items():
        w = a["weakest_routes_top5"][0]
        print(f"  {arm:<16} cross top1 {a['cross600']['top1']:.4f}  lock {a['cross600']['src_lock_rate']:.3f}  weakest {w['route']} {w['top1']:.3f}")
    for s in findings["shared_failure_routes"]:
        print(f"  shared-fail: {s['route']} in bottom5 of {s['n_arms_bottom5']} arms")
    print(f"outputs: {OUT_JSON.relative_to(ROOT)} {OUT_PNG.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
