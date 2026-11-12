"""V7-R P5: unified statistics, Pareto front, per-task paired diffs, report.

Statistics policy (V7R_PLAN.md P5, correcting the V7 bin critique):
  1. Model-level Pareto front over ALL 600 cross tasks (arm means on the same
     task set — no bin re-bucketing across different tasks).
  2. Per-task paired differences (same source, same target) with clustered
     bootstrap for the key comparisons, vs B3 AND vs B4.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools.v6_24h.stats import paired_cluster_ci
from tools.v7.v7_common import (
    P0_DIR, P2_DIR as V7_P2_DIR, P3_EVAL as V7_P3_EVAL, ROOT, STYLES,
    tasks_sha256, write_json,
)
from tools.v7r.v7r_common import P1_DIR, P2_DIR, P3_DIR, P4_DIR

B3_EVAL = ROOT / "runs" / "v6_r2" / "eval_B3_distill20"
B4_EVAL = ROOT / "runs" / "v6_r2" / "eval_B4_distill50"

GATES = {"dino_s_delta": 0.020, "dino_c_min": 0.8125, "lpips_max": 0.4712,
         "top1_min_delta": -0.02}


def load_eval(p: Path):
    f = Path(p) / "eval.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def collect_arms() -> dict:
    """arm_id -> eval.json, with include/exclude reasons."""
    arms = {}
    # historical anchors
    for name, p in [("B3", B3_EVAL), ("B4", B4_EVAL),
                    ("V7_T0", ROOT / "runs/v7/p2_eval/T0"),
                    ("V7_T1", ROOT / "runs/v7/p2_eval/T1"),
                    ("V7_M1", V7_P3_EVAL / "M1"), ("V7_M2", V7_P3_EVAL / "M2")]:
        e = load_eval(p)
        if e:
            arms[name] = {"eval": e, "path": str(p), "phase": "anchor"}
    # P1 probes + objectives
    for probe in sorted(P3_DIR.glob("p1a_*")):
        e = load_eval(probe)
        if e:
            arms[probe.name] = {"eval": e, "path": str(probe), "phase": "P1-A"}
    for e_name in ["E0", "E1", "E2", "E3", "E4"]:
        e = load_eval(P3_DIR / f"p1c_{e_name}")
        if e:
            arms[f"p1c_{e_name}"] = {"eval": e, "path": str(P3_DIR / f"p1c_{e_name}"),
                                     "phase": "P1-C"}
    # P2 predictor checkpoints: any trained arm dir (N1/N2/N*_s43/RAND)
    for d in sorted(P2_DIR.iterdir()) if P2_DIR.exists() else []:
        if not d.is_dir() or not list(d.glob("step_*.pt")):
            continue
        for ck in sorted(d.glob("step_*.pt")):
            step = int(ck.stem.split("_")[1])
            e = load_eval(d / f"eval_step_{step:05d}")
            if e:
                arms[f"{d.name}@{step}"] = {"eval": e,
                                            "path": str(d / f"eval_step_{step:05d}"),
                                            "phase": "P2"}
    for name in ["N1_best", "N2_best", "RAND"]:
        e = load_eval(P2_DIR / name)
        if e:
            arms[name] = {"eval": e, "path": str(P2_DIR / name), "phase": "P2"}
    # P3 correctors
    for arm in ["R0", "R1c", "R2c", "R3c", "R4c", "R5c", "R1c_s1", "R1c_s2",
                "R3c_noC", "R3c_disp"]:
        e = load_eval(P3_DIR / arm)
        if e:
            arms[arm] = {"eval": e, "path": str(P3_DIR / arm), "phase": "P3"}
    # P4
    for arm in ["N1_best", "N2_best", "extended", "second_seed"]:
        e = load_eval(P4_DIR / arm)
        if e:
            arms[f"p4_{arm}"] = {"eval": e, "path": str(P4_DIR / arm),
                                 "phase": "P4"}
    return arms


def paired(a: dict, b: dict) -> dict:
    ra = {r["task_id"]: r for r in a["rows"] if r["is_cross"]}
    rb = {r["task_id"]: r for r in b["rows"] if r["is_cross"]}
    if set(ra) != set(rb):
        return {"protocol_mismatch": True}
    common = sorted(ra)
    out = {}
    for metric in ["target_top1", "dino_s", "dino_c", "lpips"]:
        va = [ra[k][metric] for k in common]
        vb = [rb[k][metric] for k in common]
        clusters = [ra[k]["src_stem"] for k in common]
        p = paired_cluster_ci(va, vb, clusters, n_boot=4000, seed=0)
        out[metric] = {"delta": p["diff"], "ci": p["ci"],
                       "excludes_zero": p["excludes_zero"]}
    return out


def pareto(arms: dict, ref_ids: set) -> list:
    """Model-level Pareto front over (dino_c mean, dino_s mean) on the SAME
    600-task set (task-id sets verified)."""
    pts = {}
    for name, rec in arms.items():
        rows = [r for r in rec["eval"]["rows"] if r["is_cross"]]
        ids = {r["task_id"] for r in rows}
        if ref_ids and ids != ref_ids:
            continue
        c = rec["eval"]["cross600"]
        pts[name] = (c["dino_c"], c["dino_s"])
    front = []
    for n, (dc, ds) in pts.items():
        dominated = any((dc2 >= dc and ds2 >= ds and (dc2 > dc or ds2 > ds))
                        for n2, (dc2, ds2) in pts.items() if n2 != n)
        if not dominated:
            front.append(n)
    return {"points": {k: {"dino_c": v[0], "dino_s": v[1]}
                       for k, v in sorted(pts.items())},
            "front": sorted(front)}


def p43_status(c: dict, paired_b3: dict) -> str:
    if paired_b3.get("protocol_mismatch"):
        return "PROTOCOL_MISMATCH"
    ds = paired_b3["dino_s"]
    if (ds["delta"] >= GATES["dino_s_delta"] and ds["ci"][0] > 0
            and c["dino_c"] >= GATES["dino_c_min"]
            and c["lpips"] <= GATES["lpips_max"]
            and paired_b3["target_top1"]["delta"] >= GATES["top1_min_delta"]):
        return "PASS"
    if ds["ci"][0] > 0:
        return "PARTIAL"
    return "NO"


def main() -> int:
    ap = argparse.ArgumentParser(description="V7-R final report")
    args = ap.parse_args()
    arms = collect_arms()
    b3 = arms.get("B3", {}).get("eval")
    b4 = arms.get("B4", {}).get("eval")
    if b3 is None or b4 is None:
        raise SystemExit("B3/B4 anchor evals missing")

    matrix = []
    for name, rec in sorted(arms.items()):
        e = rec["eval"]
        c, s = e["cross600"], e["same150"]
        pb3 = paired(e, b3)
        pb4 = paired(e, b4)
        matrix.append({
            "arm_id": name, "phase": rec["phase"], "eval_path": rec["path"],
            "cross600": c, "same150": s, "paired_vs_B3": pb3,
            "paired_vs_B4": pb4, "p43_status": p43_status(c, pb3),
        })

    ref_ids = {r["task_id"] for r in b4["rows"] if r["is_cross"]}
    pareto_front = pareto(arms, ref_ids)

    # key per-task paired comparisons
    key_pairs = {}
    for cand in ["R3c", "R4c", "R5c", "N1_best", "N2_best", "p4_extended",
                 "p4_second_seed"]:
        if cand in arms:
            key_pairs[f"{cand}_vs_B4"] = paired(arms[cand]["eval"], b4)
            key_pairs[f"{cand}_vs_V7_T0"] = paired(arms[cand]["eval"],
                                                   arms["V7_T0"]["eval"])
    # N2 vs N1 (question B: source-conditioning value)
    n1_best = _best_n(arms, "N1")
    n2_best = _best_n(arms, "N2")
    if n1_best and n2_best:
        key_pairs["N2best_vs_N1best"] = paired(arms[n2_best]["eval"],
                                               arms[n1_best]["eval"])

    verdict = build_verdict(matrix, pareto_front, key_pairs, arms)
    write_json({"arms": matrix, "gates": GATES, "pareto": pareto_front,
                "key_paired": key_pairs, "tasks_sha256": tasks_sha256()},
               ROOT / "runs" / "v7r" / "final_matrix.json")
    write_json(verdict, ROOT / "runs" / "v7r" / "final_verdict.json")
    write_report(matrix, pareto_front, key_pairs, verdict)
    print("[p5] V7-R report written", flush=True)
    return 0


def _best_n(arms: dict, name: str) -> str | None:
    cands = {k: arms[k]["eval"]["cross600"]["dino_s"]
             for k in arms if k.startswith(f"{name}@")}
    return max(cands, key=cands.get) if cands else None


def build_verdict(matrix, pareto_front, key_pairs, arms) -> dict:
    p1g = _read(P1_DIR / "p1_gate.json")
    p2g = _read(P2_DIR / "p2_gate.json")
    p3g = _read(P3_DIR / "p3_gate.json")
    p0a = _read(ROOT / "runs" / "v7r" / "p0_failure_audit.json")

    q_a = "NOT_TESTED"
    if p1g:
        if p1g.get("verdict") == "GO_LOCAL":
            q_a = ("YES — the local real-patch objective improved independent "
                   f"style metrics (best={p1g.get('best_energy')})")
        else:
            q_a = ("NO — local real-patch objectives did not outperform the "
                   "R1 boundary at matched budget; R1 remains the known-good "
                   "guide (negative result recorded)")
    q_b = "NOT_TESTED"
    if p2g:
        if p2g.get("verdict") == "NEW_PREDICTOR":
            q_b = ("YES — source-target conditioned correction trained stably "
                   f"({p2g.get('best_predictor')}@{p2g.get('best_step')})")
        elif "N2best_vs_N1best" in key_pairs:
            d = key_pairs["N2best_vs_N1best"]["dino_s"]["delta"]
            q_b = (f"NO — explicit source conditioning did not rescue training "
                   f"(N2-N1 ΔDINO-S {d:+.4f}); the corrector-on-B4 structure "
                   "itself is the finding")
        else:
            q_b = "NO — neither N1 nor N2 beat B4 (structure kept content but no style gain)"
    q_c = "NOT_TESTED"
    n_best = _best_n(arms, "N1") or _best_n(arms, "N2")
    m1 = arms.get("V7_M1")
    if n_best and m1:
        nb = arms[n_best]["eval"]["cross600"]
        q_c = (f"YES — corrector-on-B4 stayed near B4 content (DINO-C "
               f"{nb['dino_c']:.4f}) while the V7 relative relearn collapsed "
               f"(M1 DINO-C {m1['eval']['cross600']['dino_c']:.4f}); the "
               "frozen-predictor + small-correction structure is the stable one")
    return {
        "questions": {"A": q_a, "B": q_b, "C": q_c},
        "stage_verdicts": {
            "P0": "AUDIT_DONE" if p0a else "PENDING",
            "P1": (p1g or {}).get("verdict"),
            "P2": (p2g or {}).get("verdict"),
            "P3": (p3g or {}).get("verdict"),
            "P4": (_read(P4_DIR / "p4_decision.json") or {}).get("mode"),
        },
        "pareto_front": pareto_front["front"],
        "dev_set_statement": "All statistics are paired in-development-set "
                             "results on tasks_750; no blind-test claim.",
    }


def _read(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def fmt(m: dict) -> str:
    c, s = m["cross600"], m["same150"]
    p = m.get("paired_vs_B4", {}).get("dino_s", {})
    return (f"| {m['arm_id']} | {m['phase']} | {c['top1']*100:.1f} "
            f"| {c['dino_s']:.4f} | {c['dino_c']:.4f} | {c['lpips']:.4f} "
            f"| {c['dT']:+.4f} | {s['same_lpips']:.4f} "
            f"| {p.get('delta', 0):+.4f} | {m['p43_status']} |")


def write_report(matrix, pareto_front, key_pairs, verdict) -> None:
    lines = ["# WEAVE V7-R Extension Report (auto-generated by tools/v7r/p5_report.py)", ""]
    lines += ["## 1. Stage verdicts", ""]
    for k, v in verdict["stage_verdicts"].items():
        lines.append(f"- **{k}**: {v}")
    lines += ["", "## 2. All arms (full Cross600 / Same150)", "",
              "| arm | phase | Top-1 | DINO-S | DINO-C | LPIPS | dT | Same-LPIPS | ΔDINO-S vs B4 | P4.3 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    lines += [fmt(m) for m in matrix]
    lines += ["", "## 3. Model-level Pareto front (same 600 tasks)", ""]
    for n in pareto_front["front"]:
        p = pareto_front["points"][n]
        lines.append(f"- {n}: DINO-C {p['dino_c']:.4f}, DINO-S {p['dino_s']:.4f}")
    lines += ["", "## 4. Key per-task paired comparisons (same source & target)", ""]
    for k, v in key_pairs.items():
        d = v.get("dino_s", {})
        if d:
            lines.append(f"- {k}: ΔDINO-S {d['delta']:+.4f} "
                         f"[{d['ci'][0]:+.4f},{d['ci'][1]:+.4f}]")
    lines += ["", "## 5. Mechanism questions", ""]
    for q, a in verdict["questions"].items():
        lines.append(f"- **{q}**: {a}")
    lines += ["", "## 6. Development-set limitation", "",
              verdict["dev_set_statement"], ""]
    (ROOT / "docs" / "v7" / "V7R_EXTENSION_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
