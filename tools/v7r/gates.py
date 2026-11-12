"""V7-R gate evaluators: p1 / p2 / p3 gates, written as JSON for the runner.

Gate philosophy (V7R_PLAN.md): a failed hypothesis switches to the next
pre-registered branch — it never terminates the pipeline.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tools.v7.v7_common import ROOT, STYLES, write_json
from tools.v7r.v7r_common import P1_DIR, P2_DIR, P3_DIR, load_corrector

B4_EVAL = ROOT / "runs" / "v6_r2" / "eval_B4_distill50"
T0_EVAL = ROOT / "runs" / "v7" / "p2_eval" / "T0"


def load_eval(p: Path):
    f = Path(p) / "eval.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def _c(e):
    return e["cross600"] if e else None


def p1_gate() -> int:
    b4 = _c(load_eval(B4_EVAL))
    arms = {}
    for e in ["E0", "E1", "E2", "E3", "E4", "E3A_repr", "E3B_repr"]:
        arms[e] = _c(load_eval(P3_DIR / f"p1c_{e}"))
    probes = {}
    for o in ["S1", "S2", "R1"]:
        for v in ["fwd", "rev", "rand", "wrong"]:
            probes[f"{o}_{v}"] = _c(load_eval(P3_DIR / f"p1a_{o}_{v}"))

    sign_ok, rand_ok = {}, {}
    for o in ["S1", "S2", "R1"]:
        f, r, rd = (probes.get(f"{o}_fwd"), probes.get(f"{o}_rev"),
                    probes.get(f"{o}_rand"))
        if f and r:
            sign_ok[o] = f["dino_s"] > r["dino_s"]
        if f and rd:
            rand_ok[o] = f["dino_s"] > rd["dino_s"]

    e_stats = {k: v for k, v in arms.items() if v}
    best_global = None
    if e_stats:
        best_global = max(e_stats, key=lambda k: e_stats[k]["dino_s"])
    gate = {"arms": e_stats, "probes": probes, "sign_ok": sign_ok,
            "rand_ok": rand_ok, "best_of_E": best_global}
    # P1 gate: a LOCAL objective (E3/E4) must improve independent style metrics
    # over B4 at the same budget and not lose to the global-stat objectives.
    local_set = ("E3", "E4", "E3A_repr", "E3B_repr")
    passing = []
    for e in local_set:
        a = arms.get(e)
        if not a or not b4:
            continue
        cond_b4 = a["dino_s"] > b4["dino_s"] - 0.002
        others = [v["dino_s"] for k, v in e_stats.items() if k not in local_set]
        cond_others = (not others) or a["dino_s"] >= max(others) - 0.001
        cond_sign = sign_ok.get("S2", True) or sign_ok.get("S1", True)
        if cond_b4 and cond_others and cond_sign:
            passing.append(e)
    if passing:
        gate["verdict"] = "GO_LOCAL"
        gate["best_energy"] = max(passing, key=lambda e: arms[e]["dino_s"])
    else:
        gate["verdict"] = "NO_LOCAL"
        gate["best_energy"] = None
    write_json(gate, P1_DIR / "p1_gate.json")
    print(f"[p1gate] {gate['verdict']} best={gate['best_energy']}", flush=True)
    return 0


def p2_gate() -> int:
    b4 = _c(load_eval(B4_EVAL))
    gate = {"arms": {}, "verdict": "NO_PREDICTOR", "best_predictor": None,
            "best_step": None, "best_ckpt": None}
    arm_dirs = []
    if P2_DIR.exists():
        for d in sorted(P2_DIR.iterdir()):
            if d.is_dir() and list(d.glob("step_*.pt")) and "eval_" in " ".join(
                    x.name for x in d.iterdir()):
                arm_dirs.append(d.name)
    for name in arm_dirs:
        for ck in sorted((P2_DIR / name).glob("step_*.pt")):
            step = int(ck.stem.split("_")[1])
            ev_dir = P2_DIR / name / f"eval_step_{step:05d}"
            ev = load_eval(ev_dir)
            if not ev or not b4:
                continue
            c = ev["cross600"]
            ok = (c["dino_s"] > b4["dino_s"] + 0.002
                  and c["dino_c"] >= b4["dino_c"] - 0.01
                  and c["lpips"] <= b4["lpips"] + 0.02)
            probe_f = P2_DIR / name / "corrector_probe.json"
            resid = json.loads(probe_f.read_text(encoding="utf-8")).get(
                "residual_rms_mean") if probe_f.exists() else None
            gate["arms"][f"{name}@{step}"] = {
                "cross600": c, "beats_b4": ok, "residual_rms_mean": resid,
                "ckpt": str(ev_dir.parent / f"step_{step:05d}.pt")}
            if ok and resid is not None and resid <= 0.32:
                if (gate["best_predictor"] is None
                        or c["dino_s"] > gate["arms"][gate["best_predictor"]]
                        ["cross600"]["dino_s"]):
                    gate["verdict"] = "NEW_PREDICTOR"
                    gate["best_predictor"] = name
                    gate["best_step"] = step
                    gate["best_ckpt"] = gate["arms"][f"{name}@{step}"]["ckpt"]
    write_json(gate, P2_DIR / "p2_gate.json")
    print(f"[p2gate] {gate['verdict']} best={gate['best_predictor']}"
          f"@{gate['best_step']}", flush=True)
    return 0


def e1_summary() -> int:
    """E1 causal-intervention summary: paired deltas of the conditioned arms
    against E1_A (correct source + correct target), plus the R1-predicted
    source substitution gap."""
    from tools.v6_24h.stats import paired_cluster_ci

    def rows(e):
        return {r["task_id"]: r for r in e["rows"] if r["is_cross"]}

    def paired(a, b):
        ra, rb = rows(a), rows(b)
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

    base = load_eval(P3_DIR / "E1_A")
    out = {"base": "E1_A", "comparisons": {}}
    if base:
        for arm in ["E1_B", "E1_C", "E1_D", "E1_R1src"]:
            e = load_eval(P3_DIR / arm)
            if e:
                out["comparisons"][f"{arm}_minus_E1A"] = paired(e, base)
        # core judgment
        b = out["comparisons"].get("E1_B_minus_E1A", {}).get("dino_s")
        out["source_conditioning_used"] = bool(
            b and b["delta"] > 0 and b["ci"][0] > 0)
        r1sub = out["comparisons"].get("E1_R1src_minus_E1A", {}).get("dino_s")
        out["r1_predicted_source_gap"] = (r1sub or {}).get("delta")
    write_json(out, ROOT / "runs" / "v7r" / "e1_summary.json")
    print(f"[e1] source_conditioning_used={out.get('source_conditioning_used')}",
          flush=True)
    return 0


def p3_gate() -> int:
    b4 = _c(load_eval(B4_EVAL))
    t0 = _c(load_eval(T0_EVAL))
    gate = {"arms": {}, "verdict": "NO_CORRECTOR", "best_corrector": None,
            "t0": t0}
    for arm in ["R1c", "R2c", "R3c", "R4c", "R1c_s1", "R1c_s2", "R3c_noC",
                "R3c_disp"]:
        ev = load_eval(P3_DIR / arm)
        if ev:
            gate["arms"][arm] = ev["cross600"]
    # candidate ranking: best DINO-S among correctors from B4; the content gate
    # (DINO-C >= B4 - 0.005) is reported separately so R5c can still combine
    # the best predictor with the best available corrector.
    cand = {}
    content_passed = {}
    for arm, c in gate["arms"].items():
        if arm in ("R3c_noC", "R3c_disp"):
            continue  # variants recorded, not candidates
        cand[arm] = c["dino_s"]
        content_passed[arm] = bool(b4 and c["dino_c"] >= b4["dino_c"] - 0.005)
    if cand:
        best = max(cand, key=cand.get)
        gate["best_corrector"] = best
        gate["content_gate_passed"] = content_passed[best]
        if content_passed[best] and b4 and cand[best] > b4["dino_s"]:
            gate["verdict"] = "R_CORRECTOR_AVAILABLE"
    write_json(gate, P3_DIR / "p3_gate.json")
    print(f"[p3gate] {gate['verdict']} best={gate['best_corrector']}", flush=True)
    return 0


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="V7-R gates")
    ap.add_argument("gate", choices=["p1", "p2", "p3", "e1"])
    args = ap.parse_args()
    return {"p1": p1_gate, "p2": p2_gate, "p3": p3_gate,
            "e1": e1_summary}[args.gate]()


if __name__ == "__main__":
    raise SystemExit(main())
