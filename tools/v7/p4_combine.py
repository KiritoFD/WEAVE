"""V7 P4: combine Predictor & Corrector — two modes.

  python -m tools.v7.p4_combine plan    # P2/P3 verdicts -> p4_plan.json (+skips)
  python -m tools.v7.p4_combine judge   # after planned evals -> p4_verdict.json

Rules (V7_EXPERIMENT_MATRIX.md sections 4-6):
  - C0 (B4 direct) reuses runs/v6_r2/eval_B4_distill50; C1 (B4 + R1 gradient)
    reuses the P2 T0 arm — identical protocol, no re-run needed.
  - C2 runs only on NEW_CORRECTOR; C3 only on NEW_PREDICTOR; C4 needs both.
  - Complementarity: C4 vs C2 must reduce corrector displacement without
    degrading DINO-C / LPIPS / DINO-S / Same150.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from tools.v7.v7_common import (
    P2_DIR, P3_EVAL, P4_DIR, ROOT, STYLES, load_tasks, pick_trace_tasks,
    tasks_sha256, write_json,
)

B4_EVAL = ROOT / "runs" / "v6_r2" / "eval_B4_distill50"
ARM_PRIORITY = {"T4": 4, "T5": 3, "T2": 2, "T3": 1}


def load_eval(path: Path):
    f = Path(path) / "eval.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def grad_ok(path: Path) -> bool | None:
    f = Path(path) / "grad_norm.json"
    if not f.exists():
        return None
    a = json.loads(f.read_text(encoding="utf-8")).get("audit", {})
    return not a.get("imbalanced", False)


def dinoc_bins(eval_json: dict, bins=((0.74, 0.76), (0.76, 0.78), (0.78, 0.80),
                                     (0.80, 0.82), (0.82, 0.84))) -> dict:
    rows = [r for r in eval_json["rows"] if r["is_cross"]]
    out = {}
    for lo, hi in bins:
        sub = [r for r in rows if lo <= r["dino_c"] < hi]
        if sub:
            out[f"[{lo:.2f},{hi:.2f})"] = {
                "n": len(sub),
                "dino_s_mean": float(np.mean([r["dino_s"] for r in sub])),
                "dino_s_best": float(np.max([r["dino_s"] for r in sub])),
                "lpips_mean": float(np.mean([r["lpips"] for r in sub])),
            }
    return out


def p2_verdict() -> dict:
    arms = {}
    for name in ["T0", "T1", "T2", "T3", "T4", "T5"]:
        e = load_eval(P2_DIR / name)
        if e:
            arms[name] = e
    if "T0" not in arms:
        return {"verdict": "INCONCLUSIVE", "reason": "T0 eval missing"}
    t0 = arms["T0"]["cross600"]
    t0_same = arms["T0"]["same150"]
    qualifying, details = [], {}
    for name, e in arms.items():
        if name not in ARM_PRIORITY:
            continue
        c, s = e["cross600"], e["same150"]
        g = grad_ok(P2_DIR / name)
        trace_ok = (P2_DIR / name / "optim_trace.json").exists()
        reasons = []
        if c["dino_s"] <= t0["dino_s"]:
            reasons.append(f"dino_s {c['dino_s']:.4f} <= T0 {t0['dino_s']:.4f}")
        if c["dino_c"] < t0["dino_c"] - 0.002:
            reasons.append(f"dino_c drop {c['dino_c'] - t0['dino_c']:+.4f}")
        if s["same_dino_c"] < t0_same["same_dino_c"] - 0.005:
            reasons.append("Same150 dino_c degraded")
        if s["same_lpips"] > t0_same["same_lpips"] + 0.005:
            reasons.append("Same150 lpips degraded")
        if g is False:
            reasons.append("grad imbalanced")
        if not trace_ok:
            reasons.append("optim_trace.json missing")
        bins_a, bins_t0 = dinoc_bins(e), dinoc_bins(arms["T0"])
        bin_worse = []
        for k in bins_a:
            if k in bins_t0 and bins_a[k]["n"] >= 5 and bins_t0[k]["n"] >= 5:
                d = bins_a[k]["dino_s_mean"] - bins_t0[k]["dino_s_mean"]
                if d < -0.01:
                    bin_worse.append(f"{k}: {d:+.4f}")
        if bin_worse:
            reasons.append("same-C-bin DINO-S worse: " + "; ".join(bin_worse))
        details[name] = {
            "cross600": c, "same150": s, "grad_ok": g, "reasons": reasons,
            "dinoc_bins": bins_a, "qualifies": not reasons,
        }
        if not reasons:
            qualifying.append(name)
    if qualifying:
        best = max(qualifying, key=lambda a: (ARM_PRIORITY[a],
                                              details[a]["cross600"]["dino_s"]))
        verdict = "NEW_CORRECTOR"
    else:
        best = None
        verdict = "NO_CORRECTOR"
    return {
        "verdict": verdict,
        "best_corrector": best,
        "t0": {"cross600": t0, "same150": t0_same,
               "dinoc_bins": dinoc_bins(arms["T0"])},
        "arms": details,
        "fixed_topk_reference": "runs/v6_w/eval_topk_fix (PARTIAL, +0.0187/+0.0185)",
    }


def p3_verdict(p2v: dict) -> dict:
    b4 = load_eval(B4_EVAL)
    m1, m2 = load_eval(P3_EVAL / "M1"), load_eval(P3_EVAL / "M2")
    probes = {}
    for name in ["M1", "M2"]:
        f = ROOT / "runs" / "v7" / "p3_models" / name / "five_style_probe.json"
        if f.exists():
            probes[name] = json.loads(f.read_text(encoding="utf-8"))
    audit_p0 = ROOT / "runs" / "v7" / "p0_b4_causal_audit.json"
    b4_residual = None
    if audit_p0.exists():
        b4_residual = json.loads(audit_p0.read_text(encoding="utf-8"))[
            "overall"]["source_residual_r1"]
    if m2 is None:
        return {"verdict": "NO_PREDICTOR", "reason": "M2 eval missing"}

    checks = {}
    if b4 is not None:
        checks["m2_beats_b4"] = m2["cross600"]["dino_s"] > b4["cross600"]["dino_s"]
        checks["m2_dino_s_vs_b4"] = m2["cross600"]["dino_s"] - b4["cross600"]["dino_s"]
    if m1 is not None:
        checks["m2_beats_m1"] = m2["cross600"]["dino_s"] > m1["cross600"]["dino_s"]
        checks["m1_beats_b4"] = m1["cross600"]["dino_s"] > b4["cross600"]["dino_s"]
    pr = probes.get("M2", {}).get("overall", {})
    checks["style_outputs_differ"] = pr.get("mean_pairwise_latent_rms", 0) > 0.01
    if b4_residual is not None and pr:
        checks["source_leak_reduced"] = (pr.get("source_residual_rate_r1", 1.0)
                                         < b4_residual)
    checks["residual_rms_reasonable"] = pr.get("residual_rms_mean", 9) <= 0.32

    hard = [k for k in ("m2_beats_b4", "m2_beats_m1") if k in checks]
    if hard and all(checks[k] for k in hard) and checks["style_outputs_differ"] \
            and checks.get("source_leak_reduced", False) \
            and checks["residual_rms_reasonable"]:
        verdict = "NEW_PREDICTOR"
    elif checks.get("m1_beats_b4") and not checks.get("m2_beats_m1", False):
        verdict = "STRUCTURE_ONLY"
    else:
        verdict = "NO_PREDICTOR"
    return {
        "verdict": verdict, "checks": checks,
        "b4": None if b4 is None else b4["cross600"],
        "M1": None if m1 is None else m1["cross600"],
        "M2": m2["cross600"],
        "probes": probes,
        "b4_source_residual_r1": b4_residual,
        "p2_verdict": p2v["verdict"],
    }


def cmd_plan() -> int:
    P4_DIR.mkdir(parents=True, exist_ok=True)
    p2v = p2_verdict()
    write_json(p2v, ROOT / "runs" / "v7" / "p2_verdict.json")
    p3v = p3_verdict(p2v)
    write_json(p3v, ROOT / "runs" / "v7" / "p3_verdict.json")

    plan, skipped = [], []
    best = p2v.get("best_corrector")
    has_corr = p2v["verdict"] == "NEW_CORRECTOR"
    has_pred = p3v["verdict"] == "NEW_PREDICTOR"

    plan.append({"arm": "C0", "predictor": "B4", "corrector": None,
                 "reuse": str(B4_EVAL)})
    plan.append({"arm": "C1", "predictor": "B4", "corrector": "T0",
                 "reuse": str(P2_DIR / "T0")})
    if has_corr:
        plan.append({"arm": "C2", "predictor": "B4", "corrector": best,
                     "gen": "v7", "v7_arm": best, "ckpt": "B4",
                     "start_mode": "direct"})
    else:
        skipped.append({"arm": "C2", "reason": f"P2 verdict {p2v['verdict']}"})
    if has_pred:
        plan.append({"arm": "C3", "predictor": "M2", "corrector": None,
                     "gen": "relative", "ckpt": "M2"})
    else:
        skipped.append({"arm": "C3", "reason": f"P3 verdict {p3v['verdict']}"})
    if has_corr and has_pred:
        plan.append({"arm": "C4", "predictor": "M2", "corrector": best,
                     "gen": "v7", "v7_arm": best, "ckpt": "M2",
                     "start_mode": "relative"})
    else:
        skipped.append({"arm": "C4",
                        "reason": f"corr={p2v['verdict']}, pred={p3v['verdict']}"})

    write_json({"plan": plan, "tasks_sha256": tasks_sha256()},
               ROOT / "runs" / "v7" / "p4_plan.json")
    write_json({"skipped": skipped}, ROOT / "runs" / "v7" / "p4_skipped.json")
    print(f"[p4] plan: {[p['arm'] for p in plan]}  skipped: {[s['arm'] for s in skipped]}",
          flush=True)
    return 0


def cmd_judge() -> int:
    evals = {}
    for arm in ["C0", "C1", "C2", "C3", "C4"]:
        e = load_eval(P4_DIR / arm)
        if e:
            evals[arm] = e
    out = {"arms": {k: {"cross600": v["cross600"], "same150": v["same150"]}
                    for k, v in evals.items()}}
    if "C4" in evals and "C2" in evals:
        c4, c2 = evals["C4"]["cross600"], evals["C2"]["cross600"]

        def disp(arm):
            f = P4_DIR / arm / "optim_trace.json"
            if not f.exists():
                return None
            d = json.loads(f.read_text(encoding="utf-8"))["displacement"]
            return float(np.mean(list(d.values()))) if d else None

        d4, d2 = disp("C4"), disp("C2")
        comp = {
            "displacement_C4": d4, "displacement_C2": d2,
            "displacement_reduced": (d4 is not None and d2 is not None
                                     and d4 < d2),
            "dino_c_not_down": c4["dino_c"] >= c2["dino_c"] - 0.002,
            "lpips_not_up": c4["lpips"] <= c2["lpips"] + 0.002,
            "dino_s_not_down": c4["dino_s"] >= c2["dino_s"] - 0.002,
            "same150_not_degraded": (
                evals["C4"]["same150"]["same_dino_c"]
                >= evals["C2"]["same150"]["same_dino_c"] - 0.005
                and evals["C4"]["same150"]["same_lpips"]
                <= evals["C2"]["same150"]["same_lpips"] + 0.005),
        }
        comp["complementary"] = all(
            comp[k] for k in ["displacement_reduced", "dino_c_not_down",
                              "lpips_not_up", "dino_s_not_down",
                              "same150_not_degraded"])
        out["c4_vs_c2"] = comp
    write_json(out, ROOT / "runs" / "v7" / "p4_verdict.json")
    print(f"[p4] judge written; complementary="
          f"{out.get('c4_vs_c2', {}).get('complementary')}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="V7 P4 combine")
    ap.add_argument("mode", choices=["plan", "judge"])
    args = ap.parse_args()
    return cmd_plan() if args.mode == "plan" else cmd_judge()


if __name__ == "__main__":
    raise SystemExit(main())
