"""V7-R P4: joint validation or a pre-registered extension branch.

Decision rules (V7R_PLAN.md P4) — evidence-driven, recorded:
  1. If the P2 gate found a NEW_PREDICTOR and P3 a best corrector:
     synergy comparison B4 / N-best / B4+R / N+R (R5c already run by the
     runner); writes p4_verdict.json with the complementarity checks.
  2. Else pick ONE diagnostic-driven branch:
     - train curve still improving at 3000  -> EXTEND to 6000 steps
     - early-good-late-crash                -> ROLLBACK to best ckpt
     - unstable across ckpts                -> SECOND SEED
     - style signal insufficient            -> RAND route note (run by P2 alt)
  The runner reads p4_decision.json and executes the chosen branch.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools.v7.v7_common import ROOT, STYLES, write_json
from tools.v7r.v7r_common import P2_DIR, P3_DIR, P4_DIR

B4_EVAL = ROOT / "runs" / "v6_r2" / "eval_B4_distill50"


def load_eval(p: Path):
    f = Path(p) / "eval.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def n_evals(name: str) -> dict:
    out = {}
    for ck in sorted((P2_DIR / name).glob("step_*.pt")):
        step = int(ck.stem.split("_")[1])
        ev = load_eval(P2_DIR / name / f"eval_step_{step:05d}")
        if ev:
            out[step] = ev["cross600"]
    return out


def _trained_arms() -> dict:
    """All corrector arms with checkpoint evaluation curves:
    {name: {step: cross600}}."""
    curves = {}
    if not P2_DIR.exists():
        return curves
    for d in sorted(P2_DIR.iterdir()):
        if not d.is_dir():
            continue
        cks = sorted(d.glob("step_*.pt"))
        if not cks:
            continue
        ev = {}
        for ck in cks:
            step = int(ck.stem.split("_")[1])
            e = load_eval(d / f"eval_step_{step:05d}")
            if e:
                ev[step] = e["cross600"]
        if ev:
            curves[d.name] = ev
    return curves


def _last_ckpt(name: str) -> str:
    cks = sorted((P2_DIR / name).glob("step_*.pt"),
                 key=lambda p: int(p.stem.split("_")[1]))
    return str(cks[-1])


def cmd_decide() -> int:
    """E4 pre-registered branches (V7-R+ section 5) over ALL trained arms,
    plus the synergy decision when a NEW_PREDICTOR exists."""
    P4_DIR.mkdir(parents=True, exist_ok=True)
    p2g = json.loads((P2_DIR / "p2_gate.json").read_text(encoding="utf-8")) \
        if (P2_DIR / "p2_gate.json").exists() else {}
    b4 = load_eval(B4_EVAL)
    b4_ds = b4["cross600"]["dino_s"] if b4 else 0.0

    decision = {"mode": "none", "reason": "", "best_predictor":
                p2g.get("best_predictor")}

    curves = _trained_arms()
    positive_arms = []
    for name, ev in curves.items():
        final_step = max(ev)
        if ev[final_step]["dino_s"] > b4_ds + 0.001:
            positive_arms.append((name, ev[final_step]["dino_s"] - b4_ds))
    positive_arms.sort(key=lambda x: -x[1])
    decision["positive_arms"] = positive_arms

    # second seed: any positive Pareto improvement must be replicated (E4 rule)
    if positive_arms:
        arm = positive_arms[0][0]
        if not (P2_DIR / f"{arm}_s43").exists():
            decision["second_seed"] = {
                "arm": arm, "reason": f"positive ΔDINO-S vs B4 "
                f"{positive_arms[0][1]:+.4f} requires a second seed (E4 rule)"}

    if p2g.get("best_predictor"):
        decision["mode"] = "synergy"
        decision["reason"] = ("new predictor available; synergy = "
                              f"{p2g.get('best_predictor')} + best corrector "
                              f"vs B4 / N / R alone")
        write_json(decision, P4_DIR / "p4_decision.json")
        print(f"[p4] decision: {decision['mode']} ({decision['reason']})",
              flush=True)
        return 0

    # E4 branch rules over curves
    chosen = None
    for name, ev in sorted(curves.items()):
        if name.endswith("_s43"):
            continue
        steps = sorted(ev)
        if len(steps) < 2:
            continue
        vals = [ev[k]["dino_s"] for k in steps]
        ds_last = vals[-1] - vals[-2]
        if ds_last > 0.0005 and ev[steps[-1]]["dino_c"] > 0.70:
            chosen = (name, "EXTEND", "DINO-S still improving at the final "
                      f"checkpoint ({ds_last:+.4f})")
            break
        if max(vals) > vals[-1] + 0.002:
            chosen = (name, "ROLLBACK_LR", "early checkpoints beat the final "
                      "one; rollback + low-lr continuation对照")
            break
    if chosen is None and curves:
        # branch C: stalled — pre-registered alternative cascade
        for alt in ["E2_C", "E5", "E2_B"]:
            if alt not in curves:
                chosen = (alt, "ALTERNATIVE", "all curves stalled; train the "
                          f"next pre-registered alternative arm {alt}")
                break
    if chosen:
        decision["mode"] = chosen[1]
        decision["target_arm"] = chosen[0]
        decision["reason"] = chosen[2]
    else:
        decision["mode"] = "NONE"
        decision["reason"] = "no trainable arm curves available for E4 branches"
    write_json(decision, P4_DIR / "p4_decision.json")
    print(f"[p4] decision: {decision['mode']} ({decision['reason']})",
          flush=True)
    return 0


def _best_trained(curves: dict) -> str:
    """Arm name with the highest final DINO-S among evaluated checkpoints."""
    best, best_v = "N2", -1.0
    for name, c in curves.items():
        vals = list(c.values())
        if vals and vals[-1] > best_v:
            best, best_v = name, vals[-1]
    return best


def cmd_judge() -> int:
    """After the runner executed the decided branch: final synergy/verdict."""
    arms = {}
    e = load_eval(B4_EVAL)
    if e:
        arms["B4"] = e
    for d in sorted(P2_DIR.iterdir()) if P2_DIR.exists() else []:
        if not d.is_dir():
            continue
        best = None
        for ck in sorted(d.glob("step_*.pt")):
            step = int(ck.stem.split("_")[1])
            ev = load_eval(d / f"eval_step_{step:05d}")
            if ev and (best is None
                       or ev["cross600"]["dino_s"] > best[1]["cross600"]["dino_s"]):
                best = (step, ev)
        if best:
            arms[f"{d.name}@{best[0]}"] = best[1]
    r5 = load_eval(P3_DIR / "R5c")
    if r5:
        arms["R5c"] = r5
    out = {"arms": {k: {"cross600": v["cross600"], "same150": v["same150"]}
                    for k, v in arms.items()}}
    p3g = json.loads((P3_DIR / "p3_gate.json").read_text(encoding="utf-8")) \
        if (P3_DIR / "p3_gate.json").exists() else {}
    best_corr = p3g.get("best_corrector")
    if "R5c" in arms and best_corr and best_corr in p3g.get("arms", {}):
        r3 = p3g["arms"][best_corr]
        comp = {
            "dino_s_R5c_minus_R": arms["R5c"]["cross600"]["dino_s"]
            - r3["dino_s"],
            "dino_c_R5c_minus_R": arms["R5c"]["cross600"]["dino_c"]
            - r3["dino_c"],
            "displacement_R5c": _mean_disp(P3_DIR / "R5c"),
            "displacement_R": _mean_disp(P3_DIR / best_corr),
        }
        comp["displacement_reduced"] = (
            comp["displacement_R5c"] is not None
            and comp["displacement_R"] is not None
            and comp["displacement_R5c"] < comp["displacement_R"])
        out["synergy"] = comp
    write_json(out, P4_DIR / "p4_verdict.json")
    print("[p4] judge written", flush=True)
    return 0


def _mean_disp(p: Path) -> float | None:
    f = Path(p) / "optim_trace.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text(encoding="utf-8")).get("displacement", {})
    return float(np.mean(list(d.values()))) if d else None


def main() -> int:
    ap = argparse.ArgumentParser(description="V7-R P4")
    ap.add_argument("mode", choices=["decide", "judge"])
    args = ap.parse_args()
    return cmd_decide() if args.mode == "decide" else cmd_judge()


if __name__ == "__main__":
    raise SystemExit(main())
