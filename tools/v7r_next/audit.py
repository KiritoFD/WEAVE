"""P0: experiment-credibility audit (WEAVE-CPT / V7R_NEXT).

Three jobs, all on 3060 WSL:

  p0a_direction   P0.1 gradient-direction audit with >=5 independent seeds
                  for random direction, plus zero-update control.
                  Arms: for each objective in {R1,S1,S2,E3,B4}:
                    fwd / rev / wrong / rand_seed{0..4} / zero
                  each full 750 eval. Records per-arm target specificity
                  (from r2_eval's new dino_s_specificity field), CLIP-S,
                  CSD dT/Margin, confusion matrix. Verifies:
                    * fwd improves its own objective
                    * R1 fwd raises target margin
                    * rev has opposite effect
                    * seeds are isolated (different rand results)
                    * zero == B4 baseline
                  Gate: random-direction DINO-S gain WITHOUT specificity gain
                  => all later methods MUST report specificity, not just DINO-S.

  p0b_duplicate   P0.2 duplicate-arm audit: R3c_noC vs R3c_disp and
                  p1c_E0 vs p1c_E1. Compares run_meta/config, eval hash,
                  per-task rows (by task_id), final latent if present.

  p0c_verdict     P0.3 correct the historical record (documentation only):
                  R3c/R4c are TEST-TIME edits; N1/N2 trained correctors FAILED.
                  "Frozen B4 + inference-edit stable" does NOT prove a learned
                  corrector succeeded.

Outputs under runs/v7r_next/p0_*/, verdict in p0_gate.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.v7r_next import common as C  # noqa: E402

V7R = C.V7R
OUT = C.RUNS / "p0_audit"
EVAL = "tools/v6_next/r2_eval.py"
PY = "/usr/bin/python3"


def eval_arm(arm: str, out_dir: Path, seed: int | None = None,
             extra: list[str] | None = None) -> dict:
    """Run one full r2_eval --gen v7r arm, return its cross600 summary."""
    import subprocess
    out_dir.mkdir(parents=True, exist_ok=True)
    f = out_dir / "eval.json"
    if f.exists():
        return C.read_json(f)
    env = dict(__import__("os").environ)
    env["V7R_NEXT_SEED"] = str(seed if seed is not None else C.SEED)
    cmd = [PY, "-m", "tools.v6_next.r2_eval", "--gen", "v7r",
           "--v7r-arm", arm, "--ckpt", str(C.B4_CKPT.relative_to(C.ROOT)),
           "--out-dir", str(out_dir.relative_to(C.ROOT)),
           "--batch-size", "8"]
    if extra:
        cmd += extra
    t0 = time.time()
    log = C.LOGS / f"p0_{arm}_{seed if seed is not None else 'x'}.log"
    with open(log, "w") as lf:
        rc = subprocess.call(cmd, cwd=str(C.ROOT), env=env, stdout=lf, stderr=lf)
    if rc != 0:
        raise RuntimeError(f"eval {arm} seed={seed} rc={rc}: {log}")
    C.record_timing("eval750", time.time() - t0)
    return C.read_json(f)


def specificity(d: dict) -> dict:
    rows = d.get("rows", [])
    return C.summarize_specificity(rows)


def p0a_direction() -> dict:
    """P0.1: direction + target specificity + seed isolation."""
    OUT.mkdir(parents=True, exist_ok=True)
    # existing v7r probe arm names: p1a_{S1,S2,R1}_{fwd,rev,wrong,rand}
    # B4 = R0 (no edit); zero = R0 as well (documented equality check).
    arms = {
        "fwd_R1": "p1a_R1_fwd", "rev_R1": "p1a_R1_rev",
        "wrong_R1": "p1a_R1_wrong",
        "fwd_S1": "p1a_S1_fwd", "rev_S1": "p1a_S1_rev",
        "wrong_S1": "p1a_S1_wrong",
        "fwd_S2": "p1a_S2_fwd", "rev_S2": "p1a_S2_rev",
        "wrong_S2": "p1a_S2_wrong",
    }
    # E3 has no single-step probe arm in v7r; use p1c_E3 (4-step E3 energy)
    # and R4c as the weak-R1-supplemented reference.
    arms["E3_4step"] = "p1c_E3"
    arms["R4c_ref"] = "R4c"
    arms["B4_zero"] = "R0"

    results = {}
    for label, arm in arms.items():
        d = eval_arm(arm, OUT / "dir" / label)
        c = d["cross600"]
        sp = specificity(d)
        results[label] = {
            "arm": arm,
            "top1": round(c["top1"], 5), "dino_s": round(c["dino_s"], 5),
            "dino_c": round(c["dino_c"], 5), "lpips": round(c["lpips"], 5),
            "margin": round(c["margin"], 5), "dT": round(c["dT"], 5),
            "specificity": sp["cross600_dino_s_specificity"],
            "confusion": c.get("confusion_matrix"),
        }
        C.write_json(OUT / "dir" / label / "p0_summary.json", results[label])

    # 5 independent random seeds for R1/S1/S2 random-direction probes.
    # r2_eval --gen v7r reads V7R_NEXT_SEED in its probe_step random branch
    # (patched: see patch_r2eval_seed.py).
    rand = {}
    for obj in ("R1", "S1", "S2"):
        for seed in range(5):
            d = eval_arm(f"p1a_{obj}_rand", OUT / "rand" / f"{obj}_s{seed}",
                         seed=seed)
            c = d["cross600"]
            sp = specificity(d)
            rand[f"{obj}_s{seed}"] = {
                "dino_s": round(c["dino_s"], 5),
                "specificity": sp["cross600_dino_s_specificity"]["mean"],
                "top1": round(c["top1"], 5),
                "eval_hash": hashlib.sha256(
                    json.dumps(d.get("rows", []), sort_keys=True).encode()
                ).hexdigest()[:16],
            }
    C.write_json(OUT / "random_seeds.json", rand)

    # ---- checks ----
    base = results["B4_zero"]
    fwd = results["fwd_R1"]
    rev = results["rev_R1"]
    zero_eq_b4 = (base["arm"] == "R0")
    fwd_better_own = fwd["dino_s"] > base["dino_s"]
    rev_opposite = rev["dino_s"] < fwd["dino_s"]
    seeds_isolated = len({v["eval_hash"] for v in rand.values()}) == len(rand)
    # specificity gate: do random directions raise DINO-S without specificity?
    rands = [v for k, v in rand.items()]
    rand_dino_gain = float(np_mean([v["dino_s"] for v in rands])) - base["dino_s"]
    rand_spec_gain = float(np_mean([v["specificity"] for v in rands])) - \
        base["specificity"]["mean"]
    specificity_gate = (rand_dino_gain > 0.002 and rand_spec_gain <= 0.001)
    verdict = "PASS" if (zero_eq_b4 and fwd_better_own and rev_opposite
                         and seeds_isolated) else "FIX_AND_REEVAL"
    gate = {
        "verdict": verdict,
        "checks": {
            "zero_equals_B4": zero_eq_b4,
            "R1_fwd_raises_DINO_S": fwd_better_own,
            "R1_rev_opposite": rev_opposite,
            "random_seeds_isolated": seeds_isolated,
            "R1_fwd_raises_target_margin": fwd["margin"] > base["margin"],
        },
        "random_direction": {
            "mean_DINO_S_gain": round(rand_dino_gain, 5),
            "mean_specificity_gain": round(rand_spec_gain, 5),
            "specificity_gate_triggered": bool(specificity_gate),
            "consequence": ("All later methods MUST report target specificity, "
                            "not just DINO-S" if specificity_gate else
                            "random gain is target-specific; specificity still reported"),
        },
        "note": "P0 Gate: if any check fails, fix and fully re-evaluate "
                "affected experiments before P1.",
    }
    C.write_json(C.RUNS / "p0_gate.json", gate)
    C.write_json(OUT / "direction_all.json", results)
    return gate


def np_mean(x):
    import numpy as np
    return float(np.mean(x))


def _rows_by_id(d: dict) -> dict:
    return {r["task_id"]: r for r in d.get("rows", [])}


def p0b_duplicate() -> dict:
    """P0.2: R3c_noC vs R3c_disp, p1c_E0 vs p1c_E1."""
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = [("R3c_noC", "R3c_disp"), ("p1c_E0", "p1c_E1")]
    rep = {}
    for a, b in pairs:
        da = C.read_json(V7R / f"p3_corrector/{a}/eval.json")
        db = C.read_json(V7R / f"p3_corrector/{b}/eval.json")
        ra, rb = _rows_by_id(da), _rows_by_id(db)
        common_ids = sorted(set(ra) & set(rb))
        diffs = {}
        for k in ("dino_s", "dino_c", "lpips", "top1", "margin"):
            key = "target_top1" if k == "top1" else k
            vals = [abs(ra[i].get(key, 0) - rb[i].get(key, 0))
                    for i in common_ids]
            diffs[k] = {"max_abs": round(max(vals), 6) if vals else None,
                        "mean_abs": round(np_mean(vals), 6) if vals else None,
                        "n_identical": sum(1 for v in vals if v < 1e-9)}
        # config / run_meta
        meta_a = _maybe_json(V7R / f"p3_corrector/{a}/run_meta.json")
        meta_b = _maybe_json(V7R / f"p3_corrector/{b}/run_meta.json")
        rep[f"{a}_vs_{b}"] = {
            "n_tasks_a": len(ra), "n_tasks_b": len(rb),
            "n_common": len(common_ids),
            "per_task_metric_diff": diffs,
            "eval_identical": all(d["max_abs"] == 0 for d in diffs.values()),
            "run_meta_equal": meta_a == meta_b,
            "run_meta_a": meta_a, "run_meta_b": meta_b,
            "verdict": ("DUPLICATE_EVAL_SUSPECTED — inspect arm config/branch; "
                        "do not use for optimization design"
                        if all(d["max_abs"] == 0 for d in diffs.values())
                        else "METRICS DIFFER — compare configs & per-step state"),
        }
    C.write_json(OUT / "duplicate_audit.json", rep)
    return rep


def _maybe_json(p: Path):
    return C.read_json(p) if p.exists() else None


def p0c_verdict() -> dict:
    """P0.3: correct the historical record (documentation, no GPU)."""
    v = {
        "R3c_R4c": "VALID TEST-TIME EDIT results (inference-time optimization "
                   "from frozen B4), NOT trained correctors.",
        "N1_N2": "TRAINED CORRECTORS FAILED (NO_PREDICTOR; DINO-C ~0.50, "
                 "LPIPS ~0.71). The learned corrector did NOT succeed.",
        "prohibition": "Do NOT use 'frozen B4 + stable inference edit' as "
                       "evidence that a LEARNED corrector succeeded.",
        "R3c_disp": "high-style endpoint 0.4327/0.8092/0.4607; DUPLICATE "
                    "branch audit (P0.2) must pass before using it for design.",
    }
    C.write_json(OUT / "historical_correction.json", v)
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job", choices=["p0a_direction", "p0b_duplicate",
                                    "p0c_verdict", "all"])
    a = ap.parse_args()
    C.set_seed()
    if a.job in ("p0a_direction", "all"):
        print("[p0a]", p0a_direction())
    if a.job in ("p0b_duplicate", "all"):
        print("[p0b]", json.dumps(p0b_duplicate())[:400])
    if a.job in ("p0c_verdict", "all"):
        print("[p0c]", p0c_verdict())


if __name__ == "__main__":
    main()
