"""V7-R+ ETA bookkeeping (section 8.1/8.2 of the plan).

Separates BUDGET_TOTAL_SEC (hard cap), ELAPSED_SEC (wall since start),
GPU_ACTIVE_SEC (measured GPU work so far) and ETA_REMAINING_SEC (work-based
estimate). The runner calls this after every stage and logs the values.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from tools.v7.v7_common import ROOT, write_json

RUNS = ROOT / "runs" / "v7r"
BUDGET_TOTAL_SEC = 57600  # 16h hard cap (14h target)

# Planned GPU work inventory: (artifact_check_path, kind, default_sec).
# kind: train = one training run, eval = one full-750 evaluation, misc.
PLAN = [
    # P0
    ("p0_ckpts/M1/step_0800.pt", "train", 900),
    ("p0_ckpts/M2/step_0800.pt", "train", 1000),
    ("p0_failure_audit.json", "misc", 600),
    # P1
    ("p1_energy/patch_banks.pt", "misc", 300),
    ("p3_corrector/p1a_S1_fwd/eval.json", "eval", 0),
    ("p3_corrector/p1a_S1_rev/eval.json", "eval", 0),
    ("p3_corrector/p1a_S1_rand/eval.json", "eval", 0),
    ("p3_corrector/p1a_S1_wrong/eval.json", "eval", 0),
    ("p3_corrector/p1a_S2_fwd/eval.json", "eval", 0),
    ("p3_corrector/p1a_S2_rev/eval.json", "eval", 0),
    ("p3_corrector/p1a_S2_rand/eval.json", "eval", 0),
    ("p3_corrector/p1a_S2_wrong/eval.json", "eval", 0),
    ("p3_corrector/p1a_R1_fwd/eval.json", "eval", 0),
    ("p3_corrector/p1a_R1_rev/eval.json", "eval", 0),
    ("p3_corrector/p1a_R1_rand/eval.json", "eval", 0),
    ("p3_corrector/p1a_R1_wrong/eval.json", "eval", 0),
    ("p3_corrector/p1c_E0/eval.json", "eval", 0),
    ("p3_corrector/p1c_E1/eval.json", "eval", 0),
    ("p3_corrector/p1c_E2/eval.json", "eval", 0),
    ("p3_corrector/p1c_E3/eval.json", "eval", 0),
    ("p3_corrector/p1c_E4/eval.json", "eval", 0),
    # P2 (N1/N2 trainings + per-ckpt evals)
    ("p2_predictor/N1/step_03000.pt", "train", 2700),
    ("p2_predictor/N2/step_03000.pt", "train", 2700),
    ("p2_predictor/N1/eval_step_0500/eval.json", "eval", 0),
    ("p2_predictor/N1/eval_step_1500/eval.json", "eval", 0),
    ("p2_predictor/N1/eval_step_3000/eval.json", "eval", 0),
    ("p2_predictor/N2/eval_step_0500/eval.json", "eval", 0),
    ("p2_predictor/N2/eval_step_1500/eval.json", "eval", 0),
    ("p2_predictor/N2/eval_step_3000/eval.json", "eval", 0),
    # E1 (5 condition interventions)
    ("p3_corrector/E1_A/eval.json", "eval", 0),
    ("p3_corrector/E1_B/eval.json", "eval", 0),
    ("p3_corrector/E1_C/eval.json", "eval", 0),
    ("p3_corrector/E1_D/eval.json", "eval", 0),
    ("p3_corrector/E1_R1src/eval.json", "eval", 0),
    # E2 (2 trainings + per-ckpt evals; E2-A reuses N2)
    ("p2_predictor/E2_B/step_03000.pt", "train", 3000),
    ("p2_predictor/E2_C/step_03000.pt", "train", 3600),
    ("p2_predictor/E2_B/eval_step_1000/eval.json", "eval", 0),
    ("p2_predictor/E2_B/eval_step_2000/eval.json", "eval", 0),
    ("p2_predictor/E2_B/eval_step_3000/eval.json", "eval", 0),
    ("p2_predictor/E2_C/eval_step_1000/eval.json", "eval", 0),
    ("p2_predictor/E2_C/eval_step_2000/eval.json", "eval", 0),
    ("p2_predictor/E2_C/eval_step_3000/eval.json", "eval", 0),
    # E3 (2 repr trainings + phase1 + 2 edit evals)
    ("e3/repr_A.pt", "train", 900),
    ("e3/repr_B.pt", "train", 900),
    ("p3_corrector/p1c_E3A_repr/eval.json", "eval", 0),
    ("p3_corrector/p1c_E3B_repr/eval.json", "eval", 0),
    # E5 (1 training + 3 ckpt evals)
    ("p2_predictor/E5/step_03000.pt", "train", 2700),
    ("p2_predictor/E5/eval_step_1000/eval.json", "eval", 0),
    ("p2_predictor/E5/eval_step_2000/eval.json", "eval", 0),
    ("p2_predictor/E5/eval_step_3000/eval.json", "eval", 0),
    # E4 (branch: one continuation + up to 3 evals; second seed + 3 evals)
    ("p4_joint/p4_verdict.json", "misc", 0),
    # P3 (R0/R1c/R2c + gated R3/R4/R5 + short paths + variants)
    ("p3_corrector/R0/eval.json", "eval", 0),
    ("p3_corrector/R1c/eval.json", "eval", 0),
    ("p3_corrector/R2c/eval.json", "eval", 0),
    ("p3_corrector/R3c/eval.json", "eval", 0),
    ("p3_corrector/R4c/eval.json", "eval", 0),
    ("p3_corrector/R5c/eval.json", "eval", 0),
    ("p3_corrector/p3_gate.json", "misc", 60),
    # P5
    ("visual_grid.png", "misc", 180),
    ("final_matrix.json", "misc", 120),
]

START_FILE = RUNS / "state" / ".start_ts"


def _mean_eval_wall() -> float:
    walls = []
    for rt in RUNS.rglob("runtime.json"):
        try:
            w = json.loads(rt.read_text(encoding="utf-8")).get("wall_sec")
            if w and w > 30:
                walls.append(w)
        except Exception:  # noqa: BLE001
            pass
    return sum(walls) / len(walls) if walls else 150.0


def _mean_train_wall() -> float:
    walls = []
    for cfg in (RUNS / "p2_predictor").glob("*/config.json"):
        try:
            w = json.loads(cfg.read_text(encoding="utf-8")).get("train_wall_sec")
            if w:
                walls.append(w)
        except Exception:  # noqa: BLE001
            pass
    for logf in (RUNS / "p0_ckpts").glob("*/train.log"):
        try:
            last = ""
            for line in logf.read_text(encoding="utf-8").splitlines():
                if "saved" in line or "final" in line:
                    last = line
            m = re.search(r"wall=(\d+)", last) or re.search(r"elapsed=(\d+)", last)
            if m:
                walls.append(float(m.group(1)))
        except Exception:  # noqa: BLE001
            pass
    return sum(walls) / len(walls) if walls else 2700.0


def main() -> int:
    ap = argparse.ArgumentParser(description="V7-R+ ETA accounting")
    ap.add_argument("--start", action="store_true",
                    help="record the pipeline start time")
    args = ap.parse_args()
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / "state").mkdir(parents=True, exist_ok=True)
    if args.start or not START_FILE.exists():
        START_FILE.write_text(str(int(time.time())))
    start_ts = int(START_FILE.read_text().strip())
    elapsed = int(time.time()) - start_ts

    done_eval, done_train, done_misc = 0, 0, 0
    rem_eval, rem_train, rem_misc = 0, 0, 0
    per_item = []
    for rel, kind, default in PLAN:
        exists = (RUNS / rel).exists() or (ROOT / "runs" / "v7r" / rel).exists()
        if kind == "eval":
            if exists:
                done_eval += 1
            else:
                rem_eval += 1
        elif kind == "train":
            if exists:
                done_train += 1
            else:
                rem_train += 1
        else:
            if exists:
                done_misc += 1
            else:
                rem_misc += 1
        per_item.append({"item": rel, "kind": kind, "done": exists})
    # counted evals also include gate-skipped items (marked done for ETA purposes)
    for sk in RUNS.rglob("SKIPPED_BY_GATE.json"):
        rel_from_runs = str(sk.relative_to(RUNS)).replace("\\", "/")
        parent = str(sk.parent.relative_to(RUNS)).replace("\\", "/")
        if (parent + "/eval.json") in [p[0] for p in PLAN]:
            rem_eval = max(0, rem_eval - 1)

    mean_eval, mean_train = _mean_eval_wall(), _mean_train_wall()
    gpu_active = done_eval * mean_eval + done_train * mean_train
    eta_remaining = int(rem_eval * mean_eval + rem_train * mean_train
                        + rem_misc * 120)
    out = {
        "BUDGET_TOTAL_SEC": BUDGET_TOTAL_SEC,
        "ELAPSED_SEC": elapsed,
        "GPU_ACTIVE_SEC_EST": int(gpu_active),
        "ETA_REMAINING_SEC": eta_remaining,
        "BUDGET_REMAINING_SEC": max(0, BUDGET_TOTAL_SEC - elapsed),
        "measured_mean_eval750_sec": round(mean_eval, 1),
        "measured_mean_train_sec": round(mean_train, 1),
        "counts": {"eval_done": done_eval, "eval_remaining": rem_eval,
                   "train_done": done_train, "train_remaining": rem_train,
                   "misc_done": done_misc, "misc_remaining": rem_misc},
        "note": "ETA is a work-based estimate from measured throughput; "
                "not a promise. Never wait to fill the budget.",
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(out, RUNS / "eta.json")
    print(f"[eta] elapsed={elapsed}s gpu_active≈{int(gpu_active)}s "
          f"eta_remaining={eta_remaining}s "
          f"budget_remaining={out['BUDGET_REMAINING_SEC']}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
