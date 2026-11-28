"""Serial GPU stage runner with a persistent absolute twelve-hour deadline."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from exp.v37_editing import OUT, atomic_json, read_manifest


def stage(name, module, args=(), budget=None, final=False):
    state_path = OUT / "stages" / f"{name}.json"
    if state_path.exists():
        old = json.loads(state_path.read_text())
        if old.get("status") == "complete":
            return
    m = read_manifest()
    limit = m["deadline_epoch"] if final else m["eval_reserved_from_epoch"]
    if budget:
        limit = min(limit, time.time() + budget)
    if time.time() >= limit:
        raise TimeoutError(f"No budget for {name}")
    log_path = OUT / "logs" / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-u", "-m", module, *map(str, args)]
    with log_path.open("ab", buffering=0) as log:
        child = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        state = {"stage": name, "status": "running", "pid": child.pid, "started": time.time(),
                 "deadline": limit, "command": cmd, "log": str(log_path)}
        atomic_json(state_path, state)
        try:
            rc = child.wait(timeout=max(1, limit - time.time() - 25))
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=max(1, limit - time.time()))
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            state.update(status="deadline", finished=time.time())
            atomic_json(state_path, state)
            raise TimeoutError(name)
    state.update(status="complete" if rc == 0 else "failed", rc=rc, finished=time.time())
    atomic_json(state_path, state)
    if rc:
        raise RuntimeError(f"{name} failed with exit {rc}; inspect {log_path}")


def p1():
    mod = "exp.v37_editing.evaluation.diagnostics"
    stage("P1_seen_generate", mod, ["seen"], 2400)
    stage("P1_seen_measure", mod, ["measure"], 1800)
    stage("P1_teacher_select", mod, ["select"], 60)
    stage("P1_core_quadrants", mod, ["core"], 2400)


def train_and_evaluate():
    train_mod = "exp.v37_editing.training.train"
    ev = "exp.v37_editing.evaluation.evaluate"
    assert (OUT / "pairs/approved.json").exists(), "Proxy examples need agent inspection before training"
    stage("P2_inverse_preflight", train_mod, ["--smoke-inverse"], 180)
    stage("P2_E0", train_mod, ["--route", "E0", "--budget-s", 9000, "--max-steps", 15000], 9060)
    stage("P3_pairs", "exp.v37_editing.training.pairs", budget=1800)
    e0_step = json.loads((OUT / "E0_state.json").read_text())["step"]
    pair_state = json.loads((OUT / "stages/P3_pairs.json").read_text())
    pair_seconds = pair_state["finished"] - pair_state["started"]
    e1_budget = max(60, int(10800 - pair_seconds))
    stage("P3_E1", train_mod, ["--route", "E1", "--budget-s", e1_budget, "--max-steps", e0_step], e1_budget + 60)
    stage("P3_seen_generate", ev, ["seen_generate"], 1800)
    stage("P3_seen_measure", ev, ["seen_measure"], 1800)
    stage("P4_decide", ev, ["decide"], 60)
    decision = json.loads((OUT / "p4_decision.json").read_text())["decision"]
    remaining = read_manifest()["eval_reserved_from_epoch"] - time.time()
    if decision == "A_extend_E1" and remaining > 120:
        budget = min(10800, int(remaining - 60))
        stage("P4_E1_extension", train_mod, ["--route", "E1", "--extension", "--budget-s", budget, "--max-steps", 30000], budget + 30)
        stage("P4_select_extended", ev, ["select_extended"], 60, final=True)
    final_evaluate()


def final_evaluate():
    ev = "exp.v37_editing.evaluation.evaluate"
    stage("P5_core750", ev, ["core"], final=True)
    stage("P5_holdout_generate", ev, ["holdout_generate"], final=True)
    stage("P5_holdout_measure", ev, ["holdout_measure"], final=True)
    stage("P5_report", "exp.v37_editing.report", final=True)
    atomic_json(OUT / "completion.json", {"status": "complete", "finished": time.time(), "deadline": read_manifest()["deadline_epoch"]})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("p1", "preview", "train", "final"))
    a = p.parse_args()
    if a.mode == "p1": p1()
    elif a.mode == "preview": stage("P3_fixed_preview", "exp.v37_editing.training.pairs", ["--preview"], 600)
    elif a.mode == "train": train_and_evaluate()
    else: final_evaluate()
