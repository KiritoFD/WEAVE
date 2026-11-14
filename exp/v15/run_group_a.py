"""V15 Work Group A sequencer: 16 arms × seed 0, sequential execution.

Run on the 3060: python -m exp.v15.run_group_a
Each arm: 12k optimizer updates. Status: runs/v15/group_a_status.json,
per-arm stdout/stderr: runs/v15/logs/<tag>.log
OOM safety net: expandable_segments allocator + on OutOfMemoryError retry the
arm with micro halved / accum doubled (effective batch 128 unchanged).
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("I:/Latent_Style/WEAVE")
POOL = "I:/Latent_Style/pseudo_gt_24h/orig_mean_fp16"
OUT = REPO / "runs/v15"
LOGS = OUT / "logs"
PYTHON = sys.executable
TIMEOUT_S = 6 * 3600
ENV = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}

ARMS_A = []
for dim in (16, 32, 64, 96):
    for att in (False, True):
        for where in (False, True):
            ARMS_A.append({
                "base_dim": dim,
                "use_attention": att,
                "use_where": where,
            })

def arm_tag(cfg):
    att = "att" if cfg["use_attention"] else "conv"
    wh = "where" if cfg["use_where"] else "base"
    return f"W_dim{cfg['base_dim']}_{att}_{wh}"

def arm_cmd(cfg, out_dir, updates, seed, micro, accum):
    args = [
        PYTHON, "-m", "exp.v15.train",
        "--base-dim", str(cfg["base_dim"]),
        "--pool", POOL,
        "--out", str(out_dir),
        "--updates", str(updates),
        "--seed", str(seed),
        "--micro", str(micro),
        "--accum", str(accum),
    ]
    if cfg["use_attention"]:
        args.append("--use-attention")
    if cfg["use_where"]:
        args.append("--use-where")
    return args

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    status_file = OUT / "group_a_status.json"
    results = {}
    if status_file.exists():
        results = json.loads(status_file.read_text())
    t0 = time.time()
    n_done = 0
    for i, cfg in enumerate(ARMS_A):
        tag = arm_tag(cfg)
        if results.get(tag, {}).get("status") == "done":
            print(f"[{i+1}/16] {tag} already done, skip", flush=True)
            n_done += 1
            continue
        print(f"[{i+1}/16] starting {tag}...", flush=True)
        out_dir = OUT / tag
        log_path = LOGS / f"{tag}.log"
        t_arm = time.time()
        status, micro_used = "done", 32
        for attempt, (micro, accum) in enumerate(((32, 4), (16, 8), (8, 16))):
            micro_used = micro
            args = arm_cmd(cfg, out_dir, 12000, 0, micro, accum)
            try:
                with open(log_path, "w") as lf:
                    r = subprocess.run(args, cwd=str(REPO), timeout=TIMEOUT_S,
                                       stdout=lf, stderr=subprocess.STDOUT,
                                       env=ENV)
                if r.returncode == 0:
                    status = "done"
                    break
                status = f"error:{r.returncode}"
            except subprocess.TimeoutExpired:
                status = "timeout"
                break
            tail = log_path.read_text(errors="replace")
            if "OutOfMemoryError" not in tail and attempt < 2:
                break  # non-OOM failure: retrying won't help
            print(f"  OOM at micro={micro}, retrying with {micro//2}/"
                  f"{accum*2}", flush=True)
        results[tag] = {"status": status, "micro": micro_used if status == "done" else None,
                        "elapsed_min": round((time.time() - t_arm) / 60, 1)}
        status_file.write_text(json.dumps(results, indent=1))
        if status != "done":
            tail = log_path.read_text(errors="replace")[-400:]
            print(f"  LOG TAIL: {tail}", flush=True)
        print(f"[{i+1}/16] {tag} {status} "
              f"({results[tag]['elapsed_min']}min) "
              f"total {time.time()-t0:.0f}s", flush=True)
    print(f"GROUP_A_COMPLETE done_this_run={n_done}/16", flush=True)

if __name__ == "__main__":
    main()
