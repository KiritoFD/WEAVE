"""V15 Work Group A sequencer: 16 arms × seed 0, sequential execution.

Run on the 3060: python -m exp.v15.run_group_a
Each arm: 12k optimizer updates. Status: runs/v15/group_a_status.json,
per-arm stdout/stderr: runs/v15/logs/<tag>.log
"""
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("I:/Latent_Style/WEAVE")
POOL = "I:/Latent_Style/pseudo_gt_24h/orig_mean_fp16"
OUT = REPO / "runs/v15"
LOGS = OUT / "logs"
PYTHON = sys.executable
TIMEOUT_S = 4 * 3600  # generous: dim96+att+where arms are the slowest

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
        args = [
            PYTHON, "-m", "exp.v15.train",
            "--base-dim", str(cfg["base_dim"]),
            "--pool", POOL,
            "--out", str(OUT / tag),
            "--updates", "12000",
            "--seed", "0",
        ]
        if cfg["use_attention"]:
            args.append("--use-attention")
        if cfg["use_where"]:
            args.append("--use-where")
        t_arm = time.time()
        log_path = LOGS / f"{tag}.log"
        status = "done"
        try:
            with open(log_path, "w") as lf:
                r = subprocess.run(args, cwd=str(REPO), timeout=TIMEOUT_S,
                                   stdout=lf, stderr=subprocess.STDOUT)
            if r.returncode != 0:
                status = f"error:{r.returncode}"
                tail = log_path.read_text(errors="replace")[-400:]
                print(f"  LOG TAIL: {tail}", flush=True)
        except subprocess.TimeoutExpired:
            status = "timeout"
        results[tag] = {"status": status,
                        "elapsed_min": round((time.time() - t_arm) / 60, 1)}
        status_file.write_text(json.dumps(results, indent=1))
        print(f"[{i+1}/16] {tag} {status} "
              f"({results[tag]['elapsed_min']}min) "
              f"total {time.time()-t0:.0f}s", flush=True)
    print(f"GROUP_A_COMPLETE done_this_run={n_done}/16", flush=True)

if __name__ == "__main__":
    main()
