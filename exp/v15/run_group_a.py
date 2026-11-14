"""V15 Work Group A sequencer: 16 arms × seed 0, sequential execution.

Run on the 3060 inside tmux:
  tmux new-session -d -s v15 'python -m exp.v15.run_group_a'
Each arm: 4k optimizer updates (eff. batch 128 = micro × accum; ~37 epochs —
loss knee at ~1-2k steps. 2026-11-15 decision: run 4k first, resume/extend
later if diagnostics show under-convergence).
Status: runs/v15/group_a_status.json, per-arm logs: runs/v15/logs/<tag>.log

Memory plan (no allocator flags, plain PyTorch): per-arm micro table sized
from smoke-matrix peaks; any residual OOM retries once with micro halved /
accum doubled (effective batch unchanged, protocol-safe).
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
TIMEOUT_S = 8 * 3600

ARMS_A = []
# Phase 1: plain (base) arms, largest first — fast sweep of capacity/context
for dim in (96, 64, 32, 16):
    for att in (False, True):
        ARMS_A.append({"base_dim": dim, "use_attention": att,
                       "use_where": False})
# Phase 2: where arms, largest first — the expensive batch comes second
for dim in (96, 64, 32, 16):
    for att in (False, True):
        ARMS_A.append({"base_dim": dim, "use_attention": att,
                       "use_where": True})

def arm_tag(cfg):
    att = "att" if cfg["use_attention"] else "conv"
    wh = "where" if cfg["use_where"] else "base"
    return f"W_dim{cfg['base_dim']}_{att}_{wh}"

def arm_micro(cfg):
    """micro/accum per arm; effective batch ALWAYS 128 (registered protocol).

    Sized from smoke-matrix peaks (per-sample activation GB ≈ (peak−1.4)/32),
    target ≤ ~8GB per micro-batch, micro must divide 128.
    """
    tag = arm_tag(cfg)
    if not cfg["use_where"]:
        return 128, 1          # all base arms: tiny activations
    if tag.startswith("W_dim16"):
        return 128, 1          # dim16 where ~52MB/sample -> ~6.7GB at 128
    if tag.startswith("W_dim32"):
        return 64, 2           # dim32 where ~81MB/sample
    if tag.startswith("W_dim64"):
        return 32, 4           # dim64 where ~146-183MB/sample
    if tag == "W_dim96_conv_where":
        return 32, 4           # ~242MB/sample -> ~8GB at micro 32
    return 16, 8               # dim96_att_where ~290MB/sample

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
        micro, accum = arm_micro(cfg)
        status, micro_used = "done", micro
        for attempt in range(2):
            micro_used = micro
            args = arm_cmd(cfg, out_dir, 4000, 0, micro, accum)
            try:
                with open(log_path, "w") as lf:
                    r = subprocess.run(args, cwd=str(REPO), timeout=TIMEOUT_S,
                                       stdout=lf, stderr=subprocess.STDOUT)
                if r.returncode == 0:
                    status = "done"
                    break
                status = f"error:{r.returncode}"
            except subprocess.TimeoutExpired:
                status = "timeout"
                break
            if attempt == 0 and "OutOfMemoryError" in log_path.read_text(errors="replace"):
                micro, accum = micro // 2, accum * 2   # eff. batch 128 kept
                print(f"  OOM at micro={micro*2}, retrying micro={micro} "
                      f"accum={accum}", flush=True)
                continue
            break
        results[tag] = {"status": status,
                        "micro": micro_used if status == "done" else None,
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
