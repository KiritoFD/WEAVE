"""Serial recovery of V17 eval arms that died from the zombie-process OOM.

This script runs ONE process at a time on the 3060 (no concurrency).  It is
meant to run AFTER the main queue has left the P2/P3 training phases, or in a
GPU idle window.  Polls GPU memory; only launches the next arm when < 1GB
used; waits for that arm to fully exit before launching the next.

Run locally: python -m exp.v17.recover_evals
Log: G:/WEAVE/runs/v17/recover_evals.log
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

REPO = Path("G:/WEAVE")
LOG = REPO / "runs/v17/recover_evals.log"
LOG.parent.mkdir(parents=True, exist_ok=True)
LOG_FH = open(LOG, "a", encoding="utf-8")
ARMS = ["P2_S1_eval", "P2_S2_eval", "P2_S3_eval", "P2_S4_eval",
        "P2_S5_eval", "P2_S6_eval", "P2_S8_eval", "P2_S7_train",
        "P2_S7_eval", "P3_H2arm_eval"]


def log(msg):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    LOG_FH.write(line + "\n")
    LOG_FH.flush()
    print(line, flush=True)


def ssh(cmd, timeout=120):
    return subprocess.run(f'ssh -o BatchMode=yes 3060 "{cmd}"', shell=True,
                          capture_output=True, text=True, timeout=timeout,
                          encoding='utf-8', errors='replace').stdout


def gpu_used_gb():
    out = ssh("nvidia-smi --query-gpu=memory.used --format=csv,noheader")
    try:
        return int(out.strip().splitlines()[0]) / 1024
    except Exception:
        return 99.


def arm_done(arm):
    """Check via measured/<arm>/summary.json presence."""
    out = ssh('powershell -NoProfile -Command "Test-Path '
              'I:/Latent_Style/WEAVE/runs/v17/measured/' + arm + '/summary.json"')
    return "True" in out


def main():
    log("=== recover_evals start ===")
    for arm in ARMS:
        if arm_done(arm):
            log(f"{arm}: already measured, skip")
            continue
        for _ in range(180):                 # wait up to 3h for GPU idle
            g = gpu_used_gb()
            if g < 1.0:
                break
            log(f"{arm}: GPU busy {g:.1f}GB, waiting 60s")
            time.sleep(60)
        else:
            log(f"{arm}: GPU never idle, ABORT")
            return
        log(f"{arm}: launching single process")
        cmd = (f'cd /d I:\\Latent_Style\\WEAVE && python -m exp.v17.run '
               f'--config runs/v17/{arm}/config_resolved.json --seed 0 '
               f'--out runs/v17/{arm} --resume auto')
        # run synchronously over ssh; blocks until arm finishes
        out = ssh(cmd, timeout=7200)
        tail = [l for l in out.splitlines()
                if 'ARM_DONE' in l or 'Error' in l or 'Traceback' in l]
        log(f"{arm}: finished -> {tail[-1] if tail else '(no marker)'}")
    log("=== recover_evals done ===")


if __name__ == "__main__":
    main()
