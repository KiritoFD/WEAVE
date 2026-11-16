"""Wait for scaling completion -> launch the V17 36h queue in tmux.

Run locally: python -m exp.v17.orchestrate_launch
Log: G:/WEAVE/runs/v17/orchestrate.log
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

REPO = Path("G:/WEAVE")
LOG = REPO / "runs/v17/orchestrate.log"
LOG.parent.mkdir(parents=True, exist_ok=True)
LOG_FH = open(LOG, "a", encoding="utf-8")


def log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    LOG_FH.write(line + "\n")
    LOG_FH.flush()
    print(line, flush=True)


def ssh(cmd: str, timeout: int = 60) -> str:
    return subprocess.run(f'ssh -o BatchMode=yes 3060 "{cmd}"', shell=True,
                          capture_output=True, text=True,
                          timeout=timeout).stdout


def main():
    log("=== v17 launch chain start ===")
    for _ in range(60):
        out = ssh("type I:/Latent_Style/WEAVE/runs/v15/scaling_status.json")
        try:
            d = json.loads(out)
        except Exception:
            d = {}
        if d.get("B8k_dim96", {}).get("status") == "done" or \
                "SCALING_COMPLETE" in out:
            log("scaling complete — all 8 arms done")
            break
        pending = [k for k, v in d.items() if v.get("status") != "done"]
        log(f"scaling pending: {pending}")
        time.sleep(600)
    else:
        log("TIMEOUT waiting for scaling; aborting")
        return
    log("launching V17 queue in tmux session weave-v17")
    log(ssh("tmux kill-session -t weave-v17 2>nul & "
            "tmux new-session -d -s weave-v17 cmd /c "
            "I:\\Latent_Style\\WEAVE\\_tmp\\run_v17_queue.bat"))
    for _ in range(20):
        time.sleep(60)
        out = ssh("dir I:/Latent_Style/WEAVE/runs/v17/queue_console.log")
        if "queue_console.log" in out:
            log("queue console log present — queue is running")
            break
    log("=== v17 launch chain done ===")


if __name__ == "__main__":
    main()
