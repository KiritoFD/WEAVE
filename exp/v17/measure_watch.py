"""Wait for the main queue's current training arm to finish, then run the
batch referee (measure_only.py) on all raw payloads lacking a summary.

Run locally: python -m exp.v17.measure_watch
Log: G:/WEAVE/runs/v17/measure_watch.log
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

REPO = Path("G:/WEAVE")
LOG = REPO / "runs/v17/measure_watch.log"
LOG.parent.mkdir(parents=True, exist_ok=True)
LOG_FH = open(LOG, "a", encoding="utf-8")


def log(m):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {m}"
    LOG_FH.write(line + "\n"); LOG_FH.flush(); print(line, flush=True)


def ssh(cmd, timeout=120):
    return subprocess.run(f'ssh -o BatchMode=yes 3060 "{cmd}"', shell=True,
                          capture_output=True, text=True, timeout=timeout,
                          encoding='utf-8', errors='replace').stdout


def main():
    log("=== measure watch start ===")
    # wait until no exp.v17.run training process (queue between arms)
    for i in range(120):
        procs = ssh('wmic process where "name=\'python.exe\'" get '
                    'commandline /format:list 2>nul | findstr exp.v17.run')
        if 'exp.v17.run' not in procs:
            log("GPU idle from queue, launching batch measure")
            break
        log(f"queue busy ({i+1}/120), wait 60s")
        time.sleep(60)
    else:
        log("TIMEOUT; aborting")
        return
    log(ssh("tmux new-session -d -s batchmeas cmd /c "
            "I:\\Latent_Style\\WEAVE\\_tmp\\v17_batch_measure.bat"))
    for i in range(60):
        time.sleep(60)
        out = ssh('powershell -NoProfile -Command '
                  '"Get-Content I:/Latent_Style/WEAVE/runs/v17/'
                  'batch_measure.log -Tail 3"')
        if 'BATCH_MEASURE_ALL_DONE' in out:
            log("batch measure complete")
            log(out.strip()[-300:])
            break
    log("=== measure watch done ===")


if __name__ == "__main__":
    main()
