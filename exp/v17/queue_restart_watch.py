"""Restart the V17 queue once P3_H5 training finishes, so the 10 reset
(PENDING) arms are re-run in manifest order before P4/P5.

Run locally: python -m exp.v17.queue_restart_watch
Log: G:/WEAVE/runs/v17/restart_watch.log
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

REPO = Path("G:/WEAVE")
LOG = REPO / "runs/v17/restart_watch.log"
LOG.parent.mkdir(parents=True, exist_ok=True)
LOG_FH = open(LOG, "a", encoding="utf-8")


def log(m):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {m}"
    LOG_FH.write(line + "\n")
    LOG_FH.flush()
    print(line, flush=True)


def ssh(cmd, timeout=120):
    return subprocess.run(f'ssh -o BatchMode=yes 3060 "{cmd}"', shell=True,
                          capture_output=True, text=True, timeout=timeout,
                          encoding='utf-8', errors='replace').stdout


def main():
    log("=== restart watch start ===")
    for i in range(180):                      # up to 90min (H4 ~60min + H5)
        out = ssh('powershell -NoProfile -Command "Get-Content '
                  'I:/Latent_Style/WEAVE/runs/v17/logs/queue.log '
                  '-Tail 400"')
        done = [l for l in out.splitlines()
                if 'END P3_H5_' in l or 'END P3_H6_' in l or 'END P3_H7_' in l]
        if done:
            log("P3 training phase complete: " + done[-1].strip())
            break
        log(f"waiting ({i+1}/180) for H5/H6/H7 completion")
        time.sleep(30)
    else:
        log("TIMEOUT waiting; aborting")
        return
    log("restarting queue (kills current, state preserved, PENDING arms rerun)")
    log(ssh("taskkill /f /im python.exe 2>nul & "
            "tmux kill-session -t weave-v17 2>nul & "
            "ping -n 5 127.0.0.1 >nul & "
            "tmux new-session -d -s weave-v17 cmd /c "
            "I:\\Latent_Style\\WEAVE\\_tmp\\run_v17_queue.bat"))
    time.sleep(60)
    log(ssh('powershell -NoProfile -Command "Get-Content '
            'I:/Latent_Style/WEAVE/runs/v17/logs/queue.log -Tail 4"'))
    log("=== restart watch done ===")


if __name__ == "__main__":
    main()
