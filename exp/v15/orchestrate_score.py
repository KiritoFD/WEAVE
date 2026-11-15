"""Orchestrate v2: wait for 3060 eval -> launch remote DINO scoring -> pull
JSONs only (KBs, no image bulk).  PNG/payload download stays on-demand.

Run locally: python -m exp.v15.orchestrate_score
Progress: G:/WEAVE/runs/v15/orchestrate.log
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

REPO = Path("G:/WEAVE")
RUNS = REPO / "runs/v15"
LOG = RUNS / "orchestrate.log"
LOG_FH = open(LOG, "a", encoding="utf-8")


def log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    LOG_FH.write(line + "\n")
    LOG_FH.flush()
    print(line, flush=True)


def ssh(cmd: str, timeout: int = 120) -> str:
    return subprocess.run(f'ssh -o BatchMode=yes 3060 "{cmd}"', shell=True,
                          capture_output=True, text=True,
                          timeout=timeout).stdout


def main():
    log("=== orchestrate v2 start ===")
    for _ in range(48):
        out = ssh("type I:/Latent_Style/WEAVE/runs/v15/eval_run.log")
        if "EVAL_ALL_DONE" in out:
            log("eval complete on 3060")
            break
        log(f"eval pending (tail ...{out.strip()[-50:]})")
        time.sleep(300)
    else:
        log("TIMEOUT waiting for eval; aborting")
        return

    log("launch remote scoring")
    log(ssh("taskkill /f /im python.exe 2>nul & "
            "tmux kill-session -t score15 2>nul & "
            "tmux new-session -d -s score15 cmd /c "
            "I:\\Latent_Style\\WEAVE\\_tmp\\run_v15_score.bat"))

    for _ in range(30):
        time.sleep(120)
        out = ssh("type I:/Latent_Style/WEAVE/runs/v15/score_run.log")
        if "EVAL_SCORE_DONE" in out:
            log("scoring complete")
            break
        log(f"scoring pending (tail ...{out.strip()[-50:]})")
    else:
        log("TIMEOUT waiting for scoring; aborting")
        return

    t0 = time.time()
    r = subprocess.run(
        'scp -q "3060:I:/Latent_Style/WEAVE/runs/v15/eval/*.json" '
        f'"{RUNS / "eval"}"', shell=True, capture_output=True, text=True)
    n = len(list((RUNS / "eval").glob("*.json")))
    log(f"pulled {n} jsons rc={r.returncode} {time.time()-t0:.0f}s")
    log("=== orchestrate v2 done ===")


if __name__ == "__main__":
    main()
