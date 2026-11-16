"""1h-cadence V17 queue monitor (runs for the whole 36h).

Each cycle: read queue_state.json + per-arm progress, log transitions and
failures, pull newly landed measured eval/summary JSONs, rebuild the
performance ledger (docs/performance).  Never touches the experiment.
Run locally: python -m exp.v17.monitor_loop
Log: G:/WEAVE/runs/v17/monitor_1h.log
"""
from __future__ import annotations

import datetime
import json
import subprocess
import time
from pathlib import Path

REPO = Path("G:/WEAVE")
LOG = REPO / "runs/v17/monitor_1h.log"
LOG.parent.mkdir(parents=True, exist_ok=True)
LOG_FH = open(LOG, "a", encoding="utf-8")


def log(msg: str):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    LOG_FH.write(line + "\n")
    LOG_FH.flush()
    print(line, flush=True)


def ssh(cmd: str, timeout: int = 120) -> str:
    return subprocess.run(f'ssh -o BatchMode=yes 3060 "{cmd}"', shell=True,
                          capture_output=True, text=True, timeout=timeout,
                          encoding='utf-8', errors='replace').stdout


def pull_state() -> dict | None:
    out = ssh("type I:/Latent_Style/WEAVE/runs/v17/queue_state.json")
    try:
        return json.loads(out)
    except Exception:
        return None


def pull_measured():
    subprocess.run(
        f'scp -q -r "3060:I:/Latent_Style/WEAVE/runs/v17/measured" '
        f'"{REPO}/runs/v17/"', shell=True, capture_output=True, text=True,
        encoding='utf-8', errors='replace')


def rebuild_metrics():
    subprocess.run([sys_executable(), 'docs/performance/build_metrics.py'],
                   cwd=REPO, capture_output=True, text=True)


def sys_executable():
    import sys
    return sys.executable


def main():
    log("=== 1h monitor start ===")
    prev_states: dict[str, str] = {}
    cycles = 0
    while True:
        try:
            d = pull_state()
            if d is None:
                log("queue_state.json unreadable/missing")
            else:
                valid = d.get('valid_task_minutes', 0) / 60
                log(f"queue: valid={valid:.2f}h / target 36h")
                for arm, v in sorted(d.get('arms', {}).items()):
                    st = v.get('state', '?')
                    if st != prev_states.get(arm):
                        mark = '!' if st in ('failed', 'timeout') else '+'
                        log(f"  {mark} {arm}: {st} "
                            f"({v.get('minutes', 0):.1f}min "
                            f"retries={v.get('retries', 0)})")
                        prev_states[arm] = st
            pull_measured()
            rebuild_metrics()
            log("ledger rebuilt (docs/performance)")
        except Exception as e:
            log(f"cycle error: {type(e).__name__}: {e}")
        cycles += 1
        time.sleep(3600)


if __name__ == '__main__':
    main()
