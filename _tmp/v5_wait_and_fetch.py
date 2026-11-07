"""Wait for the frozen Route C pipeline to finish, then fetch all results.

Long-running background monitor: checks every 5 minutes, no GPU work, no
duplicate experiments. Exits early on DONE or FATAL.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REMOTE = "3060"
WSL_PROGRESS = "/mnt/i/Latent_Style/WEAVE/runs/v5/PROGRESS.md"
DEADLINE_SEC = 5 * 3600
POLL_SEC = 300


def ssh_wsl_tail() -> str:
    r = subprocess.run(
        ["ssh", REMOTE, "wsl", "tail", "-n", "30", WSL_PROGRESS],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=60,
    )
    return r.stdout if r.returncode == 0 else ""


def main() -> int:
    end = time.time() + DEADLINE_SEC
    last = ""
    while time.time() < end:
        tail = ssh_wsl_tail()
        if tail != last:
            print(f"[{time.strftime('%F %T')}]\n{tail.strip()}", flush=True)
            last = tail
        if "V5 ROUTE C ALL DONE" in tail:
            print("[monitor] pipeline DONE", flush=True)
            break
        if any(k in tail for k in ("FATAL", "FAILED-FATAL")):
            print("[monitor] pipeline FAILED", flush=True)
            break
        time.sleep(POLL_SEC)
    else:
        print("[monitor] deadline reached", flush=True)

    fetch = ROOT / "_tmp" / "fetch_v5_results.py"
    subprocess.run([sys.executable, str(fetch)], check=False)

    # Fetch the generated Markdown report if present.
    rel = "docs/v5/V5_116PLAN_EXECUTION_REPORT.md"
    dst = ROOT / rel
    r = subprocess.run(
        ["ssh", REMOTE, "wsl", "cat", "/mnt/i/Latent_Style/WEAVE/" + rel],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=60,
    )
    if r.returncode == 0 and r.stdout:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(r.stdout, encoding="utf-8")
        print(f"[monitor] fetched {rel} ({len(r.stdout)}B)", flush=True)
    print("[monitor] finished", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
