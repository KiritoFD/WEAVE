"""Stage-aware V5 monitor (no GPU work, no duplicate experiments).

Watches `runs/v5/PROGRESS.md` and the active stage log on 3060, reports step
ETA and stage transitions, then fetches artifacts once the serial pipeline
finishes.  Safe to run in the background across goal rounds.
"""
from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REMOTE = "3060"
PROGRESS = "/mnt/i/Latent_Style/WEAVE/runs/v5/PROGRESS.md"
LOG_DIR = "/mnt/i/Latent_Style/WEAVE/runs/v5/route_c/logs"
FINAL_REPORT = "/mnt/i/Latent_Style/WEAVE/docs/v5/V5_116PLAN_EXECUTION_REPORT.md"
DEADLINE_SEC = 8 * 3600
POLL_SEC = 300

STAGE_LOG = {
    "C1_train_full": "train_full.log",
    "C2_noise_check": "check.log",
    "C2b_train_lowmid": "train_lowmid.log",
    "C2c_lowmid_check": "check_lowmid.log",
    "C3_quick_sweep": "quick.log",
    "C4_final_750": "final.log",
    "C5_summary": "summary.log",
}


def ssh_wsl(args, timeout=45):
    try:
        r = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=15", "-o", "BatchMode=yes", REMOTE, "wsl", *args],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout,
        )
        return r.stdout if r.returncode == 0 else ""
    except subprocess.TimeoutExpired:
        return ""


def current_stage(progress: str) -> str:
    stages = []
    for line in progress.splitlines():
        m = re.search(r"(C[1-5][a-z]?_[A-Za-z0-9_]+|V5 ROUTE C ALL DONE) (START|END)", line)
        if m:
            stages.append((m.group(1), m.group(2)))
    for name, state in reversed(stages):
        if state == "START":
            return name
    return "unknown"


def parse_eta(log: str):
    pairs = re.findall(r"step (\d+)/(\d+)", log)
    elapsed = [int(x) for x in re.findall(r"elapsed=(\d+)s", log)]
    if pairs and elapsed:
        cur, total = int(pairs[-1][0]), int(pairs[-1][1])
        rate = elapsed[-1] / max(cur, 1)
        return cur, total, round((total - cur) * rate / 60, 1)
    return 0, 0, None


def fetch(rel: str, dst: Path):
    data = ssh_wsl(["cat", rel])
    if data:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(data, encoding="utf-8")
        return len(data)
    return 0


def main() -> int:
    end = time.time() + DEADLINE_SEC
    last_progress = ""
    last_step = -1
    stages_seen = set()
    while time.time() < end:
        progress = ssh_wsl(["tail", "-n", "50", PROGRESS])
        stage = current_stage(progress)
        log_name = STAGE_LOG.get(stage, "train_full.log")
        active_log = ssh_wsl(["tail", "-n", "12", f"{LOG_DIR}/{log_name}"])
        cur, total, eta_min = parse_eta(active_log)
        if cur != last_step or progress != last_progress:
            stamp = time.strftime("%F %T")
            print(f"[{stamp}] active={stage} log={log_name} step={cur}/{total} ETA~{eta_min}min", flush=True)
            print(progress.strip(), flush=True)
            last_step, last_progress = cur, progress
        for line in progress.splitlines():
            for name in (*STAGE_LOG, "V5 ROUTE C ALL DONE"):
                if name in line and name not in stages_seen:
                    stages_seen.add(name)
                    print(f"[stage] {line.strip()}", flush=True)
        last_line = progress.strip().splitlines()[-1] if progress.strip() else ""
        if "V5 ROUTE C ALL DONE" in progress:
            print("[monitor] pipeline DONE", flush=True)
            break
        if re.search(r"(FATAL|FAILED)$", last_line):
            print(f"[monitor] pipeline FAILED: {last_line}", flush=True)
            break
        time.sleep(POLL_SEC)
    else:
        print("[monitor] deadline reached without DONE", flush=True)

    # Always fetch whatever exists; the final summary overwrites the partial report.
    fetcher = ROOT / "_tmp" / "fetch_v5_results.py"
    subprocess.run([sys.executable, str(fetcher)], check=False)
    n = fetch(FINAL_REPORT, ROOT / "docs" / "v5" / "V5_116PLAN_EXECUTION_REPORT.md")
    if n:
        print(f"[monitor] fetched final report ({n}B)", flush=True)
    print(f"[monitor] finished stages={sorted(stages_seen)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
