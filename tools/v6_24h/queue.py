"""V6 24h plan - serial queue with an exclusive GPU lock.

Guarantees required by plan section 3:
  * strictly one GPU training process (exclusive lock + pre-flight GPU check);
  * per-arm timeout at 1.5x the calibrated budget, marked `incomplete`, never silently
    shortened;
  * state.json written after every transition, so progress survives an SSH drop;
  * full config JSON, git SHA and timings recorded per arm.

Usage
    python3 -m tools.v6_24h.queue --list
    python3 -m tools.v6_24h.queue --run --max-hours 24
    python3 -m tools.v6_24h.queue --status
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.v6_24h.arms import merged, ordered, to_cli  # noqa: E402

CFG_DIR = ROOT / "configs" / "v6_24h"
RUN_DIR = ROOT / "runs" / "v6_24h"
LOG_DIR = RUN_DIR / "train_logs"
STATE = RUN_DIR / "state.json"
LOCK = RUN_DIR / ".gpu.lock"
BUDGET_MIN = 42.0          # calibrated from historical arms
TIMEOUT_MIN = BUDGET_MIN * 1.5


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT),
                              capture_output=True, text=True).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def gpu_busy() -> tuple:
    """Return (busy, description). Blocks a second training process."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                            "--format=csv,noheader"], capture_output=True, text=True)
        lines = [x.strip() for x in r.stdout.splitlines() if x.strip()]
        return bool(lines), "; ".join(lines)[:200]
    except Exception as e:  # noqa: BLE001
        return False, f"nvidia-smi unavailable: {e}"


class Lock:
    """Exclusive lock via O_EXCL, with stale-pid recovery."""

    def __init__(self, path: Path):
        self.path = path

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            try:
                pid = int(self.path.read_text().split()[0])
                os.kill(pid, 0)
                raise RuntimeError(f"queue already running (pid {pid}); "
                                   f"remove {self.path} if stale")
            except (ValueError, ProcessLookupError, PermissionError):
                self.path.unlink(missing_ok=True)
        fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{os.getpid()}\n".encode())
        os.close(fd)
        return self

    def __exit__(self, *a):
        self.path.unlink(missing_ok=True)


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {}


def save_state(s: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")


def write_config(aid: str, cfg: dict) -> Path:
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    p = CFG_DIR / f"{aid}.json"
    p.write_text(json.dumps(cfg, ensure_ascii=False, indent=2, sort_keys=True),
                 encoding="utf-8")
    return p


def cfg_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def run_arm(arm: dict, sha: str, timeout_s: int) -> dict:
    aid = arm["id"]
    cfg = merged(arm)
    cp = write_config(aid, cfg)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = LOG_DIR / f"{aid}.log"
    cmd = [sys.executable, "-u", "tools/v6/train_eval_v6.py", "--name", aid,
           *to_cli(cfg)]
    rec = {"id": aid, "group": arm["group"], "note": arm["note"],
           "config": cfg, "config_hash": cfg_hash(cfg), "git_sha": sha,
           "config_path": str(cp.relative_to(ROOT)), "cmd": " ".join(cmd),
           "status": "running", "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    t0 = time.time()
    with log.open("a", encoding="utf-8") as f:
        f.write("\n>>> " + " ".join(cmd) + "\n")
        f.flush()
        try:
            r = subprocess.run(cmd, cwd=str(ROOT), stdout=f,
                               stderr=subprocess.STDOUT, timeout=timeout_s)
            rec["returncode"] = r.returncode
            rec["status"] = "done" if r.returncode == 0 else "failed"
        except subprocess.TimeoutExpired:
            rec["status"] = "incomplete"
            rec["error"] = f"timeout at {timeout_s/60:.0f} min (1.5x budget); " \
                           f"exposure NOT shortened"
    rec["wall_sec"] = round(time.time() - t0, 1)
    rp = ROOT / "runs" / "v6" / aid / "report.json"
    if rp.exists():
        try:
            r = json.loads(rp.read_text())
            e = r.get("eval200") or {}
            rec["eval200"] = {k: e.get(k) for k in
                              ("mean_dT", "mean_dM", "mean_dino_c", "mean_lpips",
                               "frac_dT_positive", "styles_improved", "ukiyo_dT",
                               "same50_mean_dS")}
            rec["per_target_dT"] = e.get("per_target_dT")
            rec["n_params"] = r.get("n_params")
            rec["grad_diag"] = r.get("grad_diag")
            rec["train_wall_sec"] = r.get("train_wall_sec")
            rec["peak_vram_mib"] = r.get("peak_vram_mib")
        except Exception as ex:  # noqa: BLE001
            rec["report_error"] = str(ex)
    return rec


def print_status(s: dict) -> None:
    if not s:
        print("no state yet")
        return
    print(f"{'arm':16s} {'status':11s} {'dT':>9s} {'dino':>7s} {'lpips':>7s} "
          f"{'wall_min':>9s}")
    for aid, r in s.items():
        e = r.get("eval200") or {}
        dT = f"{e['mean_dT']:+.4f}" if e.get("mean_dT") is not None else "—"
        dn = f"{e['mean_dino_c']:.4f}" if e.get("mean_dino_c") is not None else "—"
        lp = f"{e['mean_lpips']:.4f}" if e.get("mean_lpips") is not None else "—"
        wm = f"{r.get('wall_sec', 0)/60:.1f}" if r.get("wall_sec") else "—"
        print(f"{aid:16s} {r.get('status', '?'):11s} {dT:>9s} {dn:>7s} {lp:>7s} "
              f"{wm:>9s}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--arms", default="", help="comma-separated subset of ids")
    ap.add_argument("--max-hours", type=float, default=24.0)
    ap.add_argument("--budget-min", type=float, default=BUDGET_MIN)
    ap.add_argument("--only-smoke", action="store_true")
    args = ap.parse_args()

    ids = [x.strip() for x in args.arms.split(",") if x.strip()] if args.arms else None
    seq = ordered(ids)

    if args.list:
        print(f"{'arm':16s} {'group':6s} {'note'}")
        for a in seq:
            print(f"{a['id']:16s} {a['group']:6s} {a['note']}")
        print(f"\n{len(seq)} arms, {args.budget_min:.0f} min each "
              f"= {len(seq)*args.budget_min/60:.1f} h")
        return
    if args.status:
        print_status(load_state())
        return

    if args.only_smoke:
        sys.exit(subprocess.run([sys.executable, "-m", "tools.v6_24h.smoke16"],
                                cwd=str(ROOT)).returncode)

    if not args.run:
        ap.print_help()
        return

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    sha = git_sha()
    deadline = time.monotonic() + args.max_hours * 3600
    timeout_s = int(args.budget_min * 1.5 * 60)

    with Lock(LOCK):
        state = load_state()
        for arm in seq:
            if time.monotonic() > deadline:
                print(f"[queue] wall-clock budget reached, stopping before {arm['id']}")
                break
            if state.get(arm["id"], {}).get("status") == "done":
                print(f"[queue] {arm['id']} already done, skip")
                continue
            busy, desc = gpu_busy()
            if busy:
                print(f"[queue] GPU busy, refusing to start {arm['id']}: {desc}")
                state[arm["id"]] = {"id": arm["id"], "status": "blocked_gpu",
                                    "gpu": desc}
                save_state(state)
                break
            print(f"[queue] === {arm['id']} ({arm['group']}) {arm['note']}", flush=True)
            # Write state BEFORE the arm starts so a crash still leaves a record of
            # which arm was in flight (plan section 3: state must be live, not polled).
            state[arm["id"]] = {"id": arm["id"], "group": arm["group"],
                                "note": arm["note"], "config": merged(arm),
                                "config_hash": cfg_hash(merged(arm)), "git_sha": sha,
                                "status": "running",
                                "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
            save_state(state)
            rec = run_arm(arm, sha, timeout_s)
            state[arm["id"]] = rec
            save_state(state)
            e = rec.get("eval200") or {}
            print(f"[queue] {arm['id']} {rec['status']} "
                  f"dT={e.get('mean_dT')} dino={e.get('mean_dino_c')} "
                  f"wall={rec.get('wall_sec', 0)/60:.1f}min", flush=True)
        print_status(load_state())


if __name__ == "__main__":
    main()
