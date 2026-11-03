"""One-off 2026-11-04 handoff helper.

Adds the missing 2026-11-04 rows to /mnt/i/Latent_Style/runs/registry.jsonl:
  - baseline_nohh600 / b160_retest  -> delegated to runs/append_registry.py
  - a1_32 / a1_128 / a1_512         -> appended here as aborted (user stop)

Idempotent: existing run_ids are skipped.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/mnt/i/Latent_Style")
REG = ROOT / "runs" / "registry.jsonl"


def existing_ids():
    ids = set()
    if REG.exists():
        for line in REG.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                ids.add(json.loads(line).get("run_id"))
            except json.JSONDecodeError:
                continue
    return ids


def delegate(run_id, eval_rel, note):
    cmd = [
        sys.executable, str(ROOT / "runs" / "append_registry.py"),
        "--run-dir", str(ROOT / "runs" / "p0_diag" / run_id),
        "--eval-dir", str(ROOT / "runs" / "p0_diag" / eval_rel),
        "--route", "A",
        "--valid", "yes",
        "--note", note,
    ]
    subprocess.run(cmd, check=True)


def aborted_row(run_id, steps_done, steps_total, batch, note):
    return {
        "ts": "2026-11-04T04:05:00",
        "run_id": run_id,
        "route": "A1",
        "config_hash": "",
        "seed": 0,
        "epochs": [],
        "batch": batch,
        "microbatch": None,
        "peak_gb": None,
        "vram_budget_batch": batch,
        "free_at_start_mib": None,
        "peak_vram_gb": None,
        "vram_guard_failed": False,
        "sps": None,
        "train_s": None,
        "metrics_600": None,
        "metrics_150": None,
        "gate": {},
        "early_stop_armed": False,
        "early_stop_record_only": True,
        "fired": False,
        "would_fire": False,
        "fired_epoch": None,
        "rho_series": None,
        "gate_mean_series": None,
        "stop_reason": "aborted_by_user",
        "steps_done": steps_done,
        "steps_total": steps_total,
        "valid": False,
        "invalid_reason": "aborted: user called stop training line; %d/%d steps, no eval" % (steps_done, steps_total),
        "note": note,
        "status": "aborted",
    }


def main():
    ids = existing_ids()
    print("existing rows:", len(ids))

    if "baseline_nohh600" not in ids:
        delegate(
            "baseline_nohh600",
            "baseline_nohh600_eval2/baseline_nohh600",
            "plan baseline A (hf_oriented_nohh, b96, lr2e-4, enable_hh_head=false); "
            "600 cross-style cluster, NOT comparable to paper 750 all-pairs",
        )
    if "b160_retest" not in ids:
        delegate(
            "b160_retest",
            "b160_retest_eval_missing",
            "b160 controlled retest: 496 steps (16ep x 31), peak 10.38G, no OOM; "
            "throughput/VRAM check, no eval by design",
        )

    a1_rows = [
        aborted_row("a1_32", 2000, 2000, 32,
                    "A1 saturation diag arm train_sample_limit=32; finished 2000/2000 steps but series "
                    "aborted by user before eval"),
        aborted_row("a1_128", 0, 2000, 32,
                    "A1 saturation diag arm train_sample_limit=128; killed right after start by user stop"),
        aborted_row("a1_512", 4, 2000, 32,
                    "A1 saturation diag arm train_sample_limit=512; killed at step 4 by user stop"),
    ]
    with REG.open("a") as fh:
        for row in a1_rows:
            if row["run_id"] in ids:
                continue
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            print("APPENDED", row["run_id"])

    print("total rows now:", len(existing_ids()))


if __name__ == "__main__":
    main()
