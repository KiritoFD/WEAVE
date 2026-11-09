"""Backfill first-50-update timing and exposure fields from train logs.

The running suite40 process was launched before these fields were added to the
driver, so this fills them from each arm's own train log / train_summary without
touching the GPU process.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path("/mnt/i/Latent_Style/WEAVE")
SUITE = ROOT / "runs/blog_sbflow/suite40"
LOG = re.compile(r"step=(\d+)/(\d+).*?elapsed=(\d+)s")


def main() -> None:
    state_file = SUITE / "suite_state.json"
    if not state_file.exists():
        print("no suite_state.json")
        return
    state = json.loads(state_file.read_text())
    changed = 0
    for aid, rec in state.items():
        log = SUITE / "logs" / f"{aid}_train.log"
        if not log.exists():
            continue
        first50 = None
        for line in log.read_text(errors="ignore").splitlines():
            m = LOG.search(line)
            if m and int(m.group(1)) >= 50:
                first50 = int(m.group(3)) / 50
                break
        cfg = json.loads((ROOT / "configs/sbflow" / f"exp{aid}.json").read_text())
        exposures = cfg["steps"] * cfg["batch"] * 2
        old = (rec.get("first50_sec_per_update"), rec.get("exposures"))
        if first50 is not None:
            rec["first50_elapsed_sec"] = first50 * 50
            rec["first50_sec_per_update"] = first50
        rec["exposures"] = exposures
        rec.setdefault("per_pair_exposure", exposures // 10)
        if old != (rec.get("first50_sec_per_update"), rec["exposures"]):
            changed += 1
        print(f"  {aid}: first50={first50} s/upd exposures={exposures} per_pair={exposures//10}")
    if changed:
        state_file.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        from sbflow.suite40 import write_results
        write_results(state, SUITE)
        print(f"backfilled {changed} arm(s)")
    else:
        print("nothing to backfill")


if __name__ == "__main__":
    main()
