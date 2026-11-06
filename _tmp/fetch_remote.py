"""Fetch read-only result files from 3060 into _tmp/remote for local verification."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

OUT = Path(r"M:\WEAVE\_tmp\remote")
OUT.mkdir(parents=True, exist_ok=True)
BASE = "/mnt/i/Latent_Style/"

FILES = [
    "WEAVE/runs/p1/auroc_csd_dm.json",
    "WEAVE/runs/p1/auroc_dm.json",
    "WEAVE/runs/p2/pgt_p0_audit.json",
    "WEAVE/runs/p2/pgt_selection.json",
    "WEAVE/runs/p2/pgt_q0.json",
    "WEAVE/runs/p2/p5_styleid_sensitivity.json",
    "WEAVE/runs/p2/v4_pattern_report.json",
    "WEAVE/runs/p3/p3_p4.json",
    "WEAVE/runs/v4/batch_bench.txt",
    "WEAVE/runs/v4/batch_bench2.txt",
    "WEAVE/runs/v4/train_bench.txt",
    "WEAVE/runs/v4/b64_t8.eval750.log",
    "pseudo_gt_24h/p2_reach.json",
    "pseudo_gt_24h/p2_reach_e3e1.json",
    "pseudo_gt_24h/p2_reach_e8.json",
    "pseudo_gt_24h/eval_render/S1_A_P5_final/v4_eval750.json",
    "pseudo_gt_24h/eval_render/S2_E7_final/v4_eval750.json",
    "pseudo_gt_24h/bench_render/b64_t8/v4_eval750.json",
]


def main() -> int:
    rels = sys.argv[1:] or FILES
    for rel in rels:
        dst = OUT / rel.replace("/", "__")
        r = subprocess.run(
            ["ssh", "3060", f"wsl -e bash -c \"cat {BASE}{rel}\""],
            capture_output=True, text=True, encoding="utf-8",
        )
        if r.returncode != 0 or not r.stdout:
            print(f"FAIL {rel}: rc={r.returncode} {r.stderr[:120]}")
            continue
        dst.write_text(r.stdout, encoding="utf-8")
        print(f"OK {rel} -> {dst.name} ({len(r.stdout)}B)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
