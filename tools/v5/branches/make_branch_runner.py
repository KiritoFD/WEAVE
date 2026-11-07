"""Build the M1/M2/B2/D branch shell driver for the pruned decision-tree paths.

This driver is NOT part of the currently running Route C pipeline.  It exists
so every branch in 116plan is runnable with the same serial / one-GPU policy.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "tools" / "v5" / "branches" / "run_pruned_branches.sh"
BODY = """#!/usr/bin/env bash
# Pruned V5 branches (only run if an earlier decision-tree gate routes here).
set -u
cd /mnt/i/Latent_Style/WEAVE || exit 1
OUT=runs/v5/branches
LOG=$OUT/logs
mkdir -p "$OUT" "$LOG"
PROG=runs/v5/PROGRESS_BRANCHES.md
log() { echo "[$(date '+%F %T')] $*" | tee -a "$PROG"; }
run() { local n=$1 l=$2; shift 2; log "$n START"; "$@" > "$LOG/$l" 2>&1; local r=$?; log "$n END rc=$r"; if [ $r -ne 0 ]; then tail -n 20 "$LOG/$l" | tee -a "$PROG"; fi; return $r; }

# Route A: M1 direct operator vs M2 unpaired endpoint flow, same supervision.
run "M1_descriptor" m1_descriptor.log python3 -u tools/v5/branches/route_a.py train \\
  --name M1_descriptor --mode m1 --objective descriptor --structure selfsim \\
  --steps 1200 --microbatch 128 --accum 2 --ode-infer-steps 1
run "M2_descriptor" m2_descriptor.log python3 -u tools/v5/branches/route_a.py train \\
  --name M2_descriptor --mode m2 --objective descriptor --structure selfsim \\
  --steps 800 --microbatch 128 --accum 2 --ode-steps 2 --ode-infer-steps 4
run "M1_encoder" m1_encoder.log python3 -u tools/v5/branches/route_a.py train \\
  --name M1_encoder --mode m1 --objective encoder --structure selfsim \\
  --steps 1200 --microbatch 128 --accum 2 --ode-infer-steps 1

# B2: content-constraint Pareto frontier.
run "B2_pareto" b2.log python3 -u tools/v5/branches/route_a.py b2

# Third layer: Oracle-solution distillation into M1.
run "D_distill" distill.log python3 -u tools/v5/branches/route_a.py distill \\
  --n-pairs 1000 --steps 800 --microbatch 128

log "PRUNED BRANCHES ALL DONE"
"""
OUT.write_bytes(BODY.replace("\r\n", "\n").encode("utf-8"))
print(f"wrote {OUT}")
