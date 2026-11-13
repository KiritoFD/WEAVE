#!/usr/bin/env bash
set -euo pipefail
cd /mnt/i/Latent_Style/WEAVE
tar czf - \
  docs/v6/R2_12H_REPORT.md \
  runs/v6_r2/final_verdict.json \
  runs/v6_r2/provenance.txt \
  runs/v6_r2/final_paired_report.txt \
  runs/v6_r2/dino_s_control.txt \
  runs/v6_r2/queue_r2_12h.log \
  runs/v6_r2/queue_p0.log runs/v6_r2/queue_p1.log runs/v6_r2/queue_p2.log \
  runs/v6_r2/queue_p3.log runs/v6_r2/queue_p3diag.log runs/v6_r2/queue_p4.log \
  runs/v6_r2/state/ \
  runs/v6_r2/fitting_diagnostics/ \
  runs/v6_r2/visual_grids/ \
  runs/v6_r2/eval_S1/eval.json runs/v6_r2/eval_S2/eval.json \
  runs/v6_r2/eval_S3/eval.json runs/v6_r2/eval_S4/eval.json \
  runs/v6_r2/eval_P1_B2c_ctrl845/eval.json \
  2>/dev/null | base64 -w0
