#!/usr/bin/env bash
set -euo pipefail
cd /mnt/i/Latent_Style/WEAVE
tar czf - \
  runs/v6_r2/eval_identity/eval.json \
  runs/v6_r2/eval_Q2a_direct/eval.json \
  runs/v6_r2/eval_Q2a_oracle/eval.json \
  runs/v6_r2/eval_B3_distill20/eval.json \
  runs/v6_r2/eval_B4_distill50/eval.json \
  runs/v6_r2/eval_scorediff_N24/eval.json \
  runs/v6_r2/p1_analysis/p1_summary.json \
  runs/v6_r2/p2_teachers/p2_teachers_summary.json \
  runs/v6_r2/dino_s_control.txt \
  runs/v6_r2/fitting_diagnostics/ \
  runs/v6_r2/queue_p0.log runs/v6_r2/queue_p1.log runs/v6_r2/queue_p2.log \
  runs/v6_r2/queue_p3.log \
  2>/dev/null | base64 -w0
