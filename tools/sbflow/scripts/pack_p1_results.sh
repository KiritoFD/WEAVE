#!/usr/bin/env bash
set -euo pipefail
cd /mnt/i/Latent_Style/WEAVE
tar czf - \
  docs/v6/W_P1_REPORT.md \
  runs/v6_w/w_p1_verdict.json \
  runs/v6_w/w_paired_verdict.json \
  runs/v6_w/w_paired_report.txt \
  runs/v6_w/eval_guided_r1/eval.json \
  runs/v6_w/eval_guided_score/eval.json \
  runs/v6_w/eval_guided_both/eval.json \
  runs/v6_w/score_prior/train_report.json \
  runs/v6_w/state_p1/ \
  2>/dev/null | base64 -w0
