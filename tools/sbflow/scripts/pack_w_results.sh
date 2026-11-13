#!/usr/bin/env bash
set -euo pipefail
cd /mnt/i/Latent_Style/WEAVE
tar czf - \
  docs/v6/W_REPORT.md \
  runs/v6_w/queue_w_12h.log runs/v6_w/queue_w_imgcal.log \
  runs/v6_w/w_paired_report.txt runs/v6_w/w_paired_verdict.json \
  runs/v6_w/w_best_arm.json \
  runs/v6_w/eval_W1_FeatDistill_218/eval.json \
  runs/v6_w/eval_W2_LossClean_218/eval.json \
  runs/v6_w/eval_W3_LossClean_218_l50/eval.json \
  runs/v6_w/eval_W5_ImgCalib/eval.json \
  runs/v6_w/state/ \
  2>/dev/null | base64 -w0
