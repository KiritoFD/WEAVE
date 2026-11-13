#!/usr/bin/env bash
set -euo pipefail
cd /mnt/i/Latent_Style/WEAVE
tar czf - \
  runs/v6_w/eval_gated_full/eval.json \
  runs/v6_w/eval_gated_hf/eval.json \
  runs/v6_w/eval_gated_topk/eval.json \
  runs/v6_w/eval_gated_content/eval.json \
  runs/v6_w/eval_gated_style_energy/eval.json \
  runs/v6_w/w_paired_report.txt \
  runs/v6_w/w_paired_verdict.json \
  2>/dev/null | base64 -w0
