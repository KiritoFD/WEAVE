#!/usr/bin/env bash
set -euo pipefail
cd /mnt/i/Latent_Style/WEAVE
tar czf - runs/v6_r2/queue_r2_12h.log docs/v6/R2_12H_REPORT.md \
  runs/v6_r2/final_verdict.json 2>/dev/null | base64 -w0
