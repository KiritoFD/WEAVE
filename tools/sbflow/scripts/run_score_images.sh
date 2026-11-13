#!/usr/bin/env bash
# Consumer pass: read generated PNGs from disk and compute metrics.
set -euo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
GEN=${1:-runs/blog_sbflow/gen/Mid8M_b16}
OUT=${2:-runs/blog_sbflow/scores/Mid8M_b16}
BATCH=${3:-16}
rm -rf "$OUT"
python3 -u -m sbflow.score_images --repo-root . --gen "$GEN" --out "$OUT" \
  --batch "$BATCH" --workers 8
echo "=== SCORE DONE ==="
