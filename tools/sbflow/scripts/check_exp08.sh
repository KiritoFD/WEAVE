#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
ROOT=runs/blog_sbflow/suite40
echo "=== exp08 artifacts ==="
ls -lh "$ROOT/exp08"/*.pt 2>/dev/null | tail -6 || echo "  no ckpt"
ls -lh "$ROOT/exp08/train_summary.json" 2>/dev/null || echo "  no summary"
ls -lh "$ROOT/gen_08_quick200/manifest.json" 2>/dev/null || echo "  no gen"
ls -lh "$ROOT/score_08_quick200/style_scores.json" 2>/dev/null || echo "  no score"
echo "=== exp08 train tail ==="
tail -n 4 "$ROOT/logs/08_train.log" 2>/dev/null || true
echo "=== exp08 gen tail ==="
tail -n 4 "$ROOT/logs/08_gen.log" 2>/dev/null || true
echo "=== suite log tail ==="
tail -n 8 "$ROOT/logs/suite.log" 2>/dev/null || true
