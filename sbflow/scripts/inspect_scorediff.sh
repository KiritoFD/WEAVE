#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
echo "=== runs/v6/score_diff ==="
ls -la runs/v6/score_diff/ 2>/dev/null || echo "  missing"
echo
echo "=== score_diff ckpts anywhere under runs/v6 ==="
find runs/v6 -name "*.pt" 2>/dev/null | head -20
echo
echo "=== score_diff.py CLI surface ==="
grep -nE "add_argument|def main" tools/v6/score_diff.py 2>/dev/null | head -25
