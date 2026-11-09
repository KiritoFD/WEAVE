#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
echo "=== N1 training steps ==="
grep -a "step " runs/v6_24h/train_logs/N1_no_style.log | tail -n 4
echo "=== Eval200 line (if any) ==="
grep -a "Eval200" runs/v6_24h/train_logs/N1_no_style.log | tail -n 2 || echo "  not yet"
echo "=== tail ==="
tail -n 2 runs/v6_24h/train_logs/N1_no_style.log
