#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
F=runs/v6_24h/train_logs/N5_inv_off.log
echo "=== tail 40 ==="
tail -n 40 "$F"
echo
echo "=== traceback search ==="
grep -an "Traceback\|Error\|error" "$F" | tail -n 10 || echo "  none"
