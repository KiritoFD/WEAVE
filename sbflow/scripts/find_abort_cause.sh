#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
echo "=== lines matching the abort pattern ==="
grep -naE "Traceback|RuntimeError|KeyError" runs/v6_24h/postqueue.log | head -n 10 || echo "  none"
echo
echo "=== full smoke section boundaries ==="
grep -naE "^##########|^\[[0-9]{2}:" runs/v6_24h/postqueue.log | head -n 20
