#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
F=runs/v6_24h/train_logs/N3_dim32.log
echo "=== now ==="
date +%H:%M:%S
echo "=== log mtime / size ==="
ls -l --time-style=+%H:%M:%S "$F"
echo "=== raw tail ==="
tail -c 700 "$F"
echo
echo "=== N3 process ==="
ps -eo pid,etime,args --no-headers | grep "N3_dim32" | grep -v grep
