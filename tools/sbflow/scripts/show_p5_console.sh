#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
echo "=== P5 console section ==="
awk '/6\/6 P5/{f=1} f' runs/v6_24h/postqueue.log | head -n 60
