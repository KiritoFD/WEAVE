#!/bin/bash
cd /mnt/i/Latent_Style/WEAVE || exit 1
echo "== controls log tail =="
tail -n 30 runs/p2/q0_controls.log 2>/dev/null
echo "== tmux =="
tmux ls 2>/dev/null
