#!/usr/bin/env bash
cd /mnt/i/Latent_Style/WEAVE/runs/cache || exit 1
ls -la | awk '{print $5, $9}' | head -30
echo "==== grep 26144"
grep -rl "26144" . 2>/dev/null | head -10
