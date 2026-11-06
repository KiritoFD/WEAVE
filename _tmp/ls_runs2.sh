#!/usr/bin/env bash
cd /mnt/i/Latent_Style/WEAVE/runs || exit 1
for d in */; do echo "== $d"; ls "$d" 2>/dev/null | head -30; done
