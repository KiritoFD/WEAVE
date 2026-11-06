#!/bin/bash
cd /mnt/i/Latent_Style/WEAVE
echo "== anchor dirs =="
ls -d runs/12h/anchors/* 2>/dev/null
for d in runs/12h/anchors/*/; do
  echo "-- $d"
  ls "$d" | head -6
done
