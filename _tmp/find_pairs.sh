#!/usr/bin/env bash
cd /mnt/i/Latent_Style/WEAVE || exit 1
find . -maxdepth 2 -iname "*pair*" -o -maxdepth 2 -iname "*cache*" | head -40
echo "==== data dir"
ls -la data 2>/dev/null | awk '{print $5, $9}'
echo "==== v4 cache"
ls -la runs/v4/cache 2>/dev/null | awk '{print $5, $9}'
