#!/bin/bash
cd /mnt/i/Latent_Style/WEAVE
grep -a "[prof]" _tmp/prof_render.log
echo "== proc =="
ps aux | grep prof_render | grep -v grep | head -2
