#!/bin/bash
cd /mnt/i/Latent_Style/WEAVE || exit 1
python3 -u _tmp/prof_render.py --checkpoint runs/v4/S1_A_P5/step_000600.pt --n 128
