#!/usr/bin/env bash
# P4: full 750 all-pairs eval sweep for the V4 diffusion prior.
# 20-step main configs + identity / null-condition references + 50-step follow-ups.
set -u
cd /mnt/i/Latent_Style/WEAVE || exit 1
exec > runs/v4/prior/full_eval_driver.log 2>&1
root=runs/v4/prior/full
ck=runs/v4/prior/step_10000.pt

bash tools/12h/v4_full_eval.sh s0.15_cf1.0            "$ck" "$root" 20 0.15 1.0
bash tools/12h/v4_full_eval.sh s0.30_cf1.5            "$ck" "$root" 20 0.30 1.5
V4_IDENTITY=1  bash tools/12h/v4_full_eval.sh identity "$ck" "$root" 20 0.0 1.0
V4_FORCE_NULL=1 bash tools/12h/v4_full_eval.sh nullcond_s0.15_cf1.0 "$ck" "$root" 20 0.15 1.0
bash tools/12h/v4_full_eval.sh s0.15_cf1.0_50steps    "$ck" "$root" 50 0.15 1.0
bash tools/12h/v4_full_eval.sh s0.30_cf1.5_50steps    "$ck" "$root" 50 0.30 1.5
bash tools/12h/v4_full_eval.sh s0.45_cf1.0_50steps    "$ck" "$root" 50 0.45 1.0
echo ALLDONE
