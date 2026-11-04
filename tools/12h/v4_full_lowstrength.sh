#!/usr/bin/env bash
# 低强度补点：全量 750 all-pairs，20 步，cfg 1.0，强度 0.05/0.08/0.10/0.20。
set -u
cd /mnt/i/Latent_Style/WEAVE || exit 1
root=runs/v4/prior/full
ck=runs/v4/prior/step_10000.pt
exec > "$root/full_eval_lowstrength.log" 2>&1
for s in 0.05 0.08 0.10 0.20; do
  bash tools/12h/v4_full_eval.sh "s${s}_cf1.0" "$ck" "$root" 20 "$s" 1.0
  echo "s${s} done $(date +%H:%M:%S)"
done
python3 tools/12h/v4_sweep_table.py "$root"
echo LOWSTRENGTHDONE

