#!/usr/bin/env bash
# 最佳配置的多 seed 复现检查（100 对小集，paper/512/TRT，20 步，s=0.15 cfg=1.0）。
set -u
cd /mnt/i/Latent_Style/WEAVE || exit 1
export HF_HOME=/mnt/i/Latent_Style/eval_cache/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
root=runs/v4/prior/sweep_10k
exec >> "$root/seeds.log" 2>&1
for sd in 1 2; do
  out="$root/seed${sd}_s0.15_cf1.0"
  mkdir -p "$out"
  V4_NUM_STEPS=20 V4_EDIT_STRENGTH=0.15 V4_CFG=1.0 V4_SEED="$sd" \
  python3 tools/12h/eval_fast_trt_fp16.py --protocol paper --checkpoint runs/v4/prior/step_10000.pt \
    --output "$out" --device cuda --batch 8 --metric-batch 32 --decode-batch 8 --image-size 512 \
    --workers 4 --no-save-images --gen-dtype fp32 --vae-dtype fp32 --sync-timing --decoder trt \
    --max-src-samples 4 --num-steps 20 --seed "$sd" --backbone "v4_seed${sd}_s0.15_cf1.0" \
    > "$out/eval.log" 2>&1
  echo "seed${sd} EXIT=$? $(date +%H:%M:%S)"
done
echo SEEDSDONE
