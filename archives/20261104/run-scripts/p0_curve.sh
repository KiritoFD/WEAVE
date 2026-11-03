#!/usr/bin/env bash
# P0.3 Long-range learning curve with the current baseline recipe (no new modules).
# One continuous run: epochs 11..25, 2000 updates/epoch -> 2k ... 30k updates.
# Checkpoints are saved every epoch; we evaluate 2k / 10k / 20k / 30k.
set -u
# 2026-11-04: user paused model design; long-range curve cancelled. Keep this
# file as the record of the planned config, but never start it from the queue.
echo "P0_CURVE_SKIPPED: long-range curve cancelled by user (2026-11-04)"
exit 0
cd /mnt/i/Latent_Style/WEAVE || exit 1
export HF_HOME=/mnt/i/Latent_Style/eval_cache/hf
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/mnt/i/Latent_Style/runs/p0_diag
dir="$R/curve"; mkdir -p "$dir"
python3 - "$dir/config.json" <<'PY'
import json, sys
d = {"_base": "/mnt/i/Latent_Style/WEAVE/config.json",
     "checkpoint": {"save_dir": "/mnt/i/Latent_Style/runs/p0_diag/curve/ckpt"},
     "training": {"batch_size": 14, "num_epochs": 25, "seed": 0, "log_interval": 10,
                  "use_tqdm": False, "use_amp": True, "amp_dtype": "bf16",
                  "resume_optimizer": False,
                  "learning_rate": 5e-5, "min_learning_rate": 5e-5,
                  "resume_checkpoint": "exp/dino_s_break/brk_a_ll03_10ep/epoch_0010.pt",
                  "save_interval": 1,
                  "internal_probe_enabled": True, "internal_early_stop_enabled": True,
                  "internal_probe_every": 1, "internal_early_stop_min_steps": 2000},
     "model": {"target_latent_hf_subband_fusion_enabled": True,
               "target_latent_hf_head_fusion_gate_init": 0.0},
     "data": {"virtual_length_multiplier": 5.6}}
json.dump(d, open(sys.argv[1], 'w'), indent=2)
PY
echo "===== $(date '+%F %T') TRAIN curve epochs 11..25 (2000 steps/epoch) =====" | tee -a "$R/curve.log"
timeout 7200 python3 run.py --config "$dir/config.json" > "$dir/train.log" 2>&1
echo "[train rc=$?] curve" | tee -a "$R/curve.log"
for ep in 11 15 20 25; do
  ck="$dir/ckpt/epoch_00$ep.pt"
  case "$ep" in
    11) tag=curve_2k ;; 15) tag=curve_10k ;; 20) tag=curve_20k ;; 25) tag=curve_30k ;;
  esac
  if [ -f "$ck" ]; then
    bash /mnt/i/Latent_Style/runs/eval_one.sh "$tag" "$ck" "$R" "$dir/config.json" 2>&1 | tail -3 | tee -a "$R/curve.log"
  else
    echo "[NO CKPT] $tag $ck" | tee -a "$R/curve.log"
  fi
done
echo P0_CURVE_DONE | tee -a "$R/curve.log"
