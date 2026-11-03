#!/usr/bin/env bash
# Item 1: fixed plan-internal baseline (hf_oriented_nohh), 600 optimizer
# steps at batch 96 (= 52 steps/epoch), then the standard 750-image eval
# whose split report yields 600 cross-style / 150 same-style.
set -u
cd /mnt/i/Latent_Style/WEAVE || exit 1
source runs/vram_guard.sh
export HF_HOME=/mnt/i/Latent_Style/eval_cache/hf
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT=/mnt/i/Latent_Style/runs/p0_diag/baseline_nohh600
BATCH=${BATCH:-96}
STEPS=${STEPS:-600}
mkdir -p "$OUT"
rm -rf "$OUT/ckpt" "$OUT/train.log" "$OUT/vram_budget.json"

cfg="$OUT/config.json"
python3 - "$cfg" "$BATCH" "$STEPS" "$OUT" <<'PY'
import json, sys
cfg, b, steps, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
d = {
  "_base": "/mnt/i/Latent_Style/WEAVE/experiments/architecture/hf_oriented_nohh.json",
  "checkpoint": {"save_dir": out + "/ckpt"},
  "training": {"batch_size": b, "num_epochs": 12, "seed": 0, "log_interval": 25,
               "use_tqdm": False, "use_amp": True, "amp_dtype": "bf16",
               "resume_checkpoint": "", "resume_optimizer": False,
               "save_interval": 9999, "stop_after_global_steps": steps,
               "internal_probe_enabled": False,
               "internal_early_stop_enabled": False},
  "data": {"packed_limit_per_style": 0, "virtual_length_multiplier": 1.0},
}
json.dump(d, open(cfg, "w"), indent=2)
PY
echo "CONFIG baseline=hf_oriented_nohh batch=$BATCH steps=$STEPS"

budget_start "$OUT" "$BATCH" 4000 5500 7500 || exit 1

start=$(date +%s)
python3 run.py --config "$cfg" > "$OUT/train.log" 2>&1 &
pid=$!
guard_watch "$pid" "$OUT" 4000
grc=$?
if [ $grc -ne 0 ]; then echo "GUARD_FAILED: low VRAM, run killed"; exit 1; fi
wait "$pid"; rc=$?
end=$(date +%s)
echo "TRAIN_RC=$rc WALL=$((end-start))s"
grep -E "Epoch .*epoch=" "$OUT/train.log" | tail -4
echo BASELINE_TRAIN_DONE
