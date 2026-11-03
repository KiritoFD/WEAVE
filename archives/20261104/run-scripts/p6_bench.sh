#!/usr/bin/env bash
# P6 before/after bench: same b160 config, only the probe-scalar sync schedule differs.
#   PROBE_INTERVAL=1  -> emulate pre-P6 behaviour (one multi-.item() sweep every step)
#   PROBE_INTERVAL=25 -> post-P6 behaviour (one .tolist() per source on the throttle)
# Internal probe / early stop are off so the only difference is the sync sweep.
set -u
cd /mnt/i/Latent_Style/WEAVE || exit 1
source runs/vram_guard.sh
export HF_HOME=/mnt/i/Latent_Style/eval_cache/hf
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BATCH=${BATCH:-160}
STEPS=${STEPS:-100}
INTERVAL=${PROBE_INTERVAL:-1}
OUT=/mnt/i/Latent_Style/runs/p0_diag/p6_bench_i${INTERVAL}
mkdir -p "$OUT"
rm -rf "$OUT/ckpt" "$OUT/train.log" "$OUT/vram_budget.json"

cfg="$OUT/config.json"
python3 - "$cfg" "$BATCH" "$STEPS" "$OUT" "$INTERVAL" <<'PY'
import json, sys
cfg, b, steps, out, interval = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], int(sys.argv[5])
d = {
  "_base": "/mnt/i/Latent_Style/WEAVE/experiments/architecture/hf_oriented_nohh.json",
  "checkpoint": {"save_dir": out + "/ckpt"},
  "training": {"batch_size": b, "num_epochs": 8, "seed": 0, "log_interval": 5,
               "use_tqdm": False, "use_amp": True, "amp_dtype": "bf16",
               "resume_checkpoint": "", "resume_optimizer": False,
               "save_interval": 9999, "stop_after_global_steps": steps,
               "internal_probe_enabled": False, "internal_early_stop_enabled": False,
               "probe_log_interval": interval},
  "data": {"packed_limit_per_style": 0, "virtual_length_multiplier": 1.0},
}
json.dump(d, open(cfg, "w"), indent=2)
PY

budget_start "$OUT" "$BATCH" 8000 9728 10752 || exit 1
start=$(date +%s)
python3 run.py --config "$cfg" > "$OUT/train.log" 2>&1
rc=$?
end=$(date +%s)
echo "P6_BENCH interval=$INTERVAL rc=$rc wall=$((end-start))s"
grep -a "Epoch " "$OUT/train.log" | tail -5
echo P6_BENCH_DONE
