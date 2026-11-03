#!/usr/bin/env bash
# Blocking check: default config (internal_probe_enabled + internal_early_stop_enabled
# both default True) must survive every per-epoch probe point. 200 steps at batch 96
# spans ~4 epochs, so it crosses >=3 probe / early-stop evaluations.
set -u
cd /mnt/i/Latent_Style/WEAVE || exit 1
source runs/vram_guard.sh
export HF_HOME=/mnt/i/Latent_Style/eval_cache/hf
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT=/mnt/i/Latent_Style/runs/p0_diag/probe_smoke
BATCH=${BATCH:-96}
STEPS=${STEPS:-200}
mkdir -p "$OUT"
rm -rf "$OUT/ckpt" "$OUT/train.log" "$OUT/vram_budget.json"

cfg="$OUT/config.json"
python3 - "$cfg" "$BATCH" "$STEPS" "$OUT" <<'PY'
import json, sys
cfg, b, steps, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
d = {
  "_base": "/mnt/i/Latent_Style/WEAVE/experiments/architecture/hf_oriented_nohh.json",
  "checkpoint": {"save_dir": out + "/ckpt"},
  "training": {"batch_size": b, "num_epochs": 8, "seed": 0, "log_interval": 10,
               "use_tqdm": False, "use_amp": True, "amp_dtype": "bf16",
               "resume_checkpoint": "", "resume_optimizer": False,
               "save_interval": 9999, "stop_after_global_steps": steps},
  "data": {"packed_limit_per_style": 0, "virtual_length_multiplier": 1.0},
}
json.dump(d, open(cfg, "w"), indent=2)
print("CONFIG", cfg, "batch", b, "steps", steps)
PY

budget_start "$OUT" "$BATCH" 4000 5500 7500 || exit 1
start=$(date +%s)
python3 run.py --config "$cfg" > "$OUT/train.log" 2>&1 &
pid=$!
guard_watch "$pid" "$OUT" 4000
grc=$?
if [ $grc -ne 0 ]; then echo "GUARD_FAILED: low VRAM, run killed"; exit 1; fi
wait "$pid"; rc=$?
end=$(date +%s)
echo "SMOKE_RC=$rc WALL=$((end-start))s"
echo "--- probe / early-stop evidence ---"
grep -nE "NameError|Traceback|internal probe|early_stop|probe \|" "$OUT/train.log" | tail -24
echo "--- tail ---"
tail -3 "$OUT/train.log"
echo PROBE_SMOKE_DONE
