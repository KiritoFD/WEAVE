#!/usr/bin/env bash
# Controlled b160 retest (item 8): 500 optimizer steps on the flow-only path,
# spanning ~15 epoch boundaries, to record peak VRAM / OOM / s-per-step.
# Uses the zombie-safe vram_guard (ps state, not VRAM-only) and a floor of 8G.
set -u
cd /mnt/i/Latent_Style/WEAVE || exit 1
source runs/vram_guard.sh
export HF_HOME=/mnt/i/Latent_Style/eval_cache/hf
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT=/mnt/i/Latent_Style/runs/p0_diag/b160_retest
BATCH=${BATCH:-160}
STEPS=${STEPS:-500}
EPOCHS=${EPOCHS:-16}
mkdir -p "$OUT"
rm -rf "$OUT/ckpt" "$OUT/train.log" "$OUT/vram_budget.json"
LR=$(python3 -c "print(round(5e-5*($BATCH/14.0)**0.5, 8))")

cfg="$OUT/config.json"
python3 - "$cfg" "$BATCH" "$EPOCHS" "$LR" "$OUT" "$STEPS" <<'PY'
import json, sys
cfg, b, epochs, lr, out, steps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), sys.argv[5], int(sys.argv[6])
d = {"_base": "/mnt/i/Latent_Style/WEAVE/config.json",
     "checkpoint": {"save_dir": out + "/ckpt"},
     "training": {"batch_size": b, "num_epochs": epochs, "seed": 0, "log_interval": 25,
                  "use_tqdm": False, "use_amp": True, "amp_dtype": "bf16",
                  "resume_checkpoint": "", "resume_optimizer": False,
                  "learning_rate": lr, "min_learning_rate": lr,
                  "save_interval": 9999, "stop_after_global_steps": steps,
                  "internal_probe_enabled": False,
                  "internal_early_stop_enabled": False},
     "data": {"packed_limit_per_style": 0, "virtual_length_multiplier": 1.0}}
json.dump(d, open(cfg, "w"), indent=2)
PY
echo "CONFIG batch=$BATCH steps=$STEPS epochs=$EPOCHS lr=$LR"

budget_start "$OUT" "$BATCH" 8000 9728 10752 || exit 1

start=$(date +%s)
python3 run.py --config "$cfg" > "$OUT/train.log" 2>&1 &
pid=$!
guard_watch "$pid" "$OUT" 8000
grc=$?
if [ $grc -ne 0 ]; then echo "GUARD_FAILED: low VRAM, run killed"; exit 1; fi
wait "$pid"; rc=$?
end=$(date +%s)
echo "TRAIN_RC=$rc WALL=$((end-start))s"

python3 - "$OUT" <<'PY'
import json, os, re, sys
out = sys.argv[1]
log = open(os.path.join(out, "train.log"), errors="replace").read()
epochs = [(float(m.group(1)), int(m.group(2))) for m in re.finditer(r"epoch=([0-9.]+)s", log)]
steps = [(float(m.group(1)), int(m.group(2))) for m in re.finditer(r"global_step=(\d+)", log)]
print("EPOCH_TIMES", epochs[-6:])
peak = None
for m in re.finditer(r"cuda_peak_reserved_gb=([0-9.]+)", log):
    peak = float(m.group(1))
print("CUDA_PEAK_RESERVED_GB", peak)
for line in log.splitlines():
    if "OOM" in line or "out of memory" in line:
        print("OOM_LINE", line[:200])
PY
cp "$OUT/ckpt/internal_dynamics_summary.json" "$OUT/internal_dynamics_summary.final.json" 2>/dev/null || true
echo B160_RETEST_DONE
