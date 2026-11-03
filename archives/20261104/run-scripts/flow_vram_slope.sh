#!/usr/bin/env bash
# Marginal VRAM slope for the internal-supervision main path (flow-only).
# Runs identical configs at batch 64 / 96 / 128 for 14 steps each and samples
# nvidia-smi peak.  slope = (peak(128) - peak(64)) / 64  [GB/sample].
set -u
cd /mnt/i/Latent_Style/WEAVE || exit 1
source runs/vram_guard.sh
export HF_HOME=/mnt/i/Latent_Style/eval_cache/hf
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

R=/mnt/i/Latent_Style/runs/p0_diag/vram_slope
STEPS=${STEPS:-14}
mkdir -p "$R"
: > "$R/slope.jsonl"

for B in 64 96 128; do
  cfg="$R/config_b$B.json"
  python3 - "$cfg" "$B" "$STEPS" "$R" <<'PY'
import json, sys
cfg, b, steps, r = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
d = {
  "_base": "/mnt/i/Latent_Style/WEAVE/experiments/architecture/hf_oriented_nohh.json",
  "checkpoint": {"save_dir": "%s/ckpt_b%d" % (r, b)},
  "training": {"batch_size": b, "num_epochs": 3, "seed": 0, "log_interval": 5,
               "use_tqdm": False, "use_amp": True, "amp_dtype": "bf16",
               "resume_checkpoint": "", "resume_optimizer": False,
               "save_interval": 9999, "stop_after_global_steps": steps,
               "internal_probe_enabled": False, "internal_early_stop_enabled": False},
  "data": {"packed_limit_per_style": 0, "virtual_length_multiplier": 1.0},
}
json.dump(d, open(cfg, "w"), indent=2)
PY
  echo "== batch=$B steps=$STEPS $(date '+%T') =="
  python3 run.py --config "$cfg" > "$R/b$B.log" 2>&1 &
  pid=$!
  peak=0
  while kill -0 "$pid" 2>/dev/null; do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
    if [ "${u:-0}" -gt "$peak" ]; then peak="$u"; fi
    sleep 1
  done
  wait "$pid"; rc=$?
  echo "B=$B rc=$rc peak_mib=$peak"
  echo "{\"batch\": $B, \"steps\": $STEPS, \"peak_mib\": $peak, \"rc\": $rc}" >> "$R/slope.jsonl"
done

python3 - "$R/slope.jsonl" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
by = {r["batch"]: r["peak_mib"] / 1024.0 for r in rows}
lo, hi = min(by), max(by)
slope = (by[hi] - by[lo]) / (hi - lo)
print("SLOPE %.4f GB/sample (b%d %.2fG -> b%d %.2fG)" % (slope, lo, by[lo], hi, by[hi]))
PY
echo FLOW_SLOPE_DONE
