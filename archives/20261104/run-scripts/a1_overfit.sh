#!/usr/bin/env bash
# A1: fixed small-subset overfit capacity probe (internal supervision only).
#   a1_32        32 samples, fixed pairing, no SDE noise
#   a1_128       128 samples, fixed pairing, no SDE noise
#   a1_512       512 samples, fixed pairing, no SDE noise
#   a1_128_noise 128 samples, fixed pairing, original SDE noise (bridge_sigma=0.02)
# A1-5 (simplified content-aligned target) needs a new target mode -> not run.
set -u
cd /mnt/i/Latent_Style/WEAVE || exit 1
source runs/vram_guard.sh
export HF_HOME=/mnt/i/Latent_Style/eval_cache/hf
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

R=/mnt/i/Latent_Style/runs/p0_diag/a1
STEPS=${STEPS:-2000}
mkdir -p "$R"

run_case () {
  tag="$1"; limit="$2"; sigma="$3"; save_iv="$4"; epochs="$5"
  B=32
  cfg="$R/${tag}.json"
  python3 - "$cfg" "$limit" "$B" "$sigma" "$STEPS" "$R" "$tag" "$save_iv" "$epochs" <<'PY'
import json, math, sys
cfg, limit, b, sigma, steps, r, tag, save_iv, epochs = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]), sys.argv[6], sys.argv[7], int(sys.argv[8]), int(sys.argv[9])
lr = 2e-4 * math.sqrt(b / 96.0)
d = {
  "_base": "/mnt/i/Latent_Style/WEAVE/experiments/architecture/hf_oriented_nohh.json",
  "checkpoint": {"save_dir": "%s/ckpt_%s" % (r, tag)},
  "training": {"batch_size": b, "num_epochs": epochs, "seed": 0, "log_interval": 1,
               "use_tqdm": False, "use_amp": True, "amp_dtype": "bf16",
               "resume_checkpoint": "", "resume_optimizer": False,
               "learning_rate": lr, "min_learning_rate": lr,
               "save_interval": save_iv, "stop_after_global_steps": steps,
               "internal_probe_enabled": False, "internal_early_stop_enabled": False},
  "data": {"train_sample_limit": limit, "pairing_mode": "fixed", "pairing_seed": 0,
           "identity_prob": 0.0, "virtual_length_multiplier": 1.0},
  "bridge": {"bridge_sigma": sigma},
}
json.dump(d, open(cfg, "w"), indent=2)
print("CASE", tag, "limit", limit, "batch", b, "lr", "%.3g" % lr, "sigma", sigma, "steps", steps)
PY
  echo "== $tag limit=$limit sigma=$sigma lr=$(python3 -c "import math;print('%.3g'%(2e-4*math.sqrt($B/96.0)))") steps=$STEPS $(date '+%T') =="
  python3 run.py --config "$cfg" > "$R/$tag.log" 2>&1
  echo "[$tag rc=$?] $(date '+%T')"
}

run_case a1_32        32  0.0  500  2500
run_case a1_128       128 0.0  100  600
run_case a1_512       512 0.0  25   200
run_case a1_128_noise 128 0.02 100  600
echo A1_DONE
