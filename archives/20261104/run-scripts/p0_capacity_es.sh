#!/usr/bin/env bash
# P0.2b Capacity diagnostic with internal early stop ARMED on both arms.
# Same fit32 content subset and 5000 updates as runs/p0_capacity.sh, but the
# gated subband-HF head is built so the internal probe has a real reading.
# Run AFTER the current p0_all queue finishes (cap32/cap128 pair stays unarmed
# for comparability; this pair is the consistently-armed twin).
set -u
cd /mnt/i/Latent_Style/WEAVE || exit 1
export HF_HOME=/mnt/i/Latent_Style/eval_cache/hf
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/mnt/i/Latent_Style/runs/p0_diag
mkdir -p "$R"
[ -d "$R/fit32" ] || { echo "[ERR] $R/fit32 missing; run runs/p0_capacity.sh first" >&2; exit 1; }

train_and_eval () {
  n="$1"; tag="cap${n}_es"; steps=5000; b=14
  dir="$R/$tag"; mkdir -p "$dir"
  vlm=$(python3 -c "print($steps*$b/($n*5))")
  python3 - "$dir/config.json" "$tag" "$n" "$vlm" <<'PY'
import json, sys
cfg, tag, n, vlm = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
d = {"_base": "/mnt/i/Latent_Style/WEAVE/config.json",
     "checkpoint": {"save_dir": "/mnt/i/Latent_Style/runs/p0_diag/%s/ckpt" % tag},
     "training": {"batch_size": 14, "num_epochs": 11, "seed": 0, "log_interval": 10,
                  "use_tqdm": False, "use_amp": True, "amp_dtype": "bf16",
                  "resume_optimizer": False,
                  "learning_rate": 5e-5, "min_learning_rate": 5e-5,
                  "resume_checkpoint": "exp/dino_s_break/brk_a_ll03_10ep/epoch_0010.pt",
                  "save_interval": 1,
                  "internal_probe_enabled": True, "internal_early_stop_enabled": True,
                  "internal_probe_every": 1, "internal_early_stop_min_steps": 2000},
     "model": {"target_latent_hf_subband_fusion_enabled": True,
               "target_latent_hf_head_fusion_gate_init": 0.0},
     "data": {"packed_limit_per_style": n, "virtual_length_multiplier": vlm}}
json.dump(d, open(cfg, 'w'), indent=2)
PY
  echo "===== $(date '+%F %T') TRAIN $tag limit=$n/style steps=$steps vlm=$vlm (early stop ARMED) =====" | tee -a "$R/capacity_es.log"
  timeout 3600 python3 run.py --config "$dir/config.json" > "$dir/train.log" 2>&1
  echo "[train rc=$?] $tag" | tee -a "$R/capacity_es.log"
  grep -a "Epoch 11/11" "$dir/train.log" | tail -1 | tee -a "$R/capacity_es.log"
  grep -a "Internal early stop summary" "$dir/train.log" | tail -1 | tee -a "$R/capacity_es.log"
  ck="$dir/ckpt/epoch_0011.pt"
  if [ -f "$ck" ]; then
    bash /mnt/i/Latent_Style/runs/eval_one.sh "${tag}_fit" "$ck" "$R" "$dir/config.json" "" \
      /mnt/i/Latent_Style/runs/p0_diag/fit32 2>&1 | tail -3 | tee -a "$R/capacity_es.log"
    bash /mnt/i/Latent_Style/runs/eval_one.sh "${tag}_heldout" "$ck" "$R" "$dir/config.json" 2>&1 | tail -3 | tee -a "$R/capacity_es.log"
  else
    echo "[NO CKPT] $tag" | tee -a "$R/capacity_es.log"
  fi
}

train_and_eval 32
train_and_eval 128
echo P0_CAPACITY_ES_DONE | tee -a "$R/capacity_es.log"
