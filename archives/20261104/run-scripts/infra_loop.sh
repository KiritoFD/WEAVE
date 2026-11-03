#!/usr/bin/env bash
# Infra closed loop: one real training -> full 750 eval -> results + registry.
# Resume the Route-A baseline for 2000 updates (batch 14, bf16), early stop
# armed but min_steps=2000 means it cannot fire within this single epoch.
set -u
cd /mnt/i/Latent_Style/WEAVE || exit 1
export HF_HOME=/mnt/i/Latent_Style/eval_cache/hf
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/mnt/i/Latent_Style/runs/p0_diag
tag=infra_loop
dir="$R/$tag"; mkdir -p "$dir"

python3 - "$dir/config.json" <<'PY'
import json, sys
d = {"_base": "/mnt/i/Latent_Style/WEAVE/config.json",
     "checkpoint": {"save_dir": "/mnt/i/Latent_Style/runs/p0_diag/infra_loop/ckpt"},
     "training": {"batch_size": 14, "num_epochs": 11, "seed": 0, "log_interval": 10,
                  "use_tqdm": False, "use_amp": True, "amp_dtype": "bf16",
                  "resume_optimizer": False,
                  "resume_checkpoint": "exp/dino_s_break/brk_a_ll03_10ep/epoch_0010.pt",
                  "learning_rate": 5e-5, "min_learning_rate": 5e-5,
                  "save_interval": 1,
                  "internal_probe_enabled": True, "internal_early_stop_enabled": True,
                  "internal_probe_every": 1, "internal_early_stop_min_steps": 2000},
     "model": {"target_latent_hf_subband_fusion_enabled": True,
               "target_latent_hf_head_fusion_gate_init": 0.0},
     "data": {"virtual_length_multiplier": 5.6}}
json.dump(d, open(sys.argv[1], 'w'), indent=2)
PY

echo "===== $(date '+%F %T') INFRA LOOP train 2000 steps -> 750 eval =====" | tee -a "$R/infra_loop.log"
timeout 1800 python3 run.py --config "$dir/config.json" > "$dir/train.log" 2>&1
echo "[train rc=$?] $tag" | tee -a "$R/infra_loop.log"
grep -a "Internal early stop summary" "$dir/train.log" | tail -1 | tee -a "$R/infra_loop.log"
python3 -m json.tool "$dir/ckpt/internal_dynamics_summary.json" | tee -a "$R/infra_loop.log"

ck="$dir/ckpt/epoch_0011.pt"
if [ -f "$ck" ]; then
  bash /mnt/i/Latent_Style/runs/eval_one.sh "${tag}_eval" "$ck" "$R" "$dir/config.json" 2>&1 | tail -4 | tee -a "$R/infra_loop.log"
  python3 /mnt/i/Latent_Style/runs/gate.py "$dir" A > "$dir/gate.json" 2>>"$R/infra_loop.log" || true
  python3 /mnt/i/Latent_Style/runs/append_registry.py --run-dir "$dir" --eval-dir "$R/${tag}_eval" \
    --route A --valid no \
    --invalid-reason "infra closed-loop: base config has no Route-A output supervision; gate N/A" \
    >> "$R/infra_loop.log" 2>&1 || true
  echo "[artifacts] results=$R/${tag}_eval/results.json split=$R/${tag}_eval/results_split.json" | tee -a "$R/infra_loop.log"
  echo "[registry] appended $tag" | tee -a "$R/infra_loop.log"
else
  echo "[NO CKPT] $tag" | tee -a "$R/infra_loop.log"
fi
echo INFRA_LOOP_DONE | tee -a "$R/infra_loop.log"
