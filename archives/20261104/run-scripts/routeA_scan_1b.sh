#!/usr/bin/env bash
# Route A round1b: lr=5e-5 / 2000 steps (2 epochs x 1000 steps).
# IMPORTANT: resume_optimizer=false, otherwise the baseline checkpoint's
# scheduler state (lr already decayed to 1e-5) silences the new learning rate.
set -u
cd /mnt/i/Latent_Style/WEAVE || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/mnt/i/Latent_Style/runs/routeA_scan_1b
mkdir -p "$R"
L="$R/scan.log"

run_one () {
  id="$1"; ls="$2"; lc="$3"; steps="$4"; every="$5"; seed="$6"
  dir="$R/$id"; mkdir -p "$dir"
  if [ -f "/mnt/i/Latent_Style/runs/routeA_scan_eval/$id/results_split.json" ]; then
    echo "[skip] $id" | tee -a "$L"; return 0
  fi
  python3 - "$dir/config.json" "$id" "$ls" "$lc" "$steps" "$every" "$seed" <<'PY'
import json, sys
cfg, rid, ls, lc, steps, every, seed = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])
d = {"_base": "/mnt/i/Latent_Style/runs/nextgen_routeA_post/config_routeA.json",
     "checkpoint": {"save_dir": "/mnt/i/Latent_Style/runs/routeA_scan_1b/%s/ckpt" % rid},
     "training": {"seed": seed, "batch_size": 1, "num_epochs": 12, "log_interval": 10,
                  "learning_rate": 5e-5, "min_learning_rate": 5e-5,
                  "resume_optimizer": False,
                  "numeric_debug": True, "numeric_debug_interval": 50, "numeric_debug_dump_limit": 120},
     "bridge": {"output_supervision_lambda_style": ls, "output_supervision_lambda_content": lc,
                "output_supervision_ode_steps": steps, "output_supervision_apply_every": every},
     "data": {"virtual_length_multiplier": 0.2}}
json.dump(d, open(cfg, "w"), indent=2)
PY
  echo "===== $(date '+%F %T') TRAIN $id =====" | tee -a "$L"
  timeout 3600 python3 run.py --config "$dir/config.json" > "$dir/train.log" 2>&1
  echo "[train rc=$?] $id" | tee -a "$L"
  echo "[lr_actual] $(grep -ao 'lr=[0-9.e+-]*' "$dir/train.log" | sort -u | tr '\n' ' ')" | tee -a "$L"
  grep -a "Epoch 11/12\|Epoch 12/12" "$dir/train.log" | tail -2 | tee -a "$L"
  python3 /mnt/i/Latent_Style/runs/gate.py "$dir" 2>&1 | tee -a "$L"
  ck="$dir/ckpt/epoch_0012.pt"
  if [ -f "$ck" ]; then
    bash /mnt/i/Latent_Style/runs/eval_one.sh "$id" "$ck" /mnt/i/Latent_Style/runs/routeA_scan_eval 2>&1 | tee -a "$L"
  else
    echo "[NO CKPT] $id" | tee -a "$L"
  fi
}

run_one b_ls03_lc01_s2_e1_s0 0.3 0.1 2 1 0
run_one b_ls01_lc01_s2_e1_s0 0.1 0.1 2 1 0
run_one b_ls03_lc03_s2_e1_s0 0.3 0.3 2 1 0
run_one b_ls03_lc01_s4_e1_s0 0.3 0.1 4 1 0
run_one b_ls03_lc01_s2_e2_s0 0.3 0.1 2 2 0
echo "SCAN_ROUND1B_DONE" | tee -a "$L"
