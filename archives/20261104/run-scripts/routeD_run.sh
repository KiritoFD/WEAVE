#!/usr/bin/env bash
# Route D formal run: multi-scale texture refiner ON vs the Style-ID control
# (control = runs/routeC_run/c_styleid_*, identical config except the texture flag).
set -u
cd /mnt/i/Latent_Style/WEAVE || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/mnt/i/Latent_Style/runs/routeD_run
L="$R/scan.log"
mkdir -p "$R"

run_one () {
  id="$1"; seed="$2"
  dir="$R/$id"; mkdir -p "$dir"
  if [ -f "/mnt/i/Latent_Style/runs/routeD_eval/$id/results_split.json" ]; then
    echo "[skip] $id" | tee -a "$L"; return 0
  fi
  python3 - "$dir/config.json" "$id" "$seed" <<'PY'
import json, sys
cfg, rid, seed = sys.argv[1:4]
d = {"_base": "/mnt/i/Latent_Style/WEAVE/config_d5_adain2.json",
     "checkpoint": {"save_dir": "/mnt/i/Latent_Style/runs/routeD_run/%s/ckpt" % rid},
     "training": {"seed": int(seed), "batch_size": 14, "num_epochs": 11, "log_interval": 10,
                  "use_tqdm": False, "resume_optimizer": False,
                  "learning_rate": 5e-5, "min_learning_rate": 5e-5,
                  "resume_checkpoint": "exp/dino_s_break/brk_a_ll03_10ep/epoch_0010.pt",
                  "numeric_debug": True, "numeric_debug_interval": 50, "numeric_debug_dump_limit": 120,
                  "internal_probe_enabled": True, "internal_early_stop_enabled": True,
                  "internal_probe_every": 1, "internal_early_stop_min_steps": 2000},
     "model": {"conditioner_variant": "style_id", "reference_encoder_in_dim": 384,
               "texture_refiner_enabled": True, "texture_refiner_hidden": 32,
               "target_latent_hf_subband_fusion_enabled": True,
               "target_latent_hf_head_fusion_gate_init": 0.0},
     "data": {"virtual_length_multiplier": 5.6}}
json.dump(d, open(cfg, "w"), indent=2)
PY
  echo "===== $(date '+%F %T') TRAIN $id (texture_on) =====" | tee -a "$L"
  timeout 3600 python3 run.py --config "$dir/config.json" > "$dir/train.log" 2>&1
  echo "[train rc=$?] $id" | tee -a "$L"
  grep -a "Route D texture refiner ON" "$dir/train.log" | tail -1 | tee -a "$L"
  grep -a "Epoch 11/11" "$dir/train.log" | tail -1 | tee -a "$L"
  python3 /mnt/i/Latent_Style/runs/gate.py "$dir" D 2>&1 | tee -a "$L"
  ck="$dir/ckpt/epoch_0011.pt"
  if [ -f "$ck" ]; then
    python3 - "$ck" <<'PY' | tee -a "$R/scan.log"
import sys, torch
sd = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
sd = sd.get("model_state_dict", sd.get("model", sd.get("state_dict", sd)))
w = sd.get("texture_refiner.out.weight")
print("[tex_w_norm]", None if w is None else round(float(w.float().norm()), 6))
PY
    bash /mnt/i/Latent_Style/runs/eval_one.sh "$id" "$ck" /mnt/i/Latent_Style/runs/routeD_eval \
      "$dir/config.json" "" 2>&1 | tee -a "$L"
  else
    echo "[NO CKPT] $id" | tee -a "$L"
  fi
}

run_one d_tex_on_s0 0
run_one d_tex_on_s1 1
echo "ROUTED_RUN_DONE" | tee -a "$L"
