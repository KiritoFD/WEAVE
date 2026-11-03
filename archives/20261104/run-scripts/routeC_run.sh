#!/usr/bin/env bash
# Route C formal run: DINOv2 patch-token reference branch + InfoNCE vs the
# legacy Style-ID path, same steps/lr/seed for a fair A/B.
set -u
cd /mnt/i/Latent_Style/WEAVE || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/mnt/i/Latent_Style/runs/routeC_run
L="$R/scan.log"
mkdir -p "$R"

run_one () {
  id="$1"; variant="$2"; in_dim="$3"; nce="$4"; seed="$5"
  dir="$R/$id"; mkdir -p "$dir"
  if [ -f "/mnt/i/Latent_Style/runs/routeC_eval/$id/results_split.json" ]; then
    echo "[skip] $id" | tee -a "$L"; return 0
  fi
  python3 - "$dir/config.json" "$id" "$variant" "$in_dim" "$nce" "$seed" <<'PY'
import json, sys
cfg, rid, variant, in_dim, nce, seed = sys.argv[1:7]
d = {"_base": "/mnt/i/Latent_Style/WEAVE/config_d5_adain2.json",
     "checkpoint": {"save_dir": "/mnt/i/Latent_Style/runs/routeC_run/%s/ckpt" % rid},
     "training": {"seed": int(seed), "batch_size": 14, "num_epochs": 11, "log_interval": 10,
                  "use_tqdm": False, "resume_optimizer": False,
                  "learning_rate": 5e-5, "min_learning_rate": 5e-5,
                  "resume_checkpoint": "exp/dino_s_break/brk_a_ll03_10ep/epoch_0010.pt",
                  "numeric_debug": True, "numeric_debug_interval": 50, "numeric_debug_dump_limit": 120,
                  "internal_probe_enabled": True, "internal_early_stop_enabled": True,
                  "internal_probe_every": 1, "internal_early_stop_min_steps": 2000},
     "model": {"conditioner_variant": variant, "reference_encoder_in_dim": int(in_dim),
               "target_latent_hf_subband_fusion_enabled": True,
               "target_latent_hf_head_fusion_gate_init": 0.0},
     "bridge": {"reference_info_nce_weight": float(nce)},
     "data": {"virtual_length_multiplier": 5.6}}
if variant == "reference_hybrid":
    d["data"]["reference_dino_cache_dir"] = "/mnt/i/Latent_Style/dino_cache/content_5k"
json.dump(d, open(cfg, "w"), indent=2)
PY
  echo "===== $(date '+%F %T') TRAIN $id ($variant) =====" | tee -a "$L"
  timeout 3600 python3 run.py --config "$dir/config.json" > "$dir/train.log" 2>&1
  echo "[train rc=$?] $id" | tee -a "$L"
  echo "[lr_actual] $(grep -ao 'lr=[0-9.e+-]*' "$dir/train.log" | sort -u | tr '\n' ' ')" | tee -a "$L"
  grep -a "Epoch 11/11" "$dir/train.log" | tail -1 | tee -a "$L"
  grp=C
  [ "$variant" = "style_id" ] && grp=CTRL
  python3 /mnt/i/Latent_Style/runs/gate.py "$dir" "$grp" 2>&1 | tee -a "$L"
  ck="$dir/ckpt/epoch_0011.pt"
  if [ -f "$ck" ]; then
    reftok=""
    [ "$variant" = "reference_hybrid" ] && reftok=/mnt/i/Latent_Style/dino_cache/refs_test_first.pt
    bash /mnt/i/Latent_Style/runs/eval_one.sh "$id" "$ck" /mnt/i/Latent_Style/runs/routeC_eval \
      "$dir/config.json" "$reftok" 2>&1 | tee -a "$L"
  else
    echo "[NO CKPT] $id" | tee -a "$L"
  fi
}

run_one c_ref_s0    reference_hybrid 768 0.1 0
run_one c_styleid_s0 style_id        384 0.0 0
run_one c_ref_s1    reference_hybrid 768 0.1 1
run_one c_styleid_s1 style_id        384 0.0 1
echo "ROUTEC_RUN_DONE" | tee -a "$L"
