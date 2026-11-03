#!/usr/bin/env bash
# Hyperparameter-sufficiency queue (SOTA protocol: >=3 levels x 2 seeds per route).
# Order: A (slowest) -> C -> D. Each arm: train -> gate -> full 750 eval -> finalize.
set -u
cd /mnt/i/Latent_Style/WEAVE || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ROUTES="${1:-CDA}"

# ---------------- Route A: lambda_style sweep (apply_every=1, steps=2, b14/mb14) ----------
A_ROOT=/mnt/i/Latent_Style/runs/routeA_seed
mkdir -p "$A_ROOT"
A_LOG="$A_ROOT/scan.log"

run_a () {
  id="$1"; ls_="$2"; seed="$3"
  dir="$A_ROOT/$id"; mkdir -p "$dir"
  if [ -f "/mnt/i/Latent_Style/runs/routeA_seed_eval/$id/results_split.json" ]; then
    echo "[skip] $id" | tee -a "$A_LOG"; return 0
  fi
  python3 - "$dir/config.json" "$id" "$ls_" "$seed" <<'PY'
import json, sys
cfg, rid, ls_, seed = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
d = {"_base": "/mnt/i/Latent_Style/runs/nextgen_routeA_post/config_routeA.json",
     "checkpoint": {"save_dir": "/mnt/i/Latent_Style/runs/routeA_seed/%s/ckpt" % rid},
     "training": {"batch_size": 14, "num_epochs": 11, "seed": seed, "log_interval": 10,
                  "use_tqdm": False, "resume_optimizer": False,
                  "learning_rate": 5e-5, "min_learning_rate": 5e-5,
                  "resume_checkpoint": "exp/dino_s_break/brk_a_ll03_10ep/epoch_0010.pt",
                  "use_amp": True, "amp_dtype": "bf16",
                  "numeric_debug": True, "numeric_debug_interval": 50, "numeric_debug_dump_limit": 200},
     "bridge": {
        "output_supervision_enabled": True,
        "output_supervision_extractor": "dinov2",
        "output_supervision_extractor_local_dir": "/mnt/i/Latent_Style/weights/dinov2-base",
        "output_supervision_feature_res": 384,
        "output_supervision_chunk_rows": 4,
        "output_supervision_patch_size": 3,
        "output_supervision_patch_stride": 3,
        "output_supervision_style_layers": "block4,block8,block11",
        "output_supervision_content_layers": "block2,block4",
        "output_supervision_selfsim_layer": "block8",
        "output_supervision_ode_steps": 2,
        "output_supervision_apply_every": 1,
        "output_supervision_microbatch": 14,
        "output_supervision_lambda_style": ls_,
        "output_supervision_lambda_content": 0.1,
     },
     "data": {"virtual_length_multiplier": 2.8}}
json.dump(d, open(cfg, "w"), indent=2)
PY
  echo "===== $(date '+%F %T') TRAIN $id ls=$ls_ seed=$seed =====" | tee -a "$A_LOG"
  timeout 3600 python3 run.py --config "$dir/config.json" > "$dir/train.log" 2>&1
  echo "[train rc=$?] $id" | tee -a "$A_LOG"
  python3 /mnt/i/Latent_Style/runs/gate.py "$dir" A 2>&1 | tee -a "$A_LOG"
  ck="$dir/ckpt/epoch_0011.pt"
  if [ -f "$ck" ]; then
    bash /mnt/i/Latent_Style/runs/eval_one.sh "$id" "$ck" /mnt/i/Latent_Style/runs/routeA_seed_eval \
      "$dir/config.json" "" 2>&1 | tee -a "$A_LOG"
    python3 /mnt/i/Latent_Style/runs/append_registry.py --run-dir "$dir" \
      --eval-dir "/mnt/i/Latent_Style/runs/routeA_seed_eval/$id" --route A --note "lambda_style sweep" 2>&1 | cut -c1-200 | tee -a "$A_LOG"
  else
    echo "[NO CKPT] $id" | tee -a "$A_LOG"
  fi
}

if [[ "$ROUTES" == *A* ]]; then
for ls_ in 0.1 0.3 1.0; do
  for seed in 0 1; do
    run_a "a_ls${ls_:2}_s${seed}" "$ls_" "$seed"
  done
done
echo "ROUTEA_SEED_DONE" | tee -a "$A_LOG"
fi

# ---------------- Route C: NCE weight sweep (reference_hybrid) ----------
C_ROOT=/mnt/i/Latent_Style/runs/routeC_nce
C_LOG="$C_ROOT/scan.log"
mkdir -p "$C_ROOT"

run_c () {
  id="$1"; nce="$2"; seed="$3"
  dir="$C_ROOT/$id"; mkdir -p "$dir"
  if [ -f "/mnt/i/Latent_Style/runs/routeC_nce_eval/$id/results_split.json" ]; then
    echo "[skip] $id" | tee -a "$C_LOG"; return 0
  fi
  python3 - "$dir/config.json" "$id" "$nce" "$seed" <<'PY'
import json, sys
cfg, rid, nce, seed = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
d = {"_base": "/mnt/i/Latent_Style/WEAVE/config_d5_adain2.json",
     "checkpoint": {"save_dir": "/mnt/i/Latent_Style/runs/routeC_nce/%s/ckpt" % rid},
     "training": {"seed": seed, "batch_size": 14, "num_epochs": 11, "log_interval": 10,
                  "use_tqdm": False, "resume_optimizer": False,
                  "learning_rate": 5e-5, "min_learning_rate": 5e-5,
                  "resume_checkpoint": "exp/dino_s_break/brk_a_ll03_10ep/epoch_0010.pt",
                  "numeric_debug": True, "numeric_debug_interval": 50, "numeric_debug_dump_limit": 160},
     "model": {"conditioner_variant": "reference_hybrid", "reference_encoder_in_dim": 768},
     "bridge": {"reference_info_nce_weight": nce, "reference_info_nce_temperature": 0.1},
     "data": {"reference_dino_cache_dir": "/mnt/i/Latent_Style/dino_cache/content_5k",
              "virtual_length_multiplier": 5.6}}
json.dump(d, open(cfg, "w"), indent=2)
PY
  echo "===== $(date '+%F %T') TRAIN $id nce=$nce seed=$seed =====" | tee -a "$C_LOG"
  timeout 3600 python3 run.py --config "$dir/config.json" > "$dir/train.log" 2>&1
  echo "[train rc=$?] $id" | tee -a "$C_LOG"
  python3 /mnt/i/Latent_Style/runs/gate.py "$dir" C 2>&1 | tee -a "$C_LOG"
  ck="$dir/ckpt/epoch_0011.pt"
  if [ -f "$ck" ]; then
    bash /mnt/i/Latent_Style/runs/eval_one.sh "$id" "$ck" /mnt/i/Latent_Style/runs/routeC_nce_eval \
      "$dir/config.json" /mnt/i/Latent_Style/dino_cache/refs_test_first.pt 2>&1 | tee -a "$C_LOG"
    python3 /mnt/i/Latent_Style/runs/append_registry.py --run-dir "$dir" \
      --eval-dir "/mnt/i/Latent_Style/runs/routeC_nce_eval/$id" --route C --note "nce sweep" 2>&1 | cut -c1-200 | tee -a "$C_LOG"
  else
    echo "[NO CKPT] $id" | tee -a "$C_LOG"
  fi
}

if [[ "$ROUTES" == *C* ]]; then
for nce in 0.05 0.3; do
  for seed in 0 1; do
    run_c "c_nce${nce/./}_s${seed}" "$nce" "$seed"
  done
done
echo "ROUTEC_NCE_DONE" | tee -a "$C_LOG"
fi

# ---------------- Route D: texture hidden-width sweep ----------
D_ROOT=/mnt/i/Latent_Style/runs/routeD_hidden
D_LOG="$D_ROOT/scan.log"
mkdir -p "$D_ROOT"

run_d () {
  id="$1"; hid="$2"; seed="$3"
  dir="$D_ROOT/$id"; mkdir -p "$dir"
  if [ -f "/mnt/i/Latent_Style/runs/routeD_hidden_eval/$id/results_split.json" ]; then
    echo "[skip] $id" | tee -a "$D_LOG"; return 0
  fi
  python3 - "$dir/config.json" "$id" "$hid" "$seed" <<'PY'
import json, sys
cfg, rid, hid, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
d = {"_base": "/mnt/i/Latent_Style/WEAVE/config_d5_adain2.json",
     "checkpoint": {"save_dir": "/mnt/i/Latent_Style/runs/routeD_hidden/%s/ckpt" % rid},
     "training": {"seed": seed, "batch_size": 14, "num_epochs": 11, "log_interval": 10,
                  "use_tqdm": False, "resume_optimizer": False,
                  "learning_rate": 5e-5, "min_learning_rate": 5e-5,
                  "resume_checkpoint": "exp/dino_s_break/brk_a_ll03_10ep/epoch_0010.pt",
                  "numeric_debug": True, "numeric_debug_interval": 50, "numeric_debug_dump_limit": 160},
     "model": {"conditioner_variant": "style_id", "reference_encoder_in_dim": 384,
               "texture_refiner_enabled": True, "texture_refiner_hidden": hid},
     "data": {"virtual_length_multiplier": 5.6}}
json.dump(d, open(cfg, "w"), indent=2)
PY
  echo "===== $(date '+%F %T') TRAIN $id hidden=$hid seed=$seed =====" | tee -a "$D_LOG"
  timeout 3600 python3 run.py --config "$dir/config.json" > "$dir/train.log" 2>&1
  echo "[train rc=$?] $id" | tee -a "$D_LOG"
  python3 /mnt/i/Latent_Style/runs/gate.py "$dir" D 2>&1 | tee -a "$D_LOG"
  ck="$dir/ckpt/epoch_0011.pt"
  if [ -f "$ck" ]; then
    bash /mnt/i/Latent_Style/runs/eval_one.sh "$id" "$ck" /mnt/i/Latent_Style/runs/routeD_hidden_eval \
      "$dir/config.json" "" 2>&1 | tee -a "$D_LOG"
    python3 /mnt/i/Latent_Style/runs/append_registry.py --run-dir "$dir" \
      --eval-dir "/mnt/i/Latent_Style/runs/routeD_hidden_eval/$id" --route D --note "hidden sweep" 2>&1 | cut -c1-200 | tee -a "$D_LOG"
  else
    echo "[NO CKPT] $id" | tee -a "$D_LOG"
  fi
}

if [[ "$ROUTES" == *D* ]]; then
for hid in 16 64; do
  for seed in 0 1; do
    run_d "d_h${hid}_s${seed}" "$hid" "$seed"
  done
done
echo "ROUTED_HIDDEN_DONE" | tee -a "$D_LOG"
fi
echo "SWEEP_QUEUE_DONE"
