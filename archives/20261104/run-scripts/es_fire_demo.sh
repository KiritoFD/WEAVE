#!/usr/bin/env bash
# Infra evidence run: early-stop "fired" demo at large physical batch,
# closed loop = train -> full 750 eval (big eval batch) -> dino -> gate -> registry.
# Selected batch from runs/mem_probe (flow-only path): b180 = 9.66G, 0.24 s/step.
set -u
cd /mnt/i/Latent_Style/WEAVE || exit 1
source runs/vram_guard.sh
export HF_HOME=/mnt/i/Latent_Style/eval_cache/hf
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT=/mnt/i/Latent_Style/runs/p0_diag/es_fire_big
EVAL_ROOT=/mnt/i/Latent_Style/runs/p0_diag/es_fire_big_eval
# Selected batch: probes on the flow-only path gave b180=9.43-9.66G but the real run
# OOM'd during backward (probe under-reads by ~2G), so step down to b160.
BATCH=${BATCH:-160}
STEPS_PER_EPOCH=${STEPS_PER_EPOCH:-500}
EPOCHS=${EPOCHS:-10}
EVAL_BATCH=${EVAL_BATCH:-32}
DINO_BATCH=${DINO_BATCH:-32}
VLM=$(python3 -c "print($STEPS_PER_EPOCH*$BATCH/160.0)")
LR=$(python3 -c "print(5e-5*($BATCH/14.0)**0.5)")
mkdir -p "$OUT" "$EVAL_ROOT"
rm -rf "$OUT/ckpt" "$OUT/train.log" "$OUT/internal_dynamics_summary.final.json"

cfg="$OUT/config.json"
python3 - "$cfg" "$BATCH" "$VLM" "$EPOCHS" "$LR" "$OUT" <<'PY'
import json, sys
cfg, b, vlm, epochs, lr, out = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]), sys.argv[6]
d = {"_base": "/mnt/i/Latent_Style/WEAVE/config.json",
     "checkpoint": {"save_dir": out + "/ckpt"},
     "training": {"batch_size": b, "num_epochs": epochs, "seed": 0, "log_interval": 25,
                  "use_tqdm": False, "use_amp": True, "amp_dtype": "bf16",
                  "resume_checkpoint": "", "resume_optimizer": False,
                  "learning_rate": lr, "min_learning_rate": lr,
                  "save_interval": 1,
                  "internal_probe_enabled": True, "internal_probe_every": 1,
                  "internal_early_stop_enabled": True, "internal_early_stop_record_only": False,
                  "internal_early_stop_min_epoch": 3, "internal_early_stop_min_steps": 2000,
                  "internal_early_stop_gate_delta_threshold": 0.0,
                  "internal_early_stop_shared_ratio_drop_threshold": 0.65},
     "model": {"target_latent_hf_subband_fusion_enabled": True,
               "target_latent_hf_head_fusion_gate_init": 0.05},
     "data": {"packed_limit_per_style": 32, "virtual_length_multiplier": vlm}}
json.dump(d, open(cfg, "w"), indent=2)
PY
echo "CONFIG batch=$BATCH vlm=$VLM lr=$LR epochs=$EPOCHS steps/epoch=$STEPS_PER_EPOCH"

budget_start "$OUT" "$BATCH" 8000 9728 10752 || exit 1

python3 run.py --config "$cfg" > "$OUT/train.log" 2>&1 &
pid=$!
guard_watch "$pid" "$OUT" 8000
grc=$?
if [ $grc -ne 0 ]; then echo "GUARD_FAILED: low VRAM, run killed"; exit 1; fi
wait "$pid"; rc=$?
echo "TRAIN_RC=$rc"
cp "$OUT/ckpt/internal_dynamics_summary.json" "$OUT/internal_dynamics_summary.final.json" 2>/dev/null

python3 - "$OUT" <<'PY'
import json, sys
p = sys.argv[1] + "/ckpt/internal_dynamics_summary.json"
try:
    d = json.load(open(p))
except Exception as exc:
    print("SUMMARY_MISSING", exc); raise SystemExit(0)
print("EARLY_STOP", json.dumps({k: d.get(k) for k in
      ("armed", "record_only", "fired", "would_fire", "fired_epoch", "stop_reason", "global_step",
       "rho_series", "gate_mean_series")}, ensure_ascii=False))
PY

ckpt=$(ls -1 "$OUT/ckpt"/epoch_*.pt 2>/dev/null | sort | tail -1)
echo "CKPT=$ckpt"
if [ -n "${ckpt:-}" ]; then
  EVAL_BATCH="$EVAL_BATCH" DINO_BATCH="$DINO_BATCH" bash runs/eval_one.sh \
    es_fire_big "$ckpt" "$EVAL_ROOT" "$cfg" "" data/test
fi

python3 runs/gate.py "$OUT" A > "$OUT/gate.stdout" 2>&1 || true
python3 runs/append_registry.py --run-dir "$OUT" --eval-dir "$EVAL_ROOT/es_fire_big" \
  --route A --valid no \
  --invalid-reason "infra early-stop fired demo: base config has no Route-A output supervision" \
  | tee "$OUT/registry.stdout"
echo ES_FIRE_BIG_DONE
