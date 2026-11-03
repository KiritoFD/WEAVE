#!/usr/bin/env bash
# 200-step smoke for the default-on internal early stop.
# Goal: prove it does NOT fire when it should not (min_steps not reached).
# Batch 4 => ~1.5GB, safe to run next to the P0 queue when free VRAM >= 4GB.
set -u
cd /mnt/i/Latent_Style/WEAVE || exit 1
export HF_HOME=/mnt/i/Latent_Style/eval_cache/hf
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/mnt/i/Latent_Style/runs/p0_diag
S="$R/es_smoke"; mkdir -p "$S"

python3 - "$S/config.json" <<'PY'
import json, sys
d = {"_base": "/mnt/i/Latent_Style/WEAVE/config.json",
     "checkpoint": {"save_dir": "/mnt/i/Latent_Style/runs/p0_diag/es_smoke/ckpt"},
     "training": {"batch_size": 4, "num_epochs": 11, "seed": 0, "log_interval": 10,
                  "use_tqdm": False, "use_amp": True, "amp_dtype": "bf16",
                  "resume_optimizer": False,
                  "learning_rate": 5e-5, "min_learning_rate": 5e-5,
                  "resume_checkpoint": "exp/dino_s_break/brk_a_ll03_10ep/epoch_0010.pt",
                  "save_interval": 1, "stop_after_global_steps": 200,
                  "internal_probe_enabled": True, "internal_early_stop_enabled": True,
                  "internal_probe_every": 1, "internal_early_stop_min_steps": 2000},
     "model": {"target_latent_hf_subband_fusion_enabled": True,
               "target_latent_hf_head_fusion_gate_init": 0.0},
     "data": {"virtual_length_multiplier": 0.16}}
json.dump(d, open(sys.argv[1], 'w'), indent=2)
PY

for _ in $(seq 1 60); do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  [ "${free:-0}" -ge 4096 ] && break
  echo "[wait] free VRAM ${free}MB < 4096MB"; sleep 30
done

echo "===== $(date '+%F %T') ES SMOKE 200 steps =====" | tee -a "$R/es_smoke.log"
timeout 900 python3 run.py --config "$S/config.json" > "$S/train.log" 2>&1
rc=$?
echo "[train rc=$rc] es_smoke" | tee -a "$R/es_smoke.log"
grep -a "Internal early stop summary" "$S/train.log" | tail -1 | tee -a "$R/es_smoke.log"

python3 - "$S/ckpt/internal_dynamics_summary.json" "$S/ckpt/internal_dynamics.jsonl" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1]))
print(json.dumps(summary, ensure_ascii=False, indent=2))
assert summary["armed"] is True, "early stop must be armed"
assert summary["fired"] is False, "early stop must NOT fire before min_steps"
assert summary["stop_reason"] == "min_steps 未达", summary["stop_reason"]
assert len(summary["probe_epochs"]) == 1, summary["probe_epochs"]
rows = [line for line in open(sys.argv[2]) if line.strip()]
assert len(rows) == 1, len(rows)
print("[OK] early stop did not fire; summary written")
PY
assert_rc=$?
if [ "$assert_rc" -eq 0 ]; then
  date '+%F %T' > "$R/es_smoke.ok"
  echo "[OK] wrote $R/es_smoke.ok" | tee -a "$R/es_smoke.log"
else
  echo "[FAIL] smoke assertions failed" | tee -a "$R/es_smoke.log"
fi
exit "$assert_rc"
