#!/usr/bin/env bash
# Route C smoke: reference-hybrid conditioning + InfoNCE from DINOv2 patch cache.
# 100 steps only; checks that loss_reference_nce is non-zero and the reference
# encoder participates in forward before spending GPU time on the full run.
set -u
cd /mnt/i/Latent_Style/WEAVE || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/mnt/i/Latent_Style/runs/routeC_smoke
mkdir -p "$R"
python3 - "$R/config.json" <<'PY'
import json, sys
d = {"_base": "/mnt/i/Latent_Style/WEAVE/config_d5_adain2.json",
     "checkpoint": {"save_dir": "/mnt/i/Latent_Style/runs/routeC_smoke/ckpt"},
     "training": {"seed": 0, "batch_size": 14, "num_epochs": 11, "log_interval": 5,
                  "use_tqdm": False, "resume_optimizer": False,
                  "learning_rate": 1e-4, "min_learning_rate": 1e-4,
                  "resume_checkpoint": "exp/dino_s_break/brk_a_ll03_10ep/epoch_0010.pt",
                  "numeric_debug": True, "numeric_debug_interval": 10, "numeric_debug_dump_limit": 60},
     "model": {"conditioner_variant": "reference_hybrid", "reference_encoder_in_dim": 768},
     "bridge": {"reference_info_nce_weight": 0.1, "reference_info_nce_temperature": 0.1},
     "data": {"virtual_length_multiplier": 0.14,
              "reference_dino_cache_dir": "/mnt/i/Latent_Style/dino_cache/content_5k"}}
json.dump(d, open(sys.argv[1], "w"), indent=2)
PY
timeout 1800 python3 run.py --config "$R/config.json" > "$R/train.log" 2>&1
echo "[train rc=$?]"
grep -a "Route C\|reference\|reference_feats" "$R/train.log" | head -5
python3 /mnt/i/Latent_Style/runs/gate.py "$R" 2>&1
echo "--- nce check ---"
python3 - "$R" <<'PY'
import glob, json, os, sys
R = sys.argv[1]
vals = []
for f in glob.glob(os.path.join(R, "numeric_debug", "*.jsonl")):
    for line in open(f):
        try:
            rec = json.loads(line)
        except Exception:
            continue
        for k, v in (rec.get("loss_dict") or rec).items():
            if "nce" in str(k) and isinstance(v, (int, float)):
                vals.append(float(v))
print("nce_values_found", len(vals), "min", min(vals) if vals else None, "max", max(vals) if vals else None)
print("NCE_CHECK", "PASS" if vals and max(vals) > 0 else "FAIL")
PY
echo ROUTEC_SMOKE_DONE
