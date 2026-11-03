#!/usr/bin/env bash
# Route A memory-optimised throughput ramp:
#   DINOv2-base feature branch + 384 loss resolution + chunked patch descriptors + bf16 AMP.
# Each batch size runs 12 optimizer steps; OOM stops the sweep.
set -u
cd /mnt/i/Latent_Style/WEAVE || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/mnt/i/Latent_Style/runs/routeA_mem_ramp
mkdir -p "$R"
: > "$R/ramp.jsonl"
for b in 1 2 4 6; do
  vlm=$(python3 -c "print(12*$b/5000)")
  cfg="$R/config_b$b.json"
  python3 - "$cfg" "$b" "$vlm" <<'PY'
import json, sys
cfg, b, vlm = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
d = {
    "_base": "/mnt/i/Latent_Style/runs/nextgen_routeA_post/config_routeA.json",
    "checkpoint": {"save_dir": "/mnt/i/Latent_Style/runs/routeA_mem_ramp/ckpt_b%d" % b},
    "training": {"batch_size": b, "num_epochs": 12, "log_interval": 1,
                 "use_amp": True, "amp_dtype": "bf16"},
    "bridge": {
        "output_supervision_extractor": "dinov2",
        "output_supervision_extractor_local_dir": "/mnt/i/Latent_Style/weights/dinov2-base",
        "output_supervision_feature_res": 384,
        "output_supervision_chunk_rows": 4,
        "output_supervision_patch_size": 3,
        "output_supervision_patch_stride": 3,
        "output_supervision_style_layers": "block4,block8,block11",
        "output_supervision_content_layers": "block2,block4",
        "output_supervision_selfsim_layer": "block8",
    },
    "data": {"virtual_length_multiplier": vlm},
}
json.dump(d, open(cfg, "w"), indent=2)
PY
  log="$R/b$b.log"
  echo "=== BATCH $b ===" | tee -a "$R/ramp.log"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader >> "$R/ramp.log"
  start=$(date +%s)
  timeout 1200 python3 run.py --config "$cfg" > "$log" 2>&1
  rc=$?
  end=$(date +%s)
  line=$(grep -a "Epoch 11/12" "$log" | tail -1)
  oom=$(grep -ac "out of memory" "$log" || true)
  echo "rc=$rc oom=$oom wall=$((end-start))s" >> "$R/ramp.log"
  echo "$line" >> "$R/ramp.log"
  python3 - "$R" "$b" "$rc" "$oom" "$((end-start))" "$line" <<'PY'
import json, re, sys
R, b, rc, oom, dur, line = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
rec = {"batch": b, "rc": rc, "oom_hits": oom, "wall_s": dur}
m = re.search(r"comp=([0-9.]+)s epoch=([0-9.]+)s sps=([0-9.]+) bstep=([0-9.]+)s.*peak=([0-9.]+)/", line)
if m:
    rec.update({"comp_s": float(m.group(1)), "epoch_s": float(m.group(2)), "sps": float(m.group(3)),
                "step_s": float(m.group(4)), "peak_gb": float(m.group(5)), "batch_sps": float(m.group(3)) * b})
else:
    rec.update({"error": (line or "no epoch line")[-200:]})
open(R + "/ramp.jsonl", "a").write(json.dumps(rec) + "\n")
PY
  echo "---" >> "$R/ramp.log"
  if [ "$oom" -gt 0 ]; then
    echo "OOM at batch $b, stop" | tee -a "$R/ramp.log"
    break
  fi
done
echo RAMP_MEM_DONE | tee -a "$R/ramp.log"
cat "$R/ramp.jsonl"
