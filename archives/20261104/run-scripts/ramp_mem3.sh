#!/usr/bin/env bash
# Third ramp: find the highest batch that stays under ~10GB peak with the ODE
# unroll fully parallel (microbatch == batch).
set -u
cd /mnt/i/Latent_Style/WEAVE || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/mnt/i/Latent_Style/runs/routeA_mem_ramp3
mkdir -p "$R"
: > "$R/ramp.jsonl"
for pair in 10:10 12:12 12:4 14:1 14:14; do
  b=${pair%%:*}; mb=${pair##*:}
  vlm=$(python3 -c "print(20*$b/5000)")
  cfg="$R/config_b${b}_mb${mb}.json"
  log="$R/b${b}_mb${mb}.log"
  python3 - "$cfg" "$b" "$mb" "$vlm" <<'PY'
import json, sys
cfg, b, mb, vlm = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
d = {"_base": "/mnt/i/Latent_Style/runs/nextgen_routeA_post/config_routeA.json",
     "checkpoint": {"save_dir": "/mnt/i/Latent_Style/runs/routeA_mem_ramp3/ckpt_b%d_mb%d" % (b, mb)},
     "training": {"batch_size": b, "num_epochs": 11, "log_interval": 1,
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
        "output_supervision_microbatch": mb,
     },
     "data": {"virtual_length_multiplier": vlm}}
json.dump(d, open(cfg, "w"), indent=2)
PY
  echo "=== B${b} MB${mb} ===" | tee -a "$R/ramp.log"
  start=$(date +%s)
  timeout 900 python3 run.py --config "$cfg" > "$log" 2>&1
  rc=$?; end=$(date +%s)
  line=$(grep -a "Epoch 11/11" "$log" | tail -1)
  oom=$(grep -ac "out of memory" "$log" || true)
  echo "rc=$rc oom=$oom wall=$((end-start))s $line" >> "$R/ramp.log"
  python3 - "$R" "$b" "$mb" "$rc" "$oom" "$((end-start))" "$line" <<'PY'
import json, re, sys
R, b, mb, rc, oom, dur, line = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), sys.argv[7]
rec = {"batch": b, "microbatch": mb, "rc": rc, "oom_hits": oom, "wall_s": dur}
m = re.search(r"comp=([0-9.]+)s epoch=([0-9.]+)s sps=([0-9.]+) bstep=([0-9.]+)s.*peak=([0-9.]+)/", line)
if m:
    rec.update({"comp_s": float(m.group(1)), "epoch_s": float(m.group(2)), "sps": float(m.group(3)),
                "step_s": float(m.group(4)), "peak_gb": float(m.group(5)), "img_s": float(m.group(3)) * b})
else:
    rec.update({"error": (line or "no epoch line")[-160:]})
open(R + "/ramp.jsonl", "a").write(json.dumps(rec) + "\n")
PY
  if [ "$oom" -gt 0 ]; then
    echo "OOM at b=$b mb=$mb" | tee -a "$R/ramp.log"
  fi
done
echo RAMP3_DONE | tee -a "$R/ramp.log"
cat "$R/ramp.jsonl"
