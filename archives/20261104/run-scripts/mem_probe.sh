#!/usr/bin/env bash
# VRAM decomposition v2: keep physical batch large, shrink only the supervision path.
# usage: run_case <tag> <batch> <steps> <mode flow|vgg|dino> <microbatch>
set -u
cd /mnt/i/Latent_Style/WEAVE || exit 1
export HF_HOME=/mnt/i/Latent_Style/eval_cache/hf
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/mnt/i/Latent_Style/runs/p0_diag/mem_probe
mkdir -p "$R"
CKPT=exp/dino_s_break/brk_a_ll03_10ep/epoch_0010.pt
: > "$R/mem2.jsonl"

run_case () {
  tag="$1"; b="$2"; steps="$3"; mode="$4"; mb="$5"
  cfg="$R/config_$tag.json"
  vlm=$(python3 -c "print($steps*$b/160.0)")
  python3 - "$cfg" "$b" "$vlm" "$steps" "$CKPT" "$mode" "$mb" <<'PY'
import json, sys
cfg, b, vlm, steps, ckpt, mode, mb = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6], int(sys.argv[7])
d = {"_base": "/mnt/i/Latent_Style/runs/nextgen_routeA_post/config_routeA.json",
     "checkpoint": {"save_dir": "/mnt/i/Latent_Style/runs/p0_diag/mem_probe/ckpt_%s" % mode},
     "training": {"batch_size": b, "num_epochs": 11, "seed": 0, "log_interval": max(1, steps // 5),
                  "use_tqdm": False, "use_amp": True, "amp_dtype": "bf16",
                  "resume_checkpoint": ckpt, "resume_prefer_local_checkpoint": False,
                  "resume_optimizer": False,
                  "learning_rate": 5e-5, "min_learning_rate": 5e-5,
                  "stop_after_global_steps": steps, "save_interval": 999},
     "model": {"target_latent_hf_subband_fusion_enabled": True,
               "target_latent_hf_head_fusion_gate_init": 0.0},
     "data": {"packed_limit_per_style": 32, "virtual_length_multiplier": vlm}}
br = {"output_supervision_microbatch": mb, "output_supervision_grad_checkpoint": True}
if mode == "vgg":
    br.update({"output_supervision_enabled": True, "output_supervision_extractor": "vgg16",
               "output_supervision_decoder": "taesd",
               "output_supervision_style_layers": "relu2_2,relu3_3",
               "output_supervision_content_layers": "relu1_2,relu2_2",
               "output_supervision_selfsim_layer": "relu3_3",
               "output_supervision_ode_steps": 2,
               "output_supervision_lambda_style": 1.0,
               "output_supervision_lambda_content": 0.5})
elif mode == "dino":
    br.update({"output_supervision_enabled": True, "output_supervision_extractor": "dinov2",
               "output_supervision_extractor_local_dir": "/mnt/i/Latent_Style/weights/dinov2-base",
               "output_supervision_feature_res": 384, "output_supervision_chunk_rows": 4,
               "output_supervision_patch_size": 3, "output_supervision_patch_stride": 3,
               "output_supervision_style_layers": "block4,block8,block11",
               "output_supervision_content_layers": "block2,block4",
               "output_supervision_selfsim_layer": "block8",
               "output_supervision_ode_steps": 2,
               "output_supervision_lambda_style": 1.0,
               "output_supervision_lambda_content": 0.5})
else:
    br.update({"output_supervision_enabled": False})
d["bridge"] = br
json.dump(d, open(cfg, "w"), indent=2)
PY

  echo "== $tag batch=$b mb=$mb steps=$steps mode=$mode vlm=$vlm $(date '+%T') ==" | tee -a "$R/mem2.log"
  start=$(date +%s)
  python3 run.py --config "$cfg" > "$R/$tag.log" 2>&1 &
  pid=$!
  peak=0
  while kill -0 "$pid" 2>/dev/null; do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
    [ "${u:-0}" -gt "$peak" ] && peak=$u
    sleep 1
  done
  wait "$pid"; rc=$?
  dur=$(( $(date +%s) - start ))
  python3 - "$tag" "$b" "$steps" "$peak" "$dur" "$rc" "$R/$tag.log" "$R/mem2.jsonl" <<'PY'
import json, sys, re
tag, b, steps, peak, dur, rc, log, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), sys.argv[7], sys.argv[8]
text = open(log, errors="ignore").read()
m = re.findall(r"epoch=([0-9.]+)s", text)
epoch_s = float(m[-1]) if m else 0.0
bstep = epoch_s / steps if epoch_s > 0 else max(0.0, (dur - 8) / max(steps, 1))
oom = "out of memory" in text.lower() or "OutOfMemoryError" in text
row = {"tag": tag, "batch": b, "steps": steps, "peak_used_mib": peak,
       "peak_gb": round(peak / 1024, 2), "s_per_step": round(bstep, 3),
       "s_per_sample": round(bstep / b, 4), "wall_s": dur, "rc": rc, "oom": bool(oom)}
print("RESULT", json.dumps(row))
open(out, "a").write(json.dumps(row) + "\n")
PY
  grep -a "Epoch 11/11" "$R/$tag.log" | tail -1 | cut -c1-200 | tee -a "$R/mem2.log"
}

run_case flow_b180    180 25 flow 180
echo MEM_PROBE5_DONE | tee -a "$R/mem2.log"
