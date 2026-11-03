#!/usr/bin/env bash
# 20-step measured throughput with the new subband gate head present (gate=0).
# Route-A output supervision + dinov2@384 + chunked patches + bf16, batch 14.
set -u
cd /mnt/i/Latent_Style/WEAVE || exit 1
export HF_HOME=/mnt/i/Latent_Style/eval_cache/hf
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/mnt/i/Latent_Style/runs/p0_diag
tag=infra_bench_b14
dir="$R/$tag"; mkdir -p "$dir"
python3 - "$dir/config.json" <<'PY'
import json, sys
d = {"_base": "/mnt/i/Latent_Style/runs/nextgen_routeA_post/config_routeA.json",
     "checkpoint": {"save_dir": "/mnt/i/Latent_Style/runs/p0_diag/infra_bench_b14/ckpt"},
     "training": {"batch_size": 14, "num_epochs": 11, "seed": 0, "log_interval": 1,
                  "use_tqdm": False, "use_amp": True, "amp_dtype": "bf16",
                  "resume_optimizer": False, "stop_after_global_steps": 20,
                  "resume_checkpoint": "exp/dino_s_break/brk_a_ll03_10ep/epoch_0010.pt",
                  "learning_rate": 5e-5, "min_learning_rate": 5e-5, "save_interval": 999,
                  "internal_probe_enabled": False, "internal_early_stop_enabled": False},
     "model": {"target_latent_hf_subband_fusion_enabled": True,
               "target_latent_hf_head_fusion_gate_init": 0.0},
     "data": {"virtual_length_multiplier": 0.056}}
json.dump(d, open(sys.argv[1], 'w'), indent=2)
PY
echo "===== $(date '+%F %T') INFRA BENCH b14 20 steps =====" | tee -a "$R/infra_bench.log"
timeout 600 python3 run.py --config "$dir/config.json" > "$dir/train.log" 2>&1
echo "[rc=$?] $tag" | tee -a "$R/infra_bench.log"
grep -a "Epoch 11/11" "$dir/train.log" | tail -1 | tee -a "$R/infra_bench.log"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | tee -a "$R/infra_bench.log"
echo INFRA_BENCH_DONE | tee -a "$R/infra_bench.log"
