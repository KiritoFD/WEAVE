#!/usr/bin/env bash
# WEAVE W — student distillation matrix + S10 image-domain calibration.
# All arms share the exact R2 training recipe (800 steps, effective batch 240, bf16,
# seed 42, c1_gentle). The ONLY thing that changes is the distillation loss / teacher.
set -euo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export QLOG=runs/v6_w/queue_w_train.log

mkdir -p runs/v6_w
source sbflow/scripts/q_lib.sh
wait_for_gpu 240 || exit 3
qlog "==================== WEAVE W TRAINING MATRIX ===================="

COMMON="--mode direct --out-base runs/v6_w --steps 800 --lr 0.0003 --microbatch 80 --accum 3 --amp-dtype bf16 --canon-mode c1_gentle --w-rec 1.0 --lambda-cross 0.35 --lambda-canon 0.525 --w-style-dist 0.25 --ref-top-k 1024 --seed 42"
TEACH="--distill-teacher runs/v6/Q2a_embed_style/Q2a_embed_style.pt --distill-critic runs/v6_next/probes/r1_r1.pt --distill-mode oracle --distill-steps 1 --distill-rms 0.25"

# train_arm <name> <init> <base_dim> <lambda> <extra...>
train_arm () {
  local name=$1 init=$2 dim=$3 lam=$4; shift 4
  local ck="runs/v6_w/${name}/${name}.pt"
  if [ -s "$ck" ]; then qlog "===== $name ALREADY TRAINED ====="; return 0; fi
  run_step "train_$name" "$ck" \
    python3 -u -m tools.v6.train_eval_v6 \
      --name "$name" --base-dim "$dim" --init-from "$init" \
      $TEACH --distill-lambda "$lam" $COMMON "$@"
}

Q2A=runs/v6/Q2a_embed_style/Q2a_embed_style.pt
L8=runs/v6/L8_k1024_dim32/L8_k1024_dim32.pt

# ---- P1: W_FeatDistill (experiment 1 + 3): R1 feature/KL/repulsion/LL, class RMS, anti-attractor
train_arm W1_FeatDistill_218 "$Q2A" 16 20 \
  --w-distill --w-feat 1.0 --w-kl 1.0 --w-rep 0.5 --w-ll 1.0 \
  --w-class-rms --w-anti-attractor

# ---- P2: W_LossClean (experiment 2): same + disable old DWT cross conflict + LL-only canon
train_arm W2_LossClean_218 "$Q2A" 16 20 \
  --w-distill --w-feat 1.0 --w-kl 1.0 --w-rep 0.5 --w-ll 1.0 \
  --w-class-rms --w-anti-attractor --w-clean-cross --w-ll-canon

# ---- P2b: LossClean at lambda=50 (supervision strength)
train_arm W3_LossClean_218_l50 "$Q2A" 16 50 \
  --w-distill --w-feat 1.0 --w-kl 1.0 --w-rep 0.5 --w-ll 1.0 \
  --w-class-rms --w-anti-attractor --w-clean-cross --w-ll-canon

# ---- P2c: 845K(dim32) + Style Keys (experiment 5) — DISABLED after a measured
#      50-step stability check: loss rose 3.59 -> 5.55 (vs 218K arms 2.06 -> 1.92).
#      Style Keys + strong feature distill interact badly at this width/lr; the plan
#      explicitly says to cut low-priority arms under stability/budget pressure.
#      Recorded as a negative result in W_REPORT.md. Re-enable only with a dedicated
#      lr/regularisation study.
# train_arm W4_LossClean_845_SK "$L8" 32 20 \
#   --w-distill ... --style-keys --style-keys-rank 32


# ---- P3: W_S10_ImgCalib (experiment 4): 150-step frozen A0-Encoder + S10 decoder
#      RGB-domain CSD margin + DINO-C calibration on the best P2 student.
#      Deferred to after P4 eval so we calibrate whichever arm actually wins.
qlog "==================== W TRAINING MATRIX DONE failures=$QFAIL ===================="
