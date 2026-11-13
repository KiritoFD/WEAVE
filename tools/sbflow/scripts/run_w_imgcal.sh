#!/usr/bin/env bash
# W P3 — W_S10_ImgCalib: 150-step frozen A0-Encoder + S10 decoder RGB-domain tuning.
set -euo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export QLOG=runs/v6_w/queue_w_imgcal.log

mkdir -p runs/v6_w
source sbflow/scripts/q_lib.sh
wait_for_gpu 240 || exit 3
qlog "==================== WEAVE W S10 IMG-CALIB ===================="

BEST=$(python3 -m tools.v6_next.w_select_best)
qlog "Best W student: $BEST"

OUT="runs/v6_w/W5_ImgCalib/W5_ImgCalib.pt"
if [ -s "$OUT" ]; then
  qlog "===== W5 S10 ImgCalib ALREADY DONE ====="
else
  # microbatch=6 OOMs on the 12GB 3060 (S10 decoder + CSD ViT-L + DINO + grads).
  # The 2-step smoke used microbatch=2 at 6874MiB peak, so keep <=2 here.
  run_step "W5_imgcalib" "$OUT" \
    python3 -u -m tools.v6_next.w_img_calib \
      --init-ckpt "$BEST" --name W5_ImgCalib --out-base runs/v6_w \
      --steps 150 --microbatch 2 --lr 5e-5 --lam-csd 1.0 --lam-dino-c 1.0 --seed 42
fi

qlog "==================== W S10 IMG-CALIB DONE failures=$QFAIL ===================="
