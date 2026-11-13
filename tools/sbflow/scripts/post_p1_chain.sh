#!/usr/bin/env bash
# Chain the post-P1 work so the GPU never idles:
#   1. wait for the v11_run driver to exit (all four arms trained+scored)
#   2. rho probe for d1 (2 updates, deterministic first batch)
#   3. P3: train the five-style conditional generator, sample each style, score it
#   4. trajectory-consistency check for d1 and the control
#
# P3 comes before the trajectory check on purpose: P3 answers the question that decides
# the research direction (can one shared model generate the five styles at all?), while
# the trajectory check only refines how d1/d2 are interpreted.
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ROOT=runs/blog_sbflow/v11
mkdir -p "$ROOT/logs"

echo "[chain] $(date +%H:%M:%S) waiting for the P1 driver to exit"
while pgrep -f "sbflow.v11_run" >/dev/null 2>&1; do
  sleep 120
done
echo "[chain] $(date +%H:%M:%S) P1 driver finished"

echo "[chain] $(date +%H:%M:%S) === rho probe (d1) ==="
bash -u sbflow/scripts/probe_rho.sh d1 2>&1 | tee -a "$ROOT/logs/probe_rho.log"

echo "[chain] $(date +%H:%M:%S) === P3: five-style conditional generation ==="
C="/mnt/i/Latent_Style/pseudo_gt_24h/orig_mean_fp16/00_Early_Renaissance.pt
/mnt/i/Latent_Style/pseudo_gt_24h/orig_mean_fp16/01_Impressionism.pt
/mnt/i/Latent_Style/pseudo_gt_24h/orig_mean_fp16/02_Minimalism.pt
/mnt/i/Latent_Style/pseudo_gt_24h/orig_mean_fp16/03_Rococo.pt
/mnt/i/Latent_Style/pseudo_gt_24h/orig_mean_fp16/04_Ukiyo_e.pt"
python3 -u -m sbflow.train5 --config configs/v11/p3.json --caches $C \
  2>&1 | tee -a "$ROOT/logs/p3_train.log"
python3 -u -m sbflow.gen_uncond --ckpt "$ROOT/p3/model_equiv4000.pt" \
  --out "$ROOT/p3_gen" --per-style 100 --steps 4 --method heun --weights ema \
  2>&1 | tee -a "$ROOT/logs/p3_gen.log"
python3 -u -m sbflow.score_uncond --gen "$ROOT/p3_gen" --out "$ROOT/p3_score" \
  --threads 4 2>&1 | tee -a "$ROOT/logs/p3_score.log"
echo "[chain] $(date +%H:%M:%S) P3 finished"

echo "[chain] $(date +%H:%M:%S) === trajectory check: d1 ==="
python3 -u -m sbflow.traj_check --ckpt "$ROOT/d1/model_step4000.pt" --tag d1 \
  --threads 4 2>&1 | tee -a "$ROOT/logs/traj_d1.log"

echo "[chain] $(date +%H:%M:%S) === trajectory check: control ==="
python3 -u -m sbflow.traj_check \
  --ckpt runs/blog_sbflow/suite40/exp01/model_equiv4000.pt --tag ctrl \
  --threads 4 2>&1 | tee -a "$ROOT/logs/traj_ctrl.log"

echo "[chain] $(date +%H:%M:%S) ALL POST-P1 WORK DONE"
