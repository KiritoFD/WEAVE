#!/usr/bin/env bash
# V11 P3: train the five-style conditional generator, then sample each style
# unconditionally and score it. Strictly serial, detached in tmux.
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SESSION=${1:-v11p3}
ROOT=runs/blog_sbflow/v11
C="/mnt/i/Latent_Style/pseudo_gt_24h/orig_mean_fp16/00_Early_Renaissance.pt
/mnt/i/Latent_Style/pseudo_gt_24h/orig_mean_fp16/01_Impressionism.pt
/mnt/i/Latent_Style/pseudo_gt_24h/orig_mean_fp16/02_Minimalism.pt
/mnt/i/Latent_Style/pseudo_gt_24h/orig_mean_fp16/03_Rococo.pt
/mnt/i/Latent_Style/pseudo_gt_24h/orig_mean_fp16/04_Ukiyo_e.pt"
mkdir -p "$ROOT/logs"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session '$SESSION' already exists"
  tmux list-panes -t "$SESSION" -F '  #{pane_id} #{pane_pid} #{pane_current_command}'
  exit 0
fi

tmux new-session -d -s "$SESSION" -n p3 "
  set -e
  python3 -u -m sbflow.train5 --config configs/v11/p3.json --caches $C \
    2>&1 | tee -a $ROOT/logs/p3_train.log
  CKPT=$ROOT/p3/model_equiv4000.pt
  python3 -u -m sbflow.gen_uncond --ckpt \$CKPT --out $ROOT/p3_gen \
    --per-style 100 --steps 4 --method heun --weights ema \
    2>&1 | tee -a $ROOT/logs/p3_gen.log
  python3 -u -m sbflow.score_uncond --gen $ROOT/p3_gen --out $ROOT/p3_score \
    --threads 4 2>&1 | tee -a $ROOT/logs/p3_score.log
  echo P3_DONE
"
sleep 3
echo "started tmux '$SESSION'"
tmux list-panes -t "$SESSION" -F '  #{pane_id} #{pane_pid} #{pane_current_command}'
