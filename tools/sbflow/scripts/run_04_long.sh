#!/usr/bin/env bash
# Long reference run for the five-style model.
#
# The screening arms only get 50-320 updates per style pair, which is 12-80x below the
# per-pair budget that produced usable transfer in V9 (about 4000 updates on a single
# pair). Before spending hours on 38 undertrained arms, establish what the five-style
# model can actually do with a real budget.
#
# 6000 updates / 10 pairs = 600 updates per pair, batch 512, patch 8.
# Measured 0.375 s/update -> about 38 minutes.
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ROOT=runs/blog_sbflow/v10
C="/mnt/i/Latent_Style/pseudo_gt_24h/orig_mean_fp16/00_Early_Renaissance.pt \
/mnt/i/Latent_Style/pseudo_gt_24h/orig_mean_fp16/01_Impressionism.pt \
/mnt/i/Latent_Style/pseudo_gt_24h/orig_mean_fp16/02_Minimalism.pt \
/mnt/i/Latent_Style/pseudo_gt_24h/orig_mean_fp16/03_Rococo.pt \
/mnt/i/Latent_Style/pseudo_gt_24h/orig_mean_fp16/04_Ukiyo_e.pt"

STEPS=${1:-6000}
OUT="$ROOT/exp_04L"
mkdir -p "$ROOT/logs"

echo "[04L] $(date +%H:%M:%S) training $STEPS updates (batch 512, patch 8)"
python3 -u -m sbflow.train5 --experiment-id 04L --caches $C --output "$OUT" \
  --steps "$STEPS" --batch 512 --patch 8 --width 256 --depth 6 --heads 4 \
  --objective bidir --eps 0.04 --warmup 200 --lr 2e-4 \
  --log-interval 250 --save-interval 1000 \
  2>&1 | tee -a "$ROOT/logs/04L_train.log"

CK="$OUT/model_step$STEPS.pt"
if [ ! -f "$CK" ]; then echo "[04L] no checkpoint, abort"; exit 1; fi

echo "[04L] $(date +%H:%M:%S) generating quick200"
python3 -u -m sbflow.gen5 --repo-root . --ckpt "$CK" \
  --out "$ROOT/gen_04L_quick200" --subset quick200 --steps 4 --method heun --batch 16 \
  2>&1 | tee -a "$ROOT/logs/04L_gen.log"

echo "[04L] $(date +%H:%M:%S) CPU scoring quick200"
python3 -u -m sbflow.score_cpu --repo-root . \
  --gen "$ROOT/gen_04L_quick200" --out "$ROOT/score_04L_quick200" \
  --batch 8 --threads 6 2>&1 | tee -a "$ROOT/logs/04L_score.log"

echo "[04L] $(date +%H:%M:%S) DONE"
python3 - <<'PY'
import json
from pathlib import Path
p = Path("runs/blog_sbflow/v10/score_04L_quick200/style_scores.json")
if p.exists():
    d = json.loads(p.read_text())
    o = d["aggregate"]["overall"]
    print("overall:", {k: round(v, 4) for k, v in o.items() if isinstance(v, float)})
    for name, v in sorted(d["aggregate"]["by_direction"].items()):
        print(f"  {name:44s} top1={v['target_top1']:.3f} dT={v['csd_dT']:+.4f} "
              f"dS={v['csd_dS']:+.4f} swd={v['csd_swd']:.4f}")
PY
