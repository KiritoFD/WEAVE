#!/usr/bin/env bash
# New five-style baseline with MIXED batches: all 10 pairs and both directions in every
# update, equal-weight-per-style global statistics.
#
# 6000 updates at batch 512 gives each of the 20 (pair, direction) cells 300 updates
# and 153,600 sample exposures; each style pair sees 307,200 exposures total, which is
# above the V9 single-pair budget of 256,000.
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
BATCH=${2:-512}
TAG=${3:-04M}
OUT="$ROOT/exp_$TAG"
mkdir -p "$ROOT/logs"

echo "[$TAG] $(date +%H:%M:%S) mixed-batch training: steps=$STEPS batch=$BATCH patch=8"
python3 -u -m sbflow.train5 --experiment-id "$TAG" --caches $C --output "$OUT" \
  --steps "$STEPS" --batch "$BATCH" --patch 8 --width 256 --depth 6 --heads 4 \
  --objective bidir --eps 0.04 --warmup 200 --lr 2e-4 \
  --mixed-batch --equal-weight-stats \
  --log-interval 250 --save-interval 1000 \
  2>&1 | tee -a "$ROOT/logs/${TAG}_train.log"

CK="$OUT/model_step$STEPS.pt"
[ -f "$CK" ] || { echo "[$TAG] no checkpoint, abort"; exit 1; }

echo "[$TAG] $(date +%H:%M:%S) generating quick200"
python3 -u -m sbflow.gen5 --repo-root . --ckpt "$CK" \
  --out "$ROOT/gen_${TAG}_quick200" --subset quick200 --steps 4 --method heun --batch 16 \
  2>&1 | tee -a "$ROOT/logs/${TAG}_gen.log"

echo "[$TAG] $(date +%H:%M:%S) CPU scoring quick200"
python3 -u -m sbflow.score_cpu --repo-root . \
  --gen "$ROOT/gen_${TAG}_quick200" --out "$ROOT/score_${TAG}_quick200" \
  --batch 8 --threads 6 2>&1 | tee -a "$ROOT/logs/${TAG}_score.log"

echo "[$TAG] $(date +%H:%M:%S) DONE"
python3 - <<PY
import json
from pathlib import Path
p = Path("runs/blog_sbflow/v10/score_${TAG}_quick200/style_scores.json")
if p.exists():
    d = json.loads(p.read_text())
    o = d["aggregate"]["overall"]
    print("overall:", {k: round(v, 4) for k, v in o.items() if isinstance(v, float)})
    print("by target:")
    for k, v in sorted(d["aggregate"]["by_target"].items()):
        print(f"  {k:22s} top1={v['target_top1']:.3f} dT={v['csd_dT']:+.4f} dS={v['csd_dS']:+.4f}")
    pos = sum(1 for v in d["aggregate"]["by_direction"].values() if v["csd_dT"] > 0)
    print(f"directions with positive dT: {pos}/20")
PY
