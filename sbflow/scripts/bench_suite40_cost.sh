#!/usr/bin/env bash
# Real-data 50-step speed calibration for every distinct cost class in suite40.
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
C="/mnt/i/Latent_Style/pseudo_gt_24h/orig_mean_fp16/00_Early_Renaissance.pt \
/mnt/i/Latent_Style/pseudo_gt_24h/orig_mean_fp16/01_Impressionism.pt \
/mnt/i/Latent_Style/pseudo_gt_24h/orig_mean_fp16/02_Minimalism.pt \
/mnt/i/Latent_Style/pseudo_gt_24h/orig_mean_fp16/03_Rococo.pt \
/mnt/i/Latent_Style/pseudo_gt_24h/orig_mean_fp16/04_Ukiyo_e.pt"
ROOT=runs/blog_sbflow/suite40/_bench
mkdir -p "$ROOT"

bench() {  # name extra...
  local NAME=$1; shift
  echo "=== $NAME ==="
  rm -rf "$ROOT/$NAME"
  python3 -u -m sbflow.train5 --experiment-id "$NAME" --caches $C \
    --output "$ROOT/$NAME" --steps 50 --batch 512 --patch 8 \
    --width 256 --depth 6 --heads 4 --warmup 5 --lr-cycle-steps 4000 \
    --log-interval 50 --save-interval 100000 "$@" 2>&1 | \
    grep -E "params=|step=50/50|CUDA out of memory|Error" || true
  python3 - <<PY
import json, pathlib
p = pathlib.Path("$ROOT/$NAME/train_summary.json")
if p.exists():
    s = json.loads(p.read_text())
    u = s["sec_per_update"]
    print(f"  {u:.4f} s/update -> 4000 updates {u*4000/60:.1f} min, "
          f"peak {s['peak_vram_mib']:.0f} MiB")
PY
}

bench plain
bench swd --dist-loss both --style-enc fixed --lambda-dist 0.1 --ode-steps-train 2 --grad-microbatch 64
bench online --online sde --online-steps 4 --online-warmup 10 --grad-microbatch 64
bench patch4 --patch 4 --batch 256
echo "=== BENCH DONE ==="
