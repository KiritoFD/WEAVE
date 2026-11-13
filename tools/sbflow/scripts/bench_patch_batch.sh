#!/usr/bin/env bash
# Patch/batch trade-off on the real SD-VAE caches (64x64 latent).
set -euo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
C="/mnt/i/Latent_Style/pseudo_gt_24h/orig_mean_fp16/00_Early_Renaissance.pt \
/mnt/i/Latent_Style/pseudo_gt_24h/orig_mean_fp16/01_Impressionism.pt \
/mnt/i/Latent_Style/pseudo_gt_24h/orig_mean_fp16/02_Minimalism.pt \
/mnt/i/Latent_Style/pseudo_gt_24h/orig_mean_fp16/03_Rococo.pt \
/mnt/i/Latent_Style/pseudo_gt_24h/orig_mean_fp16/04_Ukiyo_e.pt"

run() {  # patch batch
  local P=$1 B=$2
  local NAME="pb_p${P}_b${B}"
  echo "=== patch=$P tokens=$(( (64 / P) * (64 / P) )) batch=$B ==="
  rm -rf "runs/blog_sbflow/v10/_bench_$NAME"
  python3 -u -m sbflow.train5 --experiment-id "$NAME" --caches $C \
    --output "runs/blog_sbflow/v10/_bench_$NAME" \
    --steps 8 --batch "$B" --patch "$P" --width 256 --depth 6 --heads 4 \
    --warmup 2 --log-interval 4 --save-interval 1000 2>&1 | \
    grep -E "params=|step=8/8|OutOfMemory" || echo "  (no step output)"
  python3 - <<PY
import json, pathlib
p = pathlib.Path("runs/blog_sbflow/v10/_bench_$NAME/train_summary.json")
if p.exists():
    s = json.loads(p.read_text())
    print(f"  -> {s['sec_per_update']} s/update, peak={s['peak_vram_mib']} MiB, 800upd={s['sec_per_update']*800/60:.1f} min")
PY
}

run 8 512
run 4 512
run 4 256
