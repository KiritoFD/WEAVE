#!/usr/bin/env bash
# Loss-quality curve: A1/A2 x {200,400,600,1000,2000} steps on the SAME eval path.
# Env: DECODER=torch|trt (default torch).  Missing ckpt -> writes MISSING marker.
set -u
cd /mnt/i/Latent_Style/WEAVE || exit 1
decoder="${DECODER:-torch}"
mkdir -p runs/12h/eval_curve
for arm in A1 A2; do
  for step in 200 400 600 1000 2000; do
    ck=$(printf "runs/12h/%s/step_%06d.pt" "$arm" "$step")
    out=$(printf "runs/12h/eval_curve/%s_%d" "$arm" "$step")
    if [ ! -f "$ck" ]; then
      echo "MISSING $ck" > "$out.missing"
      echo "MISSING $ck"
      continue
    fi
    if [ -f "$out/dino_summary.json" ]; then
      echo "SKIP $out (exists)"
      continue
    fi
    if [ "$decoder" = "trt" ]; then
      bash tools/12h/eval_path_trt.sh "curve_${arm}_${step}_b8" "$ck" "$out" 8
    else
      bash tools/12h/eval_path.sh "curve_${arm}_${step}_b8" "$ck" "$out" 8 torch
    fi
    echo "DONE $out"
  done
done
date

