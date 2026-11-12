#!/bin/bash
set -e
cd /root/Workspace/WEAVE_VAE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=/opt/conda/envs/cu121/bin/python
L=runs/vae_style_vae

echo "=== A0 baseline (A0 encoder + A0 decoder) ==="
$P -u tools/vae_style_vae/eval.py --a0-decoder --ckpt "" --tag A0base > $L/eval_A0base.log 2>&1
echo "done A0base"

echo "=== A_joint ==="
$P -u tools/vae_style_vae/eval.py --ckpt $L/A_joint_best.pt --tag A > $L/eval_A.log 2>&1
echo "done A"

echo "=== B_style ==="
$P -u tools/vae_style_vae/eval.py --ckpt $L/B_style_best.pt --tag B > $L/eval_B.log 2>&1
echo "done B"

echo "=== C_edit ==="
$P -u tools/vae_style_vae/eval.py --ckpt $L/C_edit_best.pt --tag C > $L/eval_C.log 2>&1
echo "done C"

echo ALL_EVALS_DONE
