#!/usr/bin/env bash
# V6 missing-ablation grid.
#
# Base config = the best existing V6 arm, D1_c1_gentle, reproduced EXACTLY:
#   mode=direct, canon=c1_gentle, base_dim=48 -> 1,881,792 params, 800 steps,
#   coherence_beta=0, w_rec=1.0, w_style_dist=0.25, w_cross=0.35, w_canon_inv=1.5, seed=42
#   (existing eval200: dT +0.0323 / DINO-C 0.9167 / LPIPS 0.3658)
# NOTE base_dim=56 gives 2,553,784 params, i.e. a DIFFERENT (larger) model, which is why
# the reference arm is 48 and not the C3UNet default.
#
# Existing ablations already cover: canon {c1,c1_soft,c1_gentle}, w_style_dist
# {0.1,0.25,0.3,0.5}, w_cross {0,0.35,0.45,0.5,1.2,2.0}, w_canon_inv {0,0.2,1.5},
# mode {direct,flow}, seed {42,43}.
#
# Missing and run here (one factor changed at a time):
#   A1 no_content   zero the content condition (identical params)
#   A2 no_style     pin Style-ID to 0 (identical params)
#   A3 canon_none   raw latent as content (no canonicalization)
#   A4..A6          base_dim 24/32/40 -> minimal size
#   A7 steps400     half the training budget
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs/v6/logs

BASE=(--mode direct --canon-mode c1_gentle --base-dim 48 --steps 800 --microbatch 80
      --accum 3 --lr 3e-4 --coherence-beta 0.0 --w-rec 1.0 --w-style-dist 0.25
      --w-cross 0.35 --w-canon-inv 1.5 --seed 42)

run() {  # name extra...
  local NAME=$1; shift
  echo "########## $NAME ##########"
  python3 -u tools/v6/train_eval_v6.py --name "$NAME" "${BASE[@]}" "$@" \
    2>&1 | tee -a "runs/v6/logs/${NAME}.log" | grep -aE "params=|Eval200|Error|error" | tail -6
}

run A0_ref        # exact reproduction of D1_c1_gentle, to confirm the harness matches
run A1_no_content --no-content
run A2_no_style   --no-style
run A3_canon_none --canon-mode none
run A4_dim24      --base-dim 24
run A5_dim32      --base-dim 32
run A6_dim40      --base-dim 40
run A7_steps400   --steps 400

echo "########## V6 ABLATION GRID DONE ##########"
