#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
echo "=== checkpoints ==="
for f in runs/v6_next/probes/r1_r1.pt \
         runs/v6/Q2a_embed_style/Q2a_embed_style.pt \
         runs/v6/P2_G0_bf16/P2_G0_bf16.pt \
         runs/v6/P2_G1a_bf16/P2_G1a_bf16.pt \
         runs/v6/P2_G1b_bf16/P2_G1b_bf16.pt \
         runs/v6/P1_B0_ctrl218/P1_B0_ctrl218.pt \
         runs/v6/P1_B1_distill218/P1_B1_distill218.pt \
         runs/v6/P1_B2c_ctrl845/P1_B2c_ctrl845.pt \
         runs/v6/P1_B2_distill845/P1_B2_distill845.pt \
         runs/v6/P1_B3_distill20/P1_B3_distill20.pt \
         runs/v6/P1_B4_distill50/P1_B4_distill50.pt \
         runs/v6/L11_k1024_dim16/L11_k1024_dim16.pt \
         runs/v6/L8_k1024_dim32/L8_k1024_dim32.pt \
         runs/v6/score_diff/score_diff_net.pt; do
  if [ -f "$f" ]; then
    printf "  %-56s %8s\n" "$f" "$(du -h "$f" | cut -f1)"
  else
    printf "  %-56s MISSING\n" "$f"
  fi
done
echo
echo "=== total ==="
du -ch runs/v6_next/probes/r1_r1.pt runs/v6/Q2a_embed_style/Q2a_embed_style.pt \
   runs/v6/P2_G*_bf16/*.pt runs/v6/P1_B*/*.pt runs/v6/L11_k1024_dim16/*.pt \
   runs/v6/L8_k1024_dim32/*.pt runs/v6/score_diff/*.pt 2>/dev/null | tail -1
