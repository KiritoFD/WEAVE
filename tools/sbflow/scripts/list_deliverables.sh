#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
echo "=== result artefacts to deliver ==="
for f in runs/v6_next/amortize/p0_variants.json \
         runs/v6_next/amortize/p1_fit_error.json \
         runs/v6_next/eval_750_P1_B0_ctrl218/eval.json \
         runs/v6_next/eval_750_P1_B3_distill20/eval.json \
         runs/v6_next/eval_750_P1_B4_distill50/eval.json \
         runs/v6_next/eval_heldout400_Q2a_embed_style/eval.json \
         runs/v6_next/eval_same150_Q2a_embed_style/eval.json \
         runs/v6_next/eval_heldout400_P2_G1b_bf16/eval.json \
         runs/v6_next/eval_same150_P2_G1b_bf16/eval.json \
         runs/v6_next/probes/r1_r1.json \
         runs/v6_next/probes/gate_r1_top50.json \
         runs/v6_next/probes/direction_L11.json; do
  if [ -f "$f" ]; then
    printf "  %-62s %s\n" "$f" "$(du -h "$f" | cut -f1)"
  else
    printf "  %-62s MISSING\n" "$f"
  fi
done
echo
echo "=== arm reports ==="
for a in Q2a_embed_style P2_G0_bf16 P2_G1a_bf16 P2_G1b_bf16 \
         P1_B0_ctrl218 P1_B1_distill218 P1_B2c_ctrl845 P1_B2_distill845 \
         P1_B3_distill20 P1_B4_distill50; do
  f="runs/v6/$a/report.json"
  if [ -f "$f" ]; then printf "  %-22s %s\n" "$a" "$(du -h "$f" | cut -f1)"; else printf "  %-22s MISSING\n" "$a"; fi
done
echo
echo "=== total payload estimate ==="
du -ch runs/v6_next/amortize/*.json runs/v6_next/eval_750_*/eval.json 2>/dev/null | tail -1
