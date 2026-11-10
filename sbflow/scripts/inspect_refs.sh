#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
echo "=== model-required caches ==="
for f in runs/v6/cache/canonicalizer_stats.pt \
         runs/v6/cache/indomain_style_targets_v2_k1024.pt \
         runs/v6/cache/r1_style_embeddings.pt \
         runs/v6/cache/trust_region_ref.pt \
         runs/v6/cache/indomain_style_targets_k64.pt \
         runs/v6/cache/indomain_style_targets_v2_k64.pt \
         runs/v6/cache/indomain_style_targets_v2_k256.pt; do
  if [ -f "$f" ]; then printf "  %-58s %8s\n" "$f" "$(du -h "$f" | cut -f1)"; else printf "  %-58s MISSING\n" "$f"; fi
done
echo
echo "=== metric references ==="
for f in runs/v5/cache/eval_src_latents.pt \
         runs/v5/cache/eval_src_csd_feats.pt \
         runs/v5/cache/d5_style_multiscale_descriptors.pt \
         /mnt/i/Latent_Style/pseudo_gt_24h/csd_proto.pt \
         /mnt/i/Latent_Style/pseudo_gt_24h/orig_mean_fp16/00_Early_Renaissance.pt \
         /mnt/i/Latent_Style/pseudo_gt_24h/orig_mean_fp16/01_Impressionism.pt \
         /mnt/i/Latent_Style/pseudo_gt_24h/orig_mean_fp16/02_Minimalism.pt \
         /mnt/i/Latent_Style/pseudo_gt_24h/orig_mean_fp16/03_Rococo.pt \
         /mnt/i/Latent_Style/pseudo_gt_24h/orig_mean_fp16/04_Ukiyo_e.pt; do
  if [ -f "$f" ]; then printf "  %-62s %8s\n" "$(basename "$f")" "$(du -h "$f" | cut -f1)"; else printf "  %-62s MISSING\n" "$(basename "$f")"; fi
done
echo
echo "=== total for test images + refs ==="
du -ch data/test runs/v5/cache/eval_src_latents.pt runs/v5/cache/eval_src_csd_feats.pt \
  runs/v5/cache/d5_style_multiscale_descriptors.pt /mnt/i/Latent_Style/pseudo_gt_24h/csd_proto.pt \
  runs/v6/cache/canonicalizer_stats.pt runs/v6/cache/indomain_style_targets_v2_k1024.pt 2>/dev/null | tail -1
