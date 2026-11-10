#!/bin/bash
# clean any leftover swap_full processes without matching this shell
for p in $(pgrep -f "vae_swap_full"); do
  if [ "$p" != "$$" ]; then kill -9 "$p" 2>/dev/null; fi
done
sleep 2
rm -f /root/Workspace/WEAVE_VAE/runs/v6/cache/canonicalizer_stats.pt
rm -f /root/Workspace/WEAVE_VAE/runs/vae_swap_full.log
echo "cleaned"
pgrep -f "vae_swap_full" | wc -l
