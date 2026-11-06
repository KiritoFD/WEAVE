#!/usr/bin/env bash
cd /mnt/i/Latent_Style/WEAVE || exit 1
echo "--- procs ---"; pgrep -af fix_effective_coverage || echo none
echo "--- summary exists? ---"; ls -la /mnt/i/Latent_Style/pseudo_gt_24h/pairs/coverage_fix_summary.json 2>/dev/null || echo no_summary
echo "--- summary ---"; cat /mnt/i/Latent_Style/pseudo_gt_24h/pairs/coverage_fix_summary.json 2>/dev/null
echo "--- covfix.log ---"; tail -n 8 runs/p2/covfix.log | cut -c1-400
echo "--- pairs sizes ---"
for a in base positive reverse; do
  echo "$a shards=$(ls -1 /mnt/i/Latent_Style/pseudo_gt_24h/pairs/$a | grep -c '^shard_') du=$(du -sh /mnt/i/Latent_Style/pseudo_gt_24h/pairs/$a | cut -f1)"
done
echo "--- build logs ---"
for f in runs/p2/pairs_base.log runs/p2/pairs_positive.log runs/p2/pairs_reverse.log; do
  echo "== $f"; head -n 2 $f | cut -c1-200; tail -n 3 $f | cut -c1-200
done
