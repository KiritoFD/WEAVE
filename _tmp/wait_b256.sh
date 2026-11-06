#!/usr/bin/env bash
cd /mnt/i/Latent_Style/WEAVE || exit 1
for i in $(seq 1 180); do
  if [ -f runs/pgt/B256_DONE ]; then
    echo "B256_DONE at $(date -u '+%F %T')"
    tail -n 8 runs/pgt/PROGRESS.md
    exit 0
  fi
  if tail -n 5 runs/pgt/PROGRESS.md | grep -q 'FAILED'; then
    echo "FAILED marker:"; tail -n 6 runs/pgt/PROGRESS.md; exit 1
  fi
  sleep 20
done
echo "TIMEOUT waiting B256_DONE"; tail -n 6 runs/pgt/PROGRESS.md; exit 2
