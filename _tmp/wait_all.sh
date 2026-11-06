#!/usr/bin/env bash
cd /mnt/i/Latent_Style/WEAVE || exit 1
for i in $(seq 1 300); do
  if tail -n 20 runs/pgt/PROGRESS.md | grep -q 'S2B DONE'; then
    echo "S2B_DONE at $(date -u '+%F %T')"; tail -n 20 runs/pgt/PROGRESS.md; exit 0
  fi
  if tail -n 8 runs/pgt/PROGRESS.md | grep -q 'FAILED'; then
    echo "FAILED:"; tail -n 10 runs/pgt/PROGRESS.md; exit 1
  fi
  sleep 60
done
echo "TIMEOUT (5h) waiting S2B_DONE"; tail -n 12 runs/pgt/PROGRESS.md; exit 2
