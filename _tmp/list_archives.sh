#!/usr/bin/env bash
set -uo pipefail
D=/mnt/i/Latent_Style/WEAVE/_baseline_stage
echo "=== archives (WSL path | Windows path | size) ==="
for f in "$D"/baseline_*.tar.gz "$D"/inventory.json; do
  [ -f "$f" ] || continue
  win=$(echo "$f" | sed 's#^/mnt/i/#I:\\#; s#/#\\#g')
  printf "%-72s %s  %s\n" "$f" "$win" "$(du -h "$f" | cut -f1)"
done
echo
echo "=== sha256 ==="
sha256sum "$D"/baseline_*.tar.gz
echo
echo "=== archive contents summary ==="
for f in "$D"/baseline_*.tar.gz; do
  echo "--- $(basename "$f") ---"
  tar -tzf "$f" | awk -F/ '{print $1"/"$2}' | sort | uniq -c | sort -k2
done
