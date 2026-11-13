#!/usr/bin/env bash
# Stage the test images plus the small metric references, split for chunked base64 transfer.
# Deliberately EXCLUDES the 160 MB indomain_style_targets_v2_k1024.pt and the 492 MB
# orig_mean_fp16 latents: those are training artefacts, not the test set. They are listed in the
# manifest so a reader knows they exist and where they live.
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
STAGE=/tmp/v6test
ACTION=${1:-prepare}

if [ "$ACTION" = "prepare" ]; then
  rm -rf "$STAGE"; mkdir -p "$STAGE/bundle"
  # 1. the 150 test source images, keeping the data/test/<Style>/ layout
  mkdir -p "$STAGE/bundle/data"
  cp -a data/test "$STAGE/bundle/data/"
  # 2. small metric references
  mkdir -p "$STAGE/bundle/refs"
  cp -f runs/v5/cache/eval_src_latents.pt "$STAGE/bundle/refs/"
  cp -f runs/v5/cache/eval_src_csd_feats.pt "$STAGE/bundle/refs/"
  cp -f runs/v5/cache/d5_style_multiscale_descriptors.pt "$STAGE/bundle/refs/"
  cp -f /mnt/i/Latent_Style/pseudo_gt_24h/csd_proto.pt "$STAGE/bundle/refs/"
  # 3. the task manifest that defines the protocol
  mkdir -p "$STAGE/bundle/protocol"
  cp -f eval_protocol/tasks_750.json "$STAGE/bundle/protocol/"
  cp -f runs/v6_24h/heldout400_tasks.json "$STAGE/bundle/protocol/" 2>/dev/null || true
  cp -f runs/v6_24h/same150_tasks.json "$STAGE/bundle/protocol/" 2>/dev/null || true

  python3 - "$STAGE/bundle" <<'PY'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1])
man = {"files": {}, "not_included": {
    "runs/v6/cache/indomain_style_targets_v2_k1024.pt": "160 MB training reference cache",
    "/mnt/i/Latent_Style/pseudo_gt_24h/orig_mean_fp16/*.pt": "492 MB training-pool latents",
}}
for p in sorted(root.rglob("*")):
    if p.is_file():
        man["files"][p.relative_to(root).as_posix()] = {
            "sha256_16": hashlib.sha256(p.read_bytes()).hexdigest()[:16],
            "bytes": p.stat().st_size}
(root / "MANIFEST.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
imgs = [k for k in man["files"] if k.startswith("data/test/")]
print(f"  {len(man['files'])} files, {len(imgs)} test images")
PY

  tar czf "$STAGE/v6_testset.tgz" -C "$STAGE/bundle" .
  split -b 5m -d -a 2 "$STAGE/v6_testset.tgz" "$STAGE/part"
  echo "  staged $(du -h "$STAGE/v6_testset.tgz" | cut -f1) in $(ls "$STAGE"/part* | wc -l) parts"
  exit 0
fi

if [ "$ACTION" = "list" ]; then
  ls -l --time-style=+ "$STAGE"/part* | awk '{printf "  %s %s\n", $NF, $5}'
  exit 0
fi

if [ "$ACTION" = "emit" ]; then
  for n in $2; do
    P=$(printf "%s/part%02d" "$STAGE" "$n")
    [ -f "$P" ] || { echo "no such part: $P" >&2; exit 2; }
    echo "###PART${n}###"
    base64 -w0 "$P"
    echo
  done
  exit 0
fi

echo "unknown action" >&2
exit 1
