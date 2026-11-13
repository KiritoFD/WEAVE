#!/usr/bin/env bash
# Stage a checkpoint bundle and split it into 5 MB parts for chunked base64 transfer.
# Usage:
#   bash stage_ckpts.sh prepare     -> builds /tmp/v6ck/v6_ckpts.tgz and splits it
#   bash stage_ckpts.sh list        -> prints the part names and sizes
#   bash stage_ckpts.sh emit <NN>   -> base64 of part NN on stdout
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
STAGE=/tmp/v6ck
ACTION=${1:-prepare}

if [ "$ACTION" = "prepare" ]; then
  rm -rf "$STAGE"; mkdir -p "$STAGE/bundle"
  FILES="runs/v6_next/probes/r1_r1.pt
runs/v6/Q2a_embed_style/Q2a_embed_style.pt
runs/v6/P2_G0_bf16/P2_G0_bf16.pt
runs/v6/P2_G1a_bf16/P2_G1a_bf16.pt
runs/v6/P2_G1b_bf16/P2_G1b_bf16.pt
runs/v6/P1_B0_ctrl218/P1_B0_ctrl218.pt
runs/v6/P1_B1_distill218/P1_B1_distill218.pt
runs/v6/P1_B2c_ctrl845/P1_B2c_ctrl845.pt
runs/v6/P1_B2_distill845/P1_B2_distill845.pt
runs/v6/P1_B3_distill20/P1_B3_distill20.pt
runs/v6/P1_B4_distill50/P1_B4_distill50.pt
runs/v6/L11_k1024_dim16/L11_k1024_dim16.pt
runs/v6/L8_k1024_dim32/L8_k1024_dim32.pt
runs/v6/score_diff/score_diff_net.pt"
  n=0
  for f in $FILES; do
    if [ -f "$f" ]; then
      d="$STAGE/bundle/$(dirname "$f")"
      mkdir -p "$d"
      cp -f "$f" "$d/"
      n=$((n+1))
    else
      echo "  MISSING $f"
    fi
  done
  # a small manifest so the local side can verify what should be inside
  python3 - "$STAGE/bundle" <<'PY'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1])
man = {}
for p in sorted(root.rglob("*.pt")):
    h = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    man[p.relative_to(root).as_posix()] = {"sha256_16": h, "bytes": p.stat().st_size}
(root / "MANIFEST.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
print(f"  manifest: {len(man)} checkpoints")
PY
  tar czf "$STAGE/v6_ckpts.tgz" -C "$STAGE/bundle" .
  split -b 5m -d -a 2 "$STAGE/v6_ckpts.tgz" "$STAGE/part"
  echo "  staged: $(du -h "$STAGE/v6_ckpts.tgz" | cut -f1) in $(ls "$STAGE"/part* | wc -l) parts, $n files"
  exit 0
fi

if [ "$ACTION" = "list" ]; then
  ls -l --time-style=+ "$STAGE"/part* | awk '{printf "  %s %s\n", $NF, $5}'
  exit 0
fi

if [ "$ACTION" = "emit" ]; then
  P=$(printf "%s/part%02d" "$STAGE" "$2")
  [ -f "$P" ] || { echo "no such part: $P" >&2; exit 2; }
  base64 -w0 "$P"
  exit 0
fi

echo "unknown action: $ACTION" >&2
exit 1
