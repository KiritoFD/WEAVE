#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
echo "=== goal deliverables ==="
for f in docs/v6/V6_NEXT_STYLE_PLAN.md \
         docs/v6/V6_NEXT_RESULTS.md \
         docs/v6/V6_NEXT_DECISIONS.md \
         docs/v6/V6_NEXT_P0_CORRECTIONS.md; do
  if [ -f "$f" ]; then
    echo "  OK   $f ($(wc -l < "$f") lines)"
  else
    echo "  MISS $f"
  fi
done
echo
echo "=== runs/v6_next ==="
find runs/v6_next -maxdepth 2 -type d 2>/dev/null | sort
echo "  files: $(find runs/v6_next -type f 2>/dev/null | wc -l)"
echo
echo "=== code ==="
ls -1 tools/v6_next/*.py 2>/dev/null
echo
echo "=== P0/P1 artefacts ==="
for f in runs/v6_next/probes/direction_L11.json \
         runs/v6_next/probes/direction_L8.json \
         runs/v6_next/probes/r1_r1.json \
         runs/v6_next/probes/r1_r1.pt \
         runs/v6_next/probes/gate_r1_top1.json \
         runs/v6_next/eval_sdA_full/eval.json \
         runs/v6_next/eval_sdB_full/eval.json \
         runs/v6_24h/eval_heldout400_L11/eval.json \
         runs/v6_24h/eval_same150_L11/eval.json; do
  [ -f "$f" ] && echo "  OK   $f" || echo "  MISS $f"
done
echo
echo "=== gates from json ==="
python3 - <<'PY'
import json
from pathlib import Path
r = json.loads(Path("runs/v6_next/probes/r1_r1.json").read_text())
print(f"  R1 balanced_top1={r['val']['balanced_top1']:.3f} "
      f"gate_top1={r['gate_top1']} gate_recall={r['gate_recall']}")
g = json.loads(Path("runs/v6_next/probes/gate_r1_top1.json").read_text())
print(f"  gate best d_m_target={g['best']['d_m_target']:+.4f} "
      f"top1={g['best']['top1']:.3f} dino={g['best']['dino_c']:.4f}")
print(f"  cos(grad, V6)={g['cos_grad_v6']:+.4f}")
PY
