#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
echo "=== CSD manifold (full) ==="
python3 - <<'PY'
import json
from pathlib import Path
p = Path("runs/v6_24h/csd_manifold_best.json")
if not p.exists():
    print("  missing"); raise SystemExit
d = json.loads(p.read_text())
print(f"  ckpt: {d.get('ckpt')}")
for pop in d.get("populations", []):
    print(f"  {pop['label']:26s} n={pop['n']:4d}  max_cos={pop['max_cos_mean']:.4f} "
          f"(p10 {pop['max_cos_p10']:.4f})  margin={pop['margin_mean']:.4f} "
          f"(p10 {pop['margin_p10']:.4f})  spread={pop['spread_mean']:.4f}")
PY
echo
echo "=== visual grid ==="
ls -la runs/v6_24h/grid/ 2>/dev/null | head -n 8 || echo "  missing"
echo
echo "=== grid manifest ==="
python3 - <<'PY'
import json
from pathlib import Path
p = Path("runs/v6_24h/grid/manifest.json")
if not p.exists():
    print("  missing"); raise SystemExit
d = json.loads(p.read_text())
for t in d.get("tasks", []):
    sc = t["scores"]
    print(f"  {t['file']:36s} src={t['src_style']:20s}")
    for s, v in sc.items():
        print(f"      {s:20s} target_cos={v['target_cos']:+.4f} "
              f"argmax={v['argmax']:20s} margin={v['margin']:+.4f}")
PY
