#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
python3 - <<'PY'
import json
from pathlib import Path
p = Path("runs/blog_sbflow/v10/score_04L_quick200/style_scores.json")
d = json.loads(p.read_text())
a = d["aggregate"]
o = a["overall"]
print("=== 04L (6000 updates, 600 per pair, batch512 patch8) overall ===")
print({k: round(v, 4) for k, v in o.items() if isinstance(v, float)})
print("\n=== by target style ===")
for k, v in sorted(a["by_target"].items()):
    print(f"  {k:22s} top1={v['target_top1']:.3f} margin={v['target_margin']:+.4f} "
          f"dT={v['csd_dT']:+.4f} dS={v['csd_dS']:+.4f} mmd={v['csd_mmd2_unbiased']:.4f} "
          f"swd={v['csd_swd']:.4f} n={v['n']}")
print("\n=== by direction (20) ===")
pos = 0
for k, v in sorted(a["by_direction"].items()):
    flag = "OK " if v["csd_dT"] > 0 else "NEG"
    if v["csd_dT"] > 0:
        pos += 1
    print(f"  {flag} {k:44s} top1={v['target_top1']:.3f} dT={v['csd_dT']:+.4f} "
          f"dS={v['csd_dS']:+.4f} swd={v['csd_swd']:.4f}")
print(f"\n  directions with positive dT: {pos}/20")
PY
echo
echo "=== comparison against the 800-update screening gate (arm 04) ==="
python3 - <<'PY'
import json
from pathlib import Path
for tag, f in (("04 (800 upd, 80/pair)", "score_04_full600"),
               ("04L (6000 upd, 600/pair)", "score_04L_quick200")):
    p = Path("runs/blog_sbflow/v10") / f / "style_scores.json"
    if not p.exists():
        print(f"  {tag}: missing")
        continue
    o = json.loads(p.read_text())["aggregate"]["overall"]
    print(f"  {tag:28s} top1={o['target_top1']:.4f} margin={o['target_margin']:+.4f} "
          f"dT={o['csd_dT']:+.4f} dS={o['csd_dS']:+.4f} swd={o['csd_swd']:.4f} n={o['n']}")
PY
