#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
python3 - <<'PY'
import json
from pathlib import Path
for aid in ("01","04"):
 p=Path(f"runs/blog_sbflow/suite40/score_{aid}_quick200/style_scores.json")
 if not p.exists():
  print(f"  exp{aid}: no score yet"); continue
 d=json.loads(p.read_text())["aggregate"]
 print(f"=== exp{aid} ===")
 print("  overall", {k:round(v,4) for k,v in d["overall"].items() if isinstance(v,float) and k in ("target_top1","target_margin","csd_dT","csd_dS","csd_swd")})
 for s in ("Rococo","Early_Renaissance","Ukiyo_e","Impressionism","Minimalism"):
  v=d["by_target"].get(s)
  if v: print(f"  {s:22s} top1={v['target_top1']:.3f} dT={v['csd_dT']:+.4f} swd={v['csd_swd']:.4f}")
 pos=sum(1 for v in d["by_direction"].values() if v["csd_dT"]>0)
 print(f"  positive dT directions {pos}/20")
PY
