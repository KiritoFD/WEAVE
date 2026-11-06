import json
import sys
from pathlib import Path

BASE = Path("/mnt/i/Latent_Style")
FILES = [
    "WEAVE/runs/p1/auroc_csd_dm.json",
    "WEAVE/runs/p1/auroc_dm.json",
    "WEAVE/runs/p2/pgt_p0_audit.json",
    "WEAVE/runs/p2/p2_cross_style.json",
    "WEAVE/runs/p2/p5_styleid_sensitivity.json",
    "WEAVE/runs/p3/p3_p4.json",
    "pseudo_gt_24h/p2_reach.json",
    "pseudo_gt_24h/p2_reach_e8.json",
    "pseudo_gt_24h/p2_reach_e3e1.json",
]
MAXS = 260


def show(obj, prefix="", depth=0, out=None):
    out = out if out is not None else []
    if depth > 4:
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, dict):
                out.append(f"{prefix}{k}:")
                show(v, prefix + "  ", depth + 1, out)
            elif isinstance(v, list):
                if all(not isinstance(x, (dict, list)) for x in v):
                    out.append(f"{prefix}{k}: [{len(v)}] {str(v[:6])[:MAXS]}")
                else:
                    out.append(f"{prefix}{k}: [list {len(v)}] first={str(v[0])[:120] if v else ''}")
            else:
                out.append(f"{prefix}{k}: {str(v)[:MAXS]}")
    return out


for rel in FILES:
    p = BASE / rel
    print(f"\n===== {rel} =====")
    if not p.exists():
        print("MISSING")
        continue
    try:
        d = json.load(open(p, encoding="utf-8"))
    except Exception as exc:
        print("ERR", exc)
        continue
    for line in show(d)[:90]:
        print(line)
