"""Dump measured 600/150/750 metrics + early-stop facts for every finished arm."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/mnt/i/Latent_Style/WEAVE")
CELLS = ["S0_base", "S1_positive", "S2_reverse", "S0_base_b256", "S1_positive_b256", "S2_reverse_b256"]
WANT = ["csd_dT", "csd_dS", "csd_dM", "dino_m", "dino_c", "dino_s", "clip_s", "lpips", "patch_swd", "pswd"]


def flat(block: dict) -> dict:
    out = {}
    for k, v in block.items():
        if isinstance(v, (int, float)):
            out[k] = v
        elif isinstance(v, dict):
            for kk in ("mean", "value", "score"):
                if isinstance(v.get(kk), (int, float)):
                    out[k] = v[kk]
                    break
    return out


for cell in CELLS:
    p = ROOT / "runs/p2" / f"pgt_{cell}.json"
    es = ROOT / "runs/pgt" / cell / "internal_dynamics_summary.json"
    print(f"===== {cell} =====")
    if not p.exists():
        print("  metrics: MISSING")
    else:
        data = json.loads(p.read_text(encoding="utf-8"))
        print("  top keys:", sorted(data.keys()))
        main = data.get("metrics", {})
        print("  600 keys:", sorted(main.keys()))
        for scope, blk in (("600", main),):
            f = flat(blk)
            print(f"  {scope}:", json.dumps({k: round(v, 4) for k, v in f.items()}, ensure_ascii=False))
        ss = (data.get("same_style_150") or {}).get("metrics", {})
        print("  150 keys:", sorted(ss.keys()))
        print("  150:", json.dumps({k: round(v, 4) for k, v in flat(ss).items()}, ensure_ascii=False))
        hist = (data.get("historical_750") or {}).get("new_render", {}).get(cell, {})
        print("  750 keys:", sorted(hist.keys()))
        print("  750:", json.dumps({k: round(v, 4) for k, v in flat(hist).items()}, ensure_ascii=False))
    if es.exists():
        s = json.loads(es.read_text(encoding="utf-8"))
        print("  early_stop:", json.dumps({k: (v if not isinstance(v, list) else [round(x, 4) for x in v])
                                          for k, v in s.items()}, ensure_ascii=False))
    print()
