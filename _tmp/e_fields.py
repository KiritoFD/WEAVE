from __future__ import annotations

import json

q = json.load(open("/mnt/i/Latent_Style/WEAVE/runs/p2/pgt_q0.json"))
c = q["configs"]
for key in ("A_P5", "D_P5", "E_E7"):
    v = c[key]
    print(key, "keys:", sorted(v.keys()))
    flat = {k: (vv.get("mean") if isinstance(vv, dict) else vv) for k, vv in v.items()}
    print(" ", json.dumps(flat, ensure_ascii=False)[:600])
