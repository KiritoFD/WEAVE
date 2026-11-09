#!/usr/bin/env bash
# Read-only sanity check: did the models actually change the latents/images?
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
ROOT=runs/blog_sbflow/suite40
python3 - <<'PY'
import json, numpy as np
from pathlib import Path
R = Path("runs/blog_sbflow/suite40")
for aid in ("01","04","08","10","24"):
    m = R / f"gen_{aid}_quick200" / "manifest.json"
    s = R / f"score_{aid}_quick200" / "style_scores.json"
    if not (m.exists() and s.exists()):
        print(f"  exp{aid}: artifacts missing"); continue
    man = json.loads(m.read_text())
    l2 = [t.get("latent_l2") for t in man["tasks"] if t.get("latent_l2") is not None]
    rows = json.loads(s.read_text())["rows"]
    dt = [r["csd_dT"] for r in rows]
    ds = [r["csd_dS"] for r in rows]
    lp = [r["lpips_to_input"] for r in rows]
    dc = [r["dino_c"] for r in rows]
    print(f"  exp{aid}: n={len(rows)} "
          f"latentL2 mean={np.mean(l2) if l2 else float('nan'):.2f} "
          f"| dT mean={np.mean(dt):+.4f} std={np.std(dt):.4f} "
          f"| dS mean={np.mean(ds):+.4f} "
          f"| lpips={np.mean(lp):.3f} dino_c={np.mean(dc):.3f}")
# how much do the generated images differ BETWEEN arms? (same task_id)
print("\n=== cross-arm generated-image agreement (proxy: per-row dT correlation) ===")
import itertools
data = {}
for aid in ("01","04","08","10","24"):
    p = R / f"score_{aid}_quick200" / "style_scores.json"
    if p.exists():
        d = json.loads(p.read_text())["rows"]
        data[aid] = {r["task_id"]: r["csd_dT"] for r in d}
for a,b in itertools.combinations(sorted(data),2):
    common = set(data[a]) & set(data[b])
    x = np.array([data[a][k] for k in common]); y = np.array([data[b][k] for k in common])
    c = float(np.corrcoef(x,y)[0,1])
    print(f"  {a} vs {b}: corr(dT per task)={c:.3f}  mean|diff|={np.mean(np.abs(x-y)):.4f}")
PY
