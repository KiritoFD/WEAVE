#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
echo "=== Q1 (seed-43 replication) queue ==="
grep -aE "STEP |OK rc=|FAILED|FATAL|DONE|GPU free" runs/v6_24h/queue_q1.log 2>/dev/null | tail -n 10 || echo "  no Q1 log yet"
echo
echo "=== running ==="
ps -eo etime,args --no-headers | grep train_eval_v6 | grep -v grep | head -1 | cut -c1-64
echo
echo "=== Q1 arms ==="
python3 - <<'PY'
import json
from pathlib import Path
for a in ("Q1_G0_s43", "Q1_G1b_s43"):
    p = Path(f"runs/v6/{a}/report.json")
    if not p.exists():
        print(f"  {a:14s} pending"); continue
    d = json.loads(p.read_text()); e = d.get("eval200") or {}
    pt = e.get("per_target_top1") or {}
    t1 = sum(pt.values())/len(pt) if pt else 0
    roc = pt.get("Rococo", 0.0)
    print(f"  {a:14s} dT={e.get('mean_dT',0):+.4f} Top1={t1:.3f} Rococo={roc:.3f} "
          f"dM={e.get('mean_dM',0):+.4f} DINO={e.get('mean_dino_c',0):.4f} "
          f"LPIPS={e.get('mean_lpips',0):.4f} sty+={e.get('styles_improved')}/5")
PY
echo
echo "=== seed-43 weight guard ==="
python3 - <<'PY'
import torch
from pathlib import Path
p0 = Path("runs/v6/Q1_G0_s43/Q1_G0_s43.pt")
p1 = Path("runs/v6/Q1_G1b_s43/Q1_G1b_s43.pt")
if not p0.exists() or not p1.exists():
    print("  pending"); raise SystemExit
a = torch.load(p0, map_location="cpu", weights_only=False)["model"]
b = torch.load(p1, map_location="cpu", weights_only=False)["model"]
mx = max(float((a[k].float()-b[k].float()).abs().max()) for k in a)
nd = sum(1 for k in a if float((a[k].float()-b[k].float()).abs().max()) > 0)
print(f"  max|dW|={mx:.3e} {nd}/{len(a)} -> "
      + ("FATAL: ranking term did nothing" if mx == 0.0 else "reached the weights"))
PY
echo
echo "=== Rococo replication check (seed 42 vs 43) ==="
python3 - <<'PY'
import json
from pathlib import Path
def roc(a):
    p = Path(f"runs/v6/{a}/report.json")
    if not p.exists():
        return None
    pt = (json.loads(p.read_text()).get("eval200") or {}).get("per_target_top1") or {}
    return pt.get("Rococo")
for s, g0, g1 in (("42", "P2_G0_bf16", "P2_G1b_bf16"), ("43", "Q1_G0_s43", "Q1_G1b_s43")):
    a, b = roc(g0), roc(g1)
    if a is None or b is None:
        print(f"  seed {s}: pending"); continue
    print(f"  seed {s}: G0 Rococo={a:.3f}  G1b Rococo={b:.3f}  delta={b-a:+.3f}")
PY
echo
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
