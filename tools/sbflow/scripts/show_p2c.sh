#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
echo "=== P2c (bf16) queue ==="
grep -aE "STEP |OK rc=|FAILED|FATAL|DONE" runs/v6_24h/queue_p2c.log 2>/dev/null | tail -n 10 || echo "  no log"
echo
echo "=== running ==="
ps -eo etime,args --no-headers | grep train_eval_v6 | grep -v grep | head -1 | cut -c1-60
echo
echo "=== bf16 arms ==="
python3 - <<'PY'
import json
from pathlib import Path
for a in ("P2_G0_bf16", "P2_G1a_bf16", "P2_G1b_bf16"):
    p = Path(f"runs/v6/{a}/report.json")
    if not p.exists():
        print(f"  {a:16s} pending"); continue
    d = json.loads(p.read_text()); e = d.get("eval200") or {}
    pt = e.get("per_target_top1") or {}
    t1 = sum(pt.values())/len(pt) if pt else 0
    print(f"  {a:16s} dT={e.get('mean_dT',0):+.4f} Top1={t1:.3f} "
          f"dM={e.get('mean_dM',0):+.4f} DINO={e.get('mean_dino_c',0):.4f} "
          f"LPIPS={e.get('mean_lpips',0):.4f} sty+={e.get('styles_improved')}/5")
PY
echo
echo "=== weight guard (G1 vs G0, bf16) ==="
python3 - <<'PY'
import torch
from pathlib import Path
p0 = Path("runs/v6/P2_G0_bf16/P2_G0_bf16.pt")
if not p0.exists():
    print("  control not ready"); raise SystemExit
base = torch.load(p0, map_location="cpu", weights_only=False)["model"]
for a in ("P2_G1a_bf16", "P2_G1b_bf16"):
    p = Path(f"runs/v6/{a}/{a}.pt")
    if not p.exists():
        print(f"  {a:16s} pending"); continue
    b = torch.load(p, map_location="cpu", weights_only=False)["model"]
    mx = max(float((base[k].float()-b[k].float()).abs().max()) for k in base)
    nd = sum(1 for k in base if float((base[k].float()-b[k].float()).abs().max()) > 0)
    verdict = "DID NOTHING (fatal)" if mx == 0.0 else "reached the weights"
    print(f"  {a:16s} max|dW|={mx:.3e} {nd}/{len(base)}  -> {verdict}")
PY
echo
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
