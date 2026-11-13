#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
echo "=== queue_p2b.log (last 12) ==="
tail -n 12 runs/v6_24h/queue_p2b.log 2>/dev/null || echo "  none"
echo
echo "=== running ==="
ps -eo etime,args --no-headers | grep train_eval_v6 | grep -v grep | head -1
echo
echo "=== G1a new report mtime ==="
ls -l --time-style=+%H:%M runs/v6/G1a_rank_weak/report.json 2>/dev/null
ls -l --time-style=+%H:%M runs/v6/G0_ctrl/report.json 2>/dev/null
echo
echo "=== weight comparison G0 vs G1a (the guard) ==="
python3 - <<'PY'
import torch
from pathlib import Path
p0, p1 = Path("runs/v6/G0_ctrl/G0_ctrl.pt"), Path("runs/v6/G1a_rank_weak/G1a_rank_weak.pt")
if not (p0.exists() and p1.exists()):
    print("  missing checkpoint"); raise SystemExit
a = torch.load(p0, map_location="cpu", weights_only=False)["model"]
b = torch.load(p1, map_location="cpu", weights_only=False)["model"]
mx = max(float((a[k].float()-b[k].float()).abs().max()) for k in a)
nd = sum(1 for k in a if float((a[k].float()-b[k].float()).abs().max()) > 0)
print(f"  max|dW|={mx:.3e}  tensors_differing={nd}/{len(a)}")
print("  -> RANKING TERM DID NOTHING" if mx == 0.0 else "  -> ranking term changed the weights")
PY
