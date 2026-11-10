#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
echo "=== Q2 (mechanism ablations) queue ==="
grep -aE "STEP |OK rc=|FAILED|FATAL|DONE|GPU free" runs/v6_24h/queue_q2.log 2>/dev/null | tail -n 10 || echo "  no Q2 log"
echo
echo "=== running ==="
ps -eo etime,args --no-headers | grep train_eval_v6 | grep -v grep | head -1 | cut -c1-70
echo
echo "=== Q2 arms ==="
python3 - <<'PY'
import json
from pathlib import Path
arms = ("Q2a_embed_style", "Q2b_trust", "Q2c_embed_trust")
print(f"  {'arm':20s}{'dT':>9s}{'Top-1':>8s}{'Rococo':>9s}{'Uki':>7s}"
      f"{'dM':>9s}{'DINO':>8s}{'LPIPS':>8s}{'sty':>5s}")
for a in arms:
    p = Path(f"runs/v6/{a}/report.json")
    if not p.exists():
        print(f"  {a:20s} pending"); continue
    d = json.loads(p.read_text()); e = d.get("eval200") or {}
    pt = e.get("per_target_top1") or {}
    t1 = sum(pt.values())/len(pt) if pt else 0
    print(f"  {a:20s}{e.get('mean_dT',0):+9.4f}{t1:8.3f}{pt.get('Rococo',0):9.3f}"
          f"{pt.get('Ukiyo_e',0):7.3f}{e.get('mean_dM',0):+9.4f}"
          f"{e.get('mean_dino_c',0):8.4f}{e.get('mean_lpips',0):8.4f}"
          f"{e.get('styles_improved')}/5")
PY
echo
echo "=== weight guard vs the ranking-only baseline (P2_G1b_bf16) ==="
python3 - <<'PY'
import torch
from pathlib import Path
base = Path("runs/v6/P2_G1b_bf16/P2_G1b_bf16.pt")
if not base.exists():
    print("  baseline missing"); raise SystemExit
a = torch.load(base, map_location="cpu", weights_only=False)["model"]
for n in ("Q2a_embed_style", "Q2b_trust", "Q2c_embed_trust"):
    p = Path(f"runs/v6/{n}/{n}.pt")
    if not p.exists():
        print(f"  {n:20s} pending"); continue
    b = torch.load(p, map_location="cpu", weights_only=False)["model"]
    mx = max(float((a[k].float()-b[k].float()).abs().max()) for k in a)
    nd = sum(1 for k in a if float((a[k].float()-b[k].float()).abs().max()) > 0)
    v = "FATAL: mechanism did nothing" if mx == 0.0 else "mechanism reached the weights"
    print(f"  {n:20s} max|dW|={mx:.3e} {nd}/{len(a)} -> {v}")
PY
echo
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
