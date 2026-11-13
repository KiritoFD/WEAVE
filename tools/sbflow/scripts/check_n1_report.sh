#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
echo "=== N1 log: eval stages ==="
grep -aE "Eval200|Conditioning audit|Running 200-task|train_wall|peak" \
  runs/v6_24h/train_logs/N1_no_style.log | tail -8
echo
echo "=== N1 report: which sections exist ==="
python3 - <<'PY'
import json
from pathlib import Path
p = Path("runs/v6/N1_no_style/report.json")
if not p.exists():
    print("  no report.json"); raise SystemExit
r = json.loads(p.read_text())
print(f"  top-level keys: {sorted(r.keys())}")
e = r.get("eval200") or {}
print(f"  eval200 keys: {sorted(e.keys())}")
print(f"  n_cross={e.get('n_cross')} mean_dT={e.get('mean_dT'):+.4f} "
      f"dino={e.get('mean_dino_c'):.4f} lpips={e.get('mean_lpips'):.4f}")
print(f"  per_target_dT={ {k: round(v,4) for k,v in (e.get('per_target_dT') or {}).items()} }")
print(f"  same50_mean_dS={e.get('same50_mean_dS')} same50_dino={e.get('same50_mean_dino_c')}")
c = r.get("conditioning_validity") or {}
print(f"  conditioning_validity present={bool(c)} all_three={c.get('all_three_passed')}")
print(f"  train_wall_sec={r.get('train_wall_sec')} peak_vram={r.get('peak_vram_mib')}")
print(f"  grad_diag={r.get('grad_diag')}")
print(f"  lam_s={r.get('lambda_cross_abs')} lam_c={r.get('lambda_canon_abs')} "
      f"lam_same={r.get('lambda_same_abs')} b1={r.get('b1_status')}")
PY
