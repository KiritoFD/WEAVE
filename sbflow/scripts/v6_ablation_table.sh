#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
python3 - <<'PY'
import json
from pathlib import Path
print(f"{'variant':22s} {'mode':7s} {'canon':10s} {'params':>9s} {'steps':>6s} "
      f"{'w_sd':>6s} {'w_cross':>7s} {'w_inv':>6s} {'dT':>9s} {'dino':>7s} {'lpips':>7s} "
      f"{'frac+':>6s} {'Ukiyo':>8s} {'Roc':>8s} {'ER':>8s}")
for d in sorted(Path("runs/v6").iterdir()):
    rp = d / "report.json"
    if not rp.exists() or d.name == "cache":
        continue
    r = json.loads(rp.read_text())
    e = r.get("eval200", {})
    if not e:
        print(f"{d.name:22s} (no eval200)")
        continue
    pt = e.get("per_target_dT", {})
    print(f"{d.name:22s} {str(r.get('mode','')):7s} {str(r.get('canon_mode','')):10s} "
          f"{r.get('n_params',0):9d} {r.get('steps',0):6d} "
          f"{r.get('w_style_dist',0):6.2f} {r.get('w_cross',0):7.2f} "
          f"{r.get('w_canon_inv',0):6.2f} {e.get('mean_dT',0):+9.4f} "
          f"{e.get('mean_dino_c',0):7.4f} {e.get('mean_lpips',0):7.4f} "
          f"{e.get('frac_dT_positive',0):6.2f} {pt.get('Ukiyo_e',0):+8.4f} "
          f"{pt.get('Rococo',0):+8.4f} {pt.get('Early_Renaissance',0):+8.4f}")

print()
print("=== score_diff report ===")
r = json.loads(Path("runs/v6/score_diff/report.json").read_text())
for k, v in r.items():
    s = json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
    print(f"  {k}: {str(s)[:300]}")
PY
