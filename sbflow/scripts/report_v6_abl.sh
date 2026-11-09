#!/usr/bin/env bash
# Final V6 ablation report: A0_ref vs the one-factor ablations, with deltas.
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
python3 - <<'PY'
import json
from pathlib import Path

ORDER = ["A0_ref", "A1_no_content", "A2_no_style", "A3_canon_none",
         "A4_dim24", "A5_dim32", "A6_dim40", "A7_steps400",
         "D1_c1_gentle", "D1_c1_gentle_s43", "D1_c1_soft", "D1_T1_strong"]

def load(name):
    p = Path("runs/v6") / name / "report.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())

rows = {}
for n in ORDER:
    r = load(n)
    if r is None:
        continue
    e = r.get("eval200") or {}
    if not e:
        continue
    pt = e.get("per_target_dT", {})
    rows[n] = dict(
        mode=r.get("mode"), canon=r.get("canon_mode"), params=r.get("n_params"),
        steps=r.get("steps"), nc=r.get("no_content"), ns=r.get("no_style"),
        dT=e.get("mean_dT"), dino=e.get("mean_dino_c"), lpips=e.get("mean_lpips"),
        frac=e.get("frac_dT_positive"), ukiyo=pt.get("Ukiyo_e"),
        roc=pt.get("Rococo"), er=pt.get("Early_Renaissance"),
        imp=pt.get("Impressionism"), mn=pt.get("Minimalism"),
        styles=e.get("styles_improved"),
    )

hdr = (f"{'variant':16s} {'canon':10s} {'params':>8s} {'steps':>5s} {'nc':>3s} {'ns':>3s} "
       f"{'dT':>9s} {'dino':>7s} {'lpips':>7s} {'frac+':>6s} {'sty+':>4s} "
       f"{'ER':>8s} {'Imp':>8s} {'Min':>8s} {'Roc':>8s} {'Uki':>8s}")
print(hdr)
print("-" * len(hdr))
for n in ORDER:
    if n not in rows:
        continue
    v = rows[n]
    print(f"{n:16s} {str(v['canon']):10s} {v['params']:8d} {v['steps']:5d} "
          f"{str(int(bool(v['nc']))):>3s} {str(int(bool(v['ns']))):>3s} "
          f"{v['dT']:+9.4f} {v['dino']:7.4f} {v['lpips']:7.4f} {v['frac']:6.2f} "
          f"{str(v['styles']):>4s} {v['er']:+8.4f} {v['imp']:+8.4f} {v['mn']:+8.4f} "
          f"{v['roc']:+8.4f} {v['ukiyo']:+8.4f}")

ref = rows.get("A0_ref") or rows.get("D1_c1_gentle")
if ref:
    print()
    print(f"=== deltas vs {('A0_ref' if 'A0_ref' in rows else 'D1_c1_gentle')} ===")
    print(f"{'variant':16s} {'d_dT':>9s} {'d_dino':>9s} {'d_lpips':>9s} {'d_ukiyo':>9s} "
          f"{'verdict':s}")
    for n in ORDER:
        if n not in rows or n in ("A0_ref", "D1_c1_gentle"):
            continue
        v = rows[n]
        dd = v['dT'] - ref['dT']
        ddn = v['dino'] - ref['dino']
        dl = v['lpips'] - ref['lpips']
        du = v['ukiyo'] - ref['ukiyo']
        bad = (dd < -0.005) or (ddn < -0.02)
        print(f"{n:16s} {dd:+9.4f} {ddn:+9.4f} {dl:+9.4f} {du:+9.4f} "
              f"{'NECESSARY (degrades)' if bad else 'no measurable effect'}")
PY
