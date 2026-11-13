#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
python3 - <<'PY'
import json
from pathlib import Path
rows = []
for name, path in (("A0 (dim48)", "runs/v6/D1_c1_gentle/report.json"),
                   ("N3_dim32", "runs/v6/N3_dim32/report.json"),
                   ("N7_dim24", "runs/v6/N7_dim24/report.json")):
    p = Path(path)
    if not p.exists():
        print(f"{name}: missing"); continue
    r = json.loads(p.read_text())
    e = r.get("eval200") or {}
    rows.append((name, r.get("n_params"), r.get("train_wall_sec"),
                 r.get("peak_vram_mib"), e.get("mean_dT"), e.get("mean_dino_c"),
                 e.get("mean_lpips")))
print(f"{'arm':14s}{'params':>10s}{'train_s':>9s}{'vram':>7s}{'dT':>9s}{'DINO-C':>8s}{'LPIPS':>8s}")
for n, p, t, v, d, dc, lp in rows:
    print(f"{n:14s}{(f'{p:,}' if p else '—'):>10s}{(f'{t:.0f}' if t else '—'):>9s}"
          f"{(f'{v:.0f}' if v else '—'):>7s}{(f'{d:+.4f}' if d else '—'):>9s}"
          f"{(f'{dc:.4f}' if dc else '—'):>8s}{(f'{lp:.4f}' if lp else '—'):>8s}")
if len(rows) >= 2:
    a, b = rows[0][1], rows[-1][1]
    if a and b:
        print(f"\ndim24 vs dim48: params {b:,} vs {a:,} = {b/a*100:.1f}%  (-{(1-b/a)*100:.1f}%)")
PY
