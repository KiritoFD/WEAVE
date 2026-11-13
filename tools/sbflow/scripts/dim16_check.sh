#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
python3 - <<'PY'
import json
from pathlib import Path

arms = [("A0", "runs/v6/D1_c1_gentle/report.json"),
        ("L4_dim48_k1024", "runs/v6/L4_ref_full/report.json"),
        ("L8_dim32_k1024", "runs/v6/L8_k1024_dim32/report.json"),
        ("L10_dim24_k1024", "runs/v6/L10_k1024_dim24/report.json"),
        ("L11_dim16_k1024", "runs/v6/L11_k1024_dim16/report.json")]

hdr = "%-18s%10s%9s%8s%8s%5s%8s%6s" % ("arm", "params", "dT", "DINO-C", "LPIPS",
                                      "sty", "sameDS", "top1")
print(hdr)
for name, p in arms:
    f = Path(p)
    if not f.exists():
        print("%-18s  missing" % name)
        continue
    r = json.loads(f.read_text())
    e = r.get("eval200") or {}
    pt = e.get("per_target_top1") or {}
    t1 = sum(pt.values()) / len(pt) if pt else None
    n = r.get("n_params")
    dt = e.get("mean_dT")
    dc = e.get("mean_dino_c")
    lp = e.get("mean_lpips")
    st = e.get("styles_improved")
    sd = e.get("same50_mean_dS")
    print("%-18s%10s%9s%8s%8s%5s%8s%6s" % (
        name,
        "{:,}".format(n) if n else "-",
        "%+.4f" % dt if dt is not None else "-",
        "%.4f" % dc if dc is not None else "-",
        "%.4f" % lp if lp is not None else "-",
        "%d/5" % st if st is not None else "-",
        "%+.4f" % sd if sd is not None else "-",
        "%.3f" % t1 if t1 is not None else "-"))
PY
