"""Mean DINO-S / DINO-C / structure-MSE over the 750 rendered pairs (from dino_metrics.csv)."""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path("/mnt/i/Latent_Style/eval_assets/p2_render")
for cell in ["S0_base", "S1_positive", "S2_reverse", "A"]:
    p = ROOT / cell / "dino_metrics.csv"
    if not p.exists():
        print(f"[{cell}] missing")
        continue
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    n = len(rows)
    means = {k: sum(float(r[k]) for r in rows) / n for k in
             ("dino_style_similarity", "dino_content_similarity", "dino_structure_mse")}
    print(f"[{cell}] n={n} " + " ".join(f"{k}={v:.4f}" for k, v in means.items()))
