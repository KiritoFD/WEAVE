#!/usr/bin/env bash
# Dump the full V6 ablation table from the existing report.json files.
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
python3 - <<'PY'
import json
from pathlib import Path

rows = []
for d in sorted(Path("runs/v6").iterdir()):
    rp = d / "report.json"
    if not rp.exists() or d.name == "cache":
        continue
    r = json.loads(rp.read_text())
    rows.append((d.name, r))

print("=" * 100)
print("A. TRAINING CONFIG OF EACH EXISTING VARIANT")
print("=" * 100)
keys = ["name", "mode", "canon_mode", "n_params", "steps", "effective_batch",
        "coherence_beta", "w_style_dist", "w_cross", "w_canon_inv", "lr", "seed",
        "style_noise", "content_dropout"]
hdr = [k for k in keys]
print(" | ".join(f"{h[:12]:>12s}" for h in hdr))
for name, r in rows:
    if "n_params" not in r:
        continue
    vals = []
    for h in hdr:
        v = r.get(h, "")
        if isinstance(v, float):
            v = f"{v:.4g}"
        vals.append(f"{str(v)[:12]:>12s}")
    print(" | ".join(vals))

print()
print("=" * 100)
print("B. ANY EVAL METRICS PRESENT IN THE REPORTS")
print("=" * 100)
for name, r in rows:
    found = {}
    def walk(o, path=""):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, f"{path}.{k}" if path else k)
        elif isinstance(o, (int, float)) and not isinstance(o, bool):
            kl = path.lower()
            if any(t in kl for t in ("dt", "d_t", "dino", "lpips", "top1", "margin",
                                     "swd", "mmd", "dS", "acc")):
                found[path] = o
    walk(r)
    if found:
        print(f"\n  --- {name} ---")
        for k, v in list(found.items())[:20]:
            print(f"      {k} = {v:.6f}" if isinstance(v, float) else f"      {k} = {v}")
    else:
        print(f"\n  --- {name}: no eval metrics stored in report.json ---")

print()
print("=" * 100)
print("C. TOP-LEVEL SUMMARY FILES")
print("=" * 100)
for f in ("stage4_final750_summary.json", "sweep_inference_strength.json",
          "stage3_pareto_calibration.json", "baseline_reval_summary.json",
          "score_diff_diagnosis_and_calibration.json"):
    p = Path("runs/v6") / f
    if not p.exists():
        print(f"  {f}: missing"); continue
    d = json.loads(p.read_text())
    print(f"\n  --- {f} ---")
    if isinstance(d, dict):
        for k, v in list(d.items())[:14]:
            s = json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
            print(f"      {k}: {str(s)[:160]}")
PY
