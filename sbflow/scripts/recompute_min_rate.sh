#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
python3 - <<'PY'
import json
from pathlib import Path
STY = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
MIN = STY.index("Minimalism")

def rate_from_conf(conf):
    hits = sum(conf[i][MIN] for i in range(5) if i != MIN)
    tot = sum(sum(conf[i]) for i in range(5) if i != MIN)
    return hits, tot

print("CORRECTED non-target Minimalism misclassification")
print("(the earlier 0.000 was a str-vs-int type bug in q0_verify.offdiag_min)\n")
print(f"  {'run':46s}{'hits':>7s}{'total':>7s}{'rate':>8s}")
for tag, p in (
    ("Q2a Heldout400", "runs/v6_next/eval_heldout400_Q2a_embed_style/eval.json"),
    ("G1b Heldout400", "runs/v6_next/eval_heldout400_P2_G1b_bf16/eval.json"),
    ("L11 Heldout400", "runs/v6_24h/eval_heldout400_L11/eval.json"),
    ("L8 Heldout400", "runs/v6_24h/eval_heldout400_L8/eval.json"),
    ("Q2a Same150", "runs/v6_next/eval_same150_Q2a_embed_style/eval.json"),
):
    p = Path(p)
    if not p.exists():
        print(f"  {tag:46s}  missing"); continue
    d = json.loads(p.read_text())
    conf = d.get("confusion_target_vs_pred")
    if not conf:
        print(f"  {tag:46s}  no confusion"); continue
    h, t = rate_from_conf(conf)
    print(f"  {tag:46s}{h:7d}{t:7d}{h/max(1,t):8.3f}")

d = json.loads(Path("runs/v6_next/amortize/p0_variants.json").read_text())
for v in ("A0", "A1", "A2", "A4"):
    conf = d["variants"][v].get("confusion")
    if not conf:
        continue
    h, t = rate_from_conf(conf)
    print(f"  {'P0 '+v+' Test600':46s}{h:7d}{t:7d}{h/max(1,t):8.3f}")
PY
