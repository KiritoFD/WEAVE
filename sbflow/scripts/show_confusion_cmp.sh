#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
python3 - <<'PY'
import json
from pathlib import Path
STY = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]

def show(tag, p):
    p = Path(p)
    if not p.exists():
        print(f"{tag}: missing"); return
    d = json.loads(p.read_text())
    conf = d.get("confusion_target_vs_pred")
    if not conf:
        print(f"{tag}: no confusion matrix"); return
    print(f"\n{tag}  (rows=target, cols=pred)")
    print(f"  {'':22s}" + "".join(f"{s[:9]:>11s}" for s in STY))
    for i, s in enumerate(STY):
        print(f"  {s:22s}" + "".join(f"{c:11d}" for c in conf[i]))
    nm = [i for i, s in enumerate(STY) if s != "Minimalism"]
    hits = sum(conf[i][2] for i in nm)
    tot = sum(sum(conf[i]) for i in nm)
    print(f"  non-Min targets predicted Minimalism: {hits}/{tot} = {hits/max(1,tot):.3f}")

show("Q2a Heldout400", "runs/v6_next/eval_heldout400_Q2a_embed_style/eval.json")
show("Q2a Test600 (P0 A0)", "runs/v6_next/amortize/p0_variants.json")
PY
