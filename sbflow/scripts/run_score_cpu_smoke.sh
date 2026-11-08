#!/usr/bin/env bash
set -euo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
rm -f eval_protocol/cache/references_v1_csd_cpu.pt
python3 -u -m sbflow.score_cpu --repo-root . \
  --gen runs/blog_sbflow/v10/_e2e_sdvae/gen \
  --out runs/blog_sbflow/v10/_e2e_sdvae/score \
  --batch 4 --threads 8 --limit 6
python3 - <<'PY'
import json
d = json.load(open("runs/blog_sbflow/v10/_e2e_sdvae/score/style_scores.json"))
print("gamma", d["protocol"]["mmd_gamma"])
o = d["aggregate"]["overall"]
print({k: round(v, 4) for k, v in o.items() if isinstance(v, float)})
PY
