#!/usr/bin/env bash
set -u
cd /mnt/i/Latent_Style/WEAVE || exit 1
export HF_HOME=/mnt/i/Latent_Style/eval_cache/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
for d in s0.30_cf1.5 identity nullcond_s0.15; do
  python3 tools/12h/v4_refpool_eval.py "runs/v4/prior/refpool/$d" --tag "$d" > "runs/v4/prior/refpool/$d/refpool.log" 2>&1
  echo "$d EXIT=$?"
  python3 - "$d" <<'PY'
import json, sys
d = sys.argv[1]
p = f"runs/v4/prior/refpool/{d}/refpool.json"
try:
    j = json.load(open(p))
    print("  " + " ".join(f"{k}={j[k]}" for k in ("n", "dino_style_pool", "dino_style_pool_x",
                                                 "clip_style_pool", "clip_style_pool_x",
                                                 "dino_content_pool", "dino_content_pool_x")))
except Exception as e:
    print("  FAILED", e)
PY
done
echo REFPOOL_ALLDONE

