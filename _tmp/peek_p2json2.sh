#!/usr/bin/env bash
cd /mnt/i/Latent_Style/WEAVE || exit 1
python3 - <<'PY'
import json
d = json.load(open("runs/p2/pgt_S0_base_b256.json"))
print("same keys", list(d["same_style_150"]["metrics"].keys()))
print("metrics ci sample", json.dumps(d["metrics"]["csd_dT"], ensure_ascii=False))
print("per_target keys", list(d["per_target_style"]["Rococo"].keys()))
print("per_source_style", json.dumps(d["per_source_style"], ensure_ascii=False)[:300])
print("timing", d.get("timing_sec"))
PY
