#!/usr/bin/env bash
cd /mnt/i/Latent_Style/WEAVE || exit 1
python3 - <<'PY'
import json
d = json.load(open("runs/p2/pgt_S0_base_b256.json"))
m = d["metrics"]
print("top keys", sorted(d.keys()))
print("metrics keys", sorted(m.keys())[:40])
print("same_style_150", json.dumps(d["same_style_150"], ensure_ascii=False)[:800])
print("per_target_style", json.dumps(d["per_target_style"], ensure_ascii=False)[:800])
print("n_cross", d["n_cross_pairs"], "n_src", d["n_sources"])
PY
