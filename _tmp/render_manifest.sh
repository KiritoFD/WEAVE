#!/usr/bin/env bash
cd /mnt/i/Latent_Style/pseudo_gt_24h/bench_render || exit 1
python3 - <<'PY'
import json
for tag in ("bench_b32", "bench_b64", "bench_b96", "bench_b128", "b64_t8"):
    d = json.load(open(f"{tag}/render_manifest.json"))
    print(tag, {k: (v if not isinstance(v, (list, dict)) else len(v)) for k, v in d.items()})
PY
