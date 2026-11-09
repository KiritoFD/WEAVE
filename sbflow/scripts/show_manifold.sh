#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
echo "=== manifold files present ==="
ls -la runs/v6_24h/*.json 2>/dev/null | grep -i manifold
echo
echo "=== content ==="
for f in runs/v6_24h/csd_manifold_best.json runs/v6_24h/csd_manifold.json; do
  [ -f "$f" ] || continue
  echo "-- $f"
  python3 - "$f" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print("  ckpt:", d.get("ckpt"))
for pop in d.get("populations", []):
    print("  %-26s n=%4d  max_cos=%.4f (p10 %.4f)  margin=%.4f (p10 %.4f)  spread=%.4f"
          % (pop["label"], pop["n"], pop["max_cos_mean"], pop["max_cos_p10"],
             pop["margin_mean"], pop["margin_p10"], pop["spread_mean"]))
PY
done
