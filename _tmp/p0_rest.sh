#!/usr/bin/env bash
cd /mnt/i/Latent_Style/WEAVE || exit 1
python3 - <<'PY'
import json
d = json.load(open("runs/p2/pgt_p0_audit.json"))
for s in ("P5", "P6", "P9"):
    v = d["specs"].get(s)
    if v:
        print(s, {k: v[k] for k in ("n", "failed", "nominal_coverage", "coverage_mean", "coverage_min", "patches_mean", "frac_images_with_drop")})
for k, v in d["reverse_algebra"]["configs"].items():
    print(k, "dT_R=%.4f[%.4f,%.4f] dS_R=%.4f dM_R=%.4f legacy_dT=%.4f" % (
        v["dT_R_mean"], v["dT_R_ci"]["lo"], v["dT_R_ci"]["hi"], v["dS_R_mean"], v["dM_R_mean"], v["legacy_dT_mean"]))
r = d["cache"].get("reverse", {})
print("reverse_cache", {k: r.get(k) for k in ("records", "sources", "dup", "nominal_cov_mean", "effective_cov_mean")})
PY
echo "--- supp progress"
tail -n 3 runs/p2/supp_full.log 2>/dev/null
echo "rows=$(grep -c '' runs/p2/supp_records.jsonl 2>/dev/null) pool2=$(grep -c '' runs/p2/supp_pool2.jsonl 2>/dev/null) pool3=$(grep -c '' runs/p2/supp_pool3.jsonl 2>/dev/null)"
echo "--- progress log"; tail -n 12 runs/gt/PROGRESS.md
