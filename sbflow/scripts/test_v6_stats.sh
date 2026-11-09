#!/usr/bin/env bash
# Unit-test the source-clustered bootstrap: it must be WIDER than the naive task-level
# bootstrap when tasks are correlated within a source, and the paired form must be
# narrower than comparing two independent arms.
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
python3 - <<'PY'
import numpy as np
from tools.v6_24h.stats import cluster_ci, paired_cluster_ci

rng = np.random.default_rng(0)
n_cl, per_cl = 40, 5
# strong within-source correlation: every task of a source shares a source effect
src_eff = rng.normal(0, 1.0, n_cl)
vals, clust = [], []
for c in range(n_cl):
    for _ in range(per_cl):
        vals.append(src_eff[c] + rng.normal(0, 0.05))
        clust.append(f"s{c}")
vals = np.asarray(vals)

naive = [float(np.percentile(vals[rng.integers(0, len(vals), (4000, len(vals)))].mean(1), q))
         for q in (2.5, 97.5)]
clu = cluster_ci(vals, clust, n_boot=4000, seed=1)
w_naive = naive[1] - naive[0]
w_clu = clu["ci"][1] - clu["ci"][0]
print(f"naive CI width  = {w_naive:.4f}")
print(f"cluster CI width= {w_clu:.4f}   (clusters={clu['n_clusters']}, n={clu['n']})")
assert w_clu > w_naive * 1.5, "cluster CI should be much wider under strong clustering"
print("PASS clustered CI is wider than naive under within-source correlation")

# paired: arm A = vals, arm B = vals - 0.3 (same sources) -> difference must be tight
a = vals
b = vals - 0.3
p = paired_cluster_ci(a, b, clust, n_boot=4000, seed=2)
print(f"paired diff = {p['diff']:+.4f}  CI [{p['ci'][0]:+.4f}, {p['ci'][1]:+.4f}]  "
      f"excludes_zero={p['excludes_zero']}")
assert abs(p["diff"] - 0.3) < 1e-6, "paired diff must recover the constant offset"
assert (p["ci"][1] - p["ci"][0]) < 0.05, "paired CI must be tight for a constant offset"
print("PASS paired cluster CI recovers a constant offset with a tight interval")

# null case: A and B identical -> difference 0, CI must contain 0
p0 = paired_cluster_ci(vals, vals, clust, n_boot=2000, seed=3)
assert abs(p0["diff"]) < 1e-9 and not p0["excludes_zero"]
print("PASS identical arms give diff 0 with CI containing 0")
print("STATS TESTS PASSED")
PY
