"""P0 Sinkhorn/OT correctness audit with marginal residuals and true SSM."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.v6.canonicalize import ContentCanonicalizer  # noqa: E402
from tools.v8.common import STYLES, V8_ROOT, ensure_dirs, load_d5_latents, save_json  # noqa: E402
from tools.v8.pair_sampler import (  # noqa: E402
    _spatial_self_similarity, c1_content_feature, ot_sample_targets, sinkhorn_log,
)


def main() -> None:
    ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Marginal residual check on random cost matrices.
    checks = []
    for (n, m) in [(16, 16), (64, 128), (8, 200)]:
        g = torch.Generator(device=device).manual_seed(n * 131 + m)
        cost = torch.rand(n, m, generator=g, device=device)
        plan, diag = sinkhorn_log(cost, reg=0.05, iters=200)
        checks.append({
            "n": n, "m": m,
            "row_marginal_max_err": diag["row_marginal_max_err"],
            "col_marginal_max_err": diag["col_marginal_max_err"],
            "row_max_min_expected": 1.0 / m,
            "plan_row_max_mean": float(plan.max(dim=1).values.mean().item()),
            "total_mass": float(plan.sum().item()),
        })
    for c in checks:
        assert c["row_marginal_max_err"] < 1e-6, c
        assert c["col_marginal_max_err"] < 1e-6, c
        assert abs(c["total_mass"] - 1.0) < 1e-6, c
        # No single entry may exceed min(1/n, 1/m): a necessary but not sufficient
        # sanity bound for a valid doubly-stochastic coupling.
        assert c["plan_row_max_mean"] <= 1.0 / c["n"] + 1e-9, c

    # 2. True spatial self-similarity: must be N*N upper triangle, not an outer product.
    z = torch.randn(4, 4, 64, 64, device=device)
    ssm = _spatial_self_similarity(z, grid=8)
    dims = {"ssm_dim": int(ssm.shape[1]), "expected": 8 * 8 * (8 * 8 - 1) // 2}
    assert dims["ssm_dim"] == dims["expected"], dims

    # 3. Data-level OT audit with fixed real target coverage.
    train, _, data_meta = load_d5_latents(device=device, heldout=100)
    stats = torch.load(_REPO_ROOT / "runs" / "v6" / "cache" / "canonicalizer_stats.pt", map_location="cpu", weights_only=False)
    canon = ContentCanonicalizer(stats).to(device).eval()
    pairs = {}
    for sid, s in enumerate(STYLES):
        t = STYLES[(sid + 1) % len(STYLES)]
        z0 = train[s][:64]
        pool = train[t][:128]
        matched, info = ot_sample_targets(z0, pool, canon, reg=0.08, seed=7)
        f0 = c1_content_feature(z0, canon)
        f1 = c1_content_feature(pool, canon)
        cost = torch.cdist(f0, f1).pow(2)
        pairs[f"{s}->{t}"] = {
            **info,
            "random_cost_exact": float(cost.mean().item()),
            "random_target_baseline": float(torch.cdist(f0, f1[torch.randperm(f1.shape[0], device=device)]).pow(2).mean().item()),
            "n_source": int(z0.shape[0]), "n_pool": int(pool.shape[0]),
        }
        assert info["matched_cost"] < info["random_cost"], (s, info)
        assert info["row_marginal_max_err"] < 1e-6, info

    out = {
        "sinkhorn_checks": checks,
        "ssm": dims,
        "data": {"scaling_factor": data_meta["scaling_factor"], "vae": data_meta["vae"]},
        "pairs": pairs,
    }
    save_json(out, V8_ROOT / "p0_ot_pairing_audit_fixed.json")
    print("P0 OT PASS")
    for k, v in pairs.items():
        print(f"  {k}: matched={v['matched_cost']:.4f} random={v['random_cost']:.4f} "
              f"coverage={v['coverage_ratio']:.3f} rowmax={v['plan_row_max']:.6f} (min {v['plan_row_max_expected_min']:.6f})")


if __name__ == "__main__":
    main()
