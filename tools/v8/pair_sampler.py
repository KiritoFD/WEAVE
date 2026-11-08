"""V8 endpoint pairing: random coupling and content-aware log-domain Sinkhorn OT.

The OT cost uses C1-canonicalized LL pooled structure + a true spatial
self-similarity matrix distance, computed on cached D5 latents.
No ScoreNet, no RGB generation, no barycentric average.

Fix log (P0):
  * Sinkhorn now uses the correct log-domain dual recursion without confusing the
    entropic regularization with the dual potential scaling:
        log_u = log_a - logsumexp(K + log_v, dim=1)
        log_v = log_b - logsumexp(K + log_u, dim=0)
        plan  = exp(K + log_u + log_v)
  * The feature vector now includes a genuine SSM (N x N patch cosine matrix),
    not an outer product of a flattened vector.
  * OT sampling uses a dedicated generator so it never consumes the training
    generator, keeping the B0-B3 source/target/tau/noise stream identical.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from wavelet import dwt2_haar
from tools.v6.canonicalize import ContentCanonicalizer


def _spatial_self_similarity(feat: torch.Tensor, grid: int = 8) -> torch.Tensor:
    """True N x N spatial self-similarity matrix of pooled patch features.

    feat: (B, C, H, W) -> (B, grid*grid) upper-triangular, normalized.
    """
    p = F.adaptive_avg_pool2d(feat, (grid, grid))          # (B, C, g, g)
    B, C = p.shape[:2]
    v = p.reshape(B, C, grid * grid).transpose(1, 2)        # (B, N, C)
    v = F.normalize(v, dim=-1)
    ssm = torch.bmm(v, v.transpose(1, 2))                   # (B, N, N)
    idx = torch.triu_indices(grid * grid, grid * grid, offset=1, device=feat.device)
    return F.normalize(ssm[:, idx[0], idx[1]], dim=-1)


def c1_content_feature(z: torch.Tensor, canon: ContentCanonicalizer, pool: int = 8) -> torch.Tensor:
    """Pooled C1 LL structure + true spatial self-similarity pattern."""
    with torch.no_grad():
        c = canon(z.float(), mode="c1", augment=False)
        ll, _, _, _ = dwt2_haar(c)
        p = F.adaptive_avg_pool2d(ll, (pool, pool)).flatten(1)
        p = F.normalize(p, dim=-1)
        ssm = _spatial_self_similarity(ll, grid=pool)
        return F.normalize(torch.cat([p, 0.25 * ssm], dim=-1), dim=-1)


def sinkhorn_log(cost: torch.Tensor, reg: float = 0.08, iters: int = 200) -> tuple[torch.Tensor, dict]:
    """Correct log-domain Sinkhorn with uniform marginals.

    K = -C / reg; dual updates are applied to the exponents (log_u, log_v) with
    scaling 1, so plan = exp(K + log_u + log_v) satisfies both marginals.
    """
    n, m = cost.shape
    K = -cost / max(reg, 1e-8)
    log_a = torch.full((n, 1), -float(np.log(float(n))), device=cost.device, dtype=cost.dtype)
    log_b = torch.full((1, m), -float(np.log(float(m))), device=cost.device, dtype=cost.dtype)
    log_u = torch.zeros(n, 1, device=cost.device, dtype=cost.dtype)
    log_v = torch.zeros(1, m, device=cost.device, dtype=cost.dtype)
    for _ in range(iters):
        log_u = log_a - torch.logsumexp(K + log_v, dim=1, keepdim=True)
        log_v = log_b - torch.logsumexp(K + log_u, dim=0, keepdim=True)
    plan = torch.exp(K + log_u + log_v)
    row_err = float((plan.sum(dim=1) - 1.0 / n).abs().max().item())
    col_err = float((plan.sum(dim=0) - 1.0 / m).abs().max().item())
    return plan, {"row_marginal_max_err": row_err, "col_marginal_max_err": col_err,
                  "row_min_expected": 1.0 / m}


def ot_sample_targets(
    z0: torch.Tensor,
    z1_pool: torch.Tensor,
    canon: ContentCanonicalizer,
    reg: float = 0.08,
    seed: int | None = None,
) -> tuple[torch.Tensor, dict]:
    """Sample one real target endpoint per source from the OT plan."""
    with torch.no_grad():
        f0 = c1_content_feature(z0, canon)
        f1 = c1_content_feature(z1_pool, canon)
        cost = torch.cdist(f0, f1).pow(2)
        plan, diag = sinkhorn_log(cost, reg=reg, iters=200)
        B, M = plan.shape
        logits = torch.log(plan.clamp_min(1e-30))
        # Dedicated generator: never touches the training sampling stream.
        gen = torch.Generator(device=plan.device)
        gen.manual_seed(int(seed) if seed is not None else 12345)
        u = torch.rand(logits.shape, generator=gen, device=plan.device, dtype=plan.dtype).clamp_(1e-6, 1 - 1e-6)
        g = -torch.log(-torch.log(u))
        score = logits + g
        used = torch.zeros(M, dtype=torch.bool, device=plan.device)
        chosen = []
        for i in range(B):
            s = score[i].masked_fill(used, -1e9)
            j = int(torch.argmax(s).item())
            chosen.append(j)
            if not used.all():
                used[j] = True
        idx = torch.tensor(chosen, device=z0.device, dtype=torch.long)
        matched_cost = cost[torch.arange(B, device=cost.device), idx]
        return z1_pool[idx], {
            "coverage_unique": int(torch.unique(idx).numel()),
            "coverage_ratio": float(torch.unique(idx).numel() / max(B, 1)),
            "matched_cost": float(matched_cost.mean().item()),
            "random_cost": float(cost.mean().item()),
            "plan_row_max": float(plan.max(dim=1).values.mean().item()),
            "plan_row_max_expected_min": diag["row_min_expected"],
            "row_marginal_max_err": diag["row_marginal_max_err"],
            "col_marginal_max_err": diag["col_marginal_max_err"],
        }
