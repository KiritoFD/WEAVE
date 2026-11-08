"""V8 endpoint pairing: random coupling and content-aware log-domain Sinkhorn OT.

The OT cost uses only C1 canonicalized LL pooled structure + spatial self-similarity,
computed on cached D5 latents; no ScoreNet, no RGB generation, no barycentric avg.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from wavelet import dwt2_haar
from tools.v6.canonicalize import ContentCanonicalizer


def c1_content_feature(z: torch.Tensor, canon: ContentCanonicalizer, pool: int = 8) -> torch.Tensor:
    """Pooled C1 LL structure + flattened normalized self-similarity."""
    with torch.no_grad():
        c = canon(z.float(), mode="c1", augment=False)
        ll, _, _, _ = dwt2_haar(c)
        p = F.adaptive_avg_pool2d(ll, (pool, pool)).flatten(1)
        p = F.normalize(p, dim=-1)
        ssm = torch.bmm(p.unsqueeze(1).transpose(1, 2), p.unsqueeze(1)).squeeze(1).flatten(1)
        ssm = F.normalize(ssm, dim=-1)
        return F.normalize(torch.cat([p, 0.25 * ssm], dim=-1), dim=-1)


def sinkhorn_log(cost: torch.Tensor, reg: float = 0.08, iters: int = 80) -> torch.Tensor:
    """Numerically stable log-domain Sinkhorn with uniform marginals."""
    n, m = cost.shape
    K = -cost / max(reg, 1e-6)
    f = torch.zeros(n, 1, device=cost.device, dtype=cost.dtype)
    g = torch.zeros(1, m, device=cost.device, dtype=cost.dtype)
    log_a = -torch.full((n, 1), float(n), device=cost.device, dtype=cost.dtype).log()
    log_b = -torch.full((1, m), float(m), device=cost.device, dtype=cost.dtype).log()
    for _ in range(iters):
        f = reg * (log_a - torch.logsumexp(K + g, dim=1, keepdim=True))
        g = reg * (log_b - torch.logsumexp(K + f, dim=0, keepdim=True))
    return torch.exp(K + f + g)


def ot_sample_targets(
    z0: torch.Tensor,
    z1_pool: torch.Tensor,
    canon: ContentCanonicalizer,
    reg: float = 0.08,
    seed: int | None = None,
) -> tuple[torch.Tensor, dict]:
    """Sample one real target endpoint per source from an OT plan with coverage constraint."""
    with torch.no_grad():
        f0 = c1_content_feature(z0, canon)
        f1 = c1_content_feature(z1_pool, canon)
        cost = torch.cdist(f0, f1).pow(2)
        plan = sinkhorn_log(cost, reg=reg, iters=80)
        B, M = plan.shape
        # Gumbel-top-k with each column used at most once: ensures real endpoints and
        # records target coverage instead of silently collapsing to a few samples.
        logits = torch.log(plan.clamp_min(1e-12))
        if seed is None:
            g = torch.distributions.Gumbel(torch.zeros_like(logits), torch.ones_like(logits)).sample()
        else:
            gen = torch.Generator(device=plan.device).manual_seed(seed)
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
        rand_cost = cost.mean()
        return z1_pool[idx], {
            "coverage_unique": int(torch.unique(idx).numel()),
            "coverage_ratio": float(torch.unique(idx).numel() / max(B, 1)),
            "matched_cost": float(matched_cost.mean().item()),
            "random_cost": float(rand_cost.item()),
            "plan_row_max": float(plan.max(dim=1).values.mean().item()),
        }
