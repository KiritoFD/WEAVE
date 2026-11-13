"""Style energies over real-painting patch banks.

Design decision recorded here because it determined the whole V8 objective
design: a FIXED-weight squared distance to a prototype set is algebraically
just regression to a weighted mean,

    E(h) = sum_j pi_j ||h - a_j||^2 = ||h - mu||^2 + const,  mu = sum_j pi_j a_j

which cannot represent a multi-modal style. So the default support energy is a
mixture soft-min whose responsibilities depend on the CURRENT generated patch,
and OT compares the generated patch SET to the prototype set rather than each
patch to one averaged vector.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .features import N_PATCH


def squared_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """(B,N,D) vs (K,D) or (B,N,D) vs (B,K,D) -> (B,N,K)."""
    if y.dim() == 2:
        return (x.square().sum(-1, keepdim=True)
                - 2 * x @ y.t() + y.square().sum(-1)[None, None, :]).clamp_min_(0)
    return (x.square().sum(-1, keepdim=True)
            - 2 * (x @ y.transpose(-1, -2)) + y.square().sum(-1).unsqueeze(1)).clamp_min_(0)


def geometry_prior(q_source: torch.Tensor, q_proto: torch.Tensor,
                   temperature: float = 1.0) -> torch.Tensor:
    """(B,N,K) geometry weights, detached: geometry only chooses which real
    patches are relevant, it never receives gradient."""
    d = squared_distance(q_source.detach(), q_proto.detach())
    return torch.softmax(-d / temperature, dim=-1)


class CentroidEnergy:
    """M1: fixed-weight mean regression. Kept as the explicit low bound."""

    def __call__(self, features, prototypes, weights=None):
        d = squared_distance(features, prototypes)
        w = (torch.full_like(d, 1.0 / d.shape[-1]) if weights is None
             else weights / weights.sum(-1, keepdim=True).clamp_min(1e-12))
        loss = (w * d).sum(-1).mean(-1)
        return {"loss": loss, "support": loss, "distance": d, "weights": w.detach(),
                "gradient_mode": "fixed_weight_mean"}


class MixtureSupportEnergy:
    """M2: -tau * logsumexp_j (log w_j - d_j/tau). Responsibilities depend on h."""

    def __init__(self, temperature: float = 1.0):
        if temperature <= 0:
            raise ValueError("positive temperature required")
        self.temperature = float(temperature)

    def __call__(self, features, prototypes, weights=None):
        d = squared_distance(features, prototypes)
        if weights is None:
            logw = torch.zeros_like(d) - torch.log(torch.tensor(d.shape[-1], dtype=d.dtype, device=d.device))
        else:
            logw = (weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)).clamp_min(1e-12).log()
        logits = logw - d / self.temperature
        per_patch = -self.temperature * torch.logsumexp(logits, -1)
        resp = torch.softmax(logits, -1)
        ent = -(resp * resp.clamp_min(1e-12).log()).sum(-1)
        return {"loss": per_patch.mean(-1), "support": per_patch.mean(-1),
                "per_patch": per_patch, "responsibilities": resp,
                "effective_modes": ent.exp().mean(-1),
                "prototype_usage": resp.mean(1),
                "gradient_mode": "mixture_responsibility"}


class PatchOTEnergy:
    """M3: per-image entropic OT between the generated patch set and prototypes.

    Marginals are fixed per image, so different images in a batch cannot couple.
    Solved in the log domain. The prototype set is CHUNKED: 8x64 patches against
    several thousand prototypes is a large cost tensor, and building it in one
    shot OOMs a 12 GB card. A non-converged solve raises rather than silently
    returning a partial plan.
    """

    def __init__(self, *, epsilon: float = 0.1, geometry_weight: float = 0.0,
                 max_iter: int = 2000, tolerance: float = 1e-3,
                 max_prototypes: int = 512):
        self.epsilon = epsilon
        self.geometry_weight = geometry_weight
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.max_prototypes = max_prototypes

    def _solve(self, cost: torch.Tensor) -> dict:
        """Log-domain Sinkhorn on one (B,N,K) cost tensor with fixed marginals."""
        b, n, k = cost.shape
        logk = -cost / self.epsilon
        logr = torch.full((b, n, 1), -torch.log(torch.tensor(float(n))), device=cost.device)
        logc = torch.full((b, 1, k), -torch.log(torch.tensor(float(k))), device=cost.device)
        r_target = torch.exp(logr); c_target = torch.exp(logc)
        u = torch.zeros_like(logr); v = torch.zeros_like(logc)
        plan = None
        row_err = col_err = float("inf")
        used = 0
        for it in range(self.max_iter):
            used = it + 1
            u = logr - torch.logsumexp(logk + v, dim=2, keepdim=True)
            v = logc - torch.logsumexp(logk + u, dim=1, keepdim=True)
            if (it + 1) % 10 == 0 or it == self.max_iter - 1:
                plan = torch.exp(logk + u + v)
                row_err = float((plan.sum(2) - r_target.squeeze(2)).abs().max())
                col_err = float((plan.sum(1) - c_target.squeeze(1)).abs().max())
                if max(row_err, col_err) <= self.tolerance:
                    return {"plan": plan, "row_error": row_err,
                            "column_error": col_err, "iterations": used}
        raise RuntimeError(
            f"Sinkhorn did not converge in {used} iters "
            f"(row_err={row_err:.2e}, col_err={col_err:.2e}); refusing a partial plan")

    def __call__(self, features, prototypes, *, q_source=None, q_prototypes=None):
        b, n, _ = features.shape
        k = prototypes.shape[-2] if prototypes.dim() == 3 else prototypes.shape[0]
        chunk = self.max_prototypes
        total = features.new_zeros(b)
        usage = features.new_zeros(b, k)
        worst_row = worst_col = 0.0
        iters = 0
        for start in range(0, k, chunk):
            stop = min(start + chunk, k)
            proto = prototypes[..., start:stop, :] if prototypes.dim() == 3 \
                else prototypes[start:stop]
            # The plan is solved with detached marginals ONLY; the returned loss
            # keeps the autograd path features -> cost -> plan*cost, which is the
            # fixed-plan envelope gradient.
            with torch.no_grad():
                cost_d = squared_distance(features.detach(), proto.detach())
                if self.geometry_weight and q_source is not None and q_prototypes is not None:
                    qp = q_prototypes[..., start:stop, :] if q_prototypes.dim() == 3 \
                        else q_prototypes[start:stop]
                    cost_d = cost_d + self.geometry_weight * squared_distance(
                        q_source.detach(), qp.detach())
                sol = self._solve(cost_d)
                plan = sol["plan"]
            cost = squared_distance(features, proto)
            if self.geometry_weight and q_source is not None and q_prototypes is not None:
                qp = q_prototypes[..., start:stop, :] if q_prototypes.dim() == 3 \
                    else q_prototypes[start:stop]
                cost = cost + self.geometry_weight * squared_distance(
                    q_source.detach(), qp.detach())
            total = total + (plan * cost).sum((-2, -1))
            usage[:, start:stop] = plan.sum(1)
            worst_row = max(worst_row, sol["row_error"])
            worst_col = max(worst_col, sol["column_error"])
            iters += sol["iterations"]
        return {"loss": total, "support": total,
                "row_error": worst_row, "column_error": worst_col,
                "iterations": iters, "converged": True,
                "prototype_usage": usage,
                "gradient_mode": "converged_detached_envelope"}


class SlicedWassersteinEnergy:
    """M3s: set-level SWD with fixed random directions, as an OT-free control."""

    def __init__(self, feature_dim: int, n_projections: int = 64, seed: int = 42):
        g = torch.Generator().manual_seed(seed)
        d = torch.randn(feature_dim, n_projections, generator=g, dtype=torch.float64)
        self.directions = (d / d.norm(dim=0, keepdim=True)).float()
        self.n_projections = n_projections

    def __call__(self, features, prototypes, *, n_quantiles: int = 128, **kw):
        dirs = self.directions.to(features.device)
        a = (features @ dirs).transpose(1, 2)          # (B,P,N)
        b = (prototypes @ dirs).transpose(1, 2)        # (B,P,K) or (P,K)
        if b.dim() == 2:
            b = b.unsqueeze(0).expand(a.shape[0], -1, -1)
        qs = torch.linspace(0, 1, n_quantiles, device=features.device)
        qa = torch.quantile(a, qs, dim=2)
        qb = torch.quantile(b, qs, dim=2)
        w = (qa - qb).square().mean(dim=(1, 2))
        return {"loss": w, "support": w, "gradient_mode": "sliced_quantile"}


class BoundedSpecificityEnergy:
    """M4/M5: absolute target support plus a BOUNDED competitor term.

    b(E) = L*tanh(E/L) is finite and its derivative is bounded, so a model
    cannot gain by running away from every real painting; the target support
    term is always retained.
    """

    def __init__(self, margin: float = 0.2, competitor_bound: float = 5.0,
                 support_weight: float = 1.0, contrast_weight: float = 0.25):
        self.margin = margin
        self.bound = competitor_bound
        self.support_weight = support_weight
        self.contrast_weight = contrast_weight

    def __call__(self, energies, target_ids):
        """energies: (B,S) calibrated absolute energies; lower is better."""
        b, s = energies.shape
        tgt = energies.gather(1, target_ids[:, None]).squeeze(1)
        mask = F.one_hot(target_ids, s).bool()
        others = energies.masked_fill(mask, float("inf"))
        comp = torch.tanh(others / self.bound).mean(1) * self.bound
        gap = tgt - comp
        loss = self.support_weight * tgt + self.contrast_weight * F.softplus(gap + self.margin)
        return {"loss": loss, "support": tgt, "absolute_support": tgt,
                "contrast": comp, "gap": gap, "gradient_mode": "bounded_contrast"}
