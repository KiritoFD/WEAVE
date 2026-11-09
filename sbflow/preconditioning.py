"""EDM-style drift preconditioning for bridge and direct-velocity targets.

Two coefficient families are provided:

`moment_coefficients` / `independent_unit_coefficients`
    the original blog-family preconditioner for the conditional bridge drifts
    b+ = (z1-zt)/(1-t) and b- = (z0-zt)/t.

`direct_velocity_coefficients`
    a re-derivation for the SINGLE-VELOCITY target (experiment 05)
        v* = (z1-z0) + sqrt(eps)(1-2t)/(2 sqrt(t(1-t))) xi
    The doc requires the input / skip / output coefficients be recomputed for this
    target rather than reusing the drift ones.

Style-aware statistics
----------------------
`StyleStats` holds one shared global mean/std plus per-style means and variances.
For a style pair with independent endpoints the cross covariance is zero, so the
ten pairs' EDM coefficients follow analytically from the five per-style statistics.
`StyleStats.pair(a, b)` builds the `PairMoments` for that pair, and
`StyleStats.pair(a, b, cross=...)` accepts an explicitly estimated cross
covariance for online alpha-DSBM (experiment 31).
"""
from dataclasses import dataclass

import torch


# --------------------------------------------------------------------------- #
# statistics containers
# --------------------------------------------------------------------------- #
@dataclass
class PairMoments:
    mu0: torch.Tensor
    mu1: torch.Tensor
    var0: torch.Tensor
    var1: torch.Tensor
    cov01: torch.Tensor

    def reversed(self):
        return PairMoments(self.mu1, self.mu0, self.var1, self.var0, self.cov01)

    def state_dict(self):
        return {k: getattr(self, k).detach().cpu() for k in ("mu0", "mu1", "var0", "var1", "cov01")}

    @classmethod
    def from_state_dict(cls, d):
        return cls(*(d[k] for k in ("mu0", "mu1", "var0", "var1", "cov01")))


class MomentAccumulator:
    """Accumulate per-channel first/second moments of paired endpoints."""

    def __init__(self):
        self.count = 0
        self.a = self.b = self.aa = self.bb = self.ab = None

    @torch.no_grad()
    def update(self, x0, x1):
        if x0.shape != x1.shape or x0.ndim != 4:
            raise ValueError("Require equal B,C,H,W endpoints")
        x0 = x0.detach().double()
        x1 = x1.detach().double()
        axes = (0, 2, 3)
        n = x0.shape[0] * x0.shape[2] * x0.shape[3]
        parts = (x0.sum(axes), x1.sum(axes), (x0 * x0).sum(axes),
                 (x1 * x1).sum(axes), (x0 * x1).sum(axes))
        if self.a is None:
            self.a, self.b, self.aa, self.bb, self.ab = [v.cpu() for v in parts]
        else:
            for attr, p in zip(("a", "b", "aa", "bb", "ab"), parts):
                setattr(self, attr, getattr(self, attr) + p.cpu())
        self.count += n

    def finalize(self):
        if self.count == 0:
            raise ValueError("No samples")
        a = self.a / self.count
        b = self.b / self.count
        va = self.aa / self.count - a * a
        vb = self.bb / self.count - b * b
        co = self.ab / self.count - a * b
        return PairMoments(a.float(), b.float(), va.float().clamp_min(1e-10),
                           vb.float().clamp_min(1e-10), co.float())


@dataclass
class StyleStats:
    """Shared global standardization plus per-style endpoint statistics.

    global_mean / global_std  : [C]
    style_mean / style_var    : [S, C]
    """

    global_mean: torch.Tensor
    global_std: torch.Tensor
    style_mean: torch.Tensor
    style_var: torch.Tensor

    @property
    def num_styles(self) -> int:
        return int(self.style_mean.shape[0])

    def pair(self, a: int, b: int, cross: torch.Tensor | None = None) -> PairMoments:
        """EDM moments for the unordered pair (a, b). Independent endpoints => cross = 0."""
        mu0 = self.style_mean[a]
        mu1 = self.style_mean[b]
        v0 = self.style_var[a].clamp_min(1e-10)
        v1 = self.style_var[b].clamp_min(1e-10)
        cov = torch.zeros_like(mu0) if cross is None else cross
        return PairMoments(mu0, mu1, v0, v1, cov)

    def all_pairs(self) -> dict:
        out = {}
        for a in range(self.num_styles):
            for b in range(a + 1, self.num_styles):
                out[(a, b)] = self.pair(a, b)
        return out

    def state_dict(self):
        return {k: getattr(self, k).detach().cpu()
                for k in ("global_mean", "global_std", "style_mean", "style_var")}

    @classmethod
    def from_state_dict(cls, d):
        return cls(*(d[k] for k in ("global_mean", "global_std", "style_mean", "style_var")))


@torch.no_grad()
def style_stats_from_caches(latents, eps=1e-10, equal_weight: bool = True) -> StyleStats:
    """Build StyleStats from a list of per-style latent tensors (already scaled, unnormalized).

    `equal_weight=True` gives every style the same weight in the global mean/std,
    which avoids letting the largest pool pull the shared latent centre toward itself
    (Impressionism is 57.8% of the 15,545 rows). Per-style statistics still use all of
    that style's data.
    """
    means = torch.stack([z.float().mean((0, 2, 3)) for z in latents])
    vars_ = torch.stack([z.float().var((0, 2, 3), unbiased=False) for z in latents])
    if equal_weight:
        gmean = means.mean(0)
        # E[var] + Var(E) with equal style weights
        gvar = (vars_ + means.square()).mean(0) - gmean.square()
    else:
        allz = torch.cat([z.float() for z in latents], dim=0)
        gmean = allz.mean((0, 2, 3))
        gvar = allz.var((0, 2, 3), unbiased=False)
    gstd = gvar.clamp_min(eps).sqrt()
    sm, sv = [], []
    for z in latents:
        zn = (z.float() - gmean[None, :, None, None]) / gstd[None, :, None, None]
        sm.append(zn.mean((0, 2, 3)))
        sv.append(zn.var((0, 2, 3), unbiased=False).clamp_min(eps))
    return StyleStats(gmean, gstd, torch.stack(sm), torch.stack(sv))


# --------------------------------------------------------------------------- #
# bridge-drift coefficients (original family)
# --------------------------------------------------------------------------- #
def independent_unit_coefficients(t, epsilon):
    """Closed forms for independent, zero-mean, unit-variance endpoints."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    v = (1 - t) ** 2 + t * t + epsilon * t * (1 - t)
    ci = torch.rsqrt(v)
    cs = ((2 - epsilon) * t - 1) / v
    co2 = ((1 - t) + epsilon * t) / ((1 - t) * v)
    return ci, cs, torch.sqrt(co2)


def moment_coefficients(t, epsilon, moments: PairMoments, tiny=1e-8):
    """Coefficients for the locally forward bridge drift, including endpoint means."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if (t <= 0).any() or (t >= 1).any():
        raise ValueError("t must be in (0,1)")
    dev = t.device
    dt = t.dtype

    def ch(v):
        return v.to(device=dev, dtype=dt).view(1, -1, 1, 1)

    a = (1 - t).view(-1, 1, 1, 1)
    b = t.view(-1, 1, 1, 1)
    mu0, mu1 = ch(moments.mu0), ch(moments.mu1)
    v0, v1, cov = ch(moments.var0), ch(moments.var1), ch(moments.cov01)
    mx = a * mu0 + b * mu1
    my = mu1 - mu0
    vx = (a * a * v0 + b * b * v1 + 2 * a * b * cov + epsilon * a * b).clamp_min(tiny)
    var_d = (v0 + v1 - 2 * cov).clamp_min(0)
    cyx = a * (cov - v0) + b * (v1 - cov) - epsilon * b
    vy = var_d + epsilon * b / a
    ci = vx.rsqrt()
    cs = cyx / vx
    resvar = (vy - cyx * cyx / vx).clamp_min(tiny)
    co = resvar.sqrt()
    return mx, my, ci, cs, co


def precondition_input(x, t, epsilon, moments):
    mx, my, ci, cs, co = moment_coefficients(t, epsilon, moments)
    return (x - mx) * ci, (mx, my, ci, cs, co)


def reconstruct_drift(x, raw, coeffs):
    mx, my, ci, cs, co = coeffs
    return my + cs * (x - mx) + co * raw


def normalized_target(x, y, coeffs):
    mx, my, ci, cs, co = coeffs
    return (y - my - cs * (x - mx)) / co


# --------------------------------------------------------------------------- #
# direct single-velocity coefficients (experiment 05)
# --------------------------------------------------------------------------- #
def direct_velocity_coefficients(t, epsilon, moments: PairMoments, tiny=1e-8):
    """EDM coefficients for v* = (z1-z0) + sqrt(eps)(1-2t)/(2 sqrt(t(1-t))) xi.

    Derivation (per channel, independent noise xi):
        u   = z0 - mu0                      Var = v0
        d   = (z1 - z0) - (mu1 - mu0)       Var = vd = v0 + v1 - 2 cov
        cvu = Cov(u, d)                     = cov - v0
        z_t - m(t) = u + t d + s xi,        s = sqrt(eps t (1-t))
        v*  - mu_d = d + c xi,              c = sqrt(eps)(1-2t)/(2 sqrt(t(1-t)))
    so
        Var_z  = v0 + t^2 vd + 2 t cvu + s^2
        Cov_zy = cvu + t vd + s c            (s c = eps (1-2t)/2)
        Var_y  = vd + c^2
        c_in = Var_z^{-1/2}, c_skip = Cov_zy / Var_z,
        c_out = sqrt(Var_y - Cov_zy^2 / Var_z)
    """
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    if (t <= 0).any() or (t >= 1).any():
        raise ValueError("t must be in (0,1)")
    dev = t.device
    dt = t.dtype

    def ch(v):
        return v.to(device=dev, dtype=dt).view(1, -1, 1, 1)

    tt = t.view(-1, 1, 1, 1)
    mu0, mu1 = ch(moments.mu0), ch(moments.mu1)
    v0, v1, cov = ch(moments.var0), ch(moments.var1), ch(moments.cov01)

    mx = (1 - tt) * mu0 + tt * mu1          # mean of z_t
    my = mu1 - mu0                          # mean of v*
    vd = (v0 + v1 - 2 * cov).clamp_min(0)
    cvu = cov - v0
    s2 = epsilon * tt * (1 - tt)
    c2 = epsilon * (1 - 2 * tt) ** 2 / (4 * tt * (1 - tt)).clamp_min(tiny)
    s_c = epsilon * (1 - 2 * tt) / 2

    vx = (v0 + tt * tt * vd + 2 * tt * cvu + s2).clamp_min(tiny)
    cyx = cvu + tt * vd + s_c
    vy = (vd + c2).clamp_min(tiny)
    ci = vx.rsqrt()
    cs = cyx / vx
    resvar = (vy - cyx * cyx / vx).clamp_min(tiny)
    co = resvar.sqrt()
    return mx, my, ci, cs, co


def reconstruct_velocity(x, raw, coeffs):
    mx, my, ci, cs, co = coeffs
    return my + cs * (x - mx) + co * raw


def normalized_velocity_target(x, y, coeffs):
    mx, my, ci, cs, co = coeffs
    return (y - my - cs * (x - mx)) / co


# --------------------------------------------------------------------------- #
# score / current decomposition (experiment 10)
# --------------------------------------------------------------------------- #
def pair_stats_per_sample(stats: "StyleStats", pa, pb, dirs, dtype):
    """Per-sample (mu0, mu1, var0, var1, cov) for a batch mixing many style pairs.

    `dirs==1` selects the forward orientation (a -> b) and `dirs==0` the reverse one,
    so the canonical bridge direction and the EDM statistics always agree.
    """
    sm = stats.style_mean.to(device=pa.device, dtype=dtype)
    sv = stats.style_var.to(device=pa.device, dtype=dtype)
    C = sm.shape[1]
    idx_a = pa.view(-1, 1).expand(-1, C)
    idx_b = pb.view(-1, 1).expand(-1, C)
    mu_a, mu_b = sm.gather(0, idx_a), sm.gather(0, idx_b)
    v_a, v_b = sv.gather(0, idx_a), sv.gather(0, idx_b)
    fwd_c = (dirs == 1).view(-1, 1)
    mu0 = torch.where(fwd_c, mu_a, mu_b)
    mu1 = torch.where(fwd_c, mu_b, mu_a)
    v0 = torch.where(fwd_c, v_a, v_b)
    v1 = torch.where(fwd_c, v_b, v_a)
    return mu0, mu1, v0, v1, torch.zeros_like(mu0)


def moment_coefficients_batch(t, epsilon, mu0, mu1, var0, var1, cov01, tiny=1e-8):
    """Per-sample version of `moment_coefficients`.

    All statistics are (B, C) tensors, so one batch can mix bridge samples from
    different style pairs (each sample carries its own pair moments).
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if (t <= 0).any() or (t >= 1).any():
        raise ValueError("t must be in (0,1)")
    a = (1 - t).view(-1, 1)
    b = t.view(-1, 1)
    mu0, mu1 = mu0.to(t.dtype), mu1.to(t.dtype)
    v0, v1, cov = var0.to(t.dtype), var1.to(t.dtype), cov01.to(t.dtype)
    mx = a * mu0 + b * mu1
    my = mu1 - mu0
    vx = (a * a * v0 + b * b * v1 + 2 * a * b * cov + epsilon * a * b).clamp_min(tiny)
    var_d = (v0 + v1 - 2 * cov).clamp_min(0)
    cyx = a * (cov - v0) + b * (v1 - cov) - epsilon * b
    vy = var_d + epsilon * b / a
    ci = vx.rsqrt()
    cs = cyx / vx
    co = (vy - cyx * cyx / vx).clamp_min(tiny).sqrt()
    # broadcast to (B, C, 1, 1)
    return [v.view(v.shape[0], v.shape[1], 1, 1) for v in (mx, my, ci, cs, co)]


def precondition_input_batch(x, t, epsilon, mu0, mu1, var0, var1, cov01):
    mx, my, ci, cs, co = moment_coefficients_batch(t, epsilon, mu0, mu1, var0, var1, cov01)
    return (x - mx) * ci, (mx, my, ci, cs, co)


def reconstruct_drift_batch(x, raw, coeffs):
    mx, my, ci, cs, co = coeffs
    return my + cs * (x - mx) + co * raw


def normalized_target_batch(x, y, coeffs):
    mx, my, ci, cs, co = coeffs
    return (y - my - cs * (x - mx)) / co


def direct_velocity_coefficients_batch(t, epsilon, mu0, mu1, var0, var1, cov01, tiny=1e-8):
    """Per-sample direct-velocity EDM coefficients (see `direct_velocity_coefficients`)."""
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    if (t <= 0).any() or (t >= 1).any():
        raise ValueError("t must be in (0,1)")
    tt = t.view(-1, 1)
    mu0, mu1 = mu0.to(t.dtype), mu1.to(t.dtype)
    v0, v1, cov = var0.to(t.dtype), var1.to(t.dtype), cov01.to(t.dtype)
    mx = (1 - tt) * mu0 + tt * mu1
    my = mu1 - mu0
    vd = (v0 + v1 - 2 * cov).clamp_min(0)
    cvu = cov - v0
    s2 = epsilon * tt * (1 - tt)
    c2 = epsilon * (1 - 2 * tt) ** 2 / (4 * tt * (1 - tt)).clamp_min(tiny)
    s_c = epsilon * (1 - 2 * tt) / 2
    vx = (v0 + tt * tt * vd + 2 * tt * cvu + s2).clamp_min(tiny)
    cyx = cvu + tt * vd + s_c
    vy = (vd + c2).clamp_min(tiny)
    ci = vx.rsqrt()
    cs = cyx / vx
    co = (vy - cyx * cyx / vx).clamp_min(tiny).sqrt()
    return [v.view(v.shape[0], v.shape[1], 1, 1) for v in (mx, my, ci, cs, co)]


def endpoint_coefficients_batch(t, epsilon, mu0, mu1, var0, var1, cov01, tiny=1e-8):
    """Per-sample endpoint-expectation coefficients (see `endpoint_coefficients`)."""
    if (t <= 0).any() or (t >= 1).any():
        raise ValueError("t must be in (0,1)")
    tt = t.view(-1, 1)
    mu0, mu1 = mu0.to(t.dtype), mu1.to(t.dtype)
    v0, v1, cov = var0.to(t.dtype), var1.to(t.dtype), cov01.to(t.dtype)
    mx = (1 - tt) * mu0 + tt * mu1
    my = mu1
    vx = ((1 - tt) ** 2 * v0 + tt * tt * v1 + 2 * tt * (1 - tt) * cov
          + epsilon * tt * (1 - tt)).clamp_min(tiny)
    c1z = (1 - tt) * cov + tt * v1
    ci = vx.rsqrt()
    cs = c1z / vx
    co = (v1 - c1z * c1z / vx).clamp_min(tiny).sqrt()
    return [v.view(v.shape[0], v.shape[1], 1, 1) for v in (mx, my, ci, cs, co)]


def endpoint_coefficients(t, epsilon, moments: PairMoments, tiny=1e-8):
    """Coefficients for predicting the forward endpoint expectation E[z1 | z_t] (exp 14).

    Cov(z1, z_t) = (1-t) cov01 + t v1
    Var(z_t)     = (1-t)^2 v0 + t^2 v1 + 2 t (1-t) cov01 + eps t (1-t)
    E[z1|z_t]    = mu1 + Cov/Var (z_t - m(t)),   Var(z1|z_t) = v1 - Cov^2/Var
    """
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    if (t <= 0).any() or (t >= 1).any():
        raise ValueError("t must be in (0,1)")
    dev, dt = t.device, t.dtype

    def ch(v):
        return v.to(device=dev, dtype=dt).view(1, -1, 1, 1)

    tt = t.view(-1, 1, 1, 1)
    mu0, mu1 = ch(moments.mu0), ch(moments.mu1)
    v0, v1, cov = ch(moments.var0), ch(moments.var1), ch(moments.cov01)
    mx = (1 - tt) * mu0 + tt * mu1
    my = mu1
    vx = ((1 - tt) ** 2 * v0 + tt * tt * v1 + 2 * tt * (1 - tt) * cov
          + epsilon * tt * (1 - tt)).clamp_min(tiny)
    c1z = (1 - tt) * cov + tt * v1
    ci = vx.rsqrt()
    cs = c1z / vx
    resvar = (v1 - c1z * c1z / vx).clamp_min(tiny)
    return mx, my, ci, cs, resvar.sqrt()


def reconstruct_endpoint(x, raw, coeffs):
    mx, my, ci, cs, co = coeffs
    return my + cs * (x - mx) + co * raw


def normalized_endpoint_target(x, y, coeffs):
    mx, my, ci, cs, co = coeffs
    return (y - my - cs * (x - mx)) / co


def score_current_targets(bplus, bminus, epsilon):
    """v* = (b+ - b-)/2 and s* = (b+ + b-)/epsilon, and the exact inverse."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    v = 0.5 * (bplus - bminus)
    s = (bplus + bminus) / epsilon
    return v, s


def drifts_from_score_current(v, s, epsilon):
    return v + 0.5 * epsilon * s, -v + 0.5 * epsilon * s
