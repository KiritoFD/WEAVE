"""P3: content-safe constrained latent optimization driven by the V10 KDE score.

Pre-registered (plan §6):
  g_safe      = g_s - max(0, <g_s, grad L_c>)/ (||grad L_c||^2+eps) * grad L_c
  z_{k+1}     = Proj_C( z_k + eta_k * normalize(g_safe) )
  budget C    : cost_tot(z, z0, z_src) <= eps (N2 hard budget, rho RMS projection)
                + content gate  C_LL(z, z0) <= 1.2 * C_LL(z_n2, z0)
  acceptance  : budget feasible AND content gate AND per-sample score improves
  backtrack   : eta in {1.0, 0.5, 0.25}
"""
from __future__ import annotations

import torch
import torch.nn.functional as Fn

from exp.v8.guidance import project_rms

EPS = 0.10
RHO = 0.15
ETAS = (1.0, 0.5, 0.25)
CONTENT_GATE = 1.2
LAMBDA_STRUCT = 0.5


def haar_ll(z: torch.Tensor) -> torch.Tensor:
    return Fn.avg_pool2d(z, 4)


def c_ll(z: torch.Tensor, z0: torch.Tensor) -> torch.Tensor:
    """Per-sample Haar-LL structural distance to the B4 output, (B,)."""
    return (haar_ll(z) - haar_ll(z0)).square().flatten(1).mean(1)


def selfsim_gram(z: torch.Tensor) -> torch.Tensor:
    """Per-sample self-similarity Gram of the 16x16 LL cells, (B,256,256)."""
    ll = haar_ll(z)                                    # (B,4,16,16)
    b = z.shape[0]
    f = ll.reshape(b, 4, -1).transpose(1, 2)           # (B,256,4)
    f = Fn.normalize(f, dim=2)
    return f @ f.transpose(1, 2)


def content_loss(z: torch.Tensor, z0: torch.Tensor) -> torch.Tensor:
    """L_c = C_LL + 0.5 * selfsim distance (per-sample (B,) -> mean)."""
    ss = (selfsim_gram(z) - selfsim_gram(z0)).square().flatten(1).mean(1)
    return c_ll(z, z0) + LAMBDA_STRUCT * ss


def safe_direction(g_s: torch.Tensor, z: torch.Tensor, z0: torch.Tensor) -> torch.Tensor:
    """First-order content projection of the style gradient (per sample)."""
    with torch.enable_grad():
        zl = z.detach().clone().requires_grad_(True)
        Lc = content_loss(zl, z0)
        gc = torch.autograd.grad(Lc.sum(), zl)[0].detach()
    g = g_s
    dot = (g.flatten(1) * gc.flatten(1)).sum(1, keepdim=True)
    denom = gc.flatten(1).square().sum(1, keepdim=True).clamp_min(1e-12)
    coeff = dot.clamp_min(0) / denom                    # (B,1)
    g_safe = g - coeff.view(-1, 1, 1, 1) * gc
    flat = g_safe.flatten(1)
    nrm = flat.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return (flat / nrm).view_as(z), gc.detach()


def cost_tot(z: torch.Tensor, z0: torch.Tensor, z_src: torch.Tensor) -> torch.Tensor:
    num = (z - z0).square().flatten(1).mean(1)
    den = (z0 - z_src).square().flatten(1).mean(1).clamp_min(1e-8)
    return num / den


@torch.no_grad()
def region_mask_v10(bank, stats, hpsi, z: torch.Tensor, tid: torch.Tensor,
                    variant: str = "B", frac: float = 0.25) -> torch.Tensor:
    """Top-frac patches by style headroom (-logp under the target bank).

    Returns the soft mask (B,1,64,64) in the same layout as v9's region_mask.
    """
    from .features import patchify, q_feats, f_feats
    from .kde import VARIANTS
    space, view, cond = VARIANTS[variant]
    b = z.shape[0]
    p = patchify(z)
    q = ((q_feats(p) - stats["q_mean"]) / stats["q_std"]).float()
    f = ((f_feats(p) - stats["f_mean"]) / stats["f_std"]).float()
    if space == "res":
        v = ((f - hpsi(q)) - stats["r_mean"]) / stats["r_std"]
    else:
        v = f
    room = torch.zeros(b, 64, device=z.device)
    for i in range(b):
        lp = bank.logp_per_patch(q[i * 64:(i + 1) * 64], v[i * 64:(i + 1) * 64],
                                 int(tid[i]), cond, space, view)
        room[i] = -lp
    k = max(1, int(round(frac * 64)))
    thr = room.topk(k, dim=1).values[:, -1:]
    m = (room >= thr).float().view(b, 1, 8, 8)
    m = Fn.interpolate(m, size=(64, 64), mode="bilinear", align_corners=False)
    return (m > 0.5).float() * m


@torch.no_grad()
def sample_scores(bank, stats, hpsi, z: torch.Tensor, tid: torch.Tensor,
                  variant: str = "B") -> torch.Tensor:
    """Per-sample truncated log-density sum (B,) — acceptance criterion."""
    from .features import patchify, q_feats, f_feats
    from .kde import VARIANTS
    space, view, cond = VARIANTS[variant]
    b = z.shape[0]
    p = patchify(z)
    q = ((q_feats(p) - stats["q_mean"]) / stats["q_std"]).float()
    f = ((f_feats(p) - stats["f_mean"]) / stats["f_std"]).float()
    if space == "res":
        v = ((f - hpsi(q)) - stats["r_mean"]) / stats["r_std"]
    else:
        v = f
    out = torch.zeros(b, device=z.device)
    for i in range(b):
        lp = bank.logp_per_patch(q[i * 64:(i + 1) * 64], v[i * 64:(i + 1) * 64],
                                 int(tid[i]), cond, space, view)
        out[i] = lp.sum()
    return out


def direction_batch(bank, feat_fn, z: torch.Tensor, tid: torch.Tensor,
                    variant: str = "B") -> torch.Tensor:
    """Per-sample normalized KDE directions for a mixed-tid batch."""
    b = z.shape[0]
    g = torch.zeros_like(z)
    with torch.enable_grad():
        for i in range(b):
            g[i:i + 1] = bank.direction(z[i:i + 1], int(tid[i]), variant, feat_fn,
                                        samples_per_call=1)
    return g


def optimize_batch(bank, stats, hpsi, z_n2: torch.Tensor, z0: torch.Tensor,
                   z_src: torch.Tensor, sid: torch.Tensor, tid: torch.Tensor,
                   variant: str = "B", feat_fn=None, mask=None,
                   k_steps: int = 2, eps: float = EPS, rho: float = RHO,
                   etas=ETAS, gate: float = CONTENT_GATE):
    """Constrained score-ascent; returns (z_out, infos).

    Acceptance per candidate: cost_tot <= eps AND C_LL <= gate*base AND score up.
    """
    b = z_n2.shape[0]
    z = z_n2.clone()
    base_ll = c_ll(z_n2, z0)
    gate_ll = (gate * base_ll).clamp_min(1e-8)
    infos = [{"steps": 0, "accepted": 0, "over": False} for _ in range(b)]
    for _ in range(k_steps):
        g_s = direction_batch(bank, feat_fn, z, tid, variant)
        if mask is not None:
            g_s = g_s * mask
        g_safe, _gc = safe_direction(g_s, z, z0)
        cur = sample_scores(bank, stats, hpsi, z, tid, variant)
        best_z = z.clone()
        best_s = cur.clone()
        acc = torch.zeros(b, dtype=torch.bool, device=z.device)
        for eta in etas:
            cand = project_rms(z + eta * g_safe, z0, rho)
            feas = (cost_tot(cand, z0, z_src) <= eps) & (c_ll(cand, z0) <= gate_ll)
            if not feas.any():
                continue
            s = sample_scores(bank, stats, hpsi, cand[feas], tid[feas], variant)
            bi = feas.nonzero(as_tuple=True)[0]
            better = s > best_s[bi]
            if better.any():
                sel = bi[better]
                best_z[sel] = cand[sel]
                best_s[sel] = s[better]
                acc[sel] = True
        for i in range(b):
            if acc[i]:
                z[i] = best_z[i]
                infos[i]["steps"] += 1
                infos[i]["accepted"] += 1
            else:
                infos[i]["over"] = infos[i]["steps"] == 0
        if not acc.any():
            break
    # final safety: over-budget -> roll back to z_n2 (per-sample budget respected)
    bad = cost_tot(z, z0, z_src) > eps
    if bad.any():
        z[bad] = z_n2[bad]
        for i in bad.nonzero(as_tuple=True)[0].tolist():
            infos[i]["over"] = True
    return z, infos

