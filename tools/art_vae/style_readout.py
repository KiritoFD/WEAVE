"""Fixed, parameter-free style readout operator for the native 4x64x64 latent.

    r(z) = [ mu(z), log sigma(z), vec(G_0(z)), vec(G_1(z)) ]

where G_k is the centered covariance of 3x3 unfolded patches at two spatial
scales (64x64 and 32x32).  No learned parameters, no random projection.

Design notes
------------
* mu / log-sigma are global channel statistics (4 + 4 dims).
* G_k uses the UPPER TRIANGLE of the (36x36) patch covariance (666 dims each),
  so the readout is 4+4+666+666 = 1340 dims. Vectorising the full matrix would
  double-count symmetric entries.
* "Scale normalisation by statistic type": each group is divided by its own
  per-sample RMS before concatenation, then the whole vector is L2-normalised.
  This (a) stops the 666-dim covariance groups from dominating the 8-dim
  statistics and (b) makes r invariant to a global rescaling of z, which is
  exactly the cheating route the plan warns about (shrinking the latent to
  reduce the contrastive loss).
* log(sigma+eps) instead of raw sigma keeps the operator numerically stable
  when the posterior collapses a channel.

The gradient of r with respect to z is well defined everywhere, so the
contrastive loss can shape the encoder directly.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

EPS = 1e-5
PATCH = 3
C_LATENT = 4
D_PATCH = C_LATENT * PATCH * PATCH          # 36
D_TRI = D_PATCH * (D_PATCH + 1) // 2        # 666
D_OUT = 4 + 4 + 2 * D_TRI                   # 1340


def _tri_indices(device):
    return torch.triu_indices(D_PATCH, D_PATCH, device=device)


def _patch_cov(z: torch.Tensor, patch: int = PATCH, eps: float = EPS):
    """Centered covariance of unfolded patches, upper-triangle vector.

    z: (B, C, H, W) -> (B, D_TRI)
    """
    B, C, H, W = z.shape
    U = F.unfold(z, kernel_size=patch)                  # (B, C*p*p, N)
    U = U - U.mean(dim=2, keepdim=True)
    N = U.shape[2]
    G = torch.bmm(U, U.transpose(1, 2)) / max(N - 1, 1)
    i, j = _tri_indices(z.device)
    return G[:, i, j]


def style_readout_raw(z: torch.Tensor, eps: float = EPS) -> list[torch.Tensor]:
    """Return the four STATISTIC GROUPS before normalisation.

    Each element is (B, d_g). Kept as a list so group-wise scaling and
    diagnostics can use the natural block structure.
    """
    if z.dim() != 4 or z.shape[1] != C_LATENT:
        raise ValueError(f"expected (B,4,H,W) latent, got {tuple(z.shape)}")
    z = z.float()
    mu = z.mean(dim=(2, 3))                              # (B,4)
    sigma = z.std(dim=(2, 3), unbiased=False)            # (B,4)
    log_sigma = torch.log(sigma + eps)

    z1 = F.avg_pool2d(z, kernel_size=2, stride=2)        # (B,4,32,32)
    g0 = _patch_cov(z, eps=eps)
    g1 = _patch_cov(z1, eps=eps)
    return [mu, log_sigma, g0, g1]


def style_readout(z: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Normalised readout q = Normalize( group-RMS-normalised [mu,logσ,G0,G1] ).

    Returns (B, D_OUT), L2-normalised per sample.
    """
    groups = style_readout_raw(z, eps=eps)
    out = []
    for g in groups:
        rms = g.pow(2).mean(dim=1, keepdim=True).sqrt() + eps
        out.append(g / rms)
    r = torch.cat(out, dim=1)
    return F.normalize(r, dim=1, eps=eps)


def readout_scale_diagnostics(z: torch.Tensor) -> dict:
    """Monitoring helper: group norms, latent RMS, and per-dim std of q.

    Used to detect the failure modes the plan lists: overall latent scaling,
    common-mode drift, variance collapse.
    """
    with torch.no_grad():
        groups = style_readout_raw(z)
        q = style_readout(z)
        return {
            "z_rms": round(float(z.float().pow(2).mean().sqrt()), 5),
            "z_mean": round(float(z.float().mean()), 5),
            "group_rms": [round(float(g.pow(2).mean(dim=1).sqrt().mean()), 5)
                          for g in groups],
            "q_per_dim_std_mean": round(float(q.std(dim=0).mean()), 6),
            "q_per_dim_std_min": round(float(q.std(dim=0).min()), 6),
            "q_norm_mean": round(float(q.norm(dim=1).mean()), 5),
        }
