"""Candidate directions for LD arms.

g_R = P_N2C . grad_z m_R1(z,t)      (N2C SPDHead rank-0 = diagonal preconditioner)
g_L = -grad_z L_loc(z,t)            (local style energy descent)
g_M = 0.5*R1norm + 0.5*Localnorm     (mixed, RMS-normalised)

Content risk h = grad_z C_LL(z, z0): Haar-LL structural distance to the B4 output.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from exp.v8.guidance import SPDHead, normalize, target_margin
from .local_style import LocalStyleEnergy


def haar_ll(z: torch.Tensor) -> torch.Tensor:
    """(B,4,64,64) -> LL band 16x16 (avg 4x4 blocks) per channel."""
    return F.avg_pool2d(z, 4)


def content_loss(z: torch.Tensor, z0: torch.Tensor) -> torch.Tensor:
    """C_LL: mean squared Haar-LL difference vs z0 (per-sample scalar)."""
    return (haar_ll(z) - haar_ll(z0)).square().flatten(1).mean(1)


class DirectionComputer:
    def __init__(self, critic, head: SPDHead, lse: LocalStyleEnergy, device):
        self.critic = critic
        self.head = head
        self.lse = lse
        self.device = device

    def _grad(self, scalar_fn, z: torch.Tensor):
        zr = z.detach().clone().requires_grad_(True)
        g = torch.autograd.grad(scalar_fn(zr).sum(), zr)[0]
        return g.detach()

    def directions(self, z: torch.Tensor, z0: torch.Tensor,
                   tid: torch.Tensor, mode: str):
        """Return dict of candidate direction tensors (already normalised)."""
        out = {}
        need_r1 = mode in ("r1", "mix", "all")
        need_loc = mode in ("local", "mix", "all")
        if need_r1:
            g = self._grad(lambda zz: target_margin(self.critic, zz.float(), tid), z)
            with torch.no_grad():
                diag, u = self.head(z0, z.detach(), tid, g)
            dg = diag * g
            ug = (u * g.unsqueeze(1)).sum(2)
            gr = dg + (u * ug.unsqueeze(2)).sum(1)
            out["r1"] = normalize(gr)
        if need_loc:
            gl = self._grad(lambda zz: self.lse.loss(zz, tid), z)
            out["local"] = normalize(-gl)
        if mode == "mix":
            out["mix"] = normalize(0.5 * out["r1"] + 0.5 * out["local"])
        return out

    @torch.no_grad()
    def content_risk(self, z: torch.Tensor, z0: torch.Tensor) -> torch.Tensor:
        """Per-patch content risk proxy: local LL energy vs z0, (B,64)."""
        b = z.shape[0]
        a = haar_ll(z)      # (B,4,16,16)
        r = haar_ll(z0)
        d = (a - r).square().mean(1)          # (B,16,16)
        # upsample risk map to 8x8 latent cells -> each cell covers 2x2 of the 16x16
        d = d.reshape(b, 8, 2, 8, 2).mean(dim=(2, 4))  # (B,8,8)
        return d.flatten(1)                   # (B,64)

    @torch.no_grad()
    def region_mask(self, z: torch.Tensor, z0: torch.Tensor, tid: torch.Tensor,
                    frac: float = 0.25) -> torch.Tensor:
        """Top-frac patches by expected style gain / content risk -> (B,1,64,64) soft mask."""
        gain = self.lse.patch_scores(z, tid)          # (B,64)
        risk = self.content_risk(z, z0)               # (B,64)
        score = gain / risk.clamp_min(1e-6)
        b = z.shape[0]
        k = max(1, int(round(frac * 64)))
        thr = score.topk(k, dim=1).values[:, -1:]     # (B,1)
        m = (score >= thr).float().view(b, 1, 8, 8)
        m = F.interpolate(m, size=(64, 64), mode="bilinear",
                          align_corners=False)
        m = (m > 0.5).float() * m                     # keep soft edges, hard core
        return m
