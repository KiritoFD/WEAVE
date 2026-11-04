"""Differentiable latent-space style losses used in the audit (no RGB backprop).

Each loss compares a *batch of latents* (generated / optimised) against a
distribution of REAL latents of the target style (reference split).  Targets
are frozen; only the query side carries gradient.

Candidates
----------
swd       sliced-Wasserstein over local patch descriptors (fixed projections)
moment    global per-channel moments (mean/std/abs/q10/q90), standardised
mmd       RBF-MMD over the same global descriptor
subband   Haar subband moments (level 1-2), standardised
cov       channel covariance (Frobenius to the style mean)
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import audit_core as C

KINDS = ("swd", "moment", "mmd", "subband", "cov")


@dataclass
class Targets:
    patch: list[torch.Tensor]      # per style [n_ref, N, 11]
    glob: list[torch.Tensor]       # per style [n_ref, 20]
    subband: list[torch.Tensor]    # per style [n_ref, 64]
    cov: list[torch.Tensor]        # per style [n_ref, 10]
    gmu: torch.Tensor
    gsd: torch.Tensor
    smu: torch.Tensor
    ssd: torch.Tensor

    @staticmethod
    def build(z_ref: torch.Tensor, *, device: str = "cpu") -> "Targets":
        patch = [C.feat_patch(z_ref[s]).float().to(device) for s in range(5)]
        glob = [C.feat_global(z_ref[s]).float().to(device) for s in range(5)]
        sub = [C.feat_subband(z_ref[s]).float().to(device) for s in range(5)]
        cov = [C.feat_chan_cov(z_ref[s]).float().to(device) for s in range(5)]
        allg = torch.cat(glob)
        alls = torch.cat(sub)
        gmu, gsd = C.standardize(allg)
        smu, ssd = C.standardize(alls)
        return Targets(patch, glob, sub, cov, gmu, gsd, smu, ssd)


class LatentStyleLoss:
    def __init__(self, targets: Targets, *, patch_dim: int = 11, n_proj: int = 32,
                 patch_subsample: int = 256, target_images: int = 64):
        self.t = targets
        self.D = patch_dim
        self.P = n_proj
        self.S = patch_subsample
        self.K = target_images
        self._proj_cache: dict[int, torch.Tensor] = {}

    # -- helpers -----------------------------------------------------------
    def _proj(self, seed: int, device) -> torch.Tensor:
        key = seed
        if key not in self._proj_cache:
            g = torch.Generator().manual_seed(seed)
            p = torch.randn(self.D, self.P, generator=g)
            self._proj_cache[key] = (p / p.norm(dim=0, keepdim=True)).to(device)
        return self._proj_cache[key]

    def _target_subset(self, s: int, seed: int, device) -> torch.Tensor:
        n = self.t.glob[s].shape[0]
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(n, generator=g)[: self.K].to(device)
        return idx

    # -- individual losses -------------------------------------------------
    def swd(self, z: torch.Tensor, s: int, *, seed: int = 0) -> torch.Tensor:
        desc = C.feat_patch(z)                              # [B, N, D]
        idx = self._target_subset(s, seed, z.device)
        tgt = self.t.patch[s].to(z.device).index_select(0, idx)
        a = desc.reshape(-1, desc.shape[-1])
        if a.shape[0] > self.S * z.shape[0]:
            g = torch.Generator().manual_seed(seed + 1)
            pick = torch.randperm(a.shape[0], generator=g)[: self.S * z.shape[0]]
            a = a.index_select(0, pick)
        b = tgt.reshape(-1, tgt.shape[-1])
        proj = self._proj(seed, z.device).to(a.dtype)
        return C.swd2_proj(a, b, proj=proj, n_q=48)

    def moment(self, z: torch.Tensor, s: int, *, seed: int = 0) -> torch.Tensor:
        f = C.feat_global(z)
        fs = (f - self.t.gmu.to(z.device)) / self.t.gsd.to(z.device)
        idx = self._target_subset(s, seed, z.device)
        ft = self.t.glob[s].to(z.device).index_select(0, idx)
        ts = (ft - self.t.gmu.to(z.device)) / self.t.gsd.to(z.device)
        return (fs.mean(0) - ts.mean(0)).pow(2).mean() + (fs.std(0) - ts.std(0)).pow(2).mean()

    def mmd(self, z: torch.Tensor, s: int, *, seed: int = 0) -> torch.Tensor:
        f = (C.feat_global(z) - self.t.gmu.to(z.device)) / self.t.gsd.to(z.device)
        idx = self._target_subset(s, seed, z.device)
        ft = self.t.glob[s].to(z.device).index_select(0, idx)
        ft = (ft - self.t.gmu.to(z.device)) / self.t.gsd.to(z.device)
        return C.mmd2(f, ft)

    def subband(self, z: torch.Tensor, s: int, *, seed: int = 0) -> torch.Tensor:
        f = (C.feat_subband(z) - self.t.smu.to(z.device)) / self.t.ssd.to(z.device)
        idx = self._target_subset(s, seed, z.device)
        ft = self.t.subband[s].to(z.device).index_select(0, idx)
        ft = (ft - self.t.smu.to(z.device)) / self.t.ssd.to(z.device)
        return (f.mean(0) - ft.mean(0)).pow(2).mean() + (f.std(0) - ft.std(0)).pow(2).mean()

    def cov(self, z: torch.Tensor, s: int, *, seed: int = 0) -> torch.Tensor:
        f = C.feat_chan_cov(z)
        idx = self._target_subset(s, seed, z.device)
        t = self.t.cov[s].to(z.device).index_select(0, idx)
        return (f.mean(0) - t.mean(0)).pow(2).mean()

    def __call__(self, kind: str, z: torch.Tensor, s: int, *, seed: int = 0) -> torch.Tensor:
        return getattr(self, kind)(z, s, seed=seed)


# ---------------------------------------------------------------------------
# content constraint (latent space only)
# ---------------------------------------------------------------------------
def content_rms(z: torch.Tensor, z0: torch.Tensor) -> torch.Tensor:
    """Per-element RMS distance, [B]."""
    return (z - z0).pow(2).flatten(1).mean(1).sqrt()


def content_ll(z: torch.Tensor, z0: torch.Tensor) -> torch.Tensor:
    from wavelet import dwt2_haar_lowpass
    d = dwt2_haar_lowpass(z, levels=2) - dwt2_haar_lowpass(z0, levels=2)
    return d.pow(2).flatten(1).mean(1).sqrt()


def content_structure(z: torch.Tensor, z0: torch.Tensor) -> torch.Tensor:
    """Self-similarity + low-pass gradient structure difference (latent analogue)."""
    import torch.nn.functional as F
    total = z.new_zeros(())
    for grid in (8, 16):
        pool = z.shape[-1] // grid
        pa = F.avg_pool2d(z, pool).flatten(2)
        pb = F.avg_pool2d(z0, pool).flatten(2)
        na = F.normalize(pa, dim=1)
        nb = F.normalize(pb, dim=1)
        sa = torch.bmm(na.transpose(1, 2), na)
        sb = torch.bmm(nb.transpose(1, 2), nb)
        total = total + (sa - sb).abs().mean()
    return total


__all__ = ["KINDS", "Targets", "LatentStyleLoss", "content_rms", "content_ll", "content_structure"]
