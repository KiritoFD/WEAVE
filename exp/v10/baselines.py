"""Baseline style directions for the V10 experiments (plan §5-A).

  r1        : grad_z of the R1 critic target margin (plain gradient, no precond)
  ld3       : -grad_z of the v9 prototype local style energy
  v6_global : -grad_z of per-channel global moment matching to the style pool

All return per-sample unit-normalized directions (B,4,64,64); zero rows stay zero.
"""
from __future__ import annotations

import torch

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from exp.v8.guidance import target_margin  # noqa: E402
from exp.v9_local.local_style import LocalStyleEnergy  # noqa: E402

V9_BANK = REPO / "runs/v9_local_4070/bank/style_bank.pt"


def normalize_dir(g: torch.Tensor) -> torch.Tensor:
    flat = g.flatten(1)
    nrm = flat.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return (flat / nrm).view_as(g)


def r1_direction(critic, z: torch.Tensor, tid: torch.Tensor) -> torch.Tensor:
    zl = z.detach().clone().requires_grad_(True)
    m = target_margin(critic, zl.float(), tid)
    g = torch.autograd.grad(m.sum(), zl)[0]
    return normalize_dir(g.detach())


class Ld3Direction:
    def __init__(self, device="cuda", bank_path=V9_BANK):
        self.lse = LocalStyleEnergy(str(bank_path), device)
        self.device = device

    def __call__(self, z: torch.Tensor, tid: torch.Tensor) -> torch.Tensor:
        zl = z.detach().clone().requires_grad_(True)
        loss = self.lse.loss(zl, tid)
        g = torch.autograd.grad(loss, zl)[0]
        return normalize_dir(-g.detach())


class V6GlobalDirection:
    """Per-channel (mean, std) moment matching to the train-split pool stats."""

    def __init__(self, device="cuda"):
        from .splits import POOL_DIR, STYLES
        mus, sds = [], []
        for si, s in enumerate(STYLES):
            d = torch.load(POOL_DIR / f"{si:02d}_{s}.pt",
                           map_location="cpu", weights_only=False)
            z = d["latents"].float()
            mus.append(z.mean(dim=(0, 2, 3)))
            sds.append(z.std(dim=(0, 2, 3)))
        self.mu = torch.stack(mus).to(device)     # (5,4)
        self.sd = torch.stack(sds).to(device)     # (5,4)
        self.device = device

    def __call__(self, z: torch.Tensor, tid: torch.Tensor) -> torch.Tensor:
        zl = z.detach().clone().requires_grad_(True)
        b = zl.shape[0]
        m = zl.mean(dim=(2, 3))                    # (B,4)
        s = zl.std(dim=(2, 3), unbiased=False).clamp_min(1e-6)
        mu_t = self.mu[tid]
        sd_t = self.sd[tid]
        loss = ((m - mu_t).square().sum(1) / 4.0
                + ((s - sd_t) / sd_t.clamp_min(1e-4)).square().sum(1) / 4.0).sum()
        g = torch.autograd.grad(loss, zl)[0]
        return normalize_dir(-g.detach())
