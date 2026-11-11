"""Two small style readers over FROZEN A0 latents (4x64x64), 64-d output.

Local Style Reader (Deep Sets / ALADIN-style local statistics)
    r_phi(z) = rho( mean_p phi(Patch_p(z)) )
  - Patch_p: overlapping 3x3 patches via F.unfold -> (B, 36, N)
  - phi: shared MLP applied to EACH patch independently (permutation-equivariant)
  - mean pooling over patches -> permutation-invariant, position-insensitive
  - rho: small head -> 64-d, L2-normalised
  This learns WHICH local statistics carry style instead of assuming all channel
  covariances do (the fixed-readout failure mode).

Plain shallow CNN reader (CONTROL)
  - ordinary conv stack over the whole latent + global average pool
  - comparable parameter budget
  - answers: is any trainable nonlinear reader enough, or does the local-set
    inductive bias specifically help?

No VAE, no decoder, no CSD/DINO gradient. Input is a cached latent tensor.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

PATCH = 3
C_LAT = 4
D_PATCH = C_LAT * PATCH * PATCH      # 36
D_OUT = 64


class LocalStyleReader(nn.Module):
    """Permutation-invariant reader over local patch statistics."""

    def __init__(self, patch: int = PATCH, width: int = 96, out: int = D_OUT,
                 pool: str = "mean"):
        super().__init__()
        self.patch = patch
        self.pool = pool
        d_in = C_LAT * patch * patch
        self.phi = nn.Sequential(
            nn.Linear(d_in, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
        )
        rho_in = width * (2 if pool == "mean+max" else 1)
        self.rho = nn.Sequential(
            nn.Linear(rho_in, width // 2), nn.SiLU(),
            nn.Linear(width // 2, out),
        )
        self.norm = nn.LayerNorm(width)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        U = F.unfold(z.float(), kernel_size=self.patch)      # (B, 36, N)
        U = U.transpose(1, 2).contiguous()                   # (B, N, 36)
        h = self.phi(U)                                      # (B, N, W)
        h = self.norm(h)
        if self.pool == "mean":
            h = h.mean(dim=1)                                # (B, W)
        elif self.pool == "mean+max":
            h = torch.cat([h.mean(dim=1), h.amax(dim=1)], dim=1)
        else:
            raise ValueError(self.pool)
        return F.normalize(self.rho(h), dim=1)


class ShallowCNNReader(nn.Module):
    """Plain small CNN reader, parameter-matched CONTROL."""

    def __init__(self, width: int = 40, out: int = D_OUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(C_LAT, width, 3, padding=1), nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, out),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.net(z.float())
        h = h.mean(dim=(2, 3))                               # global avg pool
        return F.normalize(self.head(h), dim=1)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def build(name: str, **kw) -> nn.Module:
    if name == "local":
        return LocalStyleReader(**kw)
    if name == "cnn":
        return ShallowCNNReader(**kw)
    raise ValueError(f"unknown reader {name}")


if __name__ == "__main__":
    z = torch.randn(8, 4, 64, 64)
    for name, kw in [("local", {}), ("cnn", {})]:
        m = build(name, **kw)
        y = m(z)
        print(f"{name:<8} params={n_params(m):>7,}  out={tuple(y.shape)}  "
              f"norm={float(y.norm(dim=1).mean()):.4f}")
