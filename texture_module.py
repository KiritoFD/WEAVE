"""Route D: multi-scale texture refiner.

The VAE reconstruction probe (`tools/vae_recon_probe.py`) returned `latent_ok`
(HF energy ratio ~0.996), so latent HF can carry brushwork detail and a texture
refiner is worth opening.

Contract:
  * Input: backbone hidden feature h (B, dim, H, W) at latent resolution.
  * Output: per-subband velocity residuals for LH / HL (HH is disabled upstream).
  * Spatial placement follows CONTENT edges (derived from h), so the module adds
    high-frequency texture where the content already has structure instead of
    uniformly sharpening the latent.
  * The last layer is zero-initialised: enabling the module is a no-op at init,
    and the residual only grows once gradients move it (same trick as the
    reference conditioner gate).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class MultiscaleTextureRefiner(nn.Module):
    """Parallel dilated convolutions (rates 1/2/4) fused into an edge-guided residual."""

    def __init__(self, dim: int, latent_channels: int, hidden: int = 32,
                 dilations: tuple[int, ...] = (1, 2, 4)) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(dim, hidden, kernel_size=1),
            nn.SiLU(),
        )
        self.branches = nn.ModuleList(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=d, dilation=d)
            for d in dilations
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden * len(dilations), hidden, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
        )
        self.out = nn.Conv2d(hidden, latent_channels * 2, kernel_size=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    @staticmethod
    def _edge_map(h: torch.Tensor) -> torch.Tensor:
        """Content-edge gate from the backbone feature, normalised to mean ~1."""
        gx = torch.zeros_like(h)
        gy = torch.zeros_like(h)
        gx[..., 1:] = (h[..., 1:] - h[..., :-1]).abs()
        gy[..., :, 1:] = (h[..., :, 1:] - h[..., :, :-1]).abs()
        edge = (gx + gy).mean(dim=1, keepdim=True)
        edge = edge / (edge.mean(dim=(2, 3), keepdim=True) + 1e-6)
        return edge.clamp(max=3.0)

    def forward(self, h: torch.Tensor, edge_map: torch.Tensor | None = None,
                style_tokens: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        del style_tokens  # style enters globally through the conditioner path
        f = self.proj(h)
        f = self.fuse(torch.cat([b(f) for b in self.branches], dim=1))
        gate = self._edge_map(h) if edge_map is None else edge_map
        f = f * (1.0 + 0.5 * gate)
        if f.shape[-2:] != h.shape[-2:]:
            f = F.interpolate(f, size=h.shape[-2:], mode="bilinear", align_corners=False)
        delta_lh, delta_hl = self.out(f).chunk(2, dim=1)
        return {"lh": delta_lh, "hl": delta_hl}
