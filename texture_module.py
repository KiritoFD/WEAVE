"""Route D: multi-scale texture module — interface + config slot only.

Per the review doc (路线D), a heavier texture-generation module should only be
built AFTER Route A/B confirm the supervision fix, and only if the VAE
reconstruction probe shows latent HH can still carry brushwork. This file
therefore provides ONLY the interface and a minimal identity implementation,
gated behind ``model.texture_refiner_enabled = false`` (default off).

Design contract for the future real implementation:
  * Input: hidden feature h (B, dim, H/2, W/2), content edge map, style tokens.
  * Output: per-subband velocity residual added to v_lh / v_hl / (v_hh).
  * Spatial placement must be driven by CONTENT edges, style must enter only
    as global tokens (same rule as the reference conditioner).
"""
from __future__ import annotations

import torch
from torch import nn


class MultiscaleTextureRefiner(nn.Module):
    """Minimal placeholder refiner.

    Currently an identity-scale gated residual: projects h, produces a
    zero-initialised per-subband delta. Exists so configs/checkpoints/tests can
    reference the interface; the heavy multi-scale logic is intentionally NOT
    implemented yet (see module docstring).
    """

    def __init__(self, dim: int, latent_channels: int, hidden: int = 32) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(dim, hidden, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden, latent_channels * 2, kernel_size=1),
        )
        # Zero-init last layer: enabling the module is a no-op at init.
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, h: torch.Tensor, edge_map: torch.Tensor | None = None, style_tokens: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        del edge_map, style_tokens  # placeholder: reserved for the real implementation
        delta_lh, delta_hl = self.proj(h).chunk(2, dim=1)
        return {"lh": delta_lh, "hl": delta_hl}
