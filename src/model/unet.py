"""C3UNet: the multi-scale conditional latent generator (Q2a / B3 / B4 family).

Re-exported from tools/v6/models_v6.py so every existing checkpoint loads with
bit-identical numerics. Do not fork this file to make a variant: add a clearly
named new class here instead, and register it in src.infer.
"""
from __future__ import annotations

from tools.v6.models_v6 import (  # noqa: F401
    C3UNet,
    ConditionalScoreNet,
    StyleAdaLNBlock,
    StyleChannelPaletteHead,
    StyleKeyResidual,
    sinusoidal_embedding,
)

__all__ = [
    "C3UNet",
    "ConditionalScoreNet",
    "StyleAdaLNBlock",
    "StyleChannelPaletteHead",
    "StyleKeyResidual",
    "sinusoidal_embedding",
]
