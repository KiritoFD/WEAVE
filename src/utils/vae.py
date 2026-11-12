"""VAE / TensorRT loading and decoding.

Kept separate from model definitions: this is deployment plumbing (which engine,
which batch), not architecture. The fixed batch matters: the decoder engine is
compiled for a specific batch, so a short tail batch is padded rather than
falling back to a slower PyTorch path.
"""
from __future__ import annotations

from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
DEFAULT_BATCH = 8


def load_vae(device: str = "cuda"):
    from tools.v5.route_c import load_vae as _load
    return _load(device)


def load_trt(vae, device: str = "cuda"):
    from tools.v5.route_c import load_trt as _load
    return _load(vae, device)


def decode(trt, vae, z: torch.Tensor, batch: int = DEFAULT_BATCH) -> torch.Tensor:
    """Decode with the compiled engine, padding a short tail batch."""
    from tools.v5.route_c import decode_trt
    n = z.shape[0]
    if n == 0:
        side = z.shape[-1] * 8
        return z.new_zeros((0, 3, side, side))
    if n == batch:
        return decode_trt(trt, vae, z, batch=batch)
    pad = torch.zeros(batch - n, *z.shape[1:], device=z.device, dtype=z.dtype)
    return decode_trt(trt, vae, torch.cat([z, pad], 0), batch=batch)[:n]


__all__ = ["DEFAULT_BATCH", "decode", "load_trt", "load_vae"]
