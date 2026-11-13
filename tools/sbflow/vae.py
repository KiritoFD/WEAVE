"""SD-VAE helpers for the five-style pipeline (replaces the DCAE path).

The project's latent space is `stabilityai/sd-vae-ft-ema` at `4x64x64` with
`scaling_factor = 0.18215`, because the existing pseudo-GT caches, the V8/V9
experiments and the CSD protocol were all built around it. DCAE was only ever a
blog-reproduction detail and is not used here.

Decoding prefers the prebuilt TensorRT fp16 decoder when present (fixed batch 8,
returns [0,1]); otherwise it falls back to the diffusers AutoencoderKL.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import torch

REPO = Path(__file__).resolve().parents[2]


def load_sdvae(device: str, precision: str = "fp16") -> Tuple[object, Optional[object]]:
    """Return (vae, trt_decoder_or_None)."""
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "tools"))
    sys.path.insert(0, str(REPO / "tools" / "trt"))
    from tools.v5.route_c import load_trt, load_vae

    vae = load_vae(device)
    trt = None
    try:
        engines = REPO / "runs" / "trt" / "engines"
        if engines.exists() and any(engines.glob("*.engine")):
            trt = load_trt(vae, device)
            print("[sdvae] using TensorRT decoder", flush=True)
        else:
            print("[sdvae] no TRT engine, using diffusers decode", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[sdvae] TRT unavailable ({type(e).__name__}), using diffusers", flush=True)
        trt = None
    return vae, trt


@torch.inference_mode()
def encode_images(vae, x: torch.Tensor, scaling_factor: float) -> torch.Tensor:
    """x in [-1,1] (B,3,512,512) -> scaled latents (B,4,64,64) float32."""
    return vae.encode(x).latent_dist.mean.float() * scaling_factor


@torch.inference_mode()
def decode_latents(vae, trt, z_scaled: torch.Tensor, batch: int = 8) -> torch.Tensor:
    """Scaled latents -> images in [-1,1] float32."""
    out = []
    for s in range(0, z_scaled.shape[0], batch):
        chunk = z_scaled[s:s + batch]
        if trt is not None:
            n = chunk.shape[0]
            padded = chunk.half()
            if n < 8:
                padded = torch.cat([padded, padded[-1:].expand(8 - n, -1, -1, -1)], 0)
            im = trt.decode(padded)[:n].float().clamp(0.0, 1.0)
        else:
            im = vae.decode(chunk.half()).sample.float()
            im = ((im + 1) / 2).clamp(0.0, 1.0)
        out.append(im)
    return torch.cat(out, 0) * 2 - 1
