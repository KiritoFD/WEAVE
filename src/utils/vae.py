"""VAE / TensorRT loading and decoding.

Kept separate from model definitions: this is deployment plumbing (which engine,
which batch), not architecture. The fixed batch matters: the decoder engine is
compiled for a specific batch, so a short tail batch is padded rather than
falling back to a slower PyTorch path.

Windows fallback: when the TensorRT path is unavailable (no tensorrt /
trt_vae_decoder / engines, e.g. local dev machines), decode() falls back to the
diffusers AutoencoderKL path ``vae.decode(z / scaling_factor)`` which is the
same math the TRT engine was built from. Metrics impact is limited to tiny
fp16 kernel-ordering differences.
"""
from __future__ import annotations

from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
DEFAULT_BATCH = 8


def _diffusers_vae_dir() -> Path:
    """Locate an sd-vae-ft-ema snapshot without going through tools.v5.route_c."""
    import paths
    candidates = [
        paths.eval_cache_root("hf") / "hub" / "models--stabilityai--sd-vae-ft-ema" / "snapshots",
        paths.eval_cache_root("hf") / "modelscope" / "stabilityai_sd-vae-ft-ema" / "models" / "stabilityai--sd-vae-ft-ema" / "snapshots",
    ]
    for c in candidates:
        if c.exists():
            for snap in sorted(c.glob("*")):
                if (snap / "config.json").exists():
                    return snap
    raise FileNotFoundError("SD1.5 VAE local snapshot not found")


def load_vae(device: str = "cuda"):
    try:
        from tools.v5.route_c import load_vae as _load
        return _load(device)
    except Exception:
        from diffusers import AutoencoderKL
        return AutoencoderKL.from_pretrained(
            str(_diffusers_vae_dir()), torch_dtype=torch.float16,
            local_files_only=True,
        ).to(device).eval()


def load_trt(vae, device: str = "cuda"):
    try:
        from tools.v5.route_c import load_trt as _load
        return _load(vae, device)
    except Exception:
        return None


def _vae_decode(vae, z: torch.Tensor) -> torch.Tensor:
    """diffusers decode of VAE-scaled latents -> [0,1] float, matching decode_trt."""
    sf = float(vae.config.scaling_factor)
    dtype = next(vae.parameters()).dtype
    with torch.inference_mode():
        x = vae.decode((z / sf).to(dtype)).sample
    return x.mul(0.5).add(0.5).clamp(0.0, 1.0)


S10_DECODER = REPO / "deliver" / "checkpoints" / "runs" / "vae_decoder_prune" / "s10_p5_lowlr" / "decoder.pt"


def load_s10_decoder(device: str = "cuda"):
    """S10 pruned decoder (10.6M) + its bundled post_quant_conv.

    Protocol (2026-11): all evaluation decoding uses S10, NOT the original
    sd-vae-ft-ema decoder. Input contract matches AutoencoderKL.decode():
    caller passes latents / scaling_factor; output is [-1,1] float.
    """
    import torch.nn as nn
    from tools.vae_suite.decoder_prune import build_decoder, StudentDecoder
    ck = torch.load(S10_DECODER, map_location="cpu", weights_only=False)
    pqc = nn.Conv2d(4, 4, 1)
    pqc.load_state_dict(ck["post_quant_conv"])
    dec = build_decoder(ck["meta"]["widths_forward"])
    dec.load_state_dict(ck["state_dict"])
    return StudentDecoder(pqc, dec).to(device).eval().half()


def s10_scaling_factor() -> float:
    """SD1.5 VAE scaling factor (0.18215); S10 shares the A0 latent space."""
    return 0.18215


def decode_s10(model, z: torch.Tensor, batch: int = DEFAULT_BATCH,
               dtype=torch.float16) -> torch.Tensor:
    """Decode VAE-scaled latents via S10 -> [0,1] float, same contract as decode()."""
    n = z.shape[0]
    if n == 0:
        side = z.shape[-1] * 8
        return z.new_zeros((0, 3, side, side))
    sf = s10_scaling_factor()
    outs = []
    with torch.inference_mode():
        for s in range(0, n, batch):
            x = model(((z[s:s + batch] / sf).to(dtype)))
            outs.append(x.float().mul(0.5).add(0.5).clamp(0.0, 1.0).cpu())
    return torch.cat(outs, 0)


def decode(trt, vae, z: torch.Tensor, batch: int = DEFAULT_BATCH) -> torch.Tensor:
    """Decode with the compiled engine, padding a short tail batch."""
    n = z.shape[0]
    if n == 0:
        side = z.shape[-1] * 8
        return z.new_zeros((0, 3, side, side))
    if trt is None:
        return torch.cat([_vae_decode(vae, z[s:s + batch]) for s in range(0, n, batch)], 0)
    from tools.v5.route_c import decode_trt
    if n == batch:
        return decode_trt(trt, vae, z, batch=batch)
    pad = torch.zeros(batch - n, *z.shape[1:], device=z.device, dtype=z.dtype)
    return decode_trt(trt, vae, torch.cat([z, pad], 0), batch=batch)[:n]


__all__ = ["DEFAULT_BATCH", "decode", "load_trt", "load_vae"]
