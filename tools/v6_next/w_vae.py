"""WEAVE W — canonical A0 Encoder + S10 Decoder pairing (frozen latent space).

This module exists because of a measured failure: feeding sd-vae-ft-ema latents
into S10 collapses DINO-C to 0.22. S10 was trained on the A0 Encoder's latent
distribution (4090 `build_variant('s10')`), so the ONLY correct pairing is:

    z = A0.encode(x).latent_dist.mean * sf          # scaled, generator domain
    x_hat = StudentDecoder(A0.post_quant_conv, S10_dec)(z / sf)   # [-1,1]
    x01 = (x_hat.clamp(-1,1) / 2 + 0.5)

Verified on 3060 (2026-11-11):
    DINO-C  standard sd-vae : 0.9657
    DINO-C  A0 enc + A0 dec : 0.9666
    DINO-C  A0 enc + S10    : 0.9387   <- correct
    DINO-C  sdvae latent + S10 (WRONG): 0.2185

A0 has the SAME architecture as sd-vae-ft-ema (only encoder-side weights were
fine-tuned on art data; conv_in diff ~2e-4), so A0 latents remain compatible with
the existing Q2a/B4/S1-S4 generators -- the "frozen latent space" premise holds.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL

from tools.v5.route_c import find_vae_dir
from tools.vae_suite.decoder_prune import StudentDecoder, build_decoder

ROOT = Path(__file__).resolve().parents[2]

A0_ENCODER_CANDIDATES = [
    ROOT / "runs/a0/a0_encoder_only.pt",
    ROOT / "deliver/checkpoints/runs/a0/a0_encoder_only.pt",
]
S10_DECODER_CANDIDATES = [
    ROOT / "deliver/checkpoints/runs/vae_decoder_prune/s10_p5_lowlr/decoder.pt",
    ROOT / "runs/vae_decoder_prune/s10_p5_lowlr/decoder.pt",
]


def _resolve(cands, label):
    for p in cands:
        if p.exists():
            return p
    raise FileNotFoundError(f"{label} not found; tried {[str(c) for c in cands]}")


def build_a0_encoder(device: str = "cuda") -> AutoencoderKL:
    """Frozen A0 encoder-side VAE (encoder + quant_conv + post_quant_conv).

    A0 is architecture-identical to sd-vae-ft-ema, so we clone the local sd-vae
    snapshot and overwrite only the encoder-side weights. The decoder stays the
    standard one (unused when S10 is attached) and is never trained.
    """
    ck = torch.load(_resolve(A0_ENCODER_CANDIDATES, "A0 encoder"),
                    map_location="cpu", weights_only=False)
    vae = AutoencoderKL.from_pretrained(
        str(find_vae_dir()), torch_dtype=torch.float32).to(device).eval()
    missing, unexpected = vae.load_state_dict(ck["state_dict"], strict=False)
    if unexpected:
        raise ValueError(f"unexpected A0 keys: {unexpected[:3]}")
    if any(not k.startswith("decoder.") for k in missing):
        raise ValueError(f"A0 missing non-decoder keys: {missing[:3]}")
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


def build_s10_decoder(vae_a0: AutoencoderKL, device: str = "cuda") -> nn.Module:
    """Frozen S10 decoder, using A0's post_quant_conv (NOT its checkpoint's copy).

    Do NOT use ck["post_quant_conv"] from the S10 checkpoint: that is a backup and
    pairs with the wrong latent distribution. 4090's reference implementation
    uses ``StudentDecoder(vae_a0.post_quant_conv, dec)``.
    """
    ck = torch.load(_resolve(S10_DECODER_CANDIDATES, "S10 decoder"),
                    map_location="cpu", weights_only=False)
    dec = build_decoder(ck["meta"]["widths_forward"])
    dec.load_state_dict(ck["state_dict"])
    model = StudentDecoder(vae_a0.post_quant_conv, dec).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def encode_a0(vae_a0: AutoencoderKL, x_pm1: torch.Tensor) -> torch.Tensor:
    """x_pm1: (B,3,H,W) in [-1,1] -> scaled latent (B,4,64,64), no grad."""
    sf = float(vae_a0.config.scaling_factor)
    with torch.inference_mode():
        return (vae_a0.encode(x_pm1).latent_dist.mean * sf).float()


def decode_s10(student_dec: nn.Module, z_scaled: torch.Tensor,
               sf: float) -> torch.Tensor:
    """Scaled latent -> [0,1] RGB, DIFFERENTIABLE in z (for image-domain calib).

    StudentDecoder mirrors AutoencoderKL.decode(): feed z/sf so post_quant_conv
    sees the unscaled distribution it was trained with.
    """
    out = student_dec(z_scaled.float() / sf)
    return (out.clamp(-1.0, 1.0) / 2.0 + 0.5)


def raw_to_pm1(x01: torch.Tensor) -> torch.Tensor:
    return x01 * 2.0 - 1.0
