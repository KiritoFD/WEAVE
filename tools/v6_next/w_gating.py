"""WEAVE W P2 — spatially adaptive gating for latent style editing.

All modes start from an R1 target-margin gradient and apply a spatial/channel gate
before the latent update. The purpose is to avoid the blunt full-latent edit used
in the Oracle/one-step baseline.

Modes:
  full          : uniform R1 gradient (baseline)
  hf            : keep only Haar LH/HL/HH high-frequency components
  topk          : keep only the top-k fraction of spatial positions by gradient magnitude
  content       : tangentially remove the component that increases latent content loss
  style_energy  : gate by target-style high-frequency energy statistics from real refs

All edits are normalized per sample to the same RMS budget, so comparisons are
budget-matched.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tools.legacy_root.wavelet import dwt2_haar, idwt2_haar

from tools.v6_next.w_guided import r1_gradient, rms_normalize

STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]


def haar_band_filter(g: torch.Tensor, band: str = "hf") -> torch.Tensor:
    """Filter a latent tensor to Haar subbands and restore its original RMS."""
    if band == "all":
        return g
    orig_rms = g.flatten(1).pow(2).mean(dim=1, keepdim=True).sqrt().view(-1, 1, 1, 1)
    ll, lh, hl, hh = dwt2_haar(g.float())
    zero = torch.zeros_like(ll)
    if band == "ll":
        filtered = idwt2_haar(ll, zero, zero, zero)
    elif band == "hf":
        filtered = idwt2_haar(zero, lh, hl, hh)
    else:
        raise ValueError(f"unknown band: {band}")
    filt_rms = filtered.flatten(1).pow(2).mean(dim=1, keepdim=True).sqrt().view(-1, 1, 1, 1)
    return (filtered / (filt_rms + 1e-12)) * orig_rms


def topk_spatial_mask(
    g: torch.Tensor,
    frac: float = 0.25,
) -> torch.Tensor:
    B, C, H, W = g.shape
    # 每张图独立计算空间梯度能量
    energy = g.detach().square().sum(dim=1)  # [B,H,W]
    k = max(1, min(H * W, round(frac * H * W)))
    # 每张图独立选 TopK，避免阈值并列导致数量变化
    idx = energy.flatten(1).topk(k, dim=1).indices
    mask = torch.zeros_like(energy.flatten(1))
    mask.scatter_(1, idx, 1.0)
    return g * mask.view(B, 1, H, W)


def topk_spatial_mask_legacy(
    g: torch.Tensor,
    frac: float = 0.25,
) -> torch.Tensor:
    """Old batch-shared TopK implementation. Kept only for fix audit comparisons."""
    b, c, h, w = g.shape
    energy = g.detach().pow(2).mean(dim=(0, 1))  # (H, W) across batch and channels
    k = max(1, int(round(frac * energy.numel())))
    threshold = torch.topk(energy.flatten(), k).values[-1]
    mask = (energy >= threshold).float().view(1, 1, h, w)
    return g * mask


def content_tangential(
    z: torch.Tensor,
    z_src: torch.Tensor,
    g: torch.Tensor,
    canon: nn.Module | None,
    canon_mode: str = "c1_gentle",
) -> torch.Tensor:
    """Remove the R1-gradient component aligned with increasing latent content loss."""
    from tools.v6_next.r2_teacher import compute_content_gradient

    g_c, _ = compute_content_gradient(z, z_src, canon, canon_mode)
    flat_g = g.flatten(1)
    flat_c = g_c.flatten(1)
    dot = (flat_g * flat_c).sum(dim=1, keepdim=True)
    norm_sq = flat_c.pow(2).sum(dim=1, keepdim=True).clamp(min=1e-8)
    coeff = torch.clamp(dot / norm_sq, min=0.0)
    return (flat_g - coeff * flat_c).view_as(g)


def style_energy_gate(
    z: torch.Tensor,
    target_id: torch.Tensor,
    style_energy_maps: dict[int, torch.Tensor],
    source_weight: float = 0.5,
) -> torch.Tensor:
    """Build a (B,1,32,32) gate from target-style and source high-frequency energy.

    `style_energy_maps[t]` is a normalized 32x32 map of where the target style
    tends to carry high-frequency energy in real paintings. The source energy map
    is used to avoid amplifying positions that already have very high energy.
    """
    b = z.shape[0]
    gates = []
    ll, lh, hl, hh = dwt2_haar(z.float())
    energy = lh.pow(2) + hl.pow(2) + hh.pow(2)
    energy = F.adaptive_avg_pool2d(energy, (32, 32))
    for i in range(b):
        target_map = style_energy_maps[int(target_id[i])]
        source_map = energy[i]
        source_map = (source_map - source_map.min()) / (
            source_map.max() - source_map.min() + 1e-8
        )
        gate = target_map / (target_map.max() + 1e-8)
        gate = gate * (1.0 - source_weight * source_map)
        gates.append(gate)
    gate = torch.stack(gates, dim=0)
    # gates may broadcast to (B,1,32,32) or (B,1,4,32,32); reduce any channel-like
    # dimensions to a single spatial gate before upsampling.
    gate = gate.reshape(b, -1, gate.shape[-2], gate.shape[-1])
    if gate.shape[1] > 1:
        gate = gate.mean(dim=1, keepdim=True)
    gate = (gate - gate.amin(dim=(2, 3), keepdim=True)) / (
        gate.amax(dim=(2, 3), keepdim=True) - gate.amin(dim=(2, 3), keepdim=True) + 1e-8
    )
    # Broadcast across latent channels and upsample to the latent grid.
    return F.interpolate(gate, size=(z.shape[2], z.shape[3]), mode="bilinear", align_corners=False)


def apply_gated_edit(
    z: torch.Tensor,
    z_src: torch.Tensor,
    target_id: torch.Tensor,
    *,
    critic: torch.nn.Module,
    canon: nn.Module | None,
    canon_mode: str = "c1_gentle",
    mode: str = "full",
    steps: int = 4,
    per_step_rms: float = 0.08,
    topk_frac: float = 0.25,
    style_energy_maps: dict[int, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Apply a gated multi-step R1 edit. Returns a detached latent."""
    with torch.inference_mode(False), torch.enable_grad():
        z_cur = z.detach().clone()
        tid = target_id.detach().clone()
        for _ in range(max(1, steps)):
            g = r1_gradient(z_cur, tid, critic)
            if mode == "hf":
                g = haar_band_filter(g, "hf")
            elif mode == "topk":
                g = topk_spatial_mask(g, topk_frac)
            elif mode == "content":
                g = content_tangential(z_cur, z_src, g, canon, canon_mode)
            elif mode == "style_energy":
                if style_energy_maps is None:
                    raise ValueError("style_energy mode requires style_energy_maps")
                gate = style_energy_gate(z_cur, tid, style_energy_maps)
                g = g * gate
            elif mode != "full":
                raise ValueError(f"unknown gating mode: {mode}")
            g = rms_normalize(g)
            z_cur = (z_cur + per_step_rms * g).detach()
    return z_cur.detach()


def compute_style_energy_maps(
    encoder_vae: nn.Module,
    style_to_paths: dict[str, list[str | Path]],
    device: str = "cuda",
) -> dict[int, torch.Tensor]:
    """Compute normalized target-style HF energy maps from real reference latents."""
    maps = {}
    for sid, style in enumerate(STYLES):
        paths = style_to_paths.get(style, [])
        energies = []
        for i in range(0, len(paths), 8):
            ims = [Image.open(p).convert("RGB").resize((512, 512)) for p in paths[i:i+8]]
            x = torch.stack([
                torch.from_numpy(np.array(im, dtype=np.float32) / 255.0).permute(2, 0, 1)
                for im in ims
            ]).to(device)
            x = x * 2.0 - 1.0
            with torch.inference_mode():
                z = encoder_vae.encode(x).latent_dist.mean.float() * encoder_vae.config.scaling_factor
            _, lh, hl, hh = dwt2_haar(z)
            energy = F.adaptive_avg_pool2d(lh.pow(2) + hl.pow(2) + hh.pow(2), (32, 32)).mean(dim=1, keepdim=True)
            energies.append(energy.cpu())
        energy = torch.cat(energies, dim=0).mean(dim=0, keepdim=True)
        energy = (energy - energy.min()) / (energy.max() - energy.min() + 1e-8)
        maps[sid] = energy.to(device)
    return maps
