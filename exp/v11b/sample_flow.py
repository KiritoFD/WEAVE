"""Sampling / editing with the trained wavelet flow (V11-B P3+).

SDEdit-style warm start on wavelet detail bands:
  h_tau0 = tau0*h_base + (1-tau0)*eps, integrate tau0 -> 1 in K Euler steps.
Then H_out = H_base + alpha * M_edge * (H_flow - H_base), per-band RMS clip.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from .bands import band_rms_clip, sobel_mag
from .model import WaveletFlow


def load_flow(ckpt: Path, device="cuda", width: int = 40) -> WaveletFlow:
    model = WaveletFlow(width=width).to(device)
    sd = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(sd["model"] if "model" in sd else sd)
    model.eval()
    return model


@torch.no_grad()
def sample_band(model: WaveletFlow, which: str, h_base: torch.Tensor,
                ll: torch.Tensor, edge: torch.Tensor,
                style_id: int, sigmas: dict, tau0: float = 0.6,
                K: int = 4, cfg_w: float = 1.0, device="cuda",
                unconditional_style: int = 0):
    """Integrate one band from tau0 to 1. All inputs on device fp32."""
    B = h_base.shape[0]
    band = "H1" if which == "H1" else "H2"
    sig = torch.tensor([sigmas[str(style_id)][band]] * B,
                       device=device, dtype=torch.float32
                       ).view(-1, 1, 1, 1)
    eps = torch.randn_like(h_base) * sig
    ta0 = tau0
    h = ta0 * h_base + (1 - ta0) * eps
    sid = torch.full((B,), style_id, dtype=torch.long, device=device)
    # Euler steps tau0 -> 1
    taus = torch.linspace(ta0, 1.0, K + 1, device=device)
    for i in range(K):
        t_cur, t_nxt = taus[i], taus[i + 1]
        dt = t_nxt - t_cur
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=str(device) == "cuda"):
            v_c = model.forward_band(which, h, ll, edge, sid,
                                     t_cur.expand(B),
                                     unconditional=False).float()
            if cfg_w != 1.0:
                v_u = model.forward_band(which, h, ll, edge, sid,
                                         t_cur.expand(B),
                                         unconditional=True).float()
                v = v_u + cfg_w * (v_c - v_u)
            else:
                v = v_c
        h = h + dt * v
    return h


@torch.no_grad()
def edit_bands(model, bands_base: dict, style_id: int, sigmas: dict,
               band: str = "H1", tau0=0.6, K=4, cfg_w=1.0,
               alpha=0.5, edge_decay=0.5, rms_clip_mult=1.5,
               device="cuda", generator=None):
    """bands_base: dict with LL2,H2,H1,LL1,edge1,edge2 at FULL 512 res
    (or crop res). Returns edited band tensor (same shape as base band).

    edge mask: M = 1 - edge_decay * normalized_sobel(LL_match)
    clipped to [0,1]; protects strong contours.
    """
    which = "H1" if band == "H1" else "H2"
    ll_key, edge_key = ("LL1", "edge1") if band == "H1" else ("LL2", "edge2")
    h_base = bands_base[band]
    h_flow = sample_band(model, which, h_base, bands_base[ll_key],
                         bands_base[edge_key], style_id, sigmas,
                         tau0=tau0, K=K, cfg_w=cfg_w, device=device)
    delta = h_flow - h_base
    # edge-protection mask at band resolution
    e = bands_base[edge_key]
    if e.shape[-2:] != delta.shape[-2:]:
        e = F.interpolate(e, size=delta.shape[-2:], mode="bilinear",
                          align_corners=False)
    e = e.float()
    e_n = e / e.amax(dim=(1, 2, 3)).view(-1, 1, 1, 1).clamp_min(1e-8)
    mask = (1 - edge_decay * e_n.clamp(max=1.0)).clamp(0, 1)
    delta = delta * mask
    # RMS clip relative to base band energy
    base_rms = h_base.flatten(1).pow(2).mean(1).sqrt().clamp_min(1e-8)
    radius = rms_clip_mult * base_rms.view(-1, 1, 1, 1)
    delta = band_rms_clip(delta, radius)
    return h_base + alpha * delta
