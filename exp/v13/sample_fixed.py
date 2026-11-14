"""V13 fixed sampling/editing: generator-seeded noise, sigma decoupled from the
style condition, width read from checkpoint metadata, gradient-capable editing.

Fixes (V13 plan §1): #1 randn w/o generator, #2 sigma follows the (possibly
wrong) style id, #6 edit_bands @no_grad blocks GCL gradients, #8 load_flow
default width=40 vs actual 56.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as Fn

from exp.v11b.bands import band_rms_clip
from exp.v11b.model import WaveletFlow


def load_flow_v13(ckpt: Path, device="cuda") -> tuple[WaveletFlow, dict]:
    """Load a V11b checkpoint; width/args come from the checkpoint metadata."""
    sd = torch.load(ckpt, map_location=device, weights_only=False)
    args = sd.get("args", {})
    width = int(args.get("width", 56))
    model = WaveletFlow(width=width).to(device)
    model.load_state_dict(sd["model"] if "model" in sd else sd)
    model.eval()
    return model, {"args": args, "step": sd.get("step"),
                   "sigmas": sd.get("sigmas")}


def make_noise(shape, device, sigma, generator: torch.Generator | None = None):
    g = generator
    if g is not None and g.device != torch.device(device):
        g = None  # cross-device generator unsupported; fall back to global RNG
    eps = torch.randn(shape, generator=g, device=device, dtype=torch.float32)
    if isinstance(sigma, (int, float)):
        return eps * float(sigma)
    return eps * sigma


@torch.no_grad()
def sample_band_fixed(model: WaveletFlow, which: str, h_base: torch.Tensor,
                      ll: torch.Tensor, edge: torch.Tensor,
                      style_id: int, sigmas: dict, tau0: float = 0.6,
                      K: int = 4, cfg_w: float = 1.0, device="cuda",
                      generator: torch.Generator | None = None,
                      noise_sigma: float | None = None) -> torch.Tensor:
    """Fixed sampling: per-task generator + explicit noise sigma.

    noise_sigma OVERRIDES sigmas[str(style_id)] so that a wrong-style
    condition changes ONLY the conditioning, never the noise scale
    (V13 plan §1 bug #2).
    """
    B = h_base.shape[0]
    band = which
    if noise_sigma is None:
        noise_sigma = sigmas[str(style_id)][band]
    eps = make_noise(h_base.shape, device, noise_sigma, generator)
    ta0 = tau0
    h = ta0 * h_base + (1 - ta0) * eps
    sid = torch.full((B,), style_id, dtype=torch.long, device=device)
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


def edit_band_fixed(model: WaveletFlow, bands_base: dict, style_id: int,
                    sigmas: dict, band: str = "H2", tau0=0.6, K=4, cfg_w=1.0,
                    alpha=0.5, edge_decay=0.5, rms_clip_mult=1.5,
                    device="cuda", generator: torch.Generator | None = None,
                    noise_sigma: float | None = None,
                    grad: bool = False) -> torch.Tensor:
    """Fixed edit. grad=True keeps the graph (for the differentiable GCL stage);
    default grad=False reproduces the V11b behaviour with the bug fixes."""
    which = band
    ll_key, edge_key = ("LL1", "edge1") if band == "H1" else ("LL2", "edge2")
    h_base = bands_base[band]
    ctx = (torch.enable_grad() if grad else torch.no_grad())
    with ctx:
        h_flow = sample_band_fixed(model, which, h_base, bands_base[ll_key],
                                   bands_base[edge_key], style_id, sigmas,
                                   tau0=tau0, K=K, cfg_w=cfg_w, device=device,
                                   generator=generator,
                                   noise_sigma=noise_sigma)
        delta = h_flow - h_base
        e = bands_base[edge_key]
        if e.shape[-2:] != delta.shape[-2:]:
            e = Fn.interpolate(e, size=delta.shape[-2:], mode="bilinear",
                               align_corners=False)
        e = e.float()
        e_n = e / e.detach().amax(dim=(1, 2, 3)).view(-1, 1, 1, 1).clamp_min(1e-8)
        mask = (1 - edge_decay * e_n.clamp(max=1.0)).clamp(0, 1)
        delta = delta * mask
        base_rms = h_base.detach().flatten(1).pow(2).mean(1).sqrt().clamp_min(1e-8)
        radius = rms_clip_mult * base_rms.view(-1, 1, 1, 1)
        delta = band_rms_clip(delta, radius)
        return h_base + alpha * delta
