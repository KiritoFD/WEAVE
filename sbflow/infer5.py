"""Inference for the five-style model: any source/target style, both directions.

Canonical convention (docs/bridgev10.md 1.1)
-------------------------------------------
For a style pair the canonical order is a = min(i, j), b = max(i, j), and the bridge
is always trained as z0 ~ p_a -> z1 ~ p_b. Therefore:

    a -> b   integrates the probability-flow ODE forward,  t: 0 -> 1
    b -> a   integrates the same ODE backward,             t: 1 -> 0
    i -> i   is an exact identity, no network call

Objectives are handled uniformly: everything is reduced to the probability-flow
velocity v = (b+ - b-)/2 before integration.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .network import BidirDiT
from .preconditioning import (PairMoments, StyleStats, direct_velocity_coefficients,
                              endpoint_coefficients, moment_coefficients,
                              reconstruct_endpoint, reconstruct_velocity)


def load_model5(ckpt_path: str, device: str, which: str = "raw"):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = BidirDiT(**ck["model_kwargs"]).to(device).eval()
    state = ck["model"] if which == "raw" else ck["ema"]
    model.load_state_dict(state)
    stats = StyleStats.from_state_dict(ck["style_stats"])
    stats = StyleStats(*(getattr(stats, k).to(device) for k in
                         ("global_mean", "global_std", "style_mean", "style_var")))
    return ck, model, stats


def canonical(i: int, j: int) -> Tuple[int, int, bool]:
    """Return (a, b, reverse) with a <= b; reverse=True means the request is b -> a."""
    if i == j:
        raise ValueError("identity transfer must not call the network")
    a, b = (i, j) if i < j else (j, i)
    return a, b, i != a


@torch.inference_mode()
def probability_flow_velocity(model, z, t, a, b, stats: StyleStats, eps: float,
                              t_floor: float, objective: str) -> torch.Tensor:
    """v(z, t) for the canonical pair (a, b), t in (0, 1)."""
    tt = t.clamp(t_floor, 1 - t_floor)
    pm = stats.pair(a, b)
    B = z.shape[0]
    sa = torch.full((B,), a, device=z.device, dtype=torch.long)
    sb = torch.full((B,), b, device=z.device, dtype=torch.long)

    if objective == "single_velocity":
        mx, my, ci, cs, co = direct_velocity_coefficients(tt, max(eps, 1e-8), pm)
        raw = model((z - mx) * ci, tt, torch.ones_like(sa), sa, sb).float()
        return reconstruct_velocity(z, raw, (mx, my, ci, cs, co))

    if objective == "score_current":
        mx, my, ci, cs, co = direct_velocity_coefficients(tt, max(eps, 1e-8), pm)
        pv, ps = model((z - mx) * ci, tt, torch.ones_like(sa), sa, sb, return_both=True)
        return reconstruct_velocity(z, pv.float(), (mx, my, ci, cs, co))

    if objective == "endpoint":
        # b+ = (E[z1|zt] - zt)/(1-t), b- = (E[z0|zt] - zt)/t, v = (b+ - b-)/2
        ef = endpoint_coefficients(tt, eps, pm)
        mx, my, ci, cs, co = ef
        raw_f = model((z - mx) * ci, tt, torch.ones_like(sa), sa, sb).float()
        z1_hat = reconstruct_endpoint(z, raw_f, ef)
        er = endpoint_coefficients(1 - tt, eps, pm.reversed())
        mxr, myr, cir, csr, cor = er
        raw_b = model((z - mxr) * cir, 1 - tt, torch.zeros_like(sa), sa, sb).float()
        z0_hat = reconstruct_endpoint(z, raw_b, er)
        bplus = (z1_hat - z) / (1 - tt).clamp_min(t_floor).view(-1, 1, 1, 1)
        bminus = (z0_hat - z) / tt.clamp_min(t_floor).view(-1, 1, 1, 1)
        return 0.5 * (bplus - bminus)

    # bidir / bidir_joint: reconstruct both conditional drifts
    cf = moment_coefficients(tt, eps, pm)
    mxf, myf, cif, csf, cof = cf
    raw_f = model((z - mxf) * cif, tt, torch.ones_like(sa), sa, sb).float()
    bplus = myf + csf * (z - mxf) + cof * raw_f

    pmm = pm.reversed()
    cm = moment_coefficients(1 - tt, eps, pmm)
    mxm, mym, cim, csm, com = cm
    xb = (z - mxm) * cim
    raw_b = model(xb, 1 - tt, torch.zeros_like(sa), sa, sb).float()
    bminus = mym + csm * (z - mxm) + com * raw_b
    return 0.5 * (bplus - bminus)


@torch.inference_mode()
def solve_transfer(model, z, src_style: int, tgt_style: int, stats: StyleStats,
                   eps: float, steps: int = 4, method: str = "heun",
                   t_floor: float = 0.002, objective: str = "bidir",
                   reverse_override: Optional[bool] = None) -> torch.Tensor:
    """Transfer latents from src_style to tgt_style. Identity when they are equal."""
    if src_style == tgt_style:
        return z.clone()
    a, b, reverse = canonical(src_style, tgt_style)
    if reverse_override is not None:
        reverse = reverse_override

    h = 1.0 / steps
    out = z.clone()
    for i in range(steps):
        s0 = i * h
        t0 = torch.full((len(out),), (1 - s0) if reverse else s0, device=out.device)
        k0 = probability_flow_velocity(model, out, t0, a, b, stats, eps, t_floor, objective)
        if reverse:
            k0 = -k0
        if method == "euler":
            out = out + h * k0
        else:
            s1 = (i + 1) * h
            t1 = torch.full((len(out),), (1 - s1) if reverse else s1, device=out.device)
            k1 = probability_flow_velocity(model, out + h * k0, t1, a, b, stats, eps,
                                           t_floor, objective)
            if reverse:
                k1 = -k1
            out = out + h * 0.5 * (k0 + k1)
        if not torch.isfinite(out).all():
            raise FloatingPointError(f"non-finite state at step {i}")
    return out
