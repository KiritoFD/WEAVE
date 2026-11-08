"""True online alpha-DSBM (group E, experiments 24-31).

The original algorithm is not "Brownian bridge on fixed independent endpoints".
After pretraining it samples NEW endpoints with the current model and rebuilds the
bridge coupling:

    x0 ~ p_i,  x1 ~ p_j
    x1_hat ~ P_theta^{->}(.|x0)      (forward, stochastic)
    x0_hat ~ P_theta^{<-}(.|x1)      (backward, stochastic)
    L_online = 1/2 [ L_+(x0_hat, x1) + L_-(x0, x1_hat) ]

with stop_gradient on every generated endpoint.

The endpoint sampler must be a genuine SDE, never an ODE with terminal noise:

    forward :  dz = b+(z,t) dt + sqrt(eps) dW          t: 0 -> 1
    reverse :  dz = -b-(z,t) ds + sqrt(eps) dW         s: 0 -> 1, t = 1 - s

This follows because the reverse-time drift of dz = b+ dt + sqrt(eps) dW is exactly
b- = b+ - eps * grad log p_t.
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

import torch

from .bridge import bridge_sample
from .preconditioning import (PairMoments, StyleStats, moment_coefficients,
                              normalized_target, precondition_input)


# --------------------------------------------------------------------------- #
# drift reconstruction
# --------------------------------------------------------------------------- #
def both_drifts(model, z, t, a, b, stats: StyleStats, eps: float, t_floor: float):
    """Return (b+, b-) for the canonical pair (a, b) at physical time t."""
    tt = t.clamp(t_floor, 1 - t_floor)
    pm = stats.pair(a, b)
    B = z.shape[0]
    sa = torch.full((B,), a, device=z.device, dtype=torch.long)
    sb = torch.full((B,), b, device=z.device, dtype=torch.long)

    cf = moment_coefficients(tt, eps, pm)
    mxf, myf, cif, csf, cof = cf
    raw_f = model((z - mxf) * cif, tt, torch.ones_like(sa), sa, sb).float()
    bplus = myf + csf * (z - mxf) + cof * raw_f

    cm = moment_coefficients(1 - tt, eps, pm.reversed())
    mxm, mym, cim, csm, com = cm
    raw_b = model((z - mxm) * cim, 1 - tt, torch.zeros_like(sa), sa, sb).float()
    bminus = mym + csm * (z - mxm) + com * raw_b
    return bplus, bminus


# --------------------------------------------------------------------------- #
# SDE samplers
# --------------------------------------------------------------------------- #
@torch.no_grad()
def sde_forward(model, z, a, b, stats, eps, steps: int, t_floor: float,
                gen: torch.Generator) -> torch.Tensor:
    """Sample from P_theta^{->}(.|z) with the forward SDE, t: 0 -> 1."""
    h = 1.0 / steps
    out = z.clone()
    for i in range(steps):
        t = torch.full((len(out),), i * h, device=out.device)
        bp, _ = both_drifts(model, out, t, a, b, stats, eps, t_floor)
        noise = torch.randn(out.shape, device=out.device, generator=gen, dtype=out.dtype)
        out = out + h * bp + (eps * h) ** 0.5 * noise
    return out


@torch.no_grad()
def sde_reverse(model, z, a, b, stats, eps, steps: int, t_floor: float,
                gen: torch.Generator) -> torch.Tensor:
    """Sample from P_theta^{<-}(.|z) with the reverse SDE, s: 0 -> 1 with t = 1 - s."""
    h = 1.0 / steps
    out = z.clone()
    for i in range(steps):
        t = torch.full((len(out),), 1.0 - i * h, device=out.device)
        _, bm = both_drifts(model, out, t, a, b, stats, eps, t_floor)
        noise = torch.randn(out.shape, device=out.device, generator=gen, dtype=out.dtype)
        out = out - h * bm + (eps * h) ** 0.5 * noise
    return out


@torch.no_grad()
def ode_forward(model, z, a, b, stats, eps, steps: int, t_floor: float,
                gen: Optional[torch.Generator] = None) -> torch.Tensor:
    """Deterministic ODE endpoints; the exp 29 reflow control.

    `gen` is accepted and ignored so this is a drop-in for the SDE samplers.
    """
    h = 1.0 / steps
    out = z.clone()
    for i in range(steps):
        t = torch.full((len(out),), i * h, device=out.device)
        bp, bm = both_drifts(model, out, t, a, b, stats, eps, t_floor)
        out = out + h * 0.5 * (bp - bm)
    return out


# --------------------------------------------------------------------------- #
# bridge loss on a given coupling
# --------------------------------------------------------------------------- #
def bridge_loss(model, z0, z1, a, b, stats, eps, min_t, gen, t_floor=0.002) -> torch.Tensor:
    """Standard dual-drift bridge MSE for one coupling (half forward, half reverse)."""
    B = z0.shape[0]
    half = B // 2
    t = torch.rand(B, device=z0.device, generator=gen, dtype=z0.dtype) * (1 - 2 * min_t) + min_t
    xi = torch.randn(z0.shape, device=z0.device, generator=gen, dtype=z0.dtype)
    xt, bplus, bminus = bridge_sample(z0, z1, t, eps, xi)
    pm = stats.pair(a, b)
    xf, cf = precondition_input(xt[:half], t[:half], eps, pm)
    xb, cb = precondition_input(xt[half:], 1 - t[half:], eps, pm.reversed())
    yf = normalized_target(xt[:half], bplus[:half], cf)
    yb = normalized_target(xt[half:], bminus[half:], cb)
    tau = torch.cat([t[:half], 1 - t[half:]], 0)
    d = torch.cat([torch.ones(half, device=z0.device, dtype=torch.long),
                   torch.zeros(B - half, device=z0.device, dtype=torch.long)])
    sa = torch.full((B,), a, device=z0.device, dtype=torch.long)
    sb = torch.full((B,), b, device=z0.device, dtype=torch.long)
    pred = model(torch.cat([xf, xb], 0), tau, d, sa, sb).float()
    return torch.nn.functional.mse_loss(pred, torch.cat([yf, yb], 0).float())


# --------------------------------------------------------------------------- #
# replay buffer (exp 30)
# --------------------------------------------------------------------------- #
class EndpointReplayBuffer:
    def __init__(self, capacity: int = 4096, channels: int = 32, size: int = 16, seed: int = 0):
        self.capacity = capacity
        self.buf: Dict[Tuple[int, int, str], deque] = {}
        self.channels, self.size = channels, size
        self.gen = torch.Generator().manual_seed(seed)

    def push(self, a: int, b: int, side: str, z: torch.Tensor) -> None:
        """Store individual latents (not whole batches) so sampling is per-sample."""
        key = (a, b, side)
        q = self.buf.setdefault(key, deque(maxlen=self.capacity))
        zc = z.detach().cpu().half()
        for i in range(zc.shape[0]):
            q.append(zc[i].clone())

    def sample(self, a: int, b: int, side: str, k: int, device: str,
               gen: Optional[torch.Generator] = None) -> Optional[torch.Tensor]:
        q = self.buf.get((a, b, side))
        if not q or len(q) < k:
            return None
        idx = torch.randint(0, len(q), (k,), generator=self.gen)
        return torch.stack([q[i] for i in idx.tolist()], 0).float().to(device)

    def __len__(self) -> int:
        return sum(len(v) for v in self.buf.values())


# --------------------------------------------------------------------------- #
# dynamic cross covariance (exp 31)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def estimate_cross_cov(z_a: torch.Tensor, z_b: torch.Tensor, ema: Optional[torch.Tensor] = None,
                       decay: float = 0.9) -> torch.Tensor:
    """Per-channel Cov(z_a, z_b) for the current coupling."""
    a = z_a.double()
    b = z_b.double()
    ma = a.mean((0, 2, 3), keepdim=True)
    mb = b.mean((0, 2, 3), keepdim=True)
    cov = ((a - ma) * (b - mb)).mean((0, 2, 3)).float()
    return cov if ema is None else (decay * ema + (1 - decay) * cov)


def online_loss(model, teacher, z0, z1, a, b, stats: StyleStats, eps: float, min_t: float,
                gen: torch.Generator, cfg: dict, replay: Optional[EndpointReplayBuffer] = None,
                cross: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
    """The doc's L_online = 1/2 [L+(x0_hat, x1) + L_-(x0, x1_hat)]."""
    steps = cfg.get("online_steps", 4)
    t_floor = cfg.get("t_floor", 0.002)
    mode = cfg.get("online_mode", "sde")          # sde | ode
    direction = cfg.get("online_direction", "both")  # both | forward | reverse
    mix = float(cfg.get("online_mix", 0.0))       # fraction of independent endpoints
    use_replay = bool(cfg.get("online_replay", False))

    st = stats
    if cross is not None:
        pm = StyleStats(st.global_mean, st.global_std, st.style_mean, st.style_var)
        pm_orig = pm.pair
        def pair_with_cross(i, j):
            return pm_orig(i, j, cross=cross)
        pm.pair = pair_with_cross  # type: ignore[assignment]
        st = pm

    sampler = sde_forward if mode == "sde" else ode_forward

    if use_replay and replay is not None:
        x1_hat = replay.sample(a, b, "fwd", len(z0), z0.device, gen)
        x0_hat = replay.sample(a, b, "rev", len(z1), z1.device, gen)
        if x1_hat is None:
            x1_hat = sampler(teacher, z0, a, b, st, eps, steps, t_floor, gen)
            replay.push(a, b, "fwd", x1_hat)
        if x0_hat is None:
            x0_hat = sde_reverse(teacher, z1, a, b, st, eps, steps, t_floor, gen)
            replay.push(a, b, "rev", x0_hat)
    else:
        x1_hat = sampler(teacher, z0, a, b, st, eps, steps, t_floor, gen)
        x0_hat = sde_reverse(teacher, z1, a, b, st, eps, steps, t_floor, gen)

    # optional anchoring on the original independent coupling (exp 28)
    if mix > 0:
        n = len(z0)
        k = int(round(n * mix))
        if k > 0:
            x1_hat = x1_hat.clone()
            x0_hat = x0_hat.clone()
            x1_hat[:k] = z1[:k]
            x0_hat[:k] = z0[:k]

    l_fwd = bridge_loss(model, x0_hat.detach(), z1, a, b, st, eps, min_t, gen, t_floor)
    l_rev = bridge_loss(model, z0, x1_hat.detach(), a, b, st, eps, min_t, gen, t_floor)

    if direction == "forward":
        loss = l_fwd
    elif direction == "reverse":
        loss = l_rev
    else:
        loss = 0.5 * (l_fwd + l_rev)

    info = {"online_fwd": float(l_fwd), "online_rev": float(l_rev),
            "endpoint_shift_fwd": float((x1_hat - z1).flatten(1).norm(dim=1).mean()),
            "endpoint_shift_rev": float((x0_hat - z0).flatten(1).norm(dim=1).mean()),
            "_x1_hat": x1_hat, "_x0_hat": x0_hat}
    return loss, info
