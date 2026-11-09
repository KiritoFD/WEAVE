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
                              moment_coefficients_batch, normalized_target,
                              normalized_target_batch, pair_stats_per_sample,
                              precondition_input, precondition_input_batch,
                              reconstruct_drift, reconstruct_drift_batch)


# --------------------------------------------------------------------------- #
# mixed-pair drifts (all ten style pairs in one batch)
# --------------------------------------------------------------------------- #
def mixed_drifts(model, z, t, pa, pb, stats: StyleStats, eps: float, t_floor: float):
    """(b+, b-) for a batch whose samples carry their own canonical pair."""
    tt = t.clamp(t_floor, 1 - t_floor)
    B = z.shape[0]
    fwd = torch.ones(B, device=z.device, dtype=torch.long)
    rev = torch.zeros(B, device=z.device, dtype=torch.long)
    mu0, mu1, v0, v1, cov = pair_stats_per_sample(stats, pa, pb, fwd, z.dtype)
    xin, cf = precondition_input_batch(z, tt, eps, mu0, mu1, v0, v1, cov)
    bplus = reconstruct_drift_batch(z, model(xin, tt, fwd, pa.long(), pb.long()).float(), cf)
    mu0r, mu1r, v0r, v1r, covr = pair_stats_per_sample(stats, pa, pb, rev, z.dtype)
    xinb, cb = precondition_input_batch(z, 1 - tt, eps, mu0r, mu1r, v0r, v1r, covr)
    bminus = reconstruct_drift_batch(z, model(xinb, 1 - tt, rev, pa.long(), pb.long()).float(), cb)
    return bplus, bminus


@torch.no_grad()
def mixed_sde_forward(model, z, pa, pb, stats, eps, steps: int, t_floor: float,
                      gen: torch.Generator) -> torch.Tensor:
    h = 1.0 / steps
    out = z.clone()
    for i in range(steps):
        t = torch.full((len(out),), i * h, device=out.device)
        bp, _ = mixed_drifts(model, out, t, pa, pb, stats, eps, t_floor)
        noise = torch.randn(out.shape, device=out.device, generator=gen, dtype=out.dtype)
        out = out + h * bp + (eps * h) ** 0.5 * noise
    return out


@torch.no_grad()
def mixed_sde_reverse(model, z, pa, pb, stats, eps, steps: int, t_floor: float,
                      gen: torch.Generator) -> torch.Tensor:
    h = 1.0 / steps
    out = z.clone()
    for i in range(steps):
        t = torch.full((len(out),), 1.0 - i * h, device=out.device)
        _, bm = mixed_drifts(model, out, t, pa, pb, stats, eps, t_floor)
        noise = torch.randn(out.shape, device=out.device, generator=gen, dtype=out.dtype)
        out = out - h * bm + (eps * h) ** 0.5 * noise
    return out


@torch.no_grad()
def mixed_ode_forward(model, z, pa, pb, stats, eps, steps: int, t_floor: float,
                      gen: Optional[torch.Generator] = None) -> torch.Tensor:
    """Deterministic mixed-pair ODE endpoints (the reflow control, exp 32).

    Deterministic for BOTH directions: no noise term is ever added here, so the ODE
    control can never silently fall back to a stochastic sampler.
    """
    h = 1.0 / steps
    out = z.clone()
    for i in range(steps):
        t = torch.full((len(out),), i * h, device=out.device)
        bp, bm = mixed_drifts(model, out, t, pa, pb, stats, eps, t_floor)
        out = out + h * 0.5 * (bp - bm)
    return out


@torch.no_grad()
def mixed_ode_reverse(model, z, pa, pb, stats, eps, steps: int, t_floor: float,
                      gen: Optional[torch.Generator] = None) -> torch.Tensor:
    """Deterministic reverse-time mixed ODE, t: 1 -> 0."""
    h = 1.0 / steps
    out = z.clone()
    for i in range(steps):
        t = torch.full((len(out),), 1.0 - i * h, device=out.device)
        bp, bm = mixed_drifts(model, out, t, pa, pb, stats, eps, t_floor)
        out = out - h * 0.5 * (bp - bm)
    return out


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
def bridge_loss(model, z0, z1, a, b, stats, eps, min_t, gen, t_floor=0.002,
                mode: str = "both") -> torch.Tensor:
    """Dual-drift bridge MSE for one coupling.

    `mode="both"` splits the batch (half forward, half reverse) as a training objective.
    `mode="fwd"` / `mode="rev"` supervise ONLY one drift type for every sample, which is
    what alpha-DSBM Algorithm 1 requires for its two endpoint pairings (V11 fix).
    """
    B = z0.shape[0]
    t = torch.rand(B, device=z0.device, generator=gen, dtype=z0.dtype) * (1 - 2 * min_t) + min_t
    xi = torch.randn(z0.shape, device=z0.device, generator=gen, dtype=z0.dtype)
    xt, bplus, bminus = bridge_sample(z0, z1, t, eps, xi)
    pm = stats.pair(a, b)

    if mode == "both":
        half = B // 2
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

    fwd = (mode == "fwd")
    p_use = pm if fwd else pm.reversed()
    tau = t if fwd else 1 - t
    xin, cf = precondition_input(xt, tau, eps, p_use)
    y = normalized_target(xt, bplus if fwd else bminus, cf)
    d = torch.full((B,), 1 if fwd else 0, device=z0.device, dtype=torch.long)
    sa = torch.full((B,), a, device=z0.device, dtype=torch.long)
    sb = torch.full((B,), b, device=z0.device, dtype=torch.long)
    pred = model(xin, tau, d, sa, sb).float()
    return torch.nn.functional.mse_loss(pred, y.float())


def mixed_bridge_loss(model, z0, z1, pa, pb, dirs, stats, eps, min_t, gen,
                      t_floor=0.002) -> torch.Tensor:
    """Bridge MSE for a batch mixing many style pairs (per-sample statistics)."""
    B = z0.shape[0]
    t = torch.rand(B, device=z0.device, generator=gen, dtype=z0.dtype) * (1 - 2 * min_t) + min_t
    xi = torch.randn(z0.shape, device=z0.device, generator=gen, dtype=z0.dtype)
    xt, bplus, bminus = bridge_sample(z0, z1, t, eps, xi)
    fwd = (dirs == 1)
    fwd_c = fwd.view(-1, 1, 1, 1)
    mu0, mu1, v0, v1, cov = pair_stats_per_sample(stats, pa, pb, dirs, z0.dtype)
    tau = torch.where(fwd, t, 1 - t)
    xin, cf = precondition_input_batch(xt, tau, eps, mu0, mu1, v0, v1, cov)
    y = normalized_target_batch(xt, torch.where(fwd_c, bplus, bminus), cf)
    pred = model(xin, tau, dirs.long(), pa.long(), pb.long()).float()
    return torch.nn.functional.mse_loss(pred, y.float())


def mixed_bridge_loss_fwd(model, z0, z1, pa, pb, stats, eps, min_t, gen,
                          t_floor=0.002) -> torch.Tensor:
    """Supervise ONLY the forward drift b+ for every sample in the batch."""
    ones = torch.ones(len(z0), dtype=torch.long, device=z0.device)
    return mixed_bridge_loss(model, z0, z1, pa, pb, ones, stats, eps, min_t, gen, t_floor)


def mixed_bridge_loss_rev(model, z0, z1, pa, pb, stats, eps, min_t, gen,
                          t_floor=0.002) -> torch.Tensor:
    """Supervise ONLY the reverse drift b- for every sample in the batch."""
    zeros = torch.zeros(len(z0), dtype=torch.long, device=z0.device)
    return mixed_bridge_loss(model, z0, z1, pa, pb, zeros, stats, eps, min_t, gen, t_floor)


def online_loss_mixed(model, teacher, z0, z1, pa, pb, dirs, stats: StyleStats, eps: float,
                      min_t: float, gen: torch.Generator, cfg: dict,
                      replay: Optional[EndpointReplayBuffer] = None) -> Tuple[torch.Tensor, dict]:
    """Online alpha-DSBM with all ten style pairs inside one batch.

    Same construction as `online_loss` but every sample keeps its own canonical pair, so
    the online update no longer has to fall back to a single-pair batch.
    """
    steps = cfg.get("online_steps", 4)
    t_floor = cfg.get("t_floor", 0.002)
    mode = cfg.get("online_mode", "sde")
    direction = cfg.get("online_direction", "both")
    mix = float(cfg.get("online_mix", 0.0))
    refresh = int(cfg.get("online_refresh_interval", 0))
    step = int(cfg.get("global_step", 0))
    refresh_due = (refresh <= 0) or (step % refresh == 0)

    if refresh > 0 and replay is None:
        raise ValueError("online_refresh_interval requires an endpoint replay buffer")

    if refresh > 0 and replay is None:
        raise ValueError("online_refresh_interval requires an endpoint replay buffer")

    if refresh > 0 and replay is not None:
        # exp 33: use cached endpoints between refresh steps; only regenerate when due.
        x1_hat = replay.sample(pa, pb, "fwd", len(z0), z0.device, gen)
        x0_hat = replay.sample(pa, pb, "rev", len(z1), z1.device, gen)
        if x1_hat is None or refresh_due:
            x1_hat = (mixed_sde_forward(teacher, z0, pa, pb, stats, eps, steps, t_floor, gen)
                      if mode == "sde" else
                      mixed_ode_forward(teacher, z0, pa, pb, stats, eps, steps, t_floor))
            replay.push(pa, pb, "fwd", x1_hat)
        if x0_hat is None or refresh_due:
            x0_hat = (mixed_sde_reverse(teacher, z1, pa, pb, stats, eps, steps, t_floor, gen)
                      if mode == "sde" else
                      mixed_ode_reverse(teacher, z1, pa, pb, stats, eps, steps, t_floor))
            replay.push(pa, pb, "rev", x0_hat)
    elif mode == "sde":
        x1_hat = mixed_sde_forward(teacher, z0, pa, pb, stats, eps, steps, t_floor, gen)
        x0_hat = mixed_sde_reverse(teacher, z1, pa, pb, stats, eps, steps, t_floor, gen)
    else:
        # deterministic for BOTH directions (exp 32 control)
        x1_hat = mixed_ode_forward(teacher, z0, pa, pb, stats, eps, steps, t_floor)
        x0_hat = mixed_ode_reverse(teacher, z1, pa, pb, stats, eps, steps, t_floor)

    if mix > 0:
        k = int(round(len(z0) * mix))
        if k > 0:
            x1_hat = x1_hat.clone()
            x0_hat = x0_hat.clone()
            x1_hat[:k] = z1[:k]
            x0_hat[:k] = z0[:k]

    # alpha-DSBM Algorithm 1 (V11 fix, docs/v11/V11_THEORY.md 7, defect 3).
    #
    # V10 called `mixed_bridge_loss` for both pairings. That helper picks b+ or b- per
    # sample from `dirs`, so the (x0, x1_hat) pairing also supervised b+ for some
    # samples. That is NOT the paper's update:
    #
    #   reverse SDE gives x0_hat  ->  supervise the FORWARD drift on (x0_hat, x1)
    #   forward SDE gives x1_hat  ->  supervise the REVERSE drift on (x0, x1_hat)
    #
    # Each pairing now carries exactly one drift type for every sample, which makes the
    # loss direction-specific as in the paper.
    l_fwd = mixed_bridge_loss_fwd(model, x0_hat.detach(), z1, pa, pb, stats, eps, min_t,
                                  gen, t_floor)
    l_rev = mixed_bridge_loss_rev(model, z0, x1_hat.detach(), pa, pb, stats, eps, min_t,
                                  gen, t_floor)
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

    # alpha-DSBM Algorithm 1: (x0_hat, x1) -> b+ only, (x0, x1_hat) -> b- only.
    l_fwd = bridge_loss(model, x0_hat.detach(), z1, a, b, st, eps, min_t, gen, t_floor,
                        mode="fwd")
    l_rev = bridge_loss(model, z0, x1_hat.detach(), a, b, st, eps, min_t, gen, t_floor,
                        mode="rev")

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
