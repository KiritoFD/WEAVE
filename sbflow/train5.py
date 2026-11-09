"""Five-style Schrödinger-flow trainer (docs/bridgev10.md).

One shared network, ten unordered style pairs, 20 directed transfers.

Objectives
----------
`bidir`          half the batch supervises b+ (local time t), half supervises b-
                 (local time 1-t). This is the blog-family baseline (exp 04).
`bidir_joint`    every bridge sample supervises BOTH directions (exp 08).
`single_velocity` predicts v* = (z1-z0) + sqrt(eps)(1-2t)/(2 sqrt(t(1-t))) xi
                 with its own EDM coefficients (exp 05), or without EDM (exp 06).
`score_current`  two heads for (v, s) with b+ = v + eps/2 s, b- = -v + eps/2 s (exp 10).

Ablations wired here: `--style-cond {decomposed,pair}`, `--heads-out {1,2}`,
`--eps`, `--time-sampling {uniform,stratified}`, `--target {drift,endpoint}`,
`--sampler {perm,replacement}`, `--ot-ratio`, `--ot-cost`, `--local-conv`,
`--pair-rank`.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .bridge import bridge_sample
from .data5 import (STYLE_NAMES, load_caches, make_sampler, pair_schedule,
                    record_experiment)
from .network import BidirDiT
from .ot import paired_batch
from .preconditioning import (StyleStats, direct_velocity_coefficients,
                              direct_velocity_coefficients_batch,
                              endpoint_coefficients, endpoint_coefficients_batch,
                              moment_coefficients, moment_coefficients_batch,
                              normalized_endpoint_target, normalized_target,
                              normalized_target_batch, normalized_velocity_target,
                              pair_stats_per_sample, precondition_input,
                              precondition_input_batch, reconstruct_drift_batch,
                              reconstruct_velocity)
from .time_sampling import sample_time


# --------------------------------------------------------------------------- #
def build_model(args, channels, image_size, num_styles) -> BidirDiT:
    return BidirDiT(
        channels=channels, image_size=image_size, patch=args.patch,
        width=args.width, depth=args.depth, heads=args.heads,
        num_styles=num_styles, style_cond=args.style_cond,
        heads_out=(2 if args.objective == "score_current" else args.heads_out),
        two_head=("score_current" if args.objective == "score_current" else "drift"),
        local_conv=args.local_conv, pair_rank=args.pair_rank,
        style_film=args.style_film,
    )


# --------------------------------------------------------------------------- #
# objectives
# --------------------------------------------------------------------------- #
def _bridge_terms(z0, z1, t, eps, gen, stats: StyleStats, a, b, dtype):
    xi = torch.randn(z0.shape, device=z0.device, generator=gen, dtype=dtype)
    xt, bplus, bminus = bridge_sample(z0, z1, t, eps, xi)
    return xt, bplus, bminus


def objective_bidir(model, z0, z1, a, b, stats, eps, min_t, gen, args, t_weight=None):
    """Blog-family dual drift. Half batch forward, half reverse (or both if joint)."""
    B = z0.shape[0]
    t, tw = sample_time(args.time_sampling, B, min_t, gen, z0.device,
                        args.strata, z0.dtype)
    if t_weight is not None:
        tw = tw * t_weight
    xt, bplus, bminus = _bridge_terms(z0, z1, t, eps, gen, stats, a, b, z0.dtype)
    pm = stats.pair(a, b)
    sa = torch.full((B,), a, device=z0.device, dtype=torch.long)
    sb = torch.full((B,), b, device=z0.device, dtype=torch.long)

    if args.objective == "bidir_joint":
        # both directions on the same bridge sample (exp 08)
        xf, cf = precondition_input(xt, t, eps, pm)
        xb, cb = precondition_input(xt, 1 - t, eps, pm.reversed())
        tf = normalized_target(xt, bplus, cf)
        tb = normalized_target(xt, bminus, cb)
        if args.target == "endpoint":
            ef = endpoint_coefficients(t, eps, pm)
            eb = endpoint_coefficients(1 - t, eps, pm.reversed())
            xe_f, _ = _ep_input(xt, ef)
            xe_b, _ = _ep_input(xt, eb)
            tf = normalized_endpoint_target(xt, z1, ef)
            tb = normalized_endpoint_target(xt, z0, eb)
            pf = model(xe_f, t, torch.ones_like(sa), sa, sb).float()
            pb = model(xe_b, 1 - t, torch.zeros_like(sa), sa, sb).float()
        else:
            pf = model(xf, t, torch.ones_like(sa), sa, sb).float()
            pb = model(xb, 1 - t, torch.zeros_like(sa), sa, sb).float()
        lf = F.mse_loss(pf, tf)
        lb = F.mse_loss(pb, tb)
        w = tw.mean()
        return w * 0.5 * (lf + lb), {"f_mse": float(lf), "b_mse": float(lb)}

    half = B // 2
    xf, cf = precondition_input(xt[:half], t[:half], eps, pm)
    xb, cb = precondition_input(xt[half:], 1 - t[half:], eps, pm.reversed())
    if args.target == "endpoint":
        ef = endpoint_coefficients(t[:half], eps, pm)
        eb = endpoint_coefficients(1 - t[half:], eps, pm.reversed())
        yf = normalized_endpoint_target(xt[:half], z1[:half], ef)
        yb = normalized_endpoint_target(xt[half:], z0[half:], eb)
        xin = torch.cat([_ep_input(xt[:half], ef)[0], _ep_input(xt[half:], eb)[0]], 0)
    else:
        yf = normalized_target(xt[:half], bplus[:half], cf)
        yb = normalized_target(xt[half:], bminus[half:], cb)
        xin = torch.cat([xf, xb], 0)
    tau = torch.cat([t[:half], 1 - t[half:]], 0)
    d = torch.cat([torch.ones(half, device=z0.device, dtype=torch.long),
                   torch.zeros(B - half, device=z0.device, dtype=torch.long)])
    y = torch.cat([yf, yb], 0)
    pred = model(xin, tau, d, torch.full_like(d, a), torch.full_like(d, b)).float()
    per = (pred - y.float()).square().mean(dim=(1, 2, 3))
    loss = (per * tw).mean()
    return loss, {"f_mse": float((pred[:half] - y[:half]).square().mean()),
                  "b_mse": float((pred[half:] - y[half:]).square().mean())}


def _ep_input(xt, coeffs):
    mx, my, ci, cs, co = coeffs
    return (xt - mx) * ci, coeffs


def objective_bidir_mixed(model, z0, z1, pa, pb, dirs, stats, eps, min_t, gen, args,
                          t_weight=None):
    """One batch, many style pairs: every sample carries its own pair and direction.

    Bridge MSE is additive over samples, so a batch does not have to be restricted to a
    single style pair. Each sample gets:
      * its own canonical pair (pa, pb) with pa < pb,
      * its own direction: 1 = forward drift b+ at local time t, 0 = reverse drift b-
        at local time 1-t with the reversed endpoint moments,
      * its own per-channel EDM coefficients from that pair's statistics.
    """
    B = z0.shape[0]
    t, tw = sample_time(args.time_sampling, B, min_t, gen, z0.device, args.strata, z0.dtype)
    if t_weight is not None:
        tw = tw * t_weight
    xi = torch.randn(z0.shape, device=z0.device, generator=gen, dtype=z0.dtype)
    xt, bplus, bminus = bridge_sample(z0, z1, t, eps, xi)

    fwd = (dirs == 1)
    fwd_c = fwd.view(-1, 1)
    mu0, mu1, v0, v1, cov = _mixed_stats(stats, pa, pb, dirs, z0.dtype)
    tau = torch.where(fwd, t, 1 - t)

    xin, cf = precondition_input_batch(xt, tau, eps, mu0, mu1, v0, v1, cov)
    y = normalized_target_batch(xt, torch.where(fwd_c.view(-1, 1, 1, 1), bplus, bminus), cf)
    pred = model(xin, tau, dirs.long(), pa.long(), pb.long()).float()
    per = (pred - y.float()).square().mean(dim=(1, 2, 3))
    loss = (per * tw).mean()
    f_mse = float(per[fwd].mean()) if fwd.any() else 0.0
    b_mse = float(per[~fwd].mean()) if (~fwd).any() else 0.0
    return loss, {"f_mse": f_mse, "b_mse": b_mse}


def _mixed_stats(stats, pa, pb, dirs, dtype):
    """Per-sample endpoint statistics; delegates to the shared helper."""
    return pair_stats_per_sample(stats, pa, pb, dirs, dtype)


def objective_mixed(model, z0, z1, pa, pb, dirs, stats, eps, min_t, gen, args, s_scale=None):
    """Dispatch the per-sample mixed-batch objective for ANY supported target.

    Every sample carries its own canonical pair, direction and EDM statistics, so the
    same mixed sampler serves `bidir`, `bidir_joint`, `single_velocity`, `score_current`
    and `endpoint`. This removes the training-schedule confound where non-bidir arms
    silently fell back to one pair per update.
    """
    if args.objective in ("monge", "pivot_cfm"):
        # One-step map T(z,i,j) = z + Delta(z,i,j): no ODE and no bridge time.
        tau = torch.full((z0.shape[0],), 0.5, device=z0.device, dtype=z0.dtype)
        delta = model(z0, tau, dirs.long(), pa.long(), pb.long()).float()
        z_out = z0 + delta
        base = torch.nn.functional.mse_loss(z_out, z1.float())
        if getattr(args, "cycle_weight", 0) > 0:
            # weak cycle/move regulariser: penalise excessive displacement only
            base = base + args.cycle_weight * delta.square().mean()
        return base, {"f_mse": float(base), "b_mse": 0.0}

    B = z0.shape[0]
    t, tw = sample_time(args.time_sampling, B, min_t, gen, z0.device, args.strata, z0.dtype)
    xi = torch.randn(z0.shape, device=z0.device, generator=gen, dtype=z0.dtype)
    xt, bplus, bminus = bridge_sample(z0, z1, t, eps, xi)
    mu0, mu1, v0, v1, cov = _mixed_stats(stats, pa, pb, dirs, z0.dtype)
    fwd = (dirs == 1)
    fwd_c = fwd.view(-1, 1, 1, 1)

    if args.objective == "single_velocity":
        tt = t.view(-1, 1, 1, 1)
        s = (eps * tt * (1 - tt)).clamp_min(0).sqrt()
        c = ((eps ** 0.5) * (1 - 2 * tt) / (2 * (tt * (1 - tt)).clamp_min(1e-8).sqrt())
             if eps > 0 else 0.0)
        vstar = (z1 - z0) + c * xi
        if args.edm:
            # A single velocity field is defined on the canonical orientation only, so the
            # endpoint statistics must NOT be swapped by the direction flag here.
            mu0f, mu1f, v0f, v1f, covf = pair_stats_per_sample(
                stats, pa, pb, torch.ones_like(dirs), z0.dtype)
            mx, my, ci, cs, co = direct_velocity_coefficients_batch(t, max(eps, 1e-8),
                                                                    mu0f, mu1f, v0f, v1f, covf)
            xin = (xt - mx) * ci
            y = (vstar - my - cs * (xt - mx)) / co
        else:
            xin, y = xt, vstar
        pred = model(xin, t, torch.ones_like(pa), pa.long(), pb.long()).float()
        per = (pred - y.float()).square().mean(dim=(1, 2, 3))
        return (per * tw).mean(), {"f_mse": float(per.mean()), "b_mse": 0.0}

    if args.objective == "score_current":
        tau = torch.where(fwd, t, 1 - t)
        mx, my, ci, cs, co = direct_velocity_coefficients_batch(tau, eps, mu0, mu1, v0, v1, cov)
        v_star = 0.5 * (bplus - bminus)
        s_star = (bplus + bminus) / eps
        xin = (xt - mx) * ci
        v_norm = (v_star - my - cs * (xt - mx)) / co
        s_norm = s_star * s_scale.view(1, -1, 1, 1)
        pv, ps = model(xin, tau, dirs.long(), pa.long(), pb.long(), return_both=True)
        lv = torch.nn.functional.mse_loss(pv.float(), v_norm.float())
        ls = torch.nn.functional.mse_loss(ps.float(), s_norm.float())
        return lv + args.score_weight * ls, {"f_mse": float(lv), "b_mse": float(ls)}

    if args.objective == "endpoint":
        tau = torch.where(fwd, t, 1 - t)
        mx, my, ci, cs, co = endpoint_coefficients_batch(tau, eps, mu0, mu1, v0, v1, cov)
        y_end = torch.where(fwd_c, z1, z0)
        xin = (xt - mx) * ci
        y = (y_end - my - cs * (xt - mx)) / co
        pred = model(xin, tau, dirs.long(), pa.long(), pb.long()).float()
        per = (pred - y.float()).square().mean(dim=(1, 2, 3))
        return (per * tw).mean(), {"f_mse": float(per[fwd].mean()) if fwd.any() else 0.0,
                                   "b_mse": float(per[~fwd].mean()) if (~fwd).any() else 0.0}

    if args.objective == "bidir_joint":
        # both directions on the same bridge sample, per sample pair statistics
        tau_f = t
        tau_b = 1 - t
        cf = precondition_input_batch(xt, tau_f, eps, mu0, mu1, v0, v1, cov)
        xf_in, cf = cf[0], cf[1]
        yf = normalized_target_batch(xt, bplus, cf)
        mub0, mub1, vb0, vb1, covb = _mixed_stats(stats, pa, pb, torch.zeros_like(dirs),
                                                  z0.dtype)
        cb = precondition_input_batch(xt, tau_b, eps, mub0, mub1, vb0, vb1, covb)
        xb_in, cb = cb[0], cb[1]
        yb = normalized_target_batch(xt, bminus, cb)
        pf = model(xf_in, tau_f, torch.ones_like(pa), pa.long(), pb.long()).float()
        pb_ = model(xb_in, tau_b, torch.zeros_like(pa), pa.long(), pb.long()).float()
        lf = torch.nn.functional.mse_loss(pf, yf.float())
        lb = torch.nn.functional.mse_loss(pb_, yb.float())
        return tw.mean() * 0.5 * (lf + lb), {"f_mse": float(lf), "b_mse": float(lb)}

    # default: bidir (half forward, half reverse, per-sample statistics)
    tau = torch.where(fwd, t, 1 - t)
    xin, cf = precondition_input_batch(xt, tau, eps, mu0, mu1, v0, v1, cov)
    y = normalized_target_batch(xt, torch.where(fwd_c, bplus, bminus), cf)
    pred = model(xin, tau, dirs.long(), pa.long(), pb.long()).float()
    per = (pred - y.float()).square().mean(dim=(1, 2, 3))
    return (per * tw).mean(), {"f_mse": float(per[fwd].mean()) if fwd.any() else 0.0,
                               "b_mse": float(per[~fwd].mean()) if (~fwd).any() else 0.0}


def mixed_velocity(model, z, t, pa, pb, stats, eps, objective: str = "bidir"):
    """Per-sample probability-flow velocity for a mixed pair batch, honouring `objective`.

    The distribution loss must differentiate the SAME velocity form the model was trained
    with; hard-coding the dual-drift reconstruction would silently supervise the wrong
    trajectory for single-velocity / score-current / endpoint runs.
    """
    B = z.shape[0]
    t = t.clamp(2e-3, 1 - 2e-3)
    fwd = torch.ones(B, device=z.device, dtype=torch.long)
    rev = torch.zeros(B, device=z.device, dtype=torch.long)

    if objective == "single_velocity":
        mu0, mu1, v0, v1, cov = pair_stats_per_sample(stats, pa, pb, fwd, z.dtype)
        mx, my, ci, cs, co = direct_velocity_coefficients_batch(t, max(eps, 1e-8),
                                                                mu0, mu1, v0, v1, cov)
        return reconstruct_velocity(z, model((z - mx) * ci, t, fwd, pa.long(), pb.long()).float(),
                                    (mx, my, ci, cs, co))

    if objective == "score_current":
        mu0, mu1, v0, v1, cov = pair_stats_per_sample(stats, pa, pb, fwd, z.dtype)
        mx, my, ci, cs, co = direct_velocity_coefficients_batch(t, max(eps, 1e-8),
                                                                mu0, mu1, v0, v1, cov)
        pv, _ = model((z - mx) * ci, t, fwd, pa.long(), pb.long(), return_both=True)
        return reconstruct_velocity(z, pv.float(), (mx, my, ci, cs, co))

    if objective == "endpoint":
        mu0, mu1, v0, v1, cov = pair_stats_per_sample(stats, pa, pb, fwd, z.dtype)
        ef = endpoint_coefficients_batch(t, eps, mu0, mu1, v0, v1, cov)
        xin = (z - ef[0]) * ef[2]
        z1_hat = reconstruct_drift_batch(z, model(xin, t, fwd, pa.long(), pb.long()).float(), ef)
        mu0r, mu1r, v0r, v1r, covr = pair_stats_per_sample(stats, pa, pb, rev, z.dtype)
        er = endpoint_coefficients_batch(1 - t, eps, mu0r, mu1r, v0r, v1r, covr)
        xinb = (z - er[0]) * er[2]
        z0_hat = reconstruct_drift_batch(z, model(xinb, 1 - t, rev, pa.long(), pb.long()).float(), er)
        bplus = (z1_hat - z) / (1 - t).clamp_min(2e-3).view(-1, 1, 1, 1)
        bminus = (z0_hat - z) / t.clamp_min(2e-3).view(-1, 1, 1, 1)
        return 0.5 * (bplus - bminus)

    # bidir / bidir_joint: dual-drift reconstruction with per-sample statistics
    mu0, mu1, v0, v1, cov = pair_stats_per_sample(stats, pa, pb, fwd, z.dtype)
    xin, cf = precondition_input_batch(z, t, eps, mu0, mu1, v0, v1, cov)
    bplus = reconstruct_drift_batch(z, model(xin, t, fwd, pa.long(), pb.long()).float(), cf)
    mu0r, mu1r, v0r, v1r, covr = pair_stats_per_sample(stats, pa, pb, rev, z.dtype)
    xinb, cb = precondition_input_batch(z, 1 - t, eps, mu0r, mu1r, v0r, v1r, covr)
    bminus = reconstruct_drift_batch(z, model(xinb, 1 - t, rev, pa.long(), pb.long()).float(), cb)
    return 0.5 * (bplus - bminus)


def _mixed_pair_ids(pairs, batch, device, gen_cpu):
    """Evenly assign (pair, direction) to a batch: 10 pairs x 2 directions.

    `gen_cpu` must be a CPU generator (torch.randperm defaults to CPU).
    """
    n_cell = batch // (len(pairs) * 2)
    if n_cell < 1:
        raise ValueError(f"batch {batch} too small for {len(pairs)} pairs x 2 directions")
    pa, pb, dirs = [], [], []
    for (a, b) in pairs:
        pa.extend([a] * n_cell * 2)
        pb.extend([b] * n_cell * 2)
        dirs.extend([1] * n_cell + [0] * n_cell)
    used = len(pa)
    for i in range(batch - used):                     # leftovers cycle through cells
        k = i % (len(pairs) * 2)
        a, b = pairs[k // 2]
        pa.append(a)
        pb.append(b)
        dirs.append(1 if k % 2 == 0 else 0)
    perm = torch.randperm(batch, generator=gen_cpu)
    return (torch.tensor(pa)[perm].to(device), torch.tensor(pb)[perm].to(device),
            torch.tensor(dirs)[perm].to(device))


def _mixed_batch(data, pa, pb, samplers, device, gen):
    """Draw z0 from style pa[i] and z1 from style pb[i] for every sample."""
    B = len(pa)
    z0 = torch.empty((B,) + tuple(data.z[0].shape[1:]), device=device, dtype=data.z[0].dtype)
    z1 = torch.empty_like(z0)
    for s in range(data.num_styles):
        pos0 = (pa == s).nonzero(as_tuple=True)[0]
        if len(pos0):
            z0[pos0] = data.z[s][samplers[s].draw(len(pos0)).to(device)]
        pos1 = (pb == s).nonzero(as_tuple=True)[0]
        if len(pos1):
            z1[pos1] = data.z[s][samplers[s].draw(len(pos1)).to(device)]
    return z0, z1


def objective_single_velocity(model, z0, z1, a, b, stats, eps, min_t, gen, args, t_weight=None):
    """Direct velocity target v* = (z1-z0) + c(t) xi, with re-derived EDM (exp 05/06)."""
    B = z0.shape[0]
    t, tw = sample_time(args.time_sampling, B, min_t, gen, z0.device, args.strata, z0.dtype)
    if t_weight is not None:
        tw = tw * t_weight
    xi = torch.randn(z0.shape, device=z0.device, generator=gen, dtype=z0.dtype)
    tt = t.view(-1, 1, 1, 1)
    xt = (1 - tt) * z0 + tt * z1 + (eps * tt * (1 - tt)).clamp_min(0).sqrt() * xi
    c = (eps ** 0.5) * (1 - 2 * tt) / (2 * (tt * (1 - tt)).clamp_min(1e-8).sqrt()) if eps > 0 else 0.0
    vstar = (z1 - z0) + c * xi

    pm = stats.pair(a, b)
    if args.edm:
        mx, my, ci, cs, co = direct_velocity_coefficients(t, eps if eps > 0 else 1e-8, pm)
        xin = (xt - mx) * ci
        y = (vstar - my - cs * (xt - mx)) / co
    else:
        # exp 06: no EDM. Whiten only by the marginal velocity std so the loss scale
        # is comparable, but keep the target in physical units.
        xin = xt
        y = vstar
    sa = torch.full((B,), a, device=z0.device, dtype=torch.long)
    sb = torch.full((B,), b, device=z0.device, dtype=torch.long)
    d = torch.ones(B, device=z0.device, dtype=torch.long)
    pred = model(xin, t, d, sa, sb).float()
    per = (pred - y.float()).square().mean(dim=(1, 2, 3))
    return (per * tw).mean(), {"f_mse": float(per.mean()), "b_mse": 0.0}


def objective_score_current(model, z0, z1, a, b, stats, eps, min_t, gen, args,
                            s_scale: torch.Tensor, t_weight=None):
    """Two heads: transport velocity v and score s, b+/- rebuilt from them (exp 10)."""
    B = z0.shape[0]
    t, tw = sample_time(args.time_sampling, B, min_t, gen, z0.device, args.strata, z0.dtype)
    if t_weight is not None:
        tw = tw * t_weight
    xt, bplus, bminus = _bridge_terms(z0, z1, t, eps, gen, stats, a, b, z0.dtype)
    pm = stats.pair(a, b)
    v_star = 0.5 * (bplus - bminus)
    s_star = (bplus + bminus) / eps

    mx, my, ci, cs, co = direct_velocity_coefficients(t, eps, pm)
    xin = (xt - mx) * ci
    v_norm = (v_star - my - cs * (xt - mx)) / co
    s_norm = s_star * s_scale.view(1, -1, 1, 1)

    sa = torch.full((B,), a, device=z0.device, dtype=torch.long)
    sb = torch.full((B,), b, device=z0.device, dtype=torch.long)
    d = torch.ones(B, device=z0.device, dtype=torch.long)
    pv, ps = model(xin, t, d, sa, sb, return_both=True)
    lv = F.mse_loss(pv.float(), v_norm.float())
    ls = F.mse_loss(ps.float(), s_norm.float())
    # Equal weight after normalization; the doc forbids letting the score scale dominate.
    return lv + args.score_weight * ls, {"f_mse": float(lv), "b_mse": float(ls)}


def _apply_config(p, argv):
    """Load a flat JSON config and feed it to argparse as explicit CLI overrides.

    Keys are argparse *dest* names (underscores). Unknown / bookkeeping keys are
    ignored rather than rejected, so the suite driver can keep its metadata in the
    same file. Explicit command-line arguments take precedence (appended last).
    """
    import json as _json

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="")
    known, rest = pre.parse_known_args(argv)
    if not known.config:
        return p.parse_args(argv)
    cfg = _json.loads(Path(known.config).read_text(encoding="utf-8"))

    dests = {a.dest: a for a in p._actions if a.dest != "help"}
    bool_true = {d for d, a in dests.items() if isinstance(a, argparse._StoreTrueAction)}
    skip = {"id", "group", "note", "fork_from", "priority", "quick200", "full750",
            "config", "vae", "scaling_factor", "latent_shape",
            "dataset_styles", "dataset_images"}
    # store_false flags share a dest with a store_true option; map a false value back
    # to the negative CLI flag. (single_pair / frequency_stats in JSON are redundant
    # aliases and are simply ignored because they are not real dests.)
    neg_alias = {"mixed_batch": "--single-pair",
                 "equal_weight_stats": "--frequency-stats"}
    pos_alias = {"mixed_batch": "--mixed-batch",
                 "equal_weight_stats": "--equal-weight-stats"}
    extra = []
    for k, v in cfg.items():
        k = str(k).replace("-", "_")
        if k in skip or v is None or k not in dests:
            continue
        key = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            if k in neg_alias:
                extra.append(neg_alias[k] if not v else pos_alias[k])
            elif k in bool_true and v:
                extra.append(key)
            # store_true dest with false value: keep the parser default
        elif isinstance(v, list):
            extra.extend([key] + [str(x) for x in v])
        else:
            extra.extend([key, str(v)])
    return p.parse_args(extra + rest)


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-id", required=True)
    p.add_argument("--caches", nargs="+", required=True, help="one latent cache per style")
    p.add_argument("--output", required=True)
    p.add_argument("--config", default="", help="flat JSON config; explicit CLI args win")
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--microbatch", type=int, default=0,
                   help="split each update into chunks of this size; 0 = single chunk")
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--eps", type=float, default=0.04)
    p.add_argument("--min-t", type=float, default=0.002)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--objective", choices=["bidir", "bidir_joint", "single_velocity",
                                           "score_current", "monge", "pivot_cfm"],
                   default="bidir")
    p.add_argument("--edm", type=int, default=1, help="0 disables EDM for single_velocity (exp 06)")
    p.add_argument("--heads-out", type=int, choices=[1, 2], default=1)
    p.add_argument("--style-cond", choices=["decomposed", "pair"], default="decomposed")
    p.add_argument("--target", choices=["drift", "endpoint"], default="drift")
    p.add_argument("--time-sampling", choices=["uniform", "stratified"], default="uniform")
    p.add_argument("--strata", type=int, default=8)
    p.add_argument("--sampler", choices=["perm", "replacement"], default="perm")
    p.add_argument("--score-weight", type=float, default=1.0)

    p.add_argument("--ot-ratio", type=float, default=0.0)
    p.add_argument("--ot-cost", choices=["rgb", "dino"], default="rgb")
    p.add_argument("--ot-reg", type=float, default=0.05)
    p.add_argument("--ot-tol", type=float, default=1e-6)

    p.add_argument("--width", type=int, default=256)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--patch", type=int, default=2)
    p.add_argument("--local-conv", action="store_true")
    p.add_argument("--pair-rank", type=int, default=0)
    p.add_argument("--style-film", action="store_true",
                   help="per-layer Style-FiLM modulation (exp 16 / 18)")
    p.add_argument("--cycle-weight", type=float, default=0.0,
                   help="weak move regulariser for the one-step Monge arm (exp 38)")

    p.add_argument("--log-interval", type=int, default=20)
    p.add_argument("--save-interval", type=int, default=200)
    p.add_argument("--resume", default="")
    p.add_argument("--lr-cycle-steps", type=int, default=0,
                   help="length of the LR cosine cycle. Defaults to --steps. Set this to the "
                        "TOTAL budget (e.g. 4000) when resuming a shared prefix so the fork "
                        "does not restart warmup or rescale the schedule.")
    p.add_argument("--save-every", type=int, default=0,
                   help="alias for --save-interval in equivalent steps")

    # --- group D: direct target-distribution supervision ---
    p.add_argument("--dist-loss", choices=["none", "global", "local", "both"], default="none")
    p.add_argument("--style-enc", choices=["learned", "fixed"], default="learned")
    p.add_argument("--lambda-dist", type=float, default=0.1)
    p.add_argument("--style-enc-epochs", type=int, default=60)
    p.add_argument("--style-enc-ckpt", default="",
                   help="shared frozen style-encoder checkpoint; trained once, reused by "
                        "every learned-SWD arm")
    p.add_argument("--ode-steps-train", type=int, default=2,
                   help="short differentiable ODE used to produce endpoints for L_dist")
    p.add_argument("--ode-microbatch", type=int, default=64,
                   help="cap the differentiable ODE graph for L_dist (32-128 per book)")
    p.add_argument("--grad-microbatch", type=int, default=0,
                   help="when >0, split an update's forward+backward into chunks of this "
                        "size; used by heavy differentiable-SWD / online arms")
    p.add_argument("--dist-nproj", type=int, default=64)

    # --- group E: online alpha-DSBM ---
    p.add_argument("--online", choices=["none", "sde", "ode"], default="none")
    p.add_argument("--online-steps", type=int, default=4)
    p.add_argument("--online-direction", choices=["both", "forward", "reverse"], default="both")
    p.add_argument("--online-mix", type=float, default=0.0)
    p.add_argument("--online-replay", action="store_true")
    p.add_argument("--online-raw-teacher", action="store_true")
    p.add_argument("--online-dynamic-cov", action="store_true")
    p.add_argument("--online-warmup", type=int, default=0,
                   help="steps of plain bridge training before online updates start")
    p.add_argument("--online-refresh-interval", type=int, default=0,
                   help="exp 33: refresh cached online endpoints every N steps; "
                        "0 means fully online per-step SDE")
    p.add_argument("--triangle-weight", type=float, default=0.0,
                   help="exp 39: weight of T_jk(T_ij(z)) ~= T_ik(z) consistency")

    # --- group F: shared Gaussian pivot ---
    p.add_argument("--mixed-batch", action="store_true", default=True,
                   help="mix all 10 unordered pairs inside every batch (default)")
    p.add_argument("--single-pair", dest="mixed_batch", action="store_false",
                   help="legacy: one style pair per update")
    p.add_argument("--equal-weight-stats", action="store_true", default=True,
                   help="equal-weight-per-style global mean/std (default)")
    p.add_argument("--frequency-stats", dest="equal_weight_stats", action="store_false")
    p.add_argument("--pivot", action="store_true",
                   help="train N(0,I)->style flows and compose T_ij = F_j o F_i^-1")

    # --- suite40 bookkeeping (accepted so configs can be passed verbatim) ---
    p.add_argument("--equiv-steps", type=int, default=0)
    p.add_argument("--equiv-batch", type=int, default=512)
    p.add_argument("--save-equiv", default="")
    p.add_argument("--full-equiv", type=int, default=0)
    args = _apply_config(p, sys.argv[1:])

    if args.batch % 2:
        raise ValueError("even batch required")
    if args.microbatch and args.batch % args.microbatch:
        raise ValueError("microbatch must divide batch")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    # Independent RNG streams: data order, bridge noise/time, OT, init.
    g_data = torch.Generator(device="cpu").manual_seed(args.seed * 17 + 1)
    g_noise = torch.Generator(device=device).manual_seed(args.seed * 31 + 7)
    g_time = torch.Generator(device=device).manual_seed(args.seed * 53 + 13)
    g_ot = torch.Generator(device=device).manual_seed(args.seed * 7919 + 131)
    g_sched = torch.Generator().manual_seed(args.seed * 101 + 3)
    g_mix = torch.Generator().manual_seed(args.seed * 13 + 5)

    data = load_caches([Path(c) for c in args.caches], device, STYLE_NAMES[: len(args.caches)],
                       equal_weight=args.equal_weight_stats)
    channels, H, W = data.z[0].shape[1:]
    if args.pivot:
        from .pivot import add_pivot_style, pivot_pairs
        n_real = data.num_styles
        data = add_pivot_style(data, n_pivot=min(4096, args.batch * 2), seed=args.seed)
        pairs = pivot_pairs(n_real)
        print(f"[train5] pivot mode: {n_real} real styles + Gaussian pivot, "
              f"{len(pairs)} pairs", flush=True)
    else:
        pairs = data.pairs()
    sched = pair_schedule(pairs, args.steps, g_sched)

    model = build_model(args, channels, H, data.num_styles).to(device)
    ema = build_model(args, channels, H, data.num_styles).to(device)
    ema.load_state_dict(model.state_dict())
    ema.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    warm = max(1, args.warmup)
    cycle = args.lr_cycle_steps or args.steps

    def lr_ratio(s):
        if s < warm:
            return max(0.02, (s + 1) / warm)
        v = (s - warm) / max(1, cycle - warm)
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1, v)))
    sched_lr = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_ratio)

    samplers = [make_sampler(args.sampler, n, g_data) for n in data.counts]
    s_scale = torch.ones(channels, device=device)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # ---- real resume: model, EMA, optimizer, scheduler, step and every RNG stream ----
    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            sched_lr.load_state_dict(ckpt["scheduler"])
        start_step = int(ckpt.get("step", 0))
        if "score_scale" in ckpt:
            s_scale = ckpt["score_scale"].to(device)
        # The resumed run must reuse the SAME global statistics; otherwise the shared
        # prefix comparison is invalid.
        prev = ckpt.get("style_stats")
        if prev is not None:
            cur = data.stats.state_dict()
            for k in cur:
                if k in prev and not torch.allclose(prev[k].float(), cur[k].float(), atol=1e-4):
                    raise ValueError(f"resume statistics mismatch on '{k}': the shared prefix "
                                     f"was trained with different global mean/std")
        rng = ckpt.get("rng", {})
        if rng:
            g_data.set_state(rng["g_data"])
            g_noise.set_state(rng["g_noise"])
            g_time.set_state(rng["g_time"])
            g_ot.set_state(rng["g_ot"])
            g_sched.set_state(rng["g_sched"])
            g_mix.set_state(rng["g_mix"])
            if "torch" in rng:
                torch.set_rng_state(rng["torch"])
            if "cuda" in rng and device == "cuda" and rng["cuda"] is not None:
                torch.cuda.set_rng_state_all(rng["cuda"])
        print(f"[train5] resumed from {args.resume} at step {start_step} "
              f"(lr cycle = {cycle} steps)", flush=True)
        # Restore the exact pair schedule and sampler positions so the fork continues the
        # same data stream instead of re-drawing a different one.
        if "pair_sched" in ckpt:
            saved = [tuple(x) for x in ckpt["pair_sched"]]
            end = start_step + args.steps
            # Keep the prefix's schedule intact and only generate the missing tail.
            if len(saved) < end:
                saved = saved + pair_schedule(pairs, end - len(saved), g_sched)
            sched = saved
        for s, st in zip(samplers, ckpt.get("samplers", [])):
            if st.get("perm") is not None and hasattr(s, "perm"):
                s.perm = st["perm"]
                s.pos = int(st.get("pos") or 0)

    # ---- optional group D distribution loss ----
    dist_loss = None
    if args.dist_loss != "none":
        from .style_loss import DistributionLoss, StyleEncoder, train_style_encoder
        enc = None
        if args.style_enc == "learned":
            # Train ONCE and reuse: every learned-SWD arm must load the same frozen encoder,
            # otherwise exp 22-26 differ in the representation as well as the supervision.
            enc_path = args.style_enc_ckpt or str(out / "style_encoder.pt")
            if Path(enc_path).exists():
                blob = torch.load(enc_path, map_location=device, weights_only=False)
                enc = StyleEncoder(channels=channels).to(device)
                enc.load_state_dict(blob["state"])
                enc.eval()
                for q in enc.parameters():
                    q.requires_grad_(False)
                print(f"[train5] loaded frozen style encoder from {enc_path}", flush=True)
            else:
                enc = train_style_encoder(data.z, epochs=args.style_enc_epochs, device=device,
                                          seed=args.seed,
                                          log=lambda m: print(m, flush=True))
                Path(enc_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save({"state": enc.state_dict(), "channels": channels}, enc_path)
                print(f"[train5] trained and saved style encoder -> {enc_path}", flush=True)
        dist_loss = DistributionLoss(args.dist_loss, args.style_enc, enc, args.dist_nproj)
        print(f"[train5] dist_loss={args.dist_loss} style_enc={args.style_enc} "
              f"lambda={args.lambda_dist}", flush=True)

    # ---- optional group E online teacher ----
    online_teacher = None
    replay = None
    online_cfg = {}
    cross_state: dict = {}
    if args.online != "none":
        import copy
        from .online_bridge import EndpointReplayBuffer
        online_teacher = copy.deepcopy(model).eval()
        for q in online_teacher.parameters():
            q.requires_grad_(False)
        replay = (EndpointReplayBuffer(seed=args.seed)
                  if (args.online_replay or args.online_refresh_interval > 0) else None)
        online_cfg = {"online_steps": args.online_steps, "online_mode": args.online,
                      "online_direction": args.online_direction, "online_mix": args.online_mix,
                      "online_replay": args.online_replay or args.online_refresh_interval > 0,
                      "online_refresh_interval": args.online_refresh_interval,
                      "t_floor": 0.002, "global_step": 0}
        print(f"[train5] online={args.online} steps={args.online_steps} "
              f"dir={args.online_direction} mix={args.online_mix} "
              f"warmup={args.online_warmup}", flush=True)
    if args.objective == "score_current":
        # Calibrate the score head: without this the score target is O(1/eps) and
        # dominates the loss (docs/bridgev10.md 3B exp 10).
        with torch.no_grad():
            acc, nb = torch.zeros(channels, device=device), 0
            cal_samplers = [make_sampler(args.sampler, n, torch.Generator().manual_seed(args.seed + 5))
                            for n in data.counts]
            for i in range(8):
                a_, b_ = pairs[i % len(pairs)]
                k = min(64, args.batch)
                z0 = data.z[a_][cal_samplers[a_].draw(k).to(device)]
                z1 = data.z[b_][cal_samplers[b_].draw(k).to(device)]
                tt, _ = sample_time("uniform", k, args.min_t, g_time, device, args.strata, z0.dtype)
                xi = torch.randn(z0.shape, device=device, generator=g_time, dtype=z0.dtype)
                xt, bp, bm = bridge_sample(z0, z1, tt, args.eps, xi)
                s_star = (bp + bm) / args.eps
                acc += s_star.square().mean((0, 2, 3))
                nb += 1
            rms = (acc / max(nb, 1)).sqrt().clamp_min(1e-6)
            s_scale = (1.0 / rms).to(device)
        print(f"[train5] score scale = {[round(float(v), 4) for v in s_scale[:6]]}...", flush=True)

    npars = sum(q.numel() for q in model.parameters())
    print(f"[train5] exp={args.experiment_id} params={npars} styles={data.num_styles} "
          f"pairs={len(pairs)} counts={data.counts} objective={args.objective} "
          f"batch={args.batch} accum={args.accum} eps={args.eps} device={device}", flush=True)

    def _equiv(step: int) -> int:
        """Equivalent exposure in patch8 / B512 update units."""
        return round(step * args.batch / max(1, args.equiv_batch))

    save_equiv = sorted({int(x) for x in args.save_equiv.split(",") if x.strip()}) \
        if args.save_equiv else []
    full_equiv = int(args.full_equiv or 0)
    saved_equiv: set = set()

    def save(step, tag: str = ""):
        samp = []
        for s in samplers:
            samp.append({"perm": s.perm.clone() if hasattr(s, "perm") else None,
                         "pos": getattr(s, "pos", None)})
        payload = {
            "model": model.state_dict(), "ema": ema.state_dict(),
            "optimizer": opt.state_dict(), "scheduler": sched_lr.state_dict(), "step": step,
            "equiv_step": _equiv(step),
            "style_stats": data.stats.state_dict(), "styles": data.styles,
            "model_kwargs": dict(channels=channels, image_size=H, patch=args.patch,
                                 width=args.width, depth=args.depth, heads=args.heads,
                                 num_styles=data.num_styles, style_cond=args.style_cond,
                                 heads_out=(2 if args.objective == "score_current" else args.heads_out),
                                 two_head=("score_current" if args.objective == "score_current"
                                           else "drift"),
                                 local_conv=args.local_conv, pair_rank=args.pair_rank,
                                 style_film=args.style_film),
            "objective": args.objective, "config": vars(args),
            "score_scale": s_scale.cpu(),
            "ae": data.ae, "ae_scaling_factor": data.scaling_factor,
            "rng": {"g_data": g_data.get_state(), "g_noise": g_noise.get_state(),
                    "g_time": g_time.get_state(), "g_ot": g_ot.get_state(),
                    "g_sched": g_sched.get_state(), "g_mix": g_mix.get_state(),
                    "torch": torch.get_rng_state(),
                    "cuda": (torch.cuda.get_rng_state_all() if device == "cuda" else None)},
            "samplers": samp,
            "pair_sched": [list(x) for x in sched],
        }
        name = (f"model_{tag}.pt" if tag else f"model_step{step}.pt")
        torch.save(payload, out / name)
        # Also expose a stable equivalent-step alias for fork resolution.
        eq = f"equiv{_equiv(step)}"
        alias = out / f"model_{eq}.pt"
        if tag != eq and not alias.exists():
            try:
                torch.save(payload, alias)
            except Exception:
                pass

    def sample_pair_batch(a, b, k):
        ia = samplers[a].draw(k)
        ib = samplers[b].draw(k)
        z0 = data.z[a][ia.to(device)]
        z1 = data.z[b][ib.to(device)]
        if args.ot_ratio > 0:
            t0 = data.thumb[a][ia.to(device)]
            t1 = data.thumb[b][ib.to(device)]
            z0, z1, _ = paired_batch(z0, z1, t0, t1, args.ot_ratio, args.ot_reg,
                                     g_ot, mode=args.ot_cost, ot_tol=args.ot_tol)
        return z0, z1

    t_start = time.perf_counter()
    hist = []
    dist_seen = {}
    # For a fresh run `steps` is the absolute end step. For a resumed fork it is the
    # number of ADDITIONAL updates, so the shared prefix keeps its global step index.
    end_step = start_step + args.steps
    for step in range(start_step + 1, end_step + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        a, b = sched[step - 1] if step - 1 < len(sched) else sched[-1]
        base_mb = args.microbatch or args.batch
        # Heavy arms accumulate several smaller forward/backward chunks per update so a
        # differentiable ODE or online teacher never holds a full-batch graph.
        mb = min(base_mb, args.grad_microbatch) if args.grad_microbatch > 0 else base_mb
        if args.batch % mb:
            raise ValueError(f"grad microbatch {mb} must divide batch {args.batch}")
        total, rec = 0.0, {}
        for _ in range(args.batch // mb):
            ctx = (torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda"
                   else contextlib.nullcontext())
            use_online = (online_teacher is not None and step > args.online_warmup)
            mixed = args.mixed_batch
            with ctx:
                if mixed:
                    mpa, mpb, mdirs = _mixed_pair_ids(pairs, mb, device, g_mix)
                    mz0, mz1 = _mixed_batch(data, mpa, mpb, samplers, device, g_data)
                    pair_tag = "mixed"
                    if use_online:
                        from .online_bridge import online_loss_mixed
                        online_cfg["global_step"] = step
                        if not args.online_raw_teacher:
                            with torch.no_grad():
                                for pt, ps in zip(online_teacher.parameters(), ema.parameters()):
                                    pt.copy_(ps)
                        else:
                            with torch.no_grad():
                                for pt, ps in zip(online_teacher.parameters(), model.parameters()):
                                    pt.copy_(ps)
                        loss, r = online_loss_mixed(model, online_teacher, mz0, mz1,
                                                    mpa, mpb, mdirs, data.stats, args.eps,
                                                    args.min_t, g_noise, online_cfg, replay)
                        r = {k: v for k, v in r.items() if not k.startswith("_")}
                        r.setdefault("f_mse", r.get("online_fwd", 0.0))
                        r.setdefault("b_mse", r.get("online_rev", 0.0))
                    else:
                        loss, r = objective_mixed(model, mz0, mz1, mpa, mpb, mdirs, data.stats,
                                                  args.eps, args.min_t, g_noise, args, s_scale)
                else:
                    z0, z1 = sample_pair_batch(a, b, mb)
                    pair_tag = f"({a},{b})"
                    if use_online:
                        from .online_bridge import online_loss
                        if not args.online_raw_teacher:
                            with torch.no_grad():
                                for pt, ps in zip(online_teacher.parameters(), ema.parameters()):
                                    pt.copy_(ps)
                        else:
                            with torch.no_grad():
                                for pt, ps in zip(online_teacher.parameters(), model.parameters()):
                                    pt.copy_(ps)
                        loss, r = online_loss(model, online_teacher, z0, z1, a, b, data.stats,
                                              args.eps, args.min_t, g_noise, online_cfg, replay,
                                              cross=cross_state.get((a, b)))
                        if args.online_dynamic_cov and "_x1_hat" in r:
                            from .online_bridge import estimate_cross_cov
                            cc = estimate_cross_cov(r["_x0_hat"], z1,
                                                    ema=cross_state.get((a, b)), decay=0.9)
                            cross_state[(a, b)] = cc
                            r["cross_cov_mean"] = float(cc.mean())
                        r = {k: v for k, v in r.items() if not k.startswith("_")}
                    elif args.objective in ("bidir", "bidir_joint"):
                        loss, r = objective_bidir(model, z0, z1, a, b, data.stats, args.eps,
                                                  args.min_t, g_noise, args)
                    elif args.objective == "single_velocity":
                        loss, r = objective_single_velocity(model, z0, z1, a, b, data.stats,
                                                            args.eps, args.min_t, g_noise, args)
                    else:
                        loss, r = objective_score_current(model, z0, z1, a, b, data.stats,
                                                          args.eps, args.min_t, g_noise, args,
                                                          s_scale)

                # Distribution supervision is applied uniformly to BOTH sampling paths,
                # so `bidir + mixed + dist_loss` can no longer silently skip L_dist.
                if args.triangle_weight > 0 and mixed:
                    # exp 39: T_jk(T_ij(z)) ~= T_ik(z). Small-batch differentiable
                    # two-step rollout on the SAME real source endpoints.
                    i = torch.randint(0, data.num_styles, (len(mz0),), device=device,
                                      generator=g_noise)
                    j = (i + 1 + torch.randint(0, data.num_styles - 1, (len(mz0),),
                                              device=device, generator=g_noise)) % data.num_styles
                    k = (j + 1 + torch.randint(0, data.num_styles - 1, (len(mz0),),
                                              device=device, generator=g_noise)) % data.num_styles
                    pa2, pb2 = torch.minimum(i, j), torch.maximum(i, j)
                    pa3, pb3 = torch.minimum(j, k), torch.maximum(j, k)
                    pad, pbd = torch.minimum(i, k), torch.maximum(i, k)
                    hstep2 = 0.5
                    with torch.autocast("cuda", enabled=False):
                        def _one(model_, z_, t_, a_, b_):
                            return mixed_velocity(model_, z_, t_, a_, b_, data.stats,
                                                  args.eps, args.objective)
                        t0 = torch.full((len(mz0),), 0.25, device=device, dtype=mz0.dtype)
                        t1 = torch.full((len(mz0),), 0.75, device=device, dtype=mz0.dtype)
                        zj = mz0 + hstep2 * _one(model, mz0, t0, pa2, pb2)
                        zjk = zj + hstep2 * _one(model, zj, t1, pa3, pb3)
                        zik = mz0 + hstep2 * _one(model, mz0, t0, pad, pbd) \
                            + hstep2 * _one(model, mz0 + hstep2 * _one(model, mz0, t0, pad, pbd),
                                            t1, pad, pbd)
                        ltri = torch.nn.functional.mse_loss(zjk, zik.detach())
                    loss = loss + args.triangle_weight * ltri
                    r = {**r, "triangle": float(ltri)}

                if dist_loss is not None:
                    hstep = 1.0 / max(1, args.ode_steps_train)
                    with torch.autocast("cuda", enabled=False):
                        if mixed:
                            zs = mz0.detach()
                            # Chunked differentiable ODE: keep the full batch for the
                            # bridge loss but cap the graph held by L_dist at 64 samples.
                            # Each chunk is an independent, complete ODE rollout, then the
                            # SWD is accumulated so the total gradient equals the sum.
                            dist_chunk = min(64, args.ode_microbatch or 64)
                            zs_out = []
                            for cs in range(0, len(zs), dist_chunk):
                                zc = zs[cs:cs + dist_chunk]
                                for k in range(args.ode_steps_train):
                                    tk = torch.full((len(zc),), k * hstep, device=zc.device,
                                                    dtype=zc.dtype)
                                    zc = zc + hstep * mixed_velocity(
                                        model, zc, tk, mpa[cs:cs + dist_chunk],
                                        mpb[cs:cs + dist_chunk], data.stats, args.eps,
                                        args.objective)
                                zs_out.append(zc)
                            zs_end = torch.cat(zs_out, 0)
                            # group by TARGET style: each target domain must be matched
                            # separately, never merged into one overall SWD
                            ld = dist_loss(zs_end.float(), mz1.float(), seed=step, groups=mpb)
                        else:
                            from .online_bridge import both_drifts
                            zs = z0.detach()
                            for k in range(args.ode_steps_train):
                                tk = torch.full((len(zs),), k * hstep, device=zs.device,
                                                dtype=zs.dtype)
                                bp_, bm_ = both_drifts(model, zs, tk, a, b, data.stats,
                                                       args.eps, 0.002)
                                zs = zs + hstep * 0.5 * (bp_ - bm_)
                            ld = dist_loss(zs.float(), z1.float(), seed=step,
                                           groups=torch.full((len(zs),), b, device=zs.device,
                                                             dtype=torch.long))
                    loss = loss + args.lambda_dist * ld
                    r = {**r, "dist": float(ld)}
                    dist_seen["grad_nonzero"] = float(ld) != 0.0
            (loss * (mb / args.batch)).backward()
            total += float(loss) * (mb / args.batch)
            rec = r
        if dist_loss is not None and step == 1:
            assert dist_seen.get("grad_nonzero"), (
                "dist_loss was enabled but contributed an exactly-zero value on step 1; "
                "the mixed-batch path is not actually adding L_dist")
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched_lr.step()
        with torch.no_grad():
            decay = min(args.ema_decay, (step + 1) / (step + 10))
            for pe, pv in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pv, 1 - decay)

        if step % args.log_interval == 0 or step == start_step + 1:
            dt = time.perf_counter() - t_start
            print(f"step={step}/{args.steps} {pair_tag} loss={total:.4f} "
                  f"f={rec.get('f_mse', -1):.3f} b={rec.get('b_mse', -1):.3f} "
                  f"grad={float(grad):.3f} lr={opt.param_groups[0]['lr']:.2e} "
                  f"elapsed={dt:.0f}s", flush=True)
            hist.append({"step": step, "loss": total, **rec,
                         "grad": float(grad), "elapsed": round(dt, 1)})
        if step % args.save_interval == 0 or step == end_step:
            save(step)
        # Equivalent-exposure checkpoints: the equiv3000 file is a FULL resumable
        # payload so every fork shares model/EMA/optimizer/scheduler/RNG state.
        eq = _equiv(step)
        if eq in save_equiv and eq not in saved_equiv:
            save(step, tag=f"equiv{eq}")
            saved_equiv.add(eq)

    wall = time.perf_counter() - t_start
    summary = record_experiment(args.experiment_id, vars(args), {
        "n_params": npars, "styles": data.styles, "counts": data.counts,
        "unordered_pairs": [list(x) for x in pairs],
        "batch": args.batch, "updates": args.steps,
        "endpoint_exposures": args.steps * args.batch * 2,
        "train_seconds": round(wall, 1),
        "sec_per_update": round(wall / max(args.steps, 1), 4),
        "peak_vram_mib": round(torch.cuda.max_memory_allocated(0) / 1048576, 1) if device == "cuda" else 0,
        "loss_history": hist,
    })
    (out / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
    print(f"[train5] done in {wall:.0f}s ({summary['sec_per_update']:.3f}s/update), "
          f"peak={summary['peak_vram_mib']} MiB -> {out}", flush=True)


if __name__ == "__main__":
    main()
