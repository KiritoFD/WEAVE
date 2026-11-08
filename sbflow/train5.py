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
                              endpoint_coefficients, moment_coefficients,
                              normalized_endpoint_target, normalized_target,
                              normalized_velocity_target, precondition_input,
                              reconstruct_velocity)
from .time_sampling import sample_time


# --------------------------------------------------------------------------- #
def build_model(args, channels, image_size, num_styles) -> BidirDiT:
    return BidirDiT(
        channels=channels, image_size=image_size, patch=args.patch,
        width=args.width, depth=args.depth, heads=args.heads,
        num_styles=num_styles, style_cond=args.style_cond,
        heads_out=(2 if (args.objective in ("bidir", "bidir_joint") and args.heads_out == 2)
                   else (2 if args.objective == "score_current" else 1)),
        two_head=("score_current" if args.objective == "score_current" else "drift"),
        local_conv=args.local_conv, pair_rank=args.pair_rank,
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


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-id", required=True)
    p.add_argument("--caches", nargs="+", required=True, help="one DCAE cache per style")
    p.add_argument("--output", required=True)
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

    p.add_argument("--objective", choices=["bidir", "bidir_joint", "single_velocity", "score_current"],
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

    p.add_argument("--log-interval", type=int, default=20)
    p.add_argument("--save-interval", type=int, default=200)
    p.add_argument("--resume", default="")

    # --- group D: direct target-distribution supervision ---
    p.add_argument("--dist-loss", choices=["none", "global", "local", "both"], default="none")
    p.add_argument("--style-enc", choices=["learned", "fixed"], default="learned")
    p.add_argument("--lambda-dist", type=float, default=0.1)
    p.add_argument("--style-enc-epochs", type=int, default=60)
    p.add_argument("--ode-steps-train", type=int, default=2,
                   help="short differentiable ODE used to produce endpoints for L_dist")
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

    # --- group F: shared Gaussian pivot ---
    p.add_argument("--pivot", action="store_true",
                   help="train N(0,I)->style flows and compose T_ij = F_j o F_i^-1")
    args = p.parse_args()

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

    data = load_caches([Path(c) for c in args.caches], device, STYLE_NAMES[: len(args.caches)])
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
    def lr_ratio(s):
        if s < warm:
            return max(0.02, (s + 1) / warm)
        v = (s - warm) / max(1, args.steps - warm)
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1, v)))
    sched_lr = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_ratio)

    samplers = [make_sampler(args.sampler, n, g_data) for n in data.counts]
    s_scale = torch.ones(channels, device=device)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # ---- optional group D distribution loss ----
    dist_loss = None
    if args.dist_loss != "none":
        from .style_loss import DistributionLoss, StyleEncoder, train_style_encoder
        enc = None
        if args.style_enc == "learned":
            enc = train_style_encoder(data.z, epochs=args.style_enc_epochs, device=device,
                                      seed=args.seed,
                                      log=lambda m: print(m, flush=True))
            torch.save({"state": enc.state_dict(), "channels": channels},
                       out_style_enc := (out / "style_encoder.pt"))
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
        replay = EndpointReplayBuffer(seed=args.seed) if args.online_replay else None
        online_cfg = {"online_steps": args.online_steps, "online_mode": args.online,
                      "online_direction": args.online_direction, "online_mix": args.online_mix,
                      "online_replay": args.online_replay, "t_floor": args.t_floor
                      if hasattr(args, "t_floor") else 0.002}
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

    def save(step):
        torch.save({
            "model": model.state_dict(), "ema": ema.state_dict(),
            "optimizer": opt.state_dict(), "scheduler": sched_lr.state_dict(), "step": step,
            "style_stats": data.stats.state_dict(), "styles": data.styles,
            "model_kwargs": dict(channels=channels, image_size=H, patch=args.patch,
                                 width=args.width, depth=args.depth, heads=args.heads,
                                 num_styles=data.num_styles, style_cond=args.style_cond,
                                 heads_out=(2 if args.objective == "score_current" else args.heads_out),
                                 two_head=("score_current" if args.objective == "score_current"
                                           else "drift"),
                                 local_conv=args.local_conv, pair_rank=args.pair_rank),
            "objective": args.objective, "config": vars(args),
            "score_scale": s_scale.cpu(),
        }, out / f"model_step{step}.pt")

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
    for step in range(1, args.steps + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        a, b = sched[step - 1]
        mb = args.microbatch or args.batch
        total, rec = 0.0, {}
        for _ in range(args.batch // mb):
            z0, z1 = sample_pair_batch(a, b, mb)
            ctx = (torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda"
                   else contextlib.nullcontext())
            with ctx:
                use_online = (online_teacher is not None and step > args.online_warmup)
                if use_online:
                    from .online_bridge import online_loss
                    if not args.online_raw_teacher:
                        # EMA teacher: smoother endpoints, the doc's default (exp 24)
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
                                                      args.eps, args.min_t, g_noise, args, s_scale)
                if dist_loss is not None:
                    from .infer5 import probability_flow_velocity
                    from .online_bridge import both_drifts
                    # Differentiable short ODE from the real source endpoints, then SWD.
                    zs = z0.detach()
                    hstep = 1.0 / max(1, args.ode_steps_train)
                    with torch.autocast("cuda", enabled=False):
                        for k in range(args.ode_steps_train):
                            tk = torch.full((len(zs),), k * hstep, device=zs.device, dtype=zs.dtype)
                            bp_, bm_ = both_drifts(model, zs, tk, a, b, data.stats,
                                                   args.eps, 0.002)
                            zs = zs + hstep * 0.5 * (bp_ - bm_)
                        ld = dist_loss(zs.float(), z1.float(), seed=step)
                    loss = loss + args.lambda_dist * ld
                    r = {**r, "dist": float(ld)}
            (loss * (mb / args.batch)).backward()
            total += float(loss) * (mb / args.batch)
            rec = r
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched_lr.step()
        with torch.no_grad():
            decay = min(args.ema_decay, (step + 1) / (step + 10))
            for pe, pv in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pv, 1 - decay)

        if step % args.log_interval == 0 or step == 1:
            dt = time.perf_counter() - t_start
            print(f"step={step}/{args.steps} pair=({a},{b}) loss={total:.4f} "
                  f"f={rec.get('f_mse', -1):.3f} b={rec.get('b_mse', -1):.3f} "
                  f"grad={float(grad):.3f} lr={opt.param_groups[0]['lr']:.2e} "
                  f"elapsed={dt:.0f}s", flush=True)
            hist.append({"step": step, "loss": total, **rec,
                         "grad": float(grad), "elapsed": round(dt, 1)})
        if step % args.save_interval == 0 or step == args.steps:
            save(step)

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
