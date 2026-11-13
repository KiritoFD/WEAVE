"""Registered inference variants. Each entry is ckpt -> latent generator.

Coverage of the variants that actually exist in this repository:
  direct      Q2a / B3 / B4 style one-shot C3UNet mapping
  relative    z_s + F(c, t) - F(c, s)
  oracle      direct start + one R1 target-margin step
  r1_steps    n-step R1 edit from a chosen start (full / topk)
  energy      n-step edit with a registered style energy
  student     trained corrector checkpoint (delta / policy / capacity)
  identity    source passed through unchanged (control)
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from . import (Generator, InferSpec, canonicalizer, load_c3unet, load_critic,
               register, REPO)


def _start(model, canon, canon_mode, z, tid):
    c = canon(z.float(), mode=canon_mode, augment=False)
    return model(c, 0.0, c, tid), c


def _r1_margin(critic, z, tid):
    q = critic(z.float()).float()
    onehot = F.one_hot(tid, 5).bool()
    others = q.masked_fill(onehot, torch.finfo(q.dtype).min)
    return q.gather(1, tid[:, None]).squeeze(1) - torch.logsumexp(others, 1)


def _normalize(g):
    rms = g.flatten(1).pow(2).mean(1).sqrt().clamp_min(1e-12)
    return g / rms.view(-1, 1, 1, 1)


@register("random")
def _random(spec, device):
    """Target-independent random perturbation control (same seed => same noise)."""
    from . import REPO
    import hashlib
    base, _ = load_c3unet(spec.ckpt or REPO / "runs/v6/P1_B4_distill50/P1_B4_distill50.pt", device) \
        if spec.ckpt else (None, None)
    canon = canonicalizer(device)
    def gen(z, sid, tid, tasks):
        with torch.no_grad():
            if base is not None:
                z0, _ = _start(base, canon, spec.canon_mode, z, tid)
            else:
                z0 = z
            noise = []
            for t in tasks:
                seed = int(hashlib.sha256(
                    f"{t['source_stem']}:{spec.seed}".encode()).hexdigest()[:15], 16)
                g = torch.Generator().manual_seed(seed)
                noise.append(torch.randn(z.shape[1:], generator=g))
            n = torch.stack(noise).to(z.device)
            out = z0 + spec.step_rms * _normalize(n)
        return {"z0": z0, "final": out, "trace": []}
    return gen


@register("identity")
def _identity(spec, device):
    def gen(z, sid, tid, tasks):
        return {"z0": z, "final": z, "trace": []}
    return gen


@register("direct")
def _direct(spec, device):
    if spec.ckpt is None:
        raise ValueError("direct requires --ckpt")
    model, _ = load_c3unet(spec.ckpt, device)
    canon = canonicalizer(device)
    def gen(z, sid, tid, tasks):
        with torch.no_grad():
            out, c = _start(model, canon, spec.canon_mode, z, tid)
        return {"z0": out, "final": out, "content": c, "trace": []}
    return gen


@register("relative")
def _relative(spec, device):
    if spec.ckpt is None:
        raise ValueError("relative requires --ckpt")
    model, _ = load_c3unet(spec.ckpt, device)
    canon = canonicalizer(device)
    def gen(z, sid, tid, tasks):
        with torch.no_grad():
            zt, _ = _start(model, canon, spec.canon_mode, z, tid)
            zs, _ = _start(model, canon, spec.canon_mode, z, sid)
            out = (z + zt - zs).detach()
        return {"z0": zt, "final": out, "trace": []}
    return gen


def _critic_for(spec, device):
    """R1 critic, optionally swapped for a trained checkpoint (e.g. C3).

    extra['critic_ckpt'] is a resolved .pt path whose file holds
    {'model': state_dict} — the same layout load_critic already accepts.
    """
    path = spec.extra.get("critic_ckpt")
    return load_critic(device, Path(path) if path else None)


def _support_banks(spec, device, critic):
    """Load the real-painting support bank(s) required by this spec.

    Returns (global_support, local_support); either may be None. Raises rather
    than silently degrading when lambda_ratio demands a missing bank.
    """
    from exp.v8.energy import RealSupport, SpatialSupport
    lam = float(spec.extra.get("lambda_ratio", 0.0))
    local = bool(spec.extra.get("local_support", False))
    glob = loc = None
    if lam and not local:
        sp = REPO / "runs/v8/real_support.pt"
        if not sp.exists():
            raise FileNotFoundError(f"lambda_ratio={lam} needs {sp}")
        st = torch.load(sp, map_location="cpu", weights_only=False)
        glob = RealSupport(st["prototypes"].to(device), st["sigma"].to(device))
    if lam and local:
        sp = REPO / "runs/v8/local_support.pt"
        if not sp.exists():
            raise FileNotFoundError(f"local_support needs {sp}")
        st = torch.load(sp, map_location="cpu", weights_only=False)
        loc = SpatialSupport(st["prototypes"].to(device), st["sigma"].to(device),
                             grid=int(st.get("grid", 4)))
    return glob, loc


def _content_feat(c):
    """Content-conditioning features for A-route policies: channel-wise
    mean + std of the canonicalized source latent. Shape-agnostic and cheap;
    identical computation in train.py and here so train/infer match."""
    c = c.float()
    return torch.cat([c.mean(dim=(2, 3)), c.std(dim=(2, 3))], dim=1)


def _load_policy(ckpt_path: str, device: str):
    """Load a trained A-route control policy.

    train.py saves {'policy': MLP state_dict, 'spec': ControlSpec dict,
    'feat_dim': int, ...}. The MLP is Linear(feat_dim+5 -> 256) -> SiLU ->
    Linear(256 -> control_dim); its input is [content features of the
    canonicalized source, 5-way target onehot]. `feat_dim` falls back to the
    control dim for checkpoints trained before content conditioning.
    """
    import torch.nn as nn
    from exp.v8.controls import ControlSpec
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["spec"]
    cspec = ControlSpec(global_dim=sd.get("global_dim", 32),
                        per_scale_dim=sd.get("per_scale_dim", 32),
                        scales=tuple(sd.get("scales", ("64", "32", "16", "out"))),
                        palette=sd.get("palette", True),
                        local_rank=sd.get("local_rank", 0))
    dim = cspec.dim
    feat_dim = int(ck.get("feat_dim", dim))
    policy = nn.Sequential(nn.Linear(feat_dim + 5, 256), nn.SiLU(),
                           nn.Linear(256, dim)).to(device).eval()
    policy.load_state_dict(ck["policy"])
    for p in policy.parameters():
        p.requires_grad_(False)
    return policy, cspec


def _policy_u0(policy, cspec, tid, device, feat=None):
    b = tid.shape[0]
    in_dim = policy[0].in_features - 5
    if feat is None:
        feat = torch.zeros(b, in_dim, device=device)
    oh = F.one_hot(tid.long(), 5).float()
    with torch.no_grad():
        return policy(torch.cat([feat.to(device), oh], 1))


@register("oracle")
def _oracle(spec, device):
    if spec.ckpt is None:
        raise ValueError("oracle requires --ckpt")
    model, _ = load_c3unet(spec.ckpt, device)
    canon = canonicalizer(device)
    critic = _critic_for(spec, device)
    steps = int(spec.extra.get("oracle_steps", 1))
    total_rms = float(spec.extra.get("oracle_rms", 0.25))
    per = total_rms / max(steps, 1)
    def gen(z, sid, tid, tasks):
        with torch.enable_grad():
            z0, _ = _start(model, canon, spec.canon_mode, z, tid)
            cur = z0.detach()
            for _ in range(steps):
                cur = cur.detach().clone().requires_grad_(True)
                g = torch.autograd.grad(_r1_margin(critic, cur, tid).sum(), cur)[0]
                cur = (cur + per * _normalize(g)).detach()
        return {"z0": z0, "final": cur, "trace": []}
    return gen


@register("r1_steps")
def _r1_steps(spec, device):
    """R1 edit from a direct start. full = all patches, topk = top fraction."""
    if spec.ckpt is None:
        raise ValueError("r1_steps requires --ckpt for the start model")
    model, _ = load_c3unet(spec.ckpt, device)
    canon = canonicalizer(device)
    critic = _critic_for(spec, device)
    frac = float(spec.extra.get("topk_frac", 0.0))
    def gen(z, sid, tid, tasks):
        trace = []
        with torch.enable_grad():
            z0, _ = _start(model, canon, spec.canon_mode, z, tid)
            cur = z0.detach()
            for step in range(spec.steps):
                cur = cur.detach().clone().requires_grad_(True)
                g = torch.autograd.grad(_r1_margin(critic, cur, tid).sum(), cur)[0]
                if frac > 0:
                    # Same rule as tools/v6_next/w_gating.topk_spatial_mask:
                    # per-image SPATIAL top-k on channel-summed squared
                    # gradient energy — NOT signed elementwise topk.
                    energy = g.detach().square().sum(1)          # (B,H,W)
                    bh, bw = energy.shape[-2:]
                    k = max(1, int(round(frac * bh * bw)))
                    idx = energy.flatten(1).topk(k, dim=1).indices
                    mask = torch.zeros_like(energy.flatten(1))
                    mask.scatter_(1, idx, 1.0)
                    g = g * mask.view(-1, 1, bh, bw).to(g.dtype)
                d = _normalize(g)
                cur = (cur + spec.step_rms * d).detach()
                trace.append({"step": step,
                              "displacement_rms": float(
                                  (cur - z0).flatten(1).pow(2).mean(1).sqrt().mean())})
        return {"z0": z0, "final": cur, "trace": trace}
    return gen


@register("control")
def _control(spec, device):
    """A: optimise low-dimensional generator controls against the R1 margin.

    z(u) = G(c, t; u); the gradient grad_u F = J_u^T grad_z F arrives through the
    generator, so the update lives inside the transform family Q2a already
    learned. u starts at exactly zero, so step 0 reproduces the checkpoint.
    """
    from exp.v8.controls import ControlSpec, Controls, wrap
    from exp.v8.guidance import target_margin
    from exp.v8.energy import SupportedEnergy, RealSupport

    base_path = spec.ckpt or (REPO / "runs/v6/Q2a_embed_style/Q2a_embed_style.pt")
    base, _ = load_c3unet(base_path, device)
    canon = canonicalizer(device)
    critic = _critic_for(spec, device)
    spec_map = {"global": ControlSpec(palette=False, scales=()),
                "hier": ControlSpec(palette=False),
                "hier_palette": ControlSpec(),
                "hier_palette_local": ControlSpec(local_rank=2)}
    # A trained control policy overrides BOTH the control spec and the u init:
    # its saved spec decides the control space, and its output warm-starts u.
    policy = None
    ck_path = spec.extra.get("control_ckpt")
    if ck_path:
        policy, ck_spec = _load_policy(ck_path, device)
        cspec = ck_spec
        gen = wrap(base, cspec, device)
    else:
        cspec = spec_map[spec.extra.get("spec", "hier_palette")]
        gen = wrap(base, cspec, device)
    lam = float(spec.extra.get("lambda_ratio", 0.0))
    glob_sup, loc_sup = _support_banks(spec, device, critic)
    support = loc_sup if (loc_sup is not None and spec.extra.get("local_support")) else glob_sup
    scorer = SupportedEnergy(critic, support, lam,
                             local_support=bool(spec.extra.get("local_support", False)) and loc_sup is not None)

    def gen_fn(z, sid, tid, tasks):
        with torch.enable_grad():
            c = canon(z.float(), mode=spec.canon_mode, augment=False)
            feat = _content_feat(c) if policy is not None else None
            u0 = _policy_u0(policy, cspec, tid, device, feat=feat) \
                if policy is not None \
                else torch.zeros(z.shape[0], cspec.dim, device=device)
            u = u0.detach().clone().requires_grad_(True)
            with torch.no_grad():
                z0 = gen.base(c, 0.0, c, tid)
            for _ in range(spec.steps):
                ur = u.detach().clone().requires_grad_(True)
                out = gen(c, 0.0, c, tid, controls=Controls.from_tensor(ur, cspec))
                v = scorer.value(out, tid)
                g = torch.autograd.grad(v.sum(), ur)[0]
                gr = g / g.flatten(1).pow(2).mean(1).sqrt().clamp_min(1e-12).view(-1, 1)
                u = (ur + spec.step_rms * gr).detach().clamp(-spec.rho, spec.rho)
            with torch.no_grad():
                final = gen(c, 0.0, c, tid, controls=Controls.from_tensor(u, cspec))
        return {"z0": z0, "final": final, "u": u,
                "trace": [{"control_dim": cspec.dim, "steps": spec.steps}]}
    return gen_fn


@register("supported")
def _supported(spec, device):
    """C: one R1 margin step with an optional real-painting support term."""
    from exp.v8.energy import RealSupport, SupportedEnergy
    base_path = spec.ckpt or (REPO / "runs/v6/Q2a_embed_style/Q2a_embed_style.pt")
    base, _ = load_c3unet(base_path, device)
    canon = canonicalizer(device)
    critic = _critic_for(spec, device)
    lam = float(spec.extra.get("lambda_ratio", 0.1))
    glob_sup, loc_sup = _support_banks(spec, device, critic)
    support = loc_sup if (loc_sup is not None and spec.extra.get("local_support")) else glob_sup
    scorer = SupportedEnergy(
        critic, support, lam,
        local_support=bool(spec.extra.get("local_support", False)) and loc_sup is not None)

    def gen_fn(z, sid, tid, tasks):
        with torch.enable_grad():
            c = canon(z.float(), mode=spec.canon_mode, augment=False)
            z0 = base(c, 0.0, c, tid)
            cur = z0.detach()
            for _ in range(spec.steps):
                g, _ = scorer.gradient(cur, tid)
                cur = cur + spec.rho * g
            return {"z0": z0, "final": cur, "trace": []}
    return gen_fn


@register("spd_edit")
def _spd_edit(spec, device):
    """B-u: K-step SPD-preconditioned R1 edit inside the generator control space.

    u_{k+1} = clamp(u_k + step_rms * normalize(P_theta g_u), +/-rho) with
    g_u = grad_u margin_R1(G(c,t;u), t). Optional control_ckpt warm-starts u
    from a trained A-route policy (the I2/I3 joint arms).
    """
    from exp.v8.controls import ControlSpec, Controls, wrap
    from exp.v8.guidance import ControlSPDHead, generate_control_edit
    base_path = spec.ckpt or (REPO / "runs/v6/Q2a_embed_style/Q2a_embed_style.pt")
    base, _ = load_c3unet(base_path, device)
    canon = canonicalizer(device)
    critic = _critic_for(spec, device)

    policy = None
    ck_path = spec.extra.get("control_ckpt")
    if ck_path:
        policy, cspec = _load_policy(ck_path, device)
    else:
        spec_map = {"global": ControlSpec(palette=False, scales=()),
                    "hier": ControlSpec(palette=False),
                    "hier_palette": ControlSpec(),
                    "hier_palette_local": ControlSpec(local_rank=2)}
        cspec = spec_map[spec.extra.get("spec", "hier_palette")]
    sp = spec.extra.get("spd_ckpt")
    if not sp:
        raise ValueError("spd_edit requires extra['spd_ckpt']")
    ck = torch.load(sp, map_location="cpu", weights_only=False)
    head = ControlSPDHead(dim=cspec.dim, rank=ck.get("rank", 2)).to(device).eval()
    head.load_state_dict(ck["head"])
    for p in head.parameters():
        p.requires_grad_(False)

    def gen_fn(z, sid, tid, tasks):
        feat = None
        if policy is not None:
            with torch.no_grad():
                c = canon(z.float(), mode="c1_gentle", augment=False)
            feat = _content_feat(c)
        u0 = _policy_u0(policy, cspec, tid, device, feat=feat) \
            if policy is not None else None
        return generate_control_edit(critic, head, base, canon, cspec, z, tid,
                                     steps=spec.steps, step_rms=spec.step_rms,
                                     rho=spec.rho, u0=u0, device=device)
    return gen_fn


@register("actuated")
def _actuated(spec, device):
    """F: trainable style actuation — ctrl = A_phi(u, feat(c), t) through the
    frozen B4 control space, with per-instance u optimised by the same inner
    R1-margin loop used in training (steps from spec)."""
    from exp.v8.actuation import StyleActuator
    from exp.v8.controls import ControlSpec, Controls, wrap
    from exp.v8.guidance import target_margin
    base_path = spec.ckpt or (REPO / "runs/v6/Q2a_embed_style/Q2a_embed_style.pt")
    base, _ = load_c3unet(base_path, device)
    canon = canonicalizer(device)
    critic = _critic_for(spec, device)

    ck_path = spec.extra.get("act_ckpt")
    if not ck_path:
        raise ValueError("actuated requires extra['act_ckpt']")
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    sd = ck["spec"]
    cspec = ControlSpec(global_dim=sd.get("global_dim", 32),
                        per_scale_dim=sd.get("per_scale_dim", 32),
                        scales=tuple(sd.get("scales", ("64", "32", "16", "out"))),
                        palette=sd.get("palette", True),
                        local_rank=sd.get("local_rank", 0))
    actuator = StyleActuator(int(ck["u_dim"]), int(ck["feat_dim"]),
                             cspec.dim).to(device).eval()
    actuator.load_state_dict(ck["actuator"])
    for p in actuator.parameters():
        p.requires_grad_(False)
    gen_w = wrap(base, cspec, device)
    u_dim = int(ck["u_dim"])
    eta = float(spec.extra.get("eta", 0.05))

    def gen_fn(z, sid, tid, tasks):
        with torch.enable_grad():
            c = canon(z.float(), mode=spec.canon_mode, augment=False)
            with torch.no_grad():
                z0 = base(c, 0.0, c, tid)
                feat = _content_feat(c)
            u = torch.zeros(z.shape[0], u_dim, device=device,
                            requires_grad=True)
            ctrl = None
            for _ in range(spec.steps):
                ctrl = actuator(u, feat, tid)
                out = gen_w(c, 0.0, c, tid,
                            controls=Controls.from_tensor(ctrl, cspec))
                m = target_margin(critic, out, tid)
                g = torch.autograd.grad(m.sum(), u)[0]
                gr = g / g.flatten(1).pow(2).mean(1).sqrt() \
                    .clamp_min(1e-12).view(-1, 1)
                u = (u + eta * gr).detach().requires_grad_(True)
            with torch.no_grad():
                ctrl = actuator(u, feat, tid)
                final = gen_w(c, 0.0, c, tid,
                              controls=Controls.from_tensor(ctrl, cspec))
        return {"z0": z0, "final": final, "u": ctrl,
                "trace": [{"actuator": True, "u_dim": u_dim,
                           "steps": spec.steps}]}
    return gen_fn


@register("spd_latent")
def _spd_latent(spec, device):
    """B-z: K-step SPD-preconditioned R1 edit directly on the latent.

    Replicates UnsupervisedUnroll's latent branch at inference time:
    g = grad_z margin; delta = P_theta g; z <- project_rms(z + rms*norm(delta)).
    """
    from exp.v8.guidance import SPDHead, normalize, project_rms, target_margin
    base_path = spec.ckpt or (REPO / "runs/v6/Q2a_embed_style/Q2a_embed_style.pt")
    model, _ = load_c3unet(base_path, device)
    canon = canonicalizer(device)
    critic = _critic_for(spec, device)
    sp = spec.extra.get("spd_ckpt")
    if not sp:
        raise ValueError("spd_latent requires extra['spd_ckpt']")
    ck = torch.load(sp, map_location="cpu", weights_only=False)
    head = SPDHead(rank=int(ck.get("rank", 0))).to(device).eval()
    head.load_state_dict(ck["head"])
    for p in head.parameters():
        p.requires_grad_(False)

    def spd_apply(diag, u, g):
        dg = diag * g
        ug = (u * g.unsqueeze(1)).sum(2)
        return dg + (u * ug.unsqueeze(2)).sum(1)

    eta_search = bool(spec.extra.get("eta_search", False))
    budget_eps = float(spec.extra.get("budget_eps", 0.15))
    etas = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

    def gen_fn(z, sid, tid, tasks):
        with torch.no_grad():
            c = canon(z.float(), mode=spec.canon_mode, augment=False)
            z0 = model(c, 0.0, c, tid).detach()
        cur = z0.clone()
        for _ in range(spec.steps):
            with torch.enable_grad():
                zr = cur.detach().clone().requires_grad_(True)
                g = torch.autograd.grad(
                    target_margin(critic, zr.float(), tid).sum(), zr)[0].detach()
                diag, u = head(c, cur, tid, g)
                delta = spd_apply(diag, u, g)
            step = normalize(delta)
            if eta_search:
                # eta* = largest step whose edit stays inside the per-sample
                # content budget AND actually raises the R1 margin.
                d_gen = (z0 - z).square().flatten(1).mean(1).clamp_min(1e-8)
                m_cur = target_margin(critic, cur.float(), tid)
                chosen = cur.clone()
                remaining = torch.ones(cur.shape[0], dtype=torch.bool,
                                       device=cur.device)
                for mult in sorted(etas, reverse=True):
                    cand = project_rms(
                        cur + spec.step_rms * mult * step, z0, spec.rho)
                    cost = (cand - z0).square().flatten(1).mean(1) / d_gen
                    with torch.no_grad():
                        m_cand = target_margin(critic, cand.float(), tid)
                    ok = ((cost <= budget_eps) & (m_cand > m_cur)
                          & remaining)
                    if ok.any():
                        chosen = torch.where(ok[:, None, None, None], cand, chosen)
                        remaining &= ~ok
                    if not remaining.any():
                        break
                cur = chosen.detach()
            else:
                cur = project_rms(cur + spec.step_rms * step,
                                  z0, spec.rho).detach()
        return {"z0": z0, "final": cur, "trace": []}
    return gen_fn


@register("student")
def _student(spec, device):
    """Trained corrector checkpoint produced by exp/v8 training."""
    if spec.ckpt is None:
        raise ValueError("student requires --ckpt")
    model, _ = load_c3unet(spec.ckpt, device) if spec.extra.get("start_ckpt") else (None, None)
    canon = canonicalizer(device)
    ck = torch.load(spec.ckpt, map_location="cpu", weights_only=False)
    from exp.v8.models import build_corrector
    corrector = build_corrector(ck).to(device).eval()
    start_ck = spec.extra.get("start_ckpt") or spec.extra.get("b4_ckpt")
    if start_ck is None:
        raise ValueError("student requires extra['start_ckpt'] for its frozen start")
    base, _ = load_c3unet(Path(start_ck), device) if isinstance(start_ck, (str, Path)) else (start_ck, None)
    def gen(z, sid, tid, tasks):
        with torch.no_grad():
            z0, c = _start(base, canon, spec.canon_mode, z, tid)
            out = corrector(c, z0, sid, tid)
        return {"z0": z0, "final": out, "trace": []}
    return gen
