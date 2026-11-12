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


@register("oracle")
def _oracle(spec, device):
    if spec.ckpt is None:
        raise ValueError("oracle requires --ckpt")
    model, _ = load_c3unet(spec.ckpt, device)
    canon = canonicalizer(device)
    critic = load_critic(device)
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
    critic = load_critic(device)
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
                    flat = g.flatten(1)
                    k = max(1, int(flat.shape[1] * frac))
                    thresh = flat.topk(k, dim=1).values[:, -1:]
                    g = g * (flat >= thresh).view_as(g).float()
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
    critic = load_critic(device)
    spec_map = {"global": ControlSpec(palette=False, scales=()),
                "hier": ControlSpec(palette=False),
                "hier_palette": ControlSpec(),
                "hier_palette_local": ControlSpec(local_rank=2)}
    cspec = spec_map[spec.extra.get("spec", "hier_palette")]
    gen = wrap(base, cspec, device)
    lam = float(spec.extra.get("lambda_ratio", 0.0))
    support = None
    if lam:
        sp = REPO / "runs/v8/real_support.pt"
        if not sp.exists():
            raise FileNotFoundError(
                f"lambda_ratio={lam} needs the real support bank at {sp}")
        st = torch.load(sp, map_location="cpu", weights_only=False)
        support = RealSupport(st["prototypes"].to(device), st["sigma"].to(device))
    scorer = SupportedEnergy(critic, support, lam)

    def gen_fn(z, sid, tid, tasks):
        with torch.enable_grad():
            c = canon(z.float(), mode=spec.canon_mode, augment=False)
            u = torch.zeros(z.shape[0], cspec.dim, device=device, requires_grad=True)
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
    critic = load_critic(device)
    lam = float(spec.extra.get("lambda_ratio", 0.1))
    support = None
    if lam:
        sp = REPO / "runs/v8/real_support.pt"
        if not sp.exists():
            raise FileNotFoundError(f"support bank missing: {sp}")
        st = torch.load(sp, map_location="cpu", weights_only=False)
        support = RealSupport(st["prototypes"].to(device), st["sigma"].to(device))
    scorer = SupportedEnergy(critic, support, lam)

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
