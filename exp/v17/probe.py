"""P0: model-internal condition probe + no-training CFG arms.

All quantities are NETWORK ESTIMATES of the conditional response, never the
theoretical optimal velocity difference (plan §0.1).  t=1 is endpoint
diagnostics only; no 1/(1-t) formula is applied there.
"""
from __future__ import annotations

import torch

from .common import STYLES, autocast, save
from .eval750 import evaluate_arm, make_latents
from exp.v16 import load_base, tasks_and_sources

T_GRID = [0., .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, 1.]


@torch.no_grad()
def t_curve(pool_z: torch.Tensor, pool_sid: torch.Tensor, device='cuda',
            n_per_class=20, seed=20261117):
    """R_style(t), correct/wrong FM MSE, supervised-direction cosine vs t."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.cat([torch.randperm((pool_sid == s).nonzero(as_tuple=True)[0],
                                    generator=g)[:n_per_class]
                     for s in range(5)])
    z_real = pool_z[idx].to(device)
    sid = pool_sid[idx].to(device)
    eps = torch.randn(z_real.shape, generator=g).to(device)
    model, _ = load_base()
    rows = []
    for t in T_GRID:
        zt = t * z_real + (1 - t) * eps
        tau = torch.full((len(zt),), t, device=device)
        vs = []
        for s in range(6):                      # 0..4 labels + unconditional
            lab = sid if s < 5 else sid         # label value unused for uncond
            with autocast(device):
                v = model(zt, lab, tau, unconditional=(s == 5)).float()
            vs.append(v)
        v_star = z_real - eps
        v_uncond = vs[5]
        row = {'t': t,
               'r_style_norm_ratio': float((vs[0] - v_uncond).norm() /
                                           v_uncond.norm().clamp_min(1e-8)),
               'per_label_norm_ratio': [float((vs[s] - v_uncond).norm() /
                                              v_uncond.norm().clamp_min(1e-8))
                                        for s in range(5)],
               'fm_mse_correct': float((vs[0] - v_star).square().mean()),
               'fm_mse_wrong': float((vs[4] - v_star).square().mean()),
               'cos_supervised_alignment': float(torch.nn.functional.cosine_similarity(
                   (vs[0] - v_uncond).flatten(1),
                   (v_star - v_uncond).flatten(1), dim=1).mean()),
               'per_target_cos': [float(torch.nn.functional.cosine_similarity(
                   (vs[s] - v_uncond).flatten(1),
                   (v_star - v_uncond).flatten(1), dim=1).mean())
                   for s in range(5)],
               'endpoint_note': 't=1 diagnostics only; no 1/(1-t) rescale'
               if t == 1. else None}
        rows.append(row)
        print(f"P0 t={t}: R={row['r_style_norm_ratio']:.4f} "
              f"cos={row['cos_supervised_alignment']:.4f}", flush=True)
    del model
    torch.cuda.empty_cache()
    return rows


@torch.no_grad()
def label_swap(base_z: torch.Tensor, sid: torch.Tensor, tid_fixed: torch.Tensor,
               eps: torch.Tensor, tau0: float, device='cuda', batch=32):
    """Generate each task under all five labels with fixed eps; returns
    generated latents (5, N, 4, 64, 64).  Caller scores CSD confusion."""
    model, _ = load_base()
    outs = torch.empty(5, len(base_z), 4, 64, 64, device='cpu')
    for s in range(5):
        for st in range(0, len(base_z), batch):
            sl = slice(st, st + batch)
            tau = torch.full((len(base_z[sl]),), tau0, device=device)
            zt = tau0 * base_z[sl].to(device) + (1 - tau0) * eps[sl].to(device)
            lab = torch.full((len(base_z[sl]),), s, device=device,
                             dtype=torch.long)
            with autocast(device):
                pass
            dt = (1 - tau0) / 20
            zz = zt
            for k in range(20):
                tt = torch.full((len(zz),), tau0 + k * dt, device=device)
                with autocast(device):
                    v = model(zz, lab, tt).float()
                zz = zz + dt * v
            outs[s, sl] = zz.cpu()
    del model
    torch.cuda.empty_cache()
    return outs


def cfg_op(w, w_schedule=None, steps=20):
    """Operator factory: CFG generation on top of the frozen base."""
    model_holder = {}

    def op(group, z, sid, tid, eps):
        if 'm' not in model_holder:
            model_holder['m'], _ = load_base()
        return _cfg_sample(model_holder['m'], z, sid, tid, eps, .7 if
                           w_schedule is not False else .7, steps=steps,
                           w=w, w_schedule=w_schedule)
    return op


def run_cfg_arms(arms: dict, out_root=None, device='cuda'):
    """arms: name -> (w, w_schedule or None, tau0).  Each full 750."""
    from exp.v16 import tasks_and_sources
    tasks, source = tasks_and_sources()
    results = {}
    for name, (w, sched, tau0) in arms.items():
        def op(group, z, sid, tid, eps, _w=w, _sched=sched, _tau0=tau0):
            model, _ = load_base()
            zt = _cfg_sample(model, z, sid, tid, eps, _tau0, w=_w,
                             w_schedule=_sched)
            del model
            return zt
        results[name] = evaluate_arm(name, op, tau0=tau0, out_root=out_root,
                                     extra_meta={'cfg_w': w})
        print(f"P0 {name}: cross top1={results[name]['cross600']['csd_top1']:.4f}",
              flush=True)
    return results
