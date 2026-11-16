"""P4: Flow-velocity-difference residual post-training F0-F3.

u_phi(z_t, s, t) is a SMALL residual head added to the FROZEN V15 velocity.
F0  v_base_cond + u_s                       (non-centered control)
F1  v_base_cond + u_s - mean_k u_k          (class-centered increment)
F2  v_base_cond + u_target - u_source       (bridge supervision REQUIRED:
     self-paired real data has source==target -> delta == 0 -> no gradient.
     Supervised on P1 teacher pairs at t0=.7: v* = (z_teacher - z_init)/(1-t0)
     over [t0,1]; NOT a real-pair ground truth.  If pairs are missing the arm
     records blocked_invalid_supervision and the queue still runs F0/F1/F3.)
F3  F1 with t in [0,.4] oversampled (early-condition emphasis).

Real training data is class-balanced; 15% unconditional drop preserved.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import OUT, STYLES, autocast, cosine_lr, heartbeat, save, sha
from .checkpoint import find_resume, load_ckpt, save_ckpt


class VelocityResidual(nn.Module):
    """Small FiLM residual head on the noisy latent + label + tau."""

    def __init__(self, width=96):
        super().__init__()
        self.stem = nn.Conv2d(4, width, 3, padding=1)
        self.label_embed = nn.Embedding(6, width)
        self.tau_mlp = nn.Sequential(nn.Linear(1, width), nn.SiLU(True),
                                     nn.Linear(width, width))
        self.blocks = nn.ModuleList()
        for _ in range(3):
            self.blocks.append(nn.Sequential(
                nn.Conv2d(width, width, 3, padding=1, groups=width),
                nn.SiLU(True),
                nn.Conv2d(width, width, 1), nn.SiLU(True)))
        self.out = nn.Conv2d(width, 4, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)                # zero residual at init

    def forward(self, z_t, label, tau):
        x = self.stem(z_t)
        cond = (self.label_embed(label.clamp(0, 5)) +
                self.tau_mlp(tau.view(-1, 1)))
        x = x + cond[:, :, None, None]
        for blk in self.blocks:
            x = x + blk(x)
        return self.out(x)


def centered_output(net, z_t, tid, tau, mode):
    if mode == 'F0':
        return net(z_t, tid, tau)
    n = len(z_t)
    labels = torch.arange(5, device=z_t.device).view(5, 1).expand(5, n) \
        .reshape(-1)
    tau_r = tau.repeat(5)
    z_r = z_t.repeat(5, 1, 1, 1)
    all_u = net(z_r, labels, tau_r).view(5, n, *z_t.shape[1:])
    mean = all_u.mean(0)
    if mode in ('F1', 'F3'):
        return net(z_t, tid, tau) - mean
    raise ValueError(mode)


def fm_residual_loss(net, base_model, z_real, sid, tid, mode, gen,
                     drop=.15, bridge=None, p_early=.6):
    """One optimisation micro-batch.  bridge = P1 pairs for F2 supervision."""
    if mode == 'F2':
        if bridge is None:
            return None, {'blocked_invalid_supervision': True}
        # bridge supervision on [t0,1] with t0=.7 pairs
        n_b = len(bridge['base'])
        idx = torch.randint(0, n_b, (len(z_real),),
                            generator=gen).to(z_real.device)
        z_init = bridge['base'][idx].to(z_real.device)
        z_teach = z_init + bridge['delta'][idx].to(z_real.device)
        eps_b = torch.randn(z_init.shape, generator=gen).to(z_init.device)
        tau0 = .7
        zt_b = tau0 * z_init + (1 - tau0) * eps_b
        v_star_b = (z_teach - zt_b) / (1 - tau0)
        tau_b = torch.full((len(z_real),), tau0 * .5 + .15, device=z_init.device)
        u = net(zt_b, bridge['tid'][idx].to(z_init.device), tau_b)
        if base_model is None:
            return F.mse_loss(u, v_star_b), {'bridge_mse': 1.}
        with autocast():
            v_base_b = base_model(zt_b, bridge['tid'][idx].to(z_init.device),
                                  tau_b).float()
        return F.mse_loss(v_base_b + u, v_star_b), {'bridge_mse': 1.}
    n = len(z_real)
    eps = torch.randn(z_real.shape, generator=gen).to(z_real.device)
    if mode == 'F3' and torch.rand(1, generator=gen).item() < p_early:
        tau = torch.rand(n, generator=gen).to(z_real.device) * .4
    else:
        tau = torch.rand(n, generator=gen).to(z_real.device)
    zt = tau.view(-1, 1, 1, 1) * z_real + (1 - tau.view(-1, 1, 1, 1)) * eps
    v_star = z_real - eps
    drop_mask = torch.rand(n, generator=gen).to(z_real.device) < drop
    label = tid.clone()
    with autocast():
        v_base = base_model(zt, label, tau).float()
    if drop_mask.any():
        with autocast():
            v_base_u = base_model(zt[drop_mask], label[drop_mask], tau[drop_mask],
                                  unconditional=True).float()
        v_base = v_base.clone()
        v_base[drop_mask] = v_base_u
    u = centered_output(net, zt, label, tau, mode)
    pred = v_base + u
    return F.mse_loss(pred, v_star), {'fm_mse': float(F.mse_loss(pred, v_star))}


def train_fm(mode: str, pool_split: dict, out_dir: Path, *,
             bridge_path: Path | None = None, updates=8000, micro=8,
             accum=8, seed=20261117, lr=2e-4):
    from .common import class_balanced_indices
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    net = VelocityResidual().cuda()
    base, _ = __import__('exp.v16', fromlist=['load_base']).load_base()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=.01)
    gen = torch.Generator().manual_seed(seed)
    bridge = None
    if mode == 'F2':
        if not (bridge_path or (OUT / 'pairs' / 'teacher_pairs_T0.pt')).exists():
            save(out_dir / 'blocked.json', {'blocked_invalid_supervision': True})
            return None
        bridge = torch.load(bridge_path or (OUT / 'pairs' / 'teacher_pairs_T0.pt'),
                            map_location='cpu', weights_only=False)['train']
    z_all = pool_split['train']['z'].cuda()
    sid_all = pool_split['train']['sid'].cuda()
    t0 = time.time()
    start = 0
    ck = find_resume(out_dir)
    if ck is not None and (out_dir / 'ckpt_latest.pt').exists():
        st = load_ckpt(out_dir / 'ckpt_latest.pt', model=net, optimizer=opt)
        start = st['step']
    curves = []
    per_class = 16                     # 5 classes x 16 = 80 balanced pool rows
    bal = class_balanced_indices(pool_split['train']['sid'].cpu(), per_class,
                                 torch.Generator().manual_seed(seed))
    z_bal, sid_bal = z_all[bal], sid_all[bal]
    for step in range(start + 1, updates + 1):
        opt.zero_grad(set_to_none=True)
        for _ in range(accum):
            idx = torch.randint(0, len(z_bal), (micro,), generator=gen)
            z = z_bal[idx]
            s = sid_bal[idx]
            tid = s.clone()            # real-data self-pair: target == source
            loss, parts = fm_residual_loss(net, base, z, s, tid, mode, gen,
                                           bridge=bridge)
            if loss is None:
                save(out_dir / 'blocked.json',
                     {'blocked_invalid_supervision': True})
                return None
            (loss / accum).backward()
        grad = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
        if not torch.isfinite(grad):
            raise FloatingPointError('nonfinite grad')
        for pg in opt.param_groups:
            pg['lr'] = cosine_lr(step, updates, lr)
        opt.step()
        if step == 1 or step % 200 == 0:
            curves.append({'step': step, 'lr': cosine_lr(step, updates, lr),
                           'grad_norm': float(grad),
                           'wall_sec': time.time() - t0, **parts})
            print(f"{mode} {step}/{updates} "
                  f"{parts.get('fm_mse', parts.get('bridge_mse', 0)):.5f} "
                  f"{t0 and time.time()-t0:.0f}s", flush=True)
        if step % 1000 == 0 or step == updates:
            save_ckpt(out_dir / f'ckpt_step_{step}.pt', model=net,
                      optimizer=opt, step=step, gen_state=gen.get_state())
            save_ckpt(out_dir / 'ckpt_latest.pt', model=net, optimizer=opt,
                      step=step, gen_state=gen.get_state())
            heartbeat(out_dir, 'training', {'step': step})
    save(out_dir / 'train.jsonl', curves)
    torch.save({'model': net.cpu().state_dict(), 'mode': mode,
                'updates': updates, 'seed': seed},
               out_dir / 'residual.pt')
    return out_dir
