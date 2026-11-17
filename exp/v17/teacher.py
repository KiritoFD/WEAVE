"""P1: teacher variants T0-T4 producing V17 teacher-pair files.

T0  = V16 teacher replay, regenerated WITH the 25% TopK region mask stored
      (S1's BCE contract) plus per-sample trust/acceptance info.
T1  = R1 logit per-class calibration.  A unit test (tests/v17/test_teacher.py)
      proves a constant logit bias leaves the linear margin gradient unchanged;
      T1 therefore only changes the piecewise-linear max-other term / ranking.
T2  = explicit Target-Source margin gradient direction.
T3  = per-target variable low-frequency cap: avgpool4(delta) energy above the
      target's calibration quantile is scaled down (never uniformly deleted).
T4  = T2 + T3.

The T0 pair file is the single distillation source for all S-arms (S7 uses T4).
"""
from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from .common import OUT, STYLES, save
from exp.v16 import load_base, pool_samples, read, save as v16save
from exp.v16.distill import Teacher as V16Teacher


class CalibratedDirectionComputer:
    """T1: wraps V16 DirectionComputer with per-class logit bias on the critic.

    The bias enters target_margin through max-other (piecewise linear), so
    gradients change ONLY where calibration flips the argmax — asserted by
    tests/v17/test_teacher.py before the queue starts.
    """

    def __init__(self, inner, bias: torch.Tensor):
        self.inner = inner
        self.bias = bias.cuda()                       # (5,)

    def directions(self, z, z0, tid, mode):
        return self.inner.directions(z, z0, tid, mode)

    def region_mask(self, z, z0, tid, frac=.25):
        return self.inner.region_mask(z, z0, tid, frac)

    def patch_scores(self, z, tid):
        return self.inner.lse.patch_scores(z, tid) + self.bias[tid]


class TeacherV17(V16Teacher):
    """V16 teacher with pluggable direction computer + low-freq post-cap."""

    def __init__(self, variant='T0', calib_bias=None):
        super().__init__()
        self.variant = variant
        if variant == 'T1':
            self.direction = CalibratedDirectionComputer(self.direction,
                                                         calib_bias)

    def _dirs(self, base, source, sid, tid):
        if self.variant in ('T0', 'T1', 'T3'):
            return self.direction.directions(base, base, tid, 'mix')
        if self.variant in ('T2', 'T4'):
            # explicit Target-Source margin gradient, normalised
            inner = getattr(self.direction, 'inner', self.direction)
            g_t = inner._grad(lambda zz: _target_margin(inner.critic, zz, tid),
                              base)
            g_s = inner._grad(lambda zz: _target_margin(inner.critic, zz, sid),
                              base)
            from exp.v9_local.direction import normalize as _n  # noqa
            d = g_t - g_s
            return {'r1_ts': d / d.flatten(1).norm(dim=1).view(-1, 1, 1, 1)
                    .clamp_min(1e-8)}
        raise ValueError(self.variant)

    def __call__(self, base, source, sid, tid, lf_caps=None):
        from exp.v9_local.solver import solve_batch
        with torch.enable_grad():
            dirs = self._dirs(base, source, sid, tid)
        with torch.no_grad():
            radius = (.10 * (base - source).flatten(1).square().mean(1))
            radius = radius.sqrt().clamp_max(.15)
            dirs = {k: v * radius[:, None, None, None] for k, v in dirs.items()}
            mask = self.direction.region_mask(base, base, tid, frac=.25)
            out, info = solve_batch(base, base, source, sid, tid, [dirs], mask,
                                    self.critic, self.energy, eps=.10,
                                    k_steps=2, rho=.15)
            same = sid == tid
            out[same] = base[same]
            if lf_caps is not None:                      # T3/T4 low-freq cap
                delta = out - base
                lf = F.avg_pool2d(delta, 4)              # (B,4,16,16)
                src_rms = (base - source).flatten(1).square().mean(1) \
                    .sqrt().clamp_min(1e-8)
                lf_rms = lf.flatten(1).square().mean(1).sqrt()
                ratio = lf_rms / src_rms
                scale = torch.where(ratio > cap_safe(tid, lf_caps),
                                    cap_safe(tid, lf_caps) /
                                    ratio.clamp_min(1e-8),
                                    torch.ones_like(ratio))
                lf_scaled = lf * scale.view(-1, 1, 1, 1)
                hf = delta - F.interpolate(lf, scale_factor=4, mode='nearest')
                out = base + F.interpolate(lf_scaled, scale_factor=4,
                                           mode='nearest') + hf
                for i in (scale < 1).nonzero(as_tuple=True)[0].tolist():
                    if not same[i]:
                        info[i]['lf_capped'] = True
            cost = (out - base).flatten(1).square().mean(1) / \
                (base - source).flatten(1).square().mean(1).clamp_min(1e-8)
            if (cost > .100001).any():
                raise AssertionError('Teacher exceeded V15 content budget')
        return out.detach(), info


def _target_margin(critic, z, tid):
    from exp.v8.guidance import target_margin
    return target_margin(critic, z.float(), tid)


def cap_safe(tid, lf_caps):
    return lf_caps[tid].clamp_min(1e-6)


def lf_caps_from_pairs(pairs_path, quantile=.75, device='cuda'):
    """Per-target low-frequency energy quantile from the T0 pairs.

    Uses the HOLDOUT split as an independent calibration set (never the same
    samples used to construct the student's training targets).  Falls back to
    train only if holdout is absent (older pair files)."""
    d = torch.load(pairs_path, map_location='cpu', weights_only=False)
    caps = []
    tr = d.get('holdout') or d.get('calib') or d.get('train')
    for s in range(5):
        m = (tr['tid'] == s) & (tr['sid'] != tr['tid'])
        if m.sum() == 0:
            caps.append(1e9)
            continue
        lf = F.avg_pool2d(tr['delta'][m].float(), 4)
        n = int(m.sum())
        src = tr['content'][m].float() if 'content' in tr else tr['base'][m].float()
        src_rms = (tr['base'][m].float() - src).flatten(1) \
            .square().mean(1).sqrt().clamp_min(1e-8)
        assert len(src_rms) == n, (len(src_rms), n)
        rel = lf.flatten(1).square().mean(1).sqrt() / src_rms
        caps.append(float(torch.quantile(rel, quantile)))
    return torch.tensor(caps, device=device)


def r1_logit_bias(pairs_path=None, device='cuda', force=False):
    """T1 calibration: per-class bias from an INDEPENDENT holdout set.

    For each holdout sample compute the R1 target margin (critic logits);
    classes with systematically HIGH margin are over-predicted -> lower their
    bias.  Bias is stored under runs/v17/teacher/r1_logit_bias.pt.  The unit
    test proves a constant bias does not change the linear-margin gradient
    while it CAN flip a nonlinear max-other ranking — the only real T1
    effect.  The critic is never the final CSD referee."""
    from .common import OUT, save
    path = OUT / 'teacher' / 'r1_logit_bias.pt'
    if path.exists() and not force:
        return torch.load(path, map_location='cpu', weights_only=False)
    from src.infer import load_critic
    from exp.v8.guidance import target_margin
    d = torch.load(pairs_path or (OUT / 'pairs' / 'teacher_pairs_T0.pt'),
                   map_location='cpu', weights_only=False)
    ho = d.get('holdout') or d.get('train')
    critic = load_critic(device)
    per = {s: [] for s in range(5)}
    with torch.no_grad():
        for st in range(0, len(ho['tid']), 32):
            base = ho['base'][st:st + 32].float().to(device)
            tid = ho['tid'][st:st + 32].to(device)
            margins = target_margin(critic, base, tid).cpu()
            for s in range(5):
                m = tid.cpu() == s
                if m.any():
                    per[s].append(margins[m].mean())
    mean = torch.tensor([torch.stack(v).mean() if v else 0.
                         for v in per.values()])
    bias = -(mean - mean.mean())                    # over-predicted -> lower
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bias, path)
    save(OUT / 'teacher' / 'r1_calibration.json',
         {'per_class_mean_margin': mean.tolist(), 'bias': bias.tolist(),
          'source': 'T0 holdout (independent of student training pairs)',
          'note': 'constant bias cannot change linear-margin gradients; '
                  'only nonlinear max-other ranking flips'})
    print(f'R1 calibration bias: {bias.tolist()}', flush=True)
    return bias


def prepare_pairs(variant: str, pool: str, train_each=100, hold_each=20,
                  batch=8, seed=20261117, calib_bias=None, updates_info=None):
    """Generate teacher pairs for a variant; stores base/content/delta/mask."""
    out_path = OUT / 'pairs' / f'teacher_pairs_{variant}.pt'
    if out_path.exists():
        return out_path
    source, manifest = pool_samples(pool, train_each, hold_each, seed)
    model, _ = load_base()
    lf_caps = None
    if variant in ('T3', 'T4'):
        t0_path = OUT / 'pairs' / 'teacher_pairs_T0.pt'
        lf_caps = lf_caps_from_pairs(t0_path)
    teacher = TeacherV17(variant, calib_bias)
    from src.model import canonicalizer
    canon = canonicalizer('cuda')
    g = torch.Generator().manual_seed(seed)
    result, infos, t0 = {}, [], time.time()
    for split, d in source.items():
        src = d['z'].repeat_interleave(5, 0)
        sid = d['sid'].repeat_interleave(5)
        tid = torch.arange(5).repeat(len(d['z']))
        eps = torch.randn(src.shape, generator=g)
        arrays = {k: [] for k in ['base', 'content', 'delta', 'mask']}
        for st in range(0, len(src), batch):
            sl = slice(st, st + batch)
            z, si, ti = src[sl].cuda(), sid[sl].cuda(), tid[sl].cuda()
            with torch.no_grad():
                base = generate_v17(model, z, ti, eps[sl].cuda())
                content = canon(z, mode='c1_gentle', augment=False)
                corrected, report = teacher(base, z, si, ti)
                mask = teacher.direction.region_mask(base, z, ti, frac=.25)
            for key, value in [('base', base), ('content', content),
                               ('delta', corrected - base),
                               ('mask', mask.cpu())]:
                arrays[key].append(value.cpu().half() if key != 'mask'
                                   else value.cpu().half())
            infos.extend(report)
        result[split] = {k: torch.cat(v) for k, v in arrays.items()}
        result[split].update({'sid': sid, 'tid': tid, 'source_ids': d['ids'],
                              'infos': infos})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, out_path)
    stats = {}
    for split, d in result.items():
        cross = d['sid'] != d['tid']
        rms = d['delta'].float().flatten(1).square().mean(1).sqrt()
        stats[split] = {'n': len(rms), 'cross_n': int(cross.sum()),
                        'delta_rms_cross': float(rms[cross].mean()),
                        'nonzero_cross_fraction': float(
                            (rms[cross] > 1e-7).float().mean())}
    save(OUT / 'pairs' / f'{variant}_summary.json',
         {'splits': stats, 'wall_sec': time.time() - t0, 'variant': variant,
          'lf_caps': None if lf_caps is None else lf_caps.tolist(),
          **(updates_info or {})})
    print(f'P1 {variant}: {stats["train"]["cross_n"]} cross pairs '
          f'{time.time()-t0:.0f}s', flush=True)
    return out_path


@torch.no_grad()
def generate_v17(model, source, target, noise, tau0=.7, steps=20):
    from .common import autocast
    z = tau0 * source + (1 - tau0) * noise
    dt = (1 - tau0) / steps
    for k in range(steps):
        tau = torch.full((len(z),), tau0 + k * dt, device=z.device)
        with autocast(z.device):
            v = model(z, target, tau).float()
        z = z + dt * v
    return z
