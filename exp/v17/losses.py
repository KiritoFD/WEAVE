"""V17 losses: Phase I/II with exact identities and zero-teacher safety.

E_rel(a) = 1 + a^2 - 2*a*rho   (direction cosine rho, amplitude a) — asserted
numerically in tests/v17/test_losses.py.  Teacher RMS ~0 samples use MSE/Where
only and never feed the cosine term.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def relative_mse_identity(delta: torch.Tensor, teacher: torch.Tensor):
    """E_rel = ||delta - teacher||^2 / ||teacher||^2 must equal
    1 + ||delta||^2/||teacher||^2 - 2*cos*||delta||/||teacher||, with rho the
    TRUE cosine of the two vectors (asserted numerically)."""
    dt = delta.flatten()
    tt = teacher.flatten()
    e_true = (dt - tt).square().sum() / tt.square().sum()
    rho = F.cosine_similarity(dt, tt, dim=0)
    a = dt.norm() / tt.norm()
    e_ref = 1 + a ** 2 - 2 * a * rho
    return torch.allclose(e_true, e_ref, atol=1e-4), e_true, e_ref


def student_loss(delta, teacher_delta, mask_logits=None, teacher_mask=None,
                 lam_dir=.5, lam_where=.2, phase2=False, rho=None,
                 a_target=None, lam_norm=0., low_pool=4):
    """Phase I: 0.1*MSE + dir cosine + where BCE.  Phase II adds the bounded
    norm term gated by training progress, never by Test750 outcomes."""
    parts = {}
    mse = (delta - teacher_delta).square().mean()
    parts['mse'] = mse
    loss = .1 * mse
    t_rms = teacher_delta.flatten(1).square().mean(1).sqrt()
    stable = t_rms > 1e-6                                  # zero-teacher guard
    if stable.any():
        d = delta[stable]
        t = teacher_delta[stable]
        rho_s = F.cosine_similarity(d.flatten(1), t.flatten(1), dim=1)
        parts['cos'] = float(rho_s.mean())
        loss = loss + lam_dir * (1 - rho_s).mean()
    if mask_logits is not None and teacher_mask is not None:
        bce = F.binary_cross_entropy_with_logits(
            mask_logits.flatten(1), (teacher_mask.flatten(1) > .5).float())
        parts['where_bce'] = float(bce)
        loss = loss + lam_where * bce
    low = F.avg_pool2d(delta, low_pool).square().mean()
    parts['low_frequency_penalty'] = float(low)
    loss = loss + .1 * low + .01 * delta.square().mean()
    if phase2:
        # norm soft-target: a_target = stopgrad(clamp(cos,0,.8)) * RMS(teacher),
        # trust-clipped per sample; never forces 100% teacher energy.
        a = delta.flatten(1).norm(dim=1) / \
            teacher_delta.flatten(1).norm(dim=1).clamp_min(1e-8)
        a_t = (rho.clamp(0, .8) if rho is not None else
               parts.get('cos', 0.)) * (t_rms / t_rms.clamp_min(1e-8))
        rel = 1 + a[stable] ** 2 - 2 * a[stable] * a_t[stable]
        parts['rel_norm'] = float(rel.mean())
        loss = loss + lam_norm * rel.mean()
    parts['loss'] = float(loss)
    return loss, parts
