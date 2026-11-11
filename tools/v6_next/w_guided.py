"""WEAVE W P1 — guided latent editing with a real-data score prior.

Implements the three comparison arms requested for the next phase:

  1. ``r1``   : iterative R1 target-margin gradient editing (multi-step Oracle-like);
  2. ``score``: iterative editing using only a trained real-latent score prior;
  3. ``both`` : normalized R1 direction + normalized real-data score direction.

All modes use the same per-step RMS budget so comparisons are budget-matched.
The score prior is an unconditional denoiser trained on the A0-encoded real
painting latents.  At inference, ``score = -eps_theta(z, sigma) / sigma`` is used
as an approximation of the real-data score at ``z``.

This is NOT a claim that R1 is a calibrated log p(t|z); it is a controlled test
of whether adding a real-data manifold constraint improves the style--content
Pareto over R1 alone.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def r1_margin_per_sample(z: torch.Tensor, target_id: torch.Tensor, critic: torch.nn.Module) -> torch.Tensor:
    """Target-vs-other logit margin from the frozen R1 critic. Returns (B,)."""
    q = critic(z.float()).float()
    oh = F.one_hot(target_id.long(), q.shape[1]).bool()
    return q.gather(1, target_id.long()[:, None]).squeeze(1) - torch.logsumexp(
        q.masked_fill(oh, torch.finfo(q.dtype).min), dim=1
    )


def r1_gradient(z: torch.Tensor, target_id: torch.Tensor, critic: torch.nn.Module) -> torch.Tensor:
    """Raw ∇z margin, computed with gradients enabled. Caller must detach."""
    with torch.enable_grad():
        zr = z.detach().clone().requires_grad_(True)
        m = r1_margin_per_sample(zr, target_id, critic)
        g = torch.autograd.grad(m.sum(), zr)[0]
    return g.detach()


def rms_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    rms = x.flatten(1).pow(2).mean(dim=1, keepdim=True).sqrt().view(-1, 1, 1, 1)
    return x / (rms + eps)


def data_score(eps_pred: torch.Tensor, sigma: float | torch.Tensor) -> torch.Tensor:
    """Convert a denoiser's predicted noise to an approximate data score."""
    if not torch.is_tensor(sigma):
        sigma = torch.tensor(float(sigma), device=eps_pred.device, dtype=eps_pred.dtype)
    return -eps_pred.float() / sigma.float().view(-1, 1, 1, 1)


def guided_edit(
    z: torch.Tensor,
    target_id: torch.Tensor,
    *,
    critic: torch.nn.Module,
    score_net: torch.nn.Module,
    mode: str = "both",
    steps: int = 4,
    per_step_rms: float = 0.08,
    sigma: float = 0.30,
    r1_weight: float = 1.0,
    score_weight: float = 1.0,
    r1_hard_boost: float = 2.0,
) -> torch.Tensor:
    """Run iterative guided editing in latent space.

    The R1 term uses the anti-attractor margin (hard targets repel Impressionism/
    Minimalism). The score term uses the frozen unconditional real-latent denoiser.
    Each step is normalized to the same per-sample RMS budget.
    """
    if mode not in ("r1", "score", "both"):
        raise ValueError(f"unknown guided mode: {mode}")
    with torch.inference_mode(False), torch.enable_grad():
        z_cur = z.detach().clone()
        tid = target_id.detach().clone()
        for _ in range(max(1, steps)):
            if mode == "r1":
                g_r1 = r1_gradient(z_cur, tid, critic)
                g_r1_n = rms_normalize(g_r1)
                d = g_r1_n
            else:
                # The denoiser was trained on clean latents plus Gaussian noise; here we
                # evaluate it at the current latent with the chosen sigma to obtain an
                # approximate score of the real-latent distribution.
                eps_pred = score_net(z_cur.float(), torch.full(
                    (z_cur.shape[0],), float(sigma), device=z_cur.device, dtype=torch.float32,
                ), torch.zeros(z_cur.shape[0], dtype=torch.long, device=z_cur.device))
                s_data = data_score(eps_pred, sigma)
                s_data_n = rms_normalize(s_data)
                if mode == "score":
                    d = s_data_n
                else:
                    g_r1 = r1_gradient(z_cur, tid, critic)
                    g_r1_n = rms_normalize(g_r1)
                    d = r1_weight * g_r1_n + score_weight * s_data_n

            d = rms_normalize(d)
            z_cur = (z_cur + per_step_rms * d).detach()
    return z_cur.detach()
