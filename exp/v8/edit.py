"""Constrained latent editor.

Contract enforced here (each of these was a real bug class in earlier rounds):
  1. the gradient is taken at the CURRENT z, not the initial z;
  2. the proposal direction and the acceptance test use the SAME scalar energy
     (no "propose with a normalised gradient, accept with a different loss");
  3. the candidate is projected into the RMS ball FIRST, then the energy and
     content proxy are re-evaluated on the projected point;
  4. if no candidate is accepted, z stays exactly where it was (a rejected step
     is a rejection, not a silent small step);
  5. every step is per-image: no batch coupling in projection or acceptance.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class EditConfig:
    steps: int = 4
    step_rms: float = 0.08
    rho: float | None = 0.32
    alphas: tuple = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125)
    decrease_tol: float = 1e-8
    content_slack: float | None = None


def rms(x: torch.Tensor) -> torch.Tensor:
    return x.square().flatten(1).mean(1).sqrt()


def clip_rms(delta: torch.Tensor, rho) -> torch.Tensor:
    cur = rms(delta).clamp_min(1e-12)
    limit = torch.as_tensor(rho, device=delta.device, dtype=delta.dtype)
    if limit.ndim == 0:
        limit = limit.expand_as(cur)
    return delta * (limit / cur).clamp_max(1).reshape(-1, 1, 1, 1)


def normalize(g: torch.Tensor) -> torch.Tensor:
    return g / rms(g).clamp_min(1e-12).view(-1, 1, 1, 1)


def edit(z0: torch.Tensor, energy, config: EditConfig = EditConfig(),
         *, content=None) -> dict:
    """Run the constrained edit. Returns final latents, per-step trace, states.

    energy(z) -> (B,) scalar per image, differentiable.
    content(z) -> (B,) non-negative proxy, evaluated only for acceptance.
    """
    z = z0.detach().clone()
    states = [z.clone()]
    gradients = []
    trace = []
    baseline = content(z).detach() if content is not None else None

    for step in range(config.steps):
        cur = z.detach().clone().requires_grad_(True)
        before = energy(cur)
        if before.shape != (z.shape[0],):
            raise ValueError("energy must return one value per image")
        if not torch.isfinite(before).all():
            raise FloatingPointError("non-finite energy")
        g = torch.autograd.grad(before.sum(), cur)[0]
        if not torch.isfinite(g).all():
            raise FloatingPointError("non-finite gradient")
        direction = normalize(-g)

        accepted = torch.zeros(z.shape[0], device=z.device, dtype=torch.bool)
        chosen = z.detach().clone()
        alpha_used = torch.zeros(z.shape[0], device=z.device)
        trials = []
        for alpha in config.alphas:
            raw = cur.detach() + alpha * config.step_rms * direction.detach()
            if config.rho is not None:
                raw = z0 + clip_rms(raw - z0, config.rho)
            val = energy(raw).detach()
            ok = torch.isfinite(val) & (val < before.detach() - config.decrease_tol)
            cval = None
            if content is not None:
                cval = content(raw).detach()
                ok = ok & torch.isfinite(cval) & (cval <= baseline + (config.content_slack or 0.0))
            take = (~accepted) & ok
            chosen = torch.where(take[:, None, None, None], raw, chosen)
            alpha_used = torch.where(take, alpha_used.new_tensor(alpha), alpha_used)
            accepted |= take
            trials.append({"alpha": alpha, "energy": val.cpu().tolist(),
                           "safe": ok.cpu().tolist(),
                           "content": None if cval is None else cval.cpu().tolist()})
            if bool(accepted.all()):
                break

        trace.append({
            "step": step,
            "energy_before": before.detach().cpu().tolist(),
            "energy_after": energy(chosen).detach().cpu().tolist(),
            "accepted": accepted.cpu().tolist(),
            "alpha_used": alpha_used.cpu().tolist(),
            "gradient_rms": rms(g).detach().cpu().tolist(),
            "displacement_rms": rms(chosen - z0).detach().cpu().tolist(),
            "step_rms": rms(chosen - z).detach().cpu().tolist(),
            "trials": trials,
        })
        gradients.append(g.detach())
        z = chosen.detach()
        states.append(z.clone())

    return {"final": z, "trace": trace, "states": torch.stack(states, 1),
            "gradients": torch.stack(gradients, 1),
            "accepted": torch.tensor([t["accepted"] for t in trace]).T}
