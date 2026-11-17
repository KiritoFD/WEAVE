"""V12-GCL constraint layer: budget calibration + post-step gate (plan §4).

Budget B_Q is calibrated ONLY on TRAIN paintings (never Test750):
  appearance perturbation (brightness/contrast/channel gain, seeded) defines
  the "free" content-change distribution; the gate threshold is its pre-registered
  quantile (default 0.9). A separate geometric perturbation (roll/shift) must
  EXCEED the budget (sensitivity check) — otherwise the Query is not structural.

Gate: a candidate z_cand is accepted by the LD4 solver only if
  L_Q(z_cand) <= B_Q   (in addition to the existing eps / rho / C_LL gates).
"""
from __future__ import annotations

import torch
import torch.nn.functional as Fn


def appearance_perturb(z: torch.Tensor, strength: float = 1.0,
                       seed: int = 0) -> torch.Tensor:
    """Seeded brightness/contrast/channel-gain jitter (no geometry)."""
    b = z.shape[0]
    gen = torch.Generator(device=z.device.type)
    gen.manual_seed(seed)
    bright = 1.0 + (torch.rand(b, 1, 1, 1, generator=gen, device=z.device) * 2 - 1) * 0.20 * strength
    contrast = 1.0 + (torch.rand(b, 1, 1, 1, generator=gen, device=z.device) * 2 - 1) * 0.20 * strength
    gain = 1.0 + (torch.rand(b, 4, 1, 1, generator=gen, device=z.device) * 2 - 1) * 0.15 * strength
    mean = z.mean(dim=(2, 3), keepdim=True)
    return ((z - mean) * contrast * gain + mean * bright)


def geometric_perturb(z: torch.Tensor, shift: int = 3) -> torch.Tensor:
    """Deterministic content move: circular shift by `shift` cells (structure moves)."""
    return torch.roll(z, shifts=(shift, shift), dims=(2, 3))


def calibrate_budget(latents: torch.Tensor, cost_fn, device: str,
                     quantile: float = 0.9, n_paintings: int = 48,
                     strength: float = 1.0, seed: int = 0) -> dict:
    """latents: (N,4,64,64) TRAIN-split painting latents (cpu).

    Returns {B_Q, appearance_stats, geometric_stats, n}. cost_fn(z, z_ref)
    must return per-sample costs (B,) — one Q variant per calibration run.
    """
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(latents.shape[0], generator=g)[:n_paintings]
    zs = latents[idx].float().to(device)
    app, geo = [], []
    for i in range(zs.shape[0]):
        z = zs[i:i + 1]
        zp = appearance_perturb(z, strength, seed + i)
        app.append(cost_fn(zp, z).item())
        zg = geometric_perturb(z)
        geo.append(cost_fn(zg, z).item())
    app_t, geo_t = torch.tensor(app), torch.tensor(geo)
    q = torch.quantile(app_t, quantile).item()
    return {"B_Q": round(q, 6),
            "appearance": {"mean": round(app_t.mean().item(), 6),
                           "p50": round(torch.quantile(app_t, 0.5).item(), 6),
                           "p90": round(q, 6)},
            "geometric": {"mean": round(geo_t.mean().item(), 6),
                          "p10": round(torch.quantile(geo_t, 0.1).item(), 6)},
            "sensitivity_ratio_geo_over_app_mean":
                round(geo_t.mean().item() / max(app_t.mean().item(), 1e-9), 3),
            "n_paintings": int(zs.shape[0]), "quantile": quantile,
            "strength": strength}


def gate_ok(cost_cand: torch.Tensor, budget: torch.Tensor | float) -> torch.Tensor:
    """Vector gate: per-sample cost within budget."""
    if not torch.is_tensor(budget):
        budget = torch.full_like(cost_cand, float(budget))
    return cost_cand <= budget
