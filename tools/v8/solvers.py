"""V8 ODE solvers: Euler and Heun with exact same-style identity by interface."""
from __future__ import annotations

import torch


def velocity_at(model, z, tau, src_ids, tgt_ids):
    return model(z, tau, src_ids, tgt_ids)


@torch.inference_mode()
def solve_euler(model, z0, src_ids, tgt_ids, n_steps: int = 8) -> torch.Tensor:
    z = z0.clone()
    same = (src_ids == tgt_ids).view(-1, 1, 1, 1)
    h = 1.0 / n_steps
    for k in range(n_steps):
        tau = torch.full((z.shape[0],), k * h, device=z.device, dtype=torch.float32)
        v = velocity_at(model, z, tau, src_ids, tgt_ids)
        z = z + h * v
        z = torch.where(same, z0, z)
    return z


@torch.inference_mode()
def solve_heun(model, z0, src_ids, tgt_ids, n_steps: int = 4) -> torch.Tensor:
    """Heun 2nd order; NFE = 2*n_steps."""
    z = z0.clone()
    same = (src_ids == tgt_ids).view(-1, 1, 1, 1)
    h = 1.0 / n_steps
    for k in range(n_steps):
        tau0 = torch.full((z.shape[0],), k * h, device=z.device, dtype=torch.float32)
        tau1 = torch.full((z.shape[0],), (k + 1) * h, device=z.device, dtype=torch.float32)
        v0 = velocity_at(model, z, tau0, src_ids, tgt_ids)
        z_pred = z + h * v0
        z_pred = torch.where(same, z0, z_pred)
        v1 = velocity_at(model, z_pred, tau1, src_ids, tgt_ids)
        z = z + 0.5 * h * (v0 + v1)
        z = torch.where(same, z0, z)
    return z


@torch.inference_mode()
def solve(model, z0, src_ids, tgt_ids, method: str = "euler", n_steps: int = 8) -> tuple[torch.Tensor, int]:
    if method == "euler":
        return solve_euler(model, z0, src_ids, tgt_ids, n_steps), n_steps
    if method == "heun":
        return solve_heun(model, z0, src_ids, tgt_ids, n_steps), 2 * n_steps
    raise ValueError(f"Unknown solver: {method}")
