"""Per-sample hard-budget test-time gradient descent (batched inner eval).

cost_tot(z) = mean((z-z0)^2) / mean((z0-z_src)^2)
A candidate is accepted only if cost_tot <= eps AND proxy improves;
otherwise the sample rolls back to z_N2. Identity tasks short-circuit.
"""
from __future__ import annotations

import torch

from exp.v8.guidance import project_rms, target_margin

EPS = 0.10
ETAS = (0.5, 1.0)


def cost_tot(z, z0, z_src):
    num = (z - z0).square().flatten(1).mean(1)
    den = (z0 - z_src).square().flatten(1).mean(1).clamp_min(1e-8)
    return num / den


def proxy(z, tid, critic, lse, w_r1=0.3):
    """(B,) higher=better: R1 margin minus weighted local energy."""
    m = target_margin(critic, z.float(), tid)
    e = lse.loss(z, tid)
    return m - w_r1 * e


@torch.no_grad()
def solve_batch(z_n2, z0, z_src, sid, tid, dirs_list, masks, critic, lse,
                eps=EPS, etas=ETAS, k_steps=2, rho=0.15):
    """dirs_list: list of dicts name->(B,4,64,64); masks: (B,1,64,64) or None.
    Returns (z_out (B,4,64,64), infos list)."""
    b = z_n2.shape[0]
    dev = z_n2.device
    z = z_n2.clone()
    infos = [{"steps": 0, "accepted": 0, "short": False, "over": False}
             for _ in range(b)]
    active = torch.ones(b, dtype=torch.bool, device=dev)
    for _ in range(k_steps):
        if not active.any():
            break
        idx = active.nonzero(as_tuple=True)[0]
        p_cur = proxy(z[idx], tid[idx], critic, lse)
        best_p = p_cur.clone()
        best_z = z[idx].clone()
        acc = torch.zeros(len(idx), dtype=torch.bool, device=dev)
        for dirs in dirs_list:
            for eta in etas:
                for name, d in dirs.items():
                    step = d[idx] * (masks[idx] if masks is not None else 1.0)
                    cand = project_rms(z[idx] + eta * step, z0[idx], rho)
                    c = cost_tot(cand, z0[idx], z_src[idx])
                    feas = c <= eps
                    if not feas.any():
                        continue
                    p = proxy(cand[feas], tid[idx][feas], critic, lse)
                    bi = feas.nonzero(as_tuple=True)[0]
                    better = p > best_p[bi]
                    if better.any():
                        sel = bi[better]
                        best_z[sel] = cand[sel]
                        best_p[sel] = p[better]
                        acc[sel] = True
        for j, i in enumerate(idx.tolist()):
            if acc[j]:
                z[i] = best_z[j]
                infos[i]["accepted"] += 1
                infos[i]["steps"] += 1
            else:
                infos[i]["over"] = infos[i]["steps"] == 0
                active[i] = False
    # identity short-circuit handled by caller via sid==tid mask
    # final safety: anything over budget -> z_n2
    c = cost_tot(z, z0, z_src)
    bad = c > eps
    if bad.any():
        z[bad] = z_n2[bad]
        for i in bad.nonzero(as_tuple=True)[0].tolist():
            infos[i]["over"] = True
    return z, infos
