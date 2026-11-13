"""F-route: Trainable Style Actuation.

A3/A4 showed that optimising controls of a FROZEN generator cannot exceed the
reachable set the generator already expresses. The actuator instead learns
WHAT style change should be executable:

    ctrl = D_phi(feat(c), t) + [U_psi(u, feat(c), t) - U_psi(0, feat(c), t)]
    z    = G_B4(c, t; ctrl)          # frozen generator, existing injection

- D_phi: direct (content, target) -> control map. This is the learned "style
  prior" — F0 trains only this (pure forward, no per-instance u).
- U_psi: per-instance correction gated by the latent control u. The
  difference form is exactly 0 at u = 0 and keeps every weight trainable
  (a plain zero-init MLP would be a saddle: d ctrl / d w = 0 everywhere).

At inference, u is optimised per instance with the SAME inner loop used in
training (train/infer consistency). F1 = one inner R1 step, F2 = two,
F3 = F2 + V6 real style-distribution supervision.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(in_dim: int, out_dim: int, width: int, depth: int) -> nn.Sequential:
    layers = [nn.Linear(in_dim, width), nn.SiLU()]
    for _ in range(depth - 1):
        layers += [nn.Linear(width, width), nn.SiLU()]
    layers += [nn.Linear(width, out_dim)]
    return nn.Sequential(*layers)


class StyleActuator(nn.Module):
    """(feat, tid) direct head + (u, feat, tid) gated correction head."""

    def __init__(self, u_dim: int, feat_dim: int, control_dim: int,
                 width: int = 256, depth: int = 2):
        super().__init__()
        self.u_dim = u_dim
        self.feat_dim = feat_dim
        self.control_dim = control_dim
        self.direct = _mlp(feat_dim + 5, control_dim, width, depth)
        self.delta = _mlp(u_dim + feat_dim + 5, control_dim, width, depth)
        for net in (self.direct, self.delta):
            for m in net:
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.02)
                    nn.init.zeros_(m.bias)

    def forward(self, u: torch.Tensor, feat: torch.Tensor,
                tid: torch.Tensor) -> torch.Tensor:
        b = u.shape[0]
        oh = F.one_hot(tid.long(), 5).float().to(u.dtype)
        f = feat.to(u.dtype)
        base = self.direct(torch.cat([f, oh], dim=1))
        x1 = torch.cat([u, f, oh], dim=1)
        x0 = torch.cat([u.new_zeros(b, self.u_dim), f, oh], dim=1)
        return base + self.delta(x1) - self.delta(x0)


def build_actuator(u_dim: int, feat_dim: int, control_dim: int,
                   device: str = "cuda") -> StyleActuator:
    return StyleActuator(u_dim, feat_dim, control_dim).to(device)
