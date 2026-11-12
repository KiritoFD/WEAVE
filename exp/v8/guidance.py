"""B: Learned Gradient Geometry — content-sensitive positive-definite updates.

Why positive-definite and not a free residual
---------------------------------------------
With g = grad_z F_t(z) and delta = eta * P g, if P is positive definite then to
first order

    F_t(z + eta P g) - F_t(z) = eta g^T P g + O(eta^2) >= eta * eps * ||g||^2

so the update still ascends the style objective. A free residual network has no
such guarantee, which is exactly how the earlier N1/N2 correctors drifted into
content collapse. Content safety is a SEPARATE issue and is enforced on the
actual output, not assumed from definiteness.

Two parameterisations, both trained against the FINAL generated result:
  B-z : P acts on the raw latent gradient, predicted per position
  B-u : P acts in the generator's control coordinates (see exp.v8.controls)
No teacher endpoint is ever fitted.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def target_margin(critic, z: torch.Tensor, tid: torch.Tensor) -> torch.Tensor:
    """m_t(z) = logit_t - logsumexp_{j != t} logit_j, per sample."""
    logits = critic(z.float()).float()
    mask = F.one_hot(tid.long(), 5).bool()
    others = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits.gather(1, tid[:, None]).squeeze(1) - torch.logsumexp(others, 1)


def rms(x: torch.Tensor) -> torch.Tensor:
    return x.square().flatten(1).mean(1).sqrt()


def normalize(t: torch.Tensor) -> torch.Tensor:
    return t / rms(t).clamp_min(1e-12).view(-1, 1, 1, 1)


def project_rms(z: torch.Tensor, z0: torch.Tensor, rho) -> torch.Tensor:
    d = z - z0
    cur = rms(d).clamp_min(1e-12)
    lim = torch.as_tensor(rho, device=z.device, dtype=z.dtype)
    if lim.ndim == 0:
        lim = lim.expand_as(cur)
    return z0 + d * (lim / cur).clamp_max(1).view(-1, 1, 1, 1)


class SPDHead(nn.Module):
    """Predicts a per-position positive-definite operator from (c, z, t, g).

    P = diag(softplus(d) + eps) + U U^T, with U of shape (rank, 4) at each
    position. Output is initialised to the identity (d such that softplus gives
    ~1, U = 0), so a freshly built head reproduces the plain gradient step and
    training has to earn any deviation from it.
    """

    def __init__(self, rank: int = 2, in_ch: int = 4 + 4 + 4 + 5, width: int = 48,
                 eps: float = 1e-3):
        super().__init__()
        self.rank = rank
        self.eps = eps
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, padding=1), nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.SiLU(),
            nn.Conv2d(width, 4 + rank * 4, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, c, z, tid, g):
        b = z.shape[0]
        tmap = F.one_hot(tid.long(), 5).float().view(b, 5, 1, 1).expand(-1, -1, *z.shape[-2:])
        feat = torch.cat([c.float(), z.float(), normalize(g).float(), tmap], 1)
        out = self.net(feat)
        diag_raw, u_raw = out[:, :4], out[:, 4:]
        diag = F.softplus(diag_raw) + self.eps
        u = u_raw.view(b, self.rank, 4, *z.shape[-2:])
        return diag, u

    def apply(self, diag, u, g):
        """delta = P g, with P = diag(diag) + u u^T applied per position."""
        dg = diag * g
        # u: (B,r,4,H,W); g: (B,4,H,W) -> (u . g) then broadcast back
        ug = (u * g.unsqueeze(1)).sum(2)                  # (B,r,H,W)
        return dg + (u * ug.unsqueeze(2)).sum(1)


@dataclass
class UnrollConfig:
    steps: int = 2
    step_rms: float = 0.08
    rho: float = 0.25
    detach_gradient: bool = True     # first-order truncated unroll, not exact meta


class LatentSPDEditor(nn.Module):
    """B-z: SPD-modulated latent updates, K steps, all under one frozen R1."""

    def __init__(self, critic: nn.Module, head: SPDHead, cfg: UnrollConfig):
        super().__init__()
        self.critic = critic
        self.head = head
        self.cfg = cfg
        for p in self.critic.parameters():
            p.requires_grad_(False)

    def forward(self, z0, c, tid):
        z = z0.detach().clone()
        states = [z.clone()]
        trace = []
        for k in range(self.cfg.steps):
            zr = z.detach().clone().requires_grad_(True)
            # R1 must NOT be under no_grad: we need its input gradient.
            m = target_margin(self.critic, zr, tid)
            g = torch.autograd.grad(m.sum(), zr)[0].detach() if self.cfg.detach_gradient \
                else torch.autograd.grad(m.sum(), zr, create_graph=True)[0]
            diag, u = self.head(c, z, tid, g)
            delta = self.head.apply(diag, u, g)
            cand = z + self.cfg.step_rms * normalize(delta)
            cand = project_rms(cand, z0, self.cfg.rho)
            trace.append({"step": k, "margin_before": float(m.mean()),
                          "displacement_rms": float(rms(cand - z0).mean())})
            z = cand
            states.append(z.clone())
        return {"final": z, "states": torch.stack(states, 1), "trace": trace}


class UnsupervisedUnroll(nn.Module):
    """Training loop for the SPD operators, without any teacher.

    The operator is optimised so that the FINAL state after K steps scores well
    on the style objective while staying near z0 in content. This is the whole
    point of B: the network is responsible for its own output, and no stored
    optimiser trajectory is fitted.

    space='latent'  updates z directly with an SPD-modulated gradient (B-z)
    space='control' updates the generator's control vector instead (B-u)

    Gradients through the K steps are truncated at each step (first-order
    approximation, NOT an exact meta-gradient), which the plan requires to be
    stated rather than implied.
    """

    def __init__(self, critic, head, base, canon, *, space="latent",
                 steps=2, step_rms=0.08, rho=0.25, device="cuda",
                 control_spec=None):
        super().__init__()
        self.critic = critic
        self.head = head
        self.base = base
        self.canon = canon
        self.space = space
        self.steps = steps
        self.step_rms = step_rms
        self.rho = rho
        self.device = device
        for p in self.critic.parameters():
            p.requires_grad_(False)
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.z0 = None
        if space == "control":
            from .controls import ControlSpec
            self.control_spec = control_spec or ControlSpec()

    def _spd_apply(self, diag, u, g):
        dg = diag * g
        ug = (u * g.unsqueeze(1)).sum(2)
        return dg + (u * ug.unsqueeze(2)).sum(1)

    def forward(self, z_s, tid):
        from .controls import Controls, wrap
        with torch.no_grad():
            c = self.canon(z_s.float(), mode="c1_gentle", augment=False)
            z0 = self.base(c, 0.0, c, tid).detach()
        self.z0 = z0
        if self.space == "latent":
            z = z0.clone()
            for _ in range(self.steps):
                zr = z.detach().clone().requires_grad_(True)
                g = torch.autograd.grad(self.margin(zr, tid).sum(), zr)[0].detach()
                diag, u = self.head(c, z, tid, g)
                delta = self._spd_apply(diag, u, g)
                z = project_rms(z + self.step_rms * normalize(delta), z0, self.rho)
            return z
        # control space
        gen = wrap(self.base, self.control_spec, self.device)
        u_ctrl = torch.zeros(z_s.shape[0], self.control_spec.dim,
                             device=self.device, requires_grad=True)
        for _ in range(self.steps):
            ur = u_ctrl.detach().clone().requires_grad_(True)
            z = gen(c, 0.0, c, tid, controls=Controls.from_tensor(ur, self.control_spec))
            g_u = torch.autograd.grad(self.margin(z, tid).sum(), ur)[0].detach()
            scale = g_u.abs().mean().clamp_min(1e-8)
            diag_raw, u_low = self.head(c, z, tid, g_u / scale)
            diag = torch.nn.functional.softplus(diag_raw[..., :1].mean()) + 1e-3
            u_ctrl = (ur + self.step_rms * diag * g_u).clamp(-self.rho, self.rho)
        return gen(c, 0.0, c, tid, controls=Controls.from_tensor(u_ctrl, self.control_spec))

    def margin(self, z, tid):
        return target_margin(self.critic, z, tid)

    def content_penalty(self, z, z0):
        """Squared relative displacement; the plan wants it bounded, not zero."""
        return (z - z0).square().flatten(1).mean(1).mean()


# ----------------------------------------------------------------- generate
def generate_edit(critic, head, generator, canon, z_s, tid, *, steps=2,
                  step_rms=0.08, rho=0.25, space="latent", device="cuda"):
    """Inference-time version of the same operator (no parameter updates)."""
    with torch.no_grad():
        c = canon(z_s.float(), mode="c1_gentle", augment=False)
        z0 = generator(c, 0.0, c, tid).detach()
    z = z0.clone()
    for _ in range(steps):
        with torch.enable_grad():
            zr = z.detach().clone().requires_grad_(True)
            g = torch.autograd.grad(target_margin(critic, zr, tid).sum(), zr)[0].detach()
            diag, u = head(c, z, tid, g)
            delta = (diag * g) + (u * (u * g.unsqueeze(1)).sum(2).unsqueeze(2)).sum(1)
        z = project_rms(z + step_rms * normalize(delta), z0, rho)
    return {"z0": z0, "final": z}

