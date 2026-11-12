"""C: Real-Supported Discriminative Energy.

R1 gives a good instance-level DIRECTION but is only a 5-way classifier: a
correct class decision does not imply real brushwork, and its gradient can be
large at points far from any real painting. Historically the Oracle raised
Top-1 sharply while CSD dT FELL, which is exactly that failure.

C adds a real-painting support term to the frozen R1 embedding:

    F_t(z) = m_t(z) - lambda * E_t^real(z)
    E_t^real(z) = -tau log sum_j exp(-||phi(z) - mu_{t,j}||^2 / (2 tau sigma_t^2))

with MULTIPLE prototypes per style, so the energy can move toward any real
paint mode instead of regressing to one average. This is a support PROXY in a
frozen feature space, not a calibrated log-density, and lambda starts small so
it cannot swamp the validated R1 term.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]


@dataclass
class RealSupport:
    """Per-style prototype bank in frozen R1 embedding space."""
    prototypes: torch.Tensor      # (5, K, D)
    sigma: torch.Tensor           # (5,)
    tau: float = 1.0

    def energy(self, embed: torch.Tensor, tid: torch.Tensor) -> torch.Tensor:
        """(B,D) + (B,) -> (B,) support energy (lower = closer to real paintings)."""
        out = []
        for i in range(embed.shape[0]):
            t = int(tid[i])
            mu = self.prototypes[t].to(embed.device)
            d = (embed[i:i + 1] - mu).square().sum(-1) / (2 * self.tau * self.sigma[t].to(embed.device) ** 2)
            out.append(-self.tau * torch.logsumexp(-d, dim=-1).squeeze(0))
        return torch.stack(out)


def build_real_support(critic, pool_dir: Path, *, k: int = 32, seed: int = 42,
                       device: str = "cuda", max_per_style: int = 400) -> RealSupport:
    """Collect R1 embeddings of REAL training paintings and take K medoids each.

    Prototypes come from the training pool only; tasks_750 sources and the
    independent references are never used to build the support term.
    """
    from tools.v7r.v7r_common import SD_POOL
    pool = Path(pool_dir) if pool_dir else SD_POOL
    g = torch.Generator().manual_seed(seed)
    protos, sigmas = [], []
    for sid, name in enumerate(STYLES):
        d = torch.load(pool / f"{sid:02d}_{name}.pt", map_location="cpu", weights_only=False)
        x = d["latents"].float()[100:]                     # training split
        take = torch.randperm(x.shape[0], generator=g)[:max_per_style]
        xb = x[take].to(device)
        with torch.inference_mode():
            emb = _embed(critic, xb)
        emb = F.normalize(emb.float(), dim=-1)
        proto = _kmeans(emb, k, seed=seed + sid)
        # sigma from the mean distance to the assigned prototype
        assign = torch.cdist(emb, proto).argmin(1)
        d2 = (emb - proto[assign]).square().sum(-1)
        sigmas.append(d2.mean().sqrt().clamp_min(1e-3))
        protos.append(proto)
    return RealSupport(torch.stack(protos), torch.stack(sigmas))


def _embed(critic, z: torch.Tensor) -> torch.Tensor:
    """R1 embedding. probe_style exposes embed(); fall back to logits."""
    if hasattr(critic, "embed"):
        return critic.embed(z.float())
    return critic(z.float())


def _kmeans(x: torch.Tensor, k: int, *, seed: int, iters: int = 25) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(x.shape[0], generator=g)[:k]
    cent = x[idx].clone()
    for _ in range(iters):
        a = torch.cdist(x, cent).argmin(1)
        for j in range(k):
            m = a == j
            if m.any():
                cent[j] = x[m].mean(0)
    return cent


class SupportedEnergy:
    """F_t(z) = m_t(z) - lambda * E_real_t(z).

    `lambda_ratio` is expressed relative to the R1 margin gradient norm so the
    support term cannot dominate; the two registered values are 0.1 and 0.3.
    """

    def __init__(self, critic, support: RealSupport | None = None,
                 lambda_ratio: float = 0.0):
        self.critic = critic
        self.support = support
        self.lambda_ratio = lambda_ratio

    def margin(self, z, tid):
        logits = self.critic(z.float()).float()
        mask = F.one_hot(tid.long(), 5).bool()
        others = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
        return logits.gather(1, tid[:, None]).squeeze(1) - torch.logsumexp(others, 1)

    def value(self, z, tid):
        m = self.margin(z, tid)
        if self.support is None or self.lambda_ratio == 0:
            return m
        emb = F.normalize(_embed(self.critic, z).float(), dim=-1)
        return m - self.lambda_ratio * self.support.energy(emb, tid)

    def gradient(self, z, tid, *, normalize_rms: bool = True):
        zr = z.detach().clone().requires_grad_(True)
        v = self.value(zr, tid)
        g = torch.autograd.grad(v.sum(), zr)[0]
        if normalize_rms:
            g = g / g.square().flatten(1).mean(1).sqrt().clamp_min(1e-12).view(-1, 1, 1, 1)
        return g.detach(), v.detach()


def robustness_finetune_step(critic, z_real, tid, optimizer, *, delta_rms: float = 0.02,
                             lambda_aug: float = 1.0, lambda_cons: float = 0.5):
    """C3: small-radius robustness fine-tune of R1 on REAL paintings.

    One step of CE + CE(z+delta) + consistency, with delta inside a radius where
    the style label is still valid. No generator is judged, so this is not a GAN.
    """
    logits = critic(z_real.float())
    ce = F.cross_entropy(logits, tid.long())
    delta = torch.randn_like(z_real)
    delta = delta / rms_flat(delta).view(-1, 1, 1, 1) * delta_rms
    logits_aug = critic((z_real + delta).float())
    ce_aug = F.cross_entropy(logits_aug, tid.long())
    cons = (F.softmax(logits, -1) - F.softmax(logits_aug, -1)).square().mean()
    loss = ce + lambda_aug * ce_aug + lambda_cons * cons
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return {"loss": float(loss), "ce": float(ce), "ce_aug": float(ce_aug),
            "cons": float(cons)}


def rms_flat(x):
    return x.square().flatten(1).mean(1).sqrt()
