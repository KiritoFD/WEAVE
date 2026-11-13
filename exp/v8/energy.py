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


def _featmap(critic, z: torch.Tensor) -> torch.Tensor:
    """R1's spatial feature map BEFORE the global pool: (B, w*3, 16, 16).

    Replicates LatentStyleNet.embed's conv stack minus adaptive_avg_pool2d so
    the support term can be computed per region instead of only globally.
    """
    h = critic.stem(z.float())
    h = critic.b64(h)
    h = critic.b32(critic.down32(h))
    h = critic.b16(critic.down16(h))
    return h


@dataclass
class SpatialSupport:
    """Regional real-painting support in R1 feature-map space (C2 local support).

    prototypes: (5, R, K, D) — R grid regions (default 4x4 over the 16x16 map),
    K medoids per region per style. energy() averages the per-region soft-min
    support energy, so a generated image must look real LOCALLY, not just on
    average. Still a support proxy in a frozen feature space, not a density.
    """
    prototypes: torch.Tensor      # (5, R, K, D)
    sigma: torch.Tensor           # (5,)
    grid: int = 4
    tau: float = 1.0

    def regional_embed(self, critic, z: torch.Tensor) -> torch.Tensor:
        """(B, C, 16, 16) -> (B, R, D) with R = grid*grid."""
        h = _featmap(critic, z)
        h = F.adaptive_avg_pool2d(h, self.grid)          # (B,D,g,g)
        b, d, gh, gw = h.shape
        return h.flatten(2).transpose(1, 2)              # (B, R, D)

    def energy(self, critic, z: torch.Tensor, tid: torch.Tensor) -> torch.Tensor:
        reg = self.regional_embed(critic, z)             # (B,R,D)
        out = []
        for i in range(z.shape[0]):
            t = int(tid[i])
            mu = self.prototypes[t].to(reg.device)       # (R,K,D)
            d = (reg[i:i + 1, :, None, :] - mu[None]).square().sum(-1) \
                / (2 * self.tau * self.sigma[t].to(reg.device) ** 2)   # (1,R,K)
            e = -self.tau * torch.logsumexp(-d, dim=-1)  # (1,R)
            out.append(e.mean())
        return torch.stack(out)


def build_local_support(critic, pool_dir: Path, *, k: int = 8, grid: int = 4,
                        seed: int = 42, device: str = "cuda",
                        max_per_style: int = 400) -> SpatialSupport:
    """Regional prototypes from REAL training paintings (same pool rules as
    build_real_support: training split only, never eval sources)."""
    from tools.v7r.v7r_common import SD_POOL
    pool = Path(pool_dir) if pool_dir else SD_POOL
    tmp = SpatialSupport(torch.zeros(5, grid * grid, k, 1), torch.ones(5), grid)
    g = torch.Generator().manual_seed(seed)
    protos, sigmas = [], []
    for sid, name in enumerate(STYLES):
        d = torch.load(pool / f"{sid:02d}_{name}.pt", map_location="cpu", weights_only=False)
        x = d["latents"].float()[100:]
        take = torch.randperm(x.shape[0], generator=g)[:max_per_style]
        xb = x[take].to(device)
        with torch.inference_mode():
            reg = tmp.regional_embed(critic, xb)         # (n,R,D)
        n, r, dim = reg.shape
        proto_r, dist_r = [], []
        for j in range(r):
            emb = F.normalize(reg[:, j], dim=-1)
            proto = _kmeans(emb, k, seed=seed + sid * 100 + j)
            assign = torch.cdist(emb, proto).argmin(1)
            d2 = (emb - proto[assign]).square().sum(-1)
            dist_r.append(d2.mean())
            proto_r.append(proto)
        protos.append(torch.stack(proto_r))              # (R,K,D)
        sigmas.append(torch.stack(dist_r).mean().sqrt().clamp_min(1e-3))
    return SpatialSupport(torch.stack(protos), torch.stack(sigmas), grid)


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

    def __init__(self, critic, support: RealSupport | SpatialSupport | None = None,
                 lambda_ratio: float = 0.0, local_support: bool = False):
        self.critic = critic
        self.support = support
        self.lambda_ratio = lambda_ratio
        self.local_support = local_support

    def margin(self, z, tid):
        logits = self.critic(z.float()).float()
        mask = F.one_hot(tid.long(), 5).bool()
        others = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
        return logits.gather(1, tid[:, None]).squeeze(1) - torch.logsumexp(others, 1)

    def value(self, z, tid):
        m = self.margin(z, tid)
        if self.support is None or self.lambda_ratio == 0:
            return m
        if self.local_support and isinstance(self.support, SpatialSupport):
            e = self.support.energy(self.critic, z, tid)
        else:
            emb = F.normalize(_embed(self.critic, z).float(), dim=-1)
            e = self.support.energy(emb, tid)
        return m - self.lambda_ratio * e

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
