"""Direct target-distribution supervision (group D, experiments 16-23).

The generator is pushed by a distribution distance between the FREE ODE endpoint
and real target latents. Style is still the target data distribution; nothing here
is a style classifier or an adversarial loss.

Components
----------
`StyleEncoder`   small contrastive encoder trained on real paintings so that latents
                 of the same style are close. Frozen afterwards.
                 `fixed` mode instead uses raw standardized latents, which is the
                 exp 18/19 control: no extra encoder at all.
`swd2`           squared sliced Wasserstein distance, differentiable.
`chunked_swd2`   two-pass SWD gradient: sort once on a fixed batch, then recompute
                 the forward in chunks so a 1024-sample target batch fits in memory.
                 Verified against a full-backward reference in the tests.

Important constraint from the doc: SWD must act on endpoints obtained by running the
model's own ODE from real source images, never on teacher-forced bridge samples.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn


# --------------------------------------------------------------------------- #
# SWD
# --------------------------------------------------------------------------- #
def _projections(dim: int, n_proj: int, device, dtype, seed: int = 0) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    p = torch.randn(dim, n_proj, generator=g).to(device=device, dtype=dtype)
    return F.normalize(p, dim=0)


def swd2(x: torch.Tensor, y: torch.Tensor, n_proj: int = 64, seed: int = 0,
         proj: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Squared sliced Wasserstein distance between two point clouds (B, D)."""
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("swd2 expects (N, D) and (M, D)")
    p = proj if proj is not None else _projections(x.shape[1], n_proj, x.device, x.dtype, seed)
    px = x @ p
    py = y @ p
    n, m = px.shape[0], py.shape[0]
    q = torch.linspace(0, 1, max(n, m), device=x.device, dtype=x.dtype)
    xq = torch.quantile(px.transpose(0, 1), q, dim=1).transpose(0, 1)
    yq = torch.quantile(py.transpose(0, 1), q, dim=1).transpose(0, 1)
    return (xq - yq).square().mean()


def swd2_chunked(x: torch.Tensor, y: torch.Tensor, n_proj: int = 64, seed: int = 0,
                 chunk: int = 64, subsample: int = 256) -> torch.Tensor:
    """Memory-bounded SWD on large latent feature clouds.

    Under no_grad, draw a FIXED random subset of the real cloud and compute its sorted
    projected quantiles. For the generated cloud, process samples in chunks so that no
    single autograd graph holds the whole batch. The chunk endpoint statistics are
    compared to the same fixed real quantiles; this is the two-pass approximation the
    execution book allows for large target batches.

    `subsample` caps both clouds so 4x64x64 flattened latents (16384-D) cannot create a
    (512, 16384, 64) projection tensor.
    """
    with torch.no_grad():
        n, m = x.shape[0], y.shape[0]
        k = min(subsample, n, m)
        xg = x[torch.randperm(n, device=x.device)[:k]]
        yg = y[torch.randperm(m, device=y.device)[:k]]
        proj = _projections(x.shape[1], n_proj, x.device, x.dtype, seed)
        yq = torch.quantile((yg @ proj).transpose(0, 1),
                            torch.linspace(0, 1, n_proj, device=x.device, dtype=x.dtype),
                            dim=1).transpose(0, 1).detach()
    total, count = x.new_zeros(()), 0
    for s in range(0, k, chunk):
        xs = xg[s:s + chunk]
        px = xs @ proj
        # compare the chunk's sorted projections with the matching quantiles of real
        q = torch.linspace(0, 1, n_proj, device=x.device, dtype=x.dtype)
        xq = torch.quantile(px.transpose(0, 1), q, dim=1).transpose(0, 1)
        total = total + (xq - yq).square().mean() * len(px)
        count += len(px)
    return total / max(count, 1)


def swd2_local(x_patch: torch.Tensor, y_patch: torch.Tensor, n_proj: int = 32,
               seed: int = 0, chunk: int = 64, subsample: int = 256) -> torch.Tensor:
    """Local (patch) SWD: x_patch/y_patch are (B, P, D); compares pooled patch clouds."""
    if x_patch.ndim != 3 or y_patch.ndim != 3:
        raise ValueError("swd2_local expects (B, P, D)")
    xf = x_patch.reshape(-1, x_patch.shape[-1])
    yf = y_patch.reshape(-1, y_patch.shape[-1])
    return swd2_chunked(xf, yf, n_proj, seed, chunk, subsample)


def chunked_swd2(feature_fn, x: torch.Tensor, y: torch.Tensor, chunk: int = 128,
                 n_proj: int = 64, seed: int = 0) -> torch.Tensor:
    """Two-pass SWD when the endpoint batch is large.

    Pass 1 computes quantiles under no_grad with a fixed projection set; pass 2
    re-runs the feature function in chunks and compares against those fixed
    quantiles, so memory stays bounded while the gradient is still exact for the
    chosen quantile targets.

    This is only an approximation of the full-backward gradient; the tests compare
    it against the exact version on a small batch and assert closeness.
    """
    proj = _projections(x.shape[1], n_proj, x.device, x.dtype, seed)
    with torch.no_grad():
        px = feature_fn(x) @ proj
        py = feature_fn(y) @ proj
        q = torch.linspace(0, 1, max(px.shape[0], py.shape[0]), device=x.device, dtype=x.dtype)
        yq = torch.quantile(py.transpose(0, 1), q, dim=1).transpose(0, 1)
        xq = torch.quantile(px.transpose(0, 1), q, dim=1).transpose(0, 1)
    total, count = 0.0, 0
    for s in range(0, x.shape[0], chunk):
        fx = feature_fn(x[s:s + chunk]) @ proj
        n = fx.shape[0]
        xq_c = torch.quantile(fx.transpose(0, 1), q[:n], dim=1).transpose(0, 1)
        total = total + (xq_c - yq[:n]).square().sum()
        count += n
    return total / max(count, 1)


# --------------------------------------------------------------------------- #
# style encoder
# --------------------------------------------------------------------------- #
class StyleEncoder(nn.Module):
    """Small conv encoder over the 32x16x16 latent, trained with supervised contrastive loss.

    Returns a global embedding and a spatial patch embedding so group D can compare
    global style statistics (16-19) against local brush-stroke statistics (20-23).
    """

    def __init__(self, channels: int = 32, width: int = 64, out_dim: int = 64, patch: int = 2):
        super().__init__()
        self.patch = patch
        self.body = nn.Sequential(
            nn.Conv2d(channels, width, 3, padding=1), nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1, stride=2), nn.SiLU(),
            nn.Conv2d(width, out_dim, 3, padding=1), nn.SiLU(),
        )
        self.head = nn.Linear(out_dim, out_dim)

    def forward(self, z):
        h = self.body(z)                       # (B, out_dim, H/2, W/2)
        g = F.normalize(self.head(h.mean((2, 3))), dim=-1)
        # patch tokens: unfold into non-overlapping patches then flatten spatially
        p = self.patch
        B, C, H, W = h.shape
        if H % p or W % p:
            return g, h.flatten(2).transpose(1, 2)
        ph, pw = H // p, W // p
        patches = (h.reshape(B, C, ph, p, pw, p).permute(0, 2, 4, 1, 3, 5)
                   .reshape(B, ph * pw, C * p * p))
        return g, patches


def supervised_contrastive(emb: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1):
    """Supervised contrastive loss (Khosla et al.) for style labels."""
    z = F.normalize(emb, dim=-1)
    sim = z @ z.t() / temperature
    n = z.shape[0]
    mask = torch.eye(n, device=z.device, dtype=torch.bool)
    pos = (labels.view(-1, 1) == labels.view(1, -1)) & ~mask
    logits = sim.masked_fill(mask, -1e9)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    pos_count = pos.sum(1).clamp_min(1)
    return -(log_prob.masked_fill(~pos, 0.0).sum(1) / pos_count).mean()


def train_style_encoder(latents: Sequence[torch.Tensor], epochs: int = 60, batch: int = 256,
                        lr: float = 2e-3, device: str = "cuda", seed: int = 0,
                        log=None) -> StyleEncoder:
    """Fit the contrastive style encoder on real latents; returns a frozen encoder."""
    torch.manual_seed(seed)
    channels = latents[0].shape[1]
    enc = StyleEncoder(channels=channels).to(device)
    opt = torch.optim.AdamW(enc.parameters(), lr=lr, weight_decay=1e-4)
    pools = [z.to(device) for z in latents]
    labels_all = torch.cat([torch.full((len(p),), i, device=device, dtype=torch.long)
                            for i, p in enumerate(pools)])
    gen = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        idx = torch.randperm(len(labels_all), generator=gen)
        tot, nb = 0.0, 0
        for s in range(0, len(idx), batch):
            sel = idx[s:s + batch]
            if len(sel) < 8:
                continue
            # gather samples from the right style pools
            zs = []
            for i in sel.tolist():
                zs.append(pools[int(labels_all[i])][torch.randint(len(pools[int(labels_all[i])]),
                                                                 (1,), generator=gen).item()])
            z = torch.stack(zs, 0)
            emb, _ = enc(z)
            loss = supervised_contrastive(emb, labels_all[sel])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss)
            nb += 1
        if log is not None and (ep % 20 == 0 or ep == epochs - 1):
            log(f"[style-enc] epoch {ep} loss={tot / max(nb, 1):.4f}")
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc


# --------------------------------------------------------------------------- #
# distribution loss wrapper
# --------------------------------------------------------------------------- #
class DistributionLoss:
    """Computes L_dist between free-ODE endpoints and real target latents.

    mode: "global" | "local" | "both"
    encoder: "learned" (StyleEncoder) | "fixed" (identity on standardized latents)
    """

    def __init__(self, mode: str = "global", encoder: str = "learned",
                 style_enc: Optional[StyleEncoder] = None, n_proj: int = 64,
                 chunk: int = 128):
        if mode not in ("global", "local", "both"):
            raise ValueError(mode)
        if encoder not in ("learned", "fixed"):
            raise ValueError(encoder)
        self.mode, self.encoder_kind = mode, encoder
        self.enc = style_enc
        self.n_proj, self.chunk = n_proj, chunk

    def _features(self, z: torch.Tensor):
        if self.encoder_kind == "fixed":
            g = z.flatten(1)
            # local patches for the fixed-statistics variant
            B, C, H, W = z.shape
            p = 4
            ph, pw = H // p, W // p
            patches = (z.reshape(B, C, ph, p, pw, p).permute(0, 2, 4, 1, 3, 5)
                       .reshape(B, ph * pw, C * p * p))
            return g, patches
        return self.enc(z)

    def __call__(self, z_gen: torch.Tensor, z_real: torch.Tensor, seed: int = 0,
                 groups: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Distribution distance between free-ODE endpoints and real target latents.

        `groups` (optional) holds a target-style id per sample. When given, the distance is
        computed SEPARATELY for each target style and then averaged with equal weight, as
        the experiment book requires: merging a mixed batch into one SWD would only match
        the overall mixture and could not demand that each target domain be correct.
        """
        if groups is None:
            g_gen, p_gen = self._features(z_gen)
            with torch.no_grad():
                g_real, p_real = self._features(z_real)
            loss = z_gen.new_zeros(())
            if self.mode in ("global", "both"):
                loss = loss + swd2_chunked(g_gen, g_real, self.n_proj, seed,
                                           self.chunk, min(256, g_gen.shape[0]))
            if self.mode in ("local", "both"):
                loss = loss + swd2_local(p_gen, p_real, max(16, self.n_proj // 2),
                                         seed, self.chunk)
            return loss

        total, count = z_gen.new_zeros(()), 0
        for gi in torch.unique(groups):
            m = (groups == gi)
            if int(m.sum()) < 2:
                continue
            total = total + self(z_gen[m], z_real[m], seed + int(gi), groups=None)
            count += 1
        return total / max(count, 1)
