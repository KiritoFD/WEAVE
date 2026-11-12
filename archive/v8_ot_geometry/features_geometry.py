"""Per-patch descriptors aligned on a common 8x8 grid of the 64x64 latent.

geometry  : training-free structure (orientation, coherence, relative edge
            rank, local self-similarity, flat/contour/texture memberships).
            Computed from the SOURCE and held fixed during optimisation.
            Uses orientation and RELATIVE statistics, never absolute edge
            magnitude: a target style is allowed to change stroke width.
style_h   : raw Haar high-frequency patches (phase sensitive).
style_u   : amplitude/covariance statistics, more robust to phase.
style_ll  : low-frequency colour statistics.

All three style views are kept separate because HF alone is not "style".
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

N_PATCH = 64
GEOM_PATCH = 8
HF_PATCH = 4
GEOM_DIM = 48


def haar_hf(x: torch.Tensor) -> torch.Tensor:
    """2-D Haar high-frequency subbands at half resolution -> (B,12,32,32)."""
    a = x[:, :, 0::2, 0::2]
    b = x[:, :, 1::2, 0::2]
    c = x[:, :, 0::2, 1::2]
    d = x[:, :, 1::2, 1::2]
    return torch.cat([(a - b + c - d) * .5, (a + b - c - d) * .5,
                      (a - b - c + d) * .5], dim=1)


def haar_ll(x: torch.Tensor) -> torch.Tensor:
    a = x[:, :, 0::2, 0::2]
    b = x[:, :, 1::2, 0::2]
    c = x[:, :, 0::2, 1::2]
    d = x[:, :, 1::2, 1::2]
    return (a + b + c + d) * .5


class GeometryDescriptor:
    """(B,64,48) training-free geometry, differentiable w.r.t. the input."""

    def __init__(self, eps: float = 1e-6):
        self.eps = eps

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        x = z.float()
        gx = F.pad(x[..., 1:] - x[..., :-1], (0, 1, 0, 0), mode="replicate")
        gy = F.pad(x[..., 1:, :] - x[..., :-1, :], (0, 0, 0, 1), mode="replicate")
        jxx = F.avg_pool2d((gx * gx).mean(1, True), 3, 1, 1)
        jyy = F.avg_pool2d((gy * gy).mean(1, True), 3, 1, 1)
        jxy = F.avg_pool2d((gx * gy).mean(1, True), 3, 1, 1)
        anis = ((jxx - jyy).square() + 4 * jxy.square() + self.eps ** 2).sqrt()
        trace = jxx + jyy + self.eps
        co, si = (jxx - jyy) / anis, 2 * jxy / anis
        coherence = anis / trace
        edge = trace.sqrt()
        strength = edge / (edge.mean((-2, -1), True) + self.eps)
        flat = torch.sigmoid(4 * (0.4 - strength))
        texture = torch.sigmoid(3 * (strength - 1.6))
        contour = (1 - flat - texture).clamp_min(0)
        # multi-scale relative edge ranking (scale-invariant)
        ranks = []
        for k in (8, 16, 32):
            r = F.avg_pool2d(edge, k, k).flatten(1)
            ranks.append((r.argsort(1).argsort(1).float() / (r.shape[1] - 1)).view(r.shape[0], 1, 64 // k, 64 // k))
        rank8 = F.interpolate(ranks[0], size=(64, 64), mode="nearest")
        rank16 = F.interpolate(ranks[1], size=(64, 64), mode="nearest")
        rank32 = F.interpolate(ranks[2], size=(64, 64), mode="nearest")
        loc = F.avg_pool2d(co, 5, 1, 2)
        selfsim = (co - loc).square() + (si - F.avg_pool2d(si, 5, 1, 2)).square()
        # anisotropy magnitude is a distinct cue from coherence: coherence is
        # normalised, this keeps the absolute curvature of the structure tensor.
        aniso = anis / (anis.mean((-2, -1), True) + self.eps)
        # multi-scale structure: the same statistics at 3 blur levels, so a
        # "flat background" and a "detailed texture" separate at the scale the
        # style actually operates on.
        scales = []
        for k in (1, 3, 7):
            base = (co, si, coherence, strength) if k == 1 else tuple(
                F.avg_pool2d(t, k, 1, k // 2) for t in (co, si, coherence, strength))
            scales.extend(base)
        feat = torch.cat([co, si, coherence, strength, selfsim, flat, contour,
                          texture, aniso, rank8, rank16, rank32] + scales, dim=1)
        p = F.unfold(feat, kernel_size=GEOM_PATCH, stride=GEOM_PATCH).reshape(
            x.shape[0], feat.shape[1], N_PATCH, -1)
        m, sd = p.mean(-1), p.std(-1)
        g = torch.cat([m, sd], dim=1).permute(0, 2, 1)
        g = (g - g.mean(1, keepdim=True)) / (g.std(1, keepdim=True) + self.eps)
        if g.shape[-1] != GEOM_DIM:
            raise ValueError(f"geometry descriptor must be {GEOM_DIM}-d, got {g.shape[-1]}")
        return g


class StyleDescriptor:
    """Returns {'h':(B,64,192), 'u':(B,64,102), 'll':(B,64,18)}."""

    def __call__(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        x = z.float()
        hf = haar_hf(x)
        hp = F.unfold(hf, kernel_size=HF_PATCH, stride=HF_PATCH).permute(0, 2, 1)  # (B,64,192)
        # amplitude/covariance view: per-channel energy and adjacent-channel corr
        # amplitude view: per-channel energy of the 12 HF channels, so a style's
        # channel-wise stroke energy is preserved rather than summed away.
        amp = hp.reshape(x.shape[0], N_PATCH, 12, -1).square().mean(-1)   # (B,64,12)
        chan = hf.reshape(x.shape[0], 12, -1)
        cov = (chan @ chan.transpose(1, 2)) / chan.shape[-1]              # (B,12,12)
        cov = cov.reshape(x.shape[0], 1, 144).expand(-1, N_PATCH, -1)     # (B,64,144)
        u = torch.cat([amp, torch.sqrt(amp.clamp_min(1e-8)), cov], dim=-1)[..., :102]
        ll = haar_ll(x)
        lp = F.unfold(ll, kernel_size=HF_PATCH, stride=HF_PATCH).permute(0, 2, 1)  # (B,64,12)
        ll_mean = lp.mean(-1, keepdim=True).expand(-1, -1, 3)
        ll_std = lp.std(-1, keepdim=True).expand(-1, -1, 3)
        ll_norm = (lp - lp.mean(-1, keepdim=True)) / (lp.std(-1, keepdim=True) + 1e-6)
        ll_view = torch.cat([ll_mean, ll_std, ll_norm[..., :12]], dim=-1)
        return {"h": hp, "u": u, "ll": ll_view}
