"""Losses for the Art-VAE encoder.

* SupCon   : multi-positive supervised contrastive loss on the FIXED style
             readout q = r(z). Relation weights encode the three positive
             relations from the plan; no CE head, no triplet, no projector.
* VICReg   : lightweight variance floor, calibrated to the readout dimension.
* Recon    : the already-validated S10 recipe (L1 + LPIPS@512 + weak KL).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Relative weights for the three positive relations (plan section 3.1/3.3).
W_CROSS_ARTIST = 1.0     # different painter, same school  <- the important one
W_SAME_ARTIST = 0.5      # same painter, different content
W_SAME_VIEW = 0.25       # two crops of the same work (avoid dominating)

TAU = 0.1                # fixed, no temperature search


def supcon_relation_weights(style, artist, view, supervised=None,
                            w_cross: float = W_CROSS_ARTIST,
                            w_same_artist: float = W_SAME_ARTIST,
                            w_view: float = W_SAME_VIEW) -> torch.Tensor:
    """(B,B) non-negative positive-pair weight matrix.

    A pair is a positive only when the STYLE matches. Within that, the relation
    decides the weight. `view` identifies the underlying work, so two crops of
    one painting share a view id while different paintings do not.
    """
    B = style.shape[0]
    eye = torch.eye(B, dtype=torch.bool, device=style.device)
    same_style = style[:, None] == style[None, :]
    diff_artist = artist[:, None] != artist[None, :]
    same_view = view[:, None] == view[None, :]

    same_school_other_painter = same_style & diff_artist
    same_painter_other_work = same_style & ~diff_artist & ~same_view
    same_work_other_crop = same_style & same_view & ~eye

    # Schools held out from supervision must NEVER be positives for each other:
    # they may appear as content-matched negatives only. Without this gate a
    # batch would implicitly supervise the very schools we want to test.
    if supervised is not None:
        eligible = supervised[:, None] & supervised[None, :]
        same_style = same_style & eligible

    W = torch.zeros(B, B, device=style.device, dtype=torch.float32)
    W[same_style & diff_artist] = w_cross
    W[same_style & ~diff_artist & ~same_view] = w_same_artist
    W[same_style & same_view & ~eye] = w_view
    W.masked_fill_(eye, 0.0)
    return W


def supcon(q: torch.Tensor, W: torch.Tensor, temperature: float = TAU) -> torch.Tensor:
    """Multi-positive SupCon. q is (B,D) L2-normalised, W is (B,B) weights."""
    B = q.shape[0]
    eye = torch.eye(B, dtype=torch.bool, device=q.device)
    sim = (q @ q.t()) / temperature
    logits = sim.masked_fill(eye, float("-inf"))
    logp = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    # 0 * (-inf) == nan on the diagonal; neutralise before weighting
    logp = logp.masked_fill(eye, 0.0)
    num = (W * logp).sum(dim=1)
    den = W.sum(dim=1)
    valid = den > 0
    if not bool(valid.any()):
        return q.new_zeros(())
    return -(num[valid] / den[valid]).mean()


def gamma_for_dim(d: int, target: float = 0.5) -> float:
    """Variance floor calibrated to the L2-normalised readout dimension.

    For a unit-norm vector in d dims, a healthy per-dimension std is ~1/sqrt(d).
    Requiring std >= target/sqrt(d) prevents collapse without importing a
    default threshold tuned for some other (much lower-dimensional) feature.
    """
    return target / math.sqrt(d)


def vicreg_var(q: torch.Tensor, gamma: float) -> torch.Tensor:
    """mean_j max(0, gamma - Std(q[:,j])) over the logical batch."""
    std = q.std(dim=0, unbiased=False)
    return F.relu(gamma - std).mean()


def kl_divergence(post) -> tuple[torch.Tensor, torch.Tensor]:
    """(per-sample-sum mean, elementwise mean) -- diffusers vs our convention."""
    per = 0.5 * (post.logvar.exp() + post.mean.pow(2) - 1.0 - post.logvar)
    return per.sum(dim=[1, 2, 3]).mean(), per.mean()


class ReconstructionLoss(nn.Module):
    """L1 + lam_p * LPIPS@512 + beta * KL, the S10-distillation recipe."""

    def __init__(self, lpips_fn=None, lam_p: float = 0.5, beta: float = 1e-4,
                 kl_reduction: str = "elem", lpips_size: int = 512):
        super().__init__()
        self.lpips_fn = lpips_fn
        self.lam_p = lam_p
        self.beta = beta
        self.kl_reduction = kl_reduction
        self.lpips_size = lpips_size

    def forward(self, x, y, post):
        y = y.float()
        l1 = (y - x).abs().mean()
        parts = {"l1": float(l1.detach())}
        total = l1
        if self.lpips_fn is not None:
            if self.lpips_size and self.lpips_size != y.shape[-1]:
                ys = F.interpolate(y, self.lpips_size, mode="bilinear",
                                   align_corners=False)
                xs = F.interpolate(x, self.lpips_size, mode="bilinear",
                                   align_corners=False)
            else:
                ys, xs = y, x
            l_p = self.lpips_fn(ys, xs).mean()
            total = total + self.lam_p * l_p
            parts["lpips"] = float(l_p.detach())
        kl_sum, kl_elem = kl_divergence(post)
        total = total + self.beta * (kl_sum if self.kl_reduction == "sum" else kl_elem)
        parts["kl_sum"] = float(kl_sum.detach())
        parts["kl_elem"] = float(kl_elem.detach())
        return total, parts
