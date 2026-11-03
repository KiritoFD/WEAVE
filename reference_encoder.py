"""Route C: reference-image Style Encoder.

Style representation is split into two parts:

    s = (s_family, s_reference)

* ``s_family``    — the existing learnable Style Memory indexed by Style-ID,
                    kept as the category-level prior (ablations showed it helps).
* ``s_reference`` — produced HERE: frozen pretrained ViT/DINO patch features of
                    the reference image -> small trainable encoder whose
                    spatial-layout information is deliberately suppressed
                    (train-time patch permutation + per-patch InstanceNorm +
                    learned attention pooling to a small token set).

Training signal: multi-reference InfoNCE. Given several reference images of the
SAME style but different content, the encoder is trained so that same-style
embeddings attract and different-style embeddings repel. One single pooled
reference cannot separate "style" from "this painting's objects" — multiple
same-style references can.

Conditioning injection (in model.py): family tokens and reference tokens are
combined by gated addition at the global-token level and the spatial placement
of texture remains decided by CONTENT features inside the backbone blocks. We
deliberately do NOT inject raw spatial HF maps of the reference (that failed
in the July experiments: the model copied the reference layout).

Old Style-ID-only path is untouched and remains the default
(``model.conditioner_variant = "style_id"``).
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
from torch import nn


class ReferenceStyleEncoder(nn.Module):
    """Encode reference-image patch features into layout-free style tokens.

    Input:  ref_feats (B, N, D_in) — patch tokens from a frozen extractor
            (e.g. DINOv2 intermediate block, or TinyConv unfolded patches in
            smoke tests).
    Output: (B, M, model_dim) style tokens + (B, proj_dim) embedding for
            contrastive learning.
    """

    def __init__(
        self,
        in_dim: int,
        model_dim: int,
        *,
        num_tokens: int = 32,
        num_layers: int = 1,
        num_heads: int = 4,
        proj_dim: int = 128,
        shuffle_train: bool = True,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.model_dim = int(model_dim)
        self.num_tokens = int(num_tokens)
        self.shuffle_train = bool(shuffle_train)

        self.in_norm = nn.LayerNorm(self.in_dim)
        self.in_proj = nn.Linear(self.in_dim, self.model_dim)
        # Per-patch InstanceNorm over the token axis removes absolute,
        # position-specific magnitude information (keeps relative texture stats).
        self.patch_inorm = nn.InstanceNorm1d(self.model_dim, affine=True)
        layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim, nhead=num_heads,
            dim_feedforward=self.model_dim * 2, batch_first=True,
            norm_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=max(1, int(num_layers)))
        # Learned queries pool patch features into a small token set; pooling
        # weights come from content-free queries so spatial layout cannot pass
        # through positionally.
        self.queries = nn.Parameter(torch.randn(self.num_tokens, self.model_dim) * 0.02)
        self.pool_attn = nn.MultiheadAttention(self.model_dim, num_heads, batch_first=True)
        self.out_norm = nn.LayerNorm(self.model_dim)
        self.contrast_head = nn.Sequential(
            nn.LayerNorm(self.model_dim),
            nn.Linear(self.model_dim, proj_dim),
        )

    def forward(self, ref_feats: torch.Tensor) -> dict[str, torch.Tensor]:
        B, N, D = ref_feats.shape
        if D != self.in_dim:
            raise ValueError(f"ref_feats last dim must be {self.in_dim}, got {D}")
        h = ref_feats.float()
        if self.training and self.shuffle_train and N > 1:
            # Suppress spatial layout: randomly permute patch order per sample.
            perm = torch.argsort(torch.rand(B, N, device=h.device), dim=1)
            h = torch.gather(h, 1, perm.unsqueeze(-1).expand(-1, -1, D))
        h = self.in_proj(self.in_norm(h))  # (B, N, model_dim)
        h = self.patch_inorm(h.transpose(1, 2)).transpose(1, 2)
        h = self.encoder(h)
        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        tokens, _ = self.pool_attn(q, h, h)  # (B, M, model_dim)
        tokens = self.out_norm(tokens)
        emb = F.normalize(self.contrast_head(tokens.mean(dim=1)), p=2, dim=-1)  # (B, proj_dim)
        return {"tokens": tokens.to(ref_feats.dtype), "embedding": emb}


def info_nce_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    *,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Multi-reference InfoNCE over L2-normalised embeddings.

    anchor:   (B, D) embeddings of reference view A (style s, content i).
    positive: (B, D) embeddings of reference view B (same style, different content).
    In-batch negatives: all other anchors (different styles in the batch).
    """
    a = F.normalize(anchor.float(), p=2, dim=-1)
    p = F.normalize(positive.float(), p=2, dim=-1)
    logits = a @ p.t() / float(temperature)  # (B, B)
    labels = torch.arange(a.shape[0], device=a.device)
    return F.cross_entropy(logits, labels)


class ReferenceConditioner(nn.Module):
    """Combine family (Style Memory) tokens with reference style tokens.

    out = family_tokens + tanh(gate) * ref_tokens_expanded

    Gate starts at 0 by default so enabling the path is a strict no-op at
    initialisation (existing behaviour preserved until training moves it).
    """

    def __init__(self, model_dim: int, num_family_tokens: int, num_ref_tokens: int, gate_init: float = 0.0) -> None:
        super().__init__()
        self.num_family_tokens = int(num_family_tokens)
        self.num_ref_tokens = int(num_ref_tokens)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        self.ref_proj = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
        )

    def forward(self, family_tokens: torch.Tensor, ref_tokens: torch.Tensor) -> torch.Tensor:
        ref = self.ref_proj(ref_tokens.float()).to(family_tokens.dtype)
        if ref.shape[1] != family_tokens.shape[1]:
            ref = F.interpolate(
                ref.transpose(1, 2), size=family_tokens.shape[1],
                mode="linear", align_corners=False,
            ).transpose(1, 2)
        return family_tokens + torch.tanh(self.gate).to(family_tokens.dtype) * ref
