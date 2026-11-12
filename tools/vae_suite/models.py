"""WEAVE Art-VAE Suite: Multi-Scale Latent Feature Extractor, A1 Classifier & A2 MoCo Queue.

Implements:
1. Multi-scale Latent Feature Descriptor F(z):
   F(z) = [AdaptiveAvgPool(8x8), Mean, Std, DWTStats] -> 296-dimensional vector.
2. A1 Auxiliary Classifier Head:
   Linear layer mapping 296-d feature to 137 style logits.
3. A2 MoCo Momentum Encoder & Queue:
   Maintains momentum encoder (EMA = 0.999), 4096-item memory bank, and contrastive projector.
"""
from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.v6.canonicalize import dwt2_haar


class MultiScaleLatentExtractor(nn.Module):
    """Extracts a 296-dimensional multi-scale descriptor from latent z (B, 4, 64, 64)."""

    def __init__(self):
        super().__init__()
        self.pool8 = nn.AdaptiveAvgPool2d((8, 8))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, 4, 64, 64)
        B = z.shape[0]

        # 1. Spatial 8x8 pooled tokens: (B, 256)
        p8 = self.pool8(z).flatten(1)

        # 2. Global channel mean & std: (B, 4) each
        m = z.mean(dim=(2, 3))
        s = z.std(dim=(2, 3)).clamp(min=1e-5)

        # 3. DWT subband statistics (LL, LH, HL, HH): 4 channels x 4 bands x 2 stats = 32-d
        ll, lh, hl, hh = dwt2_haar(z)
        dwt_stats = []
        for band in (ll, lh, hl, hh):
            dwt_stats.append(band.mean(dim=(2, 3)))
            dwt_stats.append(band.std(dim=(2, 3)).clamp(min=1e-5))
        dwt_cat = torch.cat(dwt_stats, dim=1) # (B, 32)

        # Total: 256 + 4 + 4 + 32 = 296 dims
        feat = torch.cat([p8, m, s, dwt_cat], dim=1)
        return feat


class A1StyleClassifier(nn.Module):
    """Simple linear auxiliary classifier head for 137 style classes."""

    def __init__(self, in_features: int = 296, num_styles: int = 137):
        super().__init__()
        self.extractor = MultiScaleLatentExtractor()
        self.classifier = nn.Linear(in_features, num_styles)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        feat = self.extractor(z)
        logits = self.classifier(feat)
        return logits


class A2MoCoQueue(nn.Module):
    """MoCo-style Momentum Encoder, Projector and Memory Queue for A2 Contrastive Learning."""

    def __init__(
        self,
        base_encoder: nn.Module,
        feature_dim: int = 296,
        proj_dim: int = 128,
        queue_size: int = 4096,
        momentum: float = 0.999,
        temperature: float = 0.10,
    ):
        super().__init__()
        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature

        self.extractor = MultiScaleLatentExtractor()

        # Projector for query
        self.projector_q = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, proj_dim),
        )

        # Momentum encoder and key projector
        self.encoder_k = copy.deepcopy(base_encoder)
        self.projector_k = copy.deepcopy(self.projector_q)

        # Freeze key encoder & projector from direct autograd
        for p in self.encoder_k.parameters():
            p.requires_grad = False
        for p in self.projector_k.parameters():
            p.requires_grad = False

        # Memory queues
        self.register_buffer("queue_feats", torch.randn(proj_dim, queue_size))
        self.queue_feats = F.normalize(self.queue_feats, dim=0)
        self.register_buffer("queue_styles", torch.full((queue_size,), -1, dtype=torch.long))
        self.register_buffer("queue_artists", torch.full((queue_size,), -1, dtype=torch.long))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def update_momentum_encoder(self, base_encoder: nn.Module):
        """EMA update of key encoder and key projector."""
        for param_q, param_k in zip(base_encoder.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)
        for param_q, param_k in zip(self.projector_q.parameters(), self.projector_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)

    def extract_query(self, z_query: torch.Tensor) -> torch.Tensor:
        feat = self.extractor(z_query)
        q = F.normalize(self.projector_q(feat), dim=-1)
        return q

    @torch.no_grad()
    def extract_key(self, x_images: torch.Tensor) -> torch.Tensor:
        # Encode with key encoder (posterior mean)
        h = self.encoder_k(x_images)
        # Note: in SD1.5 AutoencoderKL, encoder outputs 8 channels (mean, logvar)
        # quant_conv maps to 8 channels, then split
        z_k = h[:, :4]
        feat_k = self.extractor(z_k)
        k = F.normalize(self.projector_k(feat_k), dim=-1)
        return k

    @torch.no_grad()
    def enqueue(self, keys: torch.Tensor, styles: torch.Tensor, artists: torch.Tensor):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr[0])

        if ptr + batch_size <= self.queue_size:
            self.queue_feats[:, ptr : ptr + batch_size] = keys.T
            self.queue_styles[ptr : ptr + batch_size] = styles
            self.queue_artists[ptr : ptr + batch_size] = artists
            ptr = (ptr + batch_size) % self.queue_size
        else:
            # Wrap around queue
            rem = self.queue_size - ptr
            self.queue_feats[:, ptr:] = keys[:rem].T
            self.queue_styles[ptr:] = styles[:rem]
            self.queue_artists[ptr:] = artists[:rem]

            overflow = batch_size - rem
            self.queue_feats[:, :overflow] = keys[rem:].T
            self.queue_styles[:overflow] = styles[rem:]
            self.queue_artists[:overflow] = artists[rem:]
            ptr = overflow

        self.queue_ptr[0] = ptr
