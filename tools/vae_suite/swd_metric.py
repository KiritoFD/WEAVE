"""WEAVE-VAE V2: Multi-Scale Sliced Wasserstein Distance & Moment Metric on Latent Space.

Defines a parameter-free, fully differentiable distance on SD VAE latents (4x64x64):
1. Normalization by channel mean & std.
2. Two-Scale Local Patches:
   - Scale 1: Unfold 3x3 on z (36-d patch, 256 grid locations)
   - Scale 2: Unfold 3x3 on AvgPool2(z) (36-d patch, 256 grid locations)
3. 32 Fixed Random Projections -> 16 Sorted Uniform Quantiles -> Sliced Wasserstein.
4. Moment Descriptor: 4-channel mean + 10-element covariance upper triangle.
5. Calibrated Distance Combination:
   d(z_i, z_j) = 0.4 * d_swd1/scale1 + 0.4 * d_swd2/scale2 + 0.2 * d_moment/scale_m
   normalized by median inter-class distance.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentSWDDescriptor(nn.Module):
    """Extracts non-learnable multi-scale SWD quantiles and moments from raw latent z."""

    def __init__(
        self,
        num_projections: int = 32,
        num_quantiles: int = 16,
        num_samples: int = 256,
        seed: int = 42,
    ):
        super().__init__()
        self.num_projections = num_projections
        self.num_quantiles = num_quantiles
        self.num_samples = num_samples

        # 36-dimensional patch (4 channels * 3 * 3 = 36)
        patch_dim = 4 * 3 * 3

        # 32 fixed random projection directions (unit norm)
        rng = torch.Generator().manual_seed(seed)
        v = torch.randn(patch_dim, num_projections, generator=rng)
        v = F.normalize(v, p=2, dim=0) # (36, 32)
        self.register_buffer("proj_directions", v)

        # Fixed 256 sampling grid indices for Scale 1 (64x64 = 4096 positions)
        idx_scale1 = torch.linspace(0, 4095, num_samples).long()
        self.register_buffer("grid_idx_scale1", idx_scale1)

        # Fixed 256 sampling grid indices for Scale 2 (32x32 = 1024 positions)
        idx_scale2 = torch.linspace(0, 1023, num_samples).long()
        self.register_buffer("grid_idx_scale2", idx_scale2)

        # 16 quantile index positions from 0 to 255
        q_idx = torch.linspace(0, num_samples - 1, num_quantiles).round().long()
        self.register_buffer("quantile_idx", q_idx)

        # Upper triangle indices for 4x4 covariance
        tri_r, tri_c = torch.triu_indices(4, 4)
        self.register_buffer("tri_r", tri_r)
        self.register_buffer("tri_c", tri_c)

        # Default A0 channel mean & std (will be updated by calibration)
        self.register_buffer("chan_mean", torch.tensor([-0.18, 0.05, -0.06, -0.12]).view(1, 4, 1, 1))
        self.register_buffer("chan_std", torch.tensor([4.65, 4.65, 4.65, 4.65]).view(1, 4, 1, 1))

        # Scaling constants for distance components (set by calibration)
        self.register_buffer("scale_swd1", torch.tensor(1.0))
        self.register_buffer("scale_swd2", torch.tensor(1.0))
        self.register_buffer("scale_moment", torch.tensor(1.0))
        self.register_buffer("median_inter_dist", torch.tensor(1.0))

    def set_calibration(
        self,
        chan_mean: torch.Tensor,
        chan_std: torch.Tensor,
        scale_swd1: float,
        scale_swd2: float,
        scale_moment: float,
        median_inter_dist: float,
    ):
        self.chan_mean.copy_(chan_mean.view(1, 4, 1, 1))
        self.chan_std.copy_(chan_std.view(1, 4, 1, 1))
        self.scale_swd1.fill_(float(scale_swd1))
        self.scale_swd2.fill_(float(scale_swd2))
        self.scale_moment.fill_(float(scale_moment))
        self.median_inter_dist.fill_(float(median_inter_dist))

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extracts descriptors from raw latent z (B, 4, 64, 64).

        Returns:
            q1: Scale 1 quantiles (B, 32, 16)
            q2: Scale 2 quantiles (B, 32, 16)
            moments: (B, 14) (4 means + 10 covariances)
        """
        B, C, H, W = z.shape
        # 1. Normalize
        z_norm = (z - self.chan_mean) / self.chan_std.clamp(min=1e-5)

        # 2. Scale 1: Unfold 3x3 on 64x64 -> (B, 36, 4096)
        patches1 = F.unfold(z_norm, kernel_size=3, padding=1)
        sampled_patches1 = patches1[:, :, self.grid_idx_scale1] # (B, 36, 256)
        # Project: (B, 256, 36) @ (36, 32) -> (B, 256, 32) -> (B, 32, 256)
        proj1 = torch.matmul(sampled_patches1.transpose(1, 2), self.proj_directions).transpose(1, 2)
        proj1_sorted = torch.sort(proj1, dim=-1)[0]
        q1 = proj1_sorted[:, :, self.quantile_idx] # (B, 32, 16)

        # 3. Scale 2: AvgPool2(z_norm) 32x32 -> Unfold 3x3 -> (B, 36, 1024)
        z_pool2 = F.avg_pool2d(z_norm, kernel_size=2, stride=2)
        patches2 = F.unfold(z_pool2, kernel_size=3, padding=1)
        sampled_patches2 = patches2[:, :, self.grid_idx_scale2] # (B, 36, 256)
        proj2 = torch.matmul(sampled_patches2.transpose(1, 2), self.proj_directions).transpose(1, 2)
        proj2_sorted = torch.sort(proj2, dim=-1)[0]
        q2 = proj2_sorted[:, :, self.quantile_idx] # (B, 32, 16)

        # 4. Moment Descriptor
        mu = z_norm.mean(dim=(2, 3)) # (B, 4)
        z_flat = z_norm.view(B, 4, -1) - mu.unsqueeze(-1)
        cov = torch.bmm(z_flat, z_flat.transpose(1, 2)) / float(H * W - 1) # (B, 4, 4)
        cov_tri = cov[:, self.tri_r, self.tri_c] # (B, 10)
        moments = torch.cat([mu, cov_tri], dim=1) # (B, 14)

        return q1, q2, moments

    def compute_pairwise_distance(
        self,
        q1_a: torch.Tensor,
        q2_a: torch.Tensor,
        mom_a: torch.Tensor,
        q1_b: torch.Tensor,
        q2_b: torch.Tensor,
        mom_b: torch.Tensor,
    ) -> torch.Tensor:
        """Computes pairwise calibrated distance matrix D between set A and set B.

        Inputs:
            q1_a: (N, 32, 16), q1_b: (M, 32, 16)
            q2_a: (N, 32, 16), q2_b: (M, 32, 16)
            mom_a: (N, 14), mom_b: (M, 14)
        Returns:
            dist_mat: (N, M) distance matrix
        """
        N, _, _ = q1_a.shape
        M, _, _ = q1_b.shape

        # SWD 1: mean squared difference across 32x16
        diff1 = q1_a.unsqueeze(1) - q1_b.unsqueeze(0) # (N, M, 32, 16)
        d_swd1 = diff1.pow(2).mean(dim=(-2, -1)) # (N, M)

        # SWD 2: mean squared difference across 32x16
        diff2 = q2_a.unsqueeze(1) - q2_b.unsqueeze(0) # (N, M, 32, 16)
        d_swd2 = diff2.pow(2).mean(dim=(-2, -1)) # (N, M)

        # Moment distance: mean squared difference across 14 components
        diff_m = mom_a.unsqueeze(1) - mom_b.unsqueeze(0) # (N, M, 14)
        d_mom = diff_m.pow(2).mean(dim=-1) # (N, M)

        # Combine with calibrated scales
        d_combined = (
            0.4 * (d_swd1 / self.scale_swd1.clamp(min=1e-5))
            + 0.4 * (d_swd2 / self.scale_swd2.clamp(min=1e-5))
            + 0.2 * (d_mom / self.scale_moment.clamp(min=1e-5))
        )

        # Normalize by median inter-class distance
        d_norm = d_combined / self.median_inter_dist.clamp(min=1e-5)
        return d_norm
