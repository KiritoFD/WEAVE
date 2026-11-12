"""WEAVE Art-VAE Suite: Mixed Reconstruction, KL Divergence, CE & MoCo-SupCon Losses.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_mixed_reconstruction_loss(
    vae: nn.Module,
    x: torch.Tensor,
    lpips_fn: nn.Module,
    sample_prob: float = 0.25,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, any, torch.Tensor]:
    """Efficient single-decode mixed reconstruction: 75% posterior mean + 25% posterior sample.

    Matches expected loss: L_rec = 0.75 * L(D(mu), x) + 0.25 * L(D(sample), x) without doubling decoder cost.
    """
    posterior = vae.encode(x).latent_dist
    mu = posterior.mean
    sample = posterior.sample()

    mask = (torch.rand(x.shape[0], device=x.device) < sample_prob)
    z_mixed = torch.where(mask[:, None, None, None], sample, mu)

    recon = vae.decode(z_mixed).sample

    # L1 + 0.25 * LPIPS-Alex
    l1 = F.l1_loss(recon, x)
    # LPIPS expects input in [-1, 1]. Chunk in groups of 4 to eliminate activation memory peaks
    if recon.shape[0] > 4:
        lp_chunks = []
        for c_idx in range(0, recon.shape[0], 4):
            lp_chunks.append(lpips_fn(recon[c_idx : c_idx + 4].float(), x[c_idx : c_idx + 4].float()))
        l_lpips = torch.cat(lp_chunks, dim=0).mean()
    else:
        l_lpips = lpips_fn(recon.float(), x.float()).mean()
    l_rec = l1 + 0.25 * l_lpips

    return l_rec, l1, l_lpips, posterior, z_mixed


def compute_kl_divergence(posterior) -> Tuple[torch.Tensor, float, float, float]:
    """Compute KL divergence and posterior statistics in FP32.

    In diffusers DiagonalGaussianDistribution, posterior.kl() returns sum over latent dims [1, 2, 3],
    shape: (B,).
    """
    kl_per_img_tensor = posterior.kl() # (B,)
    kl_per_image = float(kl_per_img_tensor.mean().item())

    # 4 channels * 64 * 64 = 16384 latent elements
    latent_dim = posterior.mean.shape[1] * posterior.mean.shape[2] * posterior.mean.shape[3]
    kl_per_element_tensor = kl_per_img_tensor / float(latent_dim)
    kl_per_element = float(kl_per_element_tensor.mean().item())

    var_mean = float(posterior.var.mean().item())
    logvar_mean = float(posterior.logvar.mean().item())

    return kl_per_element_tensor.mean(), kl_per_element, kl_per_image, var_mean


def compute_moco_supcon_loss(
    queries: torch.Tensor, # (B, D)
    q_styles: torch.Tensor, # (B,)
    q_artists: torch.Tensor, # (B,)
    queue_feats: torch.Tensor, # (D, K)
    queue_styles: torch.Tensor, # (K,)
    queue_artists: torch.Tensor, # (K,)
    temperature: float = 0.10,
) -> Tuple[torch.Tensor, float]:
    """Supervised Contrastive Loss against MoCo Memory Queue.

    Positive pairs: same style.
    Skips anchors with 0 positives in queue to avoid division by zero or NaN gradients.
    """
    B = queries.shape[0]
    device = queries.device

    # Cosine similarities: (B, K)
    logits = (queries @ queue_feats) / temperature

    # Positives mask: same style and valid queue entry
    pos_mask = (q_styles.unsqueeze(1) == queue_styles.unsqueeze(0)) & (queue_styles.unsqueeze(0) >= 0)

    # Exclude invalid entries from logits denominator
    valid_mask = (queue_styles.unsqueeze(0) >= 0)
    logits_masked = logits.masked_fill(~valid_mask, -1e4)

    # Log-sum-exp over all valid entries
    log_denom = torch.logsumexp(logits_masked, dim=1, keepdim=True)

    # Compute loss for anchors with at least 1 positive
    n_pos = pos_mask.sum(dim=1)
    has_pos = (n_pos > 0)

    if not has_pos.any():
        return torch.tensor(0.0, device=device, requires_grad=True), 0.0

    # Log probabilities of positives
    log_probs = logits - log_denom
    pos_log_probs = (log_probs * pos_mask.float()).sum(dim=1)

    anchor_loss = -(pos_log_probs[has_pos] / n_pos[has_pos].clamp(min=1.0))
    loss = anchor_loss.mean()

    valid_pos_rate = float(has_pos.float().mean().item())
    return loss, valid_pos_rate
