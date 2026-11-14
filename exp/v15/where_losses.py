"""V15 Where self-supervision losses (plan §4.3).

Three losses added on top of the V6 objective:
  1. Masked Context Reconstruction — random region occlusion, reconstruct from context
  2. Region Affinity — weak structural region labels from edge/clustering
  3. Content Invariance — representation stability under appearance perturbation
"""
from __future__ import annotations

import torch
import torch.nn.functional as Fn


def random_region_mask(B, H, W, device, min_size=4, max_size=16, n_masks=3):
    """Irregular region masks (not just random squares)."""
    masks = torch.zeros(B, 1, H, W, device=device)
    for b in range(B):
        for _ in range(n_masks):
            h = torch.randint(min_size, max_size, (1,), device=device).item()
            w = torch.randint(min_size, max_size, (1,), device=device).item()
            y = torch.randint(0, H - h, (1,), device=device).item()
            x = torch.randint(0, W - w, (1,), device=device).item()
            masks[b, 0, y:y + h, x:x + w] = 1.0
    return masks


def masked_reconstruction_loss(model, z_src, style_id, tau, mask_fraction=0.15):
    """Mask random regions of the input latent, predict the full clean output."""
    B = z_src.shape[0]
    H, W = z_src.shape[-2:]
    mask = random_region_mask(B, H, W, z_src.device,
                              n_masks=max(1, int(mask_fraction * 4)))
    z_masked = z_src * (1 - mask) + torch.randn_like(z_src) * 0.1 * mask
    pred = model(z_masked, style_id, tau)
    # only compute loss in masked regions (forces context usage)
    diff = (pred - z_src).abs() * mask
    return diff.sum() / (mask.sum() * z_src.shape[1] + 1e-8)


def region_affinity_loss(feats, edge_map, tau=0.07, n_sample=256):
    """Weak region affinity: nearby tokens with similar edge magnitude should
    have similar features; edge discontinuities should separate features.

    feats: (B, D, H, W) spatial features
    edge_map: (B, 1, H, W) sobel magnitude of LL
    Subsamples n_sample tokens so the affinity matrix stays small
    ((B, 256, 256) instead of (B, 4096, 4096) = ~2GB at B=32).
    """
    B, D, H, W = feats.shape
    tokens = feats.flatten(2).transpose(1, 2)              # (B, HW, D)
    e_flat = edge_map.flatten(2).transpose(1, 2)            # (B, HW, 1)
    HW = tokens.shape[1]
    if HW > n_sample:
        idx = torch.randperm(HW, device=feats.device)[:n_sample]
        tokens = tokens[:, idx]
        e_flat = e_flat[:, idx]
    tokens_n = Fn.normalize(tokens, dim=2)
    # affinity prediction: dot product between token pairs
    aff = tokens_n @ tokens_n.transpose(1, 2)               # (B, HW, HW)
    # weak labels: same-edge-level pairs = same region (1.0), different = boundary (0.0)
    e_bin = (e_flat > e_flat.median(dim=1, keepdim=True).values).float()
    labels = (e_bin == e_bin.transpose(1, 2)).float()       # 1 = both high or both low
    loss = ((aff - labels) ** 2).mean()
    return loss


def content_invariance_loss(feats_a, feats_b):
    """Representation consistency under appearance perturbation + variance reg."""
    loss_inv = (feats_a - feats_b).square().mean()
    # variance regularization (prevent collapse)
    zc = feats_a - feats_a.mean(0)
    std = zc.std(0)
    var_reg = Fn.relu(1.0 - std).mean()
    return loss_inv + 0.1 * var_reg


def appearance_perturb(z: torch.Tensor, strength: float = 1.0, seed: int = 0):
    """Colour/brightness/contrast perturbation (no geometry change)."""
    gen = torch.Generator(device=z.device.type)
    gen.manual_seed(seed)
    bright = 1.0 + (torch.rand(z.shape[0], 1, 1, 1, generator=gen,
                               device=z.device) * 2 - 1) * 0.20 * strength
    contrast = 1.0 + (torch.rand(z.shape[0], 1, 1, 1, generator=gen,
                                 device=z.device) * 2 - 1) * 0.20 * strength
    gain = 1.0 + (torch.rand(z.shape[0], 4, 1, 1, generator=gen,
                             device=z.device) * 2 - 1) * 0.15 * strength
    mean = z.mean(dim=(2, 3), keepdim=True)
    return ((z - mean) * contrast * gain + mean * bright)


def where_losses(model, z_src, style_id, tau, lambda_m=0.1, lambda_a=0.05,
                 lambda_i=0.05):
    """Compute all Where self-supervision losses for a training step."""
    B = z_src.shape[0]
    losses = {}

    # 1. Masked reconstruction
    pert_z = appearance_perturb(z_src, strength=0.5, seed=int(tau[0].item() * 1000))
    losses["masked"] = masked_reconstruction_loss(model, pert_z, style_id, tau)

    # 2. Region affinity (on intermediate features — needs a hook or forward pass)
    # For simplicity, apply to the model output features
    pred = model(z_src, style_id, tau)
    # Use edge map from the latent as weak structural signal
    gx = pred[:, :, :, 1:] - pred[:, :, :, :-1]          # (B,C,H,W-1)
    gy = pred[:, :, 1:, :] - pred[:, :, :-1, :]          # (B,C,H-1,W)
    gx = Fn.pad(gx, (0, 1, 0, 0))                        # -> (B,C,H,W)
    gy = Fn.pad(gy, (0, 0, 0, 1))                        # -> (B,C,H,W)
    edge = (gx.square().mean(1) + gy.square().mean(1)).sqrt()  # (B,H,W)
    edge = edge.unsqueeze(1)                                     # (B,1,H,W)
    # Use the predicted latent as a proxy for features
    feats = pred.flatten(2).transpose(1, 2)                      # (B, HW, 4)
    losses["affinity"] = region_affinity_loss(pred, edge)

    # 3. Content invariance: two appearance-perturbed versions should give
    #    consistent structure features
    pert1 = appearance_perturb(z_src, strength=0.3, seed=42)
    pert2 = appearance_perturb(z_src, strength=0.3, seed=43)
    pred1 = model(pert1, style_id, tau)
    pred2 = model(pert2, style_id, tau)
    losses["invariance"] = content_invariance_loss(pred1, pred2)

    total = (lambda_m * losses["masked"] + lambda_a * losses["affinity"] +
             lambda_i * losses["invariance"])
    return total, losses
