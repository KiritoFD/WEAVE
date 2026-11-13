"""Lightweight Score Difference inference shared with training and evaluation."""
from __future__ import annotations

import hashlib
from typing import Dict, Optional, Sequence, Tuple

import torch

from tools.legacy_root.wavelet import dwt2_haar, idwt2_haar
from tools.v6.models_v6 import ConditionalScoreNet

STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]


@torch.inference_mode()
def edit_with_score_difference(
    model: ConditionalScoreNet,
    z_src: torch.Tensor,
    src_ids: torch.Tensor,
    tgt_ids: torch.Tensor,
    *,
    num_steps: int = 12,
    step_size: float = 0.08,
    sigmas: Tuple[float, ...] = (0.50, 0.30, 0.15),
    struct_anchor_weight: float = 0.10,
    seed: int = 42,
    task_ids: Optional[Sequence[str]] = None,
    style_step_scales: Optional[Dict[str, float]] = None,
    record_steps: Optional[Sequence[int]] = None,
) -> torch.Tensor | Tuple[torch.Tensor, Dict[int, torch.Tensor], Dict[int, float]]:
    """Apply multi-scale score difference g_{s -> t} from source style s to target style t.

    Strictly satisfies g_{s -> s} == 0 when src_ids == tgt_ids!
    Guarantees deterministic, task_id-decoupled noise generation when task_ids are provided.
    Supports style-specific step scaling and step-by-step trajectory recording.
    """
    model.eval()
    B = z_src.shape[0]
    same_mask = (src_ids == tgt_ids).view(-1, 1, 1, 1)
    z = z_src.clone()

    # Per-sample deterministic generators
    if task_ids is not None:
        generators = [
            torch.Generator(device=z_src.device).manual_seed(
                int(hashlib.md5(f"{task_ids[i]}_{seed}".encode("utf-8")).hexdigest()[:8], 16)
            )
            for i in range(B)
        ]
    else:
        generators = [
            torch.Generator(device=z_src.device).manual_seed(seed + i * 10007)
            for i in range(B)
        ]

    # Style-specific step scale vector
    if style_step_scales is not None:
        scales = torch.tensor(
            [style_step_scales.get(STYLES[tgt_ids[i].item()], 1.0) for i in range(B)],
            device=z.device, dtype=z.dtype
        ).view(-1, 1, 1, 1)
        step_vec = step_size * scales
    else:
        step_vec = step_size

    trajectory = {}
    grad_norms = {}
    if record_steps is not None and 0 in record_steps:
        trajectory[0] = z.clone()

    for step in range(1, num_steps + 1):
        grad_acc = torch.zeros_like(z)
        for sig in sigmas:
            eps = torch.stack([
                torch.randn(z[i].shape, generator=generators[i], device=z.device, dtype=z.dtype)
                for i in range(B)
            ], dim=0)
            z_sig = z + sig * eps
            eps_s = model(z_sig, sig, src_ids)
            eps_t = model(z_sig, sig, tgt_ids)
            # g_{s -> t} = (eps_theta(z_sig, s) - eps_theta(z_sig, t)) / sig
            g_st = (eps_s - eps_t) / sig
            grad_acc = grad_acc + g_st / len(sigmas)

        # Zero out update wherever s == t (strictly preserving same-style identity)
        grad_acc = torch.where(same_mask, torch.zeros_like(grad_acc), grad_acc)

        # Record gradient norm
        grad_norm_val = float(grad_acc.flatten(1).norm(p=2, dim=1).mean().item())
        grad_norms[step] = grad_norm_val

        # Gentle structure anchoring toward normalized spatial self-structure of z_src
        if struct_anchor_weight > 0.0:
            ll_z, lh_z, hl_z, hh_z = dwt2_haar(z)
            ll_0, lh_0, hl_0, _ = dwt2_haar(z_src)
            mz, sz = ll_z.mean(dim=(2, 3), keepdim=True), ll_z.std(dim=(2, 3), keepdim=True).clamp(min=1e-4)
            m0, s0 = ll_0.mean(dim=(2, 3), keepdim=True), ll_0.std(dim=(2, 3), keepdim=True).clamp(min=1e-4)
            ll_anchored = ((1.0 - struct_anchor_weight) * ((ll_z - mz) / sz) + struct_anchor_weight * ((ll_0 - m0) / s0)) * sz + mz
            z = idwt2_haar(ll_anchored, lh_z, hl_z, hh_z)

        z = z + step_vec * grad_acc
        z = torch.where(same_mask, z_src, z)

        if record_steps is not None and step in record_steps:
            trajectory[step] = z.clone()

    if record_steps is not None:
        return z, trajectory, grad_norms
    return z
