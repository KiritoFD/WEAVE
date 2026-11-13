"""Sliced Wasserstein Distance (SWD) over local patch descriptors — Route A.

Distribution-level style matching: we compare the *distribution* of local
patch descriptors of the decoded output against those of the reference style
image. No pixel/coordinate correspondence is required — this is the key
difference from the current per-position FM target.

IMPORTANT — relation to historical experiments (see review doc section 五-路线A):
The repo's Cycle-NCE era already tried SWD and it failed. This implementation
differs in two essential ways, and experiments must not conflate them:
  1. SWD is computed on MULTI-LAYER features of the FINAL DECODED image
     (VAE output), not on a random target latent at a single ODE step.
  2. Gradients flow along the ACTUAL ODE inference path (2-4 unrolled Euler
     steps + VAE decode), not through a single-step surrogate.
The old negative result therefore does not transfer directly.
"""
from __future__ import annotations

from typing import Dict, Iterable, List

import torch
from torch import nn

from features import FrozenFeatureExtractor, extract_patch_descriptors, extract_patch_descriptors_chunked


def sliced_wasserstein_distance(
    desc_a: torch.Tensor,
    desc_b: torch.Tensor,
    *,
    num_projections: int = 64,
    p: int = 2,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sliced p-Wasserstein distance between two sets of descriptors.

    Args:
        desc_a, desc_b: (N, D) and (M, D) descriptor sets. May differ in count.
        num_projections: number of random 1-D projections (the "slices").
        p: Wasserstein order (1 or 2 are typical).
    Returns:
        Scalar distance. Differentiable w.r.t. both inputs (projections are
        constants sampled once per call).
    """
    if desc_a.dim() != 2 or desc_b.dim() != 2:
        raise ValueError("desc_a/desc_b must be (N, D) and (M, D)")
    D = desc_a.shape[-1]
    device, dtype = desc_a.device, torch.float32
    a = desc_a.float()
    b = desc_b.float()
    proj = torch.randn(D, num_projections, device=device, dtype=dtype, generator=generator)
    proj = proj / proj.norm(dim=0, keepdim=True).clamp_min(1e-8)
    pa = a @ proj  # (N, P)
    pb = b @ proj  # (M, P)
    pa_sorted, _ = pa.sort(dim=0)
    pb_sorted, _ = pb.sort(dim=0)
    # Align quantiles by interpolation when counts differ.
    n_a, n_b = pa_sorted.shape[0], pb_sorted.shape[0]
    if n_a != n_b:
        n = max(n_a, n_b)
        grid = torch.linspace(0, 1, n, device=device, dtype=dtype)
        ga = torch.linspace(0, 1, n_a, device=device, dtype=dtype)
        gb = torch.linspace(0, 1, n_b, device=device, dtype=dtype)
        idx_a = torch.searchsorted(ga, grid).clamp(0, n_a - 1)
        idx_b = torch.searchsorted(gb, grid).clamp(0, n_b - 1)
        pa_sorted = pa_sorted[idx_a]
        pb_sorted = pb_sorted[idx_b]
    diff = (pa_sorted - pb_sorted).abs().pow(p)
    return diff.mean().pow(1.0 / p).to(desc_a.dtype)


class PatchSWDStyleLoss(nn.Module):
    """Multi-layer patch-descriptor SWD style loss on decoded images.

    For each requested layer l of a frozen extractor phi:
        L_l = SWD(patch_desc(phi_l(I_hat)), patch_desc(phi_l(I_style)))
    where patch descriptors are local (patch_size x stride) feature patches.

    The loss is distribution-level: no spatial correspondence is assumed
    between output and style image.
    """

    def __init__(
        self,
        extractor: FrozenFeatureExtractor,
        layers: Iterable[str],
        *,
        patch_size: int = 8,
        patch_stride: int = 4,
        num_projections: int = 64,
        p: int = 2,
        layer_weights: Dict[str, float] | None = None,
        max_patches_per_side: int = 4096,
        chunk_rows: int = 0,
    ) -> None:
        super().__init__()
        self.extractor = extractor
        self.layers: List[str] = list(layers)
        self.patch_size = int(patch_size)
        self.patch_stride = int(patch_stride)
        self.num_projections = int(num_projections)
        self.p = int(p)
        self.layer_weights = dict(layer_weights or {})
        self.max_patches_per_side = int(max_patches_per_side)
        self.chunk_rows = max(0, int(chunk_rows))

    def _descriptors(self, img: torch.Tensor, layer: str, feats: Dict[str, torch.Tensor] | None) -> torch.Tensor:
        fmap = feats[layer] if feats is not None else self.extractor.forward_layers(img, [layer])[layer]
        desc = extract_patch_descriptors_chunked(  # (B, N, D)
            fmap, self.patch_size, self.patch_stride, chunk_rows=self.chunk_rows,
        )
        return desc.reshape(-1, desc.shape[-1])  # (B*N, D) pooled over batch — distribution-level

    def _subsample(self, desc: torch.Tensor) -> torch.Tensor:
        n = desc.shape[0]
        if n <= self.max_patches_per_side:
            return desc
        idx = torch.randperm(n, device=desc.device)[: self.max_patches_per_side]
        return desc[idx]

    def forward(
        self,
        img_out: torch.Tensor,
        img_style: torch.Tensor,
        *,
        feats_out: Dict[str, torch.Tensor] | None = None,
        feats_style: Dict[str, torch.Tensor] | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Returns {'loss': weighted sum, 'swd_<layer>': per-layer values}."""
        if feats_style is None:
            with torch.no_grad():
                feats_style = self.extractor.forward_layers(img_style.detach(), self.layers)
        total = img_out.new_tensor(0.0)
        logs: Dict[str, torch.Tensor] = {}
        for layer in self.layers:
            d_out = self._subsample(self._descriptors(img_out, layer, feats_out))
            d_sty = self._subsample(self._descriptors(img_style, layer, feats_style))
            val = sliced_wasserstein_distance(d_out, d_sty, num_projections=self.num_projections, p=self.p)
            w = float(self.layer_weights.get(layer, 1.0))
            total = total + w * val
            logs[f"swd_{layer}"] = val.detach()
        logs["loss"] = total
        return logs
