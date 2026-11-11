"""Gradient Cache for large-logical-batch contrastive learning.

Problem: SupCon needs all 64 samples in one softmax, but a 64-image encoder
forward at 512px does not fit in 24 GB.  Standard microbatch accumulation would
compute the contrastive loss on only `mb` samples, destroying the negative pool.

Solution (Gradient Cache, two passes over the SAME 64 images):

  pass 1 (no grad): run the encoder in microbatches, collect the tiny style
      readouts q (64 x 1340). Compute the FULL-batch SupCon + VICReg loss on
      that small tensor and differentiate it w.r.t. q  ->  g_q (64 x 1340).
  pass 2 (grad): re-run the encoder in microbatches WITH autograd, recompute q,
      and inject g_q via `(q * g_q_chunk).sum().backward()`. Chain rule makes
      this exactly dL_style/dtheta for that chunk. Reconstruction/KL gradients
      are accumulated in the same pass. One optimizer step at the end.

Why this is exact: the style loss depends on the encoder only THROUGH q, so
splitting dL/dq by row and re-using the same Jacobian per chunk reproduces the
true large-batch backward. `test_gradcache_numerical` verifies this.

Determinism: the encoder has no dropout/BN-in-train randomness, and the readout
is parameter-free, so both passes see identical q for identical inputs. The
dataset must use deterministic crops (fixed per-sample crop seed), which
`tools.art_vae.data` guarantees.
"""
from __future__ import annotations

from typing import Callable

import torch


def cache_style_gradient(encode_fn: Callable[[torch.Tensor], torch.Tensor],
                         readout_fn: Callable[[torch.Tensor], torch.Tensor],
                         style_loss_fn: Callable[[torch.Tensor], torch.Tensor],
                         x: torch.Tensor, microbatch: int):
    """Pass 1. Returns (q_detached (B,D), dL/dq (B,D), style_loss_value).

    `style_loss_fn` receives the FULL batch q and must return a scalar.
    Runs under no_grad for the encoder; autograd is used only on the tiny q.
    """
    B = x.shape[0]
    chunks = []
    with torch.no_grad():
        for i in range(0, B, microbatch):
            z = encode_fn(x[i:i + microbatch])
            chunks.append(readout_fn(z).detach())
    q = torch.cat(chunks, dim=0).detach().requires_grad_(True)
    loss = style_loss_fn(q)
    g, = torch.autograd.grad(loss, q)
    return q.detach(), g.detach(), float(loss.detach())


def inject_chunk(encode_fn, readout_fn, x_chunk, g_chunk):
    """Pass 2 for one microbatch: inject dL_style/dq and return z (with grad)."""
    z = encode_fn(x_chunk)                     # WITH grad
    q = readout_fn(z)
    (q * g_chunk).sum().backward(retain_graph=True)
    return z


def test_gradcache_numerical(dev: str, tol: float = 2e-3):
    """Verify cache gradients == a genuine single-64-image large-batch backward.

    Uses a small deterministic conv encoder and the REAL style_readout, so the
    test exercises the same Jacobian structure as training.
    """
    import torch.nn as nn
    import torch.nn.functional as F
    from tools.art_vae.style_readout import style_readout
    from tools.art_vae.losses import (supcon, supcon_relation_weights,
                                      vicreg_var, gamma_for_dim)

    print("\n=== 4. Gradient Cache numerical equivalence ===", flush=True)
    torch.manual_seed(7)
    B, mb = 16, 4
    x = torch.randn(B, 3, 32, 32, device=dev)

    enc = nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1), nn.SiLU(),
        nn.AvgPool2d(2),
        nn.Conv2d(4, 4, 3, padding=1),
    ).to(dev)

    def encode(xb):
        return enc(xb)

    def readout(zb):
        return style_readout(zb)

    style = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0,
                          1, 1, 1, 1, 1, 1, 1, 1], device=dev)
    artist = torch.arange(B, device=dev) % 11
    view = torch.arange(B, device=dev)
    W = supcon_relation_weights(style, artist, view)
    gamma = gamma_for_dim(1340)

    def loss_fn(q):
        return supcon(q, W) + 0.1 * vicreg_var(q, gamma)

    # ---- ground truth: one forward over all B ----
    enc.zero_grad(set_to_none=True)
    q_true = readout(encode(x))
    loss_true = loss_fn(q_true)
    loss_true.backward()
    g_true = torch.cat([p.grad.flatten() for p in enc.parameters()
                        if p.grad is not None]).clone()

    # ---- cached: pass 1 (no grad) + pass 2 (inject per chunk) ----
    enc.zero_grad(set_to_none=True)
    q_c, g_q, loss_c = cache_style_gradient(encode, readout, loss_fn, x, mb)
    for i in range(0, B, mb):
        inject_chunk(encode, readout, x[i:i + mb], g_q[i:i + mb])
    g_cache = torch.cat([p.grad.flatten() for p in enc.parameters()
                         if p.grad is not None]).clone()

    cos = float(F.cosine_similarity(g_true, g_cache, dim=0))
    rel = float((g_true - g_cache).norm() / (g_true.norm() + 1e-12))
    print(f"  g_true norm={g_true.norm():.6f}  g_cache norm={g_cache.norm():.6f}",
          flush=True)
    print(f"  cos={cos:.6f}  relative L2 diff={rel:.2e}", flush=True)
    print(f"  loss_true={float(loss_true):.6f}  loss_cached={loss_c:.6f}",
          flush=True)
    assert abs(float(loss_true) - loss_c) < 1e-5, "style loss mismatch between passes"
    assert cos > 1 - tol, f"gradient direction mismatch: cos={cos}"
    assert rel < 0.05, f"gradient magnitude mismatch: rel={rel}"
    print("  [PASS] Gradient Cache == true large-batch backward", flush=True)
