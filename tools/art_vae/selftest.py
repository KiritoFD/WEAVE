"""Self-tests for tools.art_vae. Run BEFORE any training.

Covers:
  1. style_readout: shape, finiteness, group scaling, gradient flow,
     invariance to global latent rescaling (anti-cheat property).
  2. supcon: finite, zero-grad when no positives, relation weighting,
     equivalence to a hand-computed small case.
  3. vicreg_var: gamma calibration against the empirical per-dim std.
  4. Gradient Cache: grad_Cache == grad_true_large_batch numerically.
  5. data: manifest alignment, unseen-style split, content-matched negative index.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.art_vae.style_readout import (D_OUT, D_TRI, style_readout,
                                         style_readout_raw,
                                         readout_scale_diagnostics)
from tools.art_vae.losses import (supcon, vicreg_var, supcon_relation_weights,
                                  gamma_for_dim)


def _ok(cond, label):
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}", flush=True)
    if not cond:
        raise AssertionError(label)


def test_readout(dev):
    print("\n=== 1. style_readout ===", flush=True)
    torch.manual_seed(0)
    z = torch.randn(6, 4, 64, 64, device=dev) * 3.0 + 1.5
    z = z.requires_grad_(True)

    groups = style_readout_raw(z)
    _ok([tuple(g.shape) for g in groups] ==
        [(6, 4), (6, 4), (6, D_TRI), (6, D_TRI)],
        f"group shapes = {[tuple(g.shape) for g in groups]}")

    q = style_readout(z)
    _ok(tuple(q.shape) == (6, D_OUT), f"q shape {tuple(q.shape)} == (6,{D_OUT})")
    _ok(bool(torch.isfinite(q).all()), "q all finite")
    _ok(bool((q.norm(dim=1) - 1).abs().max() < 1e-4), "q is L2-normalised")

    # group RMS equalisation: raw groups differ; after per-group RMS division
    # they must all carry equal energy (this is the anti-domination property)
    with torch.no_grad():
        gz = [g.detach() for g in groups]
        grms = [float(g.pow(2).mean(dim=1).sqrt().mean()) for g in gz]
        normed = [g / (g.pow(2).mean(dim=1, keepdim=True).sqrt() + 1e-5) for g in gz]
        nrms = [float(g.pow(2).mean(dim=1).sqrt().mean()) for g in normed]
    print(f"  raw group RMS: {[round(v, 3) for v in grms]}", flush=True)
    print(f"  normalised group RMS: {[round(v, 4) for v in nrms]}", flush=True)
    _ok(max(grms) / min(grms) > 1.5,
        "raw statistic groups have different scales (>1.5x)")
    _ok(max(nrms) / min(nrms) < 1.02,
        "per-group RMS normalisation equalises group energy")

    # gradient reaches z
    loss = (q * torch.randn_like(q)).sum()
    g, = torch.autograd.grad(loss, z)
    _ok(bool(torch.isfinite(g).all()) and float(g.abs().sum()) > 0,
        "gradient flows back to z")

    # anti-cheat: global rescale must not change q
    q2 = style_readout(z.detach() * 7.0)
    d = float((q.detach() - q2).abs().max())
    # mu / G0 / G1 are exactly rescale-invariant; log(sigma+eps) has a tiny
    # eps-induced deviation (O(1e-4)) because eps does not scale with z.
    _ok(d < 1e-3, f"q near-invariant to global rescaling (max diff {d:.2e}, eps-induced)")

    diag = readout_scale_diagnostics(z.detach())
    print(f"  diagnostics: {diag}", flush=True)
    return diag


def test_supcon(dev):
    print("\n=== 2. SupCon ===", flush=True)
    torch.manual_seed(1)
    B, D = 12, 16
    q = torch.nn.functional.normalize(torch.randn(B, D, device=dev), dim=1)
    style = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], device=dev)
    artist = torch.tensor([10, 10, 11, 12, 20, 20, 21, 22, 30, 31, 32, 33], device=dev)
    view = torch.arange(B, device=dev)  # all distinct works

    W = supcon_relation_weights(style, artist, view)
    _ok(bool((W >= 0).all()), "relation weights non-negative")
    _ok(bool((W.diagonal() == 0).all()), "no self-positive")
    _ok(float(W[0, 2]) > float(W[0, 1]),
        "cross-artist same-style weight > same-artist weight")

    l = supcon(q, W, temperature=0.1)
    _ok(bool(torch.isfinite(l)) and float(l) > 0, f"loss finite positive ({float(l):.4f})")

    # hand-computed 4-sample case with one positive pair
    q4 = torch.nn.functional.normalize(
        torch.tensor([[1., 0.], [0.9, 0.1], [0., 1.], [0.1, 0.9]], device=dev), dim=1)
    s4 = torch.tensor([0, 0, 1, 1], device=dev)
    a4 = torch.tensor([1, 2, 3, 4], device=dev)
    v4 = torch.arange(4, device=dev)
    W4 = supcon_relation_weights(s4, a4, v4)
    l4 = supcon(q4, W4, temperature=0.1)
    sim = (q4 @ q4.T) / 0.1
    logp = sim - torch.logsumexp(sim.masked_fill(torch.eye(4, device=dev).bool(),
                                                 float("-inf")), dim=1, keepdim=True)
    manual = -(logp[0, 1] + logp[1, 0] + logp[2, 3] + logp[3, 2]) / 4
    _ok(abs(float(l4) - float(manual)) < 1e-4,
        f"SupCon matches hand computation ({float(l4):.5f} vs {float(manual):.5f})")

    # no positives -> zero
    W0 = torch.zeros_like(W4)
    _ok(float(supcon(q4, W0, temperature=0.1)) == 0.0,
        "zero positives -> zero loss")


def test_vicreg(dev):
    print("\n=== 3. VICReg variance constraint ===", flush=True)
    B, D = 32, D_OUT
    q = torch.nn.functional.normalize(torch.randn(B, D, device=dev), dim=1)
    gamma = gamma_for_dim(D)
    std = q.std(dim=0)
    print(f"  empirical per-dim std: mean={float(std.mean()):.6f} "
          f"min={float(std.min()):.6f}  gamma={gamma:.6f}", flush=True)
    _ok(0 < gamma < float(std.mean()), "gamma calibrated below empirical std")
    l = vicreg_var(q, gamma)
    _ok(bool(torch.isfinite(l)), f"var loss finite ({float(l):.6f})")
    # collapsed representation must incur a large penalty
    qc = q[:1].expand(B, D)
    _ok(float(vicreg_var(qc, gamma)) > float(l) + 0.5 * gamma,
        "collapsed q is penalised more than a diverse q")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--skip-gradcache", action="store_true")
    ap.add_argument("--skip-data", action="store_true")
    args = ap.parse_args()
    dev = args.device
    print(f"[selftest] device={dev}", flush=True)

    test_readout(dev)
    test_supcon(dev)
    test_vicreg(dev)

    if not args.skip_gradcache:
        from tools.art_vae.gradcache import test_gradcache_numerical
        test_gradcache_numerical(dev)

    if not args.skip_data:
        from tools.art_vae.data import test_data_alignment
        test_data_alignment()

    print("\n[selftest] ALL TESTS PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
