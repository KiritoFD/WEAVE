"""Unit tests for V30 boundary/interior structure-tensor decomposition and d_b relaxation."""
from __future__ import annotations

import torch

from exp.v30_decoupled.p0_diagnose import structure_tensor_edge_mask
from exp.v30_decoupled.p12_train import (
    _DualRegionSWDCritic,
    _edge_stats_maps,
    _gather_chw,
    _interior_haar_energy,
    _pseudo_huber_velocity,
    _sample_region_indices,
    augment_y16_with_p1_clutter,
    compute_interior_relaxation_db,
)


def test_structure_tensor_and_interior_relaxation():
    # Sharp step edge image should light up edge mask; flat image should stay interior
    img = torch.zeros(2, 3, 64, 64)
    img[0, :, :, 32:] = 0.85
    img[1, :, :, :] = 0.40 + 0.02 * torch.randn(1, 3, 64, 64)
    img = img.clamp(0.0, 1.0)
    me, mi = structure_tensor_edge_mask(img, sigma_smooth=1.5)
    assert me.shape == (2, 1, 64, 64) and mi.shape == (2, 1, 64, 64)
    assert torch.allclose(me + mi, torch.ones_like(me), atol=1e-6)
    assert float(me[0, 0, :, 30:34].mean()) > float(me[0, 0, :, :16].mean())

    e_src = torch.full((2, 4), 1e-2)
    e_tgt_hi = torch.full((2, 4), 1e-2)
    e_tgt_lo = torch.full((2, 4), 1e-4)
    assert float(compute_interior_relaxation_db(e_src, e_tgt_hi)[0]) < 1e-5
    assert float(compute_interior_relaxation_db(e_src, e_tgt_lo)[0]) > 2.0

    y16 = torch.randn(2, 1, 16, 16).clamp(-1.0, 1.0)
    me16 = torch.zeros(2, 1, 16, 16)
    me16[:, :, :, 7:9] = 1.0
    y_aug, keep = augment_y16_with_p1_clutter(y16, me16, torch.tensor([0.0, 2.5]))
    assert y_aug.shape == y16.shape and keep.dtype == torch.bool

    l_int = _interior_haar_energy(img[:, :1], mi)
    assert l_int.shape == (2, 4) and torch.isfinite(l_int).all()

    ph = _pseudo_huber_velocity(torch.ones(2, 4, 16, 16) * 1.5, torch.zeros(2, 4, 16, 16), delta=0.50)
    assert float(ph) > 0.5


def test_p2_region_sampling_and_edge_stats():
    rgb = torch.rand(2, 3, 64, 64)
    ok, gm, coh = _edge_stats_maps(rgb)
    assert ok.shape == (2, 3, 64, 64) and gm.shape == (2, 1, 64, 64) and coh.shape == (2, 1, 64, 64)
    idx = _sample_region_indices(gm + 0.01, 32)
    feat = torch.randn(2, 5, 64, 64, requires_grad=True)
    g = _gather_chw(feat, idx)
    assert g.shape == (2, 32, 5)
    g.sum().backward()
    assert feat.grad is not None and torch.isfinite(feat.grad).all()
