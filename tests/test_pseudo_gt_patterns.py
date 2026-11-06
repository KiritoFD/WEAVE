"""Pattern 掩码与 mask 测量的单元测试。"""
from __future__ import annotations

import random

import numpy as np
import pytest

from tools.pseudo_gt import pattern_masks as PM


def _full(n: int):
    q = np.full((n, n), 0.8)
    m = np.ones((n, n))
    return q, m


def test_random_selects_exact_k_unique_cells():
    q, m = _full(8)
    cells = PM.select_cells("R64", qmap=q, mmap=m, rng=random.Random(3))
    assert cells is not None and len(cells) == 16
    assert len(set(cells)) == 16
    assert all(0 <= x < 8 and 0 <= y < 8 for x, y in cells)


def test_grid_is_periodic():
    q, m = _full(8)
    cells = PM.select_cells("G64", qmap=q, mmap=m)
    assert cells == [(x, y) for y in range(0, 8, 2) for x in range(0, 8, 2)]
    grid = [0.0] * 64
    for cx, cy in cells:
        grid[cy * 8 + cx] = 1.0
    r_cells = PM.select_cells("R64", qmap=q, mmap=m, rng=random.Random(0))
    assert PM.spatial_periodicity(cells, 64) > PM.spatial_periodicity(r_cells, 64)


def test_cluster_rejects_low_quality_hole():
    q, m = _full(8)
    q[3, 3] = 0.1  # 4×4 块里必然包含 (3,3) 的只有部分块
    cells = PM.select_cells("C64", qmap=q, mmap=m)
    assert cells is not None
    assert all(q[cy, cx] >= PM.Q_MIN for cx, cy in cells)
    assert PM.component_count(cells, 64) == 1


def test_cluster_infeasible_when_no_clean_block():
    q, m = _full(8)
    q[:, :] = 0.1
    assert PM.select_cells("C64", qmap=q, mmap=m) is None


def test_spiral_and_field_are_permutations():
    order = PM.spiral_order(8, random.Random(1))
    assert len(order) == 64 and len(set(order)) == 64
    f = PM.field_order(8, random.Random(1))
    assert len(f) == 64 and len(set(f)) == 64


def test_coverage_and_boundary():
    q, m = _full(8)
    cells = PM.select_cells("C12", qmap=q, mmap=m, rng=random.Random(0))
    sal = np.zeros((8, 8))
    for cx, cy in cells:
        sal[cy, cx] = 2.0
    met = PM.mask_metrics(cells, 64, qmap=q, saliency=sal)
    assert met["coverage_actual"] == pytest.approx(0.125)
    assert met["boundary_length"] > 0
    assert met["component_count"] >= 1
    assert met["match_q_mean"] == pytest.approx(0.8)
    assert met["salient_region_fraction"] == pytest.approx(1.0)


def test_infeasible_when_too_few_qualified():
    q, m = _full(8)
    q[:] = PM.Q_MIN - 0.01
    assert PM.select_cells("R64", qmap=q, mmap=m, rng=random.Random(0)) is None
