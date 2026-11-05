"""掩码几何与配置不变量测试。"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.pseudo_gt.masks import (  # noqa: E402
    E_SPECS,
    PATCH_SPECS,
    build_patch_layout,
    coverage,
    rasterize,
    subband_mask,
)


def test_layout_aligned_and_non_overlapping():
    for spec in PATCH_SPECS:
        rects = build_patch_layout(spec, seed=0)
        for r in rects:
            assert r.x % 16 == 0 and r.y % 16 == 0
            assert r.x + r.w <= 512 and r.y + r.h <= 512
        for i, a in enumerate(rects):
            for b in rects[i + 1 :]:
                assert (
                    a.x + a.w <= b.x
                    or b.x + b.w <= a.x
                    or a.y + a.h <= b.y
                    or b.y + b.h <= a.y
                ), f"{spec}: {a} 与 {b} 重叠"


def test_same_seed_reproducible():
    for spec in PATCH_SPECS:
        assert build_patch_layout(spec, seed=7) == build_patch_layout(spec, seed=7)


def test_coverage_matches_nominal():
    for spec, meta in PATCH_SPECS.items():
        mask = rasterize(build_patch_layout(spec, seed=0))
        actual = coverage(mask)
        assert abs(actual - float(meta["coverage"])) < 0.01, f"{spec}: {actual} vs {meta['coverage']}"


def test_mask_binary_inside_outside():
    rects = build_patch_layout("P1", seed=0)
    mask = rasterize(rects)
    inside = torch.zeros_like(mask, dtype=torch.bool)
    for r in rects:
        inside[r.y : r.y + r.h, r.x : r.x + r.w] = True
    assert torch.all(mask[inside] == 1.0)
    assert torch.all(mask[~inside] == 0.0)


def test_feather_bounded_and_smaller():
    rects = build_patch_layout("P2", seed=0)
    hard = coverage(rasterize(rects))
    soft = coverage(rasterize(rects, feather=True))
    assert 0.0 < soft < hard


def test_subband_mask_shape_and_area():
    rects = build_patch_layout("P7", seed=0)
    sub = subband_mask(rects)
    assert tuple(sub.shape) == (32, 32)
    assert float(sub.sum()) == 256.0  # 256px patch -> 16×16 子带


def test_e_specs_layouts_valid():
    for name in E_SPECS:
        rects = build_patch_layout(name, seed=0)
        assert len(rects) == len(E_SPECS[name]["sizes"])
