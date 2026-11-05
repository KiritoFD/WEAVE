"""构造器不变量：A 掩码外不变、C 的 LL/HH 不变、D 在等统计时近似恒等。"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.pseudo_gt.construct import (  # noqa: E402
    band_energy_share,
    construct_A,
    construct_B,
    construct_C,
    construct_D,
    seam_gradient,
)
from tools.pseudo_gt.masks import build_patch_layout, rasterize  # noqa: E402
from tools.pseudo_gt.matching import Match  # noqa: E402
from wavelet import dwt2_haar  # noqa: E402


def _matches(spec="P2"):
    rects = build_patch_layout(spec, seed=3)
    return [Match(src=r, donor=r, q=0.9, mutual_ratio=0.5) for r in rects]


def test_construct_A_outside_mask_unchanged():
    torch.manual_seed(0)
    src = torch.rand(1, 3, 512, 512)
    donor = torch.rand(1, 3, 512, 512)
    m = _matches("P2")
    out = construct_A(src, donor, m)
    mask = rasterize([x.src for x in m])[None, None].bool().expand_as(src)
    assert torch.allclose(out[~mask], src[~mask])


def test_construct_B_alpha_zero_outside():
    torch.manual_seed(0)
    src = torch.rand(1, 3, 512, 512)
    donor = torch.rand(1, 3, 512, 512)
    m = _matches("P1")
    out = construct_B(src, donor, m)
    mask = rasterize([x.src for x in m], feather=True)[None, None].expand_as(src)
    outside = mask <= 0
    assert torch.allclose(out[outside], src[outside])


def test_construct_C_keeps_LL_and_HH():
    torch.manual_seed(1)
    zs = torch.randn(1, 4, 64, 64)
    zd = torch.randn(1, 4, 64, 64)
    m = _matches("P2")
    out = construct_C(zs, zd, m)
    ll_s, lh_s, hl_s, hh_s = dwt2_haar(zs)
    ll_o, lh_o, hl_o, hh_o = dwt2_haar(out)
    assert torch.allclose(ll_o, ll_s, atol=1e-5)
    assert torch.allclose(hh_o, hh_s, atol=1e-5)
    assert not torch.allclose(lh_o, lh_s) or not torch.allclose(hl_o, hl_s)


def test_construct_D_recovers_source_when_stats_equal():
    torch.manual_seed(2)
    z = torch.randn(1, 4, 64, 64)
    m = _matches("P5")
    out = construct_D(z, z.clone(), m)
    assert torch.allclose(out, z, atol=1e-5)


def test_band_energy_sums_to_one():
    torch.manual_seed(3)
    zs = torch.randn(1, 4, 64, 64)
    zp = zs + 0.01 * torch.randn_like(zs)
    share = band_energy_share(zs, zp)
    assert abs(sum(share.values()) - 1.0) < 1e-5


def test_seam_gradient_positive():
    torch.manual_seed(4)
    img = torch.rand(1, 3, 512, 512)
    rects = build_patch_layout("P5", seed=0)
    ratio = seam_gradient(img, rects)
    assert ratio > 0
