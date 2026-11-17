"""Unit tests for WEAVE V18 mathematical contracts (lightweight CPU)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def test_c64_color_palette_invariance():
    """Two RGB images with radically different color palettes but identical BT.601
    luminance MUST produce identical C_64 structure tensors."""
    from exp.v18.structure import rgb_to_gray, rgb_to_structure_64

    y = torch.rand(2, 1, 512, 512) * 0.6 + 0.2
    # Image A: pure grayscale [Y, Y, Y]
    rgb_a = y.repeat(1, 3, 1, 1)
    # Image B: strong warm-red / cool-blue tint preserving 0.299R + 0.587G + 0.114B == Y
    r = (y + 0.15).clamp(0, 1)
    b = (y - 0.15).clamp(0, 1)
    g = ((y - 0.299 * r - 0.114 * b) / 0.587).clamp(0, 1)
    rgb_b = torch.cat([r, g, b], dim=1)

    assert torch.allclose(rgb_to_gray(rgb_a), rgb_to_gray(rgb_b), atol=1e-5)
    c_a = rgb_to_structure_64(rgb_a)
    c_b = rgb_to_structure_64(rgb_b)
    assert c_a.shape == (2, 4, 64, 64)
    assert torch.allclose(c_a, c_b, atol=1e-4), f"max diff: {(c_a - c_b).abs().max()}"


def test_fast_dino_student_identities():
    from exp.v18.ld4_plus import FastDINOStudent

    net = FastDINOStudent(width=32, blocks=2)
    zb4 = torch.randn(3, 4, 64, 64)
    cc = torch.randn(3, 4, 64, 64)
    c64 = torch.randn(3, 4, 64, 64)
    d16 = torch.randn(3, 384, 16, 16)
    sid = torch.tensor([0, 2, 4])
    tid = torch.tensor([1, 2, 0])

    # 1. Zero-init exact identity
    out0 = net(zb4, cc, c64, d16, sid, tid)
    assert float(out0.abs().max()) == 0.0

    # 2. Non-zero weights still guarantee exact zero on same-style (index 1: sid==tid==2)
    with torch.no_grad():
        net.out.weight.normal_(0, 0.02)
    out1 = net(zb4, cc, c64, d16, sid, tid)
    assert float(out1[0].abs().max()) > 0.0
    assert float(out1[1].abs().max()) == 0.0


def test_ccf_unet_drop_flags_and_grads():
    from exp.v18.ccf_model import CCFUNet

    net = CCFUNet(base_dim=32, proto_dim=48, cond_dim=64)
    z = torch.randn(2, 4, 64, 64)
    c64 = torch.randn(2, 4, 64, 64)
    d16 = torch.randn(2, 384, 16, 16)
    sid = torch.tensor([1, 3])
    proto = torch.randn(2, 48)
    tau = torch.tensor([0.0, 0.65])

    # When drop_style=True, changing sid & proto must NOT change output
    net.eval()
    with torch.no_grad():
        u1 = net(z, c64, d16, sid, proto, tau, drop_style=True, drop_content=False)
        u2 = net(z, c64, d16, torch.tensor([4, 0]), torch.randn(2, 48), tau, drop_style=True, drop_content=False)
    assert torch.allclose(u1, u2, atol=1e-6)

    # Gradient check
    net.train()
    v = net(z, c64, d16, sid, proto, tau)
    v.square().mean().backward()
    assert net.in_conv.weight.grad is not None and torch.isfinite(net.in_conv.weight.grad).all()


def test_rank_normalize_and_wrong_cross_tid():
    from exp.v18.structure import rank_normalize_structure
    from exp.v18.run_v18_p2 import _pick_wrong_cross_tid

    c = torch.randn(2, 4, 64, 64) * 3.0 + 0.5
    r = rank_normalize_structure(c, keep_coarse=True)
    assert r.shape == (2, 4, 64, 64)
    assert torch.isfinite(r).all()
    assert float(r[:, 3].abs().max()) == 0.0
    # Contrast-invariant on channel 1 (coarse LCN; 1-rank step is 2/4096 = 4.9e-4)
    r2 = rank_normalize_structure(c * 10.0 - 2.0, keep_coarse=True)
    assert torch.allclose(r[:, 1], r2[:, 1], atol=1e-3)

    sid = torch.tensor([0, 1, 2, 3, 4, 0, 1])
    tid = torch.tensor([1, 2, 3, 4, 0, 0, 0])
    w = _pick_wrong_cross_tid(sid, tid)
    cross = sid != tid
    assert bool(((w[cross] != tid[cross]) & (w[cross] != sid[cross])).all())


if __name__ == "__main__":
    test_c64_color_palette_invariance()
    test_fast_dino_student_identities()
    test_ccf_unet_drop_flags_and_grads()
    test_rank_normalize_and_wrong_cross_tid()
    print("V18_UNIT_TESTS_4_OF_4_PASS")
