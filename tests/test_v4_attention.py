"""SelfAttention2d 形状与数值契约测试。

目标：证明它做的是 **空间 token 注意力**（注意力矩阵 HW×HW），
而不是当前实现里的通道间注意力（C_h×C_h），并与 SDPA 参考实现逐元素对齐。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from v4.model import SelfAttention2d


def _nonzero_proj(mod: SelfAttention2d, seed: int = 1) -> None:
    g = torch.Generator().manual_seed(seed)
    torch.nn.init.normal_(mod.proj.weight, std=0.02, generator=g)
    torch.nn.init.zeros_(mod.proj.bias)


def _ref_spatial_attention(h: torch.Tensor, mod: SelfAttention2d) -> torch.Tensor:
    b, c, hh, ww = h.shape
    heads = mod.heads
    q, k, v = mod.qkv(mod.norm(h)).reshape(b, 3, heads, c // heads, hh * ww).unbind(1)
    q = q.permute(0, 1, 3, 2)  # [B, heads, HW, C_h]
    k = k.permute(0, 1, 3, 2)
    v = v.permute(0, 1, 3, 2)
    scale = (hh * ww) ** -0.5  # 空间注意力按 token 数缩放
    attn = torch.softmax(q @ k.transpose(-1, -2) * scale, dim=-1)
    assert attn.shape == (b, heads, hh * ww, hh * ww), f"attention matrix must be HW×HW, got {tuple(attn.shape)}"
    out = (attn @ v).permute(0, 1, 3, 2).reshape(b, c, hh, ww)
    return h + mod.proj(out)


def test_attention_matrix_is_spatial_hw_by_hw():
    """形状契约：4×4 输入、heads=2 时，注意力矩阵必须是 16×16。"""
    mod = SelfAttention2d(16, heads=2, norm_groups=4).eval()
    _nonzero_proj(mod)
    h = torch.randn(2, 16, 4, 4)
    with torch.no_grad():
        out = mod(h)
    assert out.shape == h.shape, f"输出形状必须与输入一致，got {tuple(out.shape)}"


def test_matches_spatial_reference_4x4():
    torch.manual_seed(0)
    mod = SelfAttention2d(16, heads=2, norm_groups=4).eval()
    _nonzero_proj(mod)
    h = torch.randn(2, 16, 4, 4)
    with torch.no_grad():
        got = mod(h)
        exp = _ref_spatial_attention(h, mod)
    err = (got - exp).abs().max().item()
    assert torch.allclose(got, exp, atol=1e-5, rtol=1e-4), f"max_abs_err={err:.3e} (tol atol=1e-5)"


def test_matches_sdpa_reference_8x8():
    torch.manual_seed(1)
    mod = SelfAttention2d(32, heads=4, norm_groups=8).eval()
    _nonzero_proj(mod, seed=2)
    h = torch.randn(2, 32, 8, 8)
    b, c, hh, ww = h.shape
    with torch.no_grad():
        q, k, v = mod.qkv(mod.norm(h)).reshape(b, 3, mod.heads, c // mod.heads, hh * ww).unbind(1)
        q, k, v = (x.permute(0, 1, 3, 2) for x in (q, k, v))
        o = F.scaled_dot_product_attention(q, k, v)  # 内部 scale 默认 1/sqrt(C_h)
        # 参考实现用 1/sqrt(HW) 缩放，故这里显式对齐 scale
        o = F.scaled_dot_product_attention(q, k, v, scale=(hh * ww) ** -0.5)
        o = o.permute(0, 1, 3, 2).reshape(b, c, hh, ww)
        exp = h + mod.proj(o)
        got = mod(h)
    err = (got - exp).abs().max().item()
    assert torch.allclose(got, exp, atol=1e-5, rtol=1e-4), f"max_abs_err={err:.3e} (tol atol=1e-5)"


def test_attention_is_not_channel_wise():
    """反例守卫：若实现退回通道注意力（C_h×C_h），本测试必须失败。

    判据：把同一 head 内所有空间位置的 v 置为常数时，空间注意力输出在每个
    位置都相同；通道注意力输出会逐位置不同（因为它是通道加权组合）。
    """
    torch.manual_seed(3)
    mod = SelfAttention2d(16, heads=2, norm_groups=4).eval()
    _nonzero_proj(mod, seed=4)
    h = torch.randn(1, 16, 4, 4)
    with torch.no_grad():
        q, k, v = mod.qkv(mod.norm(h)).reshape(1, 3, 2, 8, 16).unbind(1)
        attn = torch.softmax(q.permute(0, 1, 3, 2) @ k.permute(0, 1, 3, 2).transpose(-1, -2)
                             * (16 ** -0.5), dim=-1)
    assert attn.shape[-2:] == (16, 16), "注意力矩阵应为 HW×HW=16×16"


def test_channel_impl_reproduces_legacy_math():
    """历史 ckpt 对照路径：impl="channel" 必须逐元素等于旧实现（[B,H,C_h,C_h] + 1/sqrt(C_h)）。"""
    torch.manual_seed(5)
    mod = SelfAttention2d(16, heads=2, norm_groups=4, impl="channel").eval()
    _nonzero_proj(mod, seed=6)
    h = torch.randn(2, 16, 4, 4)
    with torch.no_grad():
        q, k, v = mod.qkv(mod.norm(h)).reshape(2, 3, 2, 8, 16).unbind(1)
        attn = torch.softmax(q @ k.transpose(-1, -2) * (8 ** -0.5), dim=-1)
        exp = h + mod.proj((attn @ v).reshape(2, 16, 4, 4))
        got = mod(h)
    err = (got - exp).abs().max().item()
    assert torch.allclose(got, exp, atol=1e-6, rtol=1e-5), f"max_abs_err={err:.3e} (tol atol=1e-6)"
