"""Regression tests for the ReLU² cross-attention memory path.

The reference implementation keeps two full ``[B, H, Q, K]`` tensors
(``logits`` + ``gates``) in the autograd graph. The recomputed kernel
evaluates the same expression in query chunks and rebuilds the gates in
backward. These tests pin the numerical contract so the memory fix can never
silently change what the model computes.
"""

import copy

import pytest
import torch

from tools.legacy_root.blocks import ResidualBlock, _Relu2AttentionFn, _relu2_attn_legacy


def _ref_stats(attended_inputs):
    gates = attended_inputs
    return (
        gates.detach().float().mean(),
        gates.detach().float().std(),
        (gates > 0.0).float().mean(),
    )


@torch.no_grad()
def test_recomputed_matches_legacy_forward():
    torch.manual_seed(0)
    q = torch.randn(2, 2, 16, 8, dtype=torch.float64)
    k = torch.randn(2, 2, 16, 8, dtype=torch.float64)
    v = torch.randn(2, 2, 16, 8, dtype=torch.float64)
    scale = 8 ** -0.5
    ref = _relu2_attn_legacy(q, k, v, scale)
    for chunk in (1, 4, 16):
        out, mean, std, active = _Relu2AttentionFn.apply(q, k, v, scale, chunk)
        assert torch.allclose(out, ref, atol=1e-9, rtol=0)
        logits = torch.matmul(q, k.transpose(-2, -1)) * scale
        gates = torch.relu(logits) ** 2
        ref_mean, ref_std, ref_active = _ref_stats(gates)
        assert abs(float(mean) - float(ref_mean)) < 1e-6
        assert abs(float(std) - float(ref_std)) < 1e-5
        assert abs(float(active) - float(ref_active)) < 1e-6


def test_recomputed_matches_legacy_backward():
    torch.manual_seed(1)
    q = torch.randn(2, 2, 16, 8, dtype=torch.float64, requires_grad=True)
    k = torch.randn(2, 2, 16, 8, dtype=torch.float64, requires_grad=True)
    v = torch.randn(2, 2, 16, 8, dtype=torch.float64, requires_grad=True)
    q2, k2, v2 = q.detach().clone().requires_grad_(True), k.detach().clone().requires_grad_(True), v.detach().clone().requires_grad_(True)
    scale = 8 ** -0.5
    seed_grad = torch.randn(2, 2, 16, 8, dtype=torch.float64)

    _relu2_attn_legacy(q, k, v, scale).backward(seed_grad)
    _Relu2AttentionFn.apply(q2, k2, v2, scale, 5)[0].backward(seed_grad)

    for name, a, b in (("dq", q, q2), ("dk", k, k2), ("dv", v, v2)):
        assert torch.allclose(a.grad, b.grad, atol=1e-9, rtol=0), name


def test_chunk_size_does_not_change_result():
    torch.manual_seed(2)
    q = torch.randn(3, 2, 37, 6, dtype=torch.float64)
    k = torch.randn(3, 2, 9, 6, dtype=torch.float64)
    v = torch.randn(3, 2, 9, 6, dtype=torch.float64)
    outs = []
    for chunk in (1, 7, 64):
        outs.append(_Relu2AttentionFn.apply(q, k, v, 6 ** -0.5, chunk)[0])
    for other in outs[1:]:
        assert torch.allclose(outs[0], other, atol=1e-9, rtol=0)


def test_algebraic_kernel_reorder_is_not_equivalent():
    """Guards the decision to NOT use the O(N) ``(relu(q)^2)(relu(k)^2)^T`` form.

    The model applies ``relu`` AFTER the dot product, so the linear-kernel
    reorder is a different function. It must not be mistaken for an exact fix.
    """
    torch.manual_seed(3)
    q = torch.randn(1, 2, 8, 6)
    k = torch.randn(1, 2, 8, 6)
    v = torch.randn(1, 2, 8, 6)
    ref = _relu2_attn_legacy(q, k, v, 6 ** -0.5)
    kf = torch.relu(k) ** 2
    qf = torch.relu(q) ** 2
    reordered = qf @ (kf.transpose(-2, -1) @ v)
    z = qf @ kf.sum(dim=-2, keepdim=True).transpose(-2, -1)
    reordered = reordered / z.clamp_min(1e-8)
    max_abs = (reordered - ref).abs().max().item()
    assert max_abs > 1e-2, f"unexpectedly equivalent (max_abs={max_abs:.3e})"


def test_block_forward_and_grad_match_legacy_impl():
    torch.manual_seed(4)
    block = ResidualBlock(
        dim=32,
        num_heads=4,
        style_gate_init=0.3,
        cross_attention_enabled=True,
        style_shortcut_alpha=1.0,
        layer_idx=0,
        num_layers=4,
        attn_temperature=1.0,
        dwt_route=True,
        dwt_route_train_prob=1.0,
        attn_mode="relu2",
        relu2_impl="legacy",
        relu2_chunk=4,
    )
    clone = copy.deepcopy(block)
    clone.relu2_impl = "recomputed"

    x = torch.randn(2, 32, 8, 8, requires_grad=True)
    x_clone = x.detach().clone().requires_grad_(True)
    time_emb = torch.randn(2, 32)
    style_tokens = torch.randn(2, 12, 32)

    out_ref = block(x, time_emb=time_emb, style_tokens=style_tokens)
    out_new = clone(x_clone, time_emb=time_emb, style_tokens=style_tokens)
    max_abs = (out_ref - out_new).abs().max().item()
    assert max_abs < 1e-5, f"forward drift {max_abs:.3e}"
    assert abs(float(block.last_debug["gate_mean"]) - float(clone.last_debug["gate_mean"])) < 1e-5

    out_ref.sum().backward()
    out_new.sum().backward()
    grad_drift = (x.grad - x_clone.grad).abs().max().item()
    assert grad_drift < 1e-5, f"grad drift {grad_drift:.3e}"


def test_invalid_impl_is_rejected():
    with pytest.raises(ValueError):
        ResidualBlock(dim=32, num_heads=4, relu2_impl="quantum")
