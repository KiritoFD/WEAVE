"""V26 契约 5–8: Q32 / 软极分解数值性质 (CPU)."""
import math

import pytest
import torch

from exp.v21.stiefel_ref_cache import stiefel_polar_quotient
from exp.v26.diagnose_q32 import (
    gram_feats,
    soft_stiefel_quotient,
    spectrum_stats,
    structure_retention,
)


@pytest.fixture()
def basis():
    g = torch.Generator().manual_seed(20261185)
    a = torch.randn(384, 32, generator=g)
    q, _ = torch.linalg.qr(a)
    return q  # (384, 32) 正交列


def test_soft_q32_lowrank_and_zero_stable(basis):
    """契约 5: 全零 / 秩 1 病态输入下软 Q32 无 NaN/Inf."""
    z0 = torch.zeros(2, 384, 16, 16)
    r1 = torch.zeros(2, 384, 16, 16)
    r1[:, 0] = 1.0
    for x in (z0, r1):
        q = soft_stiefel_quotient(x, basis, eta=0.3)
        assert torch.isfinite(q).all()


def test_soft_eta_zero_equals_hard(basis):
    """契约 6: eta=0 时软 Q32 退化为硬极分解 (逐元素一致)."""
    g = torch.Generator().manual_seed(42)
    p = torch.randn(2, 384, 16, 16, generator=g)
    hard = stiefel_polar_quotient(p, basis)
    soft = soft_stiefel_quotient(p, basis, eta=0.0)
    assert torch.allclose(hard, soft, atol=1e-4), \
        f"max diff = {float((hard - soft).abs().max())}"


def test_hard_q32_gram_identity(basis):
    """契约 7: 硬 Q32 满足 (1/256) Q Q^T == I_32 (≤1e-3)."""
    g = torch.Generator().manual_seed(43)
    p = torch.randn(3, 384, 16, 16, generator=g)
    q = stiefel_polar_quotient(p, basis).float().flatten(2)
    err = float((torch.bmm(q, q.transpose(1, 2)) / 256.0 - torch.eye(32)).abs().max())
    assert err < 1e-3, f"Gram 恒等误差 {err}"


def test_spectrum_and_retention():
    """契约 8: 谱统计有限且有效秩>0; 结构保持对恒等输入为 1."""
    g = torch.Generator().manual_seed(44)
    x = torch.randn(4, 16, 16, 16, generator=g)
    st = spectrum_stats(x)
    assert st["effective_rank"] > 0 and math.isfinite(st["effective_rank"])
    assert st["sigma_min_max_ratio"] > 0
    assert abs(structure_retention(x, x) - 1.0) < 1e-5
    y = torch.randn(4, 16, 16, 16, generator=g)
    r = structure_retention(x, y)
    assert -1.0 <= r <= 1.0


def test_gram_feats_shape():
    g = torch.Generator().manual_seed(45)
    x = torch.randn(2, 48, 16, 16, generator=g)
    gf = gram_feats(x)
    assert gf.shape == (2, 48 * 49 // 2)
