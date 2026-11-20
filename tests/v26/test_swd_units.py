"""V26 契约 9–11: 动态 SWD 距离性质 (CPU, 冻结 VGG-16 离线权重)."""
import pytest
import torch

from exp.v26.dynamic_style_swd import DynamicSWD


@pytest.fixture()
def swd():
    """每测试独立实例: 层归一尺度在首个调用的参考幅度上初始化, 共享实例会串扰."""
    return DynamicSWD(device="cpu", n_proj=32, n_token=256)


@pytest.fixture(scope="module")
def imgs():
    g = torch.Generator().manual_seed(20261185)
    return torch.rand(4, 3, 64, 64, generator=g)


def test_same_image_zero_loss(swd, imgs):
    """契约 9a: 同图距离为 0 (经验分布完全重合)."""
    r = swd.loss(imgs[0:1], imgs[0:1], gen=torch.Generator().manual_seed(1))
    assert float(r["loss"]) < 1e-6, f"同图 SWD loss = {float(r['loss'])}"


def test_different_images_positive(swd, imgs):
    """契约 9b: 异图距离 > 0 且有限."""
    r = swd.loss(imgs[0:1], imgs[1:2], gen=torch.Generator().manual_seed(1))
    v = float(r["loss"])
    assert 0.0 < v < 1e4 and len(r["per_layer"]) == 3


def test_projection_refresh_determinism(swd, imgs):
    """契约 6/10: 同一种子 → 同一 loss (方向共享可复现); 不同种子 → 方向刷新."""
    v1 = float(swd.loss(imgs[0:1], imgs[2:3], gen=torch.Generator().manual_seed(7))["loss"])
    v2 = float(swd.loss(imgs[0:1], imgs[2:3], gen=torch.Generator().manual_seed(7))["loss"])
    v3 = float(swd.loss(imgs[0:1], imgs[2:3], gen=torch.Generator().manual_seed(8))["loss"])
    assert v1 == v2, "同种子方向未复现"
    assert v3 != v2, "不同种子方向未刷新"


def test_gradient_flows_to_output(swd, imgs):
    """契约 11: 梯度经 loss 到达 rgb_out (有限非零), 常量侧无梯度."""
    x = imgs[0:1].clone().requires_grad_(True)
    r = swd.loss(x, imgs[1:2].clone(), gen=torch.Generator().manual_seed(3))
    r["loss"].backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert float(x.grad.abs().sum()) > 0.0


def test_layer_scales_fixed_after_first_call(swd, imgs):
    """层归一首次初始化后保持固定 (防止逐 update 漂移)."""
    swd.loss(imgs[0:1], imgs[1:2], gen=torch.Generator().manual_seed(5))
    s1 = list(swd._layer_scales)
    swd.loss(imgs[0:1], imgs[1:2], gen=torch.Generator().manual_seed(5))
    assert s1 == list(swd._layer_scales)
