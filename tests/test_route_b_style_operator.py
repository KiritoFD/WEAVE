"""路线 B（B1+B2）的 CPU 单测与结构探针。

运行：python -m pytest tests/test_route_b_style_operator.py -q
全部合成张量、CPU、秒级；不触碰 GPU/远端。
"""
import pytest
import torch
import torch.nn.functional as F

from b1_style_operator import (
    B1OperatorConfig,
    B1StyleDistributionLoss,
    B1StyleOperator,
    StyleTargetScreener,
    reference_permutation_probe,
    reference_swap_probe,
)
from b2_destyle import (
    DestyleConfig,
    ReferenceMixConfig,
    build_latent_pair,
    destyle_image,
    destyle_latent,
    lowpass_geometry_correlation,
    mix_patches,
    shuffle_patches,
)
from b_pseudo_targets import (
    BPseudoTargetProvider,
    OfflineStyleFeatureCache,
    OfflineStyleFeatureCacheWriter,
)
from b_training_diagnostics import (
    CurveRecorder,
    OverfitConfig,
    OverfitProbe,
    StyleGainDiagnostics,
    channel_adain,
    evaluate_holdout,
)
from pseudo_targets import PseudoTargetCache


SHAPE = (1, 4, 16, 16)
TOKENS = 64  # token_side=8 -> 8x8


def make_operator(seed: int = 0) -> B1StyleOperator:
    torch.manual_seed(seed)
    cfg = B1OperatorConfig(
        latent_channels=4,
        widths=(32, 64, 96),
        encoder_blocks=(1, 1, 1),
        attention_heads=(2, 4, 4),
        attention_layers=(1, 1, 1),
        token_side=8,
        decoder_hidden=32,
        num_directions=4,
        num_brush_scales=2,
        direction_kernel=3,
        block_kernel=3,
        style_gate_init=0.2,
        patch_size=4,
        patch_stride=4,
    )
    return B1StyleOperator(cfg)


def make_latents(seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    content = torch.randn(SHAPE, generator=g)
    ref_a = torch.randn(SHAPE, generator=g)
    ref_b = torch.randn(SHAPE, generator=g) * 3.0 + 2.0
    return content, ref_a, ref_b


# ---------------------------------------------------------------------------
# B1 结构 / 梯度
# ---------------------------------------------------------------------------

def test_operator_shapes_and_gradients():
    op = make_operator()
    content, ref, _ = make_latents()
    out = op(content, ref, return_attention=True)
    assert out["z_star"].shape == SHAPE
    assert out["gate"].numel() == 1
    # 内容 token 4x4，参考 token 4x4
    assert out["attention"].shape == (1, TOKENS, TOKENS)
    # 区域参数与 LL 子带同分辨率：16x16 latent -> 8x8
    params = out["region_params"]
    assert params["direction"].shape == (1, 4, 8, 8)
    assert params["brush_scale"].shape == (1, 2, 8, 8)
    assert params["block_strength"].shape == (1, 1, 8, 8)
    assert params["region_gate"].shape == (1, 1, 8, 8)
    # 概率分布成立
    assert torch.allclose(params["direction"].sum(1), torch.ones(1, 8, 8), atol=1e-5)
    assert torch.allclose(params["brush_scale"].sum(1), torch.ones(1, 8, 8), atol=1e-5)

    out["z_star"].pow(2).mean().backward()
    assert op.q_proj[0][0].weight.grad is not None
    assert op.k_proj[0][0].weight.grad is not None
    assert op.v_proj[0][0].weight.grad is not None
    assert op.repaint.renderer.ll_out.weight.grad is not None
    assert op.repaint.renderer.hf_out.weight.grad is not None
    assert op.repaint.renderer.style_gate.grad is not None


def test_reference_token_permutation_is_invariant():
    op = make_operator(seed=2)
    content, ref, _ = make_latents(seed=3)
    perm = torch.randperm(TOKENS, generator=torch.Generator().manual_seed(4))
    base = op(content, ref)["z_star"]
    shuffled = op(content, ref, reference_token_permutation=perm)["z_star"]
    assert torch.allclose(base, shuffled, atol=1e-5, rtol=0.0)

    probe = reference_permutation_probe(op, content, ref, permutation=perm)
    assert probe["output_delta_max"] < 1e-4


def test_reference_swap_changes_output():
    op = make_operator(seed=5)
    content, ref_a, ref_b = make_latents(seed=6)
    probe = reference_swap_probe(op, content, ref_a, ref_b)
    # 未训练时只验证"参考图路径确实连通"；风格强度由阶段 1 训练建立。
    assert probe["output_delta_rel"] > 1.0e-4
    assert probe["output_delta_max"] > 0.0


def test_stage3_checks_pass_on_synthetic():
    op = make_operator(seed=7)
    provider = BPseudoTargetProvider(op)
    content, ref_a, ref_b = make_latents(seed=8)
    checks = provider.stage3_checks(content, ref_a, ref_b)
    assert checks["swap_passed"]
    assert checks["permutation_passed"]
    assert checks["passed"]


def test_style_distribution_loss_backward():
    torch.manual_seed(9)
    loss_fn = B1StyleDistributionLoss(patch_size=4, patch_stride=4, num_projections=16)
    op = make_operator(seed=9)
    content, ref, _ = make_latents(seed=10)
    out = op(content, ref)
    logs = loss_fn(out["z_star"], content, ref, ll_delta=out["ll_delta"])
    logs["loss"].backward()
    assert torch.isfinite(logs["loss"])
    assert op.repaint.renderer.style_gate.grad is not None


def test_default_parameter_budget():
    """新规格：T_phi 默认配置必须落在 5–15M 可训练参数。"""
    op = B1StyleOperator(B1OperatorConfig())
    n = sum(p.numel() for p in op.parameters() if p.requires_grad)
    assert 5_000_000 <= n <= 15_000_000, f"default B1 operator has {n/1e6:.2f}M params"


def test_region_parameters_are_valid_probabilities():
    op = make_operator(seed=36)
    content, ref, _ = make_latents(seed=37)
    out = op(content, ref)
    p = out["region_params"]
    for key in ("direction", "brush_scale"):
        assert torch.allclose(p[key].sum(dim=1), torch.ones(1, 8, 8), atol=1e-5)
        assert (p[key] >= 0).all()
    for key in ("block_strength", "region_gate"):
        assert ((p[key] >= 0) & (p[key] <= 1)).all()


# ---------------------------------------------------------------------------
# B2 Destyle / 参考图扰动
# ---------------------------------------------------------------------------

def _structured_image(seed: int = 11) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    y = torch.rand(2, 3, 32, 32, generator=g)
    y[..., 10:22, 10:22] = (y[..., 10:22, 10:22] + 0.5).clamp(0, 1)
    return y


def test_destyle_preserves_geometry():
    y = _structured_image()
    cfg = DestyleConfig(
        color_prob=1.0,
        color_mode_probs=(0.0, 1.0, 0.0),
        lowfreq_prob=0.0,
        hf_suppress_prob=1.0,
        hf_suppress_min=0.2,
        hf_suppress_max=0.2,
        blur_prob=0.0,
        clamp_range=None,
    )
    x = destyle_image(y, cfg, torch.Generator().manual_seed(12))
    assert lowpass_geometry_correlation(y, x, patch=4) > 0.85
    # 逐像素位置没有被搬动：低频结构的空间相关性足够高
    assert x.shape == y.shape


def test_destyle_suppresses_hf_and_changes_color():
    y = _structured_image(seed=13)
    hf_cfg = DestyleConfig(
        color_prob=0.0,
        lowfreq_prob=0.0,
        hf_suppress_prob=1.0,
        hf_suppress_min=0.05,
        hf_suppress_max=0.05,
        blur_prob=0.0,
        clamp_range=None,
    )
    x = destyle_image(y, hf_cfg, torch.Generator().manual_seed(14))
    hf_y = (y - F.avg_pool2d(y, 3, 1, 1)).pow(2).mean()
    hf_x = (x - F.avg_pool2d(x, 3, 1, 1)).pow(2).mean()
    assert hf_x < hf_y * 0.5

    color_cfg = DestyleConfig(
        color_prob=1.0,
        color_mode_probs=(0.0, 1.0, 0.0),
        color_shift=0.35,
        color_scale=0.6,
        lowfreq_prob=0.0,
        hf_suppress_prob=0.0,
        clamp_range=None,
    )
    x2 = destyle_image(y, color_cfg, torch.Generator().manual_seed(15))
    assert (x2.mean(dim=(0, 2, 3)) - y.mean(dim=(0, 2, 3))).abs().mean() > 0.05


def test_destyle_latent_is_geometry_preserving_soft():
    g = torch.Generator().manual_seed(16)
    z = torch.randn(SHAPE, generator=g)
    x = destyle_latent(z, DestyleConfig(clamp_range=None), torch.Generator().manual_seed(17))
    assert x.shape == z.shape
    assert torch.isfinite(x).all()


def test_shuffle_patches_preserves_patch_multiset():
    g = torch.Generator().manual_seed(18)
    x = torch.randn(SHAPE, generator=g)
    y = shuffle_patches(x, 4, torch.Generator().manual_seed(19))
    px = F.unfold(x, 4, 4).sum(dim=1).sort(dim=-1).values
    py = F.unfold(y, 4, 4).sum(dim=1).sort(dim=-1).values
    assert torch.allclose(px, py, atol=1e-6)


def test_mix_patches_uses_donor():
    g = torch.Generator().manual_seed(20)
    x = torch.randn(SHAPE, generator=g)
    donor = torch.randn(SHAPE, generator=g) + 10.0
    mixed = mix_patches(x, donor, 4, ratio=0.5, generator=torch.Generator().manual_seed(21))
    assert not torch.allclose(mixed, x)
    assert mixed.mean() > x.mean()


def test_build_latent_pair_shapes():
    g = torch.Generator().manual_seed(22)
    target = torch.randn(SHAPE, generator=g)
    ref = torch.randn(SHAPE, generator=g)
    donor = torch.randn(SHAPE, generator=g)
    pair = build_latent_pair(
        target, ref, donor,
        destyle_cfg=DestyleConfig(clamp_range=None),
        mix_cfg=ReferenceMixConfig(patch_size=4),
        generator=torch.Generator().manual_seed(23),
    )
    assert pair["content_latent"].shape == target.shape
    assert pair["reference_latent"].shape == ref.shape
    assert pair["target_latent"].shape == target.shape


# ---------------------------------------------------------------------------
# 阶段 1 筛选器
# ---------------------------------------------------------------------------

def test_screener_rejects_identity_and_reference_copy():
    screener = StyleTargetScreener(patch_size=4, patch_stride=4, num_projections=32)
    g = torch.Generator().manual_seed(24)
    content = torch.randn(SHAPE, generator=g)
    ref = torch.randn(SHAPE, generator=g) * 2.0

    identity = screener.score(content, content, ref)
    assert not identity["passed"]
    assert identity["style_gain"] == pytest.approx(0.0, abs=1e-5)

    copy = screener.score(ref, content, ref)
    assert not copy["passed"]
    assert copy["reference_copy_corr"] > 0.9


def test_screener_passes_style_shifted_content():
    screener = StyleTargetScreener(patch_size=4, patch_stride=4, num_projections=32)
    g = torch.Generator().manual_seed(25)
    content = torch.randn(SHAPE, generator=g)
    ref = torch.randn(SHAPE, generator=g) * 2.0
    stylized = content * 2.0  # 只改分布尺度，不改结构
    report = screener.score(stylized, content, ref)
    assert report["style_gain"] > 0.05
    assert report["content_keep"] > 0.9
    assert report["reference_copy_corr"] < 0.5
    assert report["passed"]


# ---------------------------------------------------------------------------
# 离线特征 cache
# ---------------------------------------------------------------------------

def test_offline_style_feature_cache_roundtrip(tmp_path):
    writer = OfflineStyleFeatureCacheWriter(tmp_path, shard_size=2)
    op = torch.randn(16)
    writer.add("style/0", global_op=op, style_id=0)
    writer.add("style/1", global_op=torch.randn(16), patch_tokens=torch.randn(8, 16), style_id=1)
    writer.close(meta={"source": "unit-test"})

    cache = OfflineStyleFeatureCache(tmp_path)
    assert cache.count == 2
    assert torch.allclose(cache.get_global_op("style/0"), op)
    assert cache.get_global_op("missing") is None
    assert cache.get("style/1")["patch_tokens"].shape == (8, 16)


# ---------------------------------------------------------------------------
# 三阶段供应器
# ---------------------------------------------------------------------------

def test_stage1_requires_unpaired():
    provider = BPseudoTargetProvider(make_operator(seed=26))
    g = torch.Generator().manual_seed(27)
    paired = {
        "content_latent": torch.randn(SHAPE, generator=g),
        "reference_latent": torch.randn(SHAPE, generator=g),
        "target_latent": torch.randn(SHAPE, generator=g),
    }
    with pytest.raises(ValueError):
        provider.stage1_loss(paired, unpaired=None, require_unpaired=True)


def test_stage1_loss_and_step():
    torch.manual_seed(28)
    op = make_operator(seed=28)
    provider = BPseudoTargetProvider(op)
    g = torch.Generator().manual_seed(29)
    paired = {
        "content_latent": torch.randn(SHAPE, generator=g),
        "reference_latent": torch.randn(SHAPE, generator=g),
        "target_latent": torch.randn(SHAPE, generator=g),
    }
    unpaired = {
        "content_latent": torch.randn(SHAPE, generator=g),
        "reference_latent": torch.randn(SHAPE, generator=g),
    }
    opt = torch.optim.SGD(op.parameters(), lr=1e-2)
    logs = provider.stage1_step(paired, unpaired, optimizer=opt, grad_clip=1.0)
    assert torch.isfinite(logs["loss"])
    assert "paired/loss_style_swd" in logs
    assert "unpaired/loss_style_swd" in logs
    assert op.repaint.renderer.style_gate.grad is not None


def test_endpoint_requires_screen_and_freeze():
    provider = BPseudoTargetProvider(make_operator(seed=30))
    content, ref, _ = make_latents(seed=31)
    with pytest.raises(RuntimeError):
        provider.build_fm_endpoint(content, ref)

    provider.freeze_operator()
    with pytest.raises(RuntimeError):
        provider.build_fm_endpoint(content, ref)

    provider.mark_screened({"passed": True, "style_gain": 0.5})
    ep = provider.build_fm_endpoint(content, ref)
    assert ep is not None
    assert ep["z1"].requires_grad is False
    assert torch.allclose(ep["velocity"], ep["z1"] - content, atol=1e-6)


def test_endpoint_from_offline_cache(tmp_path):
    op = make_operator(seed=32)
    writer = OfflineStyleFeatureCacheWriter(tmp_path)
    op_vec = torch.randn(op.style_dim)
    writer.add("style/0", global_op=op_vec, style_id=0)
    writer.close()
    cache = OfflineStyleFeatureCache(tmp_path)

    provider = BPseudoTargetProvider(op, style_feature_cache=cache)
    provider.mark_screened({"passed": True})
    provider.freeze_operator()
    content, _, _ = make_latents(seed=33)
    ep = provider.build_fm_endpoint(content, reference_latent=None, style_key="style/0")
    assert ep is not None
    assert ep["z1"].shape == content.shape
    assert provider.build_fm_endpoint(content, style_key="style/missing") is None


def test_export_endpoints_roundtrip(tmp_path):
    provider = BPseudoTargetProvider(make_operator(seed=34))
    content, ref, _ = make_latents(seed=35)
    out_dir = tmp_path / "b_endpoints"
    provider.export_endpoints(
        [
            {
                "content_key": ["content/0"],
                "style_id": [0],
                "content_latent": content,
                "reference_latent": ref,
            }
        ],
        out_dir,
        only_passed=False,
    )
    cache = PseudoTargetCache(out_dir)
    latent = cache.get("content/0", 0)
    assert latent is not None
    assert latent.shape == SHAPE[1:]
    assert torch.isfinite(latent).all()


# ---------------------------------------------------------------------------
# 训练诊断
# ---------------------------------------------------------------------------

def test_curve_recorder_milestones_and_write(tmp_path):
    rec = CurveRecorder()
    for step, loss in ((1, 5.0), (2000, 3.0), (10000, 1.2), (30000, 0.4)):
        rec.record(step, "train", {"loss": loss})
    for step, gain in ((2000, 0.02), (10000, 0.06), (30000, 0.11)):
        rec.record(step, "holdout", {"style_gain": gain, "content_keep": 0.9})
    assert rec.should_record(2000) and not rec.should_record(500)

    summ = rec.summary()
    assert summ["milestones"][30000]["train"]["loss"] == pytest.approx(0.4)
    assert summ["milestones"][10000]["holdout"]["style_gain"] == pytest.approx(0.06)
    assert summ["train_fit"]["final_loss"] == pytest.approx(0.4)

    out = rec.write(tmp_path / "diag")
    assert (out / "b_curve.jsonl").exists()
    assert (out / "b_curve_summary.md").exists()
    assert "milestone 30000" in (out / "b_curve_summary.md").read_text(encoding="utf-8")


class _IdentityOperator:
    """无参数恒等算子桩，用来验证 AdaIN-off 诊断能区分"网络 vs 后处理"。"""

    def __call__(self, content, reference=None, **_kwargs):
        return {"z_star": content}


def test_style_gain_diagnostics_detects_adain_only_style():
    diag = StyleGainDiagnostics(_IdentityOperator(), adain_strength=1.0)
    g = torch.Generator().manual_seed(38)
    content = torch.randn(SHAPE, generator=g)
    ref = torch.randn(SHAPE, generator=g) * 3.0 + 2.0
    rep = diag.compare(content, ref)
    # 恒等映射：关掉 AdaIN 后没有风格增益。
    assert rep["adain_off"]["style_gain"] == pytest.approx(0.0, abs=1e-5)
    assert rep["adain_on"]["style_gain"] > 0.1
    assert rep["adain_contribution"] > 0.1
    assert not rep["network_style_ok"]


def test_channel_adain_strength_zero_is_identity():
    g = torch.Generator().manual_seed(39)
    z = torch.randn(SHAPE, generator=g)
    ref = torch.randn(SHAPE, generator=g) * 2.0
    assert torch.allclose(channel_adain(z, ref, 0.0), z)


class _QuadraticProvider:
    """把 loss 设计成 (p-3)^2，用于验证 OverfitProbe 的机制本身。"""

    def __init__(self):
        self.operator = torch.nn.Module()
        self.operator.p = torch.nn.Parameter(torch.tensor(0.0))

    def stage1_step(self, paired, unpaired, *, optimizer=None, grad_clip=0.0, unpaired_weight=0.5):
        loss = (self.operator.p - 3.0).pow(2) + 1.0
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.operator.parameters(), grad_clip)
            optimizer.step()
        return {"loss": loss.detach()}


def test_overfit_probe_mechanism_reaches_target_ratio():
    probe = OverfitProbe(
        _QuadraticProvider(),
        OverfitConfig(steps=60, lr=0.2, n_samples=4, seed=0),
    )
    g = torch.Generator().manual_seed(40)
    samples = [
        {
            "content_latent": torch.randn(SHAPE, generator=g),
            "reference_latent": torch.randn(SHAPE, generator=g),
            "target_latent": torch.randn(SHAPE, generator=g),
        }
        for _ in range(4)
    ]
    out = probe.run(samples)
    assert out["ratio"] <= out["target_ratio"]
    assert out["overfit_ok"]
    assert len(out["history"]) == 60


def test_evaluate_holdout_returns_means():
    provider = BPseudoTargetProvider(make_operator(seed=41))
    g = torch.Generator().manual_seed(42)
    samples = [
        {
            "content_latent": torch.randn(SHAPE, generator=g),
            "reference_latent": torch.randn(SHAPE, generator=g),
        }
        for _ in range(3)
    ]
    out = evaluate_holdout(provider, samples)
    assert out["num_samples"] == 3
    assert 0.0 <= out["pass_rate"] <= 1.0
    assert 0.0 <= out["content_keep"] <= 1.0
    assert torch.isfinite(torch.tensor(out["style_gain"]))
