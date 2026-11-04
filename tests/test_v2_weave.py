"""WEAVE-v2 单元测试（CPU）。

覆盖：默认路径不变、梯度回流、AdaLN 门控开关、identity 分支、PatchNCE
图内负样本、判别器条件、形状/显存/参数量断言、Style stats 与蒸馏 stopgrad。
"""
from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from pathlib import Path

from config_schema import load_experiment_config
from flow import FlowMatchingObjective

from v2.config import V2Config, expand_variant_defaults
from v2.discriminator import ConditionalPatchDiscriminator
from v2.distill import DistillObjective
from v2.model import FiLMResBlock, V2Backbone, V2Generator, count_parameters
from v2.objective import PatchNCEEncoder, PatchNCELoss, V2DistributionObjective, patch_nce_logits
from v2.style_stats import build_style_stats_from_tensors, compute_style_stats


def tiny_cfg(**kwargs) -> V2Config:
    base = V2Config(
        enabled=True,
        channels=(16, 32, 48, 48),
        blocks_per_stage=(1, 1, 1, 1),
        decoder_blocks_per_stage=(1, 1, 1, 1),
        style_embed_dim=32,
        style_tokens=8,
        style_token_heads=2,
        attn_heads=2,
        norm_groups=8,
        disc_base_channels=16,
        disc_style_dim=32,
        nce_hidden=16,
        nce_num_patches=16,
        geometry_max_patches=32,
    )
    return replace(base, **kwargs)


def test_default_path_is_unchanged():
    cfg = V2Config.from_mapping({})
    assert cfg.enabled is False
    exp = load_experiment_config(Path(__file__).parents[1] / "config.json")
    assert V2Config.from_experiment(exp).enabled is False
    # 旧目标对象仍然可以被原样构建
    obj = FlowMatchingObjective(exp)
    assert obj is not None
    # v2 包导入不改变旧模型的参数规模（历史基线约 1.04M）
    from model import build_model_from_config

    legacy = build_model_from_config(exp.model, bridge_cfg=exp.bridge)
    assert count_parameters(legacy) < 2_000_000


def test_gradients_flow_through_generator():
    torch.manual_seed(0)
    cfg = tiny_cfg()
    gen = V2Generator(cfg, latent_channels=4, num_styles=5)
    x = torch.randn(2, 4, 32, 32) * 0.2
    out = gen(x, torch.tensor([0, 3]))
    loss = out["z_hat"].pow(2).mean()
    loss.backward()
    checked = {
        "stem": gen.backbone.stem,
        "enc0": gen.backbone.enc_blocks[0][0].conv1,
        "enc3": gen.backbone.enc_blocks[3][0].conv1,
        "dec0": gen.backbone.dec_blocks[0][0].conv1,
        "head_ll": gen.backbone.head_ll,
        "head_hf": gen.backbone.head_hf,
    }
    for name, module in checked.items():
        assert module.weight.grad is not None and module.weight.grad.abs().sum() > 0, name
    # 低尺度自注意力输出投影非零梯度
    assert gen.backbone.lowest_attn.out.weight.grad is not None


def test_adaln_gate_can_be_disabled_and_enabled():
    torch.manual_seed(1)
    cfg = tiny_cfg(adaln_gate_init=0.0)
    blk = FiLMResBlock(16, 16, cfg.style_embed_dim, groups=8, adaln_enabled=True, gate_init=0.0)
    with torch.no_grad():
        torch.nn.init.normal_(blk.film.weight, std=0.2)
        torch.nn.init.zeros_(blk.film.bias)
    x = torch.randn(1, 16, 8, 8)
    c0 = torch.zeros(1, cfg.style_embed_dim)
    c1 = torch.randn(1, cfg.style_embed_dim)
    # gate=tanh(0)=0 -> 两个条件严格相同
    assert torch.allclose(blk(x, c0), blk(x, c1), atol=1e-6)
    with torch.no_grad():
        blk.gate.fill_(1.0)
    assert not torch.allclose(blk(x, c0), blk(x, c1), atol=1e-4)
    # 完全关闭 AdaLN 时条件被忽略
    off = FiLMResBlock(16, 16, cfg.style_embed_dim, groups=8, adaln_enabled=False, gate_init=0.0)
    assert torch.allclose(off(x, c0), off(x, c1), atol=1e-6)


def test_identity_branch_only_uses_identity_rows():
    torch.manual_seed(2)
    cfg = tiny_cfg(w_adv=0.0, w_patchnce=0.0, w_geometry=0.0, w_identity=1.0)
    gen = V2Generator(cfg, latent_channels=4, num_styles=5)
    obj = V2DistributionObjective(cfg, latent_channels=4, num_styles=5)
    x = torch.randn(3, 4, 32, 32) * 0.2
    target = torch.randn(3, 4, 32, 32) * 0.2
    mask = torch.tensor([1.0, 0.0, 1.0])
    res = obj.compute_generator_loss(
        gen, x, target, torch.tensor([0, 1, 2]), source_style_id=torch.tensor([0, 4, 2]),
        conditioning={"identity_mask": mask},
    )
    fake = res["z_hat"]
    expected = torch.nn.functional.l1_loss(fake[mask.bool()], x[mask.bool()])
    assert torch.allclose(res["loss_identity"], expected, atol=1e-5)
    assert float(res["identity_fraction"]) == pytest.approx(2.0 / 3.0)
    # 无 identity 行时该项为 0
    res2 = obj.compute_generator_loss(
        gen, x, target, torch.tensor([0, 1, 2]), source_style_id=torch.tensor([1, 2, 3]),
        conditioning={"identity_mask": torch.zeros(3)},
    )
    assert float(res2["loss_identity"]) == 0.0
    # identity loss 能回传到生成器（总 loss 只含 identity 项）
    res["loss"].backward()
    assert gen.backbone.head_ll.weight.grad is not None


def test_patchnce_in_image_negatives_and_batch_isolation():
    torch.manual_seed(3)
    b, c, n = 2, 8, 16
    src = torch.randn(b, c, n)
    tgt = torch.randn(b, c, n)
    pos = torch.tensor([0, 3, 7])
    logits = patch_nce_logits(src, tgt, pos, temperature=0.07)
    assert logits.shape == (b, pos.numel(), n)
    # 正样本就是采样位置本身：与 cos(src[pos], tgt[pos]) 一致
    s = torch.nn.functional.normalize(src, dim=1)
    t = torch.nn.functional.normalize(tgt, dim=1)
    for p, idx in enumerate(pos.tolist()):
        manual = torch.einsum("bc,bcn->bn", s[:, :, idx], t) / 0.07
        assert torch.allclose(logits[:, p, :], manual, atol=1e-5)
    # 同一 batch 的另一个样本的特征变化不影响第一个样本的 logits
    src2 = src.clone()
    src2[1] = src2[1] + 5.0
    logits2 = patch_nce_logits(src2, tgt, pos, temperature=0.07)
    assert torch.allclose(logits[0], logits2[0], atol=1e-6)
    # 图内其他位置确实作为负样本参与：改一个位置的特征会改变对应 logit
    tgt3 = tgt.clone()
    tgt3[0, :, 5] = tgt3[0, :, 5] + 3.0
    logits3 = patch_nce_logits(src, tgt3, pos, temperature=0.07)
    assert not torch.isclose(logits3[0, 0, 5], logits[0, 0, 5], atol=1e-4)
    assert torch.isfinite(logits).all()


def test_patchnce_loss_and_encoder_grad():
    torch.manual_seed(4)
    cfg = tiny_cfg()
    enc = PatchNCEEncoder(4, cfg.nce_hidden)
    nce = PatchNCELoss(cfg)
    nce.build_proj([cfg.nce_hidden * 2, cfg.nce_hidden * 2], cfg.nce_hidden)
    x = torch.randn(2, 4, 32, 32)
    y = x + 0.1 * torch.randn_like(x)
    loss = nce(enc(x), enc(y))
    assert torch.isfinite(loss)
    loss.backward()
    assert enc.net[0][0].weight.grad is not None and enc.net[0][0].weight.grad.abs().sum() > 0
    assert nce.proj["0"][0].weight.grad is not None


def test_discriminator_conditioning_and_shapes():
    torch.manual_seed(5)
    cfg = tiny_cfg()
    disc = ConditionalPatchDiscriminator(cfg, latent_channels=4, num_styles=5)
    x = torch.randn(2, 4, 32, 32)
    logits_a = disc(x, torch.tensor([0, 0]))
    logits_b = disc(x, torch.tensor([4, 4]))
    assert len(logits_a) == cfg.disc_num_scales
    assert logits_a[0].shape == (2, 1, logits_a[0].shape[-2], logits_a[0].shape[-1])
    mean_a = torch.cat([p.reshape(-1) for p in logits_a]).mean()
    mean_b = torch.cat([p.reshape(-1) for p in logits_b]).mean()
    assert not torch.isclose(mean_a, mean_b, atol=1e-4)
    mean_a.backward()
    assert disc.style_embed.weight.grad is not None
    assert disc.style_embed.weight.grad.abs().sum() > 0


def test_generator_shapes_and_parameter_budget():
    cfg = V2Config(enabled=True)  # 默认 96/192/256/256
    gen = V2Generator(cfg, latent_channels=4, num_styles=5)
    params = count_parameters(gen)
    assert 4_500_000 < params < 8_000_000  # 默认配置落在 5–8M 预算
    x = torch.randn(1, 4, 64, 64)
    out = gen(x, torch.tensor([0]))
    assert out["z_hat"].shape == x.shape
    assert out["delta_ll"].shape[1] == 4
    assert out["delta_lh"].shape[1] == 4
    assert out["delta_hl"].shape[1] == 4
    assert out["delta_hh"].shape[1] == 4  # HH 不再关闭

    vel = V2Backbone(cfg, latent_channels=4, num_styles=5, mode="velocity")
    vout = vel(x, t=torch.tensor([0.5]))
    assert set(vout) == {"ll", "lh", "hl", "hh"}
    assert vout["ll"].shape == (1, 4, 32, 32)  # 子带分辨率 H/2


def test_style_stats_are_style_id_only_and_gated():
    torch.manual_seed(6)
    tensors = {i: torch.randn(4, 4, 16, 16) * (0.1 + 0.1 * i) for i in range(5)}
    bank = build_style_stats_from_tensors(tensors, num_styles=5)
    assert bank.shape == (5, 32)
    assert torch.isfinite(bank).all()
    assert compute_style_stats(tensors[0]).shape == (32,)

    cfg = tiny_cfg(style_stats_gate_init=0.0)
    gen = V2Generator(cfg, latent_channels=4, num_styles=5)
    x = torch.randn(1, 4, 32, 32)
    with torch.no_grad():
        a = gen(x, torch.tensor([0]))["z_hat"]
        b = gen(x, torch.tensor([0]), style_stats=bank)["z_hat"]
        e_plain, _ = gen.backbone.style(torch.tensor([0]), batch=1, device=x.device, dtype=x.dtype)
        e_zero, _ = gen.backbone.style(
            torch.tensor([0]), batch=1, device=x.device, dtype=x.dtype, style_stats=bank
        )
    assert torch.allclose(a, b, atol=1e-6)  # gate=0 时统计条件严格不生效
    assert torch.allclose(e_plain, e_zero, atol=1e-8)
    with torch.no_grad():
        gen.backbone.style.stats_gate.fill_(1.0)
        e_on, _ = gen.backbone.style(
            torch.tensor([0]), batch=1, device=x.device, dtype=x.dtype, style_stats=bank
        )
    assert not torch.allclose(e_plain, e_on, atol=1e-5)  # 打开后统计条件进入 style embedding


def test_distill_teacher_is_stopgrad_and_off_by_default():
    cfg = tiny_cfg()
    assert cfg.distill_enabled is False
    assert expand_variant_defaults(replace(cfg, variant="V5")).distill_enabled is False

    torch.manual_seed(7)
    teacher = V2Generator(cfg, latent_channels=4, num_styles=5).eval()
    student = V2Backbone(cfg, latent_channels=4, num_styles=5, mode="velocity")
    obj = DistillObjective(cfg)
    x = torch.randn(2, 4, 32, 32)
    ids = torch.tensor([0, 2])
    res = obj.compute(student, teacher, x, ids)
    res["loss"].backward()
    assert all(p.grad is None for p in teacher.parameters())
    assert student.stem.weight.grad is not None
