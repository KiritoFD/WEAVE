"""V4Inference 与 gen_smallset 的采样口径一致性测试。

重点：正式推理入口必须把 ``cfg.prediction_type`` 透传给 ``sdedit`` / ``ddim_sample``；
默认参数是 ``"eps"``，v-pred 的 ckpt 若走默认值会被错误解释。
"""
from __future__ import annotations

from dataclasses import asdict

import torch

from v4.config import V4Config
from v4.diffusion import Schedule, ddim_sample, sdedit
from v4.inference import V4Inference
from v4.model import V4UNet


def _make_payload(pred_type: str, seed: int = 0):
    cfg = V4Config(
        channels=(16, 24, 32, 48),
        blocks_per_stage=(1, 1, 1, 1),
        mid_blocks=1,
        attn_heads=2,
        norm_groups=4,
        time_embed_dim=64,
        style_embed_dim=32,
        cond_dim=64,
        prediction_type=pred_type,
    )
    net = V4UNet(cfg)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in net.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.02)
    payload = {
        "variant": "V4DIFF",
        "v4_config": asdict(cfg),
        "model": {k: v.clone() for k, v in net.state_dict().items()},
        "latent_scale_factor": cfg.latent_scale_factor,
        "step": 0,
    }
    return cfg, net, payload


def _build(payload, monkeypatch, *, steps: int, strength: float, seed: int) -> V4Inference:
    monkeypatch.setenv("V4_NUM_STEPS", str(steps))
    monkeypatch.setenv("V4_EDIT_STRENGTH", str(strength))
    monkeypatch.setenv("V4_SEED", str(seed))
    monkeypatch.setenv("V4_CFG", "1.0")
    return V4Inference(payload, device="cpu", num_steps_fallback=steps)


def test_sdedit_parity_all_prediction_types(monkeypatch):
    """同一 ckpt / latent / Style-ID / 种子 / 采样参数，适配器与 gen_smallset 口径必须一致。"""
    x0 = torch.randn(1, 4, 8, 8, generator=torch.Generator().manual_seed(11))
    sid = torch.tensor([2])
    steps, strength, seed = 4, 0.45, 7

    for pred in ("v", "eps"):
        cfg, net, payload = _make_payload(pred)
        inf = _build(payload, monkeypatch, steps=steps, strength=strength, seed=seed)
        got = inf.generation_with_target_latent(x0.clone(), sid.clone())
        ref = sdedit(net, x0.clone(), sid.clone(), inf.schedule,
                     strength=strength, num_steps=steps, cfg_scale=1.0,
                     prediction_type=cfg.prediction_type,
                     generator=torch.Generator().manual_seed(seed))
        err = (got - ref).abs().max().item()
        assert err <= 1e-5, f"prediction_type={pred}: max_abs_err={err:.3e} > 1e-5"


def test_prediction_type_actually_matters(monkeypatch):
    """反例守卫：若适配器仍漏传 prediction_type（退回默认 eps），v 与 eps 的差异会被抹平。"""
    x0 = torch.randn(1, 4, 8, 8, generator=torch.Generator().manual_seed(12))
    sid = torch.tensor([1])
    steps, strength, seed = 4, 0.6, 3
    cfg, net, payload = _make_payload("v")
    inf = _build(payload, monkeypatch, steps=steps, strength=strength, seed=seed)
    got = inf.generation_with_target_latent(x0.clone(), sid.clone())
    as_eps = sdedit(net, x0.clone(), sid.clone(), inf.schedule,
                    strength=strength, num_steps=steps, cfg_scale=1.0,
                    prediction_type="eps",
                    generator=torch.Generator().manual_seed(seed))
    diff = (got - as_eps).abs().max().item()
    assert diff > 1e-3, f"v-pred 与 eps 解码差异过小（{diff:.3e}），无法区分口径"


def test_sample_styles_parity(monkeypatch):
    """纯风格采样：ddim_sample 也必须透传 prediction_type。"""
    shape = (1, 4, 8, 8)
    style_ids = torch.tensor([3])
    steps, seed = 4, 5
    cfg, net, payload = _make_payload("v")
    inf = _build(payload, monkeypatch, steps=steps, strength=0.45, seed=seed)
    got = inf.sample_styles(style_ids, shape, num_steps=steps, cfg_scale=1.0, seed=seed)
    z = torch.randn(shape, generator=torch.Generator().manual_seed(seed))
    ref = ddim_sample(net, z, style_ids.clone(),
                      Schedule.linear(cfg.timesteps, cfg.beta_start, cfg.beta_end, device="cpu"),
                      t_start=cfg.timesteps - 1, num_steps=steps, cfg_scale=1.0,
                      prediction_type=cfg.prediction_type)
    err = (got - ref).abs().max().item()
    assert err <= 1e-5, f"max_abs_err={err:.3e} > 1e-5"
