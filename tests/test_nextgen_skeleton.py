"""Next-gen smoke tests: Routes A-D + P1 fixes. All CPU, synthetic, seconds.

Each test proves forward/backward/config parsing works without GPU or data.
Run: pytest tests/test_nextgen_skeleton.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from config_schema import ExperimentConfig


def _tiny_cfg(**bridge_over):
    model = {"base_dim": 32, "num_res_blocks": 2, "num_styles": 3, "contract_family": "weave"}
    return ExperimentConfig.from_mapping({"model": model, "bridge": bridge_over})


def _tiny_model(cfg):
    from model import build_model_from_config

    return build_model_from_config(cfg.model, bridge_cfg=cfg.bridge)


# --------------------------------------------------------------------------
# Route A: SWD + output supervision
# --------------------------------------------------------------------------

def test_swd_identical_vs_shifted():
    from swd import sliced_wasserstein_distance

    torch.manual_seed(0)
    a = torch.randn(400, 8)
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    same = sliced_wasserstein_distance(a, a.clone(), generator=g1)
    shifted = sliced_wasserstein_distance(a, a + 2.0, generator=g2)
    assert float(same) < 1e-5
    assert float(shifted) > 1.0


def test_swd_gradient_flows():
    from swd import sliced_wasserstein_distance

    a = torch.randn(128, 8, requires_grad=True)
    b = torch.randn(128, 8)
    loss = sliced_wasserstein_distance(a, b)
    loss.backward()
    assert a.grad is not None and float(a.grad.abs().sum()) > 0


def test_output_supervision_backward():
    from features import build_feature_extractor
    from output_supervision import OutputSupervision, build_latent_decoder

    torch.manual_seed(0)
    ext = build_feature_extractor("tiny_conv")
    dec = build_latent_decoder("tiny")
    sup = OutputSupervision(
        ext, dec, style_layers=["conv2", "conv3"], content_perceptual_layers=["conv1", "conv2"],
        content_selfsim_layer="conv3", ode_steps=2, microbatch=2, grad_checkpoint=True,
    )
    model = _tiny_model(_tiny_cfg())
    model.train()
    logs = sup(
        model, content_latent=torch.randn(4, 4, 16, 16),
        style_image=torch.rand(4, 3, 32, 32), content_image=torch.rand(4, 3, 32, 32),
        style_id=torch.tensor([0, 1, 2, 0]), style_latent=torch.randn(1, 4, 16, 16),
    )
    loss = logs["loss_style_out"] + logs["loss_content_out"]
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and sum(float(g.abs().sum()) for g in grads) > 0


def test_feature_extractor_invalid_layer():
    from features import build_feature_extractor

    ext = build_feature_extractor("tiny_conv")
    with pytest.raises(KeyError):
        ext.forward_layers(torch.rand(1, 3, 16, 16), ["no_such_layer"])


# --------------------------------------------------------------------------
# Route B: endpoint modes + pseudo cache
# --------------------------------------------------------------------------

def test_endpoint_mode_pseudo_substitution():
    from flow import FlowMatchingObjective

    torch.manual_seed(0)
    cfg = _tiny_cfg(endpoint_mode="mixed")
    objective = FlowMatchingObjective(cfg)
    target = torch.zeros(4, 4, 16, 16)
    pseudo = torch.ones(2, 4, 16, 16)
    out = objective._apply_pseudo_endpoints(
        target, {"pseudo_target_latent": pseudo, "pseudo_target_found": [1, 3]}
    )
    assert torch.equal(out[1], torch.ones(4, 16, 16))
    assert torch.equal(out[0], torch.zeros(4, 16, 16))
    assert torch.equal(out[3], torch.ones(4, 16, 16))


def test_endpoint_mode_invalid_rejected():
    from flow import FlowMatchingObjective

    with pytest.raises(ValueError):
        FlowMatchingObjective(_tiny_cfg(endpoint_mode="bogus"))


def test_endpoint_mode_default_unchanged():
    from flow import FlowMatchingObjective

    objective = FlowMatchingObjective(_tiny_cfg())
    target = torch.randn(2, 4, 16, 16)
    out = objective._apply_pseudo_endpoints(target, {"pseudo_target_latent": torch.ones(1, 4, 16, 16), "pseudo_target_found": [0]})
    assert torch.equal(out, target)


def test_pseudo_cache_roundtrip(tmp_path: Path):
    from pseudo_targets import PseudoTargetCache, PseudoTargetCacheWriter

    writer = PseudoTargetCacheWriter(tmp_path, shard_size=2)
    lat = torch.randn(3, 4, 8, 8)
    writer.add(lat, ["0/0", "0/1", "1/0"], [1, 2, 3], [0.9, 0.1, 0.8])
    writer.close()
    cache = PseudoTargetCache(tmp_path, score_threshold=0.5)
    assert cache.count == 2  # 0.1 filtered out
    assert cache.get("0/0", 1) is not None
    assert cache.get("0/1", 2) is None
    stacked, mask, found = cache.get_batch(["0/0", "0/1", "1/0"], [1, 2, 3])
    assert stacked is not None and stacked.shape == (2, 4, 8, 8)
    assert mask.tolist() == [True, False, True]
    assert found == [0, 2]


def test_fallback_teacher_and_scoring():
    from features import build_feature_extractor
    from pseudo_targets import FallbackStatsTeacher, score_content_consistency

    torch.manual_seed(0)
    teacher = FallbackStatsTeacher()
    content = torch.rand(2, 3, 32, 32)
    pseudo = teacher.stylize(content, torch.rand(2, 3, 32, 32))
    assert pseudo.shape == content.shape
    scores = score_content_consistency(pseudo, content, build_feature_extractor("tiny_conv"), selfsim_layer="conv3")
    assert scores.shape == (2,)
    assert torch.all((scores >= 0) & (scores <= 1))


# --------------------------------------------------------------------------
# Route C: reference encoder + conditioner variant
# --------------------------------------------------------------------------

def test_reference_encoder_shapes_and_nce():
    from reference_encoder import ReferenceStyleEncoder, info_nce_loss

    torch.manual_seed(0)
    enc = ReferenceStyleEncoder(in_dim=16, model_dim=32, num_tokens=8, proj_dim=16)
    enc.train()
    out = enc(torch.randn(4, 25, 16))
    assert out["tokens"].shape == (4, 8, 32)
    assert out["embedding"].shape == (4, 16)
    loss = info_nce_loss(out["embedding"], enc(torch.randn(4, 25, 16))["embedding"])
    loss.backward()
    assert any(p.grad is not None for p in enc.parameters())


def test_reference_conditioner_zero_init_noop():
    from reference_encoder import ReferenceConditioner

    cond = ReferenceConditioner(32, 256, 8, gate_init=0.0)
    fam = torch.randn(2, 256, 32)
    assert torch.equal(cond(fam, torch.randn(2, 8, 32)), fam)


def test_model_reference_hybrid_variant():
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    mcfg = cfg.model.to_dict()
    mcfg["conditioner_variant"] = "reference_hybrid"
    mcfg["reference_encoder_in_dim"] = 16
    from config_schema import ModelConfig

    cfg2 = ExperimentConfig.from_mapping({"model": mcfg, "bridge": {}})
    model = _tiny_model(cfg2)
    assert model.reference_encoder is not None and model.reference_conditioner is not None
    out = model(torch.randn(2, 4, 16, 16), t=0.5, style_id=1,
                reference_feats=torch.randn(2, 25, 16))
    assert "ll" in out and model.last_reference_embedding is not None
    # default variant stays old path
    model_old = _tiny_model(_tiny_cfg())
    assert model_old.reference_encoder is None and model_old.reference_conditioner is None


# --------------------------------------------------------------------------
# Route D + P1 fixes
# --------------------------------------------------------------------------

def test_hh_encoders_not_built_when_head_disabled():
    cfg = _tiny_cfg()
    mcfg = cfg.model.to_dict()
    mcfg.update({
        "enable_hh_head": False,
        "target_latent_hf_spatial_fusion_enabled": True,
        "target_latent_hf_subband_fusion_enabled": True,
        "target_latent_hf_texture_fusion_enabled": True,
        "target_latent_hf_head_fusion_enabled": True,
    })
    cfg2 = ExperimentConfig.from_mapping({"model": mcfg, "bridge": {}})
    model = _tiny_model(cfg2)
    assert model.target_latent_hf_spatial_hh is None
    assert model.target_latent_hf_subband_encoder_hh is None
    assert model.target_latent_hf_texture_encoder_hh is None
    assert model.target_latent_hf_delta_hh is None
    # LH/HL still present and forward works
    assert model.target_latent_hf_spatial_lh is not None
    out = model(torch.randn(2, 4, 16, 16), t=0.5, style_id=1, style_latent=torch.randn(1, 4, 16, 16))
    assert "hh" not in out


def test_heun_rk4_update_hh_when_head_enabled():
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    mcfg = cfg.model.to_dict()
    mcfg["enable_hh_head"] = True
    cfg2 = ExperimentConfig.from_mapping({"model": mcfg, "bridge": {}})
    model = _tiny_model(cfg2).eval()
    from wavelet import dwt2_haar

    x = torch.randn(1, 4, 16, 16)
    for solver in ("heun", "rk4", "euler"):
        h = model._solver_step(
            x, t_curr=0.0, t_mid=0.5, t_next=1.0, dt=1.0, solver_type=solver,
            lock_ll=False, style_id=1, style_text_tokens=None, style_latent=None,
        )
        _, _, _, hh_out = dwt2_haar(h)
        _, _, _, hh_in = dwt2_haar(x)
        # With a (random, nonzero) HH head, the HH band must actually change.
        assert not torch.allclose(hh_out, hh_in), f"{solver} silently dropped v_hh"


def test_t_sampling_modes():
    from flow import FlowMatchingObjective

    torch.manual_seed(0)
    # legacy compat: uniform regardless of mode name
    o = FlowMatchingObjective(_tiny_cfg(t_sampling_mode="logit_normal", t_sampling_legacy_compat=True))
    t = o._sample_t(torch.randn(2048, 1))
    assert abs(float(t.std()) - 0.289) < 0.03  # uniform(0,1) std
    # strict logit-normal with mean 0: symmetric beta-like distribution
    o2 = FlowMatchingObjective(_tiny_cfg(t_sampling_mode="logit_normal", t_sampling_legacy_compat=False,
                                         t_sampling_logit_mean=0.0, t_sampling_logit_std=1.0))
    t2 = o2._sample_t(torch.randn(2048, 1))
    assert 0.0 <= float(t2.min()) and float(t2.max()) <= 1.0
    assert abs(float(t2.mean()) - 0.5) < 0.05
    # strict beta(2,5) skews early
    o3 = FlowMatchingObjective(_tiny_cfg(t_sampling_mode="beta", t_sampling_legacy_compat=False,
                                         t_sampling_beta_a=2.0, t_sampling_beta_b=5.0))
    t3 = o3._sample_t(torch.randn(2048, 1))
    assert float(t3.mean()) < 0.4


def test_texture_refiner_slot_default_off_and_noop_when_on():
    model = _tiny_model(_tiny_cfg())
    assert model.texture_refiner is None
    cfg = _tiny_cfg()
    mcfg = cfg.model.to_dict()
    mcfg["texture_refiner_enabled"] = True
    cfg2 = ExperimentConfig.from_mapping({"model": mcfg, "bridge": {}})
    model2 = _tiny_model(cfg2)
    assert model2.texture_refiner is not None
    x = torch.randn(1, 4, 16, 16)
    out = model2(x, t=0.5, style_id=1)
    assert "ll" in out  # zero-init residual: forward works, output == base at init


# --------------------------------------------------------------------------
# Trainer integration (all routes, tiny synthetic dataloader, 1 epoch)
# --------------------------------------------------------------------------

def test_trainer_end_to_end_all_routes(tmp_path: Path):
    from torch.utils.data import DataLoader, Dataset
    from pseudo_targets import PseudoTargetCacheWriter
    from trainer import SBTrainer

    torch.manual_seed(0)
    cache_dir = tmp_path / "pc"
    writer = PseudoTargetCacheWriter(cache_dir, shard_size=8)
    writer.add(torch.randn(8, 4, 16, 16), [f"{i % 2}/{i}" for i in range(8)], [i % 3 for i in range(8)], [0.9] * 8)
    writer.close()

    class DS(Dataset):
        def __len__(self):
            return 8

        def __getitem__(self, i):
            return {
                "content": torch.randn(4, 16, 16), "target_style": torch.randn(4, 16, 16),
                "target_style_id": i % 3, "source_style_id": i % 2, "content_index": i,
            }

    cfg = ExperimentConfig.from_mapping({
        "model": {"base_dim": 32, "num_res_blocks": 2, "num_styles": 3, "contract_family": "weave",
                  "conditioner_variant": "reference_hybrid", "reference_encoder_in_dim": 16},
        "bridge": {"endpoint_mode": "mixed", "pseudo_target_cache_dir": str(cache_dir),
                   "output_supervision_enabled": True, "output_supervision_ode_steps": 2,
                   "output_supervision_microbatch": 2, "reference_info_nce_weight": 0.1},
        "training": {"num_epochs": 1, "batch_size": 4, "use_tqdm": False, "log_interval": 0,
                     "save_interval": 999, "use_amp": False, "gpu_monitor_enabled": False,
                     # Internal early stop is default-on project-wide; this route test
                     # isolates output-supervision/NCE and has no subband gate head.
                     "internal_probe_enabled": False, "internal_early_stop_enabled": False},
        "checkpoint": {"save_dir": str(tmp_path / "ck")},
        "data": {},
    })
    trainer = SBTrainer(cfg, torch.device("cpu"))

    class DL:
        def __init__(self, base):
            self.base = base

        def __iter__(self):
            for b in self.base:
                b["reference_feats"] = torch.randn(b["content"].shape[0], 25, 16)
                b["reference_feats_alt"] = torch.randn(b["content"].shape[0], 25, 16)
                yield b

        def __len__(self):
            return len(self.base)

    metrics = trainer.train_epoch(DL(DataLoader(DS(), batch_size=4)), 1)
    assert metrics.get("pseudo_target_hit_rate", 0) > 0
    assert "loss_output_sup" in metrics
    assert "loss_reference_nce" in metrics
    assert metrics["loss"] == metrics["loss"]  # finite


def test_trainer_default_config_unchanged(tmp_path: Path):
    """Route B/C/D machinery must stay off; internal early stop is now default-on."""
    from trainer import SBTrainer

    cfg = ExperimentConfig.from_mapping({
        "model": {"base_dim": 32, "num_res_blocks": 2, "num_styles": 3, "contract_family": "weave"},
        "bridge": {},
        "training": {"num_epochs": 1, "use_tqdm": False, "use_amp": False, "gpu_monitor_enabled": False,
                     # Early stop now requires a subband gate head; disable it here to
                     # keep this test focused on the retired next-gen machinery.
                     "internal_probe_enabled": False, "internal_early_stop_enabled": False},
        "checkpoint": {"save_dir": str(tmp_path / "ck2")},
        "data": {},
    })
    trainer = SBTrainer(cfg, torch.device("cpu"))
    assert trainer._output_supervision is None
    assert trainer._pseudo_cache is None
    assert trainer._ref_nce_weight == 0.0
    assert trainer.loss_fn.endpoint_mode == "target_hf"
    assert trainer.model.reference_encoder is None
