"""WEAVE-v2 配置。

配置来源两种：
1. 标准 WEAVE config 的顶层 ``v2`` 段（``ExperimentConfig.extra_sections["v2"]``）；
2. 直接传 dict。

所有开关默认关闭；``V2Config.from_experiment()`` 在缺失 ``v2`` 段时返回
``enabled=False`` 的默认对象，因此旧配置不会走到 v2 代码。
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Mapping, Sequence


def _tuple_of_int(value: Any, default: tuple[int, ...]) -> tuple[int, ...]:
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)
    raise ValueError(f"expected a sequence of ints, got {type(value).__name__}")


@dataclass
class V2Config:
    # ---- 总开关（默认关）----
    enabled: bool = False
    variant: str = "V3"  # V1/V2/V3/V4/V5，仅用于训练入口校验

    # ---- 主干 ----
    # 64 -> 32 -> 16 -> 8 四级；默认 7.3M（5–8M 预算），10M 档见 smoke 报告
    channels: tuple[int, ...] = (80, 160, 208, 208)
    blocks_per_stage: tuple[int, ...] = (1, 1, 2, 1)
    decoder_blocks_per_stage: tuple[int, ...] = (1, 1, 1, 1)
    style_embed_dim: int = 128
    style_tokens: int = 8  # 0 表示彻底关闭可学习 style token 增强
    style_token_heads: int = 4
    attn_heads: int = 4
    norm_groups: int = 8
    adaln_enabled: bool = True
    adaln_gate_init: float = 0.0  # 0 = 初始恒等（门控关）
    residual_init_std: float = 0.02
    residual_ll_scale: float = 0.10
    residual_hf_scale: float = 0.35
    stats_dim: int = 32

    # ---- 监督目标 ----
    w_adv: float = 1.0
    w_patchnce: float = 1.0
    w_geometry: float = 1.0
    w_identity: float = 1.0
    w_recon: float = 0.0  # V4 才打开
    nce_temperature: float = 0.07
    nce_num_patches: int = 128
    nce_hidden: int = 64
    disc_base_channels: int = 64
    disc_num_scales: int = 3
    disc_layers: int = 3
    disc_style_dim: int = 128
    adv_mode: str = "hinge"
    r1_weight: float = 0.0
    geometry_scales: int = 3
    geometry_patch: int = 3
    geometry_patch_stride: int = 2
    geometry_max_patches: int = 256

    # ---- 第二阶段蒸馏（默认关，绝不自动启用）----
    distill_enabled: bool = False
    distill_steps: int = 2
    distill_weight: float = 1.0
    legacy_ode_steps: int = 2  # V1 用旧主干时的可求导 Euler 步数

    # ---- Style-ID-only 条件 ----
    style_stats_path: str = ""
    style_stats_gate_init: float = 0.0  # 0 = 统计条件初始不生效

    # ---- V3-SWD（分布匹配，无 GAN / 无 FM / 无 PatchNCE）----
    residual_head_zero_init: bool = False
    swd_library: str = ""
    latent_cache: str = "data/train/.latent_cache/packed/packed"
    w_swd: float = 1.0
    w_mmd: float = 0.5
    w_content: float = 1.0
    content_eps: float = 0.05
    content_alpha: float = 1.0
    w_identity: float = 1.0
    swd_patch_samples: int = 128
    swd_real_images: int = 16
    diag_every: int = 50
    save_every: int = 500

    # ---- 训练入口 ----
    steps: int = 0
    lr: float = 1e-4
    lr_disc: float = 2e-4
    batch_size: int = 8
    accumulation_steps: int = 1  # microbatch 数；有效 batch = batch_size * accumulation_steps
    seed: int = 42
    synthetic: bool = False
    out_dir: str = "runs/v2"

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "V2Config":
        data = dict(payload or {})
        known = {item.name for item in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown v2 config keys: {sorted(unknown)}")
        kwargs = {key: data[key] for key in known if key in data}
        cfg = cls(**kwargs)
        for name in ("channels", "blocks_per_stage", "decoder_blocks_per_stage"):
            setattr(cfg, name, _tuple_of_int(getattr(cfg, name), getattr(cls(), name)))
        cfg.validate()
        return cfg

    @classmethod
    def from_experiment(cls, config: Any) -> "V2Config":
        """从 ``ExperimentConfig`` / dict 读取顶层 ``v2`` 段。"""
        payload: Mapping[str, Any] = {}
        if config is None:
            return cls()
        if isinstance(config, Mapping):
            payload = config.get("v2", {}) or {}
        else:
            extra = getattr(config, "extra_sections", {}) or {}
            if isinstance(extra, Mapping) and isinstance(extra.get("v2"), Mapping):
                payload = extra["v2"]
            else:
                raw = getattr(config, "v2", None)
                if isinstance(raw, Mapping):
                    payload = raw
        return cls.from_mapping(payload)

    def validate(self) -> None:
        if len(self.channels) != 4:
            raise ValueError(f"v2.channels must have 4 entries, got {self.channels}")
        if len(self.blocks_per_stage) != 4 or len(self.decoder_blocks_per_stage) != 4:
            raise ValueError("v2 blocks_per_stage / decoder_blocks_per_stage must have 4 entries")
        if self.variant not in {"V1", "V2", "V3", "V4", "V5", "V3S"}:
            raise ValueError(f"v2.variant must be V1..V5 or V3S, got {self.variant!r}")
        if self.adv_mode not in {"hinge", "lsgan"}:
            raise ValueError(f"v2.adv_mode must be hinge|lsgan, got {self.adv_mode!r}")
        if self.style_tokens < 0:
            raise ValueError("v2.style_tokens must be >= 0")
        if self.style_tokens > 0 and self.style_tokens < 8:
            # 允许 8-16 的增强档；小于 8 的 token 数没有实际意义
            raise ValueError("v2.style_tokens must be 0 or in [8, 16]")
        if self.style_tokens > 16:
            raise ValueError("v2.style_tokens must be <= 16")
        if int(self.accumulation_steps) < 1:
            raise ValueError("v2.accumulation_steps must be >= 1")
        for name in ("w_adv", "w_patchnce", "w_geometry", "w_identity", "w_recon", "distill_weight"):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"v2.{name} must be >= 0")
        if not (0.0 < self.nce_temperature):
            raise ValueError("v2.nce_temperature must be > 0")
        if self.disc_num_scales < 1:
            raise ValueError("v2.disc_num_scales must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, tuple):
                value = list(value)
            out[item.name] = value
        return out


def expand_variant_defaults(cfg: V2Config) -> V2Config:
    """V4 打开同图重建；V5 的蒸馏必须由显式配置/命令行打开，不自动启用。"""
    if cfg.variant == "V4" and cfg.w_recon == 0.0:
        cfg.w_recon = 0.5
    return cfg


__all__ = ["V2Config", "expand_variant_defaults"]
