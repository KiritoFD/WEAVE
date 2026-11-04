"""WEAVE-v4 配置。纯 dataclass，字段即全部可调项；未知键直接报错。"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, asdict
from typing import Any, Mapping


def _as_int_tuple(value: Any) -> tuple[int, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)
    raise ValueError(f"expected a sequence of ints, got {type(value).__name__}")


@dataclass
class V4Config:
    # ---- 路径 ----
    data_root: str = "data/train"
    packed_dir: str = "data/train/.latent_cache/packed/packed"
    out_dir: str = "runs/v4"
    run_name: str = "v4_prior"

    # ---- 数据 ----
    val_per_style: int = 100
    image_size: int = 512
    latent_channels: int = 4
    latent_scale_factor: float = 0.18215
    num_styles: int = 5

    # ---- 主干（4 尺度 ResUNet，FiLM 条件）----
    channels: tuple[int, ...] = (64, 128, 192, 256)
    blocks_per_stage: tuple[int, ...] = (1, 1, 1, 1)
    mid_blocks: int = 1
    attn_heads: int = 4
    norm_groups: int = 8
    time_embed_dim: int = 256
    style_embed_dim: int = 64
    cond_dim: int = 256
    dropout: float = 0.0
    film_init_std: float = 0.0   # >0 时 FiLM 投影非零初始化（臂 B：条件通路开机即激活）
    # 条件注入方式："sum" = 时间+风格相加后共用一套 FiLM（A/B 臂）；
    # "split" = 时间走原 FiLM，风格走逐层独立 FiLM（E2 臂，参数几乎持平）。
    cond_mode: str = "sum"
    style_film_dim: int = 64      # split 模式下风格条件维度
    style_film_init_std: float = 0.05

    # ---- 扩散 ----
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    beta_schedule: str = "linear"
    cfg_dropout: float = 0.10
    prediction_type: str = "eps"  # "eps" | "v"

    # ---- 优化 ----
    # lr 按有效 batch 缩放：原 1e-4 @ batch 32 -> 1e-4*sqrt(256/32)=2.83e-4 @ batch 256
    lr: float = 2.8e-4
    warmup_steps: int = 500
    min_lr_ratio: float = 0.05
    weight_decay: float = 0.01
    batch_size: int = 256        # 单次前向/反传的 batch（不再做梯度累积）
    grad_accum: int = 1          # 有效 batch = 256
    max_steps: int = 20000
    ema_decay: float = 0.999
    grad_clip: float = 1.0
    use_bf16: bool = True
    seed: int = 42
    milestones: tuple[int, ...] = (1000, 3000, 6000, 10000, 15000, 20000)
    val_every: int = 250
    val_batches: int = 8
    save_best: bool = True

    # ---- 早停 ----
    early_stop_after: int = 4000
    early_stop_patience: int = 6
    early_stop_min_delta: float = 0.005

    # ---- 诊断 ----
    diag_every: int = 500
    diag_samples: int = 8

    def validate(self) -> None:
        if len(self.channels) != 4:
            raise ValueError(f"v4.channels needs 4 entries, got {self.channels}")
        if len(self.blocks_per_stage) != 4:
            raise ValueError("v4.blocks_per_stage needs 4 entries")
        if not 0.0 <= self.cfg_dropout < 1.0:
            raise ValueError("v4.cfg_dropout must be in [0, 1)")
        if self.num_styles < 1:
            raise ValueError("v4.num_styles must be >= 1")
        if self.batch_size < 1 or self.grad_accum < 1:
            raise ValueError("v4.batch_size / grad_accum must be >= 1")
        if self.beta_schedule != "linear":
            raise ValueError("only linear beta schedule is implemented")
        if self.prediction_type not in ("eps", "v"):
            raise ValueError(f"prediction_type must be 'eps' or 'v', got {self.prediction_type!r}")
        if self.cond_mode not in ("sum", "split"):
            raise ValueError(f"cond_mode must be 'sum' or 'split', got {self.cond_mode!r}")
        if not 0.0 < self.ema_decay <= 1.0:
            raise ValueError("v4.ema_decay must be in (0, 1]")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "V4Config":
        data = dict(payload or {})
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown v4 config keys: {sorted(unknown)}")
        cfg = cls(**{k: data[k] for k in known if k in data})
        for name in ("channels", "blocks_per_stage", "milestones"):
            setattr(cfg, name, _as_int_tuple(getattr(cfg, name)))
        cfg.validate()
        return cfg

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        for name in ("channels", "blocks_per_stage", "milestones"):
            out[name] = list(out[name])
        return out


__all__ = ["V4Config"]

