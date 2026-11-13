"""v2 主干（V2/V3/V4）的评测适配器。

``tools/eval_fast.py`` 走的是 legacy ``LGTInference`` 接口：
``inversion(x0)`` + ``generation_with_target_latent(x0, tgt_ids, target_style_latent=...)``。
新主干按离散 style id 条件化，不吃 target style latent，因此这里做一个最小适配：

* V2（velocity 模式）：从 content latent 出发做 n 步 Euler 子带积分；
* V3/V4（generator 模式）：单步前向直接输出 ``z_hat``。

加载一律 ``strict=True``；键不匹配直接 SystemExit，杜绝静默错评。
"""
from __future__ import annotations

import torch
from tools.legacy_root.wavelet import dwt2_haar, idwt2_haar

from .config import V2Config
from .model import V2Backbone, V2Generator

_GENERATOR_VARIANTS = {"V3", "V4", "V3S"}


class _ScaleShim:
    """只暴露 eval_fast 需要的 ``latent_scale_factor``。"""

    def __init__(self, scale: float) -> None:
        self.latent_scale_factor = float(scale)


class V2Inference:
    def __init__(
        self,
        payload: dict,
        *,
        device: str = "cuda",
        num_steps: int = 8,
        latent_channels: int = 4,
        num_styles: int = 5,
        latent_scale_factor: float = 0.18215,
    ) -> None:
        variant = str(payload.get("variant", ""))
        if variant not in {"V2"} | _GENERATOR_VARIANTS:
            raise SystemExit(f"V2Inference 不支持 variant={variant!r}")
        self.variant = variant
        self.v2cfg = V2Config.from_mapping(payload.get("v2_config"))
        if variant == "V2":
            net = V2Backbone(self.v2cfg, latent_channels=latent_channels,
                             num_styles=num_styles, mode="velocity")
            self.kind = "velocity"
        else:
            net = V2Generator(self.v2cfg, latent_channels=latent_channels, num_styles=num_styles)
            self.kind = "generator"

        state = payload.get("model")
        if not isinstance(state, dict):
            raise SystemExit("ckpt 缺少裸键 'model'，拒绝评测")
        try:
            net.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            model_keys = set(net.state_dict().keys())
            ckpt_keys = set(state.keys())
            missing = sorted(model_keys - ckpt_keys)
            unexpected = sorted(ckpt_keys - model_keys)
            raise SystemExit(
                f"键不匹配，拒绝评测: loaded_params={len(model_keys) - len(missing)}/{len(model_keys)} "
                f"(missing={len(missing)}, unexpected={len(unexpected)}); "
                f"first_missing={missing[:3]}; first_unexpected={unexpected[:3]}"
            ) from exc
        total = len(net.state_dict())
        print(f"[v2-inference] {variant} strict load OK: loaded_params={total}/{total} (100%)", flush=True)

        self.net = net.to(device).eval()
        self.model = _ScaleShim(latent_scale_factor)
        self.num_steps = max(1, int(num_steps))

    @torch.no_grad()
    def inversion(self, x: torch.Tensor) -> torch.Tensor:
        return x

    @torch.no_grad()
    def generation_with_target_latent(
        self,
        x0: torch.Tensor,
        tgt_ids: torch.Tensor,
        target_style_latent: dict | None = None,
    ) -> torch.Tensor:
        del target_style_latent  # 新主干只用离散 style id
        x = x0
        if self.kind == "generator":
            return self.net(x, style_id=tgt_ids)["z_hat"]

        b = x.shape[0]
        dt = 1.0 / float(self.num_steps)
        for step in range(self.num_steps):
            t = torch.full((b,), (step + 0.5) * dt, device=x.device, dtype=x.dtype)
            v = self.net(x, t=t, style_id=tgt_ids)
            ll, lh, hl, hh = dwt2_haar(x.float())
            ll = ll + dt * v["ll"].float()
            lh = lh + dt * v["lh"].float()
            hl = hl + dt * v["hl"].float()
            hh = hh + dt * v["hh"].float()
            x = idwt2_haar(ll, lh, hl, hh).to(dtype=x0.dtype)
        return x


def _cfg_model_field(config, name: str, default):
    """``eval_fast`` 传进来的是 dict，训练侧是 ExperimentConfig；两种都吃。"""
    model = config.get("model", {}) if isinstance(config, dict) else getattr(config, "model", None)
    if isinstance(model, dict):
        return model.get(name, default)
    if model is None:
        return default
    return getattr(model, name, default)


def build_v2_inference(payload: dict, config, *, device: str, num_steps: int = 8) -> V2Inference:
    return V2Inference(
        payload,
        device=device,
        num_steps=num_steps,
        latent_channels=int(_cfg_model_field(config, "latent_channels", 4)),
        num_styles=int(_cfg_model_field(config, "num_styles", 5)),
        latent_scale_factor=float(_cfg_model_field(config, "latent_scale_factor", 0.18215)),
    )


__all__ = ["V2Inference", "build_v2_inference"]
