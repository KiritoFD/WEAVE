"""WEAVE-v4 评测适配器：SDEdit 编辑，接 ``tools/eval_fast.py`` 的 legacy 接口。

``inversion(x0)`` 为恒等；``generation_with_target_latent(x0, tgt_ids)`` 做
「加噪到 strength -> 用目标 Style-ID 做 DDIM 去噪」。

运行时参数全部走环境变量，保证评测协议脚本零改动：
  V4_NUM_STEPS     DDIM 步数（默认 20）
  V4_EDIT_STRENGTH SDEdit 强度（默认 0.45）
  V4_CFG           CFG 系数（默认 1.0；>1 时 NFE 翻倍）
  V4_SEED          噪声种子（默认 0）
  V4_FORCE_NULL    1 = 强制空风格条件（关闭 Style-ID 的对照）
"""
from __future__ import annotations

import os

import torch

from .config import V4Config
from .diffusion import Schedule, ddim_sample, sdedit
from .model import V4UNet


class _ScaleShim:
    def __init__(self, scale: float) -> None:
        self.latent_scale_factor = float(scale)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "")
    return int(raw) if raw.strip() else default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "")
    return float(raw) if raw.strip() else default


class V4Inference:
    def __init__(self, payload: dict, *, device: str = "cuda", num_steps_fallback: int = 20) -> None:
        if str(payload.get("variant", "")) != "V4DIFF":
            raise SystemExit(f"V4Inference 只接受 variant=V4DIFF，收到 {payload.get('variant')!r}")
        cfg = V4Config.from_mapping(payload.get("v4_config"))
        state = payload.get("model")
        if not isinstance(state, dict):
            raise SystemExit("ckpt 缺少裸键 'model'，拒绝评测")
        net = V4UNet(cfg)
        try:
            net.load_state_dict({k: v.float() for k, v in state.items()}, strict=True)
        except RuntimeError as exc:
            model_keys = set(net.state_dict().keys())
            ckpt_keys = set(state.keys())
            missing = sorted(model_keys - ckpt_keys)
            unexpected = sorted(ckpt_keys - model_keys)
            raise SystemExit(
                f"键不匹配，拒绝评测: {len(model_keys) - len(missing)}/{len(model_keys)} "
                f"missing={len(missing)} unexpected={len(unexpected)} first_missing={missing[:3]} "
                f"first_unexpected={unexpected[:3]}"
            ) from exc
        total = len(net.state_dict())
        print(f"[v4-inference] V4DIFF strict load OK: loaded_params={total}/{total} "
              f"(100%) step={payload.get('step')}", flush=True)

        self.net = net.to(device).eval()
        self.prediction_type = cfg.prediction_type
        self.model = _ScaleShim(payload.get("latent_scale_factor", cfg.latent_scale_factor))
        self.schedule = Schedule.linear(cfg.timesteps, cfg.beta_start, cfg.beta_end, device=device)
        self.device = device
        self.num_steps = _env_int("V4_NUM_STEPS", num_steps_fallback)
        self.strength = _env_float("V4_EDIT_STRENGTH", 0.45)
        self.cfg_scale = _env_float("V4_CFG", 1.0)
        self.seed = _env_int("V4_SEED", 0)
        self.force_null = _env_int("V4_FORCE_NULL", 0) == 1
        self.identity = _env_int("V4_IDENTITY", 0) == 1
        self._calls = 0
        print(f"[v4-inference] steps={self.num_steps} strength={self.strength} "
              f"cfg={self.cfg_scale} seed={self.seed} force_null={self.force_null} "
              f"prediction_type={self.prediction_type} "
              f"NFE/call={self.num_steps * (2 if self.cfg_scale != 1.0 else 1)}", flush=True)

    @torch.no_grad()
    def inversion(self, x: torch.Tensor) -> torch.Tensor:
        return x

    @torch.no_grad()
    def _noise_gen(self) -> torch.Generator:
        gen = torch.Generator(device=self.device)
        gen.manual_seed(self.seed + 1000 * self._calls)
        self._calls += 1
        return gen

    @torch.no_grad()
    def generation_with_target_latent(self, x0: torch.Tensor, tgt_ids: torch.Tensor,
                                      target_style_latent: dict | None = None) -> torch.Tensor:
        del target_style_latent  # V4 只用离散 style id
        sid = tgt_ids.to(self.device).long()
        if self.identity:
            return x0.to(self.device).float()
        if self.force_null:
            sid = torch.full_like(sid, -1)
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=self.device == "cuda"):
            out = sdedit(self.net, x0.to(self.device).float(), sid, self.schedule,
                         strength=self.strength, num_steps=self.num_steps,
                         cfg_scale=self.cfg_scale,
                         prediction_type=self.prediction_type,
                         generator=self._noise_gen())
        return out.float()

    @torch.no_grad()
    def sample_styles(self, style_ids: torch.Tensor, latent_shape: tuple[int, int, int, int], *,
                      num_steps: int | None = None, cfg_scale: float | None = None,
                      seed: int | None = None) -> torch.Tensor:
        """纯风格生成（无内容图）：从纯噪声按 Style-ID 去噪。"""
        steps = self.num_steps if num_steps is None else num_steps
        cfg = self.cfg_scale if cfg_scale is None else cfg_scale
        gen = torch.Generator(device=self.device).manual_seed(self.seed if seed is None else seed)
        z = torch.randn(latent_shape, device=self.device, generator=gen)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device == "cuda"):
            out = ddim_sample(self.net, z, style_ids.long().to(self.device), self.schedule,
                              t_start=self.schedule.timesteps - 1, num_steps=steps,
                              cfg_scale=cfg, prediction_type=self.prediction_type)
        return out.float()


def build_v4_inference(payload: dict, config=None, *, device: str = "cuda", num_steps: int = 8,
                       seed: int = 0) -> V4Inference:
    # 评测脚本固定传 8（旧 ODE 口径）；V4 默认 20 步，可用 V4_NUM_STEPS 显式覆盖。
    del num_steps
    if seed >= 0:
        os.environ.setdefault("V4_SEED", str(int(seed)))
    return V4Inference(payload, device=device, num_steps_fallback=20)


__all__ = ["V4Inference", "build_v4_inference"]
