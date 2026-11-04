"""噪声日程、训练目标、DDIM 采样与 SDEdit。

标准线性 beta 日程 + 均匀时间步；预测目标支持 eps 与 v。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def _idx(t):
    """时间步索引统一转 long，允许调用方传 float 时间给网络、传同一批给日程。"""
    if isinstance(t, torch.Tensor) and t.dtype.is_floating_point:
        return t.long()
    return t


@dataclass
class Schedule:
    betas: torch.Tensor       # [T]
    alphas: torch.Tensor      # [T]
    alpha_bar: torch.Tensor   # [T]
    timesteps: int

    @classmethod
    def linear(cls, timesteps: int = 1000, beta_start: float = 1e-4,
               beta_end: float = 2e-2, device: str = "cpu") -> "Schedule":
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64, device=device)
        alphas = 1.0 - betas
        ab = torch.cumprod(alphas, dim=0)
        return cls(betas.float(), alphas.float(), ab.float(), timesteps)

    def to(self, device) -> "Schedule":
        return Schedule(self.betas.to(device), self.alphas.to(device),
                        self.alpha_bar.to(device), self.timesteps)

    def q_sample(self, z0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        ab = self.alpha_bar[_idx(t)].reshape(-1, 1, 1, 1)
        return ab.sqrt() * z0 + (1.0 - ab).sqrt() * noise

    def alpha_sigma(self, t: torch.Tensor | int) -> tuple[torch.Tensor, torch.Tensor]:
        ab = self.alpha_bar[_idx(t)].reshape(-1, 1, 1, 1)
        return ab.sqrt(), (1.0 - ab).sqrt()

    def snr(self, t: torch.Tensor | int) -> torch.Tensor:
        ab = self.alpha_bar[_idx(t)]
        return ab / (1.0 - ab)


def v_target(z0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor,
             schedule: Schedule) -> torch.Tensor:
    """v* = alpha_t * eps - sigma_t * z0（与 eps 共用同一份 t/noise 采样）。"""
    alpha, sigma = schedule.alpha_sigma(t)
    return alpha * noise - sigma * z0


def pred_to_x0_eps(pred: torch.Tensor, z: torch.Tensor, t: torch.Tensor,
                   schedule: Schedule, prediction_type: str = "eps") -> tuple[torch.Tensor, torch.Tensor]:
    """把网络输出统一换算成 (x0, eps)；v 与 eps 用同一套 alpha/sigma。"""
    alpha, sigma = schedule.alpha_sigma(t)
    if prediction_type == "v":
        return alpha * z - sigma * pred, sigma * z + alpha * pred
    eps = pred
    return (z - sigma * eps) / alpha, eps


def training_loss(
    model,
    z0: torch.Tensor,
    style_ids: torch.Tensor,
    schedule: Schedule,
    *,
    cfg_dropout: float = 0.0,
    prediction_type: str = "eps",
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, dict]:
    """``L = ||f_theta(z_t, t, s) - target||^2``；target 由 prediction_type 决定。

    CFG dropout 用 style_id=-1 表示空条件。eps / v 共用同一次 t 与 noise 采样。
    """
    b = z0.shape[0]
    device = z0.device
    t = torch.randint(0, schedule.timesteps, (b,), device=device, generator=generator)
    noise = torch.randn(z0.shape, device=device, dtype=z0.dtype, generator=generator)
    zt = schedule.q_sample(z0, t, noise)
    cond = style_ids.clone()
    if cfg_dropout > 0.0:
        drop = torch.rand(b, device=device, generator=generator) < cfg_dropout
        cond = torch.where(drop, torch.full_like(cond, -1), cond)
    pred = model(zt, t.float(), cond)
    if prediction_type == "v":
        target = v_target(z0, noise, t, schedule)
    else:
        target = noise
    loss = ((pred - target) ** 2).mean()
    with torch.no_grad():
        stats = {
            "loss": float(loss.detach()),
            "t_mean": float(t.float().mean()),
            "null_frac": float((cond < 0).float().mean()),
            "pred_std": float(pred.detach().std()),
            "target_std": float(target.detach().std()),
            "base": 0.0,
        }
    return loss, stats


def ddim_timesteps(t_start: int, num_steps: int, timesteps: int) -> list[int]:
    """从 ``t_start`` 降到 0 的等距时间步（降序），长度 ``num_steps``。"""
    seq = torch.linspace(0, t_start, num_steps + 1).round().long().tolist()
    return sorted(set(seq[::-1]), reverse=True)


@torch.no_grad()
def ddim_sample(
    model,
    z: torch.Tensor,
    style_ids: torch.Tensor,
    schedule: Schedule,
    *,
    t_start: int,
    num_steps: int = 20,
    cfg_scale: float = 1.0,
    prediction_type: str = "eps",
    guidance_fn=None,
    guidance_scale: float = 0.0,
) -> torch.Tensor:
    """DDIM(eta=0) 去噪；``cfg_scale>1`` 时每步两次前向（NFE 翻倍）。

    ``guidance_fn(z, t_batch) -> dlog p/dz`` 提供分类器引导；梯度按样本 L2 归一化后
    以 ``guidance_scale * sigma_t`` 限幅，避免低噪声区放大成纹理伪影。
    """
    steps = ddim_timesteps(t_start, num_steps, schedule.timesteps)
    for i, t_cur in enumerate(steps):
        t_next = steps[i + 1] if i + 1 < len(steps) else 0
        t_batch = torch.full((z.shape[0],), float(t_cur), device=z.device)
        pred = model(z, t_batch, style_ids)
        if cfg_scale != 1.0:
            pred_uncond = model(z, t_batch, torch.full_like(style_ids, -1))
            pred = pred_uncond + cfg_scale * (pred - pred_uncond)
        x0_pred, eps = pred_to_x0_eps(pred, z, t_cur, schedule, prediction_type)
        alpha, sigma = schedule.alpha_sigma(t_cur)
        if guidance_fn is not None and guidance_scale != 0.0:
            grad = guidance_fn(z, t_batch)
            denom = grad.flatten(1).norm(dim=1).clamp_min(1e-8).reshape(-1, 1, 1, 1)
            eps = eps - guidance_scale * sigma * (grad / denom)
            x0_pred = (z - sigma * eps) / alpha
        ab_next = schedule.alpha_bar[t_next] if t_next > 0 else torch.tensor(1.0, device=z.device)
        z = ab_next.sqrt() * x0_pred + (1 - ab_next).sqrt() * eps
        if t_next == 0:
            return x0_pred
    return z


@torch.no_grad()
def sdedit(
    model,
    x0: torch.Tensor,
    style_ids: torch.Tensor,
    schedule: Schedule,
    *,
    strength: float,
    num_steps: int = 20,
    cfg_scale: float = 1.0,
    prediction_type: str = "eps",
    guidance_fn=None,
    guidance_scale: float = 0.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """把内容 latent 加噪到 ``strength`` 对应水平，再用目标 Style-ID 去噪。"""
    t_start = max(1, min(schedule.timesteps - 1, int(round(strength * (schedule.timesteps - 1)))))
    noise = torch.randn(x0.shape, device=x0.device, dtype=x0.dtype, generator=generator)
    z = schedule.q_sample(x0, torch.full((x0.shape[0],), t_start, device=x0.device), noise)
    return ddim_sample(model, z, style_ids, schedule, t_start=t_start,
                       num_steps=num_steps, cfg_scale=cfg_scale,
                       prediction_type=prediction_type,
                       guidance_fn=guidance_fn, guidance_scale=guidance_scale)


__all__ = ["Schedule", "training_loss", "v_target", "pred_to_x0_eps",
           "ddim_timesteps", "ddim_sample", "sdedit"]
