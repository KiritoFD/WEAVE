"""噪声日程、训练目标、DDIM 采样与 SDEdit。

只实现方案要求的最小集合：标准线性 beta 日程 + 均匀时间步；预测 eps。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch


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
        ab = self.alpha_bar[t].view(-1, 1, 1, 1)
        return ab.sqrt() * z0 + (1.0 - ab).sqrt() * noise


def training_loss(
    model,
    z0: torch.Tensor,
    style_ids: torch.Tensor,
    schedule: Schedule,
    *,
    cfg_dropout: float = 0.0,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, dict]:
    """``L = ||eps_theta(z_t, t, s) - eps||^2``；CFG dropout 用 style_id=-1 表示空条件。"""
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
    loss = ((pred - noise) ** 2).mean()
    with torch.no_grad():
        base = ((noise - noise) ** 2).mean()  # 0，占位保证返回结构稳定
        stats = {
            "loss": float(loss.detach()),
            "t_mean": float(t.float().mean()),
            "null_frac": float((cond < 0).float().mean()),
            "pred_std": float(pred.detach().std()),
            "noise_std": float(noise.std()),
            "base": float(base),
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
) -> torch.Tensor:
    """DDIM(eta=0) 去噪；``cfg_scale>1`` 时每步两次前向（NFE 翻倍）。"""
    steps = ddim_timesteps(t_start, num_steps, schedule.timesteps)
    for i, t_cur in enumerate(steps):
        t_next = steps[i + 1] if i + 1 < len(steps) else 0
        t_batch = torch.full((z.shape[0],), float(t_cur), device=z.device)
        eps = model(z, t_batch, style_ids)
        if cfg_scale != 1.0:
            eps_uncond = model(z, t_batch, torch.full_like(style_ids, -1))
            eps = eps_uncond + cfg_scale * (eps - eps_uncond)
        ab_cur = schedule.alpha_bar[t_cur]
        ab_next = schedule.alpha_bar[t_next] if t_next > 0 else torch.tensor(1.0, device=z.device)
        z0_pred = (z - (1 - ab_cur).sqrt() * eps) / ab_cur.sqrt()
        z = ab_next.sqrt() * z0_pred + (1 - ab_next).sqrt() * eps
        if t_next == 0:
            return z0_pred
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
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """把内容 latent 加噪到 ``strength`` 对应水平，再用目标 Style-ID 去噪。"""
    t_start = max(1, min(schedule.timesteps - 1, int(round(strength * (schedule.timesteps - 1)))))
    noise = torch.randn(x0.shape, device=x0.device, dtype=x0.dtype, generator=generator)
    z = schedule.q_sample(x0, torch.full((x0.shape[0],), t_start, device=x0.device), noise)
    return ddim_sample(model, z, style_ids, schedule, t_start=t_start,
                       num_steps=num_steps, cfg_scale=cfg_scale)


__all__ = ["Schedule", "training_loss", "ddim_timesteps", "ddim_sample", "sdedit"]
