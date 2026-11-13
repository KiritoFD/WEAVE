"""V3-SWD 目标：局部 SWD + 整图 MMD + 内容结构约束 + Identity。

L = L_local-SWD + beta*L_global-MMD + lambda_c*[L_structure - eps]_+^2 + lambda_i*L_id

* 真实侧特征离线缓存（``runs/v2/swd/library_5styles_512.pt``），训练时不重复计算；
* 生成侧特征在 decode 后的像素图上现算，梯度可回传到 latent；
* 没有 FM / GAN / PatchNCE / 随机目标像素回归。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .config import V2Config
from .distances import content_structure_loss, mmd2_rbf, swd2
from .features import (
    FeatureStats,
    encode_style,
    global_descriptor,
    local_descriptors,
    normalize_global,
    normalize_local,
)
from .model import V2Generator


def load_library(path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("schema") != "swd-style-library-v1":
        raise ValueError(f"bad library schema: {payload.get('schema')!r}")
    return payload


def decode_latents(vae, z: torch.Tensor, *, scale: float | None = None) -> torch.Tensor:
    """可微 decode：latent -> [0,1] RGB。VAE 参数需 requires_grad=False。"""
    s = float(vae.config.scaling_factor) if scale is None else float(scale)
    img = vae.decode((z / s).float()).sample
    return (img + 1.0) * 0.5


@dataclass
class SWDRow:
    loss: torch.Tensor
    swd: torch.Tensor
    mmd: torch.Tensor
    structure: torch.Tensor
    hinge: torch.Tensor
    identity: torch.Tensor


class SWDObjective:
    def __init__(
        self,
        cfg: V2Config,
        library: dict[str, Any],
        *,
        device: torch.device,
        seed: int = 0,
    ) -> None:
        self.cfg = cfg
        self.device = device
        self.stats: FeatureStats = FeatureStats(
            local_mean=library["local_mean"].float().to(device),
            local_std=library["local_std"].float().to(device),
            local_whiten=library["local_whiten"].float().to(device),
            global_mean=library["global_mean"].float().to(device),
            global_std=library["global_std"].float().to(device),
            global_whiten=library["global_whiten"].float().to(device),
        )
        # 每个风格：list[3] of [N_images, PATCHES_PER_LEVEL, 14]，所有张量常驻显存
        self.local = [
            [t.to(device=device, dtype=torch.float32) for t in style_locals]
            for style_locals in library["local"]
        ]
        self.global_feat = [t.to(device=device, dtype=torch.float32) for t in library["global"]]
        self.gen = torch.Generator(device="cpu").manual_seed(int(seed))
        self.encoder = None
        self.encoder_size = int(library.get("encoder_size", 128))
        ckpt_path = str(library.get("encoder_ckpt") or "")
        if ckpt_path and Path(ckpt_path).exists():
            from .features import encode_style
            from .style_encoder import StyleEncoder

            payload = torch.load(Path(ckpt_path), map_location="cpu", weights_only=False)
            enc = StyleEncoder(int(payload["num_styles"]), int(payload["embed_dim"]))
            enc.load_state_dict(payload["state_dict"])
            self.encoder = enc.to(device).eval().requires_grad_(False)
        elif ckpt_path:
            raise FileNotFoundError(f"library 记录的 encoder ckpt 不存在: {ckpt_path}")

    def global_features(self, x: torch.Tensor) -> torch.Tensor:
        q = global_descriptor(x)
        if self.encoder is not None:
            q = torch.cat([q, encode_style(self.encoder, x, size=self.encoder_size)], dim=1)
        return q

    def _gen_index(self, n: int, total: int) -> torch.Tensor:
        if total <= n:
            return torch.arange(total)
        perm = torch.randperm(total, generator=self.gen)[:n]
        return perm

    def real_samples(self, style_id: int, *, n_images: int) -> tuple[list[torch.Tensor], torch.Tensor]:
        """抽 n_images 张真实同风格图的缓存特征 -> ([n*P,14] per level, [n,86])."""
        style_locals = self.local[int(style_id)]
        total = style_locals[0].shape[0]
        idx = self._gen_index(n_images, total).to(self.device)
        locs = [t.index_select(0, idx).reshape(-1, t.shape[-1]) for t in style_locals]
        glob = self.global_feat[int(style_id)].index_select(0, idx.to(self.global_feat[0].device))
        return locs, glob

    def compute(
        self,
        generator: V2Generator,
        vae,
        content: torch.Tensor,
        *,
        target_style: int,
        identity_latents: torch.Tensor,
        patch_samples: int = 128,
        real_images: int = 16,
    ) -> SWDRow:
        cfg = self.cfg
        dev = content.device
        tgt = torch.full((content.shape[0],), int(target_style), dtype=torch.long, device=dev)
        z_hat = generator(content, style_id=tgt)["z_hat"]

        with torch.no_grad():
            x_src = decode_latents(vae, content)
        x_hat = decode_latents(vae, z_hat)

        structure = content_structure_loss(x_hat, x_src, alpha=float(cfg.content_alpha))
        hinge = torch.relu(structure - float(cfg.content_eps)) ** 2

        descs = normalize_local(local_descriptors(x_hat), self.stats)
        flat = []
        for d in descs:
            b, c, h, w = d.shape
            d = d.reshape(b, c, h * w).transpose(1, 2).reshape(b * h * w, c)
            if d.shape[0] > b * patch_samples:
                pick = torch.randint(0, h * w, (patch_samples,), device=d.device)
                idx = (torch.arange(b, device=d.device)[:, None] * (h * w) + pick[None, :]).reshape(-1)
                d = d.index_select(0, idx)
            flat.append(d)
        real_locals, real_glob = self.real_samples(target_style, n_images=real_images)
        swd = x_hat.new_zeros(())
        for k in range(len(flat)):
            swd = swd + swd2(flat[k], real_locals[k])

        q_gen = normalize_global(self.global_features(x_hat), self.stats)
        mmd = mmd2_rbf(q_gen, real_glob, gen=self.gen) if cfg.w_mmd > 0 else x_hat.new_zeros(())

        z_id = generator(identity_latents, style_id=torch.full(
            (identity_latents.shape[0],), int(target_style), dtype=torch.long, device=dev
        ))["z_hat"]
        identity = (z_id - identity_latents).abs().mean() / identity_latents.abs().mean().clamp_min(1e-6)

        total = (
            float(cfg.w_swd) * swd
            + float(cfg.w_mmd) * mmd
            + float(cfg.w_content) * hinge
            + float(cfg.w_identity) * identity
        )
        return SWDRow(loss=total, swd=swd, mmd=mmd, structure=structure, hinge=hinge, identity=identity)


__all__ = ["SWDObjective", "SWDRow", "load_library", "decode_latents"]
