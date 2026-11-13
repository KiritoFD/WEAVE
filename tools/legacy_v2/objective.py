"""WEAVE-v2 分布级监督目标。

``L_G = λ_adv·L_adv + λ_p·L_PatchNCE + λ_g·L_geometry + λ_i·L_identity``

- ``L_adv``：条件多尺度 PatchGAN（真实绘画 patch vs 生成 patch）；
- ``L_PatchNCE``：输入内容与输出在对应位置的特征正对，负样本来自同一张图的
  其他位置（CUT 思路，不做跨图负样本），用于保内容布局；
- ``L_geometry``：多尺度低频 L1 + 局部 patch 自相似矩阵；
- ``L_identity``：输入风格 == 目标风格的行，输出 ≈ 输入；
- ``L_recon``：V4 可选，额外要求 G(真实目标图, 其自身风格) 重建目标图。
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .config import V2Config
from .discriminator import ConditionalPatchDiscriminator, adv_d_loss, adv_g_loss


def _ids(style_id: Any, batch: int, device: torch.device) -> torch.Tensor:
    if style_id is None:
        return torch.zeros(batch, dtype=torch.long, device=device)
    if torch.is_tensor(style_id):
        ids = style_id.to(device=device, dtype=torch.long).view(-1)
        if ids.numel() == 1 and batch > 1:
            ids = ids.expand(batch)
        return ids
    return torch.full((batch,), int(style_id), dtype=torch.long, device=device)


def identity_rows(
    batch: int,
    *,
    target_style_id: Any,
    source_style_id: Any,
    conditioning: dict[str, Any] | None = None,
    device: torch.device,
) -> torch.Tensor:
    """返回 bool 掩码；优先使用 dataset 的 ``identity_mask``。"""
    conditioning = conditioning or {}
    mask = conditioning.get("identity_mask")
    if torch.is_tensor(mask):
        return mask.to(device=device).view(-1).bool()
    if source_style_id is not None and target_style_id is not None:
        t = _ids(target_style_id, batch, device)
        s = _ids(source_style_id, batch, device)
        return t == s
    return torch.zeros(batch, dtype=torch.bool, device=device)


class PatchNCEEncoder(nn.Module):
    """CUT 风格的特征编码器：输入/输出共用权重，取 32×32 与 16×16 两层。"""

    def __init__(self, in_channels: int = 4, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, hidden, 4, stride=2, padding=1),
                    nn.SiLU(),
                    nn.Conv2d(hidden, hidden, 3, padding=1),
                ),
                nn.Sequential(
                    nn.Conv2d(hidden, hidden * 2, 4, stride=2, padding=1),
                    nn.SiLU(),
                    nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1),
                ),
                nn.Sequential(
                    nn.Conv2d(hidden * 2, hidden * 2, 4, stride=2, padding=1),
                    nn.SiLU(),
                    nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1),
                ),
            ]
        )
        self.proj = None

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats: list[torch.Tensor] = []
        h = x.float()
        for idx, layer in enumerate(self.net):
            h = layer(h)
            if idx >= 1:
                feats.append(h)
        return feats


def patch_nce_logits(
    src: torch.Tensor,
    tgt: torch.Tensor,
    sample_positions: torch.Tensor,
    *,
    temperature: float = 0.07,
) -> torch.Tensor:
    """``src``/``tgt``: (B,C,N) 同一张图的对应位置特征。

    返回 (B, P, N)，第 p 个 query 的正样本索引就是 ``sample_positions[p]``；
    负样本来自同一张图的其余 N-1 个位置，不跨 batch。
    """
    if src.dim() != 3 or tgt.dim() != 3:
        raise ValueError("patch_nce_logits expects (B,C,N) tensors")
    src = F.normalize(src.float(), dim=1)
    tgt = F.normalize(tgt.float(), dim=1)
    src_s = src.index_select(2, sample_positions.to(src.device))  # (B,C,P)
    logits = torch.einsum("bcp,bcn->bpn", src_s, tgt) / max(float(temperature), 1e-6)
    return logits


def _sample_positions(n: int, k: int, device: torch.device) -> torch.Tensor:
    if k >= n:
        return torch.arange(n, device=device)
    return torch.randperm(n, device=device)[:k].sort().values


class PatchNCELoss(nn.Module):
    def __init__(self, cfg: V2Config) -> None:
        super().__init__()
        self.temperature = float(cfg.nce_temperature)
        self.num_patches = int(cfg.nce_num_patches)
        self.proj: nn.ModuleDict | None = None

    def build_proj(self, channels: list[int], hidden: int) -> None:
        self.proj = nn.ModuleDict()
        for idx, c in enumerate(channels):
            self.proj[str(idx)] = nn.Sequential(
                nn.Conv2d(c, hidden, 1),
                nn.SiLU(),
                nn.Conv2d(hidden, hidden, 1),
            )

    def forward(self, src_feats: list[torch.Tensor], tgt_feats: list[torch.Tensor]) -> torch.Tensor:
        if not src_feats:
            raise ValueError("PatchNCE requires at least one feature map")
        if self.proj is None:
            self.build_proj([f.shape[1] for f in src_feats], max(8, src_feats[0].shape[1]))
            self.proj.to(src_feats[0].device)
        total = src_feats[0].new_tensor(0.0)
        for idx, (s, t) in enumerate(zip(src_feats, tgt_feats)):
            p_s = self.proj[str(idx)](s)
            p_t = self.proj[str(idx)](t)
            b, c, h, w = p_s.shape
            n = h * w
            pos = _sample_positions(n, self.num_patches, p_s.device)
            logits = patch_nce_logits(
                p_t.reshape(b, c, n),
                p_s.reshape(b, c, n),
                pos,
                temperature=self.temperature,
            )
            labels = torch.arange(pos.numel(), device=pos.device)
            total = total + F.cross_entropy(logits.reshape(-1, n), labels.repeat(b))
        return total / float(len(src_feats))


def geometry_loss(
    content: torch.Tensor,
    fake: torch.Tensor,
    *,
    scales: int = 3,
    patch: int = 3,
    stride: int = 2,
    max_patches: int = 256,
) -> torch.Tensor:
    """多尺度低频一致性 + 局部 patch 自相似结构。"""
    c = content.float()
    f = fake.float()
    lf = c.new_tensor(0.0)
    for s in range(int(scales)):
        k = 2 ** s
        if min(c.shape[-2:]) // k < 4:
            break
        lf = lf + F.l1_loss(F.avg_pool2d(c, k), F.avg_pool2d(f, k))
    lf = lf / max(1, int(scales))

    unfold = lambda z: F.unfold(z, kernel_size=patch, stride=stride)  # (B, C*p*p, P)
    pc = F.normalize(unfold(c), dim=1)
    pf = F.normalize(unfold(f), dim=1)
    p = pc.shape[-1]
    idx = _sample_positions(p, max_patches, pc.device)
    pc = pc.index_select(2, idx)
    pf = pf.index_select(2, idx)
    sim_c = torch.bmm(pc.transpose(1, 2), pc)
    sim_f = torch.bmm(pf.transpose(1, 2), pf)
    selfsim = F.mse_loss(sim_c, sim_f)
    return lf + selfsim


class V2DistributionObjective(nn.Module):
    """GAN + PatchNCE + geometry + identity 的组合目标。"""

    def __init__(self, cfg: V2Config, *, latent_channels: int = 4, num_styles: int = 5) -> None:
        super().__init__()
        self.cfg = cfg
        self.nce_encoder = PatchNCEEncoder(latent_channels, int(cfg.nce_hidden))
        self.nce = PatchNCELoss(cfg)
        # 预先建好投影层，保证优化器在第一次 forward 前就能拿到全部参数
        self.nce.build_proj([int(cfg.nce_hidden) * 2, int(cfg.nce_hidden) * 2], int(cfg.nce_hidden))
        self.discriminator = ConditionalPatchDiscriminator(
            cfg, latent_channels=latent_channels, num_styles=num_styles
        )

    def _gen(self, G: nn.Module, content: torch.Tensor, ids: torch.Tensor, style_stats: torch.Tensor | None):
        return G(content, style_id=ids, style_stats=style_stats)

    def compute_generator_loss(
        self,
        G: nn.Module,
        content: torch.Tensor,
        target_style: torch.Tensor,
        target_style_id: Any,
        *,
        source_style_id: Any = None,
        conditioning: dict[str, Any] | None = None,
        style_stats: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        b = content.shape[0]
        dev = content.device
        ids = _ids(target_style_id, b, dev)
        ident = identity_rows(
            b,
            target_style_id=ids,
            source_style_id=source_style_id,
            conditioning=conditioning,
            device=dev,
        )

        out = self._gen(G, content, ids, style_stats)
        fake = out["z_hat"]

        # 条件 GAN：同一 style_id 条件下 fake vs real
        fake_logits = self.discriminator(fake, ids)
        adv = adv_g_loss(torch.cat([p.reshape(-1) for p in fake_logits]), self.cfg.adv_mode)

        # PatchNCE 只在跨风格行上计算；identity 行的正样本是恒等，没有信息量
        cross = ~ident
        zero = content.new_tensor(0.0)
        nce = zero
        if cross.any():
            src_feats = self.nce_encoder(content[cross])
            tgt_feats = self.nce_encoder(fake[cross])
            nce = self.nce(src_feats, tgt_feats)
        geo = geometry_loss(
            content,
            fake,
            scales=self.cfg.geometry_scales,
            patch=self.cfg.geometry_patch,
            stride=self.cfg.geometry_patch_stride,
            max_patches=self.cfg.geometry_max_patches,
        )

        id_loss = zero
        if ident.any():
            id_loss = F.l1_loss(fake[ident], content[ident])

        recon = zero
        if float(self.cfg.w_recon) > 0.0:
            out2 = self._gen(G, target_style, ids, style_stats)
            recon = F.l1_loss(out2["z_hat"], target_style)

        total = (
            float(self.cfg.w_adv) * adv
            + float(self.cfg.w_patchnce) * nce
            + float(self.cfg.w_geometry) * geo
            + float(self.cfg.w_identity) * id_loss
            + float(self.cfg.w_recon) * recon
        )
        return {
            "loss": total,
            "loss_adv": adv.detach(),
            "loss_patchnce": nce.detach(),
            "loss_geometry": geo.detach(),
            "loss_identity": id_loss.detach(),
            "loss_recon": recon.detach(),
            "identity_fraction": ident.float().mean().detach(),
            "z_hat": fake.detach(),
        }

    def compute_discriminator_loss(
        self,
        content: torch.Tensor,
        target_style: torch.Tensor,
        target_style_id: Any,
        fake: torch.Tensor,
        *,
        source_style_id: Any = None,
        conditioning: dict[str, Any] | None = None,
        style_stats: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del style_stats  # D 的条件来自自己的 style embedding，不共用 G 的统计
        b = content.shape[0]
        dev = content.device
        ids = _ids(target_style_id, b, dev)
        ident = identity_rows(
            b,
            target_style_id=ids,
            source_style_id=source_style_id,
            conditioning=conditioning,
            device=dev,
        )
        real = target_style.clone()
        if ident.any():
            real[ident] = content[ident]
        if float(self.cfg.r1_weight) > 0.0:
            real = real.detach().requires_grad_(True)
        real_logits = self.discriminator(real, ids)
        fake_logits = self.discriminator(fake.detach(), ids)
        real_flat = torch.cat([p.reshape(-1) for p in real_logits])
        fake_flat = torch.cat([p.reshape(-1) for p in fake_logits])
        loss = adv_d_loss(real_flat, fake_flat, self.cfg.adv_mode)
        if float(self.cfg.r1_weight) > 0.0:
            grad = torch.autograd.grad(
                sum(p.float().mean() for p in real_logits),
                real,
                create_graph=False,
                retain_graph=False,
                only_inputs=True,
            )[0]
            loss = loss + 0.5 * float(self.cfg.r1_weight) * grad.pow(2).mean()
        accuracy = 0.5 * ((real_flat > 0).float().mean() + (fake_flat < 0).float().mean())
        return {
            "loss": loss,
            "loss_adv_d": loss.detach(),
            "d_real_logit": real_flat.mean().detach(),
            "d_fake_logit": fake_flat.mean().detach(),
            "d_acc": accuracy.detach(),
        }


def discriminator_parameters(objective: V2DistributionObjective):
    return list(objective.discriminator.parameters())


def generator_extra_parameters(objective: V2DistributionObjective):
    return list(objective.nce_encoder.parameters()) + list(objective.nce.parameters())


__all__ = [
    "PatchNCEEncoder",
    "PatchNCELoss",
    "patch_nce_logits",
    "geometry_loss",
    "identity_rows",
    "V2DistributionObjective",
    "discriminator_parameters",
    "generator_extra_parameters",
]
