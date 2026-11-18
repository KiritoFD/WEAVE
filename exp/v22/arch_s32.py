"""V22-P2 结构消融 S2：Coarse-16 + 32×32 低强度结构条件（修复中频内容损失）。

S2 = SLFUNet + 32×32 SPADE 结构块（亮度 LCN_5 的 32×32 粗轮廓，注入在 up32 之前的 32×32 层）。
从 slf_final.pt 热启动（新块随机初始化，strict=False），沿用 finetune_slf_core 协议 1800 updates。
"""
from __future__ import annotations

import argparse
import copy
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import OUT, ROOT, autocast, sanitize_cuda_env, save_json, seed_all, utc
from exp.v18.ccf_model import SPADEStyleBlock, sinusoidal_time_emb
from exp.v20.slf_flow import SLFUNet, coarse_y16_lcn, project_latent_y16_lcn
from exp.v20.submode_patchswd import build_or_load_core_knn_bank
from exp.v19.fiber_duality import build_or_load_v19_foundation

BASE_CKPT = ROOT / "runs/v20/slf_flow/slf_final.pt"


class SLFUNetS32(SLFUNet):
    """SLFUNet + 32×32 低强度结构 SPADE（亮度轮廓），内容修复的架构臂。"""

    def __init__(self, base_dim: int = 144, u_dim: int = 128, cond_dim: int = 288, s32_ch: int = 2):
        super().__init__(base_dim=base_dim, u_dim=u_dim, cond_dim=cond_dim)
        w1 = base_dim * 2
        # 32×32 条件 = [LCN_5(y64) 的 2× 下采样, 平滑亮度] = 2ch；可学习残差门控（初始 ≈0.18 低强度）
        self.s32_gate = nn.Parameter(torch.tensor(-1.5))
        self.mid_s32_spade = SPADEStyleBlock(w1, s32_ch, cond_dim)

    def forward(self, z_tau, y16_lcn, dino_16, style_id, u_vivid, tau, *,
                drop_style=None, drop_content=None, layout_scale=None,
                y32_struct: torch.Tensor | None = None):
        B = z_tau.shape[0]
        dev = z_tau.device
        if drop_style is None:
            drop_style = torch.zeros(B, dtype=torch.bool, device=dev)
        if drop_content is None:
            drop_content = torch.zeros(B, dtype=torch.bool, device=dev)
        cmask4 = (~drop_content).float().view(B, 1, 1, 1)
        smask2 = (~drop_style).float().view(B, 1)
        if layout_scale is None:
            lscale = (~drop_content).float()
        else:
            lscale = layout_scale.to(dev).float().view(B).clamp(0.0, 1.0) * (~drop_content).float()
        lscale4 = lscale.view(B, 1, 1, 1)

        d_feat = self.dino_proj(dino_16 * cmask4) * lscale4
        s16 = torch.cat([y16_lcn * lscale4, d_feat], dim=1)

        oh = F.one_hot(style_id.clamp(0, self.N_STYLE - 1), self.N_STYLE).float() * smask2
        uv = u_vivid * smask2
        temb = sinusoidal_time_emb(tau, 64)
        flags = torch.stack([drop_style.float(), 1.0 - lscale], dim=-1)
        cond = self.cond_mlp(torch.cat([temb, oh, uv, flags], dim=-1))

        h64 = self.in_conv(z_tau)
        for blk in self.enc64:
            h64 = blk(h64, cond)

        h32 = self.down32(h64)
        for blk in self.enc32:
            h32 = blk(h32, cond)

        h16 = self.down16(h32)
        h16 = self.mid1(h16, s16, cond)
        h16 = self.mid_attn(h16)
        h16 = self.mid2(h16, s16, cond)

        u32 = self.up32(h16.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1))
        u32 = self.skip32(torch.cat([u32, h32], dim=1))
        # S2 新增：32×32 低强度残差门控注入（内容条件恒开启时生效；门控可学习）
        if y32_struct is not None:
            g = torch.sigmoid(self.s32_gate)
            u32 = u32 + g * (self.mid_s32_spade(u32, y32_struct * cmask4, cond) - u32)
        for blk in self.dec32:
            u32 = blk(u32, cond)

        u64 = self.up64(u32.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1))
        u64 = self.skip64(torch.cat([u64, h64], dim=1))
        for blk in self.dec64:
            u64 = blk(u64, cond)

        return self.out_conv(F.silu(self.out_norm(u64)))


def smooth_y32(y64: torch.Tensor) -> torch.Tensor:
    """32×32 平滑亮度轮廓（2× 池化 + 双 3×3 平滑），供 S2 结构条件。"""
    y32 = F.avg_pool2d(y64.float(), 2, 2)
    y32 = F.avg_pool2d(F.pad(y32, (1, 1, 1, 1), mode="replicate"), 3, 1)
    y32 = F.avg_pool2d(F.pad(y32, (1, 1, 1, 1), mode="replicate"), 3, 1)
    mu = y32.mean(dim=(2, 3), keepdim=True)
    sd = y32.std(dim=(2, 3), keepdim=True, unbiased=False)
    return ((y32 - mu) / (sd + 0.04)).clamp(-2.5, 2.5) / 2.5


def lcn5_y32(y64: torch.Tensor) -> torch.Tensor:
    from src.utils.structure import local_contrast_norm
    y32 = F.avg_pool2d(y64.float(), 2, 2)
    return local_contrast_norm(y32, kernel=5, eps=0.04)


def build_y32_pair(y64: torch.Tensor) -> torch.Tensor:
    """S2 的 2 通道 32×32 结构条件: [LCN_5(y64)↓2×, 平滑亮度]。"""
    return torch.cat([lcn5_y32(y64), smooth_y32(y64)], dim=1)


def train_s32(bundle: dict, core_bank: dict, updates: int = 1800,
              micro: int = 120, accum: int = 1, device: str = "cuda") -> Path:
    out_p = OUT / "arch_s32/slf_s32.pt"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    if out_p.exists():
        print(f"[V22 S32] Reusing {out_p}", flush=True)
        return out_p

    seed_all(20261210)
    tr = bundle["v18_cache"]["train"]
    w_luma = bundle["v18_cache"]["luma_proj"]["w"].to(device).float()
    b_luma = bundle["v18_cache"]["luma_proj"]["b"].to(device).float()

    z_all = tr["z"]
    sid_all = tr["sid"].long()
    d16_all = tr["dino_16"]
    y16_all = coarse_y16_lcn(tr["y_64"].float()).half()
    y32_all = build_y32_pair(tr["y_64"].float()).half()
    u_exact_all = core_bank["u_exact_all"]

    all_cls_idx = [(sid_all == s).nonzero(as_tuple=True)[0] for s in range(5)]
    core_cls_idx = [ci.long() for ci in core_bank["core_idx_per_style"]]

    model = SLFUNetS32(base_dim=144, u_dim=128).to(device)
    init_state = torch.load(BASE_CKPT, map_location="cpu", weights_only=False)["ema"]
    missing, unexpected = model.load_state_dict(init_state, strict=False)
    # 新增块: s32_gate / mid_s32.* 保持随机初始化；mid_s32.spade 的 norm affine 也随机
    print(f"[V22 S32] init missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(model.parameters(), lr=1.3e-4, betas=(0.9, 0.99), weight_decay=1e-4)
    t0 = time.time()

    for step in range(1, updates + 1):
        prog = step / max(1, updates)
        lr = 1.5e-5 + 0.5 * (1.3e-4 - 1.5e-5) * (1.0 + math.cos(math.pi * prog))
        for pg in opt.param_groups:
            pg["lr"] = lr

        opt.zero_grad(set_to_none=True)
        loss_fm_v, loss_st_v = 0.0, 0.0

        for _ in range(accum):
            per_cls = max(1, micro // 5)
            pools = core_cls_idx if (step % 5 != 0) else all_cls_idx
            idx = torch.cat([ci[torch.randint(0, len(ci), (per_cls,))] for ci in pools], dim=0)
            B = len(idx)

            z1 = z_all[idx].to(device, non_blocking=True).float()
            sid = sid_all[idx].to(device, non_blocking=True)
            d16 = d16_all[idx].to(device, non_blocking=True).float()
            y16 = y16_all[idx].to(device, non_blocking=True).float()
            y32 = y32_all[idx].to(device, non_blocking=True).float()
            uv = u_exact_all[idx].to(device, non_blocking=True).float()

            d16_aug = d16 + 0.08 * torch.randn_like(d16)
            d16_aug = d16_aug - d16_aug.mean(dim=(2, 3), keepdim=True)
            y16_aug = (y16 * (0.85 + 0.30 * torch.rand(B, 1, 1, 1, device=device)) + 0.04 * torch.randn_like(y16)).clamp(-1.1, 1.1)
            y32_aug = (y32 * (0.85 + 0.30 * torch.rand(B, 1, 1, 1, device=device)) + 0.04 * torch.randn_like(y32)).clamp(-1.1, 1.1)

            # P2 §2.1: 只 drop_style；内容永不 dropout；layout_scale=1.0
            ds = torch.rand(B, device=device) < 0.12
            dc = torch.zeros(B, dtype=torch.bool, device=device)
            lscale = torch.ones(B, device=device)

            z0 = torch.randn_like(z1)
            tau = torch.rand(B, device=device)
            zt = (1.0 - tau).view(B, 1, 1, 1) * z0 + tau.view(B, 1, 1, 1) * z1
            target_v = z1 - z0

            with autocast(device):
                pred_v = model(zt, y16_aug, d16_aug, sid, uv, tau,
                               drop_style=ds, drop_content=dc, layout_scale=lscale,
                               y32_struct=y32_aug)
                fm_loss = F.mse_loss(pred_v.float(), target_v)

                z1_hat = (zt + (1.0 - tau).view(B, 1, 1, 1) * pred_v.float()).clamp(-5.0, 5.0)
                y16_hat = project_latent_y16_lcn(z1_hat, w_luma, b_luma)
                st_loss = ((y16_hat - y16).abs().mean(dim=(1, 2, 3))).mean()
                loss = (fm_loss + 0.10 * st_loss) / accum

            loss.backward()
            loss_fm_v += float(fm_loss.item()) / accum
            loss_st_v += float(st_loss.item()) / accum

        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for pe_w, pw in zip(ema.parameters(), model.parameters()):
                pe_w.mul_(0.996).add_(pw, alpha=0.004)

        if step == 1 or step % 200 == 0 or step == updates:
            print(f"[V22 S32] {step:4d}/{updates} fm={loss_fm_v:.4f} y16={loss_st_v:.4f} ({time.time()-t0:.0f}s)", flush=True)

    torch.save({"ema": ema.state_dict(), "updates": updates,
                "base_ckpt": str(BASE_CKPT), "arch": "s32"}, out_p)
    print(f"[V22 S32] Saved {out_p} in {(time.time()-t0)/60:.1f} min", flush=True)
    return out_p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--updates", type=int, default=1800)
    args = ap.parse_args()

    sanitize_cuda_env()
    bundle = build_or_load_v19_foundation(args.pool, device=args.device)
    core_bank = build_or_load_core_knn_bank(bundle, k_sub=8, core_per_style=180, device=args.device)
    ckpt = train_s32(bundle, core_bank, updates=args.updates, device=args.device)
    save_json(OUT / "train_s32_done.json", {"updated_utc": utc(), "ckpt": str(ckpt), "updates": args.updates})


if __name__ == "__main__":
    main()
