r"""P0-1': Dec-QR-AdaGN 双臂受控实验 (用户 2026-11-23 批准设计).

配对纯度审计结论: V27 无监督 DINO-kNN 纯度 46.7% (ER 31%/Uki 37%) ≪ 75% — 配对污染
是 ER/Uki 坍缩主因. 本轮 A0/A1 全部使用 V21 干净同画派 nbr_idx (纯度 100%),
推理仍 100% 免标签 — 单变量架构裁决.

  A0 (纯净对照): 无局部注入, cond=[temb|e_ref128|v_mean192|flags] + 干净 nbr
      → 剥离配对污染后, 纯全局条件的真实天花板 (21% 还是 35%+?).
  A1 (主力 Dec-QR-AdaGN): QuotientStyleRouter (Q/K 同在 Q32 流形, 修复 V21 Q/K 错配)
      + SpatialAdaGNMod (dec32/dec64 各 2 block, 零初始化, 乘性+tanh 门控)
      → 双向瓶颈纯净 + 解码仿射路由能否同时拿到 V21 级风格与 V27 级内容.

门控: A1 SG-5 Top-1 ≥ 50% 且 Holdout ≥ 40% 且 DINO-C ≥ 0.55.
用法:  python -m exp.v27.p01_prime --pool ../pseudo_gt_24h/orig_mean_fp16 --arms A0,A1
"""
from __future__ import annotations

import argparse
import copy
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from exp.v21 import autocast
from exp.v21.prism_attn_flow import (
    BottleneckAttention,
    PureStyleResBlock,
    SPADEStyleBlock,
    _project_latent_y16,
    chroma_align_to_exemplar,
    sinusoidal_time_emb,
)
from exp.v27 import OUT, ROOT, RawWriterV27, run_measure_v27, save_json, utc, read_json
from src.utils.runtime import seed_all

STYLES = ("Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e")
E_FEAT_DIM = 423


# ------------------------------------------------- 模块 ----

class QuotientStyleRouter(nn.Module):
    """Q/K 严格同在 Q32 商流形 (修复 V21 Q/K 模态错配); V = 192d 笔触 token."""

    def __init__(self, q_dim: int = 32, v_dim: int = 192, d_qk: int = 64, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.d_head = d_qk // heads
        self.to_q = nn.Linear(q_dim, d_qk, bias=False)
        self.to_k = nn.Linear(q_dim, d_qk, bias=False)
        self.log_scale = nn.Parameter(torch.tensor(2.3))

    def forward(self, q_src_16: torch.Tensor, k_ref_tok: torch.Tensor,
                v_ref_tok: torch.Tensor) -> torch.Tensor:
        B, _, H, W = q_src_16.shape
        L = k_ref_tok.shape[1]
        qf = q_src_16.flatten(2).transpose(1, 2)
        q = F.normalize(self.to_q(qf).view(B, H * W, self.heads, self.d_head).transpose(1, 2), dim=-1)
        k = F.normalize(self.to_k(k_ref_tok).view(B, L, self.heads, self.d_head).transpose(1, 2), dim=-1)
        v = v_ref_tok.view(B, L, self.heads, -1).transpose(1, 2)
        scale = self.log_scale.exp().clamp(1.0, 25.0)
        out = F.scaled_dot_product_attention(q * scale, k, v, scale=1.0)
        return out.transpose(1, 2).reshape(B, H * W, -1).transpose(1, 2).reshape(B, -1, H, W)


class SpatialAdaGNMod(nn.Module):
    """零初始化空间仿射增量: (Δγ, Δβ) = P_zero(R), r_map 双线性对齐 h 分辨率."""

    def __init__(self, v_dim: int, ch: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(v_dim, hidden, 1), nn.SiLU(inplace=True),
            nn.Conv2d(hidden, ch * 2, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h_norm: torch.Tensor, r_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if r_map.shape[-2:] != h_norm.shape[-2:]:
            r_map = F.interpolate(r_map, size=h_norm.shape[-2:], mode="bilinear", align_corners=False)
        dg, db = self.net(r_map).chunk(2, dim=1)
        return torch.tanh(dg), db


class V27PrimeResBlock(PureStyleResBlock):
    """PureStyleResBlock + 零初始化空间仿射注入 (norm1/norm2 两处, r_map=None 时精确退化为基类)."""

    def __init__(self, ch: int, cond_dim: int, v_dim: int = 192):
        super().__init__(ch, cond_dim)
        self.smod1 = SpatialAdaGNMod(v_dim, ch)
        self.smod2 = SpatialAdaGNMod(v_dim, ch)

    def forward(self, x: torch.Tensor, cond: torch.Tensor,
                r_map: torch.Tensor | None = None) -> torch.Tensor:
        if r_map is None:
            return super().forward(x, cond)
        g1, b1 = self.film1(cond).chunk(2, dim=-1)
        gn1 = self.norm1(x)
        dg1, db1 = self.smod1(gn1, r_map)
        h = F.silu(gn1 * (1.0 + g1[:, :, None, None] + dg1) + b1[:, :, None, None] + db1)
        h = self.conv1(h)
        g2, b2 = self.film2(cond).chunk(2, dim=-1)
        gn2 = self.norm2(h)
        dg2, db2 = self.smod2(gn2, r_map)
        h = F.silu(gn2 * (1.0 + g2[:, :, None, None] + dg2) + b2[:, :, None, None] + db2)
        h = self.conv2(h)
        return x + h


class V27PrimeUNet(nn.Module):
    """V27Zero 骨架 + dec32/dec64 换 V27PrimeResBlock + QuotientStyleRouter.

    cond = [temb64 | e_ref 128 | v_mean 192 | flags2] = 386 (与 V21 cond 同宽, 零 onehot).
    mid16 瓶颈永久零局部参考注入 (双向瓶颈纯净).
    """

    def __init__(self, base_dim: int = 144, e_dim: int = 128, v_dim: int = 192,
                 cond_dim: int = 288, y32_scale: float = 1.0, feat_dim: int = 423,
                 use_router: bool = True):
        super().__init__()
        w0, w1, w2 = base_dim, base_dim * 2, base_dim * 3
        q_proj_ch = 64
        self.y32_scale = y32_scale
        self.use_router = use_router

        self.eref = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.SiLU(inplace=True), nn.Linear(256, e_dim),
        )
        self.q_proj = nn.Sequential(
            nn.Conv2d(32, 96, 1), nn.GroupNorm(16, 96), nn.SiLU(inplace=True),
            nn.Conv2d(96, q_proj_ch, 1),
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(64 + e_dim + v_dim + 2, cond_dim), nn.SiLU(inplace=True),
            nn.Linear(cond_dim, cond_dim), nn.SiLU(inplace=True),
        )
        self.in_conv = nn.Conv2d(4, w0, 3, padding=1)
        self.enc64 = nn.ModuleList([PureStyleResBlock(w0, cond_dim), PureStyleResBlock(w0, cond_dim)])
        self.down32 = nn.Conv2d(w0, w1, 3, stride=2, padding=1)
        self.enc32 = nn.ModuleList([PureStyleResBlock(w1, cond_dim), PureStyleResBlock(w1, cond_dim)])
        self.down16 = nn.Conv2d(w1, w2, 3, stride=2, padding=1)
        s16_ch = 1 + q_proj_ch
        self.mid1 = SPADEStyleBlock(w2, s16_ch, cond_dim)
        self.mid_self_attn = BottleneckAttention(w2, heads=8)
        self.mid2 = SPADEStyleBlock(w2, s16_ch, cond_dim)
        self.up32 = nn.Conv2d(w2, w1, 3, padding=1)
        self.skip32 = nn.Conv2d(w1 * 2, w1, 1)
        self.dec32_spade = SPADEStyleBlock(w1, 1, cond_dim)
        self.dec32 = nn.ModuleList([V27PrimeResBlock(w1, cond_dim, v_dim),
                                    V27PrimeResBlock(w1, cond_dim, v_dim)])
        self.up64 = nn.Conv2d(w1, w0, 3, padding=1)
        self.skip64 = nn.Conv2d(w0 * 2, w0, 1)
        self.dec64 = nn.ModuleList([V27PrimeResBlock(w0, cond_dim, v_dim),
                                    V27PrimeResBlock(w0, cond_dim, v_dim)])
        self.out_norm = nn.GroupNorm(16, w0)
        self.out_conv = nn.Conv2d(w0, 4, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)
        if use_router:
            self.router = QuotientStyleRouter(q_dim=32, v_dim=v_dim)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def encode_ref(self, feat: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.eref(feat), dim=-1)

    def forward(self, z_tau, y16_lcn, y32_smooth, q32_src, e_ref, v_mean, tau, *,
                drop_style=None, drop_content=None, layout_scale=None,
                k_ref_tok=None, v_ref_tok=None):
        B = z_tau.shape[0]
        dev = z_tau.device
        drop_style = torch.zeros(B, dtype=torch.bool, device=dev) if drop_style is None else drop_style
        drop_content = torch.zeros(B, dtype=torch.bool, device=dev) if drop_content is None else drop_content
        cmask4 = (~drop_content).float().view(B, 1, 1, 1)
        smask2 = (~drop_style).float().view(B, 1)
        lscale = ((~drop_content).float() if layout_scale is None
                  else layout_scale.to(dev).float().view(B).clamp(0.0, 1.0) * (~drop_content).float())
        lscale4 = lscale.view(B, 1, 1, 1)

        qf16 = self.q_proj(q32_src * cmask4) * lscale4
        s16 = torch.cat([y16_lcn * lscale4, qf16], dim=1)
        s32 = y32_smooth * (lscale4 * 0.85 * self.y32_scale)
        v_mean = v_mean * smask2

        temb = sinusoidal_time_emb(tau, 64)
        flags = torch.stack([drop_style.float(), 1.0 - lscale], dim=-1)
        cond = self.cond_mlp(torch.cat([temb, e_ref * smask2, v_mean, flags], dim=-1))

        # 空间路由: A 在 16×16 只算一次, R16 上采样复用至 dec32/dec64
        r16 = None
        if self.use_router and k_ref_tok is not None and v_ref_tok is not None:
            r16 = self.router(q_src_16 * lscale4, k_ref_tok * smask2.view(B, 1, 1),
                              v_ref_tok * smask2.view(B, 1, 1))
        r32 = F.interpolate(r16, size=(32, 32), mode="bilinear", align_corners=False) if r16 is not None else None
        r64 = F.interpolate(r16, size=(64, 64), mode="bilinear", align_corners=False) if r16 is not None else None

        h64 = self.in_conv(z_tau)
        for blk in self.enc64:
            h64 = blk(h64, cond)
        h32 = self.down32(h64)
        for blk in self.enc32:
            h32 = blk(h32, cond)
        h16 = self.down16(h32)
        h16 = self.mid1(h16, s16, cond)
        h16 = self.mid_self_attn(h16)
        h16 = self.mid2(h16, s16, cond)
        u32 = self.up32(h16.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1))
        u32 = self.skip32(torch.cat([u32, h32], dim=1))
        u32 = self.dec32_spade(u32, s32, cond)
        for blk in self.dec32:
            u32 = blk(u32, cond, r32)
        u64 = self.up64(u32.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1))
        u64 = self.skip64(torch.cat([u64, h64], dim=1))
        for blk in self.dec64:
            u64 = blk(u64, cond, r64)
        return self.out_conv(F.silu(self.out_norm(u64)))


# ------------------------------------------------- 训练 ----

def train_prime(arm: str, bundle, c21, feats_train: torch.Tensor,
                updates: int = 3800, micro: int = 96, accum: int = 1,
                device: str = "cuda", use_router: bool = True) -> Path:
    seed_all(20267100 if arm == "A1" else 20267099)
    ckpt_dir = OUT / f"prime_{arm}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    final_p = ckpt_dir / "final.pt"
    if final_p.exists():
        print(f"[{arm}] Reusing {final_p}", flush=True)
        return final_p
    tr = bundle["v18_cache"]["train"]
    tr21 = c21["train"]
    w_luma = bundle["v18_cache"]["luma_proj"]["w"].to(device).float()
    b_luma = bundle["v18_cache"]["luma_proj"]["b"].to(device).float()

    z_all = tr["z"]
    sid_all = tr["sid"].long()
    valid_m = tr21["valid_mask"]
    q32_all = tr21["q32"]
    y16_all = tr21["y16"]
    y32_all = tr21["y32"]
    vref_all = tr21["vref"]
    nbr_idx = tr21["nbr_idx"]  # V21 干净同画派核心近邻 (纯度 100%)
    all_cls_idx = [((sid_all == s) & valid_m).nonzero(as_tuple=True)[0] for s in range(5)]
    core_cls_idx = [ci.long() for ci in tr21["core_idx_per_style"]]

    model = V27PrimeUNet(use_router=use_router).to(device)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=2.0e-4, betas=(0.9, 0.99), weight_decay=1e-4)
    feats_dev = feats_train.to(device)
    vref_dev = vref_all.float().to(device)
    q32_dev = q32_all.float().to(device)
    t0 = time.time()

    for step in range(1, updates + 1):
        if step <= 250:
            lr = 2.0e-4 * (step / 250.0)
        else:
            prog = (step - 250) / max(1, updates - 250)
            lr = 2.0e-5 + 0.5 * (2.0e-4 - 2.0e-5) * (1.0 + math.cos(math.pi * prog))
        for pg in opt.param_groups:
            pg["lr"] = lr
        opt.zero_grad(set_to_none=True)
        loss_fm_v, loss_st_v = 0.0, 0.0
        for _ in range(accum):
            per_cls = max(1, micro // 5)
            pools = core_cls_idx if (step % 5 != 0) else all_cls_idx
            idx = torch.cat([ci[torch.randint(0, len(ci), (per_cls,))] for ci in pools], dim=0)
            B = len(idx)
            z1 = z_all[idx].to(device).float()
            q32 = q32_dev[idx]
            y16 = y16_all[idx].to(device).float()
            y32 = y32_all[idx].to(device).float()

            is_multi = (step % 5 in (1, 3))
            n_ref = 4 if is_multi else 1
            r_ids = nbr_idx[idx][:, :n_ref].to(device)  # 干净同画派近邻 (训练配对允许)
            f_ref = feats_dev[r_ids].mean(dim=1)
            u_ref = model.encode_ref(f_ref)
            # 参考路由 token: 异图近邻的 Q32 (256×32) 与 vref (256×192)
            k_tok = q32_dev[r_ids].flatten(2).transpose(1, 2).reshape(B, n_ref * 256, 32)
            v_tok = vref_dev[r_ids].reshape(B, n_ref * 256, 192)
            t_perm = torch.randperm(n_ref * 256, device=device)
            k_tok, v_tok = k_tok[:, t_perm, :], v_tok[:, t_perm, :]
            v_mean = v_tok.mean(dim=1)

            q32_aug = q32 + 0.06 * torch.randn_like(q32)
            q32_aug = q32_aug - q32_aug.mean(dim=(2, 3), keepdim=True)
            y16_aug = (y16 * (0.85 + 0.30 * torch.rand(B, 1, 1, 1, device=device))
                       + 0.03 * torch.randn_like(y16)).clamp(-1.1, 1.1)
            y32_aug = (y32 * (0.85 + 0.30 * torch.rand(B, 1, 1, 1, device=device))
                       + 0.03 * torch.randn_like(y32)).clamp(-1.1, 1.1)

            u_r = torch.rand(B, device=device)
            drop_s = u_r < 0.14
            drop_c = (u_r >= 0.14) & (u_r < 0.34)
            both = u_r >= 0.95
            drop_s = drop_s | both
            drop_c = drop_c | both
            lscale = torch.where(drop_c, torch.zeros(B, device=device),
                                 0.20 + 0.80 * torch.rand(B, device=device))

            z0 = torch.randn_like(z1)
            tau = torch.rand(B, device=device)
            zt = (1.0 - tau).view(B, 1, 1, 1) * z0 + tau.view(B, 1, 1, 1) * z1
            target_v = z1 - z0

            with autocast(device):
                pred_v = model(zt, y16_aug, y32_aug, q32_aug, u_ref, v_mean, tau,
                               drop_style=drop_s, drop_content=drop_c, layout_scale=lscale,
                               k_ref_tok=k_tok, v_ref_tok=v_tok)
                fm_loss = F.mse_loss(pred_v.float(), target_v)
                z1_hat = (zt + (1.0 - tau).view(B, 1, 1, 1) * pred_v.float()).clamp(-5.0, 5.0)
                y16_hat = _project_latent_y16(z1_hat, w_luma, b_luma)
                w_tau = (tau > 0.25).float() * lscale.square()
                st_loss = ((y16_hat - y16).abs().mean(dim=(1, 2, 3)) * w_tau).sum() / w_tau.sum().clamp_min(1.0)
                loss = (fm_loss + 0.12 * st_loss) / accum
            loss.backward()
            loss_fm_v += float(fm_loss.item()) / accum
            loss_st_v += float(st_loss.item()) / accum

        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for pe, pw in zip(ema.parameters(), model.parameters()):
                pe.mul_(0.997).add_(pw, alpha=0.003)
        if step == 1 or step % 400 == 0 or step == updates:
            print(f"[{arm}] {step:4d}/{updates} fm={loss_fm_v:.4f} st={loss_st_v:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    torch.save({"ema": ema.state_dict(), "updates": updates, "arm": arm,
                "params": sum(p.numel() for p in ema.parameters()),
                "use_router": use_router}, final_p)
    print(f"[{arm}] Saved {final_p} ({ema.n_params()/1e6:.2f}M) in {(time.time()-t0)/60:.1f} min", flush=True)
    return final_p


# ------------------------------------------------- 推理 ----

@torch.no_grad()
def integrate_prime(model, z_anchor, eps, y16, y32, q32, e_ref, v_mean,
                    k_ref_tok, v_ref_tok, *,
                    tau_hi=0.32, steps=14,
                    ws_fig=4.4, wsp_fig_late=3.2, wsp_abs=5.5,
                    wc_fig=1.60, wc_abs=0.55, lscale_fig=0.95, lscale_abs=0.25,
                    rescale_phi=0.50) -> torch.Tensor:
    dev = z_anchor.device
    B = len(z_anchor)
    tau_v = torch.full((B,), float(tau_hi), device=dev)
    z = (1.0 - tau_v.view(B, 1, 1, 1)) * eps + tau_v.view(B, 1, 1, 1) * z_anchor
    f0 = torch.zeros(B, dtype=torch.bool, device=dev)
    f1 = torch.ones(B, dtype=torch.bool, device=dev)
    ls0 = torch.zeros(B, device=dev)
    lscale = torch.full((B,), float(lscale_fig), device=dev)

    def _vel(z_cur, prog):
        tv = tau_v + prog * (1.0 - tau_v)
        with autocast(str(dev)):
            z4 = z_cur.repeat(4, 1, 1, 1)
            y16_4 = y16.repeat(4, 1, 1, 1)
            y32_4 = y32.repeat(4, 1, 1, 1)
            q4 = q32.repeat(4, 1, 1, 1)
            e4 = e_ref.repeat(4, 1)
            vm4 = v_mean.repeat(4, 1)
            k4 = k_ref_tok.repeat(4, 1, 1)
            v4 = v_ref_tok.repeat(4, 1, 1)
            t4 = tv.repeat(4)
            ds = torch.cat([f0, f0, f1, f1], 0)
            dc = torch.cat([f0, f1, f0, f1], 0)
            lsc4 = torch.cat([lscale, ls0, lscale, ls0], 0)
            v_sc, v_s0, v_0c, v_00 = model(
                z4, y16_4, y32_4, q4, e4, vm4, t4,
                drop_style=ds, drop_content=dc, layout_scale=lsc4,
                k_ref_tok=k4, v_ref_tok=v4,
            ).float().chunk(4, 0)
        gl = min(1.0, max(0.0, (prog - 0.38) / 0.32))
        wc = wc_fig - 0.35 * gl
        ws_j = ws_fig - 0.9 * gl
        ws_p = gl * wsp_fig_late
        vr = v_00 + wc * (v_0c - v_00) + ws_j * (v_sc - v_0c) + ws_p * (v_s0 - v_00)
        if rescale_phi > 0.0:
            s_ref = v_sc.flatten(1).std(1).view(B, 1, 1, 1).clamp_min(1e-4)
            s_raw = vr.flatten(1).std(1).view(B, 1, 1, 1).clamp_min(1e-4)
            vr = rescale_phi * vr * (s_ref / s_raw) + (1.0 - rescale_phi) * vr
        return vr

    dt = (1.0 - tau_v).view(B, 1, 1, 1) / float(steps)
    for i in range(steps):
        v1 = _vel(z, i / float(steps))
        zm = z + 0.5 * dt * v1
        v2 = _vel(zm, (i + 0.5) / float(steps))
        z = z + dt * v2
    return z


# ------------------------------------------------- 门控主流程 ----

def _pil_rgb(rgb: torch.Tensor):
    from PIL import Image as PILImage
    return PILImage.fromarray((rgb.permute(1, 2, 0).float().cpu().numpy() * 255)
                              .round().clip(0, 255).astype("uint8"))


@torch.no_grad()
def eval_prime(arm: str, ckpt_p, bundle, c21, feats, tasks, z_src_750, device: str = "cuda") -> dict:
    from src.utils import load_s10_decoder
    from src.utils.protocol import EVAL_SEED

    c21d = c21
    t750 = c21["test750"]
    sg5 = c21["sg5"]
    ho21 = c21["holdout"]
    ho_z = bundle["v18_cache"]["holdout"]["z"]
    P_Y = bundle["fnd"]["priors"]["P_Y"].to(device).float()
    P_C = bundle["fnd"]["priors"]["P_C"].to(device).float()
    eps_750 = torch.randn(z_src_750.shape, generator=torch.Generator().manual_seed(20261185))
    ck = torch.load(ckpt_p, map_location=device, weights_only=False)
    model = V27PrimeUNet(use_router=ck.get("use_router", True)).to(device).eval()
    model.load_state_dict(ck["ema"])
    dec = load_s10_decoder(device)

    arms = [
        (f"{arm}_SG5_s14", "sg5", dict(tau_hi=0.32, steps=14)),
        (f"{arm}_Holdout_s14", "ho", dict(tau_hi=0.40, steps=14)),
        (f"{arm}_Holdout_wrong_s14", "ho", dict(tau_hi=0.40, steps=14)),
    ]
    out = {}
    for arm_name, ref_mode, kw in arms:
        if (OUT / f"raw/{arm_name}/meta.json").exists() and                 (OUT / f"measured/{arm_name}/summary_v27.json").exists():
            print(f"[{arm}] eval reuse {arm_name}", flush=True)
            s = read_json(OUT / f"measured/{arm_name}/summary_v27.json")
        else:
          with torch.no_grad():
            wr = RawWriterV27(arm_name, dec=dec, device=device)
            is_wrong = "wrong" in arm_name
            for st in range(0, 750, 25):
                sl = slice(st, min(st + 25, 750))
                grp = tasks[sl]
                B = len(grp)
                zs = z_src_750[sl].to(device).float()
                ep = eps_750[sl].to(device).float()
                y16 = t750["y16"][sl].to(device).float()
                y32 = t750["y32"][sl].to(device).float()
                q32 = t750["q32"][sl].to(device).float()
                sid_p = torch.tensor([t["source_style_id"] for t in grp], device=device)
                tid_t = torch.tensor([t["target_style_id"] for t in grp], device=device)
                same = sid_p == tid_t
                tid_use = torch.where(is_wrong & ~same,
                                      (tid_t + 1 + (torch.rand(B, device=device) * 4).long()) % 5,
                                      tid_t)
                tid_cpu = tid_use.cpu()
                if ref_mode == "sg5":
                    z_r1 = sg5["z"][tid_cpu].to(device).float()
                    f_ref = feats["sg5"].to(device)[tid_cpu]
                    k_tok = sg5["q32"][tid_cpu].float().flatten(2).transpose(1, 2).to(device)
                    v_tok = sg5["vref"][tid_cpu].float().to(device)
                else:
                    hi = ho21["single_idx"][tid_cpu]
                    z_r1 = ho_z[hi].to(device).float()
                    f_ref = feats["holdout"].to(device)[hi]
                    k_tok = ho21["q32"][hi].float().flatten(2).transpose(1, 2).to(device)
                    v_tok = ho21["vref"][hi].float().to(device)
                u_ref = model.encode_ref(f_ref)
                v_mean = v_tok.mean(dim=1)
                z_anc = chroma_align_to_exemplar(zs, z_r1, P_Y, P_C, alpha=0.55)
                z_gen = integrate_prime(model, z_anc, ep, y16, y32, q32, u_ref, v_mean,
                                        k_tok, v_tok, **kw)
                z_out = torch.where(same.view(B, 1, 1, 1), zs, z_gen)
                wr.write_batch(z_out, grp)
            wr.finalize(arm=arm_name, ref_mode=ref_mode, **kw)
        s = run_measure_v27(arm_name, device=device)
        c = s["cross600"]
        five = {k: round(v["csd_top1"], 4) for k, v in s.get("per_target_cross120", {}).items()}
        out[arm_name] = {"top1": round(c["csd_top1"], 4), "dino_c": round(c["dino_c"], 4),
                         "dino_s": round(c["dino_s"], 4), "lpips": round(c["lpips"], 4),
                         "per_target": five}
        from exp.v26.gpu4090.p0_common import dual_gate_record
        dual_gate_record(f"P01P_{arm_name}",
                         {"csd_top1": c["csd_top1"], "dino_c": c["dino_c"],
                          "top1_per_target_style": five}, None,
                         out_name=f"p01p_dualgate_{arm_name}.json")
    del model
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--updates", type=int, default=3800)
    ap.add_argument("--arms", default="A0,A1")
    args = ap.parse_args()
    from exp.v26 import sanitize_cuda_env
    from exp.v19.fiber_duality import build_or_load_v19_foundation
    from exp.v21.stiefel_ref_cache import build_or_load_v21_cache
    from exp.v18 import tasks_and_sources
    sanitize_cuda_env()
    OUT.mkdir(parents=True, exist_ok=True)
    seed_all(20267200)
    pack = build_or_load_v21_cache(args.pool, device=args.device)
    bundle, c21 = pack["bundle"], pack["c21"]
    from exp.v27.v27_zero import precompute_ref_feats
    feats = precompute_ref_feats(bundle, c21, device=args.device)
    tasks, z_src_750, _ = tasks_and_sources()

    verdict = {"task": "P0-1' Dec-QR-AdaGN dual-arm", "gates": {"sg5": 0.50, "ho": 0.40, "dino_c": 0.55},
               "arms": {}, "updated_utc": utc()}
    for arm in args.arms.split(","):
        use_router = (arm == "A1")
        ckpt = train_prime(arm, bundle, c21, feats["train"], updates=args.updates,
                           device=args.device, use_router=use_router)
        res = eval_prime(arm, ckpt, bundle, c21, feats, tasks, z_src_750, device=args.device)
        verdict["arms"][arm] = res
        save_json(OUT / "p01_prime_verdict.json", verdict)
    a1 = verdict["arms"].get("A1", {})
    sg = a1.get("A1_SG5_s14", {})
    ho = a1.get("A1_Holdout_s14", {})
    if sg:
        verdict["gate_pass"] = bool(sg["top1"] >= 0.50 and ho["top1"] >= 0.40 and sg["dino_c"] >= 0.55)
    save_json(OUT / "p01_prime_verdict.json", verdict)
    print(f"[P0-1p] verdict saved, A1 gate_pass={verdict.get('gate_pass')}", flush=True)


if __name__ == "__main__":
    main()
