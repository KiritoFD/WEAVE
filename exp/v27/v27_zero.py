r"""V27-Zero: 纯全局 Stiefel 噪声流 (门控 1) — V21 减法重构.

相对 exp/v21/prism_attn_flow.py 的改动:
  1. 删 mid_ref_attn / dec32_ref_attn (V26 Exp 1: 加性 V = 商空间加性污染);
  2. cond = [temb64 | E_ref 128 | flags2] — 零 onehot、零 5 类压缩、零 vref-mean;
  3. E_ref 特征 = RGB-DINO μ/σ 384 (c21 vref token 矩) + Oklab 27 + 2 级 Haar 12 → MLP 128;
  4. 配对 = RGB-DINO μ 余弦 kNN 异图近邻 (r≠i, 无监督零标签);
  5. V21 铁律保留: Noise-to-Data Flow / Q32 / y16_lcn / y32 (可关) / 4-Branch CFG /
     drop_s 14% / drop_c 20% / fm+0.12·st_loss / 结构增广.
推理: 低 τ0 (tau_hi=0.32) + chroma_align_to_exemplar 免标签热启动, 6/14 步 RK2.

门控 (红线 1+3): SG-5 1-Shot Top-1 ≥ 45% 且 Holdout 单图 ≥ 38% 且 DINO-C ≥ 0.55,
双门控记录落盘 (SG-25 OOD 抽检在 Gate-1 通过后于 3060 补测).
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
from exp.v27 import OUT, ROOT, RawWriterV27, run_measure_v27
from src.utils.runtime import seed_all

STYLES = ("Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e")
E_FEAT_DIM = 423  # 384 dino μ/σ + 27 oklab + 12 haar


# ------------------------------------------------- 模型 ----

class V27ZeroUNet(nn.Module):
    """PRISMAttnUNet 减法版: 零加性 RefAttn, 内嵌 E_ref MLP, 零类别通道 (~24.6M)."""

    def __init__(self, base_dim: int = 144, e_dim: int = 128, cond_dim: int = 288,
                 y32_scale: float = 1.0, feat_dim: int = E_FEAT_DIM):
        super().__init__()
        w0, w1, w2 = base_dim, base_dim * 2, base_dim * 3
        q_proj_ch = 64
        self.y32_scale = y32_scale

        self.eref = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.SiLU(inplace=True),
            nn.Linear(256, e_dim),
        )
        self.q_proj = nn.Sequential(
            nn.Conv2d(32, 96, 1), nn.GroupNorm(16, 96), nn.SiLU(inplace=True),
            nn.Conv2d(96, q_proj_ch, 1),
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(64 + e_dim + 2, cond_dim), nn.SiLU(inplace=True),
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
        self.dec32 = nn.ModuleList([PureStyleResBlock(w1, cond_dim), PureStyleResBlock(w1, cond_dim)])
        self.up64 = nn.Conv2d(w1, w0, 3, padding=1)
        self.skip64 = nn.Conv2d(w0 * 2, w0, 1)
        self.dec64 = nn.ModuleList([PureStyleResBlock(w0, cond_dim), PureStyleResBlock(w0, cond_dim)])
        self.out_norm = nn.GroupNorm(16, w0)
        self.out_conv = nn.Conv2d(w0, 4, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def encode_ref(self, feat: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.eref(feat), dim=-1)

    def forward(self, z_tau, y16_lcn, y32_smooth, q32_src, e_ref, tau, *,
                drop_style=None, drop_content=None, layout_scale=None):
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

        temb = sinusoidal_time_emb(tau, 64)
        flags = torch.stack([drop_style.float(), 1.0 - lscale], dim=-1)
        cond = self.cond_mlp(torch.cat([temb, e_ref * smask2, flags], dim=-1))

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
        h32d = self.up32(h16.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1))
        h32d = self.skip32(torch.cat([h32d, h32], dim=1))
        h32d = self.dec32_spade(h32d, s32, cond)
        for blk in self.dec32:
            h32d = blk(h32d, cond)
        h64d = self.up64(h32d.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1))
        h64d = self.skip64(torch.cat([h64d, h64], dim=1))
        for blk in self.dec64:
            h64d = blk(h64d, cond)
        return self.out_conv(F.silu(self.out_norm(h64d)))


# ------------------------------------------------- 特征 ----

def haar2_feats(y: torch.Tensor) -> torch.Tensor:
    """2 级 Haar 能量 RMS + 偏度 (B, 12)."""
    feats = []
    cur = y
    for _ in range(2):
        a = cur[:, :, 0::2, 0::2]
        b = cur[:, :, 0::2, 1::2]
        c = cur[:, :, 1::2, 0::2]
        d = cur[:, :, 1::2, 1::2]
        for det in ((a - b) / 2, (a - c) / 2, (a - d) / 2):
            f = det.flatten(1)
            mu = f.mean(1, keepdim=True)
            sd = f.std(1, keepdim=True).clamp_min(1e-6)
            feats.append(f.square().mean(1).clamp_min(1e-8).sqrt())
            feats.append(((f - mu) / sd).pow(3).mean(1))
        cur = (a + b + c + d) / 4
    return torch.stack(feats, dim=1)


@torch.no_grad()
def precompute_ref_feats(bundle, c21, device: str = "cuda") -> dict:
    """train/sg5/holdout 全量 E_ref 输入特征 (N, 423)."""
    from exp.v19.fiber_duality import extract_color_fiber_27d
    from src.utils import decode_s10, load_s10_decoder

    luma_w = torch.tensor([0.299, 0.587, 0.114], device=device).view(1, 3, 1, 1)
    dec = load_s10_decoder(device)
    out = {}
    for split, zs, vref in (
            ("train", bundle["v18_cache"]["train"]["z"], c21["train"]["vref"]),
            ("sg5", c21["sg5"]["z"], c21["sg5"]["vref"]),
            ("holdout", bundle["v18_cache"]["holdout"]["z"], c21["holdout"]["vref"])):
        vref = vref.float()
        dino_mu = F.normalize(vref.mean(dim=1), dim=-1)  # (N, 192)
        dino_sd = vref.std(dim=1).clamp_min(1e-4)
        feats = []
        for st in range(0, len(zs), 24):
            rgb = decode_s10(dec, zs[st:st + 24].to(device).float(), batch=24)
            rgb = torch.nan_to_num(rgb, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
            for j in range(len(rgb)):
                ok27 = extract_color_fiber_27d(rgb[j][None].to(device))[0]
                y = (rgb[j].to(device) * luma_w).sum(1, keepdim=True).clamp(0.0, 1.0)
                h12 = torch.nan_to_num(haar2_feats(y[None])[0], nan=0.0, posinf=1.0, neginf=-1.0)
                fv = torch.cat([dino_mu[st + j].to(device),
                                dino_sd[st + j].to(device),
                                ok27, h12])
                feats.append(torch.nan_to_num(fv, nan=0.0, posinf=1.0, neginf=-1.0))
        out[split] = torch.stack(feats, 0)
        print(f"[V27 Feats] {split}: {tuple(out[split].shape)}", flush=True)
    return out


def build_knn_pairs(c21, k: int = 12, device: str = "cuda") -> torch.Tensor:
    """无监督 RGB-DINO μ 余弦 kNN 异图近邻 (N, k), 排除自身."""
    vref = c21["train"]["vref"].float()
    mu = F.normalize(vref.mean(dim=1), dim=-1).to(device)
    sims = mu @ mu.T
    sims.fill_diagonal_(-1e9)
    return sims.topk(k, dim=1).indices.cpu()


# ------------------------------------------------- 训练 ----

def train_v27(bundle, c21, feats_train: torch.Tensor, nbr: torch.Tensor,
              updates: int = 3800, micro: int = 96, accum: int = 1,
              device: str = "cuda", y32_scale: float = 1.0) -> Path:
    seed_all(20267000)
    ckpt_dir = OUT / "attn_flow"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    final_p = ckpt_dir / "v27_zero_final.pt"
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
    all_cls_idx = [((sid_all == s) & valid_m).nonzero(as_tuple=True)[0] for s in range(5)]
    core_cls_idx = [ci.long() for ci in tr21["core_idx_per_style"]]

    from src.utils.runtime import tier_micro_accum
    micro, accum = tier_micro_accum(micro, default_micro_24gb=96, max_micro_12gb=24, device=device)
    model = V27ZeroUNet(base_dim=144, y32_scale=y32_scale).to(device)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=2.0e-4, betas=(0.9, 0.99), weight_decay=1e-4)
    feats_dev = feats_train.to(device)
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
            q32 = q32_all[idx].to(device).float()
            y16 = y16_all[idx].to(device).float()
            y32 = y32_all[idx].to(device).float()

            is_multi = (step % 5 in (1, 3))
            n_ref = 4 if is_multi else 1
            r_ids = nbr[idx][:, :n_ref].to(device)  # 无监督 kNN 异图近邻 (r≠i)
            f_ref = feats_dev[r_ids].mean(dim=1)
            u_ref = model.encode_ref(f_ref)

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
                pred_v = model(zt, y16_aug, y32_aug, q32_aug, u_ref, tau,
                               drop_style=drop_s, drop_content=drop_c, layout_scale=lscale)
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
            print(f"[V27 Flow] {step:4d}/{updates} fm={loss_fm_v:.4f} st={loss_st_v:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    torch.save({"ema": ema.state_dict(), "updates": updates,
                "params": sum(p.numel() for p in ema.parameters()),
                "y32_scale": y32_scale}, final_p)
    print(f"[V27 Flow] Saved {final_p} ({ema.n_params()/1e6:.2f}M) in {(time.time()-t0)/60:.1f} min", flush=True)
    return final_p


# ------------------------------------------------- 推理 ----

@torch.no_grad()
def integrate_v27(model, z_anchor, eps, y16, y32, q32, e_ref, *,
                  tau_min=0.0, tau_hi=0.32, steps=14,
                  ws_fig=4.4, wsp_fig_late=3.2, wsp_abs=5.5,
                  wc_fig=1.60, wc_abs=0.55, lscale_fig=0.95, lscale_abs=0.25,
                  rescale_phi=0.50) -> torch.Tensor:
    """4-Branch RK2 (V21 曲线) — 零 k/v (注意力已删), 零类别标签."""
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
            t4 = tv.repeat(4)
            ds = torch.cat([f0, f0, f1, f1], 0)
            dc = torch.cat([f0, f1, f0, f1], 0)
            lsc4 = torch.cat([lscale, ls0, lscale, ls0], 0)
            v_sc, v_s0, v_0c, v_00 = model(
                z4, y16_4, y32_4, q4, e4, t4,
                drop_style=ds, drop_content=dc, layout_scale=lsc4,
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


# ------------------------------------------------- 门控 1 主流程 ----

def _pil_rgb(rgb: torch.Tensor):
    from PIL import Image as PILImage
    return PILImage.fromarray((rgb.permute(1, 2, 0).float().cpu().numpy() * 255)
                              .round().clip(0, 255).astype("uint8"))


def run_gate1(pool: str, device: str = "cuda", updates: int = 3800,
              reuse_ckpt: bool = True) -> dict:
    from exp.v19.fiber_duality import build_or_load_v19_foundation
    from exp.v21.stiefel_ref_cache import build_or_load_v21_cache
    from exp.v18 import tasks_and_sources
    from src.utils import decode_s10, load_s10_decoder
    from src.utils.protocol import EVAL_SEED

    pack = build_or_load_v21_cache(pool, device=device)
    bundle, c21 = pack["bundle"], pack["c21"]
    feats = precompute_ref_feats(bundle, c21, device=device)
    nbr = build_knn_pairs(c21, k=12, device=device)

    final_p = OUT / "attn_flow" / "v27_zero_final.pt"
    if not (reuse_ckpt and final_p.exists()):
        train_v27(bundle, c21, feats["train"], nbr, updates=updates, device=device)
    model = V27ZeroUNet(base_dim=144).to(device).eval()
    model.load_state_dict(torch.load(final_p, map_location=device, weights_only=False)["ema"])
    for p in model.parameters():
        p.requires_grad_(False)

    tasks, z_src_750, _ = tasks_and_sources()
    t750 = c21["test750"]
    sg5 = c21["sg5"]
    ho21 = c21["holdout"]
    ho_z = bundle["v18_cache"]["holdout"]["z"]
    P_Y = bundle["fnd"]["priors"]["P_Y"].to(device).float()
    P_C = bundle["fnd"]["priors"]["P_C"].to(device).float()
    eps_750 = torch.randn(z_src_750.shape, generator=torch.Generator().manual_seed(20261185))

    dec = load_s10_decoder(device)
    arms = [
        ("V27_1Shot_SG5_s6", "sg5", dict(tau_hi=0.32, steps=6)),
        ("V27_1Shot_SG5_s14", "sg5", dict(tau_hi=0.32, steps=14)),
        ("V27_1Shot_Holdout_s14", "ho", dict(tau_hi=0.40, steps=14)),
        ("V27_1Shot_Holdout_wrong_s14", "ho", dict(tau_hi=0.40, steps=14)),
    ]
    verdict = {"gates": {"sg5_top1": 0.45, "holdout_top1": 0.38, "dino_c": 0.55}, "arms": {}}
    for arm_name, ref_mode, kw in arms:
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
                else:
                    hi = ho21["single_idx"][tid_cpu]
                    z_r1 = ho_z[hi].to(device).float()
                    f_ref = feats["holdout"].to(device)[hi]
                u_ref = model.encode_ref(f_ref)
                z_anc = chroma_align_to_exemplar(zs, z_r1, P_Y, P_C, alpha=0.55)
                z_gen = integrate_v27(model, z_anc, ep, y16, y32, q32, u_ref, **kw)
                z_out = torch.where(same.view(B, 1, 1, 1), zs, z_gen)
                wr.write_batch(z_out, grp)
            wr.finalize(arm=arm_name, **kw)
        s = run_measure_v27(arm_name, device=device)
        c = s["cross600"]
        five = {k: round(v["csd_top1"], 4) for k, v in s.get("per_target_cross120", {}).items()}
        verdict["arms"][arm_name] = {
            "top1": round(c["csd_top1"], 4), "dino_c": round(c["dino_c"], 4),
            "dino_s": round(c["dino_s"], 4), "lpips": round(c["lpips"], 4),
            "per_target": five}
        from exp.v26.gpu4090.p0_common import dual_gate_record
        dual_gate_record(f"P0G1_{arm_name}",
                         {"csd_top1": c["csd_top1"], "dino_c": c["dino_c"],
                          "top1_per_target_style": five}, None,
                         out_name=f"p0g1_dualgate_{arm_name}.json")

    sg = verdict["arms"]["V27_1Shot_SG5_s14"]
    ho = verdict["arms"]["V27_1Shot_Holdout_s14"]
    verdict["gate_pass"] = bool(sg["top1"] >= 0.45 and ho["top1"] >= 0.38 and sg["dino_c"] >= 0.55)
    save_json(OUT / "p0_gate1_verdict.json", verdict)
    print(f"[V27 Gate1] SG5={sg['top1']} Holdout={ho['top1']} DINO-C={sg['dino_c']} "
          f"→ {'PASS' if verdict['gate_pass'] else 'FAIL'}", flush=True)
    if not verdict["gate_pass"]:
        print("[V27 Gate1] 未达标 — 逐臂数字见 p0_gate1_verdict.json, 不进入 78k 扩容", flush=True)
    return verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--updates", type=int, default=3800)
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()
    from exp.v26 import sanitize_cuda_env
    sanitize_cuda_env()
    run_gate1(pool=args.pool, device=args.device, updates=args.updates,
              reuse_ckpt=not args.fresh)


if __name__ == "__main__":
    main()
