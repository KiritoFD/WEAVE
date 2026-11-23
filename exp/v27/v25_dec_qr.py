r"""V25_Tr_T1 + Dec-QR-AdaGN: 回到 [0.61, 0.70] 有效 DINO-C 区间测试真架构增益.

纠偏背景 (2026-11-23):
  1. V27-Zero/A0/A1 把门控降到 DINO-C>=0.55, 实际在 25% Top-1 下就已跌到 0.560-0.574
     (低于 StyleGallery 0.6113), 根因是照搬 V21 的 tau_hi=0.32 重噪声起点 + Q32;
  2. p01_prime.py:293 存在 k_tok flatten(2) 维度混洗 Bug, A1 实际喂给 router 的是乱码 K;
  3. V19_S3/n2c_z.pt["z0"] 含 b4(..., tid_true) 隐式泄漏 (实测 z0_alone=36%, sid=0 跌回 2%).

本实验严格从 V25_Tr_T1 (DINO-C=0.6613, Top-1=56.67%) 出发, 零泄漏接入 Dec-QR-AdaGN:
  - 底座: slf_Tr_T1.pt (SLFUNet, dino_16 + y16_lcn 瓶颈), 冻结 encoder+mid16 守住 DINO-C;
  - 手术: cond_mlp onehot(5) 列置零 (sid 恒 0); 解冻 cond_mlp + dec32/dec64 + out_*;
  - 空间注入: 仅在 dec32/dec64 换 V27PrimeResBlock (零初始化 SpatialAdaGNMod) +
              QuotientStyleRouter (Q/K 同在 Q32, 修复 flatten(3).transpose(2,3) 混洗 Bug);
  - 破源锁: 50% Anchor-Bridge (blur3x3(z1) + 异图/邻图色度对齐) tau0~U(0.35, 0.62) +
            100% 干净 V21 nbr_idx 异图参考 (u_ref = u_all[r_ids], k_tok, v_tok);
  - 推理: 100% 免标签 — dsd_ref 直接由 fib_ref[:,411:795].mean() 计算, z_anc = chroma_align
          (或 blur3x3(z_src) chroma_align), V25 原生 cs_j 双分支 CFG (D5/D6H 两水位).

门控 (恢复有效区间标准): DINO-C >= 0.61, SG-5 Top-1 >= 45%, Holdout >= 38%.
用法:  python -m exp.v27.v25_dec_qr --pool ../pseudo_gt_24h/orig_mean_fp16 --updates 1500
"""
from __future__ import annotations

import argparse
import copy
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from exp.v20.slf_flow import (
    BottleneckAttention,
    PureStyleResBlock,
    SPADEStyleBlock,
    coarse_y16_lcn,
    project_latent_y16_lcn,
    sinusoidal_time_emb,
)
from exp.v21.prism_attn_flow import chroma_align_to_exemplar
from exp.v22.core import D5_SOLVER, GateConfig, effective_gate_components
from exp.v25.r1_inference_ablation import D6H_SOLVER
from exp.v27 import OUT, ROOT, RawWriterV27, run_measure_v27
from exp.v27.p01_prime import QuotientStyleRouter, V27PrimeResBlock
from src.utils.runtime import bf16_autocast as autocast, read_json, save_json, seed_all, tier_micro_accum, utc

V25_T1_CKPT = ROOT / "runs/v25/r2_tmatrix/slf_Tr_T1.pt"


class V25DecQRNet(nn.Module):
    """SLFUNet (V25_Tr_T1 权重兼容) + dec32/dec64 V27PrimeResBlock + QuotientStyleRouter."""

    N_STYLE = 5

    def __init__(self, base_dim: int = 144, u_dim: int = 128, cond_dim: int = 288,
                 v_dim: int = 192, use_router: bool = True):
        super().__init__()
        w0, w1, w2 = base_dim, base_dim * 2, base_dim * 3
        dino_ch = 80
        self.use_router = use_router

        self.dino_proj = nn.Sequential(
            nn.Conv2d(384, 160, 1), nn.GroupNorm(16, 160), nn.SiLU(inplace=True),
            nn.Conv2d(160, dino_ch, 1),
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(64 + self.N_STYLE + u_dim + 2, cond_dim), nn.SiLU(inplace=True),
            nn.Linear(cond_dim, cond_dim), nn.SiLU(inplace=True),
        )
        self.in_conv = nn.Conv2d(4, w0, 3, padding=1)
        self.enc64 = nn.ModuleList([PureStyleResBlock(w0, cond_dim), PureStyleResBlock(w0, cond_dim)])
        self.down32 = nn.Conv2d(w0, w1, 3, stride=2, padding=1)
        self.enc32 = nn.ModuleList([PureStyleResBlock(w1, cond_dim), PureStyleResBlock(w1, cond_dim)])
        self.down16 = nn.Conv2d(w1, w2, 3, stride=2, padding=1)

        s16_ch = 1 + dino_ch
        self.mid1 = SPADEStyleBlock(w2, s16_ch, cond_dim)
        self.mid_attn = BottleneckAttention(w2, heads=8)
        self.mid2 = SPADEStyleBlock(w2, s16_ch, cond_dim)

        self.up32 = nn.Conv2d(w2, w1, 3, padding=1)
        self.skip32 = nn.Conv2d(w1 * 2, w1, 1)
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

    def zero_onehot(self) -> None:
        with torch.no_grad():
            self.cond_mlp[0].weight[:, 64:64 + self.N_STYLE].zero_()

    def forward(self, z_tau, y16_lcn, dino_16, u_vivid, tau, *,
                drop_style=None, drop_content=None, layout_scale=None,
                q32_src=None, k_ref_tok=None, v_ref_tok=None):
        B = z_tau.shape[0]
        dev = z_tau.device
        drop_style = torch.zeros(B, dtype=torch.bool, device=dev) if drop_style is None else drop_style
        drop_content = torch.zeros(B, dtype=torch.bool, device=dev) if drop_content is None else drop_content

        cmask4 = (~drop_content).float().view(B, 1, 1, 1)
        smask2 = (~drop_style).float().view(B, 1)
        lscale = ((~drop_content).float() if layout_scale is None
                  else layout_scale.to(dev).float().view(B).clamp(0.0, 1.0) * (~drop_content).float())
        lscale4 = lscale.view(B, 1, 1, 1)

        d_feat = self.dino_proj(dino_16 * cmask4) * lscale4
        s16 = torch.cat([y16_lcn * lscale4, d_feat], dim=1)

        oh = z_tau.new_zeros((B, self.N_STYLE), dtype=torch.float32)  # 100% 零类别标签
        uv = u_vivid.float() * smask2
        temb = sinusoidal_time_emb(tau, 64)
        flags = torch.stack([drop_style.float(), 1.0 - lscale], dim=-1)
        cond = self.cond_mlp(torch.cat([temb, oh, uv, flags], dim=-1))

        r16 = None
        if self.use_router and q32_src is not None and k_ref_tok is not None and v_ref_tok is not None:
            sm3 = smask2.view(B, 1, 1)
            r16 = self.router(q32_src * lscale4, k_ref_tok * sm3, v_ref_tok * sm3)
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
        h16 = self.mid_attn(h16)
        h16 = self.mid2(h16, s16, cond)

        u32 = self.up32(h16.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1))
        u32 = self.skip32(torch.cat([u32, h32], dim=1))
        for blk in self.dec32:
            u32 = blk(u32, cond, r32)

        u64 = self.up64(u32.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1))
        u64 = self.skip64(torch.cat([u64, h64], dim=1))
        for blk in self.dec64:
            u64 = blk(u64, cond, r64)
        return self.out_conv(F.silu(self.out_norm(u64)))


def _pack_ref_tokens(q32_src_set: torch.Tensor, vref_set: torch.Tensor, r_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """r_ids: (B, n_ref) → (k_tok (B, n_ref*256, 32), v_tok (B, n_ref*256, 192)) — 修复 flatten(3).transpose(2,3)!"""
    B, n_ref = r_ids.shape
    k_tok = q32_src_set[r_ids].float().flatten(3).transpose(2, 3).reshape(B, n_ref * 256, 32)
    v_tok = vref_set[r_ids].float().reshape(B, n_ref * 256, 192)
    return k_tok, v_tok


def train_v25_decqr(arm: str, bundle: dict, c21: dict, *, updates: int = 1500,
                    micro: int = 60, device: str = "cuda", seed: int = 20267300,
                    use_router: bool = True, ab_prob: float = 0.50,
                    freeze_mid: bool = True, smoke: bool = False) -> str:
    ckpt_dir = OUT / f"v25_{arm}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_p = ckpt_dir / "final.pt"
    if out_p.exists() and not smoke:
        print(f"[{arm}] Reusing {out_p}", flush=True)
        return str(out_p)

    seed_all(seed)
    micro, accum = tier_micro_accum(micro, default_micro_24gb=30, max_micro_12gb=15, device=device)
    tr = bundle["v18_cache"]["train"]
    tr21 = c21["train"]
    w_luma = bundle["v18_cache"]["luma_proj"]["w"].to(device).float()
    b_luma = bundle["v18_cache"]["luma_proj"]["b"].to(device).float()
    P_Y = bundle["fnd"]["priors"]["P_Y"].to(device).float()
    P_C = bundle["fnd"]["priors"]["P_C"].to(device).float()

    z_all = tr["z"]
    sid_all = tr["sid"].long()
    d16_all = tr["dino_16"]
    y16_all = coarse_y16_lcn(tr["y_64"].float()).half()
    q32_all = tr21["q32"]
    vref_all = tr21["vref"]
    u_all = tr21["u"]
    nbr_idx = tr21["nbr_idx"]
    valid_m = tr21["valid_mask"]
    all_cls_idx = [((sid_all == s) & valid_m).nonzero(as_tuple=True)[0] for s in range(5)]
    core_cls_idx = [ci.long() for ci in tr21["core_idx_per_style"]]

    model = V25DecQRNet(use_router=use_router).to(device)
    ck_base = torch.load(V25_T1_CKPT, map_location="cpu", weights_only=False)["ema"]
    miss, unexp = model.load_state_dict(ck_base, strict=False)
    print(f"[{arm}] loaded V25_Tr_T1: missing={len(miss)} (router+smod) unexpected={len(unexp)}", flush=True)
    assert len(unexp) == 0, f"Unexpected keys from V25_Tr_T1: {unexp}"
    model.zero_onehot()

    # 冻结 encoder + 16×16 瓶颈 (守住 V25 DINO-C 水位), 只训 cond_mlp + 解码端 + router + out
    unfreeze = ("cond_mlp.", "up32.", "skip32.", "dec32.", "up64.", "skip64.", "dec64.",
                "out_norm.", "out_conv.", "router.") if freeze_mid else None
    new_params, dec_params = [], []
    for n, p in model.named_parameters():
        if unfreeze is None or n.startswith(unfreeze):
            p.requires_grad_(True)
            if "smod" in n or n.startswith("router."):
                new_params.append(p)
            else:
                dec_params.append(p)
        else:
            p.requires_grad_(False)
    print(f"[{arm}] trainable new={sum(p.numel() for p in new_params)/1e6:.2f}M "
          f"dec/cond={sum(p.numel() for p in dec_params)/1e6:.2f}M (micro={micro}×accum={accum})", flush=True)

    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW([
        {"params": new_params, "lr": 2.0e-4},
        {"params": dec_params, "lr": 6.0e-5},
    ], betas=(0.9, 0.99), weight_decay=1e-4)

    t0 = time.time()
    for step in range(1, updates + 1):
        prog = step / max(1, updates)
        scale = 0.15 + 0.85 * 0.5 * (1.0 + math.cos(math.pi * prog))
        opt.param_groups[0]["lr"] = 2.0e-4 * scale
        opt.param_groups[1]["lr"] = 6.0e-5 * scale

        opt.zero_grad(set_to_none=True)
        fm_v, st_v = 0.0, 0.0
        for _ in range(accum):
            per_cls = max(1, micro // 5)
            pools = core_cls_idx if (step % 5 != 0) else all_cls_idx
            idx = torch.cat([ci[torch.randint(0, len(ci), (per_cls,))] for ci in pools], dim=0)
            B = len(idx)

            z1 = z_all[idx].to(device, non_blocking=True).float()
            d16 = d16_all[idx].to(device, non_blocking=True).float()
            y16 = y16_all[idx].to(device, non_blocking=True).float()
            q32 = q32_all[idx].to(device, non_blocking=True).float()

            # 100% 干净同画派异图参考 (60% 1-Shot / 40% 4-Shot, 无放回)
            n_ref = 4 if (step % 5 in (1, 3)) else 1
            perm = torch.rand(B, 12, device=device).topk(n_ref, dim=1).indices.cpu()
            r_ids = nbr_idx[idx[:, None], perm]  # (B, n_ref)
            u_ref = F.normalize(u_all[r_ids].float().mean(dim=1), dim=-1).to(device, non_blocking=True)
            k_tok, v_tok = _pack_ref_tokens(q32_all, vref_all, r_ids)
            t_perm = torch.randperm(n_ref * 256)
            k_tok = k_tok[:, t_perm, :].to(device, non_blocking=True)
            v_tok = v_tok[:, t_perm, :].to(device, non_blocking=True)

            d16_aug = d16 + 0.08 * torch.randn_like(d16)
            d16_aug = d16_aug - d16_aug.mean(dim=(2, 3), keepdim=True)
            y16_aug = (y16 * (0.85 + 0.30 * torch.rand(B, 1, 1, 1, device=device))
                       + 0.04 * torch.randn_like(y16)).clamp(-1.1, 1.1)

            u_r = torch.rand(B, device=device)
            ds = u_r < 0.14
            dc = (u_r >= 0.14) & (u_r < 0.44)  # 30% drop_c (保持 V25_Tr_T1)
            both = u_r >= 0.95
            ds, dc = ds | both, dc | both
            lscale = torch.ones(B, device=device)

            eps = torch.randn_like(z1)
            tau_fm = torch.rand(B, device=device)
            zt_fm = (1.0 - tau_fm).view(B, 1, 1, 1) * eps + tau_fm.view(B, 1, 1, 1) * z1
            v_fm = z1 - eps

            is_ab = torch.rand(B, device=device) < ab_prob
            if is_ab.any():
                # 免标签去风格伪源锚: 模糊削弱源笔触 + 异色度扰动再对齐目标参考图色度
                z_ref1 = z_all[r_ids[:, 0]].to(device, non_blocking=True).float()
                z_blur = F.avg_pool2d(F.pad(z1, (1, 1, 1, 1), mode="replicate"), 3, 1)
                z_anc = chroma_align_to_exemplar(z_blur, z_ref1, P_Y, P_C, alpha=0.55)
                tau0 = 0.32 + 0.32 * torch.rand(B, device=device)  # U(0.32, 0.64) 匹配 cs_j 推理区间
                z_init = (1.0 - tau0).view(B, 1, 1, 1) * eps + tau0.view(B, 1, 1, 1) * z_anc
                tau_ab = tau0 + (1.0 - tau0) * torch.rand(B, device=device)
                frac = ((tau_ab - tau0) / (1.0 - tau0).clamp_min(1e-4)).view(B, 1, 1, 1)
                zt_ab = z_init + frac * (z1 - z_init)
                v_ab = (z1 - z_init) / (1.0 - tau0).clamp_min(1e-4).view(B, 1, 1, 1)
                m4 = is_ab.view(B, 1, 1, 1)
                tau = torch.where(is_ab, tau_ab, tau_fm)
                zt = torch.where(m4, zt_ab, zt_fm)
                target_v = torch.where(m4, v_ab, v_fm)
            else:
                tau, zt, target_v = tau_fm, zt_fm, v_fm

            with autocast(device):
                pred_v = model(zt, y16_aug, d16_aug, u_ref, tau,
                               drop_style=ds, drop_content=dc, layout_scale=lscale,
                               q32_src=q32, k_ref_tok=k_tok, v_ref_tok=v_tok)
                fm_loss = F.mse_loss(pred_v.float(), target_v)
                z1_hat = (zt + (1.0 - tau).view(B, 1, 1, 1) * pred_v.float()).clamp(-5.0, 5.0)
                y16_hat = project_latent_y16_lcn(z1_hat, w_luma, b_luma)
                w_tau = (tau > 0.25).float()
                st_loss = (((y16_hat - y16).abs().mean(dim=(1, 2, 3))) * w_tau).sum() / w_tau.sum().clamp_min(1.0)
                loss = (fm_loss + 0.10 * st_loss) / accum
            loss.backward()
            fm_v += float(fm_loss.item()) / accum
            st_v += float(st_loss.item()) / accum

        nn.utils.clip_grad_norm_(new_params + dec_params, 1.0)
        opt.step()
        model.zero_onehot()  # 不变量: onehot 列永远为零
        with torch.no_grad():
            for pe, pw in zip(ema.parameters(), model.parameters()):
                pe.mul_(0.996).add_(pw, alpha=0.004)
        if step == 1 or step % 250 == 0 or step == updates:
            print(f"[{arm}] {step:4d}/{updates} fm={fm_v:.4f} y16={st_v:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if smoke and step >= 100:
            break

    if not smoke:
        torch.save({"ema": ema.state_dict(), "updates": updates, "arm": arm,
                    "use_router": use_router, "ab_prob": ab_prob}, out_p)
        print(f"[{arm}] Saved {out_p} in {(time.time()-t0)/60:.1f} min", flush=True)
    return str(out_p)


@torch.no_grad()
def eval_v25_decqr(arm: str, ckpt_p: str, bundle: dict, c21: dict,
                   tasks: list[dict], z_src_750: torch.Tensor, device: str = "cuda") -> dict:
    """100% 免标签推理评测: D5 (ws=4.2 & ws=6.5) + D6H 三工作点 × SG-5 + Holdout."""
    from exp.v20.submode_patchswd import build_or_load_core_knn_bank
    from src.utils import load_s10_decoder
    from src.utils.protocol import EVAL_SEED

    bank = build_or_load_core_knn_bank(bundle, k_sub=8, core_per_style=180, device=device)
    mid_dsd = float(bank["mid_dsd"])
    t750 = c21["test750"]
    sg5 = c21["sg5"]
    ho21 = c21["holdout"]
    ho_z = bundle["v18_cache"]["holdout"]["z"]
    d16_750 = bundle["test_struct"]["dino_16"]
    y16_750 = t750["y16"]
    P_Y = bundle["fnd"]["priors"]["P_Y"].to(device).float()
    P_C = bundle["fnd"]["priors"]["P_C"].to(device).float()
    eps_750 = torch.randn(z_src_750.shape, generator=torch.Generator().manual_seed(20261119))

    ck = torch.load(ckpt_p, map_location=device, weights_only=False)
    model = V25DecQRNet(use_router=ck.get("use_router", True)).to(device).eval()
    model.load_state_dict(ck["ema"], strict=True)
    dec = load_s10_decoder(device)

    specs = [
        (f"{arm}_SG5_d5w65", "sg5", dict(D5_SOLVER, tau_min=0.34, tau_hi=0.68, ws_fig=6.5), False),
        (f"{arm}_SG5_d6h", "sg5", dict(D6H_SOLVER, ws_fig=4.4), False),
        (f"{arm}_Holdout_d5w65", "ho", dict(D5_SOLVER, tau_min=0.34, tau_hi=0.68, ws_fig=6.5), False),
        (f"{arm}_Holdout_wrong", "ho", dict(D5_SOLVER, tau_min=0.34, tau_hi=0.68, ws_fig=6.5), True),
    ]
    out = {}
    gate = GateConfig(mode="original")
    for arm_name, ref_mode, solver, is_wrong in specs:
        if (OUT / f"raw/{arm_name}/meta.json").exists() and (OUT / f"measured/{arm_name}/summary_v27.json").exists():
            s = read_json(OUT / f"measured/{arm_name}/summary_v27.json")
        else:
            wr = RawWriterV27(arm_name, dec=dec, device=device)
            for st in range(0, 750, 25):
                sl = slice(st, min(st + 25, 750))
                grp = tasks[sl]
                B = len(grp)
                zs = z_src_750[sl].to(device).float()
                ep = eps_750[sl].to(device).float()
                y16 = y16_750[sl].to(device).float()
                d16 = d16_750[sl].to(device).float()
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
                    u_ref = sg5["u"][tid_cpu].to(device).float()
                    dsd_r = sg5["dsd"][tid_cpu].to(device).float()
                    k_tok = sg5["q32"][tid_cpu].float().flatten(2).transpose(1, 2).to(device)
                    v_tok = sg5["vref"][tid_cpu].float().to(device)
                else:
                    hi = ho21["single_idx"][tid_cpu]
                    z_r1 = ho_z[hi].to(device).float()
                    u_ref = ho21["u"][hi].to(device).float()
                    dsd_r = ho21["dsd"][hi].to(device).float()
                    k_tok = ho21["q32"][hi].float().flatten(2).transpose(1, 2).to(device)
                    v_tok = ho21["vref"][hi].float().to(device)

                # 100% 免标签热启动: blur 弱化源笔触 + 色度矩对齐单张参考图
                z_blur = F.avg_pool2d(F.pad(zs, (1, 1, 1, 1), mode="replicate"), 3, 1)
                z_anc = chroma_align_to_exemplar(z_blur, z_r1, P_Y, P_C, alpha=0.55)
                g_tau, g_layout, g_guid, _ = effective_gate_components(dsd_r, gate, mid_dsd)
                tau_v = (solver["tau_min"] + (solver["tau_hi"] - solver["tau_min"]) * g_tau).clamp(0.0, 0.78)
                lscale = (solver["lscale_abs"] + (1.0 - solver["lscale_abs"]) * g_layout).clamp(0.15, 1.0)
                c8_w = (1.0 - g_layout).clamp(0.0, 0.85)
                y_in = (1.0 - c8_w.view(B, 1, 1, 1)) * y16 + c8_w.view(B, 1, 1, 1) * F.avg_pool2d(y16, 2, 2).repeat_interleave(2, -2).repeat_interleave(2, -1)
                d_in = (1.0 - c8_w.view(B, 1, 1, 1)) * d16 + c8_w.view(B, 1, 1, 1) * F.avg_pool2d(d16, 2, 2).repeat_interleave(2, -2).repeat_interleave(2, -1)

                z = (1.0 - tau_v.view(B, 1, 1, 1)) * ep + tau_v.view(B, 1, 1, 1) * z_anc
                steps = int(solver["steps"])
                dt = (1.0 - tau_v).view(B, 1, 1, 1) / float(steps)
                f0 = torch.zeros(B, dtype=torch.bool, device=device)
                f1 = torch.ones(B, dtype=torch.bool, device=device)
                gf4 = g_guid.view(B, 1, 1, 1)

                def _vel(zz, prog):
                    tv = tau_v + prog * (1.0 - tau_v)
                    with autocast(device):
                        v_sc, v_0c = model(
                            zz.repeat(2, 1, 1, 1), y_in.repeat(2, 1, 1, 1), d_in.repeat(2, 1, 1, 1),
                            u_ref.repeat(2, 1), tv.repeat(2),
                            drop_style=torch.cat([f0, f1], 0), drop_content=torch.cat([f0, f0], 0),
                            layout_scale=lscale.repeat(2),
                            q32_src=q32.repeat(2, 1, 1, 1),
                            k_ref_tok=k_tok.repeat(2, 1, 1),
                            v_ref_tok=v_tok.repeat(2, 1, 1),
                        ).float().chunk(2, 0)
                    gl = min(1.0, max(0.0, (prog - 0.36) / 0.32))
                    ws_j = gf4 * (solver["ws_fig"] - 1.1 * gl) + (1.0 - gf4) * 1.4
                    return v_0c + ws_j * (v_sc - v_0c)

                for i in range(steps):
                    v1 = _vel(z, i / float(steps))
                    zm = z + 0.5 * dt * v1
                    v2 = _vel(zm, (i + 0.5) / float(steps))
                    z = z + dt * v2

                z_out = torch.where(same.view(B, 1, 1, 1), zs, z)
                wr.write_batch(z_out, grp)
            wr.finalize(arm=arm_name, ref_mode=ref_mode, solver=solver)
            s = run_measure_v27(arm_name, device=device)

        c = s["cross600"]
        five = {k: round(v["csd_top1"], 4) for k, v in s.get("per_target_cross120", {}).items()}
        out[arm_name] = {"top1": round(c["csd_top1"], 4), "dino_c": round(c["dino_c"], 4),
                         "dino_s": round(c["dino_s"], 4), "lpips": round(c["lpips"], 4),
                         "per_target": five}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--updates", type=int, default=1500)
    ap.add_argument("--arms", default="DQR,CTL")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    from exp.v18 import tasks_and_sources
    from exp.v21.stiefel_ref_cache import build_or_load_v21_cache
    from src.utils.runtime import sanitize_cuda_env

    sanitize_cuda_env()
    OUT.mkdir(parents=True, exist_ok=True)
    pack = build_or_load_v21_cache(args.pool, device=args.device)
    bundle, c21 = pack["bundle"], pack["c21"]
    tasks, z_src_750, _ = tasks_and_sources()

    if args.smoke:
        train_v25_decqr("SMOKE", bundle, c21, updates=100, device=args.device, use_router=True, smoke=True)
        print("[V25_DecQR] SMOKE_OK", flush=True)
        return

    verdict = {"task": "V25_Tr_T1 + Dec-QR-AdaGN (valid DINO-C>=0.61)", "arms": {}, "updated_utc": utc()}
    for arm in args.arms.split(","):
        use_router = (arm == "DQR")
        ckpt = train_v25_decqr(arm, bundle, c21, updates=args.updates, device=args.device, use_router=use_router)
        verdict["arms"][arm] = eval_v25_decqr(arm, ckpt, bundle, c21, tasks, z_src_750, device=args.device)
        save_json(OUT / "v25_decqr_verdict.json", verdict)
    print(f"[V25_DecQR] DONE → {OUT / 'v25_decqr_verdict.json'}", flush=True)


if __name__ == "__main__":
    main()
