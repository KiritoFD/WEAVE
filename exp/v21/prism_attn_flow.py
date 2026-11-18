"""Zero-Class-ID Stiefel-Quotient Exemplar Cross-Attention Flow (PRISMAttnUNet).

100% class-agnostic generative transport:
  - Structure input : Stiefel Polar Quotient Q_32(x_s) [0% 1st & 2nd-order Gram style leak!] + coarse Y_16 / smooth Y_32
  - Style input     : Arbitrary reference image(s) x_ref -> global fiber u_ref in S^127 + spatial tokens (K_sem, V_style)
                      Supports 1-Shot single reference (L=256) and Multi-Shot gallery (L=1024) identically!
"""
from __future__ import annotations

import copy
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import OUT, autocast, seed_all
from .stiefel_ref_cache import coarse_y16_lcn
from exp.v18.ccf_model import BottleneckAttention, SPADEStyleBlock, sinusoidal_time_emb
from exp.v20.slf_flow import PureStyleResBlock


class QuotientRefCrossAttention(nn.Module):
    """FlashAttention2 Cross-Attention matching Stiefel Quotient Queries Q(x_s) to Exemplar Keys K(x_ref) -> V_style(x_ref)."""

    def __init__(self, ch: int, q_sem_dim: int = 32, v_ref_dim: int = 192, heads: int = 8):
        super().__init__()
        self.heads = heads
        self.head_dim = ch // heads
        assert ch % heads == 0

        self.norm_x = nn.LayerNorm(ch)
        self.norm_k_v = nn.LayerNorm(v_ref_dim)

        self.to_q = nn.Linear(ch + q_sem_dim, ch)
        self.to_k = nn.Linear(q_sem_dim + v_ref_dim, ch)
        # Deliberately UN-NORMALIZED value projection so exact palette & stroke contrast of x_ref survive!
        self.to_v = nn.Linear(v_ref_dim, ch)
        self.to_out = nn.Linear(ch, ch)
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(
        self,
        x: torch.Tensor,
        q_src_map: torch.Tensor,
        k_ref_tok: torch.Tensor,
        v_ref_tok: torch.Tensor,
        smask: torch.Tensor,
    ) -> torch.Tensor:
        """
        x         : (B, C, H, W)
        q_src_map : (B, 32, H, W)
        k_ref_tok : (B, L, 32)
        v_ref_tok : (B, L, 192)
        smask     : (B, 1) in {0, 1}
        """
        B, C, H, W = x.shape
        L = k_ref_tok.shape[1]
        xf = x.flatten(2).transpose(1, 2)            # (B, HW, C)
        qf = q_src_map.flatten(2).transpose(1, 2)    # (B, HW, 32)

        q_in = torch.cat([self.norm_x(xf), qf], dim=-1)
        k_in = torch.cat([k_ref_tok, self.norm_k_v(v_ref_tok)], dim=-1)

        q = self.to_q(q_in).view(B, H * W, self.heads, self.head_dim).transpose(1, 2)
        k = self.to_k(k_in).view(B, L, self.heads, self.head_dim).transpose(1, 2)
        v = self.to_v(v_ref_tok).view(B, L, self.heads, self.head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v)  # (B, heads, HW, head_dim)
        out = self.to_out(attn_out.transpose(1, 2).reshape(B, H * W, C))
        out_map = out.transpose(1, 2).reshape(B, C, H, W)
        return x + out_map * smask.view(B, 1, 1, 1)


class PRISMAttnUNet(nn.Module):
    """Zero-Class-ID Stiefel-Quotient Exemplar Cross-Attention Flow U-Net (~24.6M params)."""

    def __init__(self, base_dim: int = 144, q_dim: int = 32, u_dim: int = 128, v_dim: int = 192, cond_dim: int = 288):
        super().__init__()
        w0, w1, w2 = base_dim, base_dim * 2, base_dim * 3
        q_proj_ch = 64

        self.q_proj = nn.Sequential(
            nn.Conv2d(q_dim, 96, 1),
            nn.GroupNorm(16, 96),
            nn.SiLU(inplace=True),
            nn.Conv2d(96, q_proj_ch, 1),
        )
        # sin_time(64) + u_ref(128) + mean_vref(192) + flags(2) = 386 (ZERO class one-hot!)
        self.cond_mlp = nn.Sequential(
            nn.Linear(64 + u_dim + v_dim + 2, cond_dim),
            nn.SiLU(inplace=True),
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(inplace=True),
        )

        self.in_conv = nn.Conv2d(4, w0, 3, padding=1)
        self.enc64 = nn.ModuleList([PureStyleResBlock(w0, cond_dim), PureStyleResBlock(w0, cond_dim)])

        self.down32 = nn.Conv2d(w0, w1, 3, stride=2, padding=1)
        self.enc32 = nn.ModuleList([PureStyleResBlock(w1, cond_dim), PureStyleResBlock(w1, cond_dim)])

        self.down16 = nn.Conv2d(w1, w2, 3, stride=2, padding=1)
        s16_ch = 1 + q_proj_ch
        self.mid1 = SPADEStyleBlock(w2, s16_ch, cond_dim)
        self.mid_self_attn = BottleneckAttention(w2, heads=8)
        self.mid_ref_attn = QuotientRefCrossAttention(w2, q_sem_dim=q_dim, v_ref_dim=v_dim, heads=8)
        self.mid2 = SPADEStyleBlock(w2, s16_ch, cond_dim)

        self.up32 = nn.Conv2d(w2, w1, 3, padding=1)
        self.skip32 = nn.Conv2d(w1 * 2, w1, 1)
        # Smooth 32x32 shading anchor (post-GroupNorm lscale!) + 32x32 Exemplar Cross-Attention for crisp brushwork!
        self.dec32_spade = SPADEStyleBlock(w1, 1, cond_dim)
        self.dec32_ref_attn = QuotientRefCrossAttention(w1, q_sem_dim=q_dim, v_ref_dim=v_dim, heads=8)
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

    def forward(
        self,
        z_tau: torch.Tensor,
        y16_lcn: torch.Tensor,
        y32_smooth: torch.Tensor,
        q32_src: torch.Tensor,
        u_ref: torch.Tensor,
        k_ref_tok: torch.Tensor,
        v_ref_tok: torch.Tensor,
        tau: torch.Tensor,
        *,
        drop_style: torch.Tensor | None = None,
        drop_content: torch.Tensor | None = None,
        layout_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B = z_tau.shape[0]
        dev = z_tau.device
        if drop_style is None:
            drop_style = torch.zeros(B, dtype=torch.bool, device=dev)
        if drop_content is None:
            drop_content = torch.zeros(B, dtype=torch.bool, device=dev)

        cmask4 = (~drop_content).float().view(B, 1, 1, 1)
        smask2 = (~drop_style).float().view(B, 1)
        smask3 = (~drop_style).float().view(B, 1, 1)

        if layout_scale is None:
            lscale = (~drop_content).float()
        else:
            lscale = layout_scale.to(dev).float().view(B).clamp(0.0, 1.0) * (~drop_content).float()
        lscale4 = lscale.view(B, 1, 1, 1)

        # Post-GroupNorm layout scaling!
        qf16 = self.q_proj(q32_src * cmask4) * lscale4
        s16 = torch.cat([y16_lcn * lscale4, qf16], dim=1)
        s32 = y32_smooth * (lscale4 * 0.85)

        q_src_16 = q32_src * lscale4
        q_src_32 = F.interpolate(q_src_16.float(), size=(32, 32), mode="bilinear", align_corners=False).to(z_tau.dtype)

        k_tok = k_ref_tok * smask3
        v_tok = v_ref_tok * smask3
        v_mean = v_tok.mean(dim=1)
        uv = u_ref * smask2

        temb = sinusoidal_time_emb(tau, 64)
        flags = torch.stack([drop_style.float(), 1.0 - lscale], dim=-1)
        cond = self.cond_mlp(torch.cat([temb, uv, v_mean, flags], dim=-1))

        h64 = self.in_conv(z_tau)
        for blk in self.enc64:
            h64 = blk(h64, cond)

        h32 = self.down32(h64)
        for blk in self.enc32:
            h32 = blk(h32, cond)

        h16 = self.down16(h32)
        h16 = self.mid1(h16, s16, cond)
        h16 = self.mid_self_attn(h16)
        h16 = self.mid_ref_attn(h16, q_src_16, k_tok, v_tok, smask2)
        h16 = self.mid2(h16, s16, cond)

        u32 = self.up32(h16.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1))
        u32 = self.skip32(torch.cat([u32, h32], dim=1))
        u32 = self.dec32_spade(u32, s32, cond)
        u32 = self.dec32_ref_attn(u32, q_src_32, k_tok, v_tok, smask2)
        for blk in self.dec32:
            u32 = blk(u32, cond)

        u64 = self.up64(u32.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1))
        u64 = self.skip64(torch.cat([u64, h64], dim=1))
        for blk in self.dec64:
            u64 = blk(u64, cond)

        return self.out_conv(F.silu(self.out_norm(u64)))


def _project_latent_y16(z: torch.Tensor, w_luma: torch.Tensor, b_luma: torch.Tensor) -> torch.Tensor:
    w = w_luma.to(z.device).float().view(1, 4, 1, 1)
    b = b_luma.to(z.device).float()
    y64 = ((z.float() * w).sum(dim=1, keepdim=True) + b).clamp(0.0, 1.0)
    return coarse_y16_lcn(y64)


def train_prism_attn_flow(
    bundle: dict,
    c21: dict,
    updates: int = 3800,
    micro: int = 96,
    accum: int = 1,
    device: str = "cuda",
) -> Path:
    ckpt_dir = OUT / "attn_flow"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    final_p = ckpt_dir / "prism_attn_final.pt"
    if final_p.exists():
        print(f"[V21 Flow] Reusing {final_p}", flush=True)
        return final_p

    seed_all(20261128)
    tr_v18 = bundle["v18_cache"]["train"]
    tr21 = c21["train"]
    w_luma = bundle["v18_cache"]["luma_proj"]["w"].to(device).float()
    b_luma = bundle["v18_cache"]["luma_proj"]["b"].to(device).float()

    z_all = tr_v18["z"]
    sid_all = tr_v18["sid"].long()
    valid_m = tr21["valid_mask"]
    q32_all = tr21["q32"]                  # (4400, 32, 16, 16)
    y16_all = tr21["y16"]                  # (4400, 1, 16, 16)
    y32_all = tr21["y32"]                  # (4400, 1, 32, 32)
    vref_all = tr21["vref"]                # (4400, 256, 192)
    u_all = tr21["u"]                      # (4400, 128)
    nbr_idx = tr21["nbr_idx"]              # (4400, 12) DISTINCT Core-kNN neighbors!

    all_cls_idx = [((sid_all == s) & valid_m).nonzero(as_tuple=True)[0] for s in range(5)]
    core_cls_idx = [ci.long() for ci in tr21["core_idx_per_style"]]

    model = PRISMAttnUNet(base_dim=144).to(device)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(model.parameters(), lr=2.0e-4, betas=(0.9, 0.99), weight_decay=1e-4)
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

            z1 = z_all[idx].to(device, non_blocking=True).float()
            q32 = q32_all[idx].to(device, non_blocking=True).float()
            y16 = y16_all[idx].to(device, non_blocking=True).float()
            y32 = y32_all[idx].to(device, non_blocking=True).float()

            # 60% 1-Shot DISTINCT neighbor (L=256) + 40% 4-Shot DISTINCT neighbors (L=1024)!
            is_multishot = (step % 5 in (1, 3))
            n_shots = 4 if is_multishot else 1
            # Fix 5 (V22-P0): 4-Shot 参考无放回采样，严禁同一参考重复
            perm_nbr = sample_without_replacement(B, pool=12, n_shots=n_shots, device=device)
            r_ids = nbr_idx[idx[:, None], perm_nbr]  # (B, n_shots), GUARANTEED r_ids != idx!

            # Flatten (B, n_shots, 256, ...) -> (B, n_shots*256, ...) and randomly permute token order to prevent spatial copy!
            k_ref = q32_all[r_ids].float().flatten(3).transpose(2, 3).reshape(B, n_shots * 256, 32).to(device, non_blocking=True)
            v_ref = vref_all[r_ids].float().reshape(B, n_shots * 256, 192).to(device, non_blocking=True)
            t_perm = torch.randperm(n_shots * 256, device=device)
            k_ref = k_ref[:, t_perm, :]
            v_ref = v_ref[:, t_perm, :]
            u_ref = F.normalize(u_all[r_ids].float().mean(dim=1), dim=-1).to(device, non_blocking=True)

            q32_aug = q32 + 0.06 * torch.randn_like(q32)
            q32_aug = q32_aug - q32_aug.mean(dim=(2, 3), keepdim=True)
            y16_aug = (y16 * (0.85 + 0.30 * torch.rand(B, 1, 1, 1, device=device)) + 0.03 * torch.randn_like(y16)).clamp(-1.1, 1.1)
            y32_aug = (y32 * (0.85 + 0.30 * torch.rand(B, 1, 1, 1, device=device)) + 0.03 * torch.randn_like(y32)).clamp(-1.1, 1.1)

            u_r = torch.rand(B, device=device)
            drop_s = u_r < 0.14
            drop_c = (u_r >= 0.14) & (u_r < 0.34)
            both = u_r >= 0.95
            drop_s = drop_s | both
            drop_c = drop_c | both

            lscale = torch.where(drop_c, torch.zeros(B, device=device), 0.20 + 0.80 * torch.rand(B, device=device))

            z0 = torch.randn_like(z1)
            tau = torch.rand(B, device=device)
            zt = (1.0 - tau).view(B, 1, 1, 1) * z0 + tau.view(B, 1, 1, 1) * z1
            target_v = z1 - z0

            with autocast(device):
                pred_v = model(
                    zt, y16_aug, y32_aug, q32_aug, u_ref, k_ref, v_ref, tau,
                    drop_style=drop_s, drop_content=drop_c, layout_scale=lscale,
                )
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
            for pe_w, pw in zip(ema.parameters(), model.parameters()):
                pe_w.mul_(0.997).add_(pw, alpha=0.003)

        if step == 1 or step % 400 == 0 or step == updates:
            print(f"[V21 Flow] {step:4d}/{updates} fm={loss_fm_v:.4f} y16={loss_st_v:.4f} ({time.time()-t0:.0f}s)", flush=True)

    torch.save({"ema": ema.state_dict(), "updates": updates, "params": ema.n_params()}, final_p)
    print(f"[V21 Flow] Saved {final_p} ({ema.n_params()/1e6:.2f}M) in {(time.time()-t0)/60:.1f} min", flush=True)
    return final_p


@torch.no_grad()
def sample_without_replacement(batch: int, pool: int, n_shots: int, device: str = "cuda", generator=None) -> torch.Tensor:
    """Fix 5: 从 pool 个候选中为每个样本无放回抽取 n_shots 个（topk on random scores）。"""
    n = min(int(n_shots), int(pool))
    scores = torch.rand(batch, pool, device=device, generator=generator)
    return scores.topk(n, dim=1).indices


def chroma_align_to_exemplar(z_src: torch.Tensor, z_ref: torch.Tensor, P_Y: torch.Tensor, P_C: torch.Tensor, alpha: float = 0.55) -> torch.Tensor:
    """100% class-label-free orthogonal 3D chrominance warm-start from arbitrary reference z_ref!"""
    zy_s = torch.einsum("ij,bjhw->bihw", P_Y, z_src.float())
    zc_s = torch.einsum("ij,bjhw->bihw", P_C, z_src.float())
    zc_r = torch.einsum("ij,bjhw->bihw", P_C, z_ref.float())
    mu_s = zc_s.mean(dim=(2, 3), keepdim=True)
    sd_s = zc_s.std(dim=(2, 3), keepdim=True, unbiased=False).clamp_min(1e-4)
    mu_r = zc_r.mean(dim=(2, 3), keepdim=True)
    sd_r = zc_r.std(dim=(2, 3), keepdim=True, unbiased=False).clamp_min(1e-4)
    zc_al = torch.einsum("ij,bjhw->bihw", P_C, (zc_s - mu_s) / sd_s * sd_r + mu_r)
    return zy_s + (1.0 - alpha) * zc_s + alpha * zc_al


@torch.no_grad()
def integrate_prism_attn(
    model: PRISMAttnUNet,
    z_anchor: torch.Tensor,
    eps: torch.Tensor,
    y16: torch.Tensor,
    y32: torch.Tensor,
    q32_src: torch.Tensor,
    u_ref: torch.Tensor,
    k_ref: torch.Tensor,
    v_ref: torch.Tensor,
    dsd_ref: torch.Tensor,
    mid_dsd: float,
    *,
    tau_min: float = 0.0,
    tau_hi: float = 0.35,
    steps: int = 14,
    ws_fig: float = 4.2,
    wsp_fig_late: float = 2.8,
    wsp_abs: float = 5.4,
    wc_fig: float = 1.65,
    wc_abs: float = 0.65,
    lscale_fig: float = 0.95,
    lscale_abs: float = 0.25,
    rescale_phi: float = 0.50,
) -> torch.Tensor:
    """RK2 ODE integration conditioned ONLY on (x_s, x_ref) — zero class IDs!"""
    dev = z_anchor.device
    B = len(z_anchor)
    g_fig = torch.sigmoid((dsd_ref.to(dev).float() - mid_dsd) * 550.0)
    gf4 = g_fig.view(B, 1, 1, 1)

    lscale = (lscale_abs + (lscale_fig - lscale_abs) * g_fig).clamp(0.15, 1.0)
    tau_v = (tau_min + (tau_hi - tau_min) * g_fig).clamp(0.0, 0.78)
    z = (1.0 - tau_v.view(B, 1, 1, 1)) * eps + tau_v.view(B, 1, 1, 1) * z_anchor

    f0 = torch.zeros(B, dtype=torch.bool, device=dev)
    f1 = torch.ones(B, dtype=torch.bool, device=dev)
    ls0 = torch.zeros(B, device=dev)

    def _vel(z_cur: torch.Tensor, prog: float) -> torch.Tensor:
        tv = tau_v + prog * (1.0 - tau_v)
        with autocast(str(dev)):
            z4 = z_cur.repeat(4, 1, 1, 1)
            y16_4 = y16.repeat(4, 1, 1, 1)
            y32_4 = y32.repeat(4, 1, 1, 1)
            q4 = q32_src.repeat(4, 1, 1, 1)
            u4 = u_ref.repeat(4, 1)
            k4 = k_ref.repeat(4, 1, 1)
            v4 = v_ref.repeat(4, 1, 1)
            t4 = tv.repeat(4)
            ds = torch.cat([f0, f0, f1, f1], 0)
            dc = torch.cat([f0, f1, f0, f1], 0)
            lsc4 = torch.cat([lscale, ls0, lscale, ls0], 0)
            v_sc, v_s0, v_0c, v_00 = model(
                z4, y16_4, y32_4, q4, u4, k4, v4, t4,
                drop_style=ds, drop_content=dc, layout_scale=lsc4,
            ).float().chunk(4, 0)

        gl = min(1.0, max(0.0, (prog - 0.38) / 0.32))
        wc = gf4 * (wc_fig - 0.35 * gl) + (1.0 - gf4) * wc_abs
        ws_j = gf4 * (ws_fig - 0.9 * gl) + (1.0 - gf4) * 1.5
        ws_p = gf4 * (gl * wsp_fig_late) + (1.0 - gf4) * wsp_abs
        vr = v_00 + wc * (v_0c - v_00) + ws_j * (v_sc - v_0c) + ws_p * (v_s0 - v_00)
        if rescale_phi > 0.0:
            s_ref = (gf4 * v_sc + (1.0 - gf4) * v_s0).flatten(1).std(1).view(B, 1, 1, 1).clamp_min(1e-4)
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
