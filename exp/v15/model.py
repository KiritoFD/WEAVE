"""V15 variable-capacity U-Net with optional attention and spatial conditioning.

Three independent factors (plan §4):
  Capacity: base_dim ∈ {16, 32, 64, 96}
  Context:  conv-only vs +window-attention(32×32) + global-attention(16×16)
  Where:    V6 objective only vs V6 + where self-supervision losses

Style conditioning:
  R0: global AdaLN (baseline)
  R1: Spatial FiLM — modulation depends on local features + global context
  R2: Spatial Router — K local operation experts with content-dependent routing
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as Fn


# ============================ building blocks ============================
class DSConv(nn.Module):
    def __init__(self, cin, cout, k=3):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, k, padding=k // 2, groups=cin, bias=False)
        self.pw = nn.Conv2d(cin, cout, 1, bias=False)
        self.norm = nn.GroupNorm(min(8, cout), cout)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.pw(self.dw(x))))


class FiLM(nn.Module):
    def __init__(self, cond_dim, ch):
        super().__init__()
        self.fc = nn.Linear(cond_dim, ch * 2)

    def forward(self, x, cond):
        scale, shift = self.fc(cond).chunk(2, dim=-1)
        return x * (1 + scale[:, :, None, None]) + shift[:, :, None, None]


class ResBlock(nn.Module):
    def __init__(self, ch, cond_dim):
        super().__init__()
        self.c1 = DSConv(ch, ch)
        self.film = FiLM(cond_dim, ch)
        self.c2 = DSConv(ch, ch)

    def forward(self, x, cond):
        return x + self.c2(self.film(self.c1(x), cond))


# ============================ attention blocks ============================
class WindowAttention(nn.Module):
    """Non-overlapping window multi-head self-attention (Swin-style, no shift)."""

    def __init__(self, dim, window_size=8, num_heads=4):
        super().__init__()
        self.dim = dim
        self.win = window_size
        self.heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """x: (B, C, H, W) → (B, C, H, W)."""
        B, C, H, W = x.shape
        ws = self.win
        # pad to multiple of window size
        ph = (ws - H % ws) % ws
        pw = (ws - W % ws) % ws
        if ph or pw:
            x = Fn.pad(x, (0, pw, 0, ph))
        Hp, Wp = H + ph, W + pw
        # reshape to windows: (B, H/ws, W/ws, ws, ws, C)
        xw = x.reshape(B, C, Hp // ws, ws, Wp // ws, ws)
        xw = xw.permute(0, 2, 4, 3, 5, 1).reshape(-1, ws * ws, C)
        xw = self.norm(xw)
        qkv = self.qkv(xw).reshape(-1, ws * ws, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.softmax(dim=-1)
        out = (att @ v).reshape(-1, ws * ws, self.heads * self.head_dim)
        out = self.proj(out)
        # reshape back
        n_win = out.shape[0]
        out = out.reshape(B, Hp // ws, Wp // ws, ws, ws, C)
        out = out.permute(0, 5, 1, 3, 2, 4).reshape(B, C, Hp, Wp)
        if ph or pw:
            out = out[:, :, :H, :W]
        return out


class GlobalAttention(nn.Module):
    """Standard multi-head self-attention on the full spatial grid."""

    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.head_dim = dim // num_heads

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)          # (B, HW, C)
        x_norm = self.norm(x_flat)
        out, _ = self.attn(x_norm, x_norm, x_norm)
        return x + out.transpose(1, 2).reshape(B, C, H, W)


class SpatialFiLM(nn.Module):
    """R1: spatial-conditioned modulation (gamma/beta depend on local + global)."""

    def __init__(self, ch, cond_dim):
        super().__init__()
        self.local_fc = nn.Conv2d(ch, ch * 2, 1)
        self.global_fc = nn.Linear(ch + cond_dim, ch * 2)
        self.norm = nn.GroupNorm(min(8, ch), ch)

    def forward(self, x, global_cond):
        B, C, H, W = x.shape
        xn = self.norm(x)
        local_gamma, local_beta = self.local_fc(xn).chunk(2, dim=1)
        g_feat = Fn.adaptive_avg_pool2d(x, 1).flatten(1)
        g_input = torch.cat([g_feat, global_cond], dim=1) if global_cond is not None else g_feat
        g_gamma, g_beta = self.global_fc(g_input).chunk(2, dim=-1)
        gamma = (1 + local_gamma) * (1 + g_gamma[:, :, None, None]) - 1
        beta = local_beta + g_beta[:, :, None, None]
        return xn * gamma + beta


class SpatialRouter(nn.Module):
    """R2: K local operation experts with content-dependent routing."""

    def __init__(self, ch, cond_dim, k_experts=4):
        super().__init__()
        self.k = k_experts
        self.experts = nn.ModuleList([DSConv(ch, ch) for _ in range(k_experts)])
        self.router_conv = nn.Conv2d(ch, ch // 4, 1)
        self.router_fc = nn.Linear(ch // 4 + cond_dim, k_experts)
        self.norm = nn.GroupNorm(min(8, ch), ch)

    def forward(self, x, global_cond):
        xn = self.norm(x)
        expert_outs = torch.stack([e(xn) for e in self.experts], dim=1)  # (B,K,C,H,W)
        rg = self.router_conv(xn)                                        # (B,ch/4,H,W)
        g = Fn.adaptive_avg_pool2d(rg, 1).flatten(1)                     # (B,ch/4)
        if global_cond is not None:
            g = torch.cat([g, global_cond], dim=1)
        pi = torch.softmax(self.router_fc(g), dim=-1)                    # (B,K)
        pi = pi[:, :, None, None, None]
        return (expert_outs * pi).sum(dim=1)


# ============================ main U-Net ============================
class V15UNet(nn.Module):
    """Variable-capacity U-Net with optional attention and spatial conditioning.

    Args:
        base_dim: channel width (16/32/64/96) — capacity factor
        use_attention: if True, add window attention (32×32) + global attention (16×16)
        style_mode: "adaln" (R0 baseline) | "spatial_film" (R1) | "router" (R2)
        n_experts: number of router experts (for R2)
    """

    N_STYLE = 5

    def __init__(self, base_dim=64, cond_dim=128, use_attention=False,
                 style_mode="adaln", n_experts=4, levels=3):
        super().__init__()
        self.base_dim = base_dim
        self.use_attention = use_attention
        self.style_mode = style_mode
        ws = [base_dim * (2 ** i) for i in range(levels)]
        cond_ch = cond_dim

        # conditioning network: style one-hot + tau
        self.cond_net = nn.Sequential(
            nn.Linear(self.N_STYLE + 2, cond_dim), nn.SiLU(True),
            nn.Linear(cond_dim, cond_dim), nn.SiLU(True))

        # encoder: 64→32→16
        self.enc_in = DSConv(4, ws[0])
        self.down = nn.ModuleList()
        self.enc = nn.ModuleList()
        for i in range(levels - 1):
            self.down.append(nn.Conv2d(ws[i], ws[i + 1], 3, stride=2,
                                       padding=1, bias=False))
            self.enc.append(ResBlock(ws[i + 1], cond_ch))

        # attention (if enabled)
        self.win_att = WindowAttention(ws[1], window_size=8) if use_attention else None
        self.glob_att = GlobalAttention(ws[-1]) if use_attention else None

        # mid
        self.mid = ResBlock(ws[-1], cond_ch)

        # decoder: 16→32→64
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        self.res_dec = nn.ModuleList()
        for i in range(levels - 1, 0, -1):
            self.up.append(nn.ConvTranspose2d(ws[i], ws[i - 1], 2, stride=2,
                                              bias=False))
            self.dec.append(DSConv(ws[i - 1] * 2, ws[i - 1]))
            if style_mode == "spatial_film":
                self.res_dec.append(SpatialFiLM(ws[i - 1], cond_ch))
            elif style_mode == "router":
                self.res_dec.append(SpatialRouter(ws[i - 1], cond_ch, n_experts))
            else:
                self.res_dec.append(ResBlock(ws[i - 1], cond_ch))

        self.out = nn.Conv2d(ws[0], 4, 3, padding=1)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def cond(self, style_id, tau, unconditional=False):
        B = style_id.shape[0]
        dev = style_id.device
        if unconditional:
            oh = torch.zeros(B, self.N_STYLE, device=dev)
        else:
            oh = Fn.one_hot(style_id.clamp(0, self.N_STYLE - 1),
                            self.N_STYLE).float()
        tin = torch.stack([tau, torch.full_like(tau, float(unconditional))], dim=-1)
        return self.cond_net(torch.cat([oh, tin], dim=-1))

    def forward(self, z_tau, style_id, tau, unconditional=False):
        """z_tau: (B,4,64,64) noised latent; returns predicted velocity (B,4,64,64)."""
        cond = self.cond(style_id, tau, unconditional)
        # encoder
        skips = [self.enc_in(z_tau)]
        h = skips[0]
        for i, d in enumerate(self.down):
            h = self.enc[i](d(h), cond)
            skips.append(h)
            # attention at 32×32 (after first down) and 16×16 (after second down)
            if self.win_att is not None and i == 0:
                h = self.win_att(h)
        if self.glob_att is not None:
            h = self.glob_att(h)
        h = self.mid(h, cond)
        # decoder with optional spatial conditioning
        for i, (up, dec, res) in enumerate(zip(self.up, self.dec, self.res_dec)):
            h = up(h)
            h = torch.cat([h, skips[-(i + 2)]], dim=1)
            h = dec(h)
            if isinstance(res, (SpatialFiLM, SpatialRouter)):
                h = res(h, cond)
            else:
                h = res(h, cond)
        return self.out(h)
