"""Style-conditioned wavelet detail flow model (V11-B).

Two lightweight residual UNets (one per band scale) predicting the
flow-matching velocity v = h_real - eps for wavelet detail coefficients.
Depthwise-separable convs, FiLM style/time conditioning, ~0.5-1M params total.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DSConv(nn.Module):
    """Depthwise-separable conv block."""

    def __init__(self, cin: int, cout: int, k: int = 3):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, k, padding=k // 2, groups=cin, bias=False)
        self.pw = nn.Conv2d(cin, cout, 1, bias=False)
        self.norm = nn.GroupNorm(8, cout)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.pw(self.dw(x))))


class FiLM(nn.Module):
    def __init__(self, cond_dim: int, ch: int):
        super().__init__()
        self.fc = nn.Linear(cond_dim, ch * 2)

    def forward(self, x, cond):
        scale, shift = self.fc(cond).chunk(2, dim=-1)
        return x * (1 + scale[:, :, None, None]) + shift[:, :, None, None]


class ResBlock(nn.Module):
    def __init__(self, ch: int, cond_dim: int):
        super().__init__()
        self.c1 = DSConv(ch, ch)
        self.film = FiLM(cond_dim, ch)
        self.c2 = DSConv(ch, ch)

    def forward(self, x, cond):
        h = self.c1(x)
        h = self.film(h, cond)
        return x + self.c2(h)


class WaveletUNet(nn.Module):
    """Small UNet for one band scale. in_ch = 9 (detail) + 3 (LL) + 1 (edge)."""

    def __init__(self, in_ch: int = 13, width: int = 32, cond_dim: int = 64,
                 levels: int = 3):
        super().__init__()
        ws = [width * (2 ** i) for i in range(levels)]
        self.enc_in = DSConv(in_ch, ws[0])
        self.down = nn.ModuleList()
        self.enc = nn.ModuleList()
        for i in range(levels - 1):
            self.down.append(nn.Conv2d(ws[i], ws[i + 1], 3, stride=2,
                                       padding=1, bias=False))
            self.enc.append(DSConv(ws[i + 1], ws[i + 1]))
        self.mid = ResBlock(ws[-1], cond_dim)
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        self.res = nn.ModuleList()
        for i in range(levels - 1, 0, -1):
            self.up.append(nn.ConvTranspose2d(ws[i], ws[i - 1], 2, stride=2,
                                              bias=False))
            # concat skip (ws[i-1]) + upsampled (ws[i-1]) -> fuse to ws[i-1]
            self.dec.append(DSConv(ws[i - 1] * 2, ws[i - 1]))
            self.res.append(ResBlock(ws[i - 1], cond_dim))
        self.out = nn.Conv2d(ws[0], 9, 1)

    def forward(self, x, cond):
        skips = [self.enc_in(x)]
        h = skips[0]
        for i, d in enumerate(self.down):
            h = self.enc[i](d(h))
            skips.append(h)
        h = self.mid(h, cond)
        for i in range(len(self.up)):
            h = self.up[i](h)
            h = torch.cat([h, skips[-(i + 2)]], dim=1)
            h = self.res[i](self.dec[i](h), cond)
        return self.out(h)


class WaveletFlow(nn.Module):
    """Two band-specific UNets + shared conditioning MLP.

    style_id: 0-4 or -1 (unconditional, after CFG dropout).
    tau: flow time in [0,1].
    """

    N_STYLE = 5

    def __init__(self, width: int = 32, cond_dim: int = 64):
        super().__init__()
        self.cond_net = nn.Sequential(
            nn.Linear(self.N_STYLE + 2, cond_dim), nn.SiLU(),
            nn.Linear(cond_dim, cond_dim), nn.SiLU())
        # H1 band lives at half-res of the crop (64x64 for 128 crop);
        # H2 at quarter-res (32x32). Same net weights, separate instances.
        self.net_h1 = WaveletUNet(in_ch=13, width=width, cond_dim=cond_dim,
                                  levels=3)
        self.net_h2 = WaveletUNet(in_ch=13, width=width, cond_dim=cond_dim,
                                  levels=3)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def cond(self, style_id: torch.Tensor, tau: torch.Tensor,
             unconditional: bool = False):
        """style_id [B] long, tau [B] float -> cond [B,cond_dim]."""
        B = style_id.shape[0]
        dev = style_id.device
        if unconditional:
            oh = torch.zeros(B, self.N_STYLE, device=dev)
        else:
            oh = F.one_hot(style_id.clamp(0, self.N_STYLE - 1),
                           self.N_STYLE).float()
        tin = torch.stack([tau, torch.full_like(tau, float(unconditional))],
                          dim=-1)
        return self.cond_net(torch.cat([oh, tin], dim=-1))

    def forward_band(self, which: str, h_tau: torch.Tensor, ll: torch.Tensor,
                     edge: torch.Tensor, style_id: torch.Tensor,
                     tau: torch.Tensor, unconditional: bool = False):
        """Predict velocity for one band. h_tau/ll/edge already at band res."""
        cond = self.cond(style_id, tau, unconditional)
        # ll is at LL2 res (quarter); upsample to band res
        if ll.shape[-2:] != h_tau.shape[-2:]:
            ll = F.interpolate(ll, size=h_tau.shape[-2:], mode="bilinear",
                               align_corners=False)
        if edge.shape[-2:] != h_tau.shape[-2:]:
            edge = F.interpolate(edge, size=h_tau.shape[-2:],
                                 mode="bilinear", align_corners=False)
        x = torch.cat([h_tau, ll, edge], dim=1)
        net = self.net_h1 if which == "H1" else self.net_h2
        return net(x, cond)
