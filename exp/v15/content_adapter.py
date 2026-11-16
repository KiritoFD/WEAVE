"""P3/P4: zero-init Content Adapter for V15UNet.

forward(z_tau, style_id, tau, content, unconditional=False)
  content = canon(z_src_tilde_or_src)  (B,4,64,64) frozen c1_gentle view.

Adapter predicts two residual feature maps added to the frozen backbone:
  h64 (64x64, ws0 ch) and h32 (32x32, ws1 ch). Both heads are zero-initialized
  so the wrapped model is bit-identical to the base at init.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as Fn

from .model import V15UNet


class _ZeroInitConv(nn.Conv2d):
    def reset_parameters(self):
        nn.init.zeros_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


def _block(cin, cout, stride=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
        nn.SiLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.SiLU(inplace=True))


class ContentAdapter(nn.Module):
    """~0.2M params; outputs additive residual maps at 64x64 and 32x32."""

    def __init__(self, in_ch=4, width=32, ws0=96, ws1=192):
        super().__init__()
        # 64x64 branch
        self.b64 = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, padding=1, bias=False), nn.SiLU(),
            _block(width, width),
            _ZeroInitConv(width, ws0, 3, padding=1))
        # 32x32 branch (downsample first)
        self.b32 = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, stride=2, padding=1, bias=False),
            nn.SiLU(),
            _block(width, width * 2),
            _ZeroInitConv(width * 2, ws1, 3, padding=1))

    def forward(self, content):
        return self.b64(content), self.b32(content)


class V15AdapterUNet(nn.Module):
    """Frozen-or-trainable V15UNet + ContentAdapter residual injection."""

    def __init__(self, base: V15UNet, ws0=None, ws1=None):
        super().__init__()
        self.base = base
        ws = [base.base_dim * (2 ** i) for i in range(3)]
        self.adapter = ContentAdapter(ws0=ws[0], ws1=ws[1])
        self.N_STYLE = base.N_STYLE

    def cond(self, style_id, tau, unconditional=False):
        return self.base.cond(style_id, tau, unconditional)

    def set_base_requires_grad(self, flag):
        for p in self.base.parameters():
            p.requires_grad_(flag)

    def forward(self, z_tau, style_id, tau, content=None,
                unconditional=False):
        base = self.base
        cond = base.cond(style_id, tau, unconditional)
        skips = [base.enc_in(z_tau)]
        h = skips[0]
        if content is not None:
            a64, a32 = self.adapter(content)
            h = h + a64
        skips[0] = h
        for i, d in enumerate(base.down):
            h = base.enc[i](d(h), cond)
            if i == 0 and content is not None:
                h = h + a32
            skips.append(h)
            if base.win_att is not None and i == 0:
                h = base.win_att(h)
        if base.glob_att is not None:
            h = base.glob_att(h)
        h = base.mid(h, cond)
        for i, (up, dec, res) in enumerate(
                zip(base.up, base.dec, base.res_dec)):
            h = up(h)
            h = torch.cat([h, skips[-(i + 2)]], dim=1)
            h = dec(h)
            h = res(h, cond)
        return base.out(h)
