"""A: Generator-Coordinate Editing — continuous control coordinates for C3UNet.

Idea
----
Q2a/B4 already emit a content-constrained painting transform z0 = G(c,t). The
Oracle's strength is choosing a direction per instance; its cost is walking
outside that transform family in raw 4x64x64 space. So instead of editing z
directly, edit a LOW-DIMENSIONAL control u and let the trained generator carry
it:

    z(u) = G(c, t; u),   z(0) = z0,   u in R^d, d << 16384

The gradient then arrives as grad_u F = J_u^T grad_z F, i.e. the classifier
direction is filtered through the tangent space the generator can actually
express.

How it is wired, and why this way
---------------------------------
`tools/v6/models_v6.py` is left UNTOUCHED. Controls are injected by wrapping the
forward pass, because:

  * every existing checkpoint must keep loading with no key changes;
  * `controls=None` must be bit-identical to the original forward, and wrapping
    makes that testable in one place;
  * a future control variant is added here, not by editing the base network.

Where each control enters (all additive, all zero-initialised):
  u_global  -> the shared cond vector used by every StyleAdaLN block
  u64/u32/u16/uout -> per-scale cond for enc64 / enc32 / mid16 / decoder+out
  u_palette -> sample-level offsets on the palette head's 4x4 LL mix, LL bias,
               HF scale and HF bias (the low-freq colour / HF energy knobs)
  u_local   -> small-rank injection into the 32x32 decoder feature (A4 only)

Only the AdaLN/Palette paths are touched; the content injection path
(`content_proj`, 64/32/16) is deliberately left frozen, since that is the
mechanism V6 relies on for content preservation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

COND_DIM = 32          # base_dim * 2 for the Q2a/B4 models (base_dim = 16)
PALETTE_OFFSETS = 16 + 4 + 12 + 12   # mix_ll 4x4, bias_ll, scale_hf, bias_hf


@dataclass
class ControlSpec:
    """Which control groups are active and how wide each one is."""
    global_dim: int = COND_DIM
    per_scale_dim: int = COND_DIM
    scales: Sequence[str] = ("64", "32", "16", "out")
    palette: bool = True
    local_rank: int = 0          # 0 disables A4's spatial control

    @property
    def dim(self) -> int:
        d = self.global_dim
        d += self.per_scale_dim * len(self.scales)
        d += PALETTE_OFFSETS if self.palette else 0
        d += self.local_rank * 2 if self.local_rank else 0
        return d

    def describe(self) -> dict:
        return {"global": self.global_dim,
                "per_scale": {s: self.per_scale_dim for s in self.scales},
                "palette": PALETTE_OFFSETS if self.palette else 0,
                "local_rank": self.local_rank, "total": self.dim}


class Controls:
    """A per-sample control vector, split into the groups ControlSpec declares.

    Deliberately NOT an nn.Module: the control is an optimisation VARIABLE, not
    a trained parameter, and wrapping it in a Parameter broke the gradient path
    in the first version. `tensor` is the raw control with requires_grad set by
    the caller.
    """

    def __init__(self, tensor: torch.Tensor, spec: ControlSpec):
        self.spec = spec
        self.tensor = tensor

    @property
    def raw(self) -> torch.Tensor:
        return self.tensor

    @classmethod
    def zeros(cls, batch: int, device, spec: ControlSpec | None = None,
              requires_grad: bool = True):
        spec = spec or ControlSpec()
        t = torch.zeros(batch, spec.dim, device=device, requires_grad=requires_grad)
        return cls(t, spec)

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, spec: ControlSpec):
        if tensor.shape[-1] != spec.dim:
            raise ValueError(f"control width {tensor.shape[-1]} != spec.dim {spec.dim}")
        return cls(tensor, spec)

    def groups(self):
        s = self.spec
        i = 0
        out = {}
        out["global"] = self.tensor[:, i:i + s.global_dim]; i += s.global_dim
        for name in s.scales:
            out[f"u{name}"] = self.tensor[:, i:i + s.per_scale_dim]; i += s.per_scale_dim
        if s.palette:
            out["palette"] = self.tensor[:, i:i + PALETTE_OFFSETS]; i += PALETTE_OFFSETS
        if s.local_rank:
            out["local"] = self.tensor[:, i:i + 2 * s.local_rank]; i += 2 * s.local_rank
        if i != s.dim:
            raise RuntimeError("control split does not consume the whole vector")
        return out


class ControlledGenerator(nn.Module):
    """Wrap a frozen C3UNet so its forward accepts an optional `controls`.

    With controls absent or exactly zero this must equal the wrapped model's own
    output; `assert_zero_control_equivalence` checks that.
    """

    def __init__(self, base: nn.Module, spec: ControlSpec | None = None,
                 device: str = "cuda"):
        super().__init__()
        self.base = base
        self.spec = spec or ControlSpec()
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.base.eval()
        # Sample-level palette offsets, built once and added to whatever the
        # style-ID lookup produced. Shape (1, ...) so it broadcasts per sample.
        self.register_buffer("_unused", torch.zeros(1), persistent=False)

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _palette_delta(palette_raw: torch.Tensor):
        """Split the 32 palette scalars into the head's four parameter shapes.

        Layout per sample: mix_ll 4x4 needs 16 values (a per-sample 4x4 mixing
        matrix is the low-frequency colour knob), then LL bias 4, HF scale 12,
        HF bias 12. Earlier versions sliced 4/4/12/12 and then tried to view 4
        values as a 4x4 matrix.
        """
        mix = palette_raw[:, 0:16].view(-1, 4, 4)
        bias = palette_raw[:, 16:20].view(-1, 4, 1, 1)
        scale = palette_raw[:, 20:32].view(-1, 12, 1, 1)
        hfbias = palette_raw[:, 32:44].view(-1, 12, 1, 1)
        return mix, bias, scale, hfbias

    def forward(self, c: torch.Tensor, tau, z_in: torch.Tensor,
                style_id: torch.Tensor, controls: Controls | None = None):
        """Mirror C3UNet.forward's calling convention: (z_in, tau, c, style_id).

        The public WEAVE call sites use `G(c, 0.0, c, tid)`, so we keep that
        order and forward straight through.
        """
        base = self.base
        # Fast path for an exactly-zero control. This MUST NOT be taken when the
        # control requires grad: at u=0 the control has no effect on the output,
        # so returning base() directly severs grad_u and the optimiser sees no
        # gradient at all. That silently broke the first A-route smoke test.
        if controls is None:
            return base(z_in, tau, c, style_id)
        if not controls.raw.requires_grad and bool(controls.raw.detach().abs().max() == 0):
            return base(z_in, tau, c, style_id)
        return self._controlled_forward(base, z_in, tau, c, style_id, controls)

    def _controlled_forward(self, base, z_in, tau, c, style_id, controls):
        """Re-implementation of C3UNet.forward with additive control terms.

        Kept deliberately close to the original so a reviewer can diff them; the
        only insertions are `+ u` terms and the palette offsets.
        """
        B = z_in.shape[0]
        g = controls.groups()
        if not torch.is_tensor(tau):
            tau_t = torch.full((B,), float(tau), device=z_in.device, dtype=torch.float32)
        else:
            tau_t = tau.float().view(B)

        s_emb = base.style_mlp(base.style_emb(style_id.long()))
        if base.folded:
            cond = s_emb
        else:
            t_emb = base.time_mlp(
                sinusoidal_embedding(tau_t, base.base_dim).to(s_emb.dtype))
            cond = s_emb + t_emb
        cond = cond + g["global"]
        c64 = cond + g.get("u64", 0.0)
        c32 = cond + g.get("u32", 0.0)
        c16 = cond + g.get("u16", 0.0)
        cout = cond + g.get("uout", 0.0)

        c_ll, c_lh, c_hl, c_hh = dwt2_haar(c.float())
        c32c = torch.cat([c_ll, c_lh, c_hl, c_hh], dim=1).to(dtype=z_in.dtype)
        c16c = F.avg_pool2d(c32c, kernel_size=2, stride=2)

        x0 = z_in if base.folded else torch.cat([z_in, c, z_in - c], dim=1)
        h64 = base.stem64(x0)
        i64 = c if base.inject_64 else None
        i32 = c32c if base.inject_32 else None
        i16 = c16c if base.inject_16 else None
        h64 = base.enc64_1(h64, c64, i64)
        h64 = base.enc64_2(h64, c64, i64)

        h32 = base.down32(h64)
        h32 = base.enc32_1(h32, c32, i32)
        h32 = base.enc32_2(h32, c32, i32)

        h16 = base.down16(h32)
        h16 = base.mid16_1(h16, c16, i16)
        h16 = base.mid16_2(h16, c16, i16)
        if base.skr is not None:
            h16 = base.skr(h16, style_id)

        u32 = F.interpolate(h16, scale_factor=2.0, mode="nearest")
        u32 = base.up32(u32)
        u32 = base.dec32_proj(torch.cat([u32, h32], dim=1))
        u32 = base.dec32_1(u32, c32, i32)
        u32 = base.dec32_2(u32, c32, i32)

        if "local" in g and g["local"] is not None:
            u32 = u32 + self._local_injection(g["local"], u32)

        u64 = F.interpolate(u32, scale_factor=2.0, mode="nearest")
        u64 = base.up64(u64)
        u64 = base.dec64_proj(torch.cat([u64, h64], dim=1))
        u64 = base.dec64_1(u64, c64, i64)
        u64 = base.dec64_2(u64, c64, i64)

        go, bo = base.out_adaln(cout).chunk(2, dim=-1)
        u_out = F.silu(base.out_norm(u64) * (1.0 + go[:, :, None, None])
                       + bo[:, :, None, None])
        delta_spatial = base.out_conv64(u_out)

        dwt_raw = base.out_dwt32(F.silu(u32)).float()
        d_ll, d_lh, d_hl, d_hh = dwt_raw.chunk(4, dim=1)
        delta_dwt = idwt2_haar(d_ll, d_lh, d_hl, d_hh).to(dtype=z_in.dtype)

        if "palette" in g and g["palette"] is not None:
            delta_pal = self._palette_forward(base, c, style_id, g["palette"])
        else:
            delta_pal = base.palette_head(c, style_id)

        total = delta_spatial + delta_dwt + delta_pal
        return c + total if base.mode == "direct" else total

    @staticmethod
    def _palette_forward(base, c, style_id, palette_raw):
        """StyleChannelPaletteHead with per-sample additive offsets.

        The head's learned per-style parameters stay the base; the control only
        adds an offset, and the offset is per sample, so no batch sample can
        contaminate another.
        """
        head = base.palette_head
        ll, lh, hl, hh = dwt2_haar(c.float())
        dt = head.mix_ll.dtype
        ll, lh, hl, hh = ll.to(dt), lh.to(dt), hl.to(dt), hh.to(dt)
        B, C, H, W = ll.shape
        sid = style_id.long()
        mix, bias, scale, hfbias = ControlledGenerator._palette_delta(palette_raw)
        w_ll = head.mix_ll[sid] + mix.to(dt)
        b_ll = head.bias_ll[sid] + bias.to(dt)
        ll_flat = ll.reshape(B, C, H * W)
        ll_delta = torch.bmm(w_ll, ll_flat).reshape(B, C, H, W) + b_ll
        hf = torch.cat([lh, hl, hh], dim=1)
        s_hf = head.scale_hf[sid] + scale.to(dt)
        b_hf = head.bias_hf[sid] + hfbias.to(dt)
        hf_delta = hf * s_hf + b_hf
        d_lh, d_hl, d_hh = hf_delta.chunk(3, dim=1)
        return idwt2_haar(ll_delta, d_lh, d_hl, d_hh).to(dtype=c.dtype)

    @staticmethod
    def _local_injection(local_raw: torch.Tensor, feature: torch.Tensor):
        """A4: small-rank spatial control on the decoder feature.

        Rank r means r learned low-frequency bases scaled per sample; the result
        is mean-removed and broadcast over channels, so it stays an internal
        modulation rather than a free residual written to z.

        The bases must all share the feature's (H, W); the first version built
        (1,1,H,1) and (1,1,1,W) and tried to concatenate them on dim=1.
        """
        B, C, H, W = feature.shape
        rank = local_raw.shape[1] // 2
        if rank == 0:
            return 0.0
        amps = local_raw[:, :rank]                        # (B,r)
        gains = local_raw[:, rank:]                       # (B,r)
        ys = torch.linspace(-1, 1, H, device=feature.device).view(1, 1, H, 1)
        xs = torch.linspace(-1, 1, W, device=feature.device).view(1, 1, 1, W)
        ones_h = torch.ones(1, 1, H, 1, device=feature.device)
        ones_w = torch.ones(1, 1, 1, W, device=feature.device)
        # every base is (1,1,H,W)
        bases = torch.cat([ys * ones_w, ones_h * xs, ys.square() * ones_w,
                           ones_h * xs.square()], dim=1)   # (1,4,H,W)
        bases = bases[:, :rank]                            # (1,r,H,W)
        mixed = (amps.view(B, rank, 1, 1) * bases).sum(1, keepdim=True)   # (B,1,H,W)
        # 1 + tanh: at zero init the gain factor is 1, not 0 — otherwise
        # d(out)/d(amps) = bases*tanh(0) = 0 and d(out)/d(gains) ~ mixed = 0,
        # a deadlock that keeps the local branch untrainable. u=0 still gives
        # amps=0 -> out=0, so zero control reproduces the base exactly.
        scale = (1.0 + torch.tanh(gains.mean(1, keepdim=True))).view(B, 1, 1, 1)
        out = (mixed * scale).expand(-1, C, -1, -1)
        return out - out.mean(dim=(2, 3), keepdim=True)


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    import math
    half = dim // 2
    freqs = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32)
                      * (-math.log(10000.0) / max(half - 1, 1)))
    args = t.float().view(-1, 1) * freqs.view(1, -1) * 2.0 * math.pi
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return F.pad(emb, (0, 1)) if dim % 2 == 1 else emb


def dwt2_haar(x):
    from wavelet import dwt2_haar as _d
    return _d(x)


def idwt2_haar(ll, lh, hl, hh):
    from wavelet import idwt2_haar as _i
    return _i(ll, lh, hl, hh)


def wrap(base: nn.Module, spec: ControlSpec | None = None,
         device: str = "cuda") -> ControlledGenerator:
    return ControlledGenerator(base, spec, device).to(device)


def assert_zero_control_equivalence(gen: ControlledGenerator, c, tau, z_in, style_id,
                                    *, atol: float = 0.0, rtol: float = 0.0):
    """The contract that matters: u=0 must reproduce the checkpoint exactly."""
    with torch.no_grad():
        ref = gen.base(z_in, tau, c, style_id)
        ctrl = Controls.zeros(z_in.shape[0], z_in.device, gen.spec)
        got = gen(c, tau, z_in, style_id, controls=ctrl)
    if not torch.allclose(ref, got, atol=atol, rtol=rtol):
        raise AssertionError(
            f"u=0 diverges from the base forward: max|d|={float((ref-got).abs().max())}")
    return float((ref - got).abs().max())
