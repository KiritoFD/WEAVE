"""FLOPs-aware structured channel pruning for A0 AutoencoderKL.encoder with weight inheritance.

Exact forward mirror of tools/vae_suite/decoder_prune.py:
  stream A : conv_in out, down_blocks[0] (512->256px, c=128, 29.4% FLOPs!)
  stream B : down_blocks[1] (256->128px, c=256, 26.3% FLOPs!)
  stream C : down_blocks[2] (128->64px,  c=512, 26.3% FLOPs)
  stream D : down_blocks[3], mid_block, conv_norm_out, conv_out in (64px, c=512, 17.7% FLOPs)

Notice all widths should be multiples of 32 when preserving norm_num_groups=32.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from diffusers.models.autoencoders.vae import Encoder


A0_ENC_WIDTHS = [128, 256, 512, 512]
A0_ENC_SRC = {"A": 128, "B": 256, "C": 512, "D": 512}


class StudentEncoder(nn.Module):
    """Pruned A0 Encoder + sliced 4-ch quant_conv mean projection outputting scaled z_s."""

    def __init__(self, encoder: Encoder, qc_mean: nn.Conv2d, scaling_factor: float = 0.18215) -> None:
        super().__init__()
        self.encoder = encoder
        self.qc_mean = qc_mean
        self.scaling_factor = float(scaling_factor)

    def forward(self, rgb01: torch.Tensor) -> torch.Tensor:
        h = self.qc_mean(self.encoder(rgb01 * 2.0 - 1.0))
        return h * self.scaling_factor


def _get_module(root: nn.Module, path: str) -> nn.Module:
    m: Any = root
    for part in path.split("."):
        m = m[int(part)] if part.isdigit() else getattr(m, part)
    return m


def make_encoder_plan(enc: Encoder) -> list[tuple[str, str, str | None, str | None]]:
    """Return ordered list of (module_path, kind, in_stream, out_stream)."""
    plan: list[tuple[str, str, str | None, str | None]] = [("conv_in", "conv", None, "A")]

    streams_in = ["A", "A", "B", "C"]
    streams_out = ["A", "B", "C", "D"]
    for i, blk in enumerate(enc.down_blocks):
        si, so = streams_in[i], streams_out[i]
        for j, r in enumerate(blk.resnets):
            ji = si if j == 0 else so
            plan += [
                (f"down_blocks.{i}.resnets.{j}.norm1", "norm", ji, None),
                (f"down_blocks.{i}.resnets.{j}.conv1", "conv", ji, so),
                (f"down_blocks.{i}.resnets.{j}.norm2", "norm", so, None),
                (f"down_blocks.{i}.resnets.{j}.conv2", "conv", so, so),
            ]
            if getattr(r, "conv_shortcut", None) is not None:
                plan.append((f"down_blocks.{i}.resnets.{j}.conv_shortcut", "conv", ji, so))
        if getattr(blk, "downsamplers", None):
            plan.append((f"down_blocks.{i}.downsamplers.0.conv", "conv", so, so))

    if getattr(enc, "mid_block", None) is not None:
        # resnets.0 -> attentions -> resnets.1 all on stream D
        for j, r in enumerate(enc.mid_block.resnets):
            plan += [
                (f"mid_block.resnets.{j}.norm1", "norm", "D", None),
                (f"mid_block.resnets.{j}.conv1", "conv", "D", "D"),
                (f"mid_block.resnets.{j}.norm2", "norm", "D", None),
                (f"mid_block.resnets.{j}.conv2", "conv", "D", "D"),
            ]
            if getattr(r, "conv_shortcut", None) is not None:
                plan.append((f"mid_block.resnets.{j}.conv_shortcut", "conv", "D", "D"))
        if getattr(enc.mid_block, "attentions", None):
            plan += [
                ("mid_block.attentions.0.group_norm", "norm", "D", None),
                ("mid_block.attentions.0.to_q", "linear", "D", "D"),
                ("mid_block.attentions.0.to_k", "linear", "D", "D"),
                ("mid_block.attentions.0.to_v", "linear", "D", "D"),
                ("mid_block.attentions.0.to_out.0", "linear", "D", "D"),
            ]

    plan += [("conv_norm_out", "norm", "D", None), ("conv_out", "conv", "D", None)]
    return plan


def build_encoder(widths: list[int], layers_per_block: int = 2, norm_groups: int = 32) -> Encoder:
    """Construct a diffusers Encoder with given 4-stage channel widths."""
    return Encoder(
        in_channels=3,
        out_channels=4,
        down_block_types=("DownEncoderBlock2D",) * 4,
        block_out_channels=tuple(widths),
        layers_per_block=layers_per_block,
        norm_num_groups=norm_groups,
        act_fn="silu",
        double_z=True,
    )


def _haar_high(x: torch.Tensor) -> list[torch.Tensor]:
    a = x[:, :, 0::2, 0::2]
    b = x[:, :, 1::2, 0::2]
    c = x[:, :, 0::2, 1::2]
    d = x[:, :, 1::2, 1::2]
    return [(a - b + c - d) * 0.5, (a + b - c - d) * 0.5, (a - b - c + d) * 0.5]


def taylor_encoder_importance(
    a0: nn.Module,
    calib_rgb01: torch.Tensor,
    target_z_scaled: torch.Tensor,
    device: str = "cuda",
    bs: int = 4,
    scaling_factor: float = 0.18215,
    decoder: nn.Module | None = None,
) -> dict[str, torch.Tensor]:
    """Accumulate first-order Taylor |sum(W * dW)| per encoder stream using latent + Haar losses."""
    enc = a0.encoder.to(device).float().train().requires_grad_(True)
    qc_mean = nn.Conv2d(8, 4, kernel_size=1, bias=True).to(device).float()
    with torch.no_grad():
        qc_mean.weight.copy_(a0.quant_conv.weight[:4].float())
        qc_mean.bias.copy_(a0.quant_conv.bias[:4].float())
    qc_mean.requires_grad_(False)
    plan = make_encoder_plan(enc)
    acc = {s: torch.zeros(A0_ENC_SRC[s], device=device) for s in A0_ENC_SRC}

    for bi in range(0, calib_rgb01.shape[0], bs):
        x = calib_rgb01[bi : bi + bs].to(device)
        zt = target_z_scaled[bi : bi + bs].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            zs = qc_mean(enc(x * 2.0 - 1.0)).float() * scaling_factor
            l_lat = (zs - zt).abs().mean()
            l_hf = sum((a - b).abs().mean() for a, b in zip(_haar_high(zs), _haar_high(zt))) / 3.0
            loss = l_lat + 0.5 * l_hf
            if decoder is not None:
                # Ranking only: reconstruction provides a nonzero Taylor signal at
                # the intact teacher. Training itself still targets teacher latents.
                from src.utils.vae import decode_s10
                decoded = decode_s10(decoder, zs, enable_grad=True, output_cpu=False)
                loss = (decoded - x).abs().mean()
        enc.zero_grad(set_to_none=True)
        loss.backward()
        for path, kind, si, so in plan:
            m = _get_module(enc, path)
            if kind == "conv" and m.weight.grad is not None and so is not None and m.weight.shape[0] == A0_ENC_SRC[so]:
                acc[so] += (m.weight * m.weight.grad).sum(dim=(1, 2, 3)).abs().detach()
            elif kind == "norm" and m.weight.grad is not None and si is not None:
                acc[si] += (m.weight * m.weight.grad).abs().detach()
            elif kind == "linear" and m.weight.grad is not None and so is not None:
                acc[so] += (m.weight * m.weight.grad).sum(dim=1).abs().detach()

    enc.zero_grad(set_to_none=True)
    enc.eval().requires_grad_(False)
    return {k: v.cpu() for k, v in acc.items()}


def prune_encoder(
    a0: nn.Module,
    widths: list[int],
    importance: dict[str, torch.Tensor] | None = None,
    scaling_factor: float = 0.18215,
    norm_groups: int = 32,
) -> tuple[StudentEncoder, dict[str, Any]]:
    """Structurally prune AutoencoderKL.encoder + slice quant_conv mean head with exact weight inheritance."""
    enc_src = a0.encoder
    tgt = {"A": int(widths[0]), "B": int(widths[1]), "C": int(widths[2]), "D": int(widths[3])}
    idx: dict[str, torch.Tensor] = {}
    for s, n in A0_ENC_SRC.items():
        k = tgt[s]
        if k >= n:
            idx[s] = torch.arange(n)
        elif importance is None:
            idx[s] = torch.arange(k)
        else:
            sel = torch.topk(importance[s], k, largest=True).indices
            idx[s] = torch.sort(sel).values

    plan = make_encoder_plan(enc_src)
    new_enc = build_encoder(widths, norm_groups=norm_groups)

    with torch.no_grad():
        for path, kind, si, so in plan:
            mo = _get_module(enc_src, path)
            mn = _get_module(new_enc, path)
            io = idx[so] if so is not None else None
            ii = idx[si] if si is not None else None
            io_u = io if (io is not None and mo.weight.shape[0] == A0_ENC_SRC[so]) else None
            if kind == "conv":
                w = mo.weight.float()
                ii_u = ii if (ii is not None and w.shape[1] == A0_ENC_SRC[si]) else None
                wn = w[io_u] if io_u is not None else w
                if ii_u is not None:
                    wn = wn[:, ii_u]
                mn.weight.copy_(wn)
                if mo.bias is not None:
                    mn.bias.copy_(mo.bias.float() if io_u is None else mo.bias.float()[io_u])
            elif kind == "norm":
                w = mo.weight.float()
                ii_u = ii if (ii is not None and w.numel() == A0_ENC_SRC[si]) else None
                mn.weight.copy_(w if ii_u is None else w[ii_u])
                mn.bias.copy_(mo.bias.float() if ii_u is None else mo.bias.float()[ii_u])
            elif kind == "linear":
                w = mo.weight.float()
                ii_u = ii if (ii is not None and w.shape[1] == A0_ENC_SRC[si]) else None
                wn = w[io_u] if io_u is not None else w
                if ii_u is not None:
                    wn = wn[:, ii_u]
                mn.weight.copy_(wn)
                if mo.bias is not None:
                    mn.bias.copy_(mo.bias.float() if io_u is None else mo.bias.float()[io_u])

        qc_mean = nn.Conv2d(8, 4, kernel_size=1, bias=True)
        qc_mean.weight.copy_(a0.quant_conv.weight[:4].float())
        qc_mean.bias.copy_(a0.quant_conv.bias[:4].float())

    student = StudentEncoder(new_enc, qc_mean, scaling_factor=scaling_factor)
    meta = {
        "widths": widths,
        "norm_groups": norm_groups,
        "stream_target": tgt,
        "channel_index": {s: idx[s].tolist() for s in idx},
        "params": sum(p.numel() for p in student.parameters()),
    }
    return student, meta
