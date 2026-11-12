"""Structured channel pruning of the A0 Decoder with weight inheritance.

Architecture fact (measured): the diffusers Decoder has exactly 4 channel streams,
because block widths are uniform inside each up_block:

    stream A : mid_block out == up_blocks[0] in/out   (A0: 512)
    stream B : up_blocks[1] out                       (A0: 512)
    stream C : up_blocks[2] out                       (A0: 256)
    stream D : up_blocks[3] out                       (A0: 128)

`block_out_channels=[c0,c1,c2,c3]` is FORWARD order, the decoder reverses it, so
    c3 -> stream A (mid + ub0), c2 -> stream B (ub1), c1 -> stream C (ub2), c0 -> stream D (ub3)

Every conv/norm/linear is therefore assigned (in_stream, out_stream); selecting ONE
index set per stream keeps residual shortcuts, GroupNorm and attention consistent.
Channel importance uses first-order Taylor |sum(W * dW)| accumulated per stream.

Verification: building with the A0 widths must reproduce A0 outputs bit-exactly.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/Workspace/WEAVE_VAE")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL
from diffusers.models.autoencoders.vae import Decoder

A0_DIR = "/root/Workspace/WEAVE_VAE/runs_vae/A0/final_diffusers_vae"
OUT = Path("/root/Workspace/WEAVE_VAE/runs/vae_decoder_prune")
OUT.mkdir(parents=True, exist_ok=True)
MANIFEST = Path("/root/Workspace/data/wikiart_512_raw/wikiart_512_raw_manifest.json")

A0_WIDTHS = [128, 256, 512, 512]
A0_SRC = {"A": 512, "B": 512, "C": 256, "D": 128}


# ---------------------------------------------------------------- architecture plan
def make_plan(dec):
    """List of (path, kind, in_stream, out_stream). kind in {conv,norm,linear}."""
    plan = [("conv_in", "conv", None, "A")]

    # mid block: operates on stream A
    for j, r in enumerate(dec.mid_block.resnets):
        plan += [(f"mid_block.resnets.{j}.norm1", "norm", "A", None),
                 (f"mid_block.resnets.{j}.conv1", "conv", "A", "A"),
                 (f"mid_block.resnets.{j}.norm2", "norm", "A", None),
                 (f"mid_block.resnets.{j}.conv2", "conv", "A", "A")]
        if getattr(r, "conv_shortcut", None) is not None:
            plan.append((f"mid_block.resnets.{j}.conv_shortcut", "conv", "A", "A"))

    if getattr(dec.mid_block, "attentions", None):
        plan += [("mid_block.attentions.0.group_norm", "norm", "A", None),
                 ("mid_block.attentions.0.to_q", "linear", "A", "A"),
                 ("mid_block.attentions.0.to_k", "linear", "A", "A"),
                 ("mid_block.attentions.0.to_v", "linear", "A", "A"),
                 ("mid_block.attentions.0.to_out.0", "linear", "A", "A")]

    streams_in = ["A", "A", "B", "C"]
    streams_out = ["A", "B", "C", "D"]
    for i, blk in enumerate(dec.up_blocks):
        si, so = streams_in[i], streams_out[i]
        for j, r in enumerate(blk.resnets):
            ji = si if j == 0 else so
            plan += [(f"up_blocks.{i}.resnets.{j}.norm1", "norm", ji, None),
                     (f"up_blocks.{i}.resnets.{j}.conv1", "conv", ji, so),
                     (f"up_blocks.{i}.resnets.{j}.norm2", "norm", so, None),
                     (f"up_blocks.{i}.resnets.{j}.conv2", "conv", so, so)]
            if getattr(r, "conv_shortcut", None) is not None:
                plan.append((f"up_blocks.{i}.resnets.{j}.conv_shortcut", "conv", ji, so))
        if getattr(blk, "upsamplers", None):
            plan.append((f"up_blocks.{i}.upsamplers.0.conv", "conv", so, so))

    plan += [("conv_norm_out", "norm", "D", None),
             ("conv_out", "conv", "D", None)]
    return plan


def get_module(root, path):
    m = root
    for part in path.split("."):
        m = m[int(part)] if part.isdigit() else getattr(m, part)
    return m


class StudentDecoder(nn.Module):
    """post_quant_conv + decoder, matching AutoencoderKL.decode()'s input contract.

    CRITICAL: AutoencoderKL.decode(z) runs post_quant_conv(z) BEFORE the decoder.
    Calling decoder(z) directly (as an earlier version of these scripts did) feeds
    an unprojected latent and produces garbage -- A0 itself scores PSNR ~5 that way.
    """

    def __init__(self, post_quant_conv, decoder):
        super().__init__()
        self.post_quant_conv = post_quant_conv
        self.decoder = decoder

    def forward(self, z):
        return self.decoder(self.post_quant_conv(z))


def build_decoder(widths, layers_per_block=2, norm_groups=32):
    return Decoder(in_channels=4, out_channels=3,
                   up_block_types=["UpDecoderBlock2D"] * 4,
                   block_out_channels=widths, layers_per_block=layers_per_block,
                   norm_num_groups=norm_groups, act_fn="silu",
                   mid_block_add_attention=True)


# ---------------------------------------------------------------- importance
def taylor_importance(dec, plan, calib_latents, target_images, device,
                      n_batches=8, bs=2, pre=None):
    """Accumulate |sum(W*dW)| per output channel, grouped by stream.

    `pre` must be the post_quant_conv module: AutoencoderKL.decode() applies it
    BEFORE the decoder, so feeding a raw latent straight into `dec` is wrong.
    Calibration loss = Charbonnier + multi-scale Haar high-freq (per spec).
    """
    acc = {s: torch.zeros(A0_SRC[s], device=device) for s in A0_SRC}
    dec = dec.to(device).train()
    try:
        dec.enable_gradient_checkpointing()
    except Exception:
        pass
    torch.cuda.empty_cache()

    def haar(x, levels=3):
        outs = []
        for _ in range(levels):
            a = x[:, :, 0::2, 0::2]; b = x[:, :, 1::2, 0::2]
            c = x[:, :, 0::2, 1::2]; d = x[:, :, 1::2, 1::2]
            ll = (a + b + c + d) * 0.5
            lh = (a - b + c - d) * 0.5
            hl = (a + b - c - d) * 0.5
            hh = (a - b - c + d) * 0.5
            outs.append((lh, hl, hh))
            x = ll
        return outs

    n = min(len(calib_latents), n_batches * bs)
    for bi in range(0, n, bs):
        z = calib_latents[bi:bi + bs].to(device)
        x = target_images[bi:bi + bs].to(device)
        if pre is not None:
            z = pre(z)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = dec(z)
            diff = out - x
            l_ch = torch.sqrt(diff.float() * diff.float() + 1e-6).mean()
            l_hf = 0.0
            for s_out, s_tgt in zip(haar(out.float()), haar(x.float())):
                for a, b in zip(s_out, s_tgt):
                    l_hf = l_hf + (a - b).abs().mean() / (b.abs().mean().detach() + 1e-6)
            loss = l_ch + 0.5 * l_hf
        dec.zero_grad(set_to_none=True)
        loss.backward()

        for path, kind, si, so in plan:
            if kind == "conv":
                m = get_module(dec, path)
                if m.weight.grad is not None and so is not None:
                    acc[so] += (m.weight * m.weight.grad).sum(dim=(1, 2, 3)).abs().detach()
            elif kind == "norm":
                m = get_module(dec, path)
                if m.weight.grad is not None and si is not None:
                    acc[si] += (m.weight * m.weight.grad).abs().detach()
        del z, x, out, diff, loss
        torch.cuda.empty_cache()
        if (bi // bs) % 8 == 0:
            print(f"    importance batch {bi//bs+1}/{(n+bs-1)//bs}", flush=True)
    dec.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    return {k: v.cpu() for k, v in acc.items()}


# ---------------------------------------------------------------- prune + inherit
def prune_decoder(dec_a0, widths, importance=None, seed=0):
    """Rebuild at target widths and copy weights along the selected indices."""
    tgt = {"A": widths[3], "B": widths[2], "C": widths[1], "D": widths[0]}
    g = torch.Generator().manual_seed(seed)
    idx = {}
    for s in A0_SRC:
        k, n = tgt[s], A0_SRC[s]
        if k >= n:
            idx[s] = torch.arange(n)
        elif importance is None:
            idx[s] = torch.sort(torch.randperm(n, generator=g)[:k]).values
        else:
            imp = importance[s]
            sel = torch.topk(imp, k, largest=True).indices
            idx[s] = torch.sort(sel).values

    plan = make_plan(dec_a0)
    new = build_decoder(widths)

    with torch.no_grad():
        for path, kind, si, so in plan:
            mo = get_module(dec_a0, path)
            mn = get_module(new, path)
            io = idx[so] if so is not None else None
            ii = idx[si] if si is not None else None
            # an index set selects from the SOURCE stream, so compare against A0_SRC
            io_u = io if (io is not None and mo.weight.shape[0] == A0_SRC[so]) else None
            if kind == "conv":
                w = mo.weight
                ii_u = ii if (ii is not None and w.shape[1] == A0_SRC[si]) else None
                wn = w
                if io_u is not None:
                    wn = wn[io_u]
                if ii_u is not None:
                    wn = wn[:, ii_u]
                mn.weight.copy_(wn)
                if mo.bias is not None:
                    mn.bias.copy_(mo.bias if io_u is None else mo.bias[io_u])
            elif kind == "norm":
                w = mo.weight
                ii_u = ii if (ii is not None and w.numel() == A0_SRC[si]) else None
                mn.weight.copy_(w if ii_u is None else w[ii_u])
                mn.bias.copy_(mo.bias if ii_u is None else mo.bias[ii_u])
            elif kind == "linear":
                w = mo.weight
                ii_u = ii if (ii is not None and w.shape[1] == A0_SRC[si]) else None
                wn = w
                if io_u is not None:
                    wn = wn[io_u]
                if ii_u is not None:
                    wn = wn[:, ii_u]
                mn.weight.copy_(wn)
                if mo.bias is not None:
                    mn.bias.copy_(mo.bias if io_u is None else mo.bias[io_u])

    meta = {
        "widths_forward": widths,
        "stream_target": tgt,
        "stream_source": A0_SRC,
        "channel_index": {s: idx[s].tolist() for s in idx},
        "layers_per_block": 2,
        "norm_num_groups": 32,
        "mid_block_add_attention": True,
        "params": sum(p.numel() for p in new.parameters()),
    }
    return new, meta


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="verify", choices=["verify", "build"])
    ap.add_argument("--widths", default="96,160,224,224")
    ap.add_argument("--calib", type=int, default=256)
    ap.add_argument("--name", default="decoder_p3_10m")
    args = ap.parse_args()

    device = "cuda"
    vae = AutoencoderKL.from_pretrained(A0_DIR, torch_dtype=torch.float32).eval()
    dec_a0 = vae.decoder
    a0_params = sum(p.numel() for p in dec_a0.parameters())
    print(f"[A0] decoder params {a0_params/1e6:.3f} M", flush=True)

    # --- correctness check: full width must reproduce A0 exactly ---
    print("\n=== VERIFY: full-width pruning must match A0 bit-exactly ===")
    new_id, meta_id = prune_decoder(dec_a0, A0_WIDTHS, importance=None)
    new_id.eval().to(device)
    z = torch.randn(2, 4, 64, 64, device=device)
    with torch.no_grad():
        y_a0 = dec_a0.to(device).eval()(z)
        y_id = new_id(z)
    d = (y_a0 - y_id).abs().max().item()
    print(f"  max|A0 - identity_prune| = {d:.3e}  -> "
          f"{'PASS' if d < 1e-5 else 'FAIL'}", flush=True)
    if d >= 1e-5:
        print("  ABORT: pruning mapping is wrong", flush=True)
        return
    del new_id
    torch.cuda.empty_cache()

    if args.mode == "verify":
        for w in [[128, 224, 352, 384], [96, 192, 256, 320], [96, 160, 224, 224],
                  [64, 128, 224, 224]]:
            m, _ = prune_decoder(dec_a0, w, importance=None)
            print(f"  widths={w} -> {sum(p.numel() for p in m.parameters())/1e6:.3f} M "
                  f"({sum(p.numel() for p in m.parameters())/a0_params*100:.1f}% of A0)")
            del m
        return

    # --- build: Taylor importance on real art images ---
    widths = [int(x) for x in args.widths.split(",")]
    print(f"\n=== BUILD {args.name} widths={widths} ===", flush=True)

    man = json.loads(MANIFEST.read_text())
    mmap = np.memmap(man["bin_path"], dtype=np.uint8, mode="r", shape=tuple(man["shape"]))
    idx = torch.load("/root/Workspace/WEAVE_VAE/runs/vae_style_rank20/index.pt",
                     weights_only=False)
    pool = idx["pos_list"]
    rng = np.random.default_rng(0)
    sel = rng.choice(len(pool), args.calib, replace=False)
    raws = [pool[i] for i in sel]
    print(f"[Calib] {len(raws)} art images from WikiArt train pool", flush=True)

    zs, xs = [], []
    with torch.no_grad():
        for i in range(0, len(raws), 8):
            arr = np.stack([np.ascontiguousarray(mmap[r]) for r in raws[i:i + 8]])
            x = torch.from_numpy(arr).permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0)
            zs.append(vae.encode(x.to(device)).latent_dist.mean.float().cpu())
            xs.append(x)
    Z = torch.cat(zs); X = torch.cat(xs)
    print(f"[Calib] latents {tuple(Z.shape)}", flush=True)

    imp = taylor_importance(dec_a0, make_plan(dec_a0), Z, X, device)
    for s in imp:
        v = imp[s]
        print(f"  stream {s}: n={v.numel()} imp_mean={v.mean():.3e} "
              f"max={v.max():.3e} min={v.min():.3e}", flush=True)

    new, meta = prune_decoder(dec_a0, widths, importance=imp)
    print(f"[Pruned] {meta['params']/1e6:.3f} M "
          f"({meta['params']/a0_params*100:.1f}% of A0)", flush=True)

    # sanity: output still correlates with A0 (not identical, but not garbage)
    new.eval().to(device)
    with torch.no_grad():
        y0 = dec_a0(z[:2])
        y1 = new(z[:2])
    cos = F.cosine_similarity(y0.flatten(), y1.flatten(), dim=0).item()
    print(f"[Sanity] cos(A0_out, pruned_out) before any recovery = {cos:.4f}", flush=True)

    d = OUT / args.name
    d.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": new.state_dict(), "meta": meta}, d / "decoder.pt")
    (d / "prune_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[Saved] {d}/decoder.pt + prune_meta.json", flush=True)


if __name__ == "__main__":
    main()
