"""Measure A0 Decoder: per-module params, MACs at 64x64 latent, and pruning targets.

Also probes torch_pruning availability. Nothing is pruned here yet.
"""
import json
import sys
from collections import OrderedDict

sys.path.insert(0, "/root/Workspace/WEAVE_VAE")

import torch
import torch.nn as nn
from diffusers import AutoencoderKL

A0_DIR = "/root/Workspace/WEAVE_VAE/runs_vae/A0/final_diffusers_vae"


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def macs_conv(m, shape):
    """shape = (C_in, H, W) input; returns MACs for this module (recursive)."""
    tot = 0
    for mod in m.modules():
        if isinstance(mod, nn.Conv2d):
            # find the spatial size this conv runs at by walking forward is complex;
            # approximate with hooks instead (see below)
            pass
    return tot


def main():
    print("=== environment ===")
    try:
        import torch_pruning as tp
        print(f"torch_pruning: {tp.__version__}")
    except Exception as e:
        print(f"torch_pruning: NOT AVAILABLE ({type(e).__name__})")
    print(f"torch: {torch.__version__}")

    vae = AutoencoderKL.from_pretrained(A0_DIR, torch_dtype=torch.float32).eval()
    dec = vae.decoder

    total = count_params(dec)
    print(f"\n=== A0 Decoder total params: {total/1e6:.3f} M ===")

    # per top-level module
    rows = []
    for name, mod in dec.named_children():
        rows.append((name, count_params(mod)))
    print(f"\n{'module':<20}{'params':>14}{'share':>8}")
    for n, c in rows:
        print(f"{n:<20}{c:>14,}{c/total*100:>7.2f}%")

    # per up_block
    print(f"\n=== up_blocks detail ===")
    for i, blk in enumerate(dec.up_blocks):
        print(f"  up_blocks[{i}]: {count_params(blk):>12,}  "
              f"out_ch={getattr(blk, 'out_channels', '?')}  "
              f"in_ch={getattr(blk, 'in_channels', '?')}  "
              f"n_resnets={len(blk.resnets)}")

    # per resnet inside block 0 (to see the shape)
    print(f"\n=== up_blocks[0].resnets detail ===")
    for j, r in enumerate(dec.up_blocks[0].resnets):
        print(f"  resnets[{j}]: {count_params(r):>12,}  "
              f"in={r.in_channels} out={r.out_channels}")

    # MACs via hooks at 64x64 latent
    print(f"\n=== MACs at latent 64x64 -> image 512x512 ===")
    mac = {"total": 0}
    hooks = []

    def mk_hook(name):
        def h(mod, inp, out):
            if isinstance(mod, nn.Conv2d):
                o = out.shape
                m = o[0] * o[1] * o[2] * o[3] * mod.in_channels * mod.kernel_size[0] * mod.kernel_size[1] / mod.groups
                mac["total"] += int(m)
                mac[name] = mac.get(name, 0) + int(m)
            elif isinstance(mod, nn.Linear):
                mac["total"] += int(inp[0].numel() * mod.out_features)
        return h

    for name, mod in dec.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            hooks.append(mod.register_forward_hook(mk_hook(name)))

    z = torch.randn(1, 4, 64, 64)
    with torch.no_grad():
        out = dec(z)
    for h in hooks:
        h.remove()
    print(f"  output shape: {tuple(out.shape)}")
    print(f"  total MACs : {mac['total']/1e9:.3f} GMACs "
          f"({mac['total']*2/1e9:.3f} GFLOPs) per image")

    top = sorted([(k, v) for k, v in mac.items() if k != "total"],
                 key=lambda x: -x[1])[:12]
    print(f"\n  {'module':<42}{'MACs (M)':>12}{'share':>8}")
    for k, v in top:
        print(f"  {k:<42}{v/1e6:>12.1f}{v/mac['total']*100:>7.2f}%")

    # proposed widths from the spec
    print(f"\n=== proposed widths (spec) ===")
    cfg = {
        "conv_in": (4, 256),
        "mid_block": 256,
        "up_blocks[0]": 256,
        "up_blocks[1]": 256,
        "up_blocks[2]": 160,
        "up_blocks[3]": 96,
        "conv_out": (128, 3),
    }
    print(json.dumps(cfg, indent=2))
    print("\n  NOTE: exact param count requires real pruning; measured below by")
    print("        rebuilding with reduced block_out_channels.")

    # quick estimate: rebuild decoder with reduced widths
    from diffusers.models.autoencoders.vae import Decoder
    for widths in ([128, 256, 512, 512], [128, 256, 256, 512], [128, 160, 256, 256],
                   [96, 160, 256, 256]):
        try:
            d = Decoder(
                in_channels=4, out_channels=3,
                up_block_types=["UpDecoderBlock2D"] * 4,
                block_out_channels=widths,
                layers_per_block=2, norm_num_groups=32, act_fn="silu",
                mid_block_add_attention=True,
            )
            print(f"  block_out_channels={widths} -> {count_params(d)/1e6:.3f} M")
        except Exception as e:
            print(f"  block_out_channels={widths} -> ERROR {e}")


if __name__ == "__main__":
    main()
