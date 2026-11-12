"""Search Decoder width configs hitting target param counts (multiples of 32)."""
import sys
from itertools import product

sys.path.insert(0, "/root/Workspace/WEAVE_VAE")
import torch
from diffusers.models.autoencoders.vae import Decoder


def params(widths, layers=2):
    d = Decoder(in_channels=4, out_channels=3,
                up_block_types=["UpDecoderBlock2D"] * 4,
                block_out_channels=widths, layers_per_block=layers,
                norm_num_groups=32, act_fn="silu", mid_block_add_attention=True)
    return sum(p.numel() for p in d.parameters())


def main():
    print("A0 [128,256,512,512] =", f"{params([128,256,512,512])/1e6:.3f} M")
    print()
    # block_out_channels is forward order; decoder uses reversed:
    #   c3 = mid + ub0, c2 = ub1, c1 = ub2, c0 = ub3
    targets = [30e6, 18e6, 10e6]
    cands = [32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 384, 448, 512]
    best = {t: None for t in targets}

    # c0 <= c1 <= c2 <= c3 (widths must be non-increasing towards output)
    grid = [(c0, c1, c2, c3)
            for c0 in cands for c1 in cands for c2 in cands for c3 in cands
            if c0 <= c1 <= c2 <= c3]
    print(f"searching {len(grid)} configs...")
    for w in grid:
        try:
            p = params(list(w))
        except Exception:
            continue
        for t in targets:
            if best[t] is None or abs(p - t) < abs(best[t][1] - t):
                best[t] = (w, p)
    print()
    for t in targets:
        w, p = best[t]
        print(f"target {t/1e6:>4.0f}M -> widths {list(w)} = {p/1e6:.3f} M "
              f"(err {(p-t)/1e6:+.3f} M)")

    print("\n=== a few named candidates ===")
    for w in [[96, 160, 256, 256], [64, 128, 224, 224], [96, 160, 224, 224],
              [96, 192, 256, 320], [128, 192, 256, 384], [128, 224, 320, 384],
              [128, 256, 384, 448], [96, 128, 192, 192], [64, 96, 160, 160]]:
        try:
            print(f"  {w} -> {params(w)/1e6:.3f} M")
        except Exception as e:
            print(f"  {w} -> ERR {e}")


if __name__ == "__main__":
    main()
