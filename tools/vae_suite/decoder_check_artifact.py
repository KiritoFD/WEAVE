"""CPU sanity check of a pruned decoder artifact: loads it, verifies shapes,
counts params, and does one 512px forward on CPU to confirm it runs."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/root/Workspace/WEAVE_VAE")
sys.path.insert(0, "/root/Workspace/WEAVE_VAE/tools/vae_suite")

import torch
from tools.vae_suite.decoder_prune import build_decoder

OUT = Path("/root/Workspace/WEAVE_VAE/runs/vae_decoder_prune")


def main():
    torch.set_num_threads(32)
    for name in ["p1_28m"]:
        d = OUT / name
        f = d / "decoder.pt"
        if not f.exists():
            print(f"[{name}] missing {f}")
            continue
        t0 = time.time()
        ck = torch.load(f, map_location="cpu", weights_only=False)
        meta = ck["meta"]
        widths = meta["widths_forward"]
        print(f"\n=== {name} ===")
        print(f"  widths={widths}  params={meta['params']/1e6:.3f} M")
        print(f"  stream_target={meta['stream_target']}")
        idx_ok = all(len(v) == meta["stream_target"][s] for s, v in meta["channel_index"].items())
        print(f"  channel_index sizes consistent: {idx_ok}")
        for s, v in meta["channel_index"].items():
            print(f"    {s}: n={len(v)} first={v[:6]} sorted={v == sorted(v)} "
                  f"unique={len(set(v))==len(v)}")

        dec = build_decoder(widths)
        missing, unexpected = dec.load_state_dict(ck["state_dict"], strict=True), None
        dec.eval()
        n = sum(p.numel() for p in dec.parameters())
        print(f"  rebuild ok, params={n/1e6:.3f} M  load={time.time()-t0:.1f}s")

        z = torch.randn(1, 4, 64, 64)
        t0 = time.time()
        with torch.no_grad():
            y = dec(z)
        print(f"  forward {tuple(y.shape)} in {time.time()-t0:.2f}s (CPU)  "
              f"range=[{y.min():.2f},{y.max():.2f}] finite={torch.isfinite(y).all().item()}")


if __name__ == "__main__":
    main()
