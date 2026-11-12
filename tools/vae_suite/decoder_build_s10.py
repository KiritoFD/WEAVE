"""One-shot structured pruning A0 -> S10 (~10M), with Taylor importance.

Widths are given in FORWARD order [c0,c1,c2,c3]; the decoder reverses them, so
    c3 -> mid_block + up_blocks[0]   (lowest resolution, 64x64)
    c2 -> up_blocks[1]               (128x128)
    c1 -> up_blocks[2]               (256x256)
    c0 -> up_blocks[3]               (512x512, highest resolution)

S10 keeps only 64 channels at 512x512 (A0: 128), which is where activation memory
is dominated.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/root/Workspace/WEAVE_VAE")
sys.path.insert(0, "/root/Workspace/WEAVE_VAE/tools/vae_suite")

import numpy as np
import torch
from diffusers import AutoencoderKL

from tools.vae_suite.decoder_prune import (A0_DIR, MANIFEST, make_plan,
                                           prune_decoder, taylor_importance, build_decoder)

OUT = Path("/root/Workspace/WEAVE_VAE/runs/vae_decoder_prune")
CAND = {
    "s10": [64, 128, 192, 256],
    "s15": [96, 160, 256, 288],
    "s10b": [64, 128, 224, 256],
    "s10c": [64, 160, 192, 256],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default="s10")
    ap.add_argument("--calib", type=int, default=128)
    args = ap.parse_args()

    device = "cuda"
    vae = AutoencoderKL.from_pretrained(A0_DIR, torch_dtype=torch.float32).to(device).eval()
    dec_a0 = vae.decoder
    a0_params = sum(p.numel() for p in dec_a0.parameters())
    print(f"[A0] {a0_params/1e6:.3f} M  widths=[128,256,512,512]", flush=True)

    print("\n=== candidate widths (forward order) ===")
    print(f"{'name':<8}{'widths':<26}{'params':>10}{'%A0':>7}  512px chans (c1+c0)")
    for name, w in CAND.items():
        m = build_decoder(w)
        p = sum(q.numel() for q in m.parameters())
        hi = w[1] + w[0]          # channels present at 512x512 (ub2 + ub3)
        print(f"{name:<8}{str(w):<26}{p/1e6:>9.3f}M{p/a0_params*100:>6.1f}%  "
              f"{hi} (A0: 384, -{(1-hi/384)*100:.0f}%)")
        del m

    name = args.build
    widths = CAND[name]
    print(f"\n=== building {name} {widths} ===", flush=True)

    # calibration data from the converted cache (no encoder run)
    split = json.loads((OUT / "split.json").read_text())
    lat = np.memmap(split["train_bin"], dtype=np.float16, mode="r",
                    shape=(split["n_train"], split["dim"]))
    man = json.loads(MANIFEST.read_text())
    img = np.memmap(man["bin_path"], dtype=np.uint8, mode="r", shape=tuple(man["shape"]))
    raw = np.load(OUT / "raw_index.npy")
    rng = np.random.default_rng(0)
    sel = rng.choice(split["n_train"], args.calib, replace=False)
    zs, xs = [], []
    for j in sel:
        zs.append(torch.from_numpy(lat[int(j)].astype(np.float32)).view(4, 64, 64))
        a = np.ascontiguousarray(img[int(raw[int(j)])])
        xs.append(torch.from_numpy(a).permute(2, 0, 1).float().div_(127.5).sub_(1.0))
    Zc = torch.stack(zs); Xc = torch.stack(xs)
    print(f"[Calib] {tuple(Zc.shape)}", flush=True)

    t0 = time.time()
    print("[Taylor] computing importance...", flush=True)
    imp = taylor_importance(dec_a0, make_plan(dec_a0), Zc, Xc, device)
    print(f"[Taylor] done in {time.time()-t0:.0f}s", flush=True)
    torch.save(imp, OUT / "taylor_importance.pt")

    dec_a0 = dec_a0.cpu()
    torch.cuda.empty_cache()
    new, meta = prune_decoder(dec_a0, widths, importance=imp)
    print(f"[Pruned] {meta['params']/1e6:.3f} M ({meta['params']/a0_params*100:.1f}% of A0)",
          flush=True)
    print(f"  stream_target={meta['stream_target']}", flush=True)

    # zero-training sanity: how far is the pruned output from A0 on real latents?
    new.eval().to(device)
    with torch.no_grad():
        z = Zc[:8].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            y0 = dec_a0.to(device)(z)
            y1 = new(z)
    cos = torch.nn.functional.cosine_similarity(
        y0.float().flatten(), y1.float().flatten(), dim=0).item()
    l1 = (y1.float() - Xc[:8].to(device)).abs().mean().item()
    l1_a0 = (y0.float() - Xc[:8].to(device)).abs().mean().item()
    print(f"[Sanity] cos(A0,pruned)={cos:.4f}  L1 pruned={l1:.5f} vs A0={l1_a0:.5f}", flush=True)

    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": new.cpu().state_dict(), "meta": meta}, d / "decoder.pt")
    (d / "prune_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[Saved] {d}/decoder.pt", flush=True)


if __name__ == "__main__":
    main()
