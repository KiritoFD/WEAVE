"""Precompute frozen A0 posterior-mean latents for the 20-class training pool.

Stored as float16 memmap: 78,515 x 16384 x 2B = 2.57 GB.
Used by the weak A0 latent anchor term:
    L_anchor = mean( (E_theta(x) - E_A0(x))^2 ) / 16384
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/root/Workspace/WEAVE_VAE")

import numpy as np
import torch
from diffusers import AutoencoderKL

OUT = Path("/root/Workspace/WEAVE_VAE/runs/vae_style_rank20")
MANIFEST = Path("/root/Workspace/data/wikiart_512_raw/wikiart_512_raw_manifest.json")
A0_DIR = "/root/Workspace/WEAVE_VAE/runs_vae/A0/final_diffusers_vae"
DEV = "cuda"
DIM = 16384


def main():
    idx = torch.load(OUT / "index.pt", weights_only=False)
    pos_list = idx["pos_list"]
    n = len(pos_list)
    print(f"[A0Cache] {n} images", flush=True)

    man = json.loads(MANIFEST.read_text())
    mmap = np.memmap(man["bin_path"], dtype=np.uint8, mode="r", shape=tuple(man["shape"]))

    binp = OUT / "a0_latent_fp16.bin"
    dst = np.memmap(binp, dtype=np.float16, mode="w+", shape=(n, DIM))

    vae = AutoencoderKL.from_pretrained(A0_DIR).to(DEV).eval()
    for p in vae.parameters():
        p.requires_grad = False

    t0 = time.time()
    BS = 32
    with torch.no_grad():
        for i in range(0, n, BS):
            chunk = pos_list[i:i + BS]
            arr = np.stack([mmap[p] for p in chunk])
            x = torch.from_numpy(arr).permute(0, 3, 1, 2).float().to(DEV)
            x = x.div_(127.5).sub_(1.0)
            mu = vae.encode(x).latent_dist.mean.float().flatten(1)
            dst[i:i + len(chunk)] = mu.cpu().numpy().astype(np.float16)
            if (i // BS + 1) % 100 == 0:
                el = time.time() - t0
                done = i + len(chunk)
                print(f"  {done}/{n}  {el:.0f}s  eta {el/done*(n-done):.0f}s", flush=True)
    dst.flush()
    print(f"[A0Cache] done in {time.time()-t0:.0f}s -> {binp} "
          f"({binp.stat().st_size/1024**3:.2f} GB)", flush=True)
    (OUT / "a0_latent_meta.json").write_text(json.dumps(
        {"path": str(binp), "n": n, "dim": DIM, "dtype": "float16"}, indent=2))


if __name__ == "__main__":
    main()
