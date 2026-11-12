"""Check whether the existing a0_latent_fp16.bin is correctly aligned.

Two candidate interpretations of the cache row order:
  (a) row i  <->  mmap[ pos_list[i] ]              (what cache_a0_latents.py did)
  (b) row i  <->  ds[ pos_list[i] ] = mmap[indices[pos_list[i]]]

Compare each against a fresh A0 encoding of the actual image.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/Workspace/WEAVE_VAE")

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL

from tools.vae_suite.dataset import RawMemmapWikiArtDataset

A0_DIR = "/root/Workspace/WEAVE_VAE/runs_vae/A0/final_diffusers_vae"
MANIFEST = Path("/root/Workspace/data/wikiart_512_raw/wikiart_512_raw_manifest.json")
CACHE = Path("/root/Workspace/WEAVE_VAE/runs/vae_style_rank20/a0_latent_fp16.bin")
DIM = 16384


def main():
    idx = torch.load("/root/Workspace/WEAVE_VAE/runs/vae_style_rank20/index.pt",
                     weights_only=False)
    pos_list = idx["pos_list"]
    n = len(pos_list)
    print(f"[Cache] rows={CACHE.stat().st_size // (DIM*2)}  pos_list={n}")

    man = json.loads(MANIFEST.read_text())
    mmap = np.memmap(man["bin_path"], dtype=np.uint8, mode="r", shape=tuple(man["shape"]))
    ds = RawMemmapWikiArtDataset(MANIFEST, is_training=True)
    print(f"[Dataset] len={len(ds)}  indices[0:5]={list(ds.indices[:5])}")

    lat = np.memmap(CACHE, dtype=np.float16, mode="r", shape=(n, DIM))

    vae = AutoencoderKL.from_pretrained(A0_DIR, torch_dtype=torch.bfloat16).to("cuda").eval()

    def enc(x_np):
        x = torch.from_numpy(np.ascontiguousarray(x_np)).permute(2, 0, 1).float()
        x = x.div_(127.5).sub_(1.0).unsqueeze(0).cuda()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return vae.encode(x).latent_dist.mean.float().flatten(1)[0].cpu()

    print(f"\n{'row':>5} {'interp':<12}{'cos':>9}{'relL2':>9}")
    ok_a = ok_b = 0
    for r in [0, 1, 2, 100, 5000, n - 1]:
        p = pos_list[r]
        cached = torch.from_numpy(lat[r].astype(np.float32))

        # (a) raw index == p
        za = enc(mmap[p])
        ca = float(F.cosine_similarity(cached, za, dim=0))
        ra = float((cached - za).norm() / (za.norm() + 1e-8))
        print(f"{r:>5} {'mmap[p]':<12}{ca:>9.5f}{ra:>9.5f}")
        if ca > 0.99:
            ok_a += 1

        # (b) dataset position p
        zb = enc(ds[p]["image"].permute(1, 2, 0).numpy())
        cb = float(F.cosine_similarity(cached, zb, dim=0))
        rb = float((cached - zb).norm() / (zb.norm() + 1e-8))
        print(f"{r:>5} {'ds[p]':<12}{cb:>9.5f}{rb:>9.5f}")
        if cb > 0.99:
            ok_b += 1

    print(f"\nPASS counts: mmap[p]={ok_a}/6  ds[p]={ok_b}/6")
    if ok_b == 6:
        print("=> cache is aligned with ds[p]: REUSABLE via pos_list re-indexing")
    elif ok_a == 6:
        print("=> cache is aligned with mmap[p]: usable only with the raw-index list")
    else:
        print("=> cache does NOT match either interpretation: must recompute")


if __name__ == "__main__":
    main()
