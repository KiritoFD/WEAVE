"""Cache A0 posterior-mean latents for the full WikiArt train pool (fp16 memmap).

100,207 images x 16,384 dims x 2 bytes = ~3.1 GB.
Also emits a deterministic holdout split for checkpoint selection.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/root/Workspace/WEAVE_VAE")

import numpy as np
import torch
from diffusers import AutoencoderKL

from tools.vae_suite.dataset import RawMemmapWikiArtDataset

A0_DIR = "/root/Workspace/WEAVE_VAE/runs_vae/A0/final_diffusers_vae"
MANIFEST = Path("/root/Workspace/data/wikiart_512_raw/wikiart_512_raw_manifest.json")
OUT = Path("/root/Workspace/WEAVE_VAE/runs/vae_decoder_prune")
OUT.mkdir(parents=True, exist_ok=True)
DIM = 16384
HOLDOUT_EVERY = 50          # ~1/50 of the pool held out for checkpoint selection


def main():
    ds = RawMemmapWikiArtDataset(MANIFEST, is_training=True)
    n = len(ds)
    print(f"[Pool] {n} train images", flush=True)

    binp = OUT / "latent_full_fp16.bin"
    dst = np.memmap(binp, dtype=np.float16, mode="w+", shape=(n, DIM))

    vae = AutoencoderKL.from_pretrained(A0_DIR, torch_dtype=torch.bfloat16).to("cuda").eval()
    for p in vae.parameters():
        p.requires_grad = False

    t0 = time.time()
    BS = 32
    with torch.no_grad():
        for i in range(0, n, BS):
            idx = list(range(i, min(i + BS, n)))
            x = torch.stack([ds[j]["image"] for j in idx]).cuda()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                mu = vae.encode(x).latent_dist.mean
            dst[i:i + len(idx)] = mu.float().flatten(1).cpu().numpy().astype(np.float16)
            if (i // BS + 1) % 100 == 0:
                el = time.time() - t0
                done = i + len(idx)
                print(f"  {done}/{n}  {el:.0f}s  eta {el/done*(n-done)/60:.1f} min", flush=True)
    dst.flush()
    print(f"[Done] {binp} ({binp.stat().st_size/1024**3:.2f} GB) "
          f"in {(time.time()-t0)/60:.1f} min", flush=True)

    holdout = list(range(0, n, HOLDOUT_EVERY))
    train = [i for i in range(n) if i % HOLDOUT_EVERY != 0]
    (OUT / "split_full.json").write_text(json.dumps({
        "n": n, "dim": DIM, "latent_bin": str(binp),
        "holdout": holdout, "n_holdout": len(holdout),
        "n_train": len(train),
    }), encoding="utf-8")
    np.save(OUT / "train_idx_full.npy", np.array(train, dtype=np.int32))
    np.save(OUT / "holdout_idx_full.npy", np.array(holdout, dtype=np.int32))
    print(f"[Split] train={len(train)} holdout={len(holdout)}", flush=True)


if __name__ == "__main__":
    main()
