"""Reproduce build_s10's sanity computation step by step to find the L1=1.01 bug."""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/Workspace/WEAVE_VAE")
sys.path.insert(0, "/root/Workspace/WEAVE_VAE/tools/vae_suite")

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL

OUT = Path("/root/Workspace/WEAVE_VAE/runs/vae_decoder_prune")
MANIFEST = Path("/root/Workspace/data/wikiart_512_raw/wikiart_512_raw_manifest.json")
A0_DIR = "/root/Workspace/WEAVE_VAE/runs_vae/A0/final_diffusers_vae"


def main():
    dev = "cuda"
    split = json.loads((OUT / "split.json").read_text())
    lat = np.memmap(split["train_bin"], dtype=np.float16, mode="r",
                    shape=(split["n_train"], split["dim"]))
    man = json.loads(MANIFEST.read_text())
    img = np.memmap(man["bin_path"], dtype=np.uint8, mode="r", shape=tuple(man["shape"]))
    raw = np.load(OUT / "raw_index.npy")

    rng = np.random.default_rng(0)
    sel = rng.choice(split["n_train"], 128, replace=False)
    zs, xs = [], []
    for j in sel:
        zs.append(torch.from_numpy(lat[int(j)].astype(np.float32)).view(4, 64, 64))
        a = np.ascontiguousarray(img[int(raw[int(j)])])
        xs.append(torch.from_numpy(a).permute(2, 0, 1).float().div_(127.5).sub_(1.0))
    Zc = torch.stack(zs); Xc = torch.stack(xs)
    print(f"Zc {tuple(Zc.shape)} Xc {tuple(Xc.shape)} sel[:8]={sel[:8].tolist()}")

    vae = AutoencoderKL.from_pretrained(A0_DIR, torch_dtype=torch.float32).to(dev).eval()
    dec = vae.decoder

    # (1) exact build_s10 path: decode the CACHED latent
    z = Zc[:8].to(dev)
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            y0 = dec(z)
    l1 = (y0.float() - Xc[:8].to(dev)).abs().mean().item()
    print(f"\n[path1] decode(cached latent) L1 vs Xc[:8] = {l1:.5f}")

    # (2) control: encode the image fresh then decode
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            z2 = vae.encode(Xc[:8].to(dev)).latent_dist.mean
            y2 = dec(z2)
    l2 = (y2.float() - Xc[:8].to(dev)).abs().mean().item()
    print(f"[path2] encode->decode        L1 vs Xc[:8] = {l2:.5f}")
    print(f"[check] cos(cached z, fresh z) = "
          f"{float(F.cosine_similarity(z.flatten(), z2.flatten(), dim=0)):.6f}")

    # (3) per-sample to expose any single bad row
    with torch.no_grad():
        for i in range(8):
            yi = y0[i:i + 1].float()
            xi = Xc[i:i + 1].to(dev)
            print(f"  sample {i}: L1={float((yi-xi).abs().mean()):.5f} "
                  f"y_mean={float(yi.mean()):+.3f} x_mean={float(xi.mean()):+.3f} "
                  f"cos={float(F.cosine_similarity(yi.flatten(), xi.flatten(), dim=0)):+.3f}")

    # (4) what if Xc is uint8 (un-normalized)? emulate the wrong path
    x_raw = torch.stack([
        torch.from_numpy(np.ascontiguousarray(img[int(raw[int(sel[i])])]))
        .permute(2, 0, 1).float() for i in range(8)]).to(dev)
    print(f"\n[diag] L1 vs UNNORMALIZED uint8-range target = "
          f"{float((y0.float()-x_raw).abs().mean()):.5f}")


if __name__ == "__main__":
    main()
