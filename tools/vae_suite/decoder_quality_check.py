"""Trustworthy zero-training quality check: A0 vs a pruned decoder.

Uses ONLY the verified path: image = ds[i]["image"], latent = cached row i,
output = vae.decode(latent). No re-derivation of indices.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/Workspace/WEAVE_VAE")
sys.path.insert(0, "/root/Workspace/WEAVE_VAE/tools/vae_suite")

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL

from tools.vae_suite.decoder_prune import build_decoder, StudentDecoder

OUT = Path("/root/Workspace/WEAVE_VAE/runs/vae_decoder_prune")
A0_DIR = "/root/Workspace/WEAVE_VAE/runs_vae/A0/final_diffusers_vae"


def psnr(x, y):
    mse = F.mse_loss((x.clamp(-1, 1) + 1) / 2, (y.clamp(-1, 1) + 1) / 2).item()
    return 50.0 if mse <= 1e-10 else float(10 * np.log10(1.0 / mse))


def ssim_simple(x, y):
    x01 = (x.clamp(-1, 1) + 1) / 2; y01 = (y.clamp(-1, 1) + 1) / 2
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mx = F.avg_pool2d(x01, 11, 1, 5); my = F.avg_pool2d(y01, 11, 1, 5)
    mxx, myy, mxy = mx * mx, my * my, mx * my
    sx = F.avg_pool2d(x01 * x01, 11, 1, 5) - mxx
    sy = F.avg_pool2d(y01 * y01, 11, 1, 5) - myy
    sxy = F.avg_pool2d(x01 * y01, 11, 1, 5) - mxy
    m = ((2 * mxy + C1) * (2 * sxy + C2)) / ((mxx + myy + C1) * (sx + sy + C2))
    return float(m.mean())


def haar_hf(x, levels=3):
    outs = []
    for _ in range(levels):
        a = x[:, :, 0::2, 0::2]; b = x[:, :, 1::2, 0::2]
        c = x[:, :, 0::2, 1::2]; d = x[:, :, 1::2, 1::2]
        outs.append(((a - b + c - d) * 0.5, (a + b - c - d) * 0.5, (a - b - c + d) * 0.5))
        x = (a + b + c + d) * 0.5
    return outs


def hf_corr(y, x):
    cs = []
    for so, sx in zip(haar_hf(y), haar_hf(x)):
        for a, b in zip(so, sx):
            a = a.flatten() - a.mean(); b = b.flatten() - b.mean()
            cs.append(float((a * b).sum() / (a.norm() * b.norm() + 1e-8)))
    return float(np.mean(cs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="A0,s10")
    ap.add_argument("--n", type=int, default=64)
    args = ap.parse_args()
    dev = "cuda"

    split = json.loads((OUT / "split.json").read_text())
    n_tr, dim = split["n_train"], split["dim"]
    lat = np.memmap(split["train_bin"], dtype=np.float16, mode="r", shape=(n_tr, dim))
    raw = np.load(OUT / "raw_index.npy")

    from tools.vae_suite.dataset import RawMemmapWikiArtDataset
    ds = RawMemmapWikiArtDataset(
        Path("/root/Workspace/data/wikiart_512_raw/wikiart_512_raw_manifest.json"),
        is_training=True)
    # map raw_index -> dataset position
    pos_of_raw = {}
    for p, ri in enumerate(ds.indices):
        pos_of_raw[ri] = p

    rng = np.random.default_rng(7)
    rows = rng.choice(n_tr, args.n, replace=False)

    X = torch.stack([ds[pos_of_raw[int(raw[int(j)])]]["image"] for j in rows]).to(dev)
    Z = torch.stack([torch.from_numpy(lat[int(j)].astype(np.float32)).view(4, 64, 64)
                     for j in rows]).to(dev)
    print(f"[Data] X{tuple(X.shape)} Z{tuple(Z.shape)}", flush=True)

    vae = AutoencoderKL.from_pretrained(A0_DIR, torch_dtype=torch.float32).to(dev).eval()
    dec_a0 = vae.decoder
    pqc = vae.post_quant_conv

    results = {}
    for name in args.models.split(","):
        name = name.strip()
        if name == "A0":
            dec = StudentDecoder(pqc, dec_a0).eval()
            params = sum(p.numel() for p in dec_a0.parameters())
        else:
            ckf = OUT / name / "decoder.pt"
            if not ckf.exists():
                print(f"[{name}] missing, skip", flush=True)
                continue
            ck = torch.load(ckf, map_location="cpu", weights_only=False)
            d = build_decoder(ck["meta"]["widths_forward"])
            d.load_state_dict(ck["state_dict"])
            dec = StudentDecoder(pqc, d).to(dev).eval()
            params = sum(p.numel() for p in d.parameters())
        dec.eval()

        l1s, lps, pss, sss, hcs, coss = [], [], [], [], [], []
        with torch.no_grad():
            for i in range(0, len(X), 8):
                z = Z[i:i + 8]
                x = X[i:i + 8]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    y = dec(z)
                y = y.float()
                l1s.append(float((y - x).abs().mean()))
                for k in range(len(z)):
                    coss.append(float(F.cosine_similarity(
                        y[k].flatten(), x[k].flatten(), dim=0)))
                pss += [psnr(y[k:k + 1], x[k:k + 1]) for k in range(len(z))]
                sss += [ssim_simple(y[k:k + 1], x[k:k + 1]) for k in range(len(z))]
                hcs += [hf_corr(y[k:k + 1], x[k:k + 1]) for k in range(len(z))]
        results[name] = {
            "params_M": round(params / 1e6, 3),
            "L1": round(float(np.mean(l1s)), 5),
            "PSNR": round(float(np.mean(pss)), 2),
            "SSIM": round(float(np.mean(sss)), 4),
            "HFcorr": round(float(np.mean(hcs)), 4),
            "cos_yx": round(float(np.mean(coss)), 4),
        }
        print(f"[{name}] {results[name]}", flush=True)
        if name != "A0":
            del dec
        torch.cuda.empty_cache()

    (OUT / "zero_train_quality.json").write_text(json.dumps(results, indent=2))
    print("\n=== zero-training quality ===")
    print(f"{'model':<8}{'params':>9}{'L1':>9}{'PSNR':>8}{'SSIM':>8}{'HFcorr':>8}{'cos':>8}")
    for k, v in results.items():
        print(f"{k:<8}{v['params_M']:>8.3f}M{v['L1']:>9.5f}{v['PSNR']:>8.2f}"
              f"{v['SSIM']:>8.4f}{v['HFcorr']:>8.4f}{v['cos_yx']:>8.4f}")


if __name__ == "__main__":
    main()
