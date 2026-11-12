"""Is CSD a DISCRIMINATIVE style judge?

L0 showed that for Impressionism the random-direction CSD gain (+0.131) exceeds
the "real style direction" gain (+0.059). If random perturbations systematically
push CSD toward a style, that style's CSD target vector is not discriminative and
CSD cannot serve as a style metric for it.

Three measurements per style:
  D1 self-consistency : cos between CSD of a style's own images and its centroid
                        (how tight is the style cluster)
  D2 separability     : how often a held-out image is closest to its own style
                        centroid vs another style (a k-NN style test)
  D3 random sensitivity: mean dS toward style t from a RANDOM latent perturbation,
                        as a function of magnitude. A good judge keeps this ~0.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "/root/Workspace/WEAVE_VAE")
sys.path.insert(0, "/root/Workspace/WEAVE_VAE/tools/vae_suite")
sys.path.insert(0, "/root/Workspace/WEAVE_VAE/tools/12h")

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL

from tools.vae_suite.dataset import RawMemmapWikiArtDataset

OUT = Path("/root/Workspace/WEAVE_VAE/runs/vae_decoder_prune")
A0_DIR = "/root/Workspace/WEAVE_VAE/runs_vae/A0/final_diffusers_vae"
MANIFEST = Path("/root/Workspace/data/wikiart_512_raw/wikiart_512_raw_manifest.json")
NORM_MEAN = [0.48145466, 0.4578275, 0.40821073]
NORM_STD = [0.26862954, 0.26130258, 0.27577711]
STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]


def csd_style(csd, img, nm, ns):
    x01 = (img.clamp(-1, 1) + 1.0) / 2.0
    x224 = F.interpolate(x01, (224, 224), mode="bicubic", align_corners=False)
    _, _, s = csd((x224 - nm) / ns)
    return F.normalize(s.float(), dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="images per style")
    ap.add_argument("--out", default=str(OUT / "csd_discriminability.json"))
    args = ap.parse_args()
    dev = "cuda"

    man = json.loads(MANIFEST.read_text())
    name_of = {v: k for k, v in man["style_to_id"].items()}
    ds = RawMemmapWikiArtDataset(MANIFEST, is_training=False)
    by_style = defaultdict(list)
    for i in range(len(ds)):
        nm = name_of.get(int(ds[i]["style_id"]), "")
        if nm in STYLES:
            by_style[nm].append(i)
    avail = [s for s in STYLES if len(by_style[s]) >= 10]
    print("[Data] " + ", ".join(f"{s}={len(by_style[s])}" for s in avail), flush=True)

    vae = AutoencoderKL.from_pretrained(A0_DIR, torch_dtype=torch.float32).to(dev).eval()
    pqc = vae.post_quant_conv
    import csd_model
    csd, _, _ = csd_model.load_csd(dev)
    csd = csd.float().eval()
    for p in csd.parameters():
        p.requires_grad = False
    nm_ = torch.tensor(NORM_MEAN, device=dev).view(1, 3, 1, 1)
    ns_ = torch.tensor(NORM_STD, device=dev).view(1, 3, 1, 1)

    rng = np.random.default_rng(17)
    F_, Z_ = {}, {}
    for s in avail:
        idxs = rng.choice(by_style[s], min(args.n, len(by_style[s])),
                          replace=False).tolist()
        xs = torch.stack([ds[i]["image"] for i in idxs]).to(dev)
        with torch.no_grad():
            F_[s] = torch.cat([csd_style(csd, xs[i:i + 8], nm_, ns_)
                               for i in range(0, len(xs), 8)])
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                Z_[s] = torch.cat([vae.encode(xs[i:i + 8]).latent_dist.mean.float()
                                   for i in range(0, len(xs), 8)])
    cent = {s: F_[s].mean(0, keepdim=True) for s in avail}
    tgt = {s: F.normalize(cent[s], dim=-1) for s in avail}

    # ---- D1 self-consistency ----
    print("\n--- D1: cos(CSD(image), CSD(style centroid)) ---", flush=True)
    d1 = {}
    for s in avail:
        c = float((F_[s] * F.normalize(cent[s], dim=-1)).sum(-1).mean())
        d1[s] = round(c, 4)
        print(f"  {s:<20}{c:+.4f}", flush=True)

    # ---- D2 separability (nearest-centroid accuracy) ----
    print("\n--- D2: nearest-centroid style accuracy ---", flush=True)
    keys = list(avail)
    C = torch.cat([F.normalize(cent[s], dim=-1) for s in keys])   # (K, dim)
    d2 = {}
    for i, s in enumerate(keys):
        pred = (F_[s] @ C.T).argmax(-1)
        acc = float((pred == i).float().mean())
        d2[s] = round(acc, 4)
    tot = float(np.mean(list(d2.values())))
    for s in keys:
        print(f"  {s:<20}{d2[s]*100:5.1f}%", flush=True)
    print(f"  {'mean':<20}{tot*100:5.1f}%   (chance {100/len(keys):.1f}%)", flush=True)

    # ---- D3 random sensitivity ----
    print("\n--- D3: dS toward each style from a RANDOM latent perturbation ---",
          flush=True)
    fracs = [0.05, 0.10, 0.20]
    d3 = {s: {} for s in avail}
    for frac in fracs:
        row = {}
        for s in avail:              # source style
            src_all = Z_[s]
            acc = {t: [] for t in avail}
            for ci in range(0, len(src_all), 4):
                src = src_all[ci:ci + 4]
                scale = frac * src.flatten(1).norm(dim=1).mean()
                rd = torch.randn_like(src)
                rd = rd / rd.flatten(1).norm(dim=1).view(-1, 1, 1, 1) * scale
                with torch.no_grad(), torch.autocast(device_type="cuda",
                                                     dtype=torch.bfloat16):
                    y0 = vae.decode(src).sample.float()
                    y1 = vae.decode(src + rd).sample.float()
                f0 = csd_style(csd, y0, nm_, ns_)
                f1 = csd_style(csd, y1, nm_, ns_)
                for t in avail:
                    g = tgt[t]
                    acc[t].append(float(((f1 * g).sum(-1) - (f0 * g).sum(-1)).mean()))
                del y0, y1
                torch.cuda.empty_cache()
            for t in avail:
                row[t] = row.get(t, [])
                row[t].append(float(np.mean(acc[t])))
        for t in avail:
            d3[t][str(frac)] = round(float(np.mean(row[t])), 4)
        print(f"  frac={frac:.2f}: " + "  ".join(
            f"{t}={d3[t][str(frac)]:+.4f}" for t in avail), flush=True)

    print("\n" + "=" * 84)
    print("VERDICT: a style whose random-perturbation gain is large is NOT")
    print("discriminative in CSD; CSD cannot be a style metric for it.")
    print("=" * 84)
    print(f"{'style':<20}{'D1 self-cos':>13}{'D2 acc':>9}"
          + "".join(f"{'rand@'+f:>11}" for f in [str(x) for x in fracs]))
    for s in avail:
        print(f"{s:<20}{d1[s]:>13.4f}{d2[s]*100:>8.1f}%"
              + "".join(f"{d3[s][str(f)]:>+11.4f}" for f in fracs))

    Path(args.out).write_text(json.dumps(
        {"D1_self_cos": d1, "D2_nearest_centroid_acc": d2,
         "D3_random_gain": d3, "n_per_style": args.n}, indent=2))
    print(f"\n[Saved] {args.out}")


if __name__ == "__main__":
    main()
