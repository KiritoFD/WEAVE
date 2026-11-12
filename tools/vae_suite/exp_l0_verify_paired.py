"""L0-verify: unconfounded paired test of the Minimalism->Rococo latent direction.

L0 used ABSOLUTE CSD target similarity, which is confounded: a random perturbation
already moves CSD toward Impressionism (+0.083 at 20%) because that centroid sits
near the generic direction. Comparing "toward-style" vs "random" absolute gains
therefore mixes signal with that bias.

This test avoids it by using a NEAREST-CENTROID STYLE CLASSIFIER (measured 98.3%
on real held-out images vs 33.3% chance). Classification is a RELATIVE comparison
among centroids, so a generic-direction bias that shifts all classes equally does
not create a false positive.

For each source image of style s and each direction estimated from a DIFFERENT
content pair, we compare:
    z_style = z_s + dz(s->t)          (normalised to a fraction of ||z_s||)
    z_rand  = z_s + random, same norm
and ask how often each is classified as the target style t.

Verdict needs P(style) > P(rand) consistently, otherwise the L0 +0.066 was bias.
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
    ap.add_argument("--n-ref", type=int, default=24)
    ap.add_argument("--fracs", default="0.10,0.20")
    ap.add_argument("--out", default=str(OUT / "L0_verify_paired.json"))
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
    import csd_model
    csd, _, _ = csd_model.load_csd(dev)
    csd = csd.float().eval()
    for p in csd.parameters():
        p.requires_grad = False
    nm_ = torch.tensor(NORM_MEAN, device=dev).view(1, 3, 1, 1)
    ns_ = torch.tensor(NORM_STD, device=dev).view(1, 3, 1, 1)

    rng = np.random.default_rng(41)
    Z, C = {}, {}
    for s in avail:
        idxs = rng.choice(by_style[s], min(args.n_ref, len(by_style[s])),
                          replace=False).tolist()
        xs = torch.stack([ds[i]["image"] for i in idxs]).to(dev)
        with torch.no_grad():
            C[s] = torch.cat([csd_style(csd, xs[i:i + 8], nm_, ns_)
                              for i in range(0, len(xs), 8)])
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                Z[s] = torch.cat([vae.encode(xs[i:i + 8]).latent_dist.mean.float()
                                  for i in range(0, len(xs), 8)])
    keys = list(avail)
    CENT = torch.cat([F.normalize(C[s].mean(0, keepdim=True), dim=-1) for s in keys])
    print("[Judge] centroids: " + ", ".join(keys), flush=True)

    # sanity: classifier accuracy on the real held-out images
    acc = {}
    for i, s in enumerate(keys):
        pred = (C[s] @ CENT.T).argmax(-1)
        acc[s] = float((pred == i).float().mean())
    print("[Judge] real-image accuracy: "
          + " ".join(f"{k}={v*100:.0f}%" for k, v in acc.items()), flush=True)

    def classify(img):
        f = csd_style(csd, img, nm_, ns_)
        return (f @ CENT.T).argmax(-1)

    fracs = [float(x) for x in args.fracs.split(",")]
    rows = []
    for si, s in enumerate(keys):
        for ti, t in enumerate(keys):
            if s == t:
                continue
            n = min(len(Z[s]), len(Z[t]))
            if n < 4:
                continue
            # dz estimated from pair 0 only -> applied to sources 1..
            dz = (Z[t][0:1] - Z[s][0:1]).view(1, 4, 64, 64)
            src = Z[s][1:]
            if len(src) == 0:
                continue
            for frac in fracs:
                n_style = n_rand = n_tot = 0
                for ci in range(0, len(src), 4):
                    s0 = src[ci:ci + 4]
                    scale = frac * s0.flatten(1).norm(dim=1).mean()
                    d = dz / (dz.norm() + 1e-12) * scale
                    rd = torch.randn_like(d)
                    rd = rd / (rd.norm() + 1e-12) * d.norm()
                    with torch.no_grad(), torch.autocast(device_type="cuda",
                                                         dtype=torch.bfloat16):
                        y_st = vae.decode(s0 + d).sample.float()
                        y_rd = vae.decode(s0 + rd).sample.float()
                    n_style += int((classify(y_st) == ti).sum())
                    n_rand += int((classify(y_rd) == ti).sum())
                    n_tot += len(s0)
                    del y_st, y_rd
                    torch.cuda.empty_cache()
                rows.append({"pair": f"{s}->{t}", "frac": frac,
                             "p_style": round(n_style / n_tot, 4),
                             "p_rand": round(n_rand / n_tot, 4),
                             "delta": round((n_style - n_rand) / n_tot, 4),
                             "n": n_tot})
                print(f"  {s:>18} -> {t:<18} frac={frac:.2f}  "
                      f"P(style)={n_style/n_tot:.3f}  P(rand)={n_rand/n_tot:.3f}  "
                      f"delta={((n_style-n_rand)/n_tot):+.3f}", flush=True)

    print("\n" + "=" * 90)
    print("UNCONFOUNDED PAIRED TEST (nearest-centroid classification)")
    print("=" * 90)
    print(f"{'pair':<40}{'frac':>6}{'P(style)':>10}{'P(rand)':>9}{'delta':>9}")
    for r in sorted(rows, key=lambda x: -x["delta"]):
        print(f"{r['pair']:<40}{r['frac']:>6.2f}{r['p_style']:>10.3f}"
              f"{r['p_rand']:>9.3f}{r['delta']:>+9.3f}")
    if rows:
        d = [r["delta"] for r in rows]
        print(f"\noverall delta: mean={np.mean(d):+.4f} median={np.median(d):+.4f} "
              f"frac>0={float(np.mean([x > 0 for x in d]))*100:.1f}%")

    Path(args.out).write_text(json.dumps(rows, indent=2))
    print(f"\n[Saved] {args.out}")


if __name__ == "__main__":
    main()
