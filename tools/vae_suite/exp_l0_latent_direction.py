"""P4-L0: is there a REUSABLE latent style direction?

The previous directions we tried were style-CENTROID differences, which turned out
to have ~0 specificity. This experiment uses per-PAIR differences between
content-matched images of different styles, and asks two separate questions:

  Q1 reproducibility : for a style pair (s,t), do different content pairs give
                       the SAME direction?  cos(dz_1, dz_2) across artist-disjoint
                       pairs, vs a random baseline.
  Q2 transferability : does dz estimated on content pair #1 move a DIFFERENT
                       source image of style s toward style t?
                       measured on the decoded image (A0 decoder, so the decoder
                       is not a confound), against a matched-norm random control.

If Q1 is high but Q2 is ~0, the direction is reproducible but semantically inert
(the same failure mode we already saw for centroids).
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
import lpips

OUT = Path("/root/Workspace/WEAVE_VAE/runs/vae_decoder_prune")
A0_DIR = "/root/Workspace/WEAVE_VAE/runs_vae/A0/final_diffusers_vae"
MANIFEST = Path("/root/Workspace/data/wikiart_512_raw/wikiart_512_raw_manifest.json")
NORM_MEAN = [0.48145466, 0.4578275, 0.40821073]
NORM_STD = [0.26862954, 0.26130258, 0.27577711]
TARGET_STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]


def csd_style(csd, img, nm, ns):
    x01 = (img.clamp(-1, 1) + 1.0) / 2.0
    x224 = F.interpolate(x01, (224, 224), mode="bicubic", align_corners=False)
    _, _, s = csd((x224 - nm) / ns)
    return F.normalize(s.float(), dim=-1)


def dino_feat(m, img, dev):
    x01 = (img.clamp(-1, 1) + 1.0) / 2.0
    x224 = F.interpolate(x01, (224, 224), mode="bicubic", align_corners=False)
    nm = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    ns = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    o = m(pixel_values=(x224 - nm) / ns)
    return F.normalize(o.last_hidden_state[:, 0].float(), dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-ref", type=int, default=24, help="images per style")
    ap.add_argument("--fracs", default="0.05,0.10,0.20")
    ap.add_argument("--out", default=str(OUT / "L0_latent_direction.json"))
    args = ap.parse_args()
    dev = "cuda"

    man = json.loads(MANIFEST.read_text())
    name_of = {v: k for k, v in man["style_to_id"].items()}
    # use the PROPER test split only (never trained on)
    ds = RawMemmapWikiArtDataset(MANIFEST, is_training=False)
    by_style = defaultdict(list)
    for i in range(len(ds)):
        nm = name_of.get(int(ds[i]["style_id"]), "")
        if nm in TARGET_STYLES:
            by_style[nm].append(i)
    avail = [s for s in TARGET_STYLES if len(by_style[s]) >= 8]
    print("[Data] " + ", ".join(f"{s}={len(by_style[s])}" for s in avail), flush=True)
    if len(avail) < 2:
        print("not enough styles")
        return

    vae = AutoencoderKL.from_pretrained(A0_DIR, torch_dtype=torch.float32).to(dev).eval()
    pqc = vae.post_quant_conv
    lp = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
    for p in lp.parameters():
        p.requires_grad = False
    import csd_model
    csd, _, _ = csd_model.load_csd(dev)
    csd = csd.float().eval()
    for p in csd.parameters():
        p.requires_grad = False
    from transformers import AutoModel
    dino = AutoModel.from_pretrained("facebook/dinov2-base").to(dev).eval()
    for p in dino.parameters():
        p.requires_grad = False
    nm_ = torch.tensor(NORM_MEAN, device=dev).view(1, 3, 1, 1)
    ns_ = torch.tensor(NORM_STD, device=dev).view(1, 3, 1, 1)

    rng = np.random.default_rng(5)
    idx = {s: rng.choice(by_style[s], min(args.n_ref, len(by_style[s])),
                         replace=False).tolist() for s in avail}

    def enc(idxs):
        xs = torch.stack([ds[i]["image"] for i in idxs]).to(dev)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            z = vae.encode(xs).latent_dist.mean.float()
        return xs, z

    X, Z, C, DN = {}, {}, {}, {}
    for s in avail:
        X[s], Z[s] = enc(idx[s])
        C[s] = torch.stack([csd_style(csd, X[s], nm_, ns_)])[0]
        DN[s] = dino_feat(dino, X[s], dev)
        print(f"[Ref] {s}: {len(idx[s])} images", flush=True)

    # style targets (mean CSD of that style's references)
    tgt = {s: F.normalize(C[s].mean(0, keepdim=True), dim=-1) for s in avail}

    fracs = [float(x) for x in args.fracs.split(",")]

    # ---------------- Q1 reproducibility ----------------
    print("\n" + "=" * 96)
    print("Q1: reproducibility of per-pair dz for the same style pair")
    print("=" * 96, flush=True)
    q1 = []
    for a in range(len(avail)):
        for b in range(a + 1, len(avail)):
            s, t = avail[a], avail[b]
            n = min(len(Z[s]), len(Z[t]))
            if n < 4:
                continue
            # two disjoint halves of the available pairs
            perm = rng.permutation(n)
            h1, h2 = perm[:n // 2], perm[n // 2:]
            if len(h1) < 2 or len(h2) < 2:
                continue
            dz1 = (Z[t][h1] - Z[s][h1]).mean(0)
            dz2 = (Z[t][h2] - Z[s][h2]).mean(0)
            cos = float(F.cosine_similarity(dz1.flatten(), dz2.flatten(), dim=0))
            # random baseline: cosine between two random directions
            r1 = torch.randn_like(dz1); r2 = torch.randn_like(dz2)
            rcos = float(F.cosine_similarity(r1.flatten(), r2.flatten(), dim=0))
            q1.append({"pair": f"{s}->{t}", "cos": round(cos, 4),
                       "random_cos": round(rcos, 4), "n": int(n)})
    if q1:
        cs = [r["cos"] for r in q1]
        print(f"  pairs={len(q1)}  mean cos={np.mean(cs):+.4f}  "
              f"median={np.median(cs):+.4f}  min={min(cs):+.4f}  max={max(cs):+.4f}",
              flush=True)
        for r in q1:
            print(f"    {r['pair']:<34} cos={r['cos']:+.4f}  (n={r['n']})", flush=True)

    # ---------------- Q2 transferability ----------------
    print("\n" + "=" * 96)
    print("Q2: does dz from one content pair move OTHER sources toward the target style?")
    print("=" * 96, flush=True)
    rows = []
    for a in range(len(avail)):
        for b in range(len(avail)):
            if a == b:
                continue
            s, t = avail[a], avail[b]
            n = min(len(Z[s]), len(Z[t]))
            if n < 4:
                continue
            # dz from pair 0
            dz = (Z[t][0:1] - Z[s][0:1]).mean(0, keepdim=True)
            d_raw = dz.view(1, 4, 64, 64)
            # apply to the REMAINING sources of style s
            src = Z[s][1:]
            if len(src) == 0:
                continue
            for frac in fracs:
                # chunk the decode: 23 images at 512px needs ~12 GB
                sums = {"dS": 0.0, "dR": 0.0, "ck": 0.0, "lp": 0.0}
                cnt = 0
                for ci in range(0, len(src), 4):
                    s0 = src[ci:ci + 4]
                    # per-sample FULL norm: norm(dim=1) leaves (B,1,H,W) per-pixel
                    # norms (~1.5) instead of the latent norm (~588)
                    scale = frac * s0.flatten(1).norm(dim=1).mean()
                    d = d_raw / (d_raw.norm() + 1e-12) * scale
                    z1 = s0 + d
                    rd = torch.randn_like(d)
                    rd = rd / (rd.norm() + 1e-12) * d.norm()
                    z1r = s0 + rd
                    with torch.no_grad(), torch.autocast(device_type="cuda",
                                                         dtype=torch.bfloat16):
                        y0 = vae.decode(s0).sample.float()
                        y1 = vae.decode(z1).sample.float()
                        y1r = vae.decode(z1r).sample.float()
                    tg = tgt[t]
                    f0 = csd_style(csd, y0, nm_, ns_)
                    f1 = csd_style(csd, y1, nm_, ns_)
                    fr = csd_style(csd, y1r, nm_, ns_)
                    sums["dS"] += float(((f1 * tg).sum(-1) - (f0 * tg).sum(-1)).mean()) * len(s0)
                    sums["dR"] += float(((fr * tg).sum(-1) - (f0 * tg).sum(-1)).mean()) * len(s0)
                    sums["ck"] += float((dino_feat(dino, y0, dev) *
                                         dino_feat(dino, y1, dev)).sum(-1).mean()) * len(s0)
                    sums["lp"] += float(lp(y1, y0).mean()) * len(s0)
                    cnt += len(s0)
                    del y0, y1, y1r
                    torch.cuda.empty_cache()
                dS, dR = sums["dS"] / cnt, sums["dR"] / cnt
                rows.append({"pair": f"{s}->{t}", "frac": frac, "dS": dS,
                             "dS_rand": dR, "spec": dS - dR,
                             "content_keep": sums["ck"] / cnt,
                             "lpips": sums["lp"] / cnt})
            torch.cuda.empty_cache()

    if rows:
        print(f"\n{'pair':<34}{'frac':>6}{'dS':>9}{'dS_rand':>9}{'spec':>9}"
              f"{'content':>9}{'lpips':>8}")
        for r in sorted(rows, key=lambda x: -x["spec"])[:12]:
            print(f"{r['pair']:<34}{r['frac']:>6.2f}{r['dS']:>+9.4f}"
                  f"{r['dS_rand']:>+9.4f}{r['spec']:>+9.4f}"
                  f"{r['content_keep']:>9.4f}{r['lpips']:>8.4f}")
        print("  ...")
        for r in sorted(rows, key=lambda x: x["spec"])[:4]:
            print(f"{r['pair']:<34}{r['frac']:>6.2f}{r['dS']:>+9.4f}"
                  f"{r['dS_rand']:>+9.4f}{r['spec']:>+9.4f}"
                  f"{r['content_keep']:>9.4f}{r['lpips']:>8.4f}")
        sp = [r["spec"] for r in rows]
        print(f"\n  overall specificity: mean={np.mean(sp):+.5f} "
              f"median={np.median(sp):+.5f} "
              f"frac>0={float(np.mean([x > 0 for x in sp]))*100:.1f}%", flush=True)

    Path(args.out).write_text(json.dumps({"Q1": q1, "Q2": rows}, indent=2))
    print(f"\n[Saved] {args.out}")


if __name__ == "__main__":
    main()
