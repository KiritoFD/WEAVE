"""Experiment C: does the decoder render style changes that ARE in the latent?

Fixes the same (z, z') pairs and decodes with A0 and S10, then compares the
DELTA of style similarity rather than absolute scores:

    dS_D = S(D(z')) - S(D(z))

This cancels each decoder's own reconstruction bias (A0 and S10 have different
baseline fidelity, so absolute scores are not comparable).

Direction source:
  centroid : dz = mean_latent(style_t) - mean_latent(style_s), normalised and
             rescaled to a fixed fraction of ||z0||. A real style operator emits a
             SMALL residual, so the magnitude is swept as a fraction, not used raw.
  v6       : dz from a V6 checkpoint (needs V6 rebuilt).

Controls: a random direction of the SAME norm, so any gain is attributable to
direction rather than magnitude. `dS_specificity = dS_style_dir - dS_rand_dir`.
"""
import argparse
import contextlib
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

from tools.vae_suite.decoder_prune import build_decoder, StudentDecoder
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
    ap.add_argument("--s10", default="s10_p1_l1")
    ap.add_argument("--mode", default="centroid", choices=["centroid", "v6"])
    ap.add_argument("--n-src", type=int, default=32)
    ap.add_argument("--n-ref", type=int, default=24)
    ap.add_argument("--dz-fracs", default="0.02,0.05,0.10,0.20")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-csd", action="store_true")
    ap.add_argument("--out", default=str(OUT / "expC_decoder.json"))
    args = ap.parse_args()

    dev = args.device
    cpu = dev == "cpu"

    def amp_ctx():
        if cpu:
            return contextlib.nullcontext()
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    man = json.loads(MANIFEST.read_text())
    name_of = {v: k for k, v in man["style_to_id"].items()}
    ds = RawMemmapWikiArtDataset(MANIFEST, is_training=False)
    by_style = defaultdict(list)
    for i in range(len(ds)):
        nm = name_of.get(int(ds[i]["style_id"]), "")
        if nm in TARGET_STYLES:
            by_style[nm].append(i)
    print("[Data] " + ", ".join(f"{s}={len(by_style[s])}" for s in TARGET_STYLES), flush=True)
    avail = [s for s in TARGET_STYLES if len(by_style[s]) >= 6]
    if len(avail) < 2:
        print("not enough styles")
        return

    vae = AutoencoderKL.from_pretrained(A0_DIR, torch_dtype=torch.float32).to(dev).eval()
    pqc = vae.post_quant_conv

    decs = {"A0": StudentDecoder(pqc, vae.decoder).to(dev).eval()}
    ckf = OUT / args.s10 / "decoder.pt"
    if ckf.exists():
        ck = torch.load(ckf, map_location="cpu", weights_only=False)
        d = build_decoder(ck["meta"]["widths_forward"])
        d.load_state_dict(ck["state_dict"])
        decs["S10"] = StudentDecoder(pqc, d).to(dev).eval()
        print(f"[S10] {args.s10} ({ck['meta']['params']/1e6:.3f}M)", flush=True)
    else:
        print(f"[S10] {ckf} missing -> A0 only", flush=True)

    lp = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
    for p in lp.parameters():
        p.requires_grad = False

    csd = None
    if not args.no_csd:
        try:
            import csd_model
            csd, _, _ = csd_model.load_csd(dev)
            csd = csd.float().eval()
            for p in csd.parameters():
                p.requires_grad = False
        except Exception as e:
            print(f"[CSD] unavailable: {e}", flush=True)

    from transformers import AutoModel
    dino = AutoModel.from_pretrained("facebook/dinov2-base").to(dev).eval()
    for p in dino.parameters():
        p.requires_grad = False

    # reference latents -> centroids, and CSD style targets
    rng = np.random.default_rng(3)
    cent, csd_tgt = {}, {}
    nm_ = torch.tensor(NORM_MEAN, device=dev).view(1, 3, 1, 1)
    ns_ = torch.tensor(NORM_STD, device=dev).view(1, 3, 1, 1)
    for s in avail:
        idxs = rng.choice(by_style[s], min(args.n_ref, len(by_style[s])),
                          replace=False).tolist()
        xs = torch.stack([ds[i]["image"] for i in idxs]).to(dev)
        with torch.no_grad(), amp_ctx():
            z = vae.encode(xs).latent_dist.mean.float()
        cent[s] = z.mean(0)
        if csd is not None:
            f = csd_style(csd, xs, nm_, ns_)
            csd_tgt[s] = F.normalize(f.mean(0, keepdim=True), dim=-1)
    print("[Ref] centroids: " + ", ".join(avail), flush=True)

    # source images
    per = max(2, args.n_src // len(avail))
    src = []
    for s in avail:
        src += [(i, s) for i in by_style[s][:per]]
    src = src[:args.n_src]
    Xs = torch.stack([ds[i]["image"] for i, _ in src]).to(dev)
    with torch.no_grad(), amp_ctx():
        Zs = vae.encode(Xs).latent_dist.mean.float()
    print(f"[Src] {len(src)} sources", flush=True)

    fracs = [float(x) for x in args.dz_fracs.split(",")]

    def measure_one(dec, z0, dz, s, t, frac):
        z1 = z0 + dz
        rd = torch.randn_like(dz)
        rd = rd / (rd.norm() + 1e-12) * dz.norm()
        z1r = z0 + rd
        with amp_ctx():
            y0 = dec(z0).float()
            y1 = dec(z1).float()
            y1r = dec(z1r).float()
        rec = {"style": s, "target": t, "frac": frac,
               "dz_frac": float(dz.norm() / (z0.norm() + 1e-12)),
               "lpips_style_dir": float(lp(y1, y0).mean()),
               "lpips_rand_dir": float(lp(y1r, y0).mean())}
        if csd is not None:
            tg = csd_tgt[t]
            f0 = csd_style(csd, y0, nm_, ns_)
            f1 = csd_style(csd, y1, nm_, ns_)
            fr = csd_style(csd, y1r, nm_, ns_)
            rec["dS_style_dir"] = float((f1 * tg).sum() - (f0 * tg).sum())
            rec["dS_rand_dir"] = float((fr * tg).sum() - (f0 * tg).sum())
        c0 = dino_feat(dino, y0, dev)
        c1 = dino_feat(dino, y1, dev)
        rec["dino_content_keep"] = float((c0 * c1).sum())
        return rec

    results = {}
    for name, dec in decs.items():
        rows = []
        with torch.no_grad():
            for k, (i, s) in enumerate(src):
                t = avail[(avail.index(s) + 1) % len(avail)]
                d_raw = (cent[t] - cent[s]).view(1, 4, 64, 64)
                z0 = Zs[k:k + 1]
                for frac in fracs:
                    dz = d_raw / (d_raw.norm() + 1e-12) * (frac * z0.norm())
                    rows.append(measure_one(dec, z0, dz, s, t, frac))
        results[name] = {}
        for frac in fracs:
            sub = [r for r in rows if r["frac"] == frac]
            agg = {}
            for key in sub[0]:
                if key in ("style", "target", "frac"):
                    continue
                agg[key] = round(float(np.mean([r[key] for r in sub])), 5)
            if "dS_style_dir" in agg and "dS_rand_dir" in agg:
                agg["dS_specificity"] = round(agg["dS_style_dir"] - agg["dS_rand_dir"], 5)
            results[name][str(frac)] = agg
        print(f"\n[{name}]", flush=True)
        for frac, agg in results[name].items():
            print(f"  frac={frac}: {json.dumps(agg)}", flush=True)

    print("\n" + "=" * 104)
    print("EXPERIMENT C: decoder response to a latent style direction")
    print("=" * 104)
    keys = ["dS_style_dir", "dS_rand_dir", "dS_specificity",
            "lpips_style_dir", "lpips_rand_dir", "dino_content_keep"]
    print(f"{'model':<6}{'frac':>7}" + "".join(f"{k:>19}" for k in keys))
    for name, byf in results.items():
        for frac, agg in byf.items():
            print(f"{name:<6}{frac:>7}" + "".join(
                f"{agg.get(k, float('nan')):>19.4f}" for k in keys))
    if "A0" in results and "S10" in results:
        print("\ndelta (S10 - A0) per frac:")
        for frac in results["A0"]:
            if frac not in results["S10"]:
                continue
            parts = []
            for k in keys:
                a = results["A0"][frac].get(k)
                b = results["S10"][frac].get(k)
                if a is not None and b is not None:
                    parts.append(f"{k}={b-a:+.4f}")
            print(f"  frac={frac}: " + " ".join(parts))

    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[Saved] {args.out}")


if __name__ == "__main__":
    main()
