"""Experiment A + B for the S10 decoder (no training, A0 encoder frozen).

A) What real information does the decoder lose?
   Compare x vs D_A0(E(x)) vs D_S10(E(x)) on the same real paintings:
   global colour/brightness, mid-scale structure, HF stroke correlation, edge
   recall, CSD style preservation, LPIPS/PSNR/SSIM, plus per-style breakdown.

B) Does the decoder respond to known appearance changes?
   Build controlled changes of the SAME painting (colour shift, line strength,
   texture), encode both, and compare the real change x1-x with the decoded
   change D(z1)-D(z0) per frequency band. This isolates whether the decoder
   suppresses appearance variation already present in the latent.
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

from tools.vae_suite.decoder_prune import build_decoder, StudentDecoder
from tools.vae_suite.dataset import RawMemmapWikiArtDataset
import lpips

OUT = Path("/root/Workspace/WEAVE_VAE/runs/vae_decoder_prune")
A0_DIR = "/root/Workspace/WEAVE_VAE/runs_vae/A0/final_diffusers_vae"
MANIFEST = Path("/root/Workspace/data/wikiart_512_raw/wikiart_512_raw_manifest.json")
NORM_MEAN = [0.48145466, 0.4578275, 0.40821073]
NORM_STD = [0.26862954, 0.26130258, 0.27577711]


# ---------------------------------------------------------------- metrics
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


def haar_bands(x, levels=3):
    outs = []
    for _ in range(levels):
        a = x[:, :, 0::2, 0::2]; b = x[:, :, 1::2, 0::2]
        c = x[:, :, 0::2, 1::2]; d = x[:, :, 1::2, 1::2]
        outs.append(((a - b + c - d) * 0.5, (a + b - c - d) * 0.5, (a - b - c + d) * 0.5))
        x = (a + b + c + d) * 0.5
    return outs


def band_corr(y, x):
    """Correlation per level (0=finest) averaged over LH/HL/HH."""
    out = []
    for so, sx in zip(haar_bands(y), haar_bands(x)):
        cs = []
        for a, b in zip(so, sx):
            a = a.flatten() - a.mean(); b = b.flatten() - b.mean()
            cs.append(float((a * b).sum() / (a.norm() * b.norm() + 1e-8)))
        out.append(float(np.mean(cs)))
    return out


def edge_map(x):
    x01 = (x.clamp(-1, 1) + 1) / 2
    g = x01.mean(1, keepdim=True)
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                      device=x.device).view(1, 1, 3, 3)
    gx = F.conv2d(g, kx, padding=1)
    gy = F.conv2d(g, kx.transpose(-1, -2), padding=1)
    return (gx * gx + gy * gy).sqrt()


def edge_stats(y, x):
    ey, ex = edge_map(y), edge_map(x)
    cy = ey.flatten() - ey.mean(); cx = ex.flatten() - ex.mean()
    corr = float((cy * cx).sum() / (cy.norm() * cx.norm() + 1e-8))
    # recall: fraction of strong true edges where the reconstruction also has an edge
    thr = ex.flatten().quantile(0.90)
    m = ex.flatten() > thr
    recall = float((ey.flatten()[m] > ey.flatten().quantile(0.80)).float().mean())
    return corr, recall


def colour_stats(y, x):
    """Per-channel mean/std + luminance histogram L1 distance."""
    ry, rx = (y.clamp(-1, 1) + 1) / 2, (x.clamp(-1, 1) + 1) / 2
    d_mean = float((ry.mean((2, 3)) - rx.mean((2, 3))).abs().mean())
    d_std = float((ry.std((2, 3)) - rx.std((2, 3))).abs().mean())
    ly = (0.299 * ry[:, 0] + 0.587 * ry[:, 1] + 0.114 * ry[:, 2]).flatten()
    lx = (0.299 * rx[:, 0] + 0.587 * rx[:, 1] + 0.114 * rx[:, 2]).flatten()
    hy = torch.histc(ly, bins=64, min=0, max=1); hx = torch.histc(lx, bins=64, min=0, max=1)
    hy = hy / hy.sum(); hx = hx / hx.sum()
    return d_mean, d_std, float((hy - hx).abs().sum())


def csd_style(csd, img, nm, ns):
    x01 = (img.clamp(-1, 1) + 1.0) / 2.0
    x224 = F.interpolate(x01, (224, 224), mode="bicubic", align_corners=False)
    _, _, s = csd((x224 - nm) / ns)
    return F.normalize(s.float(), dim=-1)


# ---------------------------------------------------------------- changes for B
def make_changes(x):
    """Controlled appearance changes of the SAME image."""
    x01 = (x.clamp(-1, 1) + 1) / 2
    out = {}
    # (1) colour shift: warm/cool temperature
    warm = torch.tensor([1.08, 1.00, 0.90], device=x.device).view(1, 3, 1, 1)
    out["colour_warm"] = (x01 * warm).clamp(0, 1) * 2 - 1
    # (2) line strength: add edge energy (sharpen strokes)
    e = edge_map(x)
    e = e / (e.amax((2, 3), keepdim=True) + 1e-6)
    out["line_boost"] = (x + 0.35 * (e - e.mean()) * 2).clamp(-1, 1)
    # (3) texture: fine-scale Haar energy boost
    bands = haar_bands(x, levels=1)[0]
    hf = bands[0] + bands[1] + bands[2]
    hf_up = F.interpolate(hf, size=x.shape[-2:], mode="nearest")
    out["texture_boost"] = (x + 0.30 * hf_up).clamp(-1, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s10", default="s10_p1_l1")
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-csd", action="store_true", help="skip CSD (CPU runs)")
    ap.add_argument("--bs", type=int, default=4, help="inner batch for GPU memory")
    ap.add_argument("--out", default=str(OUT / "expAB_decoder.json"))
    args = ap.parse_args()

    dev = args.device
    cpu = dev == "cpu"

    def amp_ctx():
        """autocast is a big win on GPU, harmful/slow on CPU."""
        import contextlib
        if cpu:
            return contextlib.nullcontext()
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    ds = RawMemmapWikiArtDataset(MANIFEST, is_training=False)   # held-out test split
    n = min(args.n, len(ds))
    rng = np.random.default_rng(11)
    picks = rng.choice(len(ds), n, replace=False).tolist()
    print(f"[Data] {n} images from the TEST split (never trained on) dev={dev}", flush=True)

    X, styles = [], []
    for p in picks:
        d = ds[p]
        X.append(d["image"])
        styles.append(int(d["style_id"]))
    X = torch.stack(X).to(dev)

    vae = AutoencoderKL.from_pretrained(A0_DIR, torch_dtype=torch.float32).to(dev).eval()
    pqc = vae.post_quant_conv
    dec_a0 = vae.decoder

    ckf = OUT / args.s10 / "decoder.pt"
    if not ckf.exists():
        print(f"[S10] missing {ckf}; run A0-only", flush=True)
        dec_s10 = None
    else:
        ck = torch.load(ckf, map_location="cpu", weights_only=False)
        d = build_decoder(ck["meta"]["widths_forward"])
        d.load_state_dict(ck["state_dict"])
        dec_s10 = StudentDecoder(pqc, d).to(dev).eval()
        print(f"[S10] loaded {args.s10} params={ck['meta']['params']/1e6:.3f}M", flush=True)

    decs = {"A0": StudentDecoder(pqc, dec_a0).to(dev).eval()}
    if dec_s10 is not None:
        decs["S10"] = dec_s10

    lp = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
    for p in lp.parameters():
        p.requires_grad = False
    csd = None
    try:
        if args.no_csd:
            raise RuntimeError("skipped by --no-csd")
        import csd_model
        csd, _, _ = csd_model.load_csd(dev)
        csd = csd.float().eval()
        for p in csd.parameters():
            p.requires_grad = False
        nm = torch.tensor(NORM_MEAN, device=dev).view(1, 3, 1, 1)
        ns = torch.tensor(NORM_STD, device=dev).view(1, 3, 1, 1)
    except Exception as e:
        print(f"[CSD] unavailable: {e}", flush=True)
        csd = None

    def encode_chunked(x, bs=None):
        """Encode in chunks: encoding all 96 images at 512px needs ~13 GB."""
        b = bs or args.bs
        outs = []
        with torch.no_grad():
            for i in range(0, len(x), b):
                with amp_ctx():
                    outs.append(vae.encode(x[i:i + b]).latent_dist.mean.float())
                if not cpu:
                    torch.cuda.empty_cache()
        return torch.cat(outs)

    # ---------------- Experiment A ----------------
    print("\n" + "=" * 96)
    print("EXPERIMENT A: what does the decoder lose on real paintings?")
    print("=" * 96, flush=True)

    Z = encode_chunked(X)

    resA = {}
    per_style = defaultdict(lambda: defaultdict(list))
    for name, dec in decs.items():
        acc = defaultdict(list)
        with torch.no_grad():
            for i in range(0, n, args.bs):
                x = X[i:i + args.bs]; z = Z[i:i + args.bs]
                with amp_ctx():
                    y = dec(z)
                y = y.float()
                acc["lpips"] += [float(lp(y, x).mean())]
                for k in range(len(x)):
                    yk, xk = y[k:k + 1], x[k:k + 1]
                    acc["psnr"] += [psnr(yk, xk)]
                    acc["ssim"] += [ssim_simple(yk, xk)]
                    dm, ds_, hl = colour_stats(yk, xk)
                    acc["d_colour_mean"] += [dm]
                    acc["d_colour_std"] += [ds_]
                    acc["lum_hist_L1"] += [hl]
                    bc = band_corr(yk, xk)
                    for lv, v in enumerate(bc):
                        acc[f"hf_corr_L{lv}"] += [v]
                    ec, er = edge_stats(yk, xk)
                    acc["edge_corr"] += [ec]
                    acc["edge_recall"] += [er]
                    per_style[styles[i + k]][name].append(
                        (psnr(yk, xk), ssim_simple(yk, xk), bc[0]))
                if csd is not None:
                    fy = csd_style(csd, y, nm, ns); fx = csd_style(csd, x, nm, ns)
                    acc["csd_keep"] += [float((fy * fx).sum(-1).mean())]
                if not cpu:
                    torch.cuda.empty_cache()
        resA[name] = {k: round(float(np.mean(v)), 5) for k, v in acc.items()}
        print(f"\n[{name}]", flush=True)
        for k, v in resA[name].items():
            print(f"   {k:<16}{v:>10.5f}", flush=True)

    # per-style: how much worse is S10 than A0
    if "S10" in decs:
        print("\n--- per-style degradation (S10 vs A0, PSNR) ---", flush=True)
        style_rows = []
        for s, d in per_style.items():
            if "A0" in d and "S10" in d:
                a = np.mean([t[0] for t in d["A0"]]); b = np.mean([t[0] for t in d["S10"]])
                style_rows.append((s, a, b, b - a, len(d["A0"])))
        style_rows.sort(key=lambda r: r[3])
        print(f"  {'style':>6}{'A0 PSNR':>10}{'S10 PSNR':>10}{'delta':>9}{'n':>5}")
        for s, a, b, dl, cnt in style_rows[:6]:
            print(f"  {s:>6}{a:>10.2f}{b:>10.2f}{dl:>+9.2f}{cnt:>5}")
        print("  ...")
        for s, a, b, dl, cnt in style_rows[-4:]:
            print(f"  {s:>6}{a:>10.2f}{b:>10.2f}{dl:>+9.2f}{cnt:>5}")

    # ---------------- Experiment B ----------------
    print("\n" + "=" * 96)
    print("EXPERIMENT B: does the decoder pass through latent-encoded appearance change?")
    print("=" * 96, flush=True)

    changes = make_changes(X)
    resB = {}
    with torch.no_grad():
        Z0 = encode_chunked(X)
        for cname, Xc in changes.items():
            Z1 = encode_chunked(Xc)
            row = {}
            for name, dec in decs.items():
                cs_all, amps, lzs, realhf = [], [], [], []
                for bi in range(0, n, args.bs):
                    x = X[bi:bi + args.bs]
                    xc = Xc[bi:bi + args.bs]
                    z0 = Z0[bi:bi + args.bs]
                    z1 = Z1[bi:bi + args.bs]
                    with amp_ctx():
                        Y0 = dec(z0).float()
                        Y1 = dec(z1).float()
                    d_real = xc - x
                    d_dec = Y1 - Y0
                    for ar, ad in zip(haar_bands(d_real), haar_bands(d_dec)):
                        for a, b in zip(ar, ad):
                            a = a.flatten() - a.mean(); b = b.flatten() - b.mean()
                            cs_all.append(float((a * b).sum() /
                                                (a.norm() * b.norm() + 1e-8)))
                    amps.append(float(d_dec.abs().mean() /
                                      (d_real.abs().mean() + 1e-9)))
                    lzs.append(float((z1 - z0).abs().mean()))
                    realhf.append(float(band_corr(xc, x)[0]))
                    if not cpu:
                        torch.cuda.empty_cache()
                    del Y0, Y1, d_real, d_dec
                row[name] = {
                    "delta_corr_all_bands": round(float(np.mean(cs_all)), 4),
                    "delta_amp_ratio": round(float(np.mean(amps)), 4),
                    "latent_delta": round(float(np.mean(lzs)), 5),
                    "real_change_hf_L0": round(float(np.mean(realhf)), 4),
                }
            resB[cname] = row
            print("")
            print(f"[{cname}]", flush=True)
            for name, v in row.items():
                print(f"   {name:<6} dcorr={v['delta_corr_all_bands']:+.4f} "
                      f"amp={v['delta_amp_ratio']:.4f} |dz|={v['latent_delta']:.5f} "
                      f"realHF={v['real_change_hf_L0']:.4f}", flush=True)

    Path(args.out).write_text(json.dumps({"A": resA, "B": resB,
                                          "per_style": {str(k): {kk: float(np.mean([t[0] for t in vv]))
                                                                 for kk, vv in v.items()}
                                                        for k, v in per_style.items()}},
                                         indent=2), encoding="utf-8")
    print(f"\n[Saved] {args.out}")


if __name__ == "__main__":
    main()
