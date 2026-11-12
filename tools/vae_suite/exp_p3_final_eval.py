"""P3 final acceptance for the S10 decoder.

Four blocks, as required by the plan:
  1. reconstruction  : L1 / PSNR / SSIM / LPIPS-512 / CSD-keep on the frozen
                       held-out split, plus per-style consistency
  2. style response  : the experiment-B/C probes, to confirm refinement did not
                       turn the decoder into a "real-latent-only" renderer
  3. gradient proxy  : cos(g_T, g_S) where g = d/dz L(D(z)) for an image-domain
                       loss. A usable training surrogate needs the same direction,
                       not identical magnitudes.
  4. efficiency      : decoder forward, forward+input-backward, peak memory,
                       all measured in a separate process against A0.
"""
import argparse
import json
import sys
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


def hf_corr(y, x, level=0):
    cs = []
    for a, b in zip(haar_bands(y)[level], haar_bands(x)[level]):
        a = a.flatten() - a.mean(); b = b.flatten() - b.mean()
        cs.append(float((a * b).sum() / (a.norm() * b.norm() + 1e-8)))
    return float(np.mean(cs))


def csd_style(csd, img, nm, ns):
    x01 = (img.clamp(-1, 1) + 1.0) / 2.0
    x224 = F.interpolate(x01, (224, 224), mode="bicubic", align_corners=False)
    _, _, s = csd((x224 - nm) / ns)
    return F.normalize(s.float(), dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="A0,s10_p4_full")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--grad-n", type=int, default=2,
                    help="images for the gradient proxy; 8 OOMs at 512px")
    ap.add_argument("--skip-recon", action="store_true")
    ap.add_argument("--skip-grad", action="store_true")
    ap.add_argument("--out", default=str(OUT / "p3_final_report.json"))
    args = ap.parse_args()
    dev = "cuda"

    ds = RawMemmapWikiArtDataset(MANIFEST, is_training=False)
    n = min(args.n, len(ds))
    rng = np.random.default_rng(31)
    picks = rng.choice(len(ds), n, replace=False).tolist()

    vae = AutoencoderKL.from_pretrained(A0_DIR, torch_dtype=torch.float32).to(dev).eval()
    pqc = vae.post_quant_conv

    decs = {"A0": StudentDecoder(pqc, vae.decoder).to(dev).eval()}
    for name in args.models.split(","):
        name = name.strip()
        if name == "A0":
            continue
        ckf = OUT / name / "decoder.pt"
        if not ckf.exists():
            print(f"[{name}] missing, skip", flush=True)
            continue
        ck = torch.load(ckf, map_location="cpu", weights_only=False)
        d = build_decoder(ck["meta"]["widths_forward"])
        d.load_state_dict(ck["state_dict"])
        decs[name] = StudentDecoder(pqc, d).to(dev).eval()
        print(f"[{name}] {ck['meta']['params']/1e6:.3f}M", flush=True)

    lp = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
    for p in lp.parameters():
        p.requires_grad = False
    csd, _, _ = csd_model_load(dev)
    nm_ = torch.tensor(NORM_MEAN, device=dev).view(1, 3, 1, 1)
    ns_ = torch.tensor(NORM_STD, device=dev).view(1, 3, 1, 1)

    # ---------- 1. reconstruction ----------
    print("\n" + "=" * 90)
    print("1. RECONSTRUCTION (frozen held-out split)")
    print("=" * 90, flush=True)
    res = {}
    per_style = {}
    for name, dec in ({} if args.skip_recon else decs).items():
        acc = {"l1": [], "psnr": [], "ssim": [], "lpips": [], "csd": [], "hf0": []}
        sty = {}
        with torch.no_grad():
            for i in range(0, n, args.bs):
                idx = picks[i:i + args.bs]
                x = torch.stack([ds[p]["image"] for p in idx]).to(dev)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    # dec is a StudentDecoder: it applies post_quant_conv itself.
                    z = vae.encode(x).latent_dist.mean
                    y = dec(z).float()
                fx = csd_style(csd, x, nm_, ns_)
                for k in range(len(idx)):
                    yk, xk = y[k:k + 1], x[k:k + 1]
                    acc["l1"].append(float((yk - xk).abs().mean()))
                    acc["psnr"].append(psnr(yk, xk))
                    acc["ssim"].append(ssim_simple(yk, xk))
                    acc["lpips"].append(float(lp(yk, xk).mean()))
                    acc["hf0"].append(hf_corr(yk, xk))
                    acc["csd"].append(float((csd_style(csd, yk, nm_, ns_) *
                                             fx[k:k + 1]).sum(-1).mean()))
                    sid = int(ds[idx[k]]["style_id"])
                    sty.setdefault(sid, []).append(psnr(yk, xk))
                torch.cuda.empty_cache()
        res[name] = {k: round(float(np.mean(v)), 5) for k, v in acc.items()}
        per_style[name] = {s: round(float(np.mean(v)), 2) for s, v in sty.items()}
        print(f"[{name}] " + " ".join(f"{k}={v:.5f}" for k, v in res[name].items()),
              flush=True)

    # ---------- 3. gradient proxy ----------
    print("\n" + "=" * 90)
    print("3. GRADIENT PROXY  cos(d/dz L(D(z)))  for an image-domain LPIPS loss")
    print("=" * 90, flush=True)
    grads = {}
    if args.skip_grad:
        print("  skipped", flush=True)
    xg = torch.stack([ds[p]["image"] for p in picks[:args.grad_n]]).to(dev)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        # raw latent: StudentDecoder applies post_quant_conv internally
        zg0 = vae.encode(xg).latent_dist.mean.detach()
    for name, dec in decs.items():
        z = zg0.clone().requires_grad_(True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            y = dec(z)
        l = lp(y.float(), xg).mean()
        g, = torch.autograd.grad(l, z)
        grads[name] = g.detach().float()
        print(f"  [{name}] ||g||={grads[name].norm():.6f}  loss={float(l):.5f}",
              flush=True)
    names = list(grads)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = grads[names[i]].flatten(), grads[names[j]].flatten()
            cos = float(F.cosine_similarity(a, b, dim=0))
            print(f"  cos(g_{names[i]}, g_{names[j]}) = {cos:+.4f}", flush=True)

    # ---------- 4. efficiency ----------
    print("\n" + "=" * 90)
    print("4. EFFICIENCY (decoder only, bs=1, 512px, fwd and fwd+input-grad)")
    print("=" * 90, flush=True)
    import time
    eff = {}
    for name, dec in decs.items():
        for mode in ("forward", "forward+grad"):
            z = torch.randn(1, 4, 64, 64, device=dev)
            if mode == "forward+grad":
                z.requires_grad_(True)
            for _ in range(3):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    y = dec(z)
                if mode == "forward+grad":
                    y.float().abs().mean().backward()
                torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            for _ in range(10):
                if mode == "forward+grad":
                    z.grad = None
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    y = dec(z)
                if mode == "forward+grad":
                    y.float().abs().mean().backward()
            torch.cuda.synchronize()
            dt = (time.time() - t0) / 10 * 1000
            eff[f"{name}_{mode}"] = {
                "ms_per_image": round(dt, 2),
                "peak_GB": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            }
            print(f"  {name:<18}{mode:<14}{dt:7.2f} ms  "
                  f"peak={eff[f'{name}_{mode}']['peak_GB']:.2f} GB", flush=True)
    if "A0_forward" in eff:
        for k, v in eff.items():
            if k.startswith("A0"):
                continue
            base = eff.get("A0_" + k.split("_", 1)[1])
            if base:
                print(f"  speedup {k}: {base['ms_per_image']/v['ms_per_image']:.2f}x",
                      flush=True)

    # ---------- per-style spread ----------
    if len(per_style) >= 2 and "A0" in per_style:
        print("\n--- per-style PSNR: worst gaps vs A0 ---", flush=True)
        common = set(per_style["A0"])
        for k in per_style:
            common &= set(per_style[k])
        rows = []
        for s in common:
            row = [per_style[k][s] for k in per_style]
            rows.append((s, row))
        rows.sort(key=lambda r: r[1][-1] - r[1][0])
        print(f"  {'style':>7}" + "".join(f"{k:>14}" for k in per_style))
        for s, row in rows[:5] + [("...", [])] + rows[-3:]:
            if s == "...":
                print("  ...")
                continue
            print(f"  {s:>7}" + "".join(f"{v:>14.2f}" for v in row))

    Path(args.out).write_text(json.dumps({"reconstruction": res,
                                          "per_style_psnr": per_style,
                                          "efficiency": eff,
                                          "n": n}, indent=2))
    print(f"\n[Saved] {args.out}")


def csd_model_load(dev):
    import csd_model
    m, _, _ = csd_model.load_csd(dev)
    m = m.float().eval()
    for p in m.parameters():
        p.requires_grad = False
    return m, None, None


if __name__ == "__main__":
    main()
