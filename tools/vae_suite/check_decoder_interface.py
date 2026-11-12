"""Interface consistency check for S10: does it decode raw z or post_quant_conv(z)?

Three paths on the SAME 256 test images:
  A0_std   : A0 with post_quant_conv (the official decode path)
  S10_pqc  : S10 with post_quant_conv (what the fixed trainer used)
  S10_raw  : S10 WITHOUT post_quant_conv (what the old buggy path used)

If S10 was trained with post_quant_conv, S10_pqc must beat S10_raw by a wide
margin; otherwise the 0.368 LPIPS is an artefact of the interface, not pruning.
Also dumps 16 side-by-side reconstruction panels.
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
from PIL import Image

from tools.vae_suite.decoder_prune import build_decoder, StudentDecoder
from tools.vae_suite.dataset import RawMemmapWikiArtDataset
import lpips

OUT = Path("/root/Workspace/WEAVE_VAE/runs/vae_decoder_prune")
A0_DIR = "/root/Workspace/WEAVE_VAE/runs_vae/A0/final_diffusers_vae"
MANIFEST = Path("/root/Workspace/data/wikiart_512_raw/wikiart_512_raw_manifest.json")
NORM_MEAN = [0.48145466, 0.4578275, 0.40821073]
NORM_STD = [0.26862954, 0.26130258, 0.27577711]


def to_pil(t):
    x = (t.detach().clamp(-1, 1) + 1) / 2
    return Image.fromarray((x.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))


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


def csd_style(csd, img, nm, ns):
    x01 = (img.clamp(-1, 1) + 1.0) / 2.0
    x224 = F.interpolate(x01, (224, 224), mode="bicubic", align_corners=False)
    _, _, s = csd((x224 - nm) / ns)
    return F.normalize(s.float(), dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s10", default="s10_p1_l1")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--out", default=str(OUT / "interface_check.json"))
    args = ap.parse_args()
    dev = "cuda"

    ds = RawMemmapWikiArtDataset(MANIFEST, is_training=False)
    n = min(args.n, len(ds))
    rng = np.random.default_rng(23)
    picks = rng.choice(len(ds), n, replace=False).tolist()

    vae = AutoencoderKL.from_pretrained(A0_DIR, torch_dtype=torch.float32).to(dev).eval()
    pqc = vae.post_quant_conv

    ck = torch.load(OUT / args.s10 / "decoder.pt", map_location="cpu", weights_only=False)
    s10 = build_decoder(ck["meta"]["widths_forward"])
    s10.load_state_dict(ck["state_dict"])
    s10 = s10.to(dev).eval()
    print(f"[S10] {args.s10} params={ck['meta']['params']/1e6:.3f}M "
          f"holdout_l1={ck['meta'].get('holdout_l1')}", flush=True)
    print(f"[S10] checkpoint has post_quant_conv key: "
          f"{'post_quant_conv' in ck}", flush=True)

    lp = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
    for p in lp.parameters():
        p.requires_grad = False
    import csd_model
    csd, _, _ = csd_model.load_csd(dev)
    csd = csd.float().eval()
    for p in csd.parameters():
        p.requires_grad = False
    nm = torch.tensor(NORM_MEAN, device=dev).view(1, 3, 1, 1)
    ns = torch.tensor(NORM_STD, device=dev).view(1, 3, 1, 1)

    acc = {k: {"l1": [], "lpips": [], "psnr": [], "ssim": [], "csd": []}
           for k in ("A0_std", "S10_pqc", "S10_raw")}
    panels = []

    with torch.no_grad():
        for i in range(0, n, args.bs):
            idx = picks[i:i + args.bs]
            x = torch.stack([ds[p]["image"] for p in idx]).to(dev)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                z = vae.encode(x).latent_dist.mean
                zp = pqc(z)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                y_a0 = vae.decoder(zp).float()
                y_pqc = s10(zp).float()
                y_raw = s10(z).float()

            fx = csd_style(csd, x, nm, ns)
            for name, y in (("A0_std", y_a0), ("S10_pqc", y_pqc), ("S10_raw", y_raw)):
                acc[name]["l1"] += [float((y - x).abs().mean())]
                acc[name]["lpips"] += [float(lp(y, x).mean())]
                acc[name]["psnr"] += [psnr(y, x)]
                acc[name]["ssim"] += [ssim_simple(y, x)]
                acc[name]["csd"] += [float((csd_style(csd, y, nm, ns) * fx).sum(-1).mean())]

            if len(panels) < 16:
                for k in range(len(idx)):
                    if len(panels) >= 16:
                        break
                    panels.append((x[k], y_a0[k], y_pqc[k], y_raw[k]))
            torch.cuda.empty_cache()

    print(f"\n{'path':<10}{'L1':>10}{'LPIPS':>10}{'PSNR':>9}{'SSIM':>9}{'CSD':>9}")
    print("-" * 58)
    res = {}
    for name, a in acc.items():
        res[name] = {k: round(float(np.mean(v)), 5) for k, v in a.items()}
        r = res[name]
        print(f"{name:<10}{r['l1']:>10.5f}{r['lpips']:>10.4f}{r['psnr']:>9.2f}"
              f"{r['ssim']:>9.4f}{r['csd']:>9.4f}")

    print(f"\nS10_pqc vs S10_raw:  dLPIPS={res['S10_pqc']['lpips']-res['S10_raw']['lpips']:+.4f} "
          f"dCSD={res['S10_pqc']['csd']-res['S10_raw']['csd']:+.4f} "
          f"dL1={res['S10_pqc']['l1']-res['S10_raw']['l1']:+.5f}", flush=True)
    verdict = ("S10 was trained WITH post_quant_conv (pqc path clearly better)"
               if res["S10_pqc"]["lpips"] < res["S10_raw"]["lpips"] - 0.02
               else "S10 behaves better on the RAW path -> interface mismatch suspected")
    print(f"\n[Verdict] {verdict}", flush=True)

    # 16 panels: x | A0 | S10_pqc | S10_raw
    W = H = 320
    canvas = Image.new("RGB", (W * 4, H * 16))
    for r, (x_, a_, p_, w_) in enumerate(panels):
        for c, t in enumerate((x_, a_, p_, w_)):
            canvas.paste(to_pil(t).resize((W, H)), (c * W, r * H))
    canvas.save(OUT / "interface_panels.png")
    print(f"[Saved] {OUT/'interface_panels.png'} (cols: x | A0_std | S10_pqc | S10_raw)",
          flush=True)

    Path(args.out).write_text(json.dumps({"n": n, "results": res,
                                          "verdict": verdict}, indent=2))
    print(f"[Saved] {args.out}")


if __name__ == "__main__":
    main()
