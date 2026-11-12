"""What exactly differs between A0 and S10 reconstructions?

LPIPS 5x worse and CSD -34% while L1/PSNR/SSIM are identical and the images look
the same. Diagnose the residual structure:
  - residual r = y - x statistics (bias / std / radial power spectrum)
  - LPIPS(y_s10, y_a0): perceptual distance between the two reconstructions
  - residual autocorrelation (is it noise or structured?)
  - band-wise residual energy (which scale carries the difference)
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
from tools.vae_suite.dataset import RawMemmapWikiArtDataset
import lpips

OUT = Path("/root/Workspace/WEAVE_VAE/runs/vae_decoder_prune")
A0_DIR = "/root/Workspace/WEAVE_VAE/runs_vae/A0/final_diffusers_vae"
MANIFEST = Path("/root/Workspace/data/wikiart_512_raw/wikiart_512_raw_manifest.json")


def haar_bands(x, levels=4):
    outs = []
    for _ in range(levels):
        a = x[:, :, 0::2, 0::2]; b = x[:, :, 1::2, 0::2]
        c = x[:, :, 0::2, 1::2]; d = x[:, :, 1::2, 1::2]
        outs.append(((a - b + c - d) * 0.5, (a + b - c - d) * 0.5, (a - b - c + d) * 0.5))
        x = (a + b + c + d) * 0.5
    return outs, x


def band_energy(bands, level):
    """bands: list (per level) of (lh, hl, hh) tuples. Mean |coef| at that level."""
    return float(torch.stack([t.abs().mean() for t in bands[level]]).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s10", default="s10_p1_l1")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--out", default=str(OUT / "residual_diag.json"))
    args = ap.parse_args()
    dev = "cuda"

    ds = RawMemmapWikiArtDataset(MANIFEST, is_training=False)
    n = min(args.n, len(ds))
    rng = np.random.default_rng(29)
    picks = rng.choice(len(ds), n, replace=False).tolist()

    vae = AutoencoderKL.from_pretrained(A0_DIR, torch_dtype=torch.float32).to(dev).eval()
    pqc = vae.post_quant_conv
    ck = torch.load(OUT / args.s10 / "decoder.pt", map_location="cpu", weights_only=False)
    s10 = build_decoder(ck["meta"]["widths_forward"])
    s10.load_state_dict(ck["state_dict"])
    s10 = StudentDecoder(pqc, s10.to(dev)).eval()

    lp = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
    for p in lp.parameters():
        p.requires_grad = False

    rows = []
    with torch.no_grad():
        for i in range(0, n, 4):
            idx = picks[i:i + 4]
            x = torch.stack([ds[p]["image"] for p in idx]).to(dev)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                z = pqc(vae.encode(x).latent_dist.mean)
                ya = vae.decoder(z).float()
                ys = s10(vae.encode(x).latent_dist.mean).float()
            ra, rs = ya - x, ys - x
            d = ys - ya
            ba, _ = haar_bands(ra)
            bs, _ = haar_bands(rs)
            bd, _ = haar_bands(d)
            row = {
                "res_std_A0": float(ra.std()),
                "res_std_S10": float(rs.std()),
                "res_bias_A0": float(ra.mean()),
                "res_bias_S10": float(rs.mean()),
                "diff_std": float(d.std()),
                "diff_bias": float(d.mean()),
                "lpips_s10_vs_a0": float(lp(ys, ya).mean()),
                "lpips_a0_vs_x": float(lp(ya, x).mean()),
                "lpips_s10_vs_x": float(lp(ys, x).mean()),
                # band energy of the A0<->S10 difference (level0=finest)
                "diff_band_L0": band_energy(bd, 0),
                "diff_band_L1": band_energy(bd, 1),
                "diff_band_L2": band_energy(bd, 2),
                "diff_band_L3": band_energy(bd, 3),
                "res_band_L0_A0": band_energy(ba, 0),
                "res_band_L0_S10": band_energy(bs, 0),
            }
            # residual spatial autocorrelation: structured vs white noise
            def ac(r):
                g = r.mean(1, keepdim=True)
                g = g - g.mean()
                v = (g * g).mean()
                sh = (g[:, :, :, 1:] * g[:, :, :, :-1]).mean()
                sv = (g[:, :, 1:, :] * g[:, :, :-1, :]).mean()
                return float(((sh + sv) / 2) / (v + 1e-12))
            row["ac_A0"] = ac(ra)
            row["ac_S10"] = ac(rs)
            row["ac_diff"] = ac(d)
            rows.append(row)
            torch.cuda.empty_cache()

    agg = {k: round(float(np.mean([r[k] for r in rows])), 6) for k in rows[0]}
    print(f"{'metric':<22}{'value':>12}")
    print("-" * 34)
    for k, v in agg.items():
        print(f"{k:<22}{v:>12.6f}")

    print("\n--- interpretation ---")
    print(f"residual std:   A0={agg['res_std_A0']:.5f}  S10={agg['res_std_S10']:.5f} "
          f"(ratio {agg['res_std_S10']/agg['res_std_A0']:.2f}x)")
    print(f"residual bias:  A0={agg['res_bias_A0']:+.6f}  S10={agg['res_bias_S10']:+.6f}")
    print(f"A0<->S10 diff:  std={agg['diff_std']:.5f} bias={agg['diff_bias']:+.6f}")
    print(f"LPIPS: A0vsX={agg['lpips_a0_vs_x']:.4f}  S10vsX={agg['lpips_s10_vs_x']:.4f}  "
          f"S10vsA0={agg['lpips_s10_vs_a0']:.4f}")
    print(f"autocorr (1=structured, 0=white noise): "
          f"A0={agg['ac_A0']:.3f} S10={agg['ac_S10']:.3f} diff={agg['ac_diff']:.3f}")
    print(f"diff band energy L0..L3: "
          f"{agg['diff_band_L0']:.5f} {agg['diff_band_L1']:.5f} "
          f"{agg['diff_band_L2']:.5f} {agg['diff_band_L3']:.5f}")

    Path(args.out).write_text(json.dumps({"n": n, "agg": agg}, indent=2))
    print(f"\n[Saved] {args.out}")


if __name__ == "__main__":
    main()
