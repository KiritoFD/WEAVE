"""Build the P1/P2/P3 pruned decoders with Taylor importance, then measure the
zero-training state: LPIPS/PSNR/SSIM, high-freq correlation, and real throughput.

This tells us how much recovery training each stage actually needs.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/root/Workspace/WEAVE_VAE")
sys.path.insert(0, "/root/Workspace/WEAVE_VAE/tools/vae_suite")

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL

from tools.vae_suite.decoder_prune import (A0_DIR, OUT, MANIFEST, A0_WIDTHS, A0_SRC,
                                           make_plan, prune_decoder, taylor_importance)
import lpips

LADDER = {
    "p1_28m": [128, 224, 352, 384],
    "p2_18m": [96, 192, 256, 320],
    "p3_10m": [96, 160, 224, 224],
    "p3b_9m": [64, 128, 224, 224],
}


def haar_hf(x, levels=3):
    outs = []
    for _ in range(levels):
        a = x[:, :, 0::2, 0::2]; b = x[:, :, 1::2, 0::2]
        c = x[:, :, 0::2, 1::2]; d = x[:, :, 1::2, 1::2]
        outs.append(((a - b + c - d) * 0.5, (a + b - c - d) * 0.5, (a - b - c + d) * 0.5))
        x = (a + b + c + d) * 0.5
    return outs


def hf_corr(y, x):
    """Correlation of high-frequency sub-bands between two images (higher=better)."""
    cs = []
    for so, sx in zip(haar_hf(y), haar_hf(x)):
        for a, b in zip(so, sx):
            a = a.flatten(); b = b.flatten()
            a = a - a.mean(); b = b - b.mean()
            cs.append(float((a * b).sum() / (a.norm() * b.norm() + 1e-8)))
    return float(np.mean(cs))


def hf_energy(y):
    return float(np.mean([a.abs().mean().item() for s in haar_hf(y) for a in s]))


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", type=int, default=256)
    ap.add_argument("--eval-n", type=int, default=96)
    ap.add_argument("--build", default="p1_28m,p2_18m,p3_10m,p3b_9m")
    args = ap.parse_args()

    device = "cuda"
    vae = AutoencoderKL.from_pretrained(A0_DIR, torch_dtype=torch.float32).to(device).eval()
    dec_a0 = vae.decoder
    a0_params = sum(p.numel() for p in dec_a0.parameters())
    lp = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    for p in lp.parameters():
        p.requires_grad = False

    # ---------- data: calibration pool + held-out eval ----------
    man = json.loads(MANIFEST.read_text())
    mmap = np.memmap(man["bin_path"], dtype=np.uint8, mode="r", shape=tuple(man["shape"]))
    idx = torch.load("/root/Workspace/WEAVE_VAE/runs/vae_style_rank20/index.pt",
                     weights_only=False)
    pool = idx["pos_list"]
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(pool))
    calib_raw = [pool[i] for i in perm[:args.calib]]
    eval_raw = [pool[i] for i in perm[args.calib:args.calib + args.eval_n]]
    print(f"[Data] calib={len(calib_raw)} eval={len(eval_raw)}", flush=True)

    def load(raws, bs=8):
        zs, xs = [], []
        with torch.no_grad():
            for i in range(0, len(raws), bs):
                arr = np.stack([np.ascontiguousarray(mmap[r]) for r in raws[i:i + bs]])
                x = torch.from_numpy(arr).permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0)
                zs.append(vae.encode(x.to(device)).latent_dist.mean.float().cpu())
                xs.append(x)
        return torch.cat(zs), torch.cat(xs)

    Zc, Xc = load(calib_raw)
    Ze, Xe = load(eval_raw)
    print(f"[Data] calib Z{tuple(Zc.shape)} eval Z{tuple(Ze.shape)}", flush=True)

    # ---------- Taylor importance (once, shared by all stages) ----------
    imp_file = OUT / "taylor_importance.pt"
    if imp_file.exists():
        imp = torch.load(imp_file, weights_only=False)
        print(f"[Importance] loaded {imp_file}", flush=True)
    else:
        print("[Importance] computing Taylor importance...", flush=True)
        imp = taylor_importance(dec_a0, make_plan(dec_a0), Zc, Xc, device)
        torch.save(imp, imp_file)
        print(f"[Importance] saved {imp_file}", flush=True)

    # ---------- evaluate helper ----------
    def evaluate(tag, dec, params):
        dec = dec.to(device).eval()
        lps, pss, sss, hcs, hes = [], [], [], [], []
        with torch.no_grad():
            for i in range(0, len(Ze), 8):
                z = Ze[i:i + 8].to(device)
                x = Xe[i:i + 8].to(device)
                y = dec(z)
                lps.append(lp(y.float(), x.float()).mean().item())
                for j in range(len(z)):
                    pss.append(psnr(y[j:j + 1], x[j:j + 1]))
                    sss.append(ssim_simple(y[j:j + 1], x[j:j + 1]))
                    hcs.append(hf_corr(y[j:j + 1], x[j:j + 1]))
                    hes.append(hf_energy(y[j:j + 1]))

        # throughput: forward-only (no_grad) and forward+input-grad (bf16, bs=2)
        z = Ze[:8].to(device)
        torch.cuda.synchronize(); t0 = time.time()
        with torch.no_grad():
            for _ in range(3):
                dec(z)
        torch.cuda.synchronize(); fwd = (time.time() - t0) / 3

        torch.cuda.empty_cache()
        zr = Ze[:2].to(device).requires_grad_(True)
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(3):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                y = dec(zr)
            g = torch.autograd.grad(y.float().sum(), zr, retain_graph=False)[0]
        torch.cuda.synchronize(); fwb = (time.time() - t0) / 3
        del zr, g
        torch.cuda.empty_cache()

        return {
            "params": params, "params_pct": round(params / a0_params * 100, 2),
            "lpips": round(float(np.mean(lps)), 4),
            "psnr": round(float(np.mean(pss)), 2),
            "ssim": round(float(np.mean(sss)), 4),
            "hf_corr": round(float(np.mean(hcs)), 4),
            "hf_energy": round(float(np.mean(hes)), 5),
            "fwd_img_per_sec": round(8 / fwd, 2),
            "fwd_bwd_img_per_sec": round(4 / fwb, 2),
        }

    results = {}
    print("\n=== A0 baseline ===", flush=True)
    results["A0"] = evaluate("A0", dec_a0, a0_params)
    print(f"  {results['A0']}", flush=True)

    names = [n.strip() for n in args.build.split(",") if n.strip()]
    for name in names:
        widths = LADDER[name]
        t0 = time.time()
        new, meta = prune_decoder(dec_a0, widths, importance=imp)
        d = OUT / name
        d.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": new.state_dict(), "meta": meta}, d / "decoder.pt")
        (d / "prune_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"\n=== {name} widths={widths} params={meta['params']/1e6:.3f} M "
              f"({time.time()-t0:.1f}s) ===", flush=True)
        r = evaluate(name, new, meta["params"])
        r["widths"] = widths
        r["zero_training"] = True
        results[name] = r
        print(f"  {r}", flush=True)
        del new
        torch.cuda.empty_cache()
        (OUT / "ladder_zero_training.json").write_text(json.dumps(results, indent=2),
                                                      encoding="utf-8")

    print("\n" + "=" * 104)
    print("ZERO-TRAINING PRUNED DECODER QUALITY (no recovery yet)")
    print("=" * 104)
    print(f"{'model':<12}{'params':>10}{'%A0':>7}{'LPIPS':>8}{'PSNR':>8}{'SSIM':>8}"
          f"{'HFcorr':>8}{'HFeng':>9}{'fwd/s':>9}{'fwd+bwd/s':>11}")
    print("-" * 104)
    for k, v in results.items():
        print(f"{k:<12}{v['params']/1e6:>9.2f}M{v['params_pct']:>6.1f}%{v['lpips']:>8.4f}"
              f"{v['psnr']:>8.2f}{v['ssim']:>8.4f}{v['hf_corr']:>8.4f}{v['hf_energy']:>9.5f}"
              f"{v['fwd_img_per_sec']:>9.2f}{v['fwd_bwd_img_per_sec']:>11.2f}")
    print("=" * 104)
    print(f"[Saved] {OUT/'ladder_zero_training.json'}")


if __name__ == "__main__":
    main()
