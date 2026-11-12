"""WEAVE Art-VAE Suite: B0 Benchmark & P0 Hard Rejection Threshold Evaluation.

Evaluates AutoencoderKL on the frozen 2048-image independent validation set:
1. Reconstruction Quality:
   - LPIPS (AlexNet)
   - PSNR (dB)
   - SSIM
   - L1 error
2. Posterior Distribution & Stability:
   - Mean posterior variance
   - Mean reconstruction error vs Sample reconstruction error
   - NaN / Inf check
3. P0 Hard Gate Acceptance against B0 (SD1.5 AutoencoderKL Baseline):
   - LPIPS <= 1.08 * B0
   - PSNR >= B0 - 0.5 dB
   - Posterior variance stability
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from diffusers import AutoencoderKL
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import lpips

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


from tools.vae_suite.dataset import RawMemmapWikiArtDataset


class ValDataset2048(Dataset):
    def __init__(self, manifest_path: Path, image_size: int = 512):
        self.manifest_path = Path(manifest_path)
        if "raw_manifest" in self.manifest_path.name or "wikiart_512_raw" in str(self.manifest_path):
            self.dataset = RawMemmapWikiArtDataset(self.manifest_path, is_training=False)
            self.use_raw = True
            self.items = [self.dataset.records_by_idx[idx] for idx in self.dataset.indices]
        else:
            self.use_raw = False
            self.items = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.image_size = image_size
            self.tf = T.Compose([
                T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), # [-1, 1]
            ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        if self.use_raw:
            return self.dataset[idx]
        item = self.items[idx]
        p = Path(item["rel_path"])
        if not p.is_absolute():
            p = REPO_ROOT / p
        img = Image.open(p).convert("RGB")
        tensor = self.tf(img)
        return {
            "image": tensor,
            "filename": item["filename"],
            "style": item["style"],
            "style_id": item["style_id"],
        }


def compute_psnr(x: torch.Tensor, y: torch.Tensor) -> float:
    # x, y in [-1, 1], convert to [0, 1]
    x01 = (x.clamp(-1, 1) + 1.0) / 2.0
    y01 = (y.clamp(-1, 1) + 1.0) / 2.0
    mse = torch.mean((x01 - y01) ** 2).item()
    if mse <= 1e-10:
        return 50.0
    return float(10.0 * math.log10(1.0 / mse))


def compute_ssim_simple(x: torch.Tensor, y: torch.Tensor) -> float:
    """Mean SSIM across batch using PyTorch."""
    x01 = (x.clamp(-1, 1) + 1.0) / 2.0
    y01 = (y.clamp(-1, 1) + 1.0) / 2.0
    # Window size 11
    C1 = (0.01 * 1.0) ** 2
    C2 = (0.03 * 1.0) ** 2

    mu_x = nn.functional.avg_pool2d(x01, kernel_size=11, stride=1, padding=5)
    mu_y = nn.functional.avg_pool2d(y01, kernel_size=11, stride=1, padding=5)

    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x_sq = nn.functional.avg_pool2d(x01 * x01, kernel_size=11, stride=1, padding=5) - mu_x_sq
    sigma_y_sq = nn.functional.avg_pool2d(y01 * y01, kernel_size=11, stride=1, padding=5) - mu_y_sq
    sigma_xy = nn.functional.avg_pool2d(x01 * y01, kernel_size=11, stride=1, padding=5) - mu_xy

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))
    return float(ssim_map.mean().item())


def evaluate_vae(vae_path: str, val_manifest_path: str, device: str = "cuda", batch_size: int = 8) -> dict:
    print(f"\n=== Evaluating AutoencoderKL at {vae_path} (batch_size={batch_size}) ===", flush=True)
    vae = AutoencoderKL.from_pretrained(vae_path).to(device).eval()
    lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    dataset = ValDataset2048(Path(val_manifest_path))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    lpips_scores = []
    psnr_scores = []
    ssim_scores = []
    l1_scores = []
    var_scores = []
    std_scores = []
    sample_l1_scores = []

    has_nan_or_inf = False
    t0 = time.time()

    with torch.inference_mode():
        for b_idx, batch in enumerate(loader):
            x = batch["image"].to(device)

            posterior = vae.encode(x).latent_dist
            mu = posterior.mean
            var = posterior.var.mean().item()
            var_scores.append(var)
            std_scores.append(mu.std().item())

            if torch.isnan(mu).any() or torch.isinf(mu).any():
                has_nan_or_inf = True

            # 1. Reconstruction from posterior mean (standard downstream evaluation)
            recon_mu = vae.decode(mu).sample
            l1_val = nn.functional.l1_loss(recon_mu, x).item()
            lp_val = lpips_fn(recon_mu.float(), x.float()).mean().item()
            psnr_val = compute_psnr(recon_mu, x)
            ssim_val = compute_ssim_simple(recon_mu, x)

            # 2. Reconstruction from posterior sample (stochastic neighborhood stability)
            sample = posterior.sample()
            recon_samp = vae.decode(sample).sample
            sample_l1 = nn.functional.l1_loss(recon_samp, x).item()

            l1_scores.append(l1_val)
            lpips_scores.append(lp_val)
            psnr_scores.append(psnr_val)
            ssim_scores.append(ssim_val)
            sample_l1_scores.append(sample_l1)

            if (b_idx + 1) % 50 == 0 or (b_idx + 1) == len(loader):
                print(f"  [{b_idx+1}/{len(loader)}] LPIPS={np.mean(lpips_scores):.4f} PSNR={np.mean(psnr_scores):.2f}dB SSIM={np.mean(ssim_scores):.4f}", flush=True)

    eval_sec = time.time() - t0

    res = {
        "model_path": str(vae_path),
        "n_evaluated": len(dataset),
        "eval_wall_sec": round(eval_sec, 1),
        "reconstruction": {
            "LPIPS": round(float(np.mean(lpips_scores)), 4),
            "PSNR_dB": round(float(np.mean(psnr_scores)), 2),
            "SSIM": round(float(np.mean(ssim_scores)), 4),
            "L1": round(float(np.mean(l1_scores)), 4),
            "sample_L1": round(float(np.mean(sample_l1_scores)), 4),
        },
        "latent_distribution": {
            "mean_posterior_var": round(float(np.mean(var_scores)), 6),
            "mean_latent_std": round(float(np.mean(std_scores)), 4),
            "has_nan_or_inf": has_nan_or_inf,
        },
    }
    print(f"[Done in {eval_sec:.1f}s] LPIPS={res['reconstruction']['LPIPS']:.4f} | PSNR={res['reconstruction']['PSNR_dB']:.2f}dB | SSIM={res['reconstruction']['SSIM']:.4f} | Var={res['latent_distribution']['mean_posterior_var']:.4f}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae-path", required=True)
    ap.add_argument("--val-manifest", default="/root/Workspace/WEAVE_VAE/val_2048_manifest.json")
    ap.add_argument("--b0-benchmark-json", default=None, help="Path to B0 report for P0 gate verification")
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()

    res = evaluate_vae(args.vae_path, args.val_manifest)

    # If b0 benchmark provided, verify P0 hard gate thresholds
    if args.b0_benchmark_json and Path(args.b0_benchmark_json).exists():
        b0_res = json.loads(Path(args.b0_benchmark_json).read_text(encoding="utf-8"))
        b0_rec = b0_res.get("P0_reconstruction_quality", b0_res.get("reconstruction", {}))
        b0_lpips = b0_rec["LPIPS"]
        b0_psnr = b0_rec["PSNR_dB"]

        cur_lpips = res["reconstruction"]["LPIPS"]
        cur_psnr = res["reconstruction"]["PSNR_dB"]

        gate_lpips_pass = bool(cur_lpips <= 1.08 * b0_lpips)
        gate_psnr_pass = bool(cur_psnr >= b0_psnr - 0.50)
        latent_std = float(res["latent_distribution"]["mean_latent_std"])
        gate_stability_pass = bool(not res["latent_distribution"]["has_nan_or_inf"] and latent_std > 0.50)

        all_passed = gate_lpips_pass and gate_psnr_pass and gate_stability_pass

        res["p0_hard_gate"] = {
            "b0_LPIPS": b0_lpips,
            "b0_PSNR": b0_psnr,
            "lpips_threshold_1.08x": round(1.08 * b0_lpips, 4),
            "psnr_threshold_minus0.5dB": round(b0_psnr - 0.50, 2),
            "latent_std": round(latent_std, 4),
            "gate_lpips_pass": gate_lpips_pass,
            "gate_psnr_pass": gate_psnr_pass,
            "gate_stability_pass": gate_stability_pass,
            "overall_p0_passed": all_passed,
        }
        print("\n=== P0 Hard Rejection Gate Check ===")
        print(f"LPIPS <= {res['p0_hard_gate']['lpips_threshold_1.08x']}: {'PASS' if gate_lpips_pass else 'FAIL'} ({cur_lpips})")
        print(f"PSNR >= {res['p0_hard_gate']['psnr_threshold_minus0.5dB']}: {'PASS' if gate_psnr_pass else 'FAIL'} ({cur_psnr})")
        print(f"Latent Std > 0.50: {'PASS' if gate_stability_pass else 'FAIL'} ({latent_std:.4f})")
        print(f"Overall Gate: {'PASSED' if all_passed else 'REJECTED'}")

    out_p = Path(args.output_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[Saved] Output written to {out_p}")


if __name__ == "__main__":
    main()
