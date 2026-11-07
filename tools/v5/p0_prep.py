"""V5 P0 Preparation:
1. Build canonical validation task lists:
   - v5_eval200_tasks.json (50 sources x 4 targets = 200 cross tasks)
   - v5_oracle100_tasks.json (25 sources x 4 targets = 100 cross tasks)
2. Pre-encode 50 source images to VAE posterior mean latents (4x64x64, deterministic).
3. Pre-extract and cache real D5 18K art multi-scale latent descriptors & SWD projection references.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import paths
from wavelet import dwt2_haar

STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
STYLE_IDS = {name: i for i, name in enumerate(STYLES)}


def quantile_grid(sorted_proj: np.ndarray, num_quantiles: int = 128) -> np.ndarray:
    """Extract quantile grid along projection axis."""
    q = np.linspace(0.0, 1.0, num_quantiles)
    indices = (q * (sorted_proj.shape[0] - 1)).astype(np.int64)
    return sorted_proj[indices]


def build_tasks(test_dir: Path, max_eval_src: int = 10, max_oracle_src: int = 5):
    eval_sources = []
    oracle_sources = []

    for sid, style in enumerate(STYLES):
        s_dir = test_dir / style
        if not s_dir.exists():
            raise FileNotFoundError(f"Missing style dir: {s_dir}")
        imgs = sorted(list(s_dir.glob("*.jpg")))
        if len(imgs) < max_eval_src:
            raise RuntimeError(f"Style {style} has only {len(imgs)} < {max_eval_src} images")
        
        for i, img_p in enumerate(imgs[:max_eval_src]):
            eval_sources.append({
                "src_style": style,
                "src_style_id": sid,
                "src_stem": img_p.stem,
                "src_path": str(img_p),
                "src_idx_in_style": i,
            })
            if i < max_oracle_src:
                oracle_sources.append(eval_sources[-1])

    eval200 = []
    tid = 0
    for src in eval_sources:
        for tgt_id, tgt_style in enumerate(STYLES):
            if tgt_id == src["src_style_id"]:
                continue
            eval200.append({
                "task_id": tid,
                "src_style": src["src_style"],
                "src_style_id": src["src_style_id"],
                "src_stem": src["src_stem"],
                "src_path": src["src_path"],
                "tgt_style": tgt_style,
                "tgt_style_id": tgt_id,
            })
            tid += 1

    oracle100 = []
    tid = 0
    for src in oracle_sources:
        for tgt_id, tgt_style in enumerate(STYLES):
            if tgt_id == src["src_style_id"]:
                continue
            oracle100.append({
                "task_id": tid,
                "src_style": src["src_style"],
                "src_style_id": src["src_style_id"],
                "src_stem": src["src_stem"],
                "src_path": src["src_path"],
                "tgt_style": tgt_style,
                "tgt_style_id": tgt_id,
            })
            tid += 1

    return eval_sources, eval200, oracle100


def encode_sources(sources: List[dict], device: str = "cuda") -> Dict[str, torch.Tensor]:
    """Encode 50 source images to posterior mean latents using SD1.5 VAE."""
    from diffusers import AutoencoderKL
    print("[p0] Loading SD1.5 VAE for posterior mean encoding...", flush=True)
    
    # Try local snapshot candidates first to ensure zero network dependency
    candidates = [
        paths.asset_root() / "eval_cache" / "hf" / "modelscope" / "stabilityai_sd-vae-ft-ema" / "models" / "stabilityai--sd-vae-ft-ema" / "snapshots",
        REPO_ROOT / "eval_cache" / "hf" / "modelscope" / "stabilityai_sd-vae-ft-ema" / "models" / "stabilityai--sd-vae-ft-ema" / "snapshots",
        paths.asset_root() / "eval_cache" / "hf" / "hub" / "models--stabilityai--sd-vae-ft-ema",
    ]
    vae_dir = None
    for c in candidates:
        if c.exists():
            snaps = list(c.glob("*"))
            if snaps and (snaps[0] / "config.json").exists():
                vae_dir = snaps[0]
                break
            elif (c / "config.json").exists():
                vae_dir = c
                break

    if vae_dir is not None:
        print(f"[p0] Found local VAE at: {vae_dir}", flush=True)
        vae = AutoencoderKL.from_pretrained(str(vae_dir), local_files_only=True).to(device=device, dtype=torch.float32).eval()
    else:
        from utils.inference import download_vae_with_fallback
        vae = download_vae_with_fallback("stabilityai/sd-vae-ft-ema", device=device)
        vae = vae.to(dtype=torch.float32).eval()
    
    latents = {}
    print(f"[p0] Encoding {len(sources)} unique source images...", flush=True)
    with torch.inference_mode():
        for item in sources:
            stem = item["src_stem"]
            p = Path(item["src_path"])
            im = Image.open(p).convert("RGB").resize((512, 512), Image.BICUBIC)
            arr = np.array(im, dtype=np.float32) / 127.5 - 1.0
            x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
            dist = vae.encode(x).latent_dist
            mean_lat = dist.mean * vae.config.scaling_factor  # (1, 4, 64, 64)
            latents[stem] = mean_lat.squeeze(0).cpu().float()
    del vae
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return latents


def extract_real_art_descriptors(asset_root: Path, n_proj: int = 64, seed: int = 42) -> dict:
    """Extract Haar band statistics & patch SWD reference quantile grids for real D5 art."""
    orig_dir = asset_root / "pseudo_gt_24h" / "orig_mean_fp16"
    if not orig_dir.exists():
        raise FileNotFoundError(f"Missing real latent cache: {orig_dir}")

    rng = np.random.default_rng(seed)
    # Patch projection bases
    # For full z (8x8 patch -> 4*8*8 = 256 dim)
    R_z = rng.normal(size=(n_proj, 256)).astype(np.float32)
    R_z /= np.linalg.norm(R_z, axis=1, keepdims=True)

    # For LL band (4x4 patch -> 4*4*4 = 64 dim)
    R_ll = rng.normal(size=(n_proj, 64)).astype(np.float32)
    R_ll /= np.linalg.norm(R_ll, axis=1, keepdims=True)

    descriptors = {
        "n_proj": n_proj,
        "seed": seed,
        "R_z": torch.from_numpy(R_z),
        "R_ll": torch.from_numpy(R_ll),
        "styles": {},
    }

    for sid, style in enumerate(STYLES):
        fn = f"{sid:02d}_{style}.pt"
        p = orig_dir / fn
        print(f"[p0] Processing real art latents for {style} ({fn})...", flush=True)
        data = torch.load(p, map_location="cpu", weights_only=False)
        latents = data["latents"].float()  # (N, 4, 64, 64)
        N = latents.shape[0]

        # 1. Haar DWT bands
        with torch.no_grad():
            ll, lh, hl, hh = dwt2_haar(latents)  # each (N, 4, 32, 32)
            
            # Band channel means: (4,)
            ll_mean = ll.mean(dim=(0, 2, 3))
            lh_mean = lh.mean(dim=(0, 2, 3))
            hl_mean = hl.mean(dim=(0, 2, 3))
            hh_mean = hh.mean(dim=(0, 2, 3))

            # Band channel std: (4,)
            ll_std = ll.std(dim=(0, 2, 3))
            lh_std = lh.std(dim=(0, 2, 3))
            hl_std = hl.std(dim=(0, 2, 3))
            hh_std = hh.std(dim=(0, 2, 3))

            # Band energies:
            e_ll = (ll**2).mean().item()
            e_lh = (lh**2).mean().item()
            e_hl = (hl**2).mean().item()
            e_hh = (hh**2).mean().item()
            e_tot = e_ll + e_lh + e_hl + e_hh
            energy_share = {
                "LL": e_ll / e_tot,
                "LH": e_lh / e_tot,
                "HL": e_hl / e_tot,
                "HH": e_hh / e_tot,
            }

            # Channel Covariance matrices: (4, 4)
            # Centered flat: (N*1024, 4)
            def compute_cov(band_t: torch.Tensor, mean_t: torch.Tensor) -> torch.Tensor:
                flat = (band_t.permute(0, 2, 3, 1).reshape(-1, 4) - mean_t.unsqueeze(0))
                return (flat.t() @ flat) / (flat.shape[0] - 1)

            cov_ll = compute_cov(ll, ll_mean)
            cov_lh = compute_cov(lh, lh_mean)
            cov_hl = compute_cov(hl, hl_mean)
            cov_hh = compute_cov(hh, hh_mean)

            # 2. Patch SWD reference quantile grids
            # Unfold z (N, 4, 64, 64) with 8x8 kernel, stride 8 -> 64 patches/img
            patches_z = F.unfold(latents, kernel_size=8, stride=8)  # (N, 256, 64)
            patches_z = patches_z.permute(0, 2, 1).reshape(-1, 256).numpy()  # (N*64, 256)
            proj_z = np.sort(patches_z @ R_z.T, axis=0)  # (N*64, n_proj), sorted
            qgrid_z = quantile_grid(proj_z, num_quantiles=128)  # (128, n_proj)

            # Unfold LL (N, 4, 32, 32) with 4x4 kernel, stride 4 -> 64 patches/img
            patches_ll = F.unfold(ll, kernel_size=4, stride=4)  # (N, 64, 64)
            patches_ll = patches_ll.permute(0, 2, 1).reshape(-1, 64).numpy()  # (N*64, 64)
            proj_ll = np.sort(patches_ll @ R_ll.T, axis=0)
            qgrid_ll = quantile_grid(proj_ll, num_quantiles=128)

        descriptors["styles"][style] = {
            "style_id": sid,
            "num_samples": N,
            "ll_mean": ll_mean,
            "ll_std": ll_std,
            "ll_cov": cov_ll,
            "lh_mean": lh_mean,
            "lh_std": lh_std,
            "lh_cov": cov_lh,
            "hl_mean": hl_mean,
            "hl_std": hl_std,
            "hl_cov": cov_hl,
            "hh_mean": hh_mean,
            "hh_std": hh_std,
            "hh_cov": cov_hh,
            "energy_share": energy_share,
            "qgrid_z": torch.from_numpy(qgrid_z.astype(np.float32)),
            "qgrid_ll": torch.from_numpy(qgrid_ll.astype(np.float32)),
        }
        print(f"  {style}: N={N}, Energy: LL={energy_share['LL']*100:.1f}%, LH={energy_share['LH']*100:.1f}%, HL={energy_share['HL']*100:.1f}%, HH={energy_share['HH']*100:.1f}%")

    return descriptors


def main():
    out_dir = REPO_ROOT / "runs" / "v5"
    cache_dir = out_dir / "cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    test_dir = REPO_ROOT / "data" / "test"
    asset_root = paths.asset_root()

    print("[p0] 1. Building task splits...", flush=True)
    eval_sources, eval200, oracle100 = build_tasks(test_dir, max_eval_src=10, max_oracle_src=5)
    
    (out_dir / "v5_eval200_tasks.json").write_text(json.dumps(eval200, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "v5_oracle100_tasks.json").write_text(json.dumps(oracle100, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[p0] Saved v5_eval200_tasks.json ({len(eval200)} tasks) and v5_oracle100_tasks.json ({len(oracle100)} tasks)")

    print("[p0] 2. Pre-encoding source images to posterior mean latents...", flush=True)
    src_latents = encode_sources(eval_sources, device="cuda" if torch.cuda.is_available() else "cpu")
    torch.save(src_latents, cache_dir / "eval_src_latents.pt")
    print(f"[p0] Saved eval_src_latents.pt ({len(src_latents)} latents)")

    print("[p0] 3. Extracting real D5 art multi-scale descriptors...", flush=True)
    t0 = time.time()
    descriptors = extract_real_art_descriptors(asset_root)
    torch.save(descriptors, cache_dir / "d5_style_multiscale_descriptors.pt")
    print(f"[p0] Saved d5_style_multiscale_descriptors.pt in {time.time() - t0:.1f}s")
    print("[p0] P0 Preparation COMPLETE!", flush=True)


if __name__ == "__main__":
    main()
