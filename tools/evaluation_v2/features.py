"""WEAVE Evaluator V2 - Feature Extraction & Dual Protocol Caching.

Caches precomputed representations for:
1. Sources under `paper_rgb_v1` (raw RGB source images)
2. Sources under `latent_v1` (VAE roundtrip images x_vae = VAE_dec(VAE_enc(x)))
3. Reference sets under `references_v1` (30 images per style, disjoint from test)
4. Target CSD prototypes (frozen train-pool prototypes from csd_proto.pt)
5. Real-to-Real calibration pool for CSD-MMD

Provides batch extraction for any generated PNG directory.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools" / "12h"))
sys.path.insert(0, str(REPO_ROOT / "tools" / "trt"))

import paths
import csd_model
from tools.pseudo_gt.matching import encode_dino, load_dino_model
from tools.v5.route_c import csd_prototypes, decode_trt, load_trt, load_vae
from utils.run_evaluation import _load_eval_image_tensor, to_lpips_input

STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
PROTOCOL_DIR = REPO_ROOT / "eval_protocol"
CACHE_DIR = PROTOCOL_DIR / "cache"


def ensure_dirs():
    PROTOCOL_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


class EvaluatorModels:
    """Holds shared models on GPU: CSD (ViT-L), DINOv2-small, LPIPS (Alex), CLIP (ViT-B/32)."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        print("[EvaluatorModels] Loading CSD, DINOv2, LPIPS, CLIP...", flush=True)
        t0 = time.time()
        self.csd, _, _ = csd_model.load_csd(device)
        self.dino = load_dino_model(device=device)

        import lpips
        self.lpips = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

        from transformers import CLIPModel, CLIPProcessor
        clip_cache = str(paths.eval_cache_root("hf") / "hub")
        self.clip = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32", local_files_only=True, cache_dir=clip_cache
        ).to(device).eval()
        self.clip_proc = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32", local_files_only=True, cache_dir=clip_cache
        )
        self.proto_csd = csd_prototypes()  # (5, 768)
        print(f"[EvaluatorModels] Loaded all models in {time.time()-t0:.2f}s", flush=True)

    @torch.inference_mode()
    def extract_image_features(self, pils: Sequence[Image.Image]) -> dict:
        """Extract (csd_768, dino_cls_384, dino_patch_256x384, clip_512) for a list of PIL images."""
        B = len(pils)
        # 1. CSD
        csd_feats = csd_model.feats_pils(self.csd, pils, batch=B, device=self.device)  # numpy (B, 768)

        # 2. DINO CLS and patches
        cls_t, patch_t = encode_dino(self.dino, pils, batch=B, device=self.device)
        cls_t = F.normalize(cls_t.float(), dim=-1).cpu()  # (B, 384)
        patch_t = F.normalize(patch_t.float(), dim=-1).cpu()  # (B, 256, 384)

        # 3. CLIP image feature
        clip_in = self.clip_proc(images=pils, return_tensors="pt").to(self.device)
        clip_out = self.clip.get_image_features(**clip_in)
        if hasattr(clip_out, "pooler_output") and torch.is_tensor(clip_out.pooler_output):
            clip_out = clip_out.pooler_output
        clip_feats = F.normalize(clip_out.float(), dim=-1).cpu()  # (B, 512)

        return {
            "csd": torch.from_numpy(csd_feats).float(),
            "dino_cls": cls_t,
            "dino_patch": patch_t,
            "clip": clip_feats,
        }


def build_source_features_cache(models: EvaluatorModels, force: bool = False) -> Tuple[dict, dict]:
    """Extract and cache source representations for paper_rgb_v1 and latent_v1."""
    ensure_dirs()
    rgb_cache_file = CACHE_DIR / "sources_paper_rgb_v1.pt"
    latent_cache_file = CACHE_DIR / "sources_latent_v1.pt"

    if rgb_cache_file.exists() and latent_cache_file.exists() and not force:
        print(f"[Features] Source features cache found at {CACHE_DIR}")
        rgb_data = torch.load(rgb_cache_file, map_location="cpu", weights_only=False)
        lat_data = torch.load(latent_cache_file, map_location="cpu", weights_only=False)
        return rgb_data, lat_data

    tasks_meta = json.loads((PROTOCOL_DIR / "tasks_750.json").read_text(encoding="utf-8"))
    sources = tasks_meta["sources"]
    print(f"[Features] Precomputing dual-protocol features for {len(sources)} source images...", flush=True)

    # 1. paper_rgb_v1 (Raw RGB sources)
    pils_raw = [
        Image.open(REPO_ROOT / s["source_rel_path"]).convert("RGB").resize((512, 512), Image.BICUBIC)
        for s in sources
    ]
    raw_feats = models.extract_image_features(pils_raw)

    # 2. latent_v1 (VAE roundtrip sources)
    vae = load_vae(models.device)
    trt = load_trt(vae, models.device)
    pils_vae = []
    with torch.inference_mode():
        for i in range(0, len(sources), 8):
            chunk = sources[i : i + 8]
            tensors = torch.stack([
                _load_eval_image_tensor(REPO_ROOT / s["source_rel_path"], size=512)
                for s in chunk
            ]).to(models.device)
            inp = (tensors * 2.0 - 1.0).half()
            z = (vae.encode(inp).latent_dist.mean * vae.config.scaling_factor).half()
            img_dec = decode_trt(trt, vae, z, batch=8)
            u8 = img_dec.mul(255.0).round().clamp(0, 255).to(torch.uint8).cpu().permute(0, 2, 3, 1).numpy()
            pils_vae.extend([Image.fromarray(u8[j]) for j in range(len(chunk))])
    del vae, trt
    torch.cuda.empty_cache()

    vae_feats = models.extract_image_features(pils_vae)

    # Compute spatial self-similarity matrices (SSM): P @ P.T (256x256)
    def compute_ssm(patch_tokens: torch.Tensor) -> torch.Tensor:
        # patch_tokens: (B, 256, 384)
        return torch.bmm(patch_tokens, patch_tokens.transpose(1, 2)) / patch_tokens.shape[1]

    raw_ssm = compute_ssm(raw_feats["dino_patch"])
    vae_ssm = compute_ssm(vae_feats["dino_patch"])

    rgb_cache = {
        "protocol": "paper_rgb_v1",
        "description": "Precomputed raw RGB source representations",
        "stems": [s["source_stem"] for s in sources],
        "styles": [s["source_style"] for s in sources],
        "csd": raw_feats["csd"],
        "dino_cls": raw_feats["dino_cls"],
        "dino_patch": raw_feats["dino_patch"],
        "dino_ssm": raw_ssm,
        "clip": raw_feats["clip"],
    }
    torch.save(rgb_cache, rgb_cache_file)

    latent_cache = {
        "protocol": "latent_v1",
        "description": "Precomputed VAE round-trip source representations (x_vae)",
        "stems": [s["source_stem"] for s in sources],
        "styles": [s["source_style"] for s in sources],
        "csd": vae_feats["csd"],
        "dino_cls": vae_feats["dino_cls"],
        "dino_patch": vae_feats["dino_patch"],
        "dino_ssm": vae_ssm,
        "clip": vae_feats["clip"],
    }
    torch.save(latent_cache, latent_cache_file)

    print(f"[Features] Wrote {rgb_cache_file} and {latent_cache_file}")
    return rgb_cache, latent_cache


def build_reference_features_cache(models: EvaluatorModels, force: bool = False) -> dict:
    """Precompute reference features for references_v1 (30 per style)."""
    ensure_dirs()
    ref_cache_file = CACHE_DIR / "references_v1_features.pt"
    if ref_cache_file.exists() and not force:
        return torch.load(ref_cache_file, map_location="cpu", weights_only=False)

    ref_meta = json.loads((PROTOCOL_DIR / "references_v1.json").read_text(encoding="utf-8"))
    references = ref_meta["references"]

    out = {
        "styles": {},
        "clip_prototypes": {},
        "rbf_bandwidths": {},
    }

    print("[Features] Precomputing reference features and CSD-MMD bandwidths...", flush=True)
    all_csd_for_bw = []

    for sname in STYLES:
        s_refs = references[sname]
        pils = [
            Image.open(REPO_ROOT / r["rel_path"]).convert("RGB").resize((512, 512), Image.BICUBIC)
            for r in s_refs
        ]
        feats = models.extract_image_features(pils)
        all_csd_for_bw.append(feats["csd"])

        # Reference CLIP prototype: L2-normalized mean of 30 references
        clip_proto = F.normalize(feats["clip"].mean(dim=0, keepdim=True), dim=-1)[0]
        out["clip_prototypes"][sname] = clip_proto

        out["styles"][sname] = {
            "csd": feats["csd"],               # (30, 768)
            "dino_cls": feats["dino_cls"],     # (30, 384)
            "dino_patch": feats["dino_patch"], # (30, 256, 384)
            "clip": feats["clip"],             # (30, 512)
        }

    # Calibrate RBF kernel bandwidth gamma = 1 / (2 * median_dist^2) across real artworks
    all_csd_cat = torch.cat(all_csd_for_bw, dim=0)  # (150, 768)
    pdist_sq = torch.cdist(all_csd_cat, all_csd_cat).pow(2)
    med_sq = float(pdist_sq.median().item())
    gamma = 1.0 / max(2.0 * med_sq, 1e-4)
    out["rbf_gamma"] = gamma
    out["rbf_median_sq_dist"] = med_sq
    print(f"[Features] Calibrated CSD-MMD RBF gamma={gamma:.4f} (median sq dist={med_sq:.4f})")

    torch.save(out, ref_cache_file)
    print(f"[Features] Wrote reference features cache to {ref_cache_file}")
    return out
