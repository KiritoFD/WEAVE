"""Frozen feature extractors used ONLY for measurement.

Nothing in src/utils/features is differentiable guidance and nothing here may
be used as a training loss. DINO / CLIP / CSD / LPIPS are referees, not
teachers; mixing them into optimisation is how earlier rounds produced
"metrics improved, style did not".
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]


class FeatureStack:
    """Loads CSD (ViT-L), DINOv2-small, CLIP ViT-B/32 and LPIPS once (3060-safe micro-batched)."""

    def __init__(self, device: str = "cuda"):
        from src.utils import _local_eval_refs as ref
        self._ref = ref
        self.device = device
        self.csd, _, _ = ref.csd_model.load_csd(device)
        self.dino = ref.load_dino_model(device=device)
        import lpips
        self.lpips = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        from transformers import CLIPModel, CLIPProcessor
        import paths
        cache = str(paths.eval_cache_root("hf") / "hub")
        self.clip = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32", local_files_only=True,
            cache_dir=cache).to(device).eval()
        self.clip_proc = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32", local_files_only=True, cache_dir=cache)
        self.csd_prototypes = ref.csd_prototypes()

    @torch.inference_mode()
    def extract(self, images, *, batch: int = 8) -> dict:
        ref = self._ref
        csd = ref.csd_model.feats_pils(self.csd, images, batch=batch, device=self.device)
        csd = csd / np.linalg.norm(csd, axis=1, keepdims=True)
        dino, _ = ref.encode_dino(self.dino, images, batch=batch, device=self.device)
        dino = F.normalize(dino.float(), dim=-1)
        clip_chunks = []
        for s in range(0, len(images), batch):
            inputs = self.clip_proc(images=images[s:s + batch], return_tensors="pt").to(self.device)
            out = self.clip.get_image_features(**inputs)
            c = out if torch.is_tensor(out) else out.pooler_output
            if c.shape[-1] != 512:
                raise ValueError("CLIP projected embedding must be 512-d")
            clip_chunks.append(F.normalize(c.float(), dim=-1))
        clip = torch.cat(clip_chunks, dim=0)
        return {"csd": csd, "dino": dino, "clip": clip}

    @torch.inference_mode()
    def lpips_vs(self, generated_u8: torch.Tensor, reference_rgb: torch.Tensor, *, batch: int = 8) -> torch.Tensor:
        ref = self._ref
        outs = []
        for s in range(0, len(generated_u8), batch):
            g_sub = generated_u8[s:s + batch].float().div(255.0).to(self.device)
            r_sub = reference_rgb[s:s + batch].to(self.device)
            outs.append(self.lpips(ref.to_lpips_input(g_sub), ref.to_lpips_input(r_sub)).flatten())
        return torch.cat(outs, dim=0)


def reference_bank(device: str = "cuda"):
    """Frozen per-style DINO/CLIP reference features for scoring."""
    refs = torch.load(REPO / "eval_protocol/cache/references_v1_features.pt",
                      map_location="cpu", weights_only=False)
    dino = [F.normalize(refs["styles"][s]["dino_cls"].float(), dim=-1).to(device)
            for s in STYLES]
    cp = refs["clip_prototypes"]
    cp = torch.stack([cp[s] for s in STYLES]) if isinstance(cp, dict) else cp
    clip = F.normalize(cp.float().reshape(5, 512), dim=-1).to(device)
    return {"dino": dino, "clip": clip}


__all__ = ["FeatureStack", "STYLES", "reference_bank"]
