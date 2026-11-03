"""Route B: content-aligned pseudo training endpoints.

Motivation (review doc 路线B): the current FM endpoint z_1* uses the HF
coefficients of a DIFFERENT, spatially unrelated painting. The MSE-optimal
velocity then regresses to the conditional mean over many misaligned targets,
averaging out brushwork. Instead we want endpoints T(z_c, s) produced by a
content-preserving stylisation transform T, so the model learns "what changes
when THIS scene is stylised".

This module provides:
  * ``TeacherAdapter``: pluggable interface for external style-transfer
    teachers (StyTR2 / AesPA-Net / AdaAttN / an existing WEAVE checkpoint).
  * ``FallbackStatsTeacher``: no-teacher degradation path (image-space
    AdaIN + edge-weighted detail transfer). Weak, but keeps the pipeline
    runnable; clearly marked as a fallback, not a research claim.
  * Content-structure consistency scoring (self-similarity + edge IoU) used
    to filter bad pseudo targets.
  * On-disk latent cache format + lookup, consumed by the trainer when
    ``bridge.endpoint_mode`` is "pseudo" or "mixed".

Cache layout (all under one directory):
    manifest.json         {"version": 1, "count": N, "score_threshold": ...}
    shard_000000.pt       {"latent": float16 [N,4,H,W],
                           "content_key": [str], "style_id": [int],
                           "score": float32 [N]}
content_key is "<content_style_id>/<content_index>" matching dataset indices.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Protocol, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from features import FrozenFeatureExtractor, patch_self_similarity, sobel_edges

CACHE_VERSION = 1


# ---------------------------------------------------------------------------
# Teacher interface.
# ---------------------------------------------------------------------------

class TeacherAdapter(Protocol):
    """Pluggable style-transfer teacher. Implementations wrap external models."""

    name: str

    def stylize(self, content_img: torch.Tensor, style_img: torch.Tensor) -> torch.Tensor:
        """content_img/style_img: (B, 3, H, W) in [0,1] -> stylised (B, 3, H, W)."""
        ...


class FallbackStatsTeacher:
    """Degradation path when no external teacher is available.

    Image-space AdaIN (global mean/std) plus edge-weighted high-frequency
    transfer from the style image. This is intentionally simple and weak —
    use mixed mode (which keeps real-target distribution matching) whenever
    relying on it, and prefer a real teacher on the server.
    """

    name = "fallback_stats"

    def __init__(self, hf_alpha: float = 0.3) -> None:
        self.hf_alpha = float(hf_alpha)

    @torch.no_grad()
    def stylize(self, content_img: torch.Tensor, style_img: torch.Tensor) -> torch.Tensor:
        c = content_img.float()
        s = F.interpolate(style_img.float(), size=c.shape[-2:], mode="bilinear", align_corners=False)
        # Global AdaIN colour/tone match.
        c_mean = c.mean(dim=(2, 3), keepdim=True)
        c_std = c.std(dim=(2, 3), keepdim=True).clamp_min(1e-5)
        s_mean = s.mean(dim=(2, 3), keepdim=True)
        s_std = s.std(dim=(2, 3), keepdim=True).clamp_min(1e-5)
        out = (c - c_mean) / c_std * s_std + s_mean
        # Edge-weighted HF detail: add style high-pass only near content edges
        # so texture goes where the content has structure (layout stays content's).
        c_lp = F.avg_pool2d(F.pad(out, (1, 1, 1, 1), mode="replicate"), 3, stride=1)
        s_lp = F.avg_pool2d(F.pad(s, (1, 1, 1, 1), mode="replicate"), 3, stride=1)
        s_hf = s - s_lp
        edge = sobel_edges(c)
        edge = edge / edge.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        out = c_lp + (out - c_lp) + self.hf_alpha * edge * s_hf
        return out.clamp(0.0, 1.0).to(content_img.dtype)


class CallableTeacher:
    """Wrap an arbitrary callable (content, style) -> image as a teacher.

    Use this to mount StyTR2 / AesPA-Net / AdaAttN / a WEAVE checkpoint without
    changing pipeline code; the callable owns its own model + device.
    """

    def __init__(self, fn, name: str = "callable") -> None:
        self._fn = fn
        self.name = name

    def stylize(self, content_img: torch.Tensor, style_img: torch.Tensor) -> torch.Tensor:
        return self._fn(content_img, style_img)


# ---------------------------------------------------------------------------
# Content-structure consistency scoring.
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_content_consistency(
    pseudo_img: torch.Tensor,
    content_img: torch.Tensor,
    extractor: FrozenFeatureExtractor,
    *,
    selfsim_layer: str,
    edge_threshold: float = 0.1,
) -> torch.Tensor:
    """Per-sample content consistency score in [0, 1] (higher = better).

    Combines (a) correlation between patch self-similarity matrices of pseudo
    and content features and (b) edge-map IoU. Mirrors the DINO-C / edge-layout
    screening described in the review doc; uses the same pluggable extractor
    (tiny_conv only for tests; use dinov2/vgg16 in production).
    """
    feats_p = extractor.forward_layers(pseudo_img, [selfsim_layer])[selfsim_layer]
    feats_c = extractor.forward_layers(content_img, [selfsim_layer])[selfsim_layer]
    sim_p = patch_self_similarity(feats_p.float())
    sim_c = patch_self_similarity(feats_c.float())
    n = min(sim_p.shape[-1], sim_c.shape[-1])
    vp = F.adaptive_avg_pool1d(sim_p.flatten(1), n)
    vc = F.adaptive_avg_pool1d(sim_c.flatten(1), n)
    vp = F.normalize(vp, p=2, dim=-1)
    vc = F.normalize(vc, p=2, dim=-1)
    sim_corr = ((vp * vc).sum(-1) * 0.5 + 0.5).clamp(0.0, 1.0)  # (B,)

    ep = sobel_edges(pseudo_img.float())
    ec = sobel_edges(content_img.float())
    if ep.shape[-2:] != ec.shape[-2:]:
        ec = F.interpolate(ec, size=ep.shape[-2:], mode="bilinear", align_corners=False)
    bp = (ep > edge_threshold).float()
    bc = (ec > edge_threshold).float()
    inter = (bp * bc).flatten(1).sum(-1)
    union = ((bp + bc) > 0).float().flatten(1).sum(-1).clamp_min(1.0)
    edge_iou = inter / union  # (B,)

    return (0.5 * sim_corr + 0.5 * edge_iou).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Cache writer / reader.
# ---------------------------------------------------------------------------

class PseudoTargetCacheWriter:
    """Accumulate (content_key, style_id, latent, score) and flush to shards."""

    def __init__(self, out_dir: str | Path, *, shard_size: int = 4096) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = int(shard_size)
        self._latents: List[torch.Tensor] = []
        self._keys: List[str] = []
        self._styles: List[int] = []
        self._scores: List[float] = []
        self._shard_idx = 0
        self._total = 0

    def add(
        self,
        latent: torch.Tensor,
        content_keys: List[str],
        style_ids: List[int],
        scores: List[float],
    ) -> None:
        for i, key in enumerate(content_keys):
            self._latents.append(latent[i].detach().half().cpu())
            self._keys.append(str(key))
            self._styles.append(int(style_ids[i]))
            self._scores.append(float(scores[i]))
        while len(self._latents) >= self.shard_size:
            self._flush(self.shard_size)

    def _flush(self, n: int) -> None:
        lat = torch.stack(self._latents[:n])
        payload = {
            "latent": lat,
            "content_key": self._keys[:n],
            "style_id": self._styles[:n],
            "score": torch.tensor(self._scores[:n], dtype=torch.float32),
        }
        path = self.out_dir / f"shard_{self._shard_idx:06d}.pt"
        torch.save(payload, path)
        self._shard_idx += 1
        self._total += n
        del self._latents[:n], self._keys[:n], self._styles[:n], self._scores[:n]

    def close(self, *, score_threshold: float = 0.0, meta: Dict | None = None) -> Path:
        if self._latents:
            self._flush(len(self._latents))
        manifest = {
            "version": CACHE_VERSION,
            "count": self._total,
            "shards": self._shard_idx,
            "score_threshold": float(score_threshold),
            "meta": dict(meta or {}),
        }
        (self.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        return self.out_dir


class PseudoTargetCache:
    """Random-access lookup of cached pseudo-target latents.

    Key: (content_key: str, style_id: int) -> latent tensor (float32).
    """

    def __init__(self, cache_dir: str | Path, *, score_threshold: float = 0.0) -> None:
        self.cache_dir = Path(cache_dir)
        manifest_path = self.cache_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"PseudoTargetCache: no manifest.json in {self.cache_dir}")
        manifest = json.loads(manifest_path.read_text())
        if int(manifest.get("version", 0)) != CACHE_VERSION:
            raise ValueError(f"Unsupported pseudo-target cache version {manifest.get('version')}")
        self._index: Dict[Tuple[str, int], Tuple[Path, int]] = {}
        self._shard_cache: Dict[Path, Dict] = {}
        for shard in sorted(self.cache_dir.glob("shard_*.pt")):
            payload = torch.load(shard, map_location="cpu", weights_only=False)
            keep = payload["score"] >= float(score_threshold)
            for i, key in enumerate(payload["content_key"]):
                if bool(keep[i]):
                    self._index[(key, int(payload["style_id"][i]))] = (shard, i)
        self.count = len(self._index)

    def _load_shard(self, path: Path) -> Dict:
        if path not in self._shard_cache:
            self._shard_cache[path] = torch.load(path, map_location="cpu", weights_only=False)
        return self._shard_cache[path]

    def get(self, content_key: str, style_id: int) -> torch.Tensor | None:
        hit = self._index.get((content_key, int(style_id)))
        if hit is None:
            return None
        shard, i = hit
        return self._load_shard(shard)["latent"][i].float()

    def get_batch(
        self, content_keys: List[str], style_ids: List[int]
    ) -> Tuple[torch.Tensor | None, torch.Tensor, List[int]]:
        """Returns (stacked found latents or None, found mask (B,), found row indices)."""
        found: List[int] = []
        lats: List[torch.Tensor] = []
        mask = torch.zeros(len(content_keys), dtype=torch.bool)
        for b, (key, sid) in enumerate(zip(content_keys, style_ids)):
            t = self.get(key, sid)
            if t is not None:
                mask[b] = True
                found.append(b)
                lats.append(t)
        if not lats:
            return None, mask, found
        return torch.stack(lats), mask, found
