"""Fast, protocol-equivalent full evaluation.

This is an evaluation-side companion to ``utils/run_evaluation.py``.  It keeps
the metric definition identical (same image size, same reference policy, same
CLIP / LPIPS / DINOv2 implementation) but:

* runs generation + CLIP + LPIPS + DINO in one process, so no PNG round trip;
* batches ODE / VAE / metric work (``--batch`` / ``--metric-batch``);
* quantises decoded images once, then keeps uint8 pixels in RAM;
* writes PNGs from a background thread pool while metrics keep the GPU busy
  (``--workers``, default async; ``--sync-save`` writes inline);
* reuses the on-disk source latent cache produced by the legacy evaluator;
* optionally caches reference / source CLIP + DINO features;
* supports ``--device cpu`` and ``--reuse-images`` (metrics-only parity).

Default protocol (must stay identical to ``utils/run_evaluation.py``):
test dir ``data/test``, 30 source images per style, first image of the target
style as reference, 512x512 pixels (--image-size, default 512), 8 ODE steps,
config.json full_eval section.

Usage:
  python tools/eval_fast.py --checkpoint exp/.../epoch_0010.pt \
      --output runs/fast_eval/base --device cuda --batch 32 --workers 4

  # metrics-only parity against an existing legacy eval directory
  python tools/eval_fast.py --reuse-images runs/.../base_ep10 \
      --output /tmp/parity --batch 32

  # CPU-only full pass
  python tools/eval_fast.py --checkpoint ... --output ... --device cpu --batch 8
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import random
import resource
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "tools"))
sys.path.insert(0, str(_ROOT / "tools" / "trt"))

from config_schema import load_config, resolve_full_eval_section  # noqa: E402
from utils.compute_dino_metrics import DINO_TRANSFORM, list_style_images, load_dino  # noqa: E402
from utils.inference import decode_latent, encode_image, load_vae, LGTInference  # noqa: E402
from utils.run_evaluation import (  # noqa: E402
    _clip_image_size_from_runtime,
    _clip_mean_std_from_runtime,
    _extract_clip_embeddings,
    _load_clip_from_source,
    _load_eval_image_tensor,
    _lpips_forward_safe,
    _prepare_clip_pixels,
    _source_latent_cache_hash,
    _source_path_key,
    to_lpips_input,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
REFERENCE_POLICY = "first_image_of_target_style_test_dir"
ALLOWED_CLIP_REPO = "openai/clip-vit-base-patch32"
CLIP_PROTOCOL_ERROR = (
    "本仓库只允许 openai/clip-vit-base-patch32；"
    "CLIP-L/14 口径已于 2026-11-04 废弃并删除。"
)


def require_b32_clip(value: str, where: str) -> None:
    """Hard gate: only the B/32 CLIP backbone may be used (no override, no escape hatch)."""
    v = str(value or "").strip().lower()
    if "large-patch14" in v or "vit-l/14" in v:
        raise SystemExit(f"{CLIP_PROTOCOL_ERROR} (got {where}={value!r})")
    if v == ALLOWED_CLIP_REPO or "base-patch32" in v:
        return
    raise SystemExit(f"{CLIP_PROTOCOL_ERROR} (got {where}={value!r})")


METRIC_HEADER = [
    "src_style", "tgt_style", "src_image", "gen_image",
    "clip_style", "clip_s_delta_idt", "clip_t", "clip_content", "clip_dir",
    "lpips", "clip_image_vector",
]
DINO_HEADER = [
    "src_style", "tgt_style", "src_image", "gen_image",
    "dino_style_similarity", "dino_content_similarity", "dino_structure_mse",
]


# ---------------------------------------------------------------------------
# timing helpers
# ---------------------------------------------------------------------------
class Timer:
    def __init__(self, sync_cuda: bool = False):
        self.values: dict[str, float] = {}
        self.counts: dict[str, int] = {}
        self.sync_cuda = sync_cuda
        self._t0: dict[str, float] = {}

    def start(self, key: str) -> None:
        self._sync()
        self._t0[key] = time.perf_counter()

    def stop(self, key: str) -> None:
        self._sync()
        dt = time.perf_counter() - self._t0.pop(key)
        self.values[key] = self.values.get(key, 0.0) + dt
        self.counts[key] = self.counts.get(key, 0) + 1

    def add(self, key: str, dt: float, count: int = 1) -> None:
        self.values[key] = self.values.get(key, 0.0) + float(dt)
        self.counts[key] = self.counts.get(key, 0) + int(count)

    def _sync(self) -> None:
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

    def as_dict(self) -> dict[str, float]:
        return dict(self.values)


class SaveStats:
    def __init__(self):
        self.lock = threading.Lock()
        self.encode_sec = 0.0
        self.write_sec = 0.0
        self.bytes = 0
        self.count = 0
        self.wall_sec = 0.0

    def add(self, enc: float, wr: float, nbytes: int) -> None:
        with self.lock:
            self.encode_sec += enc
            self.write_sec += wr
            self.bytes += nbytes
            self.count += 1


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_path(raw: str, base: Path) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    candidate = base / p
    if candidate.exists():
        return candidate
    return p


# ---------------------------------------------------------------------------
# protocol / discovery
# ---------------------------------------------------------------------------
def build_protocol(
    test_dir: Path,
    *,
    max_src_samples: int,
    ref_pool: str,
) -> tuple[dict[int, tuple[str, list[Path]]], list[dict], dict[int, dict]]:
    style_dirs = sorted(p for p in test_dir.iterdir() if p.is_dir())
    test_images: dict[int, tuple[str, list[Path]]] = {}
    style_meta: dict[int, dict] = {}
    for sid, style_dir in enumerate(style_dirs):
        images = list_style_images(style_dir)
        if not images:
            continue
        ref_path = images[0]
        content = images[1:] if ref_pool == "strict" else images
        rng = random.Random(42)
        rng.shuffle(content)
        if max_src_samples > 0:
            content = content[:max_src_samples]
        test_images[sid] = (style_dir.name, content)
        style_meta[sid] = {
            "name": style_dir.name,
            "dir": style_dir,
            "ref": ref_path,
            "all": images,
        }

    all_src: list[dict] = []
    for sid in sorted(test_images):
        name, paths = test_images[sid]
        for p in paths:
            all_src.append({"style_id": sid, "style": name, "path": p})

    num_styles = len(test_images)
    requests: list[dict] = []
    for item in all_src:
        for tgt in range(num_styles):
            src_stem = item["path"].stem
            out_name = f"{item['style']}_{src_stem}_to_{style_meta[tgt]['name']}.png"
            requests.append({
                "src_path": item["path"],
                "src_style_id": item["style_id"],
                "src_style": item["style"],
                "src_name": item["path"].name,
                "gen_name": out_name,
                "tgt_style_id": tgt,
                "tgt_style": style_meta[tgt]["name"],
                "same_style": item["style_id"] == tgt,
            })
    return test_images, requests, style_meta


def _request_key(row: dict) -> tuple:
    return (
        int(row.get("src_style_id", row.get("content_style_id", -1))),
        int(row.get("tgt_style_id", row.get("style_id", -1))),
        str(row.get("src_name", row.get("src_image", ""))),
        str(row.get("gen_name", row.get("gen_image", ""))),
    )


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
class MetricEngine:
    """CLIP / LPIPS / DINOv2 with the exact legacy preprocessing."""

    def __init__(self, args, test_images, style_meta, device: str):
        self.args = args
        self.device = device
        self.test_images = test_images
        self.style_meta = style_meta
        self.clip_model = None
        self.clip_proc = None
        self.lpips_fn = None
        self.dino = None
        self.clip_size = 224
        self.clip_mean = (0.48145466, 0.4578275, 0.40821073)
        self.clip_std = (0.26862954, 0.26130258, 0.27577711)
        self.text_proto: dict[int, torch.Tensor] = {}
        self.ref_clip_proto: dict[int, torch.Tensor] = {}
        self.ref_dino_cls: dict[int, torch.Tensor] = {}
        self.src_clip: dict[str, torch.Tensor] = {}
        self.src_dino_cls: dict[str, torch.Tensor] = {}
        self.src_dino_ssm: dict[str, torch.Tensor] = {}
        self.src_float: dict[str, torch.Tensor] = {}
        self.cache_hits = {"clip": False, "dino": False, "src": False}

    # -- loading ------------------------------------------------------------
    def load(self, timer: Timer) -> None:
        args = self.args
        t = time.perf_counter()
        if not args.disable_clip:
            from transformers import CLIPModel, CLIPProcessor
            self.clip_model, self.clip_proc = _load_clip_from_source(
                CLIPModel, CLIPProcessor, args.clip_model, self.device,
                local_only=args.clip_local_only,
                cache_dir=args.clip_cache_dir,
                load_processor=not args.clip_skip_processor,
            )
            self.clip_model.eval()
            self.clip_size = _clip_image_size_from_runtime(self.clip_model, self.clip_proc)
            if self.clip_proc is not None:
                self.clip_mean, self.clip_std = _clip_mean_std_from_runtime(self.clip_proc)
            if args.amp and self.device == "cuda" and self.clip_model is not None:
                self.clip_model = self.clip_model.half()
            if self.clip_proc is not None and not args.only_clip_style:
                self._prepare_text_prototypes()
        if not args.disable_lpips:
            import lpips
            self.lpips_fn = lpips.LPIPS(net=args.lpips_net, verbose=False).to(self.device).eval()
        if not args.disable_dino:
            self.dino = load_dino(args.dino_model, self.device, args.dino_cache_dir, allow_network=args.allow_network)
            if args.amp and self.device == "cuda":
                self.dino = self.dino.half()
        timer.add("load_metric_models", time.perf_counter() - t)

    def _prepare_text_prototypes(self) -> None:
        import torch.nn.functional as F
        prompts = [f"a painting in the {self.style_meta[sid]['name'].replace('_', ' ')} style"
                   for sid in sorted(self.style_meta)]
        inputs = self.clip_proc(text=prompts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            feats = _extract_clip_embeddings(self.clip_model.get_text_features(**inputs))
            feats = F.normalize(feats.float(), dim=-1)
        for sid, vec in zip(sorted(self.style_meta), feats):
            self.text_proto[sid] = vec

    @torch.inference_mode()
    def _clip_images(self, pixels01: torch.Tensor) -> torch.Tensor:
        pix = _prepare_clip_pixels(pixels01, image_size=self.clip_size, mean=self.clip_mean, std=self.clip_std)
        weight_dtype = next(self.clip_model.parameters()).dtype
        if pix.dtype != weight_dtype:
            pix = pix.to(weight_dtype)
        feats = _extract_clip_embeddings(self.clip_model.get_image_features(pixel_values=pix))
        return F.normalize(feats.float(), dim=-1)

    @torch.inference_mode()
    def _dino_forward(self, pil_images: list[Image.Image]) -> tuple[torch.Tensor, torch.Tensor]:
        pixels = torch.stack([DINO_TRANSFORM(im) for im in pil_images]).to(self.device)
        weight_dtype = next(self.dino.parameters()).dtype
        if pixels.dtype != weight_dtype:
            pixels = pixels.to(weight_dtype)
        out = self.dino(pixels, output_hidden_states=True)
        cls = F.normalize(out.last_hidden_state[:, 0, :].float(), dim=-1)
        patches = F.normalize(out.hidden_states[-2][:, 1:, :].float(), dim=-1)
        return cls, patches

    # -- precomputation -----------------------------------------------------
    def prepare(self, timer: Timer, source_paths: list[Path]) -> None:
        cache_file = None
        if self.args.ref_cache:
            cache_file = Path(self.args.ref_cache)
        sig = self._cache_signature(source_paths)
        if cache_file is not None and cache_file.exists():
            payload = torch.load(cache_file, map_location="cpu", weights_only=False)
            if payload.get("sig") == sig:
                self.ref_clip_proto = payload["ref_clip_proto"]
                self.ref_dino_cls = payload["ref_dino_cls"]
                self.src_clip = payload["src_clip"]
                self.src_dino_cls = payload["src_dino_cls"]
                self.src_dino_ssm = payload["src_dino_ssm"]
                self.cache_hits["clip"] = self.cache_hits["dino"] = self.cache_hits["src"] = True
                print(f"[eval_fast] reference/source feature cache hit: {cache_file}")
                timer.add("feature_cache_load", 0.0)

        if not self.args.disable_lpips or (not self.cache_hits["clip"] and not self.args.disable_clip):
            timer.start("src_pixels")
            with torch.inference_mode():
                for s in range(0, len(source_paths), self.args.ref_batch):
                    chunk = source_paths[s:s + self.args.ref_batch]
                    batch = torch.stack([_load_eval_image_tensor(p, size=self.args.image_size) for p in chunk]).to(self.device)
                    self.src_float.update({str(p): t for p, t in zip(chunk, batch)})
                    if not self.cache_hits["clip"] and not self.args.disable_clip:
                        feats = self._clip_images(batch).cpu()
                        for p, f in zip(chunk, feats):
                            self.src_clip[str(p)] = f
            timer.stop("src_pixels")

        if not self.cache_hits["clip"] and not self.args.disable_clip:
            timer.start("ref_clip")
            with torch.inference_mode():
                for sid in sorted(self.style_meta):
                    refs = self.style_meta[sid]["all"][: self.args.max_ref_cache]
                    feats = []
                    for s in range(0, len(refs), self.args.ref_batch):
                        batch = torch.stack([_load_eval_image_tensor(p, size=self.args.image_size) for p in refs[s:s + self.args.ref_batch]]).to(self.device)
                        feats.append(self._clip_images(batch).cpu())
                    proto = F.normalize(torch.cat(feats, 0).mean(0, keepdim=True), dim=-1)[0]
                    self.ref_clip_proto[sid] = proto
            timer.stop("ref_clip")

        if not self.cache_hits["dino"] and not self.args.disable_dino:
            timer.start("ref_dino")
            with torch.inference_mode():
                for sid in sorted(self.style_meta):
                    refs = self.style_meta[sid]["all"][: self.args.dino_max_refs]
                    cls_list = []
                    for s in range(0, len(refs), self.args.metric_batch):
                        ims = [Image.open(p).convert("RGB") for p in refs[s:s + self.args.metric_batch]]
                        cls, _ = self._dino_forward(ims)
                        cls_list.append(cls.cpu())
                    self.ref_dino_cls[sid] = torch.cat(cls_list, 0)
            timer.stop("ref_dino")

            timer.start("src_dino")
            with torch.inference_mode():
                for s in range(0, len(source_paths), self.args.metric_batch):
                    chunk = source_paths[s:s + self.args.metric_batch]
                    ims = [Image.open(p).convert("RGB") for p in chunk]
                    cls, patches = self._dino_forward(ims)
                    for p, c, pt in zip(chunk, cls.cpu(), patches.cpu()):
                        self.src_dino_cls[str(p)] = c
                        self.src_dino_ssm[str(p)] = (pt @ pt.T).cpu()
            timer.stop("src_dino")

        if cache_file is not None and not all(self.cache_hits.values()):
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "sig": sig,
                "ref_clip_proto": self.ref_clip_proto,
                "ref_dino_cls": self.ref_dino_cls,
                "src_clip": self.src_clip,
                "src_dino_cls": self.src_dino_cls,
                "src_dino_ssm": self.src_dino_ssm,
            }, cache_file)

    def _cache_signature(self, source_paths: list[Path]) -> str:
        h = hashlib.md5()
        h.update(str(self.args.test_dir).encode())
        h.update(f"|mref={self.args.max_ref_cache}|dref={self.args.dino_max_refs}|".encode())
        h.update(f"|clip={self.args.clip_model}|dino={self.args.dino_model}|px={self.args.image_size}|".encode())
        for p in sorted(source_paths):
            h.update(str(p).encode())
        for sid in sorted(self.style_meta):
            h.update(str(self.style_meta[sid]["ref"]).encode())
        return h.hexdigest()

    # -- per-batch metrics --------------------------------------------------
    def batch_metrics(self, gen01: torch.Tensor, rows: list[dict], timer: Timer) -> tuple[list[dict], list[dict]]:
        """gen01: float [N,3,H,W] on device, already quantised/dequantised like PNG pixels."""
        n = gen01.shape[0]
        clip_out: list[dict] = []
        dino_out: list[dict] = []
        full_clip = not (self.args.only_clip_style or self.args.disable_clip)
        if not self.args.disable_lpips:
            timer.start("metric_lpips")
            src = torch.stack([self.src_float[str(r["src_path"])] for r in rows])
            lp = _lpips_forward_safe(
                self.lpips_fn, gen01, src, device=self.device,
                chunk_size=self.args.lpips_chunk, cpu_fallback=False, tag="eval_fast_lpips",
            )
            timer.stop("metric_lpips")
        else:
            lp = torch.zeros(n)

        if not self.args.disable_clip:
            timer.start("metric_clip")
            feats = self._clip_images(gen01).cpu()
            timer.stop("metric_clip")
            for i, r in enumerate(rows):
                f = feats[i]
                src = self.src_clip[str(r["src_path"])]
                proto = self.ref_clip_proto[int(r["tgt_style_id"])]
                clip_s = float(torch.dot(f, proto))
                clip_c = 0.0
                clip_dir = 0.0
                clip_t = 0.0
                if full_clip:
                    clip_c = float(torch.dot(f, src))
                    dir_g = F.normalize(f - src, dim=-1, eps=1e-8)
                    dir_t = F.normalize(proto - src, dim=-1, eps=1e-8)
                    clip_dir = float(torch.dot(dir_g, dir_t))
                    if self.text_proto:
                        clip_t = float(torch.dot(f, self.text_proto[int(r["tgt_style_id"])]))
                vec = ",".join(f"{x:.7g}" for x in f.tolist())
                clip_out.append({
                    "clip_style": clip_s,
                    "clip_s_delta_idt": clip_s - self.args.clip_style_idt_baseline,
                    "clip_t": clip_t,
                    "clip_content": clip_c,
                    "clip_dir": clip_dir,
                    "clip_image_vector": vec if self.args.clip_vectors else "",
                })
        else:
            clip_out = [{"clip_style": 0.0, "clip_s_delta_idt": 0.0, "clip_t": 0.0,
                         "clip_content": 0.0, "clip_dir": 0.0, "clip_image_vector": ""} for _ in rows]

        if not self.args.disable_dino:
            timer.start("metric_dino")
            ims = [
                Image.fromarray(
                    (gen01[i].permute(1, 2, 0).float() * 255.0).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
                )
                for i in range(n)
            ]
            with torch.inference_mode():
                cls, patches = self._dino_forward(ims)
            cls = cls.cpu()
            patches = patches.cpu()
            timer.stop("metric_dino")
            for i, r in enumerate(rows):
                src_cls = self.src_dino_cls[str(r["src_path"])]
                src_ssm = self.src_dino_ssm[str(r["src_path"])]
                ref_cls = self.ref_dino_cls[int(r["tgt_style_id"])]
                dino_s = float((ref_cls @ cls[i]).max())
                dino_c = float(torch.dot(cls[i], src_cls))
                ss = patches[i] @ patches[i].T
                dino_struct = float((ss - src_ssm).pow(2).mean())
                dino_out.append({
                    "dino_style_similarity": dino_s,
                    "dino_content_similarity": dino_c,
                    "dino_structure_mse": dino_struct,
                })
        else:
            dino_out = [{"dino_style_similarity": 0.0, "dino_content_similarity": 0.0,
                         "dino_structure_mse": 0.0} for _ in rows]

        out = []
        for i, r in enumerate(rows):
            out.append({
                "src_style": r["src_style"], "tgt_style": r["tgt_style"],
                "src_image": r["src_name"], "gen_image": f"images/{r['gen_name']}",
                "_src_style_id": int(r["src_style_id"]),
                "_tgt_style_id": int(r["tgt_style_id"]),
                "_same_style": bool(r.get("same_style", False)),
                **clip_out[i], "lpips": float(lp[i]),
            })
        for row, d in zip(out, dino_out):
            row.update(d)
        return out, out


# ---------------------------------------------------------------------------
# PNG saving
# ---------------------------------------------------------------------------
def _encode_png_bytes(arr: np.ndarray) -> tuple[bytes, float]:
    t0 = time.perf_counter()
    buf = BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    data = buf.getvalue()
    return data, time.perf_counter() - t0


def save_png_async(arr: np.ndarray, path: Path, stats: SaveStats, backend: str) -> None:
    t0 = time.perf_counter()
    if backend == "pil_direct":
        Image.fromarray(arr).save(path, format="PNG")
        nbytes = path.stat().st_size
        stats.add(0.0, time.perf_counter() - t0, nbytes)
        return
    data, enc = _encode_png_bytes(arr)
    path.parent.mkdir(parents=True, exist_ok=True)
    t1 = time.perf_counter()
    with open(path, "wb") as f:
        f.write(data)
    stats.add(enc, time.perf_counter() - t1, len(data))


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------
def build_source_latents(
    lgt, vae, source_paths: list[Path], device: str, args, timer: Timer,
) -> torch.Tensor:
    model_scale = float(getattr(lgt.model, "latent_scale_factor", 0.18215))
    vae_scale = float(getattr(getattr(vae, "config", None), "scaling_factor", model_scale))
    scale_in = model_scale / max(vae_scale, 1e-8)
    # The legacy evaluator persists fp16 source latents and feeds them to the
    # model as fp16 on CUDA; CPU runs fp32.  Matching this dtype keeps parity.
    lat_dtype = torch.float16 if device == "cuda" else torch.float32

    timer.start("source_latents")
    latents = None
    if args.source_latent_cache:
        cache_dir = Path(args.cache_dir)
        cache_hash = _source_latent_cache_hash(
            source_paths, vae_model=args.vae_model, scale_in=scale_in, image_size=args.image_size,
        )
        cache_file = cache_dir / f"source_latents_{cache_hash}.pt"
        if cache_file.exists():
            payload = torch.load(cache_file, map_location="cpu", weights_only=False)
            items = payload.get("items") if isinstance(payload, dict) else None
            keys = [_source_path_key(p) for p in source_paths]
            if isinstance(items, dict) and all(k in items for k in keys):
                latents = torch.stack([items[k].to(lat_dtype) for k in keys], 0)
                print(f"[eval_fast] source latent cache hit: {cache_file}")
    if latents is None:
        chunks = []
        enc_bs = max(1, int(args.encode_batch))
        with torch.inference_mode():
            for s in range(0, len(source_paths), enc_bs):
                chunk = source_paths[s:s + enc_bs]
                imgs = torch.stack([_load_eval_image_tensor(p, size=args.image_size) for p in chunk]).to(device)
                lat = encode_image(vae, imgs, device)
                if abs(scale_in - 1.0) > 1e-4:
                    lat = lat * scale_in
                chunks.append(lat.detach().to("cpu", dtype=lat_dtype))
        latents = torch.cat(chunks, 0)
    timer.stop("source_latents")
    return latents


def build_target_latents(lgt, vae, style_meta, device: str, args, timer: Timer) -> dict[int, torch.Tensor]:
    model_scale = float(getattr(lgt.model, "latent_scale_factor", 0.18215))
    vae_scale = float(getattr(getattr(vae, "config", None), "scaling_factor", model_scale))
    scale_in = model_scale / max(vae_scale, 1e-8)
    lat_dtype = torch.float16 if device == "cuda" else torch.float32
    out = {}
    timer.start("target_latents")
    with torch.inference_mode():
        for sid in sorted(style_meta):
            img = _load_eval_image_tensor(style_meta[sid]["ref"], size=args.image_size).unsqueeze(0).to(device)
            lat = encode_image(vae, img, device)
            if abs(scale_in - 1.0) > 1e-4:
                lat = lat * scale_in
            out[sid] = lat.detach().to(lat_dtype).cpu()
    timer.stop("target_latents")
    return out


# ---------------------------------------------------------------------------
# reports
# ---------------------------------------------------------------------------
def write_csv(path: Path, rows: list[dict], header: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in header})


def build_results(metric_rows: list[dict]) -> list[dict]:
    """Match the legacy ``merge_eval_metrics.py`` record schema exactly."""
    out = []
    for r in metric_rows:
        out.append({
            "content_style_id": r["src_style"],
            "target_style_id": r["tgt_style"],
            "same_style": bool(r["_same_style"]),
            "src_image": r["src_image"],
            "gen_image": r["gen_image"],
            "dino_s": float(r["dino_style_similarity"]),
            "dino_c": float(r["dino_content_similarity"]),
            "dino_structure": float(r["dino_structure_mse"]),
            "clip_s": float(r["clip_style"]),
            "clip_dir": float(r["clip_dir"]),
            "lpips": float(r["lpips"]),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Fast protocol-equivalent full evaluation")
    ap.add_argument("--checkpoint", default="", help="checkpoint .pt; not needed with --reuse-images")
    ap.add_argument("--output", required=True)
    ap.add_argument("--protocol", choices=["paper"], default="paper",
                    help="paper: CLIP ViT-B/32 + AdaIN 2.0 + 8 steps + 750 all-pairs")
    ap.add_argument("--config", default=str(_ROOT / "config.json"))
    ap.add_argument("--config-override", default="", help="defaults to inference.json for --protocol paper")
    ap.add_argument("--test-dir", default="")
    ap.add_argument("--device", default=_default_device())
    ap.add_argument("--batch", type=int, default=32, help="ODE generation batch (sources x targets)")
    ap.add_argument("--metric-batch", type=int, default=32)
    ap.add_argument("--decode-batch", type=int, default=32)
    ap.add_argument("--image-size", type=int, default=512,
                    help="evaluation resolution; source/target images are resized to this "
                         "before VAE encode, and latents decode to this. Default 512 matches "
                         "the 512-native training latents (4x64x64 -> 512px).")
    ap.add_argument("--ref-batch", type=int, default=32)
    ap.add_argument("--latent-batch", type=int, default=8)
    ap.add_argument("--lpips-chunk", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4, help="async PNG encode/write threads; 0 disables")
    ap.add_argument("--sync-save", action="store_true", help="write PNGs inline instead of async")
    ap.add_argument("--no-save-images", action="store_true")
    ap.add_argument("--summary-grid", action="store_true", help="legacy grid image (off by default)")
    ap.add_argument("--save-backend", choices=["pil_png", "pil_direct"], default="pil_png")
    ap.add_argument("--clip-vectors", action="store_true", help="include clip_image_vector in metrics.csv")
    ap.add_argument("--no-clip-vectors", dest="clip_vectors", action="store_false")
    ap.set_defaults(clip_vectors=True)
    ap.add_argument("--reuse-images", default="", help="legacy eval dir; recompute metrics only")
    ap.add_argument("--skip-metrics", action="store_true")
    ap.add_argument("--disable-clip", action="store_true")
    ap.add_argument("--disable-lpips", action="store_true")
    ap.add_argument("--disable-dino", action="store_true")
    ap.add_argument("--only-clip-style", dest="only_clip_style", action="store_true", default=None)
    ap.add_argument("--full-clip", dest="only_clip_style", action="store_false")
    ap.add_argument("--max-src-samples", type=int, default=-1)
    ap.add_argument("--ref-pool", choices=["first", "strict"], default="first")
    ap.add_argument("--max-ref-cache", type=int, default=-1)
    ap.add_argument("--dino-max-refs", type=int, default=30)
    ap.add_argument("--num-steps", type=int, default=-1)
    ap.add_argument("--step-size", type=float, default=-1.0)
    ap.add_argument("--style-strength", type=float, default=-1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vae-model", default="")
    ap.add_argument("--cache-dir", default="")
    ap.add_argument("--hf-cache-dir", default="")
    ap.add_argument("--source-latent-cache", dest="source_latent_cache", action="store_true", default=True)
    ap.add_argument("--no-source-latent-cache", dest="source_latent_cache", action="store_false")
    ap.add_argument("--clip-model", default="")
    ap.add_argument("--clip-cache-dir", default="")
    ap.add_argument("--clip-local-only", action="store_true", default=True)
    ap.add_argument("--clip-skip-processor", action="store_true")
    ap.add_argument("--clip-style-idt-baseline", type=float, default=-1.0)
    ap.add_argument("--dino-model", default="facebook/dinov2-small")
    ap.add_argument("--dino-cache-dir", default="")
    ap.add_argument("--lpips-net", default="alex")
    ap.add_argument("--vae-dtype", choices=["fp32", "fp16"], default="fp32",
                    help="VAE compute dtype; fp32 is batch-invariant (default)")
    ap.add_argument("--gen-dtype", choices=["fp32", "bf16"], default="fp32",
                    help="generation compute dtype; fp32 is batch-invariant (default)")
    ap.add_argument("--encode-batch", type=int, default=1,
                    help="VAE encode batch; 1 keeps latents independent of --batch")
    ap.add_argument("--backbone", default="", help="label written to protocol metadata")
    ap.add_argument("--decoder", choices=["torch", "trt"], default="torch",
                    help="optional TensorRT VAE decoder; engines are fixed (size, batch, "
                         "precision), so a missing engine raises instead of falling back")
    ap.add_argument("--engine-dir", default=str(_ROOT / "runs" / "trt" / "engines"),
                    help="directory holding vae_decoder_{size}_b{batch}_{precision}.engine")
    ap.add_argument("--amp", action="store_true", help="half precision CLIP/DINO (not parity-safe)")
    ap.add_argument("--reference-policy", default=REFERENCE_POLICY)
    ap.add_argument("--ref-cache", default="")
    ap.add_argument("--allow-network", action="store_true")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--sync-timing", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    full = resolve_full_eval_section(cfg)

    # Frozen paper protocol (see docs/reproduction/hf_oriented_nohh_result.md):
    # oriented-HF checkpoint, CLIP ViT-B/32, DINOv2-small, 750 all-pairs,
    # first-image reference, 8 ODE steps, AdaIN 2.0 via inference.json,
    # 512 px native (latent 4x64x64), no postprocess.
    if not args.clip_model:
        args.clip_model = ALLOWED_CLIP_REPO
    if not args.config_override:
        args.config_override = str(_ROOT / "inference.json")
    args.adain_scale = 2.0
    args.split = "all-pairs"
    if not args.backbone:
        args.backbone = "hf_oriented_nohh"
    require_b32_clip(args.clip_model, "--clip-model")

    if args.only_clip_style is None:
        args.only_clip_style = bool(full.get("only_lpips_clip_style", True))
    if args.max_src_samples < 0:
        args.max_src_samples = int(full.get("max_src_samples", 30) or 30)
    if args.max_ref_cache < 0:
        args.max_ref_cache = int(full.get("max_ref_cache", 16) or 16)
    if args.num_steps < 0:
        args.num_steps = int(full.get("num_steps", 8) or 8)
    if args.step_size < 0:
        args.step_size = float(full.get("step_size", 1.0))
    if args.style_strength < 0:
        args.style_strength = float(full.get("style_strength", 1.0))
    if not args.vae_model:
        args.vae_model = str(full.get("vae_model", "ema"))
    cache_explicit = bool(args.cache_dir)
    if not args.cache_dir:
        args.cache_dir = str(_resolve_path(str(full.get("cache_dir", "runs/cache")), _ROOT))
    # The canonical artifacts live in <workspace>/eval_cache (source latents +
    # ref features + HF hub snapshots).  Prefer it when the configured cache
    # directory has no HF hub layout (offline machines).
    if not cache_explicit and not (Path(args.cache_dir) / "hf" / "hub").exists():
        alt = _ROOT.parent / "eval_cache"
        if (alt / "hf" / "hub").exists():
            args.cache_dir = str(alt)
            print(f"[eval_fast] cache dir -> {args.cache_dir} (workspace eval_cache)")
    if not args.hf_cache_dir:
        args.hf_cache_dir = str(Path(args.cache_dir) / "hf")
    if not args.test_dir:
        args.test_dir = str(_resolve_path("data/test", _ROOT))
    if args.clip_style_idt_baseline < 0:
        args.clip_style_idt_baseline = float(full.get("clip_style_idt_baseline", 0.0) or 0.0)
    if not args.clip_cache_dir:
        # transformers expects <cache_dir>/models--org--name; the workspace
        # keeps snapshots under <hf_cache>/hub/models--org--name.
        hub = Path(args.hf_cache_dir) / "hub"
        args.clip_cache_dir = str(hub) if hub.exists() else args.hf_cache_dir
    if not args.dino_cache_dir:
        args.dino_cache_dir = args.hf_cache_dir
    args.test_dir = str(Path(args.test_dir))

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    if args.device == "cpu":
        args.amp = False

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    images_dir = output / "images"

    timer = Timer(sync_cuda=(args.sync_timing or device == "cpu"))
    wall0 = time.perf_counter()
    save_stats = SaveStats()
    pool = None
    future_lock = threading.Lock()

    if args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(args.seed)

    reuse = bool(args.reuse_images)
    if reuse:
        reuse_dir = Path(args.reuse_images)
        metric_rows_raw = list(csv.DictReader(open(reuse_dir / "metrics.csv", encoding="utf-8")))
        requests = [{
            "src_path": (Path(args.test_dir) / r["src_style"] / r["src_image"]),
            "src_style_id": int(r["src_style"]) if str(r["src_style"]).isdigit() else -1,
            "src_style": r["src_style"],
            "src_name": r["src_image"],
            "gen_name": Path(r["gen_image"]).name,
            "gen_path": reuse_dir / r["gen_image"],
            "tgt_style_id": int(r["tgt_style"]) if str(r["tgt_style"]).isdigit() else -1,
            "tgt_style": r["tgt_style"],
            "same_style": False,
        } for r in metric_rows_raw]
        # resolve ids from style names when the legacy csv stores names
        style_names = sorted({p.name for p in Path(args.test_dir).iterdir() if p.is_dir()})
        sid_map = {n: i for i, n in enumerate(style_names)}
        for r in requests:
            r["src_style_id"] = sid_map.get(r["src_style"], r["src_style_id"])
            r["tgt_style_id"] = sid_map.get(r["tgt_style"], r["tgt_style_id"])
            r["same_style"] = r["src_style_id"] == r["tgt_style_id"]
        source_paths = sorted({r["src_path"] for r in requests})
        style_meta = {}
        test_images = {}
        for sid, sname in enumerate(style_names):
            sdir = Path(args.test_dir) / sname
            all_imgs = list_style_images(sdir)
            style_meta[sid] = {"name": sname, "dir": sdir, "ref": all_imgs[0], "all": all_imgs}
            test_images[sid] = (sname, all_imgs)
    else:
        test_images, requests, style_meta = build_protocol(
            Path(args.test_dir), max_src_samples=args.max_src_samples, ref_pool=args.ref_pool,
        )
        source_paths = sorted({r["src_path"] for r in requests})

    # ---- models -----------------------------------------------------------
    lgt = vae = None
    if not reuse:
        _ckpt = None
        if args.checkpoint:
            try:
                _ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            except Exception:
                _ckpt = None
            if isinstance(_ckpt, dict) and (
                "v2_config" in _ckpt or str(_ckpt.get("variant", "")) in {"V1", "V2", "V3", "V4", "V5", "V3S"}
            ):
                os.environ["SB_STRICT_CKPT"] = "1"
        timer.start("load_lgt")
        variant_tag = str(_ckpt.get("variant", "")) if isinstance(_ckpt, dict) else ""
        if isinstance(_ckpt, dict) and variant_tag in {"V2", "V3", "V4", "V3S"}:
            # 新主干（V2 velocity / V3-V4 generator）用 v2 适配器，strict 加载在内部硬校验。
            from v2.inference import build_v2_inference
            lgt = build_v2_inference(_ckpt, cfg, device=device, num_steps=args.num_steps)
        else:
            lgt = LGTInference(
                model_path=args.checkpoint,
                device=device,
                num_steps=args.num_steps,
                step_size=args.step_size,
                style_strength=args.style_strength,
                residual_scale=1.0,
                config_override_path=(args.config_override or None),
            )
        timer.stop("load_lgt")
        timer.start("load_vae")
        vae = load_vae(device, model_id=args.vae_model, cache_dir=args.hf_cache_dir, compile_decoder=False)
        vae = vae.to(dtype=torch.float32 if args.vae_dtype == "fp32" else torch.float16)
        timer.stop("load_vae")

        trt_decoder = None
        if args.decoder == "trt" and device == "cuda":
            from trt_vae_decoder import TrtVaeDecoder
            trt_decoder = TrtVaeDecoder(args.engine_dir, vae,
                                        precision="fp16" if args.vae_dtype == "fp16" else "fp32",
                                        device=device)

    engine = None
    if not args.skip_metrics:
        timer.start("init_metric_engine")
        engine = MetricEngine(args, test_images, style_meta, device)
        engine.load(timer)
        engine.prepare(timer, source_paths)
        timer.stop("init_metric_engine")

    # ---- generation -------------------------------------------------------
    metric_rows: list[dict] = []
    if not reuse:
        source_latents = build_source_latents(lgt, vae, source_paths, device, args, timer)
        target_latents = build_target_latents(lgt, vae, style_meta, device, args, timer)
        req_map = {(r["src_path"], r["tgt_style_id"]): r for r in requests}

        if not args.no_save_images:
            if args.sync_save:
                pool = None
            elif args.workers > 0:
                pool = ThreadPoolExecutor(max_workers=args.workers)

        num_styles = len(style_meta)
        gen_dtype = torch.float16 if (args.gen_dtype == "bf16" and device == "cuda") else torch.float32
        use_bf16 = args.gen_dtype == "bf16" and device == "cuda"
        for b_start in range(0, len(source_paths), args.batch):
            b_end = min(b_start + args.batch, len(source_paths))
            chunk_paths = source_paths[b_start:b_end]
            x0 = source_latents[b_start:b_end].to(device=device, dtype=gen_dtype, non_blocking=True)
            with torch.inference_mode():
                x0 = lgt.inversion(x0)
            n_src = x0.shape[0]
            # one target at a time keeps the legacy target_chunk_size=1 composition
            for tgt in range(num_styles):
                rows = [req_map[(p, tgt)] for p in chunk_paths]

                timer.start("generation")
                with torch.inference_mode():
                    style_lat = target_latents[tgt].to(device=device, dtype=gen_dtype).expand(n_src, -1, -1, -1).contiguous()
                    tgt_ids = torch.full((n_src,), tgt, dtype=torch.long, device=device)
                    if use_bf16:
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            latents_gen = lgt.generation_with_target_latent(
                                x0, tgt_ids, target_style_latent={"style_latent_tensor": style_lat},
                            )
                    else:
                        latents_gen = lgt.generation_with_target_latent(
                            x0, tgt_ids, target_style_latent={"style_latent_tensor": style_lat},
                        )
                timer.stop("generation")

                model_scale = float(getattr(lgt.model, "latent_scale_factor", 0.18215))
                vae_scale = float(getattr(getattr(vae, "config", None), "scaling_factor", model_scale))
                scale_out = vae_scale / max(model_scale, 1e-8)
                if abs(scale_out - 1.0) > 1e-4:
                    latents_gen = latents_gen * scale_out

                decoded = []
                timer.start("vae_decode")
                with torch.inference_mode():
                    for s in range(0, n_src, args.decode_batch):
                        chunk = latents_gen[s:s + args.decode_batch]
                        if trt_decoder is not None:
                            # TRT engines are fixed-shape: pad the tail chunk, then slice.
                            n = chunk.shape[0]
                            if n < args.decode_batch:
                                chunk = torch.cat([chunk, chunk[-1:].expand(
                                    args.decode_batch - n, -1, -1, -1)], 0)
                            decoded.append(trt_decoder.decode(chunk)[:n])
                        else:
                            decoded.append(decode_latent(vae, chunk, device))
                timer.stop("vae_decode")
                gen05 = torch.cat(decoded, 0)

                timer.start("uint8_copy")
                timer.start("u8_quant")
                u8_gpu = gen05.detach().clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8)
                timer.stop("u8_quant")
                timer.start("u8_dequant")
                gen01 = u8_gpu.float().div_(255.0)
                timer.stop("u8_dequant")
                timer.start("u8_d2h")
                u8_cpu = u8_gpu.permute(0, 2, 3, 1).contiguous().cpu()
                timer.stop("u8_d2h")
                timer.stop("uint8_copy")

                if not args.no_save_images:
                    timer.start("save_submit")
                    if pool is None:
                        for i, r in enumerate(rows):
                            save_png_async(u8_cpu[i].numpy(), images_dir / r["gen_name"], save_stats, args.save_backend)
                    else:
                        for i, r in enumerate(rows):
                            pool.submit(save_png_async, u8_cpu[i].numpy(), images_dir / r["gen_name"], save_stats, args.save_backend)
                    timer.stop("save_submit")

                if engine is not None:
                    rows_out, _ = engine.batch_metrics(gen01, rows, timer)
                    metric_rows.extend(rows_out)
                else:
                    for r in rows:
                        metric_rows.append({
                            "src_style": r["src_style"], "tgt_style": r["tgt_style"],
                            "src_image": r["src_name"], "gen_image": f"images/{r['gen_name']}",
                            "lpips": 0.0, "clip_style": 0.0, "clip_dir": 0.0,
                            "clip_s_delta_idt": 0.0, "clip_t": 0.0, "clip_content": 0.0,
                            "clip_image_vector": "",
                            "dino_style_similarity": 0.0, "dino_content_similarity": 0.0,
                            "dino_structure_mse": 0.0,
                            "_src_style_id": r["src_style_id"], "_tgt_style_id": r["tgt_style_id"],
                            "_same_style": r["same_style"],
                        })

    else:
        # metrics-only over legacy PNGs
        if engine is None:
            raise SystemExit("--reuse-images requires metrics")
        for s in range(0, len(requests), args.metric_batch):
            chunk = requests[s:s + args.metric_batch]
            timer.start("load_images")
            gen01 = torch.stack([
                T.ToTensor()(Image.open(r["gen_path"]).convert("RGB")) for r in chunk
            ]).to(device)
            timer.stop("load_images")
            rows_out, _ = engine.batch_metrics(gen01, chunk, timer)
            metric_rows.extend(rows_out)

    # ---- rank / dump ------------------------------------------------------
    req_map = {(r["src_path"], r["tgt_style_id"]): r for r in requests}
    for row in metric_rows:
        if row.get("_src_style_id", -1) == -1:
            key = (Path(args.test_dir) / row["src_style"] / row["src_image"], row.get("_tgt_style_id", -1))
            r = req_map.get(key)
            if r:
                row["_src_style_id"] = r["src_style_id"]
                row["_tgt_style_id"] = r["tgt_style_id"]
                row["_same_style"] = r["same_style"]

    metrics_ready = time.perf_counter() - wall0

    # release GPU before waiting for CPU PNG writes
    if pool is not None:
        torch.cuda.empty_cache()

    t_join0 = time.perf_counter()
    if pool is not None:
        pool.shutdown(wait=True)
    timer.values["save_join"] = time.perf_counter() - t_join0

    if not args.skip_metrics:
        timer.start("write_csv")
        rows_for_csv = []
        for r in metric_rows:
            rows_for_csv.append({k: r.get(k, "") for k in METRIC_HEADER})
        write_csv(output / "metrics.csv", rows_for_csv, METRIC_HEADER)
        dino_rows = [{
            "src_style": r["src_style"], "tgt_style": r["tgt_style"],
            "src_image": r["src_image"], "gen_image": r["gen_image"],
            "dino_style_similarity": r["dino_style_similarity"],
            "dino_content_similarity": r["dino_content_similarity"],
            "dino_structure_mse": r["dino_structure_mse"],
        } for r in metric_rows]
        write_csv(output / "dino_metrics.csv", dino_rows, DINO_HEADER)
        timer.stop("write_csv")

        def mean_of(key, subset=None):
            src = subset if subset is not None else metric_rows
            vals = [float(r[key]) for r in src]
            return float(np.mean(vals)) if vals else 0.0

        cross = [r for r in metric_rows if r.get("_src_style_id") != r.get("_tgt_style_id")]
        same = [r for r in metric_rows if r.get("_src_style_id") == r.get("_tgt_style_id")]
        dino_summary = {
            "n": len(metric_rows),
            "off_n": len(cross),
            "same_n": len(same),
            "all_dino_s": mean_of("dino_style_similarity"),
            "off_dino_s": mean_of("dino_style_similarity", cross),
            "same_dino_s": mean_of("dino_style_similarity", same),
            "all_dino_c": mean_of("dino_content_similarity"),
            "off_dino_c": mean_of("dino_content_similarity", cross),
            "same_dino_c": mean_of("dino_content_similarity", same),
            "all_dino_structure": mean_of("dino_structure_mse"),
            "off_dino_structure": mean_of("dino_structure_mse", cross),
            "same_dino_structure": mean_of("dino_structure_mse", same),
            "all_clip_s": mean_of("clip_style"),
            "all_lpips": mean_of("lpips"),
        }
        (output / "dino_summary.json").write_text(json.dumps(dino_summary, indent=1))

        results = build_results(metric_rows)
        protocol = {
            "protocol": args.protocol,
            "backbone": args.backbone,
            "clip_backbone": args.clip_model,
            "dino_backbone": args.dino_model,
            "split": args.split,
            "ref_policy": args.reference_policy,
            "steps": args.num_steps,
            "adain": float(args.adain_scale),
            "eval_batch": args.batch,
            "image_size": args.image_size,
            "subset": f"max_src_samples={args.max_src_samples}",
            "num_steps": args.num_steps,
            "dino_model": args.dino_model,
            "clip_model": args.clip_model,
            "lpips_net": args.lpips_net,
            "batch": args.batch,
            "device": device,
            "vae_dtype": args.vae_dtype,
            "gen_dtype": args.gen_dtype,
        }
        payload = {
            "results": results,
            "protocol": protocol,
            # top-level mirrors required by the paper-protocol contract
            "protocol_name": args.protocol,
            "backbone": args.backbone,
            "clip_backbone": args.clip_model,
            "dino_backbone": args.dino_model,
            "split": args.split,
            "ref_policy": args.reference_policy,
            "steps": args.num_steps,
            "adain": float(args.adain_scale),
            "eval_batch": args.batch,
            "image_size": args.image_size,
            "subset": protocol["subset"],
            "reference_selection": {"policy": REFERENCE_POLICY},
        }
        (output / "results.json").write_text(json.dumps(payload, indent=1))

        try:
            from run_eval_split import split_report
            split = split_report(results, REFERENCE_POLICY)
            split["source_file"] = str(output / "results.json")
            split["subset"] = protocol["subset"]
            split["ref_policy"] = protocol["ref_policy"]
            split["protocol"] = protocol
            (output / "results_split.json").write_text(json.dumps(split, indent=1))
        except Exception as exc:
            print(f"[eval_fast] WARNING: split report failed: {exc}")

    wall_total = time.perf_counter() - wall0
    timings = timer.as_dict()
    timings["metrics_ready"] = metrics_ready
    timings["wall_total"] = wall_total
    timings["png_encode_cpu_sum"] = save_stats.encode_sec
    timings["png_write_cpu_sum"] = save_stats.write_sec
    timings["png_bytes"] = float(save_stats.bytes)
    timings["png_files"] = float(save_stats.count)
    timings["png_write_mb_s"] = float(save_stats.bytes / 1e6 / max(save_stats.write_sec, 1e-9))
    timings["peak_rss_mb"] = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)
    meta = {
        "device": device,
        "protocol": args.protocol,
        "backbone": args.backbone,
        "clip_backbone": args.clip_model,
        "dino_backbone": args.dino_model,
        "split": args.split,
        "ref_policy": args.reference_policy,
        "steps": args.num_steps,
        "adain": float(args.adain_scale),
        "eval_batch": args.batch,
        "checkpoint": args.checkpoint,
        "reuse_images": args.reuse_images,
        "batch": args.batch,
        "metric_batch": args.metric_batch,
        "decode_batch": args.decode_batch,
        "workers": args.workers,
        "sync_save": bool(args.sync_save),
        "save_images": not args.no_save_images,
        "summary_grid": False,
        "amp": bool(args.amp),
        "decoder": args.decoder,
        "decoder_stats": (trt_decoder.summary() if trt_decoder is not None else {"decoder": "torch"}),
        "timings": timings,
        "settings": {
            "num_steps": args.num_steps,
            "step_size": args.step_size,
            "style_strength": args.style_strength,
            "max_src_samples": args.max_src_samples,
            "max_ref_cache": args.max_ref_cache,
            "dino_max_refs": args.dino_max_refs,
            "ref_pool": args.ref_pool,
            "seed": args.seed,
            "vae_model": args.vae_model,
            "vae_dtype": args.vae_dtype,
            "gen_dtype": args.gen_dtype,
            "encode_batch": args.encode_batch,
            "clip_model": args.clip_model,
            "dino_model": args.dino_model,
            "lpips_net": args.lpips_net,
        },
    }
    (output / "eval_fast_meta.json").write_text(json.dumps(meta, indent=1))

    print("\n[eval_fast] timings (s):")
    for k in sorted(timings, key=lambda x: -timings[x]):
        if timings[k] > 0.01:
            print(f"  {k:28s} {timings[k]:9.3f}")
    print(f"[eval_fast] metrics_ready={metrics_ready:.2f}s wall_total={wall_total:.2f}s "
          f"peak_rss={timings['peak_rss_mb']:.0f}MB")
    if save_stats.count:
        print(f"[eval_fast] png files={save_stats.count} bytes={save_stats.bytes/1e6:.1f}MB "
              f"encode_cpu={save_stats.encode_sec:.2f}s write_cpu={save_stats.write_sec:.2f}s")


if __name__ == "__main__":
    main()
