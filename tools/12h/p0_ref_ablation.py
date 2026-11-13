"""P0: reference-image ablation with a FROZEN evaluation pool.

Hard rule: only the *generation-side* reference changes.  The DINO-S / CLIP-S
reference pool (``style_meta[sid]["all"]``, i.e. the 30 test images per style)
is never touched, so any score difference is attributable to the generator.

Cells (one process, models + features loaded once):
  A      skip ODE, adain 0.0      -> VAE round-trip (reference independent)
  B_rX   skip ODE, adain 2.0      -> pure reference-statistics transform
  C      8-step ODE, adain 0.0    -> network only (reference independent)
  D_rX   8-step ODE, adain 2.0    -> the real pipeline

Reference schemes (all chosen from the TRAINING pool, never from eval scores):
  r0  first image of the target-style test dir (current behaviour)
  r1  train-pool medoid (DINOv2-small CLS nearest to the class centroid)
  r2  statistics aggregate of the medoid + 3 nearest (4 refs)
  r3  statistics aggregate of the medoid + 7 nearest (8 refs)

r2/r3 aggregate ONLY the per-channel mean/std of the AdaIN HF fiber
(``z - haar_lowpass(z)``) -- that is literally the only style information the
``spatial_fiber`` AdaIN consumes.  No pixel / latent averaging.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tools", ROOT / "tools" / "trt"):
    sys.path.insert(0, str(_p))

import eval_fast as ef  # noqa: E402
import trt_vae_decoder  # noqa: E402
from tools.legacy_utils.inference import encode_image, load_vae, LGTInference  # noqa: E402
from tools.legacy_utils.compute_dino_metrics import DINO_TRANSFORM, list_style_images  # noqa: E402
from tools.legacy_root.wavelet import dwt2_lowpass  # noqa: E402

IMAGE_EXTS = ef.IMAGE_EXTS


def _lp_filter(z: torch.Tensor) -> torch.Tensor:
    return dwt2_lowpass(z, levels=1, basis="haar")


def _fiber(z: torch.Tensor) -> torch.Tensor:
    return z - _lp_filter(z)


def _fiber_stats(f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return f.mean(dim=[2, 3], keepdim=True), f.std(dim=[2, 3], keepdim=True)


def synth_from_stats(base_latent: torch.Tensor, mean_agg: torch.Tensor, std_agg: torch.Tensor,
                     iters: int = 12) -> torch.Tensor:
    """Latent whose HF-fiber per-channel mean/std equals the aggregate."""
    f0 = _fiber(base_latent)
    m0, s0 = _fiber_stats(f0)
    f_unit = (f0 - m0) / s0.clamp_min(1e-6)
    target = f_unit * std_agg + mean_agg
    z = base_latent.clone()
    for _ in range(iters):
        z = _lp_filter(z) + target
    return z


def build_ref_set(style_name: str, manifest: dict, image_root: Path) -> list[Path]:
    rows = [s for s in manifest["samples"] if s.get("style") == style_name and s.get("split") == "train"]
    rows.sort(key=lambda r: int(r.get("index", 0)))
    return [image_root / r["path"] for r in rows]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--test-dir", default=str(ROOT / "data" / "test"))
    ap.add_argument("--manifest", default=str(ROOT / "data" / "wikiart_new_packed" / "manifest.json"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--decode-batch", type=int, default=8)
    ap.add_argument("--metric-batch", type=int, default=32)
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--num-steps", type=int, default=8)
    ap.add_argument("--adain-scale", type=float, default=2.0)
    ap.add_argument("--max-src-samples", type=int, default=30)
    ap.add_argument("--cache-dir", default="/mnt/i/Latent_Style/eval_cache")
    ap.add_argument("--engine-dir", default=str(ROOT / "runs" / "trt" / "engines"))
    ap.add_argument("--config", default=str(ROOT / "config.json"))
    ap.add_argument("--config-override", default=str(ROOT / "inference.json"))
    ap.add_argument("--cells", default="A,B_r0,B_r1,B_r2,B_r3,C,D_r0,D_r1,D_r2,D_r3")
    args = ap.parse_args()
    # fields consumed by ef.build_source_latents
    args.source_latent_cache = True
    args.vae_model = "ema"
    args.encode_batch = 1

    device = args.device
    output_root = Path(args.out_root)
    output_root.mkdir(parents=True, exist_ok=True)
    wall0 = time.perf_counter()

    # ---- protocol (frozen) -------------------------------------------------
    test_images, requests, style_meta = ef.build_protocol(
        Path(args.test_dir), max_src_samples=args.max_src_samples, ref_pool="first",
    )
    source_paths = sorted({r["src_path"] for r in requests})
    num_styles = len(style_meta)
    print(f"[p0] styles={[style_meta[s]['name'] for s in sorted(style_meta)]} "
          f"sources={len(source_paths)} pairs={len(requests)}", flush=True)

    # ---- models -----------------------------------------------------------
    timer = ef.Timer(sync_cuda=False)
    print("[p0] loading checkpoints/models ...", flush=True)
    lgt = LGTInference(
        model_path=args.checkpoint, device=device, num_steps=args.num_steps,
        step_size=1.0, style_strength=1.0, residual_scale=1.0,
        config_override_path=args.config_override,
    )
    vae = load_vae(device, model_id="ema", cache_dir=str(Path(args.cache_dir) / "hf"), compile_decoder=False)
    vae = vae.to(dtype=torch.float32)
    trt = trt_vae_decoder.TrtVaeDecoder(args.engine_dir, vae, precision="fp16", device=device)

    ns = argparse.Namespace(
        disable_clip=False, disable_lpips=False, disable_dino=False,
        test_dir=str(args.test_dir),
        clip_model="openai/clip-vit-base-patch32",
        clip_local_only=True, clip_cache_dir=str(Path(args.cache_dir) / "hf" / "hub"),
        clip_skip_processor=False, amp=False, lpips_net="alex", lpips_chunk=32,
        dino_model="facebook/dinov2-small",
        dino_cache_dir=str(Path(args.cache_dir) / "hf"), allow_network=False,
        ref_cache="", max_ref_cache=16, dino_max_refs=30, image_size=args.image_size,
        metric_batch=args.metric_batch, ref_batch=32, only_clip_style=True,
        clip_style_idt_baseline=0.639920825263, clip_vectors=False,
    )
    engine = ef.MetricEngine(ns, test_images, style_meta, device)
    engine.load(timer)
    engine.prepare(timer, source_paths)

    # ---- reference selection (training pool only) -------------------------
    manifest = json.loads(Path(args.manifest).read_text())
    image_root = Path(manifest["image_root"])
    scheme_sel: dict[str, dict[str, list[Path]]] = {}
    for sid in sorted(style_meta):
        name = style_meta[sid]["name"]
        cands = build_ref_set(name, manifest, image_root)
        if not cands:
            raise SystemExit(f"[p0] no train candidates for style {name}")
        print(f"[p0] {name}: pool={len(cands)} medoid search ...", flush=True)
        feats = []
        with torch.inference_mode():
            for s in range(0, len(cands), 64):
                ims = [Image.open(p).convert("RGB") for p in cands[s:s + 64]]
                cls, _ = engine._dino_forward(ims)
                feats.append(cls.cpu())
        feats = torch.cat(feats, 0)
        feats = F.normalize(feats.float(), dim=-1)
        centroid = F.normalize(feats.mean(0, keepdim=True), dim=-1)
        sims = (feats @ centroid.T).squeeze(1)
        order = torch.argsort(sims, descending=True).tolist()
        medoid = cands[order[0]]
        near4 = [cands[i] for i in order[:4]]
        near8 = [cands[i] for i in order[:8]]
        scheme_sel[name] = {"r1": [medoid], "r2": near4, "r3": near8,
                            "r0": [style_meta[sid]["ref"]]}
        print(f"  medoid={medoid.name} (centroid cos={float(sims[order[0]]):.4f})", flush=True)

    # ---- target latents per scheme ---------------------------------------
    model_scale = float(getattr(lgt.model, "latent_scale_factor", 0.18215))
    vae_scale = float(getattr(getattr(vae, "config", None), "scaling_factor", model_scale))
    scale_in = model_scale / max(vae_scale, 1e-8)

    def encode_ref(path: Path) -> torch.Tensor:
        img = ef._load_eval_image_tensor(path, size=args.image_size).unsqueeze(0).to(device)
        with torch.inference_mode():
            lat = encode_image(vae, img, device)
        if abs(scale_in - 1.0) > 1e-4:
            lat = lat * scale_in
        return lat.detach().to(torch.float16).cpu()

    target_latents: dict[str, dict[int, torch.Tensor]] = {}
    agg_report = {}
    for scheme in ("r0", "r1", "r2", "r3"):
        target_latents[scheme] = {}
        for sid in sorted(style_meta):
            name = style_meta[sid]["name"]
            paths = scheme_sel[name][scheme]
            lat0 = encode_ref(paths[0])
            if scheme in ("r2", "r3"):
                stats = []
                for p in paths:
                    z = encode_ref(p).float()
                    f = _fiber(z)
                    m, s = _fiber_stats(f)
                    stats.append((m, s))
                mean_agg = torch.stack([m for m, _ in stats], 0).mean(0)
                std_agg = torch.stack([s for _, s in stats], 0).mean(0)
                z_syn = synth_from_stats(lat0.float(), mean_agg, std_agg)
                fm, fs = _fiber_stats(_fiber(z_syn))
                agg_report[f"{scheme}/{name}"] = {
                    "n_refs": len(paths),
                    "mean_target": mean_agg.flatten().tolist(),
                    "std_target": std_agg.flatten().tolist(),
                    "mean_residual": float((fm - mean_agg).abs().max()),
                    "std_residual": float((fs - std_agg).abs().max()),
                }
                target_latents[scheme][sid] = z_syn.to(torch.float16)
            else:
                target_latents[scheme][sid] = lat0
        print(f"[p0] target latents ready: {scheme}", flush=True)

    (output_root / "reference_selection.json").write_text(json.dumps({
        "medoid_note": "DINOv2-small CLS (L2-normalised) nearest to the train-pool class centroid",
        "aggregate_note": "per-channel mean/std of z - haar_lowpass(z), averaged over refs",
        "selected": {name: {k: [str(p) for p in v] for k, v in sel.items()} for name, sel in scheme_sel.items()},
        "agg_stats": agg_report,
    }, indent=1))

    # ---- source latents ---------------------------------------------------
    source_latents = ef.build_source_latents(lgt, vae, source_paths, device, args, timer)
    req_map = {(r["src_path"], r["tgt_style_id"]): r for r in requests}

    # ---- cell generation --------------------------------------------------
    def _current_adain() -> float:
        for holder in ("model_cfg", "bridge_cfg"):
            cfg = getattr(lgt.model, holder, None)
            if cfg is not None and hasattr(cfg, "endpoint_adain_scale"):
                return float(getattr(cfg, "endpoint_adain_scale"))
        raise SystemExit("[p0] endpoint_adain_scale not found on model config")

    def set_adain_scale(scale: float) -> None:
        for holder in ("model_cfg", "bridge_cfg"):
            cfg = getattr(lgt.model, holder, None)
            if cfg is not None and hasattr(cfg, "endpoint_adain_scale"):
                setattr(cfg, "endpoint_adain_scale", float(scale))
        got = _current_adain()
        if abs(got - float(scale)) > 1e-9:
            raise SystemExit(f"[p0] adain scale not applied: want {scale} got {got}")

    print(f"[p0] adain scale from config = {_current_adain()}", flush=True)

    orig_solver = lgt.model._solver_step

    def run_cell(cell: str, skip_ode: bool, adain: float, scheme: str) -> dict:
        t0 = time.perf_counter()
        set_adain_scale(adain)
        if skip_ode:
            lgt.model._solver_step = lambda h, **kw: h
        else:
            lgt.model._solver_step = orig_solver
        rows_all: list[dict] = []
        out_dir = output_root / cell
        out_dir.mkdir(parents=True, exist_ok=True)
        steps = 1 if skip_ode else args.num_steps
        for b_start in range(0, len(source_paths), args.batch):
            chunk_paths = source_paths[b_start:b_start + args.batch]
            x0 = source_latents[b_start:b_start + len(chunk_paths)].to(device=device, dtype=torch.float32)
            with torch.inference_mode():
                x0 = lgt.inversion(x0)
            n_src = x0.shape[0]
            for tgt in range(num_styles):
                rows = [req_map[(p, tgt)] for p in chunk_paths]
                with torch.inference_mode():
                    style_lat = target_latents[scheme][tgt].to(device=device, dtype=torch.float32)
                    style_lat = style_lat.expand(n_src, -1, -1, -1).contiguous()
                    tgt_ids = torch.full((n_src,), tgt, dtype=torch.long, device=device)
                    latents_gen = lgt.generation_with_target_latent(
                        x0, tgt_ids, num_steps=steps,
                        target_style_latent={"style_latent_tensor": style_lat},
                    )
                # eval_fast keeps the latent in the model scale; VAE wants vae scale
                scale_out = vae_scale / max(model_scale, 1e-8)
                if abs(scale_out - 1.0) > 1e-4:
                    latents_gen = latents_gen * scale_out
                decoded = []
                with torch.inference_mode():
                    for s in range(0, n_src, args.decode_batch):
                        chunk = latents_gen[s:s + args.decode_batch]
                        n = chunk.shape[0]
                        if n < args.decode_batch:
                            chunk = torch.cat([chunk, chunk[-1:].expand(args.decode_batch - n, -1, -1, -1)], 0)
                        decoded.append(trt.decode(chunk)[:n])
                gen05 = torch.cat(decoded, 0)
                u8 = gen05.clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8)
                gen01 = u8.float().div_(255.0)
                rows_out, _ = engine.batch_metrics(gen01, rows, timer)
                rows_all.extend(rows_out)
        lgt.model._solver_step = orig_solver

        ef.write_csv(out_dir / "metrics.csv", rows_all, ef.METRIC_HEADER)
        ef.write_csv(out_dir / "dino_metrics.csv", [{
            "src_style": r["src_style"], "tgt_style": r["tgt_style"],
            "src_image": r["src_image"], "gen_image": r["gen_image"],
            "dino_style_similarity": r["dino_style_similarity"],
            "dino_content_similarity": r["dino_content_similarity"],
            "dino_structure_mse": r["dino_structure_mse"],
        } for r in rows_all], ef.DINO_HEADER)

        cross = [r for r in rows_all if r["_src_style_id"] != r["_tgt_style_id"]]
        same = [r for r in rows_all if r["_src_style_id"] == r["_tgt_style_id"]]

        def mean(key, subset):
            return float(np.mean([float(r[key]) for r in subset])) if subset else 0.0

        per_tgt = {}
        for name in sorted({r["tgt_style"] for r in rows_all}):
            sub = [r for r in rows_all if r["tgt_style"] == name]
            per_tgt[name] = {k: mean(k, sub) for k in
                             ("dino_style_similarity", "dino_content_similarity", "clip_style", "lpips")}
        per_src = {}
        for name in sorted({r["src_style"] for r in rows_all}):
            sub = [r for r in rows_all if r["src_style"] == name]
            per_src[name] = {k: mean(k, sub) for k in
                             ("dino_style_similarity", "dino_content_similarity", "clip_style", "lpips")}

        summary = {
            "cell": cell, "skip_ode": skip_ode, "adain_scale": adain, "scheme": scheme,
            "n": len(rows_all), "off_n": len(cross), "same_n": len(same),
            "dino_s": mean("dino_style_similarity", rows_all),
            "off_dino_s": mean("dino_style_similarity", cross),
            "same_dino_s": mean("dino_style_similarity", same),
            "dino_c": mean("dino_content_similarity", rows_all),
            "clip_s": mean("clip_style", rows_all),
            "lpips": mean("lpips", rows_all),
            "macro_per_tgt": {k: float(np.mean([v[k] for v in per_tgt.values()])) for k in
                              ("dino_style_similarity", "dino_content_similarity", "clip_style", "lpips")},
            "per_tgt_style": per_tgt, "per_src_style": per_src,
            "wall_sec": time.perf_counter() - t0,
        }
        (out_dir / "cell_summary.json").write_text(json.dumps(summary, indent=1))
        print(f"[p0] {cell:6s} dino_s={summary['dino_s']:.5f} dino_c={summary['dino_c']:.5f} "
              f"clip_s={summary['clip_s']:.5f} lpips={summary['lpips']:.5f} "
              f"({summary['wall_sec']:.1f}s)", flush=True)
        return summary

    cell_specs = {
        "A": (True, 0.0, "r0"),
        "C": (False, 0.0, "r0"),
    }
    for r in ("r0", "r1", "r2", "r3"):
        cell_specs[f"B_{r}"] = (True, args.adain_scale, r)
        cell_specs[f"D_{r}"] = (False, args.adain_scale, r)

    summaries = []
    for cell in [c.strip() for c in args.cells.split(",") if c.strip()]:
        skip_ode, adain, scheme = cell_specs[cell]
        summaries.append(run_cell(cell, skip_ode, adain, scheme))

    (output_root / "p0_summary.json").write_text(json.dumps({
        "checkpoint": args.checkpoint, "wall_sec": time.perf_counter() - wall0,
        "cells": summaries,
    }, indent=1))
    print(f"[p0] done in {time.perf_counter() - wall0:.1f}s", flush=True)


if __name__ == "__main__":
    main()

