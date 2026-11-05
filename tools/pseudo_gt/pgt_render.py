"""Render a trained pseudo-GT arm on the frozen 750-pair protocol.

Protocol is identical to P0/P2 (30 test sources per style x 5 targets = 750 pairs,
8-step ODE, AdaIN off, r0 reference images reused from runs/p0/reference_selection.json).
Differences vs p2_render.py: no medoid search (reuses the archived r0 selection),
no PNG-vs-CSV consistency gate (the gate belongs to re-rendering P0 cells), and the
cell name is supplied by the caller.

Outputs (same layout as p2_render so tools/12h/p2_metrics.py can consume them):
  <render-root>/<cell>/images/<gen_name>.png
  <render-root>/<cell>/metrics.csv, dino_metrics.csv
  <render-root>/<cell>/render_manifest.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tools", ROOT / "tools" / "trt"):
    sys.path.insert(0, str(_p))

import eval_fast as ef  # noqa: E402
import paths  # noqa: E402
import trt_vae_decoder  # noqa: E402
from utils.inference import encode_image, load_vae, LGTInference  # noqa: E402


def _mean_latent(vae, image: torch.Tensor, device: str) -> torch.Tensor:
    """Posterior mean (deterministic) instead of latent_dist.sample().

    All three arms share the same target latents, so a re-encode cannot
    introduce a run-to-run difference into the arm comparison.
    """
    with torch.inference_mode():
        dtype = next(vae.parameters()).dtype
        dist = vae.encode(image.to(device, dtype=dtype)).latent_dist
        return (dist.mean * vae.config.scaling_factor).float()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--render-root", default=str(paths.asset_root() / "eval_assets" / "p2_render"))
    ap.add_argument("--reference-selection", default=str(ROOT / "runs" / "p0" / "reference_selection.json"))
    ap.add_argument("--test-dir", default=str(ROOT / "data" / "test"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--decode-batch", type=int, default=8)
    ap.add_argument("--metric-batch", type=int, default=32)
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--num-steps", type=int, default=8)
    ap.add_argument("--max-src-samples", type=int, default=30)
    ap.add_argument("--cache-dir", default=str(paths.eval_cache_root()))
    ap.add_argument("--engine-dir", default=str(ROOT / "runs" / "trt" / "engines"))
    ap.add_argument("--config", default=str(ROOT / "config.json"))
    ap.add_argument("--config-override", default=str(ROOT / "inference.json"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.source_latent_cache = True
    args.vae_model = "ema"
    args.encode_batch = 1

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = args.device
    if not torch.cuda.is_available() or device != "cuda":
        raise SystemExit("[pgt_r] CUDA required; CPU fallback is forbidden")
    render_root = Path(args.render_root)
    out_dir = render_root / args.cell
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    t_all = time.perf_counter()

    test_images, requests, style_meta = ef.build_protocol(
        Path(args.test_dir), max_src_samples=args.max_src_samples, ref_pool="first",
    )
    source_paths = sorted({r["src_path"] for r in requests})
    num_styles = len(style_meta)
    req_map = {(r["src_path"], r["tgt_style_id"]): r for r in requests}
    print(f"[pgt_r:{args.cell}] sources={len(source_paths)} pairs={len(requests)}", flush=True)

    timer = ef.Timer(sync_cuda=False)
    lgt = LGTInference(
        model_path=args.checkpoint, device=device, num_steps=args.num_steps,
        step_size=1.0, style_strength=1.0, residual_scale=1.0,
        config_override_path=args.config_override,
    )
    vae = load_vae(device, model_id="ema", cache_dir=str(Path(args.cache_dir) / "hf"), compile_decoder=False)
    trt = trt_vae_decoder.TrtVaeDecoder(args.engine_dir, vae, precision="fp16", device=device)

    ns = argparse.Namespace(
        disable_clip=True, disable_lpips=False, disable_dino=False,
        test_dir=str(args.test_dir),
        clip_model="openai/clip-vit-base-patch32", clip_local_only=True,
        clip_cache_dir=str(Path(args.cache_dir) / "hf" / "hub"), clip_skip_processor=False,
        amp=False, lpips_net="alex", lpips_chunk=32,
        dino_model="facebook/dinov2-small",
        dino_cache_dir=str(Path(args.cache_dir) / "hf"), allow_network=False,
        ref_cache="", max_ref_cache=16, dino_max_refs=30, image_size=args.image_size,
        metric_batch=args.metric_batch, ref_batch=32, only_clip_style=True,
        clip_style_idt_baseline=0.0, clip_vectors=False,
    )
    engine = ef.MetricEngine(ns, test_images, style_meta, device)
    engine.load(timer)
    engine.prepare(timer, source_paths)

    sel = json.loads(Path(args.reference_selection).read_text())["selected"]
    model_scale = float(getattr(lgt.model, "latent_scale_factor", 0.18215))
    vae_scale = float(getattr(getattr(vae, "config", None), "scaling_factor", model_scale))
    scale_in = model_scale / max(vae_scale, 1e-8)
    target_latents: dict[int, torch.Tensor] = {}
    for sid in sorted(style_meta):
        name = style_meta[sid]["name"]
        ref_path = Path(sel[name]["r0"][0])
        img = ef._load_eval_image_tensor(ref_path, size=args.image_size).unsqueeze(0).to(device)
        lat = _mean_latent(vae, img, device)
        if abs(scale_in - 1.0) > 1e-4:
            lat = lat * scale_in
        target_latents[sid] = lat.detach().to(torch.float16).cpu()
        print(f"[pgt_r] target latent {name}: {ref_path.name}", flush=True)

    source_latents = ef.build_source_latents(lgt, vae, source_paths, device, args, timer)

    def _adain_holder():
        for holder in ("model_cfg", "bridge_cfg"):
            cfg = getattr(lgt.model, holder, None)
            if cfg is not None and hasattr(cfg, "endpoint_adain_scale"):
                return cfg
        raise SystemExit("[pgt_r] endpoint_adain_scale not found")

    cfg = _adain_holder()
    cfg.endpoint_adain_scale = 0.0
    if abs(float(cfg.endpoint_adain_scale)) > 1e-9:
        raise SystemExit("[pgt_r] failed to disable AdaIN")

    rows_all: list[dict] = []
    sizes: list[int] = []
    t_render = time.perf_counter()
    for b_start in range(0, len(source_paths), args.batch):
        chunk_paths = source_paths[b_start:b_start + args.batch]
        x0 = source_latents[b_start:b_start + len(chunk_paths)].to(device=device, dtype=torch.float32)
        with torch.inference_mode():
            x0 = lgt.inversion(x0)
        n_src = x0.shape[0]
        for tgt in range(num_styles):
            rows = [req_map[(p, tgt)] for p in chunk_paths]
            with torch.inference_mode():
                style_lat = target_latents[tgt].to(device=device, dtype=torch.float32)
                style_lat = style_lat.expand(n_src, -1, -1, -1).contiguous()
                tgt_ids = torch.full((n_src,), tgt, dtype=torch.long, device=device)
                latents_gen = lgt.generation_with_target_latent(
                    x0, tgt_ids, num_steps=args.num_steps,
                    target_style_latent={"style_latent_tensor": style_lat},
                )
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
            arr = u8.cpu().numpy()
            for i, r in enumerate(rows):
                png_path = img_dir / r["gen_name"]
                Image.fromarray(arr[i].transpose(1, 2, 0)).save(png_path, format="PNG")
                sizes.append(png_path.stat().st_size)
            rows_out, _ = engine.batch_metrics(gen01, rows, timer)
            rows_all.extend(rows_out)
        print(f"[pgt_r:{args.cell}] {min(b_start + args.batch, len(source_paths))}/"
              f"{len(source_paths)} sources ({time.perf_counter() - t_render:.1f}s)", flush=True)

    ef.write_csv(out_dir / "metrics.csv", rows_all, ef.METRIC_HEADER)
    ef.write_csv(out_dir / "dino_metrics.csv", [{
        "src_style": r["src_style"], "tgt_style": r["tgt_style"],
        "src_image": r["src_image"], "gen_image": r["gen_image"],
        "dino_style_similarity": r["dino_style_similarity"],
        "dino_content_similarity": r["dino_content_similarity"],
        "dino_structure_mse": r["dino_structure_mse"],
    } for r in rows_all], ef.DINO_HEADER)

    total = int(sum(sizes))
    manifest = {
        "cell": args.cell, "checkpoint": args.checkpoint, "num_steps": args.num_steps,
        "adain_scale": 0.0, "reference_scheme": "r0 (reused from runs/p0/reference_selection.json)",
        "num_images": len(sizes), "bytes_total": total,
        "bytes_per_image_mean": float(np.mean(sizes)) if sizes else 0.0,
        "image_dir": str(img_dir),
        "wall_sec_render": round(time.perf_counter() - t_render, 2),
        "wall_sec_total": round(time.perf_counter() - t_all, 2),
        "device": device,
    }
    (out_dir / "render_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"[pgt_r] done {json.dumps({k: manifest[k] for k in ('num_images','wall_sec_render','wall_sec_total')})}",
          flush=True)


if __name__ == "__main__":
    main()
