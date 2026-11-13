"""Full-pipeline inference latency for WEAVE (encode + ODE + decode).

Deploy-relevant path per content->style pair:
  content VAE encode + style-ref VAE encode + ODE(K) + VAE decode

- Does NOT use source-latent cache (content encode always on critical path).
- Style-ref latents are cached per target style (same as production eval).
- Uses paper protocol via --config_override inference.json (endpoint_adain_scale etc.).

Example (remote):
  python -u scripts/bench_inference_latency.py ^
    --checkpoint runs/submission/hf_oriented_internal_early_stop/epoch_0004.pt ^
    --config_override inference.json ^
    --test_dir data/test ^
    --num_steps 4 8 ^
    --batch_size 8 ^
    --seed 42 ^
    --output exp/inf_timing_full_encode
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.legacy_utils.inference import (  # noqa: E402
    LGTInference,
    decode_latent,
    encode_image,
    load_vae,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _list_styles(test_dir: Path) -> list[str]:
    styles = sorted(
        p.name for p in test_dir.iterdir() if p.is_dir() and not p.name.startswith(".")
    )
    if not styles:
        raise FileNotFoundError(f"No style subdirs under {test_dir}")
    return styles


def _list_images(style_dir: Path) -> list[Path]:
    return sorted(
        p for p in style_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def _build_pairs(
    test_dir: Path,
    styles: list[str],
    max_src_per_style: int,
    include_identity: bool,
) -> list[tuple[Path, int, int, str, str]]:
    style_images: dict[str, list[Path]] = {}
    for name in styles:
        imgs = _list_images(test_dir / name)
        if not imgs:
            raise FileNotFoundError(f"No images in {test_dir / name}")
        if max_src_per_style > 0:
            imgs = imgs[:max_src_per_style]
        style_images[name] = imgs

    pairs: list[tuple[Path, int, int, str, str]] = []
    for src_id, src_name in enumerate(styles):
        for src_path in style_images[src_name]:
            for tgt_id, tgt_name in enumerate(styles):
                if not include_identity and src_id == tgt_id:
                    continue
                pairs.append((src_path, src_id, tgt_id, src_name, tgt_name))
    return pairs


def _load_rgb_batch(paths: list[Path], size: int, device: torch.device) -> torch.Tensor:
    # Match utils/run_evaluation._load_eval_image_tensor: resize + ToTensor -> [0, 1].
    tensors = []
    for p in paths:
        with Image.open(p) as im:
            img = im.convert("RGB").resize((size, size))
            tensors.append(T.ToTensor()(img))
    return torch.stack(tensors, dim=0).to(device, non_blocking=True)


@torch.inference_mode()
def bench_one_k(
    *,
    lgt: LGTInference,
    vae,
    pairs: list[tuple[Path, int, int, str, str]],
    styles: list[str],
    test_dir: Path,
    device: torch.device,
    image_size: int,
    batch_size: int,
    num_steps: int,
    scale_in: float,
    scale_out: float,
    warmup_batches: int,
) -> dict:
    style_ref_path = {
        sid: _list_images(test_dir / name)[0] for sid, name in enumerate(styles)
    }

    t_content_encode = 0.0
    t_style_encode = 0.0
    t_ode = 0.0
    t_decode = 0.0
    t_io = 0.0
    n_pairs = 0
    n_content_encodes = 0
    n_style_encodes = 0
    n_ode = 0
    n_decode = 0

    style_latent_cache: dict[int, torch.Tensor] = {}
    model_dtype = next(lgt.model.parameters()).dtype

    def _to_model(x: torch.Tensor) -> torch.Tensor:
        return x.to(device=device, dtype=model_dtype)

    if warmup_batches > 0 and pairs:
        warm = pairs[: min(batch_size, len(pairs))]
        warm_src = _load_rgb_batch([p[0] for p in warm], image_size, device)
        warm_lat = encode_image(vae, warm_src, device=str(device))
        if abs(scale_in - 1.0) > 1e-4:
            warm_lat = warm_lat * scale_in
        warm_lat = _to_model(lgt.inversion(warm_lat))
        warm_sid = torch.tensor([p[2] for p in warm], device=device, dtype=torch.long)
        warm_ref = _load_rgb_batch(
            [style_ref_path[int(p[2])] for p in warm], image_size, device
        )
        warm_s = encode_image(vae, warm_ref, device=str(device))
        if abs(scale_in - 1.0) > 1e-4:
            warm_s = warm_s * scale_in
        warm_s = _to_model(warm_s)
        warm_out = lgt.generation_with_target_latent(
            warm_lat,
            warm_sid,
            num_steps=num_steps,
            target_style_latent={"style_latent_tensor": warm_s},
        )
        if abs(scale_out - 1.0) > 1e-4:
            warm_out = warm_out * scale_out
        _ = decode_latent(vae, warm_out, device=str(device))
        _sync(device)
        del warm_src, warm_lat, warm_sid, warm_ref, warm_s, warm_out
        if device.type == "cuda":
            torch.cuda.empty_cache()

    wall0 = time.perf_counter()
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]

        t0 = time.perf_counter()
        src_rgb = _load_rgb_batch([p[0] for p in batch], image_size, device)
        _sync(device)
        t_io += time.perf_counter() - t0

        t0 = time.perf_counter()
        content_lat = encode_image(vae, src_rgb, device=str(device))
        if abs(scale_in - 1.0) > 1e-4:
            content_lat = content_lat * scale_in
        content_lat = _to_model(lgt.inversion(content_lat))
        _sync(device)
        t_content_encode += time.perf_counter() - t0
        n_content_encodes += int(content_lat.shape[0])

        t0 = time.perf_counter()
        miss = sorted({int(p[2]) for p in batch if int(p[2]) not in style_latent_cache})
        if miss:
            ref_rgb = _load_rgb_batch(
                [style_ref_path[tid] for tid in miss], image_size, device
            )
            ref_lat = encode_image(vae, ref_rgb, device=str(device))
            if abs(scale_in - 1.0) > 1e-4:
                ref_lat = ref_lat * scale_in
            for i, tid in enumerate(miss):
                style_latent_cache[tid] = _to_model(ref_lat[i : i + 1].detach())
            n_style_encodes += len(miss)
        style_lat_batch = torch.cat(
            [style_latent_cache[int(p[2])] for p in batch], dim=0
        )
        _sync(device)
        t_style_encode += time.perf_counter() - t0

        tgt_ids = torch.tensor([p[2] for p in batch], device=device, dtype=torch.long)
        t0 = time.perf_counter()
        gen_lat = lgt.generation_with_target_latent(
            content_lat,
            tgt_ids,
            num_steps=num_steps,
            target_style_latent={"style_latent_tensor": style_lat_batch},
        )
        _sync(device)
        t_ode += time.perf_counter() - t0
        n_ode += int(gen_lat.shape[0])

        if abs(scale_out - 1.0) > 1e-4:
            gen_lat = gen_lat * scale_out
        t0 = time.perf_counter()
        _ = decode_latent(vae, gen_lat, device=str(device))
        _sync(device)
        t_decode += time.perf_counter() - t0
        n_decode += int(gen_lat.shape[0])
        n_pairs += len(batch)

        del src_rgb, content_lat, style_lat_batch, tgt_ids, gen_lat

    wall = time.perf_counter() - wall0
    pipeline = t_content_encode + t_style_encode + t_ode + t_decode
    n = max(n_pairs, 1)
    return {
        "num_steps": int(num_steps),
        "n_pairs": int(n_pairs),
        "batch_size": int(batch_size),
        "image_size": int(image_size),
        "wall_sec": float(wall),
        "pipeline_sec": float(pipeline),
        "content_encode_sec": float(t_content_encode),
        "style_encode_sec": float(t_style_encode),
        "ode_sec": float(t_ode),
        "decode_sec": float(t_decode),
        "io_sec": float(t_io),
        "n_content_encodes": int(n_content_encodes),
        "n_style_encodes": int(n_style_encodes),
        "n_ode": int(n_ode),
        "n_decode": int(n_decode),
        "ms_per_pair_pipeline": float(1000.0 * pipeline / n),
        "ms_per_pair_wall": float(1000.0 * wall / n),
        "ms_content_encode": float(1000.0 * t_content_encode / max(n_content_encodes, 1)),
        "ms_style_encode_amortized": float(1000.0 * t_style_encode / n),
        "ms_ode": float(1000.0 * t_ode / max(n_ode, 1)),
        "ms_decode": float(1000.0 * t_decode / max(n_decode, 1)),
        "pipeline_750_sec_est": float(pipeline * (750.0 / n)),
        "note": (
            "pipeline = content_encode + style_encode + ODE + decode; "
            "style-ref latent cached per target style; NO source-latent cache"
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "runs/submission/hf_oriented_internal_early_stop/epoch_0004.pt",
    )
    ap.add_argument(
        "--config_override",
        type=Path,
        default=ROOT / "inference.json",
        help="Passed to LGTInference (endpoint_adain_scale=2.0 for paper protocol)",
    )
    ap.add_argument("--test_dir", type=Path, default=ROOT / "data/test")
    ap.add_argument("--num_steps", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--image_size", type=int, default=512)
    ap.add_argument("--max_src_per_style", type=int, default=30, help="0 = all images per style")
    ap.add_argument("--include_identity", action="store_true", help="Include src_style==tgt_style pairs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--warmup_batches", type=int, default=1)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--vae_model", type=str, default="ema")
    ap.add_argument("--output", type=Path, default=ROOT / "exp/inf_timing_full_encode")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    ckpt = args.checkpoint.resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)
    test_dir = args.test_dir.resolve()
    styles = _list_styles(test_dir)
    pairs = _build_pairs(
        test_dir,
        styles,
        max_src_per_style=int(args.max_src_per_style),
        include_identity=bool(args.include_identity),
    )
    print(f"styles={styles}")
    print(f"pairs={len(pairs)} identity={'on' if args.include_identity else 'off'}")
    print(f"checkpoint={ckpt}")
    print(f"config_override={args.config_override}")
    print(f"device={device} batch_size={args.batch_size} image_size={args.image_size}")

    print("Loading VAE...")
    vae = load_vae(device=str(device), model_id=str(args.vae_model))

    print("Loading model...")
    ov = str(args.config_override) if args.config_override and args.config_override.is_file() else None
    lgt = LGTInference(
        str(ckpt),
        device=str(device),
        num_steps=int(args.num_steps[0]),
        config_override_path=ov,
    )

    model_scale = float(getattr(lgt.model, "latent_scale_factor", 0.18215))
    vae_scale = float(getattr(getattr(vae, "config", None), "scaling_factor", model_scale))
    scale_in = model_scale / max(vae_scale, 1e-8)
    scale_out = vae_scale / max(model_scale, 1e-8)
    print(
        f"latent scale model={model_scale:.6f} vae={vae_scale:.6f} "
        f"in={scale_in:.6f} out={scale_out:.6f}"
    )
    if hasattr(lgt.model, "endpoint_adain_scale"):
        print(f"endpoint_adain_scale={getattr(lgt.model, 'endpoint_adain_scale')}")

    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for k in args.num_steps:
        print("\n" + "=" * 70)
        print(f"Benchmark K={k}")
        print("=" * 70)
        row = bench_one_k(
            lgt=lgt,
            vae=vae,
            pairs=pairs,
            styles=styles,
            test_dir=test_dir,
            device=device,
            image_size=int(args.image_size),
            batch_size=int(args.batch_size),
            num_steps=int(k),
            scale_in=scale_in,
            scale_out=scale_out,
            warmup_batches=int(args.warmup_batches),
        )
        results.append(row)
        print(json.dumps(row, indent=2))

    out_json = args.output / "bench_inference_latency.json"
    payload = {
        "checkpoint": str(ckpt),
        "test_dir": str(test_dir),
        "config_override": str(args.config_override) if args.config_override else None,
        "styles": styles,
        "include_identity": bool(args.include_identity),
        "results": results,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\nSUMMARY (pipeline = content_enc + style_enc + ODE + decode)")
    print("K\tpairs\tpipeline_s\t~750s\tcontent\tstyle\tODE\tdecode\tms/pair")
    for r in results:
        print(
            f"{r['num_steps']}\t{r['n_pairs']}\t{r['pipeline_sec']:.2f}\t"
            f"{r['pipeline_750_sec_est']:.1f}\t{r['content_encode_sec']:.2f}\t"
            f"{r['style_encode_sec']:.2f}\t{r['ode_sec']:.2f}\t{r['decode_sec']:.2f}\t"
            f"{r['ms_per_pair_pipeline']:.1f}"
        )
    print(f"\nWROTE {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
