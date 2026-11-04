"""Build fixed-shape TensorRT engines for the SD1.5 VAE decoder (SM86, this 3060).

TensorRT 11 notes:
  * ``BuilderFlag.FP16`` no longer exists -- precision comes from the ONNX tensor
    types, so fp16 engines are built from ``vae_decoder_<size>_fp16.onnx`` and
    flagged ``STRONGLY_TYPED`` so TRT does not silently widen to fp32.
  * ``trtexec`` is not shipped in the pip wheel; we use the Python builder API.
  * One engine per (size, batch): min == opt == max, i.e. a fixed shape.

Usage:
  python tools/trt/build_trt_engines.py --sizes 64,512 --batches 1,8,32 --precision fp16
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import tensorrt as trt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paths import runs_root  # noqa: E402


def parse_onnx(path: Path, strongly_typed: bool) -> tuple[trt.INetworkDefinition, object]:
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 0
    if strongly_typed:
        flags |= int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    raw = path.read_bytes()
    if not parser.parse(raw):
        errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError("ONNX parse failed: " + " | ".join(errs))
    return network, builder


def build(onnx_path: Path, size: int, batch: int, *, precision: str,
          workspace_mb: int, engine_dir: Path) -> dict:
    lat = size // 8
    strongly_typed = precision == "fp16"
    t0 = time.perf_counter()
    network, builder = parse_onnx(onnx_path, strongly_typed)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * 1024 * 1024)
    if precision == "fp32":
        config.clear_flag(trt.BuilderFlag.TF32)  # exact fp32, no TF32 shortcut

    shape = (batch, 4, lat, lat)
    profile = builder.create_optimization_profile()
    profile.set_shape("z", shape, shape, shape)
    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"build returned None (size={size} batch={batch} {precision})")

    engine_dir.mkdir(parents=True, exist_ok=True)
    engine_path = engine_dir / f"vae_decoder_{size}_b{batch}_{precision}.engine"
    engine_path.write_bytes(bytes(serialized))
    return {
        "size": size,
        "batch": batch,
        "precision": precision,
        "engine": str(engine_path),
        "engine_mb": round(engine_path.stat().st_size / 1024 / 1024, 1),
        "build_sec": round(time.perf_counter() - t0, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx_dir", default=str(runs_root("trt", "onnx")))
    ap.add_argument("--engine_dir", default=str(runs_root("trt", "engines")))
    ap.add_argument("--sizes", default="64,512")
    ap.add_argument("--batches", default="1,8,32")
    ap.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--workspace_mb", type=int, default=2048)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    onnx_dir = Path(args.onnx_dir)
    engine_dir = Path(args.engine_dir)
    rows = []
    for size in [int(x) for x in args.sizes.split(",") if x.strip()]:
        suffix = "_fp16" if args.precision == "fp16" else ""
        onnx_path = onnx_dir / f"vae_decoder_{size}{suffix}.onnx"
        if not onnx_path.exists():
            print(f"skip size={size}: missing {onnx_path}")
            continue
        for batch in [int(x) for x in args.batches.split(",") if x.strip()]:
            target = engine_dir / f"vae_decoder_{size}_b{batch}_{args.precision}.engine"
            if target.exists() and not args.overwrite:
                rows.append({"size": size, "batch": batch, "precision": args.precision,
                             "engine": str(target), "cached": True})
                print(json.dumps(rows[-1]))
                continue
            row = build(onnx_path, size, batch, precision=args.precision,
                        workspace_mb=args.workspace_mb, engine_dir=engine_dir)
            rows.append(row)
            print(json.dumps(row), flush=True)

    out = engine_dir / f"build_report_{args.precision}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()

