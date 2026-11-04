"""Offline (CPU) sanity + parity check for the exported decoder ONNX files.

Question this answers: "does pure ONNX export (fp16 / mixed) break the decoder,
before we ever build a TensorRT engine?"  It needs no GPU.

Reference: torch fp32 `decode_latent` on CPU.  Candidates: fp32 / fp16 / mixed
ONNX executed with onnxruntime CPU.  Reports NaN/Inf, max|Δ| (in 255 levels),
PSNR, SSIM, per-channel bias.

Usage:
  python tools/trt/offline_check_onnx.py --size 256 --batch 4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from skimage.metrics import structural_similarity

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from paths import hf_cache_root, runs_root  # noqa: E402
from utils.inference import decode_latent, load_vae  # noqa: E402


def compare(ref: np.ndarray, got: np.ndarray) -> dict:
    diff = np.abs(ref - got)
    mse = float(np.mean((ref - got) ** 2))
    ssims = [
        structural_similarity(np.transpose(ref[i], (1, 2, 0)), np.transpose(got[i], (1, 2, 0)),
                              channel_axis=2, data_range=1.0)
        for i in range(ref.shape[0])
    ]
    return {
        "max_abs_255": round(float(diff.max() * 255), 3),
        "mean_abs_255": round(float(diff.mean() * 255), 5),
        "psnr_db": round(float(10 * np.log10(1.0 / max(mse, 1e-12))), 2),
        "ssim_mean": round(float(np.mean(ssims)), 5),
        "nan": int(np.isnan(got).sum()),
        "inf": int(np.isinf(got).sum()),
        "channel_bias": [round(float((got[:, c] - ref[:, c]).mean()), 5) for c in range(3)],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--onnx_dir", default=str(runs_root("trt", "onnx")))
    ap.add_argument("--latents", default=str(runs_root("trt", "latents") / "real_latents.pt"))
    ap.add_argument("--variants", default=",fp16,mixed", help="'' = fp32 file, else suffix")
    ap.add_argument("--vae_cache", default=str(hf_cache_root()))
    args = ap.parse_args()

    blob = torch.load(args.latents, map_location="cpu", weights_only=False)
    items = [v.float() for v in blob[f"items_{args.size}"].values()][: args.batch]
    z = torch.stack(items)
    if z.shape[-1] != args.size // 8:
        import torch.nn.functional as F
        z = F.interpolate(z, size=(args.size // 8, args.size // 8), mode="bicubic", align_corners=False)

    vae = load_vae("cpu", model_id="ema", cache_dir=args.vae_cache, compile_decoder=False).float()
    t0 = time.perf_counter()
    with torch.inference_mode():
        ref = decode_latent(vae, z, "cpu").numpy()
    ref_sec = time.perf_counter() - t0
    print(f"torch fp32 CPU reference: {ref_sec:.2f}s for {len(z)} imgs "
          f"({ref_sec/len(z)*1000:.0f} ms/img)")

    rows = [{"variant": "torch_fp32_cpu", "sec": round(ref_sec, 2),
             "ms_per_img": round(ref_sec / len(z) * 1000, 1)}]
    for suffix in [s for s in args.variants.split(",")]:
        path = Path(args.onnx_dir) / f"vae_decoder_{args.size}_{suffix}.onnx" if suffix \
            else Path(args.onnx_dir) / f"vae_decoder_{args.size}.onnx"
        if not path.exists():
            rows.append({"variant": suffix or "fp32", "error": "missing file"})
            continue
        try:
            sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            in_type = sess.get_inputs()[0].type
            feed = z.numpy().astype(np.float16) if "float16" in in_type else z.numpy()
            sess.run(None, {"z": feed})  # warmup
            t0 = time.perf_counter()
            got = sess.run(None, {"z": feed})[0].astype(np.float32)
            sec = time.perf_counter() - t0
            row = {"variant": suffix or "fp32", "io_type": in_type,
                   "sec": round(sec, 2), "ms_per_img": round(sec / len(z) * 1000, 1)}
            row.update(compare(ref, got))
        except Exception as exc:  # noqa: BLE001
            row = {"variant": suffix or "fp32", "error": repr(exc)[:300]}
        rows.append(row)
        print(json.dumps(row), flush=True)

    out = runs_root("trt", "bench") / f"offline_check_{args.size}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()

