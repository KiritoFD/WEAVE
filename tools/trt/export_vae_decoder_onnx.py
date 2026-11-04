"""Export the SD1.5 VAE decoder to ONNX (CPU only, no GPU needed).

The exported graph is the exact decode path used by ``utils.inference.decode_latent``:

    z (latent) -> z / scaling_factor -> post_quant_conv -> decoder -> (x+1)/2 -> clamp[0,1]

Only the batch dimension stays dynamic, so one ONNX file per spatial size serves
every engine batch (engines are then built with fixed min=opt=max shapes).

Usage:
  python tools/trt/export_vae_decoder_onnx.py --sizes 64,256,512
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from paths import hf_cache_root, runs_root  # noqa: E402
from utils.inference import load_vae  # noqa: E402


class DecoderExport(torch.nn.Module):
    """Mirror of `decode_latent`'s eager fallback path."""

    def __init__(self, vae) -> None:
        super().__init__()
        self.post_quant_conv = vae.post_quant_conv
        self.decoder = vae.decoder
        self.scaling_factor = float(vae.config.scaling_factor)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = z / max(self.scaling_factor, 1e-8)
        z = self.post_quant_conv(z)
        out = self.decoder(z)
        if hasattr(out, "sample"):  # older diffusers returns a DecoderOutput
            out = out.sample
        out = (out + 1.0) / 2.0
        return torch.clamp(out, 0.0, 1.0)


def export_one(mod: DecoderExport, size: int, out_path: Path) -> dict:
    lat = size // 8
    z = torch.randn(1, 4, lat, lat, dtype=torch.float32)
    t0 = time.perf_counter()
    torch.onnx.export(
        mod,
        (z,),
        str(out_path),
        input_names=["z"],
        output_names=["image"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes={"z": {0: "batch"}, "image": {0: "batch"}},
        dynamo=False,
    )
    export_sec = time.perf_counter() - t0
    return {
        "size": size,
        "latent": [lat, lat],
        "onnx_mb": round(out_path.stat().st_size / (1024 * 1024), 1),
        "export_sec": round(export_sec, 1),
    }


def check_ort(mod: DecoderExport, onnx_path: Path, size: int, n: int = 2) -> dict:
    """CPU parity: torch fp32 vs onnxruntime CPU."""
    import onnxruntime as ort

    lat = size // 8
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    worst = 0.0
    for _ in range(n):
        z = torch.randn(1, 4, lat, lat, dtype=torch.float32)
        with torch.inference_mode():
            ref = mod(z).numpy()
        got = sess.run(None, {"z": z.numpy()})[0]
        worst = max(worst, float(np.abs(ref - got).max()))
    return {"ort_max_abs_diff_fp32": worst}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="64,256,512")
    ap.add_argument("--out_dir", default=str(runs_root("trt", "onnx")))
    ap.add_argument("--cache_dir", default=str(hf_cache_root()))
    ap.add_argument("--vae_model", default="ema")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vae = load_vae("cpu", model_id=args.vae_model, cache_dir=args.cache_dir, compile_decoder=False)
    vae = vae.float().eval()
    mod = DecoderExport(vae).eval()

    report = {"vae_model": args.vae_model, "out_dir": str(out_dir), "shapes": []}
    for size in [int(x) for x in args.sizes.split(",") if x.strip()]:
        path = out_dir / f"vae_decoder_{size}.onnx"
        info = export_one(mod, size, path)
        info.update(check_ort(mod, path, size))
        info["onnx"] = str(path)
        report["shapes"].append(info)
        print(json.dumps(info))

    (out_dir / "export_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("wrote", out_dir / "export_report.json")


if __name__ == "__main__":
    main()

