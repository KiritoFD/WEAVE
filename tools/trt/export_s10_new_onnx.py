"""Export SD0-S10 / official VAE decoders to ONNX for TensorRT (CPU only).

Mirrors tools/trt/export_vae_decoder_onnx.py exactly (same graph contract):
    z -> z / scaling_factor -> post_quant_conv -> decoder -> (x+1)/2 -> clamp[0,1]
dynamic batch, opset 17, input "z", output "image".

Variants:
  official : sd-vae-ft-ema 49.49M decoder (from deliver/sd0_s10_profile/sd0_vae)
  old_s10  : runs/vae_decoder_prune/s10_p5_lowlr/decoder.pt (A0-trained S10)
  c0       : deliver/sd0_s10_profile/sd0_s10_best_recon.pt   (new SD0-S10)
  c1       : deliver/sd0_s10_profile/sd0_s10_best_overall.pt (new SD0-S10)

Usage (3060, Windows python):
  python tools/trt/export_s10_new_onnx.py --variants official,old_s10,c0,c1 --sizes 512
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

PROF = _ROOT / "deliver" / "sd0_s10_profile"
OUT_DIR = _ROOT / "runs" / "trt" / "onnx"


class Wrap(torch.nn.Module):
    """Same contract as the paper's DecoderExport."""

    def __init__(self, pqc: torch.nn.Module, dec: torch.nn.Module, sf: float) -> None:
        super().__init__()
        self.post_quant_conv = pqc
        self.decoder = dec
        self.scaling_factor = float(sf)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = z / max(self.scaling_factor, 1e-8)
        z = self.post_quant_conv(z)
        out = self.decoder(z)
        if hasattr(out, "sample"):
            out = out.sample
        out = (out + 1.0) / 2.0
        return torch.clamp(out, 0.0, 1.0)


def load_official(dev: str) -> tuple[Wrap, dict]:
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(str(PROF / "sd0_vae"), torch_dtype=torch.float32,
                                        local_files_only=True).to(dev).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return Wrap(vae.post_quant_conv, vae.decoder, vae.config.scaling_factor), {
        "params_m": round(sum(p.numel() for p in vae.decoder.parameters()) / 1e6, 3)}


def _strip(sd: dict) -> dict:
    out = {}
    for k, v in sd.items():
        for pre in ("decoder.", "model.", "_orig_mod."):
            if k.startswith(pre):
                k = k[len(pre):]
                break
        out[k] = v
    return out


def load_s10(ckpt_path: str, dev: str) -> tuple[Wrap, dict]:
    from diffusers.models.autoencoders.vae import Decoder
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = _strip(ck.get("state_dict", ck))
    dec = Decoder(in_channels=4, out_channels=3,
                  up_block_types=["UpDecoderBlock2D"] * 4,
                  block_out_channels=[64, 128, 192, 256], layers_per_block=2,
                  norm_num_groups=32, act_fn="silu", mid_block_add_attention=True)
    dec.load_state_dict(sd, strict=True)
    pq = torch.nn.Conv2d(4, 4, 1)
    pq.load_state_dict(ck.get("post_quant_conv", ck))
    dec = dec.to(dev).eval()
    pq = pq.to(dev).eval()
    for p in dec.parameters():
        p.requires_grad_(False)
    for p in pq.parameters():
        p.requires_grad_(False)
    n = sum(p.numel() for p in dec.parameters())
    return Wrap(pq, dec, 0.18215), {"params_m": round(n / 1e6, 3)}


def export_one(mod: Wrap, size: int, out_path: Path) -> dict:
    lat = size // 8
    z = torch.randn(1, 4, lat, lat, dtype=torch.float32)
    t0 = time.perf_counter()
    with torch.no_grad():
        torch.onnx.export(
            mod, (z,), str(out_path),
            input_names=["z"], output_names=["image"],
            opset_version=17, do_constant_folding=True,
            dynamic_axes={"z": {0: "batch"}, "image": {0: "batch"}},
            dynamo=False)
    return {"size": size, "latent": [lat, lat],
            "onnx_mb": round(out_path.stat().st_size / (1024 * 1024), 1),
            "export_sec": round(time.perf_counter() - t0, 1)}


def check_ort(mod: Wrap, onnx_path: Path, size: int, n: int = 2) -> dict:
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
    ap.add_argument("--variants", default="official,old_s10,c0,c1")
    ap.add_argument("--sizes", default="512")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    loaders = {
        "official": lambda: load_official(args.device),
        "old_s10": lambda: load_s10(str(_ROOT / "runs/vae_decoder_prune/s10_p5_lowlr/decoder.pt"), args.device),
        "c0": lambda: load_s10(str(PROF / "sd0_s10_best_recon.pt"), args.device),
        "c1": lambda: load_s10(str(PROF / "sd0_s10_best_overall.pt"), args.device),
    }
    report = {}
    for v in [x.strip() for x in args.variants.split(",") if x.strip()]:
        mod, info = loaders[v]()
        mod = mod.cpu()   # export on CPU like the paper script
        rows = {}
        for size in [int(x) for x in args.sizes.split(",") if x.strip()]:
            suffix = "_fp16" if False else ""   # keep base name; fp16 engines are built STRONGLY_TYPED from this onnx
            out_path = OUT_DIR / f"s10new_{v}_decoder_{size}{suffix}.onnx"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            row = export_one(mod, size, out_path)
            row.update(check_ort(mod, out_path, size))
            rows[size] = row
            print(json.dumps({"variant": v, **row}), flush=True)
        report[v] = {"info": info, "onnx": rows}
        (OUT_DIR / f"s10new_export_report.json").write_text(
            json.dumps(report, indent=1), encoding="utf-8")
    print("done", flush=True)


if __name__ == "__main__":
    main()
