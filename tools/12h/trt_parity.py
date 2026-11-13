"""TRT vs torch VAE-decoder parity check (PSNR gate for the fast eval path).

Decodes the same real latents with the torch decoder and the TensorRT engine,
then reports PSNR / max-diff / channel bias / NaN-Inf.  Gate: PSNR >= 40 dB.

Usage:
  python3 tools/12h/trt_parity.py --size 512 --batch 8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "tools" / "trt"))

from tools.legacy_utils.inference import decode_latent, load_vae  # noqa: E402
from trt_vae_decoder import TrtVaeDecoder  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--latents", default=str(_ROOT / "runs" / "trt" / "latents" / "real_latents.pt"))
    ap.add_argument("--engine-dir", default=str(_ROOT / "runs" / "trt" / "engines"))
    ap.add_argument("--vae-cache", default="/mnt/i/Latent_Style/eval_cache/hf")
    ap.add_argument("--out", default=str(_ROOT / "runs" / "trt" / "bench" / "parity_512_b8.json"))
    args = ap.parse_args()

    blob = torch.load(args.latents, map_location="cuda", weights_only=False)
    key = f"items_{args.size}"
    if key not in blob:
        raise SystemExit(f"{args.latents}: missing {key}; have {list(blob)}")
    items = [v.float() for v in blob[key].values()]
    if not items:
        raise SystemExit(f"{key} is empty")
    while len(items) < args.batch:
        items = items + items
    z = torch.stack(items[: args.batch]).cuda()

    vae = load_vae("cuda", model_id="ema", cache_dir=args.vae_cache, compile_decoder=False)
    dec = TrtVaeDecoder(args.engine_dir, vae, precision="fp16", device="cuda")
    if not dec.usable(args.batch, args.size):
        raise SystemExit(f"no engine for size={args.size} batch={args.batch}")

    with torch.inference_mode():
        ref = decode_latent(vae, z, "cuda").float()
        got = dec.decode(z).float()
    torch.cuda.synchronize()

    ref_np = ref.cpu().numpy()
    got_np = got.cpu().numpy()
    mse = float(np.mean((ref_np - got_np) ** 2))
    diff = np.abs(ref_np - got_np)
    row = {
        "size": args.size,
        "batch": args.batch,
        "latents": args.latents,
        "n": int(z.shape[0]),
        "psnr_db": round(float(10 * np.log10(1.0 / max(mse, 1e-12))), 2),
        "max_abs_255": round(float(diff.max() * 255), 3),
        "mean_abs_255": round(float(diff.mean() * 255), 4),
        "nan": int(np.isnan(got_np).sum()),
        "inf": int(np.isinf(got_np).sum()),
        "channel_bias": [round(float((got_np[:, c] - ref_np[:, c]).mean()), 5) for c in range(3)],
        "gate_psnr40": bool(float(10 * np.log10(1.0 / max(mse, 1e-12))) >= 40.0),
    }
    print(json.dumps(row, indent=1))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(row, indent=1), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()

