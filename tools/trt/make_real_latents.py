"""Build real latent sets for VAE-decoder parity / benchmarking (CPU only).

The task asked for ``data/train_d5/.latent_cache.packed``; that file does not exist
on this machine.  What does exist are the evaluator's source-latent caches
(``/mnt/i/Latent_Style/eval_cache/source_latents_*.pt``): VAE-encoded latents of
real test photographs, fp16, shape [4,32,32] (i.e. 256x256 pixels).

So:
  * 256x256 : real latents, taken straight from those caches (no resampling).
  * 64x64   : bicubic resample of the real 32x32 latents to 8x8  (marked derived).
  * 512x512 : bicubic resample of the real 32x32 latents to 64x64 (marked derived).

Derived shapes are only used for the 64/512 timing + parity sweeps; the 256 case
(the one the eval protocol actually uses) is 100% real VAE output.

Usage:
  python tools/trt/make_real_latents.py --per_style 8
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

CACHE_DIR = Path("/mnt/i/Latent_Style/eval_cache")


def load_real_latents(per_style: int) -> tuple[dict[str, torch.Tensor], dict]:
    items: dict[str, torch.Tensor] = {}
    meta: dict = {}
    for path in sorted(CACHE_DIR.glob("source_latents_*.pt")):
        blob = torch.load(path, map_location="cpu", weights_only=False)
        sub = blob.get("items", {})
        if not sub:
            continue
        style = Path(next(iter(sub))).parent.name
        chosen = sorted(sub)[:per_style]
        for key in chosen:
            items[f"{style}/{Path(key).name}"] = sub[key].float()
        meta[style] = {"file": path.name, "used": len(chosen), "count": len(sub)}
    return items, meta


def resample(items: dict[str, torch.Tensor], lat: int) -> dict[str, torch.Tensor]:
    out = {}
    for key, tensor in items.items():
        z = tensor.unsqueeze(0)
        z = F.interpolate(z, size=(lat, lat), mode="bicubic", align_corners=False)
        out[key] = z.squeeze(0).contiguous()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_style", type=int, default=8)
    ap.add_argument("--out_dir", default="/mnt/i/Latent_Style/WEAVE/runs/trt/latents")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    real, meta = load_real_latents(args.per_style)
    payload = {
        "meta": {
            "source": "eval_cache/source_latents_*.pt",
            "styles": meta,
            "dtype": "float32 (upcast from fp16 cache)",
            "per_style": args.per_style,
            "total": len(real),
            "note": "256 is real VAE encoder output; 64/512 are bicubic resamples",
        },
        "items_256": real,
        "items_64": resample(real, 8),
        "items_512": resample(real, 64),
    }
    path = out_dir / "real_latents.pt"
    torch.save(payload, path)
    print(json.dumps({"file": str(path), "total": len(real),
                      "mb": round(path.stat().st_size / 1024 / 1024, 2)}, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()

