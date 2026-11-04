"""Environment/asset probe for the supervision audit.

Checks: latent cache shapes, filename->artist parsing, DINO token cache,
VAE decode path, and the five style order used everywhere else.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from paths import dino_cache_root, hf_cache_root  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]


def main() -> None:
    packed = ROOT / "data/train/.latent_cache/packed/packed"
    files = sorted(packed.glob("*.pt"))
    out = {"packed_files": [p.name for p in files], "styles": {}}
    for p in files:
        d = torch.load(p, map_location="cpu", weights_only=False)
        lat = d["latents"]
        stem = p.stem.split("_", 1)[1]
        out["styles"][stem] = {
            "keys": list(d.keys()),
            "n": int(lat.shape[0]),
            "shape": list(lat.shape[1:]),
            "dtype": str(lat.dtype),
            "min": float(lat.min()),
            "max": float(lat.max()),
            "mean": float(lat.mean()),
            "std": float(lat.std()),
            "first_file": d["files"][0],
            "last_file": d["files"][-1],
        }
        if stem == "Early_Renaissance":
            out["artists"] = sorted({f.split("__")[1].split("_", 1)[0] for f in d["files"]})[:12]
            out["n_artists"] = len({f.split("__")[1].split("_", 1)[0] for f in d["files"]})

    dinoc = dino_cache_root("content_5k")
    out["dino_manifest"] = json.loads((dinoc / "manifest.json").read_text())
    sample = next((dinoc / "Early_Renaissance").glob("*.pt"))
    tok = torch.load(sample, map_location="cpu", weights_only=False)
    out["dino_sample"] = {"file": sample.name, "keys": list(tok), "tokens": list(tok["tokens"].shape),
                          "grid": tok["grid"], "dtype": str(tok["tokens"].dtype)}
    out["dino_n_files_er"] = len(list((dinoc / "Early_Renaissance").glob("*.pt")))

    # VAE presence
    from utils.inference import load_vae, decode_latent  # noqa: E402
    vae = load_vae("cuda", model_id="ema", cache_dir=str(hf_cache_root()),
                   compile_decoder=False)
    z = torch.load(files[0], map_location="cpu", weights_only=False)["latents"][:2].float().cuda()
    with torch.inference_mode():
        img = decode_latent(vae, z, "cuda").float()
    out["decode"] = {"shape": list(img.shape), "min": float(img.min()), "max": float(img.max()),
                     "mean": float(img.mean())}
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
