"""Verify the real DCAE checkpoint: load, encode/decode a real WikiArt image, report PSNR.

Runs inside WSL where the checkpoint is visible at
/mnt/i/Latent_Style/weights/dc-ae-f32c32-mix-1.0-diffusers
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

DCAE = "/mnt/i/Latent_Style/weights/dc-ae-f32c32-mix-1.0-diffusers"
REPO = Path("/mnt/i/Latent_Style/WEAVE")
IMAGES = [
    REPO / "datasets/distinct5_512/train/Impressionism",
    REPO / "datasets/distinct5_512/train/Ukiyo_e",
]


def image_tensor(p: Path, size: int = 512) -> torch.Tensor:
    with Image.open(p) as im:
        im = im.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
        t = torch.from_numpy(np.asarray(im, dtype=np.uint8).copy()).permute(2, 0, 1).float() / 255.0
    return t


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from diffusers import AutoencoderDC

    out = {"device": device, "dcae_path": DCAE, "exists": Path(DCAE).exists()}
    t0 = __import__("time").time()
    ae = AutoencoderDC.from_pretrained(DCAE).eval().to(device)
    for p in ae.parameters():
        p.requires_grad_(False)
    out["load_sec"] = round(__import__("time").time() - t0, 1)
    out["scaling_factor"] = float(ae.config.scaling_factor)
    out["n_params_M"] = round(sum(p.numel() for p in ae.parameters()) / 1e6, 1)

    rows = []
    for folder in IMAGES:
        files = sorted(folder.glob("*.jpg"))[:2]
        for p in files:
            x = image_tensor(p)[None].to(device) * 2 - 1
            with torch.inference_mode():
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                    z = ae.encode(x).latent.float()
                    rec = ae.decode(z).sample.float().clamp(-1, 1)
            mse = float((rec - x).square().mean())
            psnr = -10 * math.log10(max(mse / 4.0, 1e-12))
            rows.append({
                "file": p.name,
                "style": folder.name,
                "latent_shape": list(z.shape),
                "latent_mean": round(float(z.mean()), 4),
                "latent_std": round(float(z.std()), 4),
                "roundtrip_psnr_db": round(psnr, 2),
                "finite": bool(torch.isfinite(rec).all()),
            })
    out["roundtrips"] = rows
    if device == "cuda":
        out["peak_vram_mib"] = round(torch.cuda.max_memory_allocated(0) / 1048576, 1)
    print(json.dumps(out, indent=2))
    ok = all(r["finite"] for r in rows) and all(r["roundtrip_psnr_db"] > 15 for r in rows)
    print("DCAE ROUNDTRIP", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
