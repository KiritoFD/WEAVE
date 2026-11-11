"""WEAVE — re-encode the 15,545-image 5-style training pool with the A0 Encoder.

Produces `orig_mean_fp16_a0/` with EXACTLY the same schema as the existing
`orig_mean_fp16/` (sd-vae-ft-ema) so train_eval_v6.py can switch via --latent-dir.
Only the encoder is used; the latent space is never trained, just re-encoded.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, str(ROOT))

from tools.v6_next.w_vae import build_a0_encoder

STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
CANDIDATES = [
    ROOT / "pseudo_gt_24h" / "orig_mean_fp16",
    Path("/mnt/i/Latent_Style/pseudo_gt_24h/orig_mean_fp16"),
    Path("/mnt/i/Latent_Style/WEAVE/datasets/../pseudo_gt_24h/orig_mean_fp16"),
]


def resolve_src_dir() -> Path:
    for p in CANDIDATES:
        if (p / "00_Early_Renaissance.pt").exists():
            return p
    raise FileNotFoundError(f"training latent pool not found; tried {CANDIDATES}")


def main():
    ap = argparse.ArgumentParser(description="A0 re-encoding of the training pool")
    ap.add_argument("--out-dir", default="pseudo_gt_24h/orig_mean_fp16_a0")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="smoke: only first N per style")
    args = ap.parse_args()

    src = resolve_src_dir()
    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    print(f"[A0Reencode] src={src}  out={out}", flush=True)

    vae = build_a0_encoder(args.device)
    sf = float(vae.config.scaling_factor)
    manifest = {}
    t0 = time.time()

    for sid, s in enumerate(STYLES):
        p = src / f"{sid:02d}_{s}.pt"
        d = torch.load(p, map_location="cpu", weights_only=False)
        files, ids = list(d["files"]), list(d["ids"])
        if args.limit:
            files, ids = files[: args.limit], ids[: args.limit]
        count = len(files)
        lat = torch.empty((count, 4, 64, 64), dtype=torch.float16)
        t1 = time.time()
        for i in range(0, count, args.batch):
            chunk = files[i : i + args.batch]
            ims = [Image.open(f).convert("RGB").resize((512, 512)) for f in chunk]
            arr = np.stack([np.array(im, dtype=np.float32) / 255.0 for im in ims])
            x01 = torch.from_numpy(arr).permute(0, 3, 1, 2).to(args.device)
            xpm1 = x01 * 2.0 - 1.0
            with torch.inference_mode():
                z = vae.encode(xpm1).latent_dist.mean * sf
            lat[i : i + len(chunk)] = z.float().cpu().half()
            if (i // args.batch) % 40 == 0:
                print(f"  {s:20s} {min(i+args.batch,count)}/{count}", flush=True)
        rec = {
            "schema": "mean_latent_fp16_v1",
            "subdir": s,
            "count": count,
            "files": files,
            "ids": ids,
            "latents": lat,
            "scaling_factor": sf,
            "vae": "A0 (art-data finetuned sd-vae-ft-ema encoder)",
        }
        op = out / f"{sid:02d}_{s}.pt"
        torch.save(rec, op)
        manifest[s] = {"n": count, "file": str(op), "elapsed_sec": round(time.time()-t1, 1)}
        print(f"  [OK] {s:20s} n={count:5d} -> {op}  ({time.time()-t1:.0f}s)", flush=True)

    (out / "reencode_manifest.json").write_text(
        json.dumps({"source": str(src), "encoder": "A0 encoder-side",
                    "scaling_factor": sf, "total": sum(m["n"] for m in manifest.values()),
                    "styles": manifest}, indent=2), encoding="utf-8")
    print(f"[A0Reencode] DONE total={sum(m['n'] for m in manifest.values())} "
          f"in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
