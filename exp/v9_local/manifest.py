"""Write run manifest + summary for v9_local arms (provenance per plan)."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "runs/v9_local_4070"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    man = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "machine": "local RTX 4070 Laptop 8GB (~6GiB usable)",
        "n2c_ckpt_sha256": sha(REPO / "runs/v8/train/N2C/step_04000.pt"),
        "tasks_sha256": sha(REPO / "eval_protocol/tasks_750.json"),
        "bank_sha256": sha(OUT / "bank/style_bank.pt"),
        "n2c_cache": "runs/v8/N2C.e/gen_latents (stripped from remote gen, 94 batches)",
        "vae_decode": "diffusers sd-vae-ft-ema fallback (no TensorRT locally)",
        "seed": 0,
        "inference_batch": 4,
        "solver": {"k_steps": 2, "etas": [0.5, 1.0], "eps": 0.10,
                   "rho": 0.15, "mask_frac": 0.25,
                   "proxy": "margin - 0.3*L_loc"},
        "note": "latent-cache ablation; NOT end-to-end WEAVE latency",
    }
    (OUT / "manifest.json").write_text(json.dumps(man, indent=1))
    print(json.dumps(man, indent=1)[:400])


if __name__ == "__main__":
    main()
