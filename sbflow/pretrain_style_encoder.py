"""Train the single shared frozen style encoder used by every learned-SWD arm."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sbflow.data5 import STYLE_NAMES, load_caches  # noqa: E402
from sbflow.style_loss import train_style_encoder  # noqa: E402

CACHE_ROOT = "/mnt/i/Latent_Style/pseudo_gt_24h/orig_mean_fp16"
FILES = ["00_Early_Renaissance.pt", "01_Impressionism.pt", "02_Minimalism.pt",
         "03_Rococo.pt", "04_Ukiyo_e.pt"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        print(f"[style-enc] already exists: {out}")
        return
    data = load_caches([Path(f"{CACHE_ROOT}/{f}") for f in FILES], args.device, STYLE_NAMES)
    enc = train_style_encoder(data.z, epochs=args.epochs, device=args.device,
                              seed=args.seed, log=lambda m: print(m, flush=True))
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": enc.state_dict(), "channels": int(data.z[0].shape[1]),
                "seed": args.seed, "epochs": args.epochs}, out)
    print(f"[style-enc] saved {out}")


if __name__ == "__main__":
    main()
