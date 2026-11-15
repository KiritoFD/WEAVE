"""Encode V15 eval payloads (rgb_u8) to 512px lossless PNGs.

Run on the 3060: python -m exp.v15.encode_png
Writes runs/v15/eval/png/<arm>/t0_<lev>/<task_id>.png (pixel-identical to
rgb_u8, so local metric recomputation is faithful).  CPU-only task.
"""
import time
from pathlib import Path

import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "runs/v15/eval/raw"
PNG = REPO / "runs/v15/eval/png"


def main():
    n = 0
    t0 = time.time()
    metas = sorted(RAW.glob("*/*/meta.json"))
    print(f"encoding payloads under {RAW} ({len(metas)} level-dirs)", flush=True)
    for meta in metas:
        ldir = meta.parent
        rel = ldir.relative_to(RAW)
        out_dir = PNG / rel
        out_dir.mkdir(parents=True, exist_ok=True)
        for bp in sorted(ldir.glob("batch_*.pt")):
            payload = torch.load(bp, map_location="cpu", weights_only=False)
            u8 = payload["rgb_u8"]
            for tid, arr in zip(payload["ids"],
                                u8.permute(0, 2, 3, 1).numpy()):
                p = out_dir / (tid.replace("/", "__") + ".png")
                if p.exists():
                    continue
                Image.fromarray(arr).save(p, format="PNG")
                n += 1
        print(f"{rel}: new={n} {time.time()-t0:.0f}s", flush=True)
    total = sum(p.stat().st_size for p in PNG.rglob("*.png"))
    n_files = sum(1 for _ in PNG.rglob("*.png"))
    print(f"PNG_DONE n={n_files} total={total/1e9:.2f}GB "
          f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
