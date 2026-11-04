"""Resize D5 train images to clean 512x512 JPEG for fast VAE encode benches."""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paths import datasets_root  # noqa: E402

STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
EXTS = {".jpg", ".jpeg", ".png", ".webp"}
SIZE = 512


def process_one(src: Path, dst: Path, quality: int) -> tuple[str, bool]:
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and dst.stat().st_size > 0:
            return (str(src), True)
        img = Image.open(src).convert("RGB")
        if img.size != (SIZE, SIZE):
            img = img.resize((SIZE, SIZE), Image.BILINEAR)
        # Progressive off, optimize for smaller decode cost later
        img.save(dst, format="JPEG", quality=quality, optimize=False, progressive=False)
        return (str(src), True)
    except Exception as e:
        return (f"{src}: {e}", False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src",
        type=str,
        default=str(datasets_root("distinct5_512", "train")),
    )
    ap.add_argument(
        "--dst",
        type=str,
        default=str(datasets_root("train_512clean")),
    )
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--quality", type=int, default=95)
    args = ap.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)
    jobs: list[tuple[Path, Path]] = []
    for style in STYLES:
        sdir = src_root / style
        for p in sorted(sdir.iterdir()):
            if p.suffix.lower() in EXTS:
                # always write .jpg for uniform decode path
                jobs.append((p, dst_root / style / (p.stem + ".jpg")))

    print(f"jobs={len(jobs)} workers={args.workers} dst={dst_root}")
    t0 = time.perf_counter()
    ok = 0
    fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(process_one, s, d, args.quality) for s, d in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            msg, success = fut.result()
            if success:
                ok += 1
            else:
                fail += 1
                print("FAIL", msg)
            if i % 500 == 0 or i == len(futs):
                print(f"  {i}/{len(futs)} ok={ok} fail={fail} elapsed={time.perf_counter()-t0:.1f}s")
    print(f"DONE ok={ok} fail={fail} wall={time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
