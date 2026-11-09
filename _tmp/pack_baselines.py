"""Package the paper baselines' evaluation images (13 methods x 3 datasets).

For each (dataset, method) it re-encodes the generated PNGs to JPEG q=90 and writes
one tar.gz per dataset. Nested style subdirectories (e.g. seedream) are included and
de-duplicated by stem so a method never contributes more than its 750 tasks.

Also emits metrics CSV copies and a contact sheet per method, plus inventory.json
recording exact counts and what is missing.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

REPO = Path("/mnt/i/Latent_Style/WEAVE")
STAGE = REPO / "_baseline_stage"
STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
METHODS = ["identity", "adain", "wct", "cut", "samst", "samam", "sdturbo",
           "styleid", "styleshot", "stytr2", "aespa", "weave", "seedream"]
DATASETS = {
    "D5-512": REPO / "results" / "D5-512",
    "P2A-256": REPO / "results" / "P256",
    "R5-WikiArt": REPO / "results" / "R5-WikiArt",
}
EXTS = (".png", ".jpg", ".jpeg")
CELL = 160
COLS, ROWS = 5, 4


def collect(mdir: Path) -> list[Path]:
    """Flat files first; otherwise fall back to nested, dedup by stem."""
    flat = sorted(p for p in mdir.iterdir() if p.is_file() and p.suffix.lower() in EXTS)
    if flat:
        return flat
    seen: dict[str, Path] = {}
    for p in sorted(mdir.rglob("*")):
        if p.is_file() and p.suffix.lower() in EXTS:
            seen.setdefault(p.stem, p)
    return list(seen.values())


def parse_target(stem: str) -> str | None:
    for s in STYLES:
        for sep in ("__to__", "_to_"):
            if stem.endswith(f"{sep}{s}"):
                return s
    return None


def strip_target(stem: str) -> str:
    for s in STYLES:
        for sep in ("__to__", "_to_"):
            tok = f"{sep}{s}"
            if stem.endswith(tok):
                return stem[: -len(tok)]
    return stem


def build_sheet(files: list[Path], out: Path) -> bool:
    by_tgt: dict[str, list[Path]] = {s: [] for s in STYLES}
    for p in files:
        t = parse_target(p.stem)
        if t:
            by_tgt[t].append(p)
    if not any(by_tgt.values()):
        # no parseable target suffix: just lay out the first 20 images
        pick = files[: COLS * ROWS]
        W, H = CELL * COLS, CELL * ROWS + 16
        sheet = Image.new("RGB", (W, H), (255, 255, 255))
        for i, p in enumerate(pick):
            im = Image.open(p).convert("RGB").resize((CELL, CELL), Image.BICUBIC)
            sheet.paste(im, ((i % COLS) * CELL, 16 + (i // COLS) * CELL))
        out.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(out, "JPEG", quality=86, optimize=True)
        return True

    counts: dict[str, int] = {}
    for t, ps in by_tgt.items():
        for p in ps:
            k = strip_target(p.stem)
            counts[k] = counts.get(k, 0) + 1
    rows = [k for k, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:ROWS]]
    if not rows:
        return False

    W, H = CELL * COLS, CELL * ROWS + 16
    sheet = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for c, t in enumerate(STYLES):
        draw.text((c * CELL + 3, 3), t[:24], fill=(0, 0, 0))
    for r, key in enumerate(rows):
        for c, t in enumerate(STYLES):
            hit = next((p for p in by_tgt[t] if strip_target(p.stem) == key), None)
            if hit is None:
                continue
            im = Image.open(hit).convert("RGB").resize((CELL, CELL), Image.BICUBIC)
            sheet.paste(im, (c * CELL, 16 + r * CELL))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, "JPEG", quality=86, optimize=True)
    return True


def metrics_for(dataset: str, method: str) -> Path | None:
    for c in [
        REPO / "eval_outputs" / "d5_750_baselines" / method / "metrics.csv",
        REPO / "results" / dataset / f"{method}_metrics.csv",
    ]:
        if c.is_file():
            return c
    return None


def main(datasets: list[str]) -> None:
    inv = {}
    for ds in datasets:
        root = DATASETS[ds]
        if not root.is_dir():
            print(f"skip {ds}")
            continue
        entry = {}
        for m in METHODS:
            mdir = root / m
            if not mdir.is_dir():
                entry[m] = {"status": "missing_dir"}
                print(f"  {ds}/{m}: MISSING", flush=True)
                continue
            files = collect(mdir)
            if not files:
                entry[m] = {"status": "empty"}
                print(f"  {ds}/{m}: EMPTY", flush=True)
                continue
            jdir = STAGE / ds / "jpg" / m
            jdir.mkdir(parents=True, exist_ok=True)
            tot = 0
            for p in files:
                dst = jdir / (p.stem + ".jpg")
                if not dst.exists():
                    Image.open(p).convert("RGB").save(dst, "JPEG", quality=90, optimize=True)
                tot += dst.stat().st_size
            sheet_ok = build_sheet(files, STAGE / ds / "sheets" / f"{m}.jpg")
            mc = metrics_for(ds, m)
            if mc:
                (STAGE / ds / "metrics").mkdir(parents=True, exist_ok=True)
                shutil.copy2(mc, STAGE / ds / "metrics" / f"{m}.csv")
            entry[m] = {"status": "ok", "n": len(files), "jpg_MB": round(tot / 1048576, 1),
                        "sheet": sheet_ok, "metrics": bool(mc)}
            print(f"  {ds}/{m}: {len(files)} jpg, {tot/1048576:.1f} MB, sheet={sheet_ok}, metrics={bool(mc)}", flush=True)
        inv[ds] = entry
        # one archive per dataset
        tarp = STAGE / f"baseline_{ds}.tar.gz"
        if tarp.exists():
            tarp.unlink()
        subprocess.run(["tar", "-czf", str(tarp), "-C", str(STAGE / ds), "jpg", "sheets"], check=True)
        print(f"  archive {tarp.name}: {tarp.stat().st_size/1048576:.0f} MB", flush=True)
    (STAGE / "inventory.json").write_text(json.dumps(inv, indent=2, ensure_ascii=False), encoding="utf-8")
    print("inventory written")


if __name__ == "__main__":
    main(sys.argv[1:] or list(DATASETS))
