"""Prepare baseline assets for download: JPEG sets, contact sheets, metrics.

Run on 3060. Writes into /mnt/i/Latent_Style/WEAVE/_baseline_stage/<dataset>/:
  jpg/<method>/<name>.jpg      full 750-image sets re-encoded at q=90
  sheets/<method>.jpg          5 targets x 4 sources contact sheet
  metrics/<method>.csv         per-task metrics when available
  manifest.json                inventory + sizes
"""
from __future__ import annotations

import io
import json
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageDraw

REPO = Path("/mnt/i/Latent_Style/WEAVE")
STAGE = REPO / "_baseline_stage"
STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
DATASETS = {
    "D5-512": REPO / "results" / "D5-512",
    "P2A-256": REPO / "results" / "P256",
    "R5-WikiArt": REPO / "results" / "R5-WikiArt",
}
# methods we do NOT want (reference targets / non-methods / duplicates)
SKIP = {"tgt", "identity", "_pp_visual_samples", "weave_eval_baseline", "weave_yfix_eval",
        "cut_raw", "sdturbo_r5", "adain_stytr2", "wct_stytr2", "idt", "latent_wct",
        "weave_pp_affine_s05", "weave_pp_affine_s10", "weave_pp_wct_s05", "weave_pp_wct_s10"}
CELL = 160
GRID_COLS, GRID_ROWS = 5, 4


def parse_target(name: str) -> str | None:
    for s in STYLES:
        for sep in ("__to__", "_to_"):
            if name.endswith(f"{sep}{s}"):
                return s
    return None


def build_sheet(files: list[Path], out: Path) -> bool:
    """5 columns = target styles, 4 rows = 4 distinct sources."""
    by_tgt: dict[str, list[Path]] = {s: [] for s in STYLES}
    for p in files:
        t = parse_target(p.stem)
        if t:
            by_tgt[t].append(p)
    # choose 4 sources that appear in the most target columns
    counts: dict[str, int] = {}
    for t, ps in by_tgt.items():
        for p in ps[:40]:
            stem = p.stem
            for s in STYLES:
                for sep in ("__to__", "_to_"):
                    tok = f"{sep}{s}"
                    if stem.endswith(tok):
                        key = stem[: -len(tok)]
                        counts[key] = counts.get(key, 0) + 1
                        break
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])[:GRID_ROWS]
    if not ranked:
        return False
    rows = [k for k, _ in ranked]

    W, H = CELL * GRID_COLS, CELL * GRID_ROWS + 18
    sheet = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for c, t in enumerate(STYLES):
        draw.text((c * CELL + 4, 4), t[:22], fill=(0, 0, 0))
    for r, src_key in enumerate(rows):
        for c, t in enumerate(STYLES):
            hit = None
            for p in by_tgt[t]:
                if p.stem.startswith(src_key):
                    hit = p
                    break
            if hit is None:
                continue
            im = Image.open(hit).convert("RGB").resize((CELL, CELL), Image.BICUBIC)
            sheet.paste(im, (c * CELL, 18 + r * CELL))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, "JPEG", quality=88, optimize=True)
    return True


def metrics_for(dataset: str, method: str) -> Path | None:
    cands = [
        REPO / "eval_outputs" / "d5_750_baselines" / method / "metrics.csv",
        REPO / "results" / dataset / f"{method}_metrics.csv",
        REPO / "results" / dataset / method / "metrics.csv",
    ]
    for c in cands:
        if c.is_file():
            return c
    return None


def main(datasets: list[str]) -> None:
    inventory = {}
    for ds in datasets:
        root = DATASETS[ds]
        if not root.exists():
            print(f"skip missing {ds}")
            continue
        entry = {}
        for mdir in sorted(p for p in root.iterdir() if p.is_dir()):
            m = mdir.name
            if m in SKIP:
                continue
            files = sorted([p for p in mdir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
            if not files:
                continue
            # 1. JPEG set
            jdir = STAGE / ds / "jpg" / m
            jdir.mkdir(parents=True, exist_ok=True)
            n_jpg = 0
            tot = 0
            for p in files:
                dst = jdir / (p.stem + ".jpg")
                if not dst.exists():
                    im = Image.open(p).convert("RGB")
                    im.save(dst, "JPEG", quality=90, optimize=True)
                n_jpg += 1
                tot += dst.stat().st_size
            # 2. contact sheet
            sheet_ok = build_sheet(files, STAGE / ds / "sheets" / f"{m}.jpg")
            # 3. metrics
            mc = metrics_for(ds, m)
            if mc:
                (STAGE / ds / "metrics").mkdir(parents=True, exist_ok=True)
                shutil.copy2(mc, STAGE / ds / "metrics" / f"{m}.csv")
            entry[m] = {"n_src": len(files), "n_jpg": n_jpg,
                        "jpg_MB": round(tot / 1048576, 1), "sheet": sheet_ok,
                        "metrics": mc.name if mc else None}
            print(f"  {ds}/{m}: {n_jpg} jpg, {tot/1048576:.1f} MB, sheet={sheet_ok}, metrics={bool(mc)}", flush=True)
        inventory[ds] = entry
    (STAGE).mkdir(parents=True, exist_ok=True)
    (STAGE / "inventory.json").write_text(json.dumps(inventory, indent=2, ensure_ascii=False), encoding="utf-8")
    print("inventory written")


if __name__ == "__main__":
    main(sys.argv[1:] or list(DATASETS))
