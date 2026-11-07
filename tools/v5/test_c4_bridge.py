"""Exercise the C4 legacy-eval compatibility bridge on drvfs (tiny dummy cell)."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import paths
from PIL import Image


def main():
    render_root = paths.asset_root() / "eval_assets" / "v5_route_c"
    cell = "_bridge_test"
    out_dir = render_root / cell
    img_dir = out_dir / "images"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    img_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (7, 8, 9)).save(img_dir / "dummy.png")
    (out_dir / "metrics.csv").write_text("src_style,tgt_style,src_image,gen_image\n", encoding="utf-8")
    (out_dir / "dino_metrics.csv").write_text("src_style,tgt_style,src_image,gen_image\n", encoding="utf-8")

    legacy_root = paths.asset_root() / "pseudo_gt_24h" / "eval_render"
    legacy_root.mkdir(parents=True, exist_ok=True)
    legacy_cell = legacy_root / cell
    if legacy_cell.exists() or legacy_cell.is_symlink():
        if legacy_cell.is_symlink():
            legacy_cell.unlink()
        else:
            shutil.rmtree(legacy_cell)
    linked = False
    try:
        legacy_cell.symlink_to(out_dir, target_is_directory=True)
        linked = True
    except OSError:
        legacy_cell.mkdir(parents=True, exist_ok=True)
        (legacy_cell / "images").mkdir(exist_ok=True)
        for src_file in img_dir.glob("*.png"):
            dst = legacy_cell / "images" / src_file.name
            try:
                import os
                os.link(src_file, dst)
            except OSError:
                shutil.copy2(src_file, dst)
        for fn in ("metrics.csv", "dino_metrics.csv"):
            shutil.copy2(out_dir / fn, legacy_cell / fn)
    ok = (legacy_cell / "images" / "dummy.png").exists() and (legacy_cell / "metrics.csv").exists()
    print(f"BRIDGE_OK={ok} mode={'symlink' if linked else 'copy/hardlink'} legacy={legacy_cell}")
    # cleanup
    for p in (legacy_cell, out_dir):
        if p.is_symlink():
            p.unlink()
        elif p.exists():
            shutil.rmtree(p)
    print("CLEANED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
