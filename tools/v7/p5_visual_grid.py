"""V7 P5 visual case grid: 10 fixed sources (2/style) x 5 target Style-IDs.

Mechanism diagnosis only (V7_EVAL_PROTOCOL.md section 8) — never a numeric gate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from tools.v7.v7_common import (
    B4_CKPT, ROOT, STYLES, load_c3unet, load_canon, model_output,
)
from tools.v5.route_c import decode_trt, load_trt, load_vae

SRC_CACHE = ROOT / "runs" / "v6_r2" / "cache" / "tasks750_source_features.pt"


def main() -> int:
    ap = argparse.ArgumentParser(description="V7 visual grid")
    ap.add_argument("--ckpt", default=str(B4_CKPT))
    ap.add_argument("--start-mode", default="direct", choices=["direct", "relative"])
    ap.add_argument("--out", default="runs/v7/visual_grid.png")
    ap.add_argument("--per-style", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = args.device

    tasks = json.loads((ROOT / "eval_protocol" / "tasks_750.json").read_text(encoding="utf-8"))["tasks"]
    seen, sel = set(), []
    for t in tasks:
        if t["source_stem"] in seen:
            continue
        seen.add(t["source_stem"])
        n_style = sum(1 for x in sel if x["source_style"] == t["source_style"])
        if n_style < args.per_style:
            sel.append(t)
        if len(sel) >= args.per_style * len(STYLES):
            break

    src_cache = torch.load(SRC_CACHE, map_location="cpu", weights_only=False)
    model = load_c3unet(Path(args.ckpt), device)
    canon = load_canon(device)
    vae = load_vae(device)
    trt = load_trt(vae, device)

    cell, label_h = 160, 22
    cols = 6  # source + 5 targets
    rows = len(sel)
    grid = Image.new("RGB", (cols * cell, rows * (cell + label_h)), "white")
    draw = ImageDraw.Draw(grid)

    with torch.inference_mode():
        for r, t in enumerate(sel):
            z = src_cache["z_src"][t["source_stem"]].to(device)
            raw = src_cache["raw_rgb"][t["source_stem"]]
            sid = torch.full((5,), int(t["source_style_id"]), device=device)
            y = torch.arange(5, device=device)
            z_out = model_output(model, z.expand(5, -1, -1, -1).contiguous(),
                                 canon, "c1_gentle", sid, y, args.start_mode)
            img_out = decode_trt(trt, vae, z_out, batch=5).clamp(0, 1).cpu()
            u8 = img_out.mul(255).to(torch.uint8)
            imgs = [Image.fromarray(raw.permute(1, 2, 0).mul(255).to(torch.uint8).numpy())]
            imgs += [Image.fromarray(u8[j].permute(1, 2, 0).numpy()) for j in range(5)]
            top = r * (cell + label_h)
            draw.text((4, top + 4),
                      f"{t['source_style']}/{t['source_stem'][:28]}", fill="black")
            for c, im in enumerate(imgs):
                grid.paste(im.resize((cell, cell)), (c * cell, top + label_h))
            if r == 0:
                for c, name in enumerate(["SOURCE"] + STYLES):
                    draw.text((c * cell + 4, rows * (cell + label_h) - 18),
                              name[:18], fill="black")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out)
    print(f"[v7grid] wrote {out} ({rows}x{cols})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
