"""Plan section 8 - visual grid, and the input a human blind rating needs.

The plan requires visualisations, and the remaining diagnostic that cannot be automated is human
blind rating (section 1: "独立风格评价/人工盲图"). Both need the same thing: decoded images laid
out so a person can judge them.

Layout: one PNG per source image.
  row 0  the source itself (real, at 512)
  row 1  the VAE round-trip of the source (what the operator starts from)
  row 2..6  the operator's output for each of the five target styles
Columns are annotated with the style name. A companion JSON records the task ids and the
per-image metrics so a rating sheet can be filled in without ambiguity.

Images are written at native 512 so style is judgeable; no upscaling, no sharpening.

Usage
    python3 -m tools.v6_24h.make_grid --ckpt runs/v6/N7_dim24/N7_dim24.pt --n-sources 4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import Image, ImageDraw

from tools.v6.canonicalize import ContentCanonicalizer
from tools.v6.models_v6 import C3UNet
from tools.v6.train_eval_v6 import (STYLES, V5_ROOT, V6_ROOT, csd_model, csd_prototypes,
                                    decode_trt, load_trt, load_vae,
                                    _load_eval_image_tensor)

SH = {"Early_Renaissance": "ER", "Impressionism": "Imp", "Minimalism": "Min",
      "Rococo": "Roc", "Ukiyo_e": "Uki"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/v6/N7_dim24/N7_dim24.pt")
    ap.add_argument("--n-sources", type=int, default=4)
    ap.add_argument("--out", default="runs/v6_24h/grid")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = args.device

    ck = torch.load(ROOT / args.ckpt, map_location="cpu", weights_only=False)
    a = ck.get("args", {})
    model = C3UNet(base_dim=a.get("base_dim", 48), num_styles=5,
                   mode=a.get("mode", "direct"),
                   no_style=bool(a.get("no_style", False)),
                   content_mode=a.get("content_mode", "full"),
                   inject_64=not a.get("no_inject_64", False),
                   inject_32=not a.get("no_inject_32", False),
                   inject_16=not a.get("no_inject_16", False),
                   style_keys=bool(a.get("style_keys", False))).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    canon_mode = a.get("canon_mode", "c1_gentle")
    canon = ContentCanonicalizer(torch.load(V6_ROOT / "cache" / "canonicalizer_stats.pt",
                                            map_location="cpu", weights_only=False)).to(dev)

    tasks = json.loads((V5_ROOT / "v5_eval200_tasks.json").read_text(encoding="utf-8"))
    # One source per distinct SOURCE STYLE. Taking the first N distinct sources gave four
    # Early_Renaissance images, which made the grid useless for judging whether different sources
    # behave differently. Spread across styles instead.
    by_style, seen = {}, set()
    for t in tasks:
        if t["src_stem"] in seen:
            continue
        seen.add(t["src_stem"])
        by_style.setdefault(t["src_style"], []).append(t)
    picked = []
    for s in STYLES:
        if by_style.get(s):
            picked.append(by_style[s][0])
    picked = picked[: args.n_sources] if args.n_sources < len(picked) else picked
    print(f"[grid] sources: {[t['src_style'] for t in picked]}", flush=True)

    lat = torch.load(V5_ROOT / "cache" / "eval_src_latents.pt", map_location="cpu")
    vae = load_vae(dev)
    trt = load_trt(vae, dev)
    sf = float(vae.config.scaling_factor)
    proto = csd_prototypes()
    csd, _, _ = csd_model.load_csd(dev)

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    manifest = []

    for ti, t in enumerate(picked):
        z = lat[t["src_stem"]].unsqueeze(0).to(dev).float()
        with torch.inference_mode():
            img_rt = decode_trt(trt, vae, z, batch=1)
            u8_rt = img_rt.mul(255.0).round().clamp(0, 255).to(torch.uint8)[0]
            gen, gen01 = [], []
            for sid in range(5):
                y = torch.tensor([sid], device=dev, dtype=torch.long)
                c = canon(z, mode=canon_mode, augment=False)
                zg = model(c, 0.0, c, y)
                im = decode_trt(trt, vae, zg, batch=1)
                u8 = im.mul(255.0).round().clamp(0, 255).to(torch.uint8)[0]
                gen.append(u8)
                gen01.append(u8.float().div(255.0))
            # CSD scores for the generated row, so the sheet carries numbers too
            pils = [Image.fromarray(g.permute(1, 2, 0).cpu().numpy()) for g in gen]
            F = csd_model.feats_pils(csd, pils, batch=8, device=dev)

        # assemble: row 0 = source PNG, row 1 = round-trip, rows 2..6 = the five targets
        src_png = _load_eval_image_tensor(Path(t["src_path"]), size=512)
        src_u8 = (src_png.mul(255.0).round().clamp(0, 255)
                  .to(torch.uint8).permute(1, 2, 0).cpu().numpy())
        rt_u8 = u8_rt.permute(1, 2, 0).cpu().numpy()
        H, W = src_u8.shape[:2]
        pad, label_h = 6, 26
        cols = 1 + 5
        canvas = Image.new("RGB", (cols * W + (cols + 1) * pad,
                                   (2 + 5) * H + (2 + 5 + 1) * pad + label_h), "white")
        dr = ImageDraw.Draw(canvas)
        headers = ["SOURCE", "RT"] + [SH[s] for s in STYLES]
        for c, h in enumerate(headers):
            dr.text((pad + c * (W + pad) + 4, 6), h, fill="black")

        rows = [src_u8, rt_u8] + [g.permute(1, 2, 0).cpu().numpy() for g in gen]
        for r, arr in enumerate(rows):
            y0 = label_h + pad + r * (H + pad)
            canvas.paste(Image.fromarray(arr), (pad, y0))
            if r >= 2:
                # put the column label on the left of each target row as well
                dr.text((pad + 4, y0 + 4), f"{SH[STYLES[r - 2]]}", fill="yellow")
        fn = out / f"grid_{ti}_{t['src_style']}.png"
        canvas.save(fn)

        row = {"task_id": t["task_id"], "src_stem": t["src_stem"],
               "src_style": t["src_style"], "file": fn.name,
               "scores": {}}
        for sid, s in enumerate(STYLES):
            sc = F[sid] @ proto.T
            row["scores"][s] = {"target_cos": float(sc[sid]),
                                "argmax": STYLES[int(sc.argmax())],
                                "margin": float(sc[sid] - np.delete(sc, sid).max())}
        manifest.append(row)
        print(f"[grid] {fn.name}  src={t['src_style']}", flush=True)

    (out / "manifest.json").write_text(json.dumps(
        {"ckpt": args.ckpt, "canon_mode": canon_mode, "tasks": manifest},
        ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[grid] wrote {len(manifest)} grids + manifest.json to {out}")
    print("[grid] layout: row0 SOURCE, row1 VAE round-trip, rows2-6 the five target styles")
    print("[grid] a rating sheet can be filled from manifest.json, which also carries the")
    print("       per-image target cosine, argmax style and margin.")


if __name__ == "__main__":
    main()
