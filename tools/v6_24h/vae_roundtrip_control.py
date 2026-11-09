"""Diagnostic: does the VAE round-trip itself destroy style identity?

Motivation. Every arm in the 24h queue has target-style Top-1 around 0.03-0.05 while chance
is 0.20, yet dT is positive. The known real-image ceiling is ~0.98, so the CSD classifier is
fine on real PNGs. The open question is whether the loss of identifiability happens at the
decode step or at the transformation step.

This control runs encode -> decode with NO transformation and asks:
  Q1 what is the Top-1 of the decoded image against its OWN source style?
  Q2 how does that compare with the original PNG's Top-1 against its own style?
  Q3 what is the dT drift (should be ~0, since nothing was transformed)?

If Q1 is far below Q2, the VAE round-trip alone destroys style identity and every dT in the
project is being computed on a degraded baseline. If Q1 ~ Q2, the decode path is fine and the
identifiability loss comes from the transformation itself.

This is the "same-protocol identity baseline" that the V6 open questions listed as missing.

Usage
    python3 -m tools.v6_24h.vae_roundtrip_control
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
from PIL import Image

from tools.v6.train_eval_v6 import (STYLES, V5_ROOT, csd_model, csd_prototypes,
                                    decode_trt, load_trt, load_vae,
                                    _load_eval_image_tensor)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/v6_24h/vae_roundtrip.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = args.device

    tasks = json.loads((V5_ROOT / "v5_eval200_tasks.json").read_text(encoding="utf-8"))
    # one entry per unique source, carrying its own style
    src = {}
    for t in tasks:
        src.setdefault(t["src_stem"], {"path": t["src_path"], "style": t["src_style"]})
    keys = sorted(src)
    print(f"[roundtrip] {len(keys)} unique source images", flush=True)

    proto = csd_prototypes()
    csd, _, _ = csd_model.load_csd(device)
    vae = load_vae(device)
    trt = load_trt(vae, device)
    sf = float(vae.config.scaling_factor)
    # the VAE runs in fp16, so the input must match its dtype
    vae_dtype = next(vae.parameters()).dtype
    print(f"[roundtrip] vae dtype = {vae_dtype}", flush=True)

    top1_png, top1_rt, cos_png, cos_rt = [], [], [], []
    rows = []
    with torch.inference_mode():
        for s in range(0, len(keys), 8):
            chunk = keys[s:s + 8]
            x = torch.stack([_load_eval_image_tensor(Path(src[k]["path"]), size=512)
                             for k in chunk], 0).to(device)
            # original PNG through CSD (the real-image ceiling path)
            u8_png = x.mul(255.0).round().clamp(0, 255).to(torch.uint8).cpu()
            arr_png = u8_png.permute(0, 2, 3, 1).numpy()
            pils_png = [Image.fromarray(arr_png[i]) for i in range(len(chunk))]
            f_png = csd_model.feats_pils(csd, pils_png, batch=16, device=device)

            # encode -> decode, NO transformation
            z = vae.encode(x.to(vae_dtype)).latent_dist.mean * sf
            img = decode_trt(trt, vae, z, batch=8)
            u8_rt = img.mul(255.0).round().clamp(0, 255).to(torch.uint8).cpu()
            arr_rt = u8_rt.permute(0, 2, 3, 1).numpy()
            pils_rt = [Image.fromarray(arr_rt[i]) for i in range(len(chunk))]
            f_rt = csd_model.feats_pils(csd, pils_rt, batch=16, device=device)

            for i, k in enumerate(chunk):
                sid = STYLES.index(src[k]["style"])
                sc_p, sc_r = f_png[i] @ proto.T, f_rt[i] @ proto.T
                top1_png.append(int(sc_p.argmax() == sid))
                top1_rt.append(int(sc_r.argmax() == sid))
                cos_png.append(float(sc_p[sid]))
                cos_rt.append(float(sc_r[sid]))
                rows.append({"src_stem": k, "style": src[k]["style"],
                             "top1_png": int(sc_p.argmax() == sid),
                             "top1_rt": int(sc_r.argmax() == sid),
                             "cos_png": float(sc_p[sid]), "cos_rt": float(sc_r[sid]),
                             "cos_drift": float(sc_r[sid] - sc_p[sid])})
            print(f"[roundtrip] {min(s + 8, len(keys))}/{len(keys)}", flush=True)

    q1 = float(np.mean(top1_rt))
    q2 = float(np.mean(top1_png))
    drift = float(np.mean([r["cos_drift"] for r in rows]))
    per_style = {s: {"png": float(np.mean([r["top1_png"] for r in rows if r["style"] == s])),
                     "rt": float(np.mean([r["top1_rt"] for r in rows if r["style"] == s]))}
                 for s in STYLES}

    print(f"\n[roundtrip] Q2 original PNG Top-1 vs own style : {q2:.3f}")
    print(f"[roundtrip] Q1 VAE round-trip Top-1 vs own style: {q1:.3f}")
    print(f"[roundtrip] Q3 cosine drift through the round-trip: {drift:+.4f}")
    print(f"\n{'style':22s}{'png':>8s}{'roundtrip':>11s}{'loss':>8s}")
    for s in STYLES:
        d = per_style[s]
        print(f"{s:22s}{d['png']:8.3f}{d['rt']:11.3f}{d['rt'] - d['png']:+8.3f}")

    verdict = ("VAE round-trip preserves style identity"
               if q1 >= 0.9 * q2 else
               "VAE round-trip DESTROYS style identity -- every dT is computed on a "
               "degraded baseline")
    print(f"\nVERDICT: {verdict}")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "n_sources": len(keys), "top1_png": q2, "top1_roundtrip": q1,
        "cos_drift": drift, "per_style": per_style,
        "verdict": verdict, "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
