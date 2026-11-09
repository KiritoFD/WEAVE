"""How much does the target style condition actually move the output?

The visual grid showed five different target conditions producing near-identical images. That is a
different question from "is the output the target style": it asks whether the style condition has any
leverage at all on the generated content.

Measured per source, over the five target styles:
  d_cross  mean pairwise latent distance between outputs for DIFFERENT targets
  d_out    distance from the source to the output (the transformation magnitude)
  ratio    d_cross / d_out

  ratio near 1    the target condition moves the output as much as the transformation itself
  ratio near 0    the five targets produce essentially the same image; the condition is inert

This is independent of CSD and of Top-1, so it is not exposed to the metric critiques raised
elsewhere.

Usage
    python3 -m tools.v6_24h.style_condition_efficacy --ckpt <path>
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

from tools.v6.canonicalize import ContentCanonicalizer
from tools.v6.models_v6 import C3UNet
from tools.v6.train_eval_v6 import STYLES, V5_ROOT, V6_ROOT


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/v6/N7_dim24/N7_dim24.pt")
    ap.add_argument("--n-sources", type=int, default=40)
    ap.add_argument("--out", default="runs/v6_24h/style_condition_efficacy.json")
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
    canon = ContentCanonicalizer(torch.load(V6_ROOT / "cache" / "canonicalizer_stats.pt",
                                           map_location="cpu", weights_only=False)).to(dev)
    cmode = a.get("canon_mode", "c1_gentle")

    lat = torch.load(V5_ROOT / "cache" / "eval_src_latents.pt", map_location="cpu")
    keys = sorted(lat)[: args.n_sources]
    z = torch.stack([lat[k] for k in keys]).to(dev).float()

    with torch.inference_mode():
        c = canon(z, mode=cmode, augment=False)
        outs = []
        for sid in range(5):
            y = torch.full((z.shape[0],), sid, device=dev, dtype=torch.long)
            outs.append(model(c, 0.0, c, y))          # (N,4,64,64)
        outs = torch.stack(outs, 0)                    # (5,N,4,64,64)

    # flatten per source
    flat = outs.permute(1, 0, 2, 3, 4).reshape(z.shape[0], 5, -1)
    d_out = (flat - z.reshape(z.shape[0], 1, -1)).norm(dim=2).mean(dim=1)   # (N,)
    pairs = []
    for i in range(5):
        for j in range(i + 1, 5):
            pairs.append((flat[:, i] - flat[:, j]).norm(dim=1))
    d_cross = torch.stack(pairs, 0).mean(dim=0)        # (N,)
    ratio = d_cross / d_out.clamp_min(1e-8)

    res = {"ckpt": args.ckpt, "n_sources": len(keys),
           "d_out_mean": float(d_out.mean()), "d_cross_mean": float(d_cross.mean()),
           "ratio_mean": float(ratio.mean()),
           "ratio_p10": float(ratio.quantile(0.1)),
           "ratio_p90": float(ratio.quantile(0.9))}
    print(f"[efficacy] ckpt={args.ckpt}  n={res['n_sources']}")
    print(f"  transformation magnitude  d_out   = {res['d_out_mean']:.4f}")
    print(f"  cross-target difference   d_cross = {res['d_cross_mean']:.4f}")
    print(f"  ratio d_cross / d_out     = {res['ratio_mean']:.4f} "
          f"(p10 {res['ratio_p10']:.4f}, p90 {res['ratio_p90']:.4f})")
    if res["ratio_mean"] < 0.2:
        print("  -> the five target conditions produce nearly the SAME image; the style")
        print("     condition has almost no leverage on the output content.")
    elif res["ratio_mean"] < 0.5:
        print("  -> the style condition moves the output noticeably but far less than the")
        print("     transformation itself.")
    else:
        print("  -> the style condition is a major driver of the output.")

    p = ROOT / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
