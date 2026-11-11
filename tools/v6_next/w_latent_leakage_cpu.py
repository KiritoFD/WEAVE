"""CPU-only early audit: does c1_gentle still leak source style into the latent?

This is the no-GPU half of W-next P0. It uses the frozen R1 critic and the
existing 150-source feature cache to compare raw posterior-mean latents against
c1_gentle canonicalized latents. Generation/causal decoding is intentionally not
run here so it cannot disturb a GPU training pipeline.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.v6.canonicalize import ContentCanonicalizer  # noqa: E402
from tools.v6_next.probe_style import LatentStyleNet  # noqa: E402

STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
SRC_CACHE = ROOT / "runs/v6_r2/cache/tasks750_source_features.pt"
CRITIC = ROOT / "runs/v6_next/probes/r1_r1.pt"
CANON_STATS = ROOT / "runs/v6/cache/canonicalizer_stats.pt"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", type=int, default=150)
    ap.add_argument("--out", default="runs/v6_w/p0_latent_leakage_cpu.json")
    args = ap.parse_args()

    cache = torch.load(SRC_CACHE, map_location="cpu", weights_only=False)
    critic_ck = torch.load(CRITIC, map_location="cpu", weights_only=False)
    critic = LatentStyleNet(width=critic_ck.get("width", 64)).eval()
    critic.load_state_dict(critic_ck["model"])
    for p in critic.parameters():
        p.requires_grad_(False)
    canon = ContentCanonicalizer(torch.load(CANON_STATS, map_location="cpu",
                                            weights_only=False)).eval()

    stems = sorted(cache["z_src"].keys())[: args.sources]
    rows = []
    with torch.inference_mode():
        for k in stems:
            z = cache["z_src"][k].float()
            c = canon(z, mode="c1_gentle", augment=False)
            # k is a source_stem like "Early_Renaissance__painter_title", not a directory path.
            src_style = k.split("__", 1)[0]
            if src_style not in STYLES:
                raise ValueError(f"cannot parse style from stem: {k}")
            sid = STYLES.index(src_style)

            def eval_one(q: torch.Tensor) -> dict:
                pred = int(q[0].argmax())
                sorted_q = q[0].sort(descending=True).values
                return {
                    "pred": pred,
                    "pred_style": STYLES[pred],
                    "source_top1": int(pred == sid),
                    "source_margin": float(q[0, sid] - sorted_q[1]) if sid >= 0 else float("nan"),
                    "top2_gap": float(sorted_q[0] - sorted_q[1]),
                }

            rows.append({
                "stem": k,
                "src_style": src_style,
                "raw": eval_one(critic(z.float())),
                "c1_gentle": eval_one(critic(c.float())),
            })

    def summary(field: str) -> dict:
        raw_acc = float(np.mean([r["raw"]["source_top1"] for r in rows]))
        canon_acc = float(np.mean([r[field]["source_top1"] for r in rows]))
        raw_margin = float(np.nanmean([r["raw"]["source_margin"] for r in rows]))
        canon_margin = float(np.nanmean([r[field]["source_margin"] for r in rows]))
        return {
            "n": len(rows),
            "raw_source_top1": raw_acc,
            "c1_gentle_source_top1": canon_acc,
            "raw_source_margin": raw_margin,
            "c1_gentle_source_margin": canon_margin,
            "top1_drop": raw_acc - canon_acc,
            "per_style": {
                s: {
                    "n": sum(1 for r in rows if r["src_style"] == s),
                    "raw_top1": float(np.mean([r["raw"]["source_top1"] for r in rows if r["src_style"] == s])),
                    "c1_top1": float(np.mean([r[field]["source_top1"] for r in rows if r["src_style"] == s])),
                } for s in STYLES
            },
        }

    result = {
        "n": len(rows),
        "raw": summary("raw"),
        "c1_gentle": summary("c1_gentle"),
        "rows": rows,
    }
    out = ROOT / args.out
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("=" * 82)
    print("W-next P0 CPU audit: source-style information remaining in content latents")
    print("=" * 82)
    print(f"sources: {len(rows)}")
    print(f"raw latent source-style Top-1        : {result['raw']['raw_source_top1']*100:.1f}%")
    print(f"c1_gentle latent source-style Top-1  : {result['c1_gentle']['c1_gentle_source_top1']*100:.1f}%")
    print(f"raw latent source-style margin       : {result['raw']['raw_source_margin']:+.4f}")
    print(f"c1_gentle latent source-style margin : {result['c1_gentle']['c1_gentle_source_margin']:+.4f}")
    print("\nPer-style source-style Top-1 (raw -> c1_gentle):")
    for s in STYLES:
        p = result["c1_gentle"]["per_style"][s]
        raw_p = result["raw"]["per_style"][s]
        print(f"  {s:20s} {raw_p['raw_top1']*100:5.1f}% -> {p['c1_top1']*100:5.1f}%")
    print("=" * 82)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
