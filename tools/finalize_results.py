"""Finalize an evaluation dir: attach protocol/counts/reference-policy/backbone
metadata and a per-target-style breakdown to results.json.

The SOTA gate is judged on the 600 cross-style split with per-style reporting
(Hayao is historically the weakest style), so every eval must keep this table.

usage:
    python3 tools/finalize_results.py --eval-dir runs/routeC_eval/c_ref_s0 \
        [--reference-policy first_image_of_target_style_test_dir] \
        [--dino-backbone facebook/dinov2-small]
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

METRICS = ("dino_s", "clip_s", "dino_c", "lpips")


def _mean(values):
    return round(statistics.fmean(values), 4) if values else None


def _group(rows):
    out: dict[str, dict[str, list[float]]] = {}
    for r in rows:
        style = r.get("target_style_id", "?")
        bucket = out.setdefault(style, {m: [] for m in METRICS})
        for m in METRICS:
            v = r.get(m)
            if isinstance(v, (int, float)):
                bucket[m].append(float(v))
    return {style: {m: _mean(vs) for m, vs in vals.items()} for style, vals in sorted(out.items())}


def _counts(rows):
    out: dict[str, int] = {}
    for r in rows:
        style = r.get("target_style_id", "?")
        out[style] = out.get(style, 0) + 1
    return dict(sorted(out.items()))


def _mean_of(groups, metric):
    vals = [g[metric] for g in groups.values() if g.get(metric) is not None]
    return round(statistics.fmean(vals), 4) if vals else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--reference-policy", default="first_image_of_target_style_test_dir")
    ap.add_argument("--dino-backbone", default="facebook/dinov2-small")
    ap.add_argument("--clip-backbone", default="openai/clip-vit-large-patch14")
    ap.add_argument("--extra", default="", help="free-form note stored in protocol")
    args = ap.parse_args()

    path = Path(args.eval_dir) / "results.json"
    payload = json.loads(path.read_text())
    rows = payload.get("results", [])
    cross = [r for r in rows if not r.get("same_style")]
    same = [r for r in rows if r.get("same_style")]

    per_cross = _group(cross)
    per_same = _group(same)
    payload["protocol"] = {
        "split": {"cross_style": len(cross), "same_style": len(same), "total": len(rows)},
        "reference_policy": args.reference_policy,
        "metrics": {
            "dino_s": "max cosine CLS(gen), CLS(target-style reference)",
            "dino_c": "cosine CLS(gen), CLS(source)",
            "clip_s": "CLIP full-image style similarity (ViT-L/14)",
            "lpips": "LPIPS(gen, source), lower is better",
        },
        "note": args.extra,
    }
    payload["counts"] = {
        "total": len(rows),
        "cross_style": len(cross),
        "same_style": len(same),
        "per_style": {"cross_style": _counts(cross), "same_style": _counts(same)},
    }
    payload["backbone"] = {"dino": args.dino_backbone, "clip": args.clip_backbone}
    payload["per_style"] = {"cross_style": per_cross, "same_style": per_same,
                            "cross_style_mean": {m: _mean_of(per_cross, m) for m in METRICS},
                            "same_style_mean": {m: _mean_of(per_same, m) for m in METRICS}}

    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)

    print(f"[per-style] {path} cross_style n={len(cross)}")
    for style, vals in per_cross.items():
        print(f"  {style:22s} dino_s={vals['dino_s']} clip_s={vals['clip_s']} "
              f"dino_c={vals['dino_c']} lpips={vals['lpips']}")
    print(f"  {'MEAN':22s} dino_s={payload['per_style']['cross_style_mean']['dino_s']} "
          f"clip_s={payload['per_style']['cross_style_mean']['clip_s']} "
          f"dino_c={payload['per_style']['cross_style_mean']['dino_c']} "
          f"lpips={payload['per_style']['cross_style_mean']['lpips']}")


if __name__ == "__main__":
    main()
