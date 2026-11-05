"""Delta_style = S(gen, s_target) - S(gen, s_source) on a fixed pair list.

S is the DINOv2-small CLS cosine to the canonical style reference image
(first image, sorted, of each style dir in the test dir) — the same encoder,
preprocessing (512 -> 224 bicubic centre crop) and reference policy as the
canonical eval. Absolute values are only comparable within this pair list.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.compute_dino_metrics import DINO_TRANSFORM, load_dino  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def encode(paths, model, device, batch=16):
    out = []
    with torch.no_grad():
        for i in range(0, len(paths), batch):
            imgs = torch.stack([DINO_TRANSFORM(Image.open(p).convert("RGB"))
                                for p in paths[i:i + batch]]).to(device)
            cls = model(imgs).last_hidden_state[:, 0]
            out.append(F.normalize(cls.float(), dim=-1).cpu())
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", default="eval/p2_val75")
    ap.add_argument("--arms", nargs="+", required=True, help="arm=outdir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache-dir", default="/mnt/i/Latent_Style/eval_cache/hf")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-refs", type=int, default=30,
                    help="refs per style; must match eval_fast --dino-max-refs")
    args = ap.parse_args()

    device = args.device
    model = load_dino("facebook/dinov2-small", device, args.cache_dir, False)
    test_dir = Path(args.test_dir)
    styles = sorted(p.name for p in test_dir.iterdir() if p.is_dir())
    refs = {}
    for s in styles:
        files = sorted(p for p in (test_dir / s).iterdir() if p.suffix.lower() in IMAGE_EXTS)
        refs[s] = files[:args.max_refs]
    proto = {}
    for s in styles:
        proto[s] = encode(refs[s], model, device)

    summary = {"pairs_file": "eval/p2_75pairs_manifest.json", "max_refs": args.max_refs,
               "style_ref_first": {k: str(v[0]) for k, v in refs.items()}, "arms": {}}
    for spec in args.arms:
        tag, outdir = spec.split("=", 1)
        rows = list(csv.DictReader(open(Path(outdir) / "metrics.csv")))
        gen_paths = [Path(outdir) / r["gen_image"] for r in rows]
        feats = encode(gen_paths, model, device)
        deltas, s_tgt_all, s_src_all, cross = [], [], [], []
        for r, f in zip(rows, feats):
            src_style = r["src_image"].split("__")[0]
            tgt_style = r["tgt_style"]
            s_t = float((f @ proto[tgt_style].T).max())
            s_s = float((f @ proto[src_style].T).max())
            d = s_t - s_s
            deltas.append(d); s_tgt_all.append(s_t); s_src_all.append(s_s)
            if src_style != tgt_style:
                cross.append(d)
        res = {
            "n": len(deltas),
            "delta_mean": sum(deltas) / len(deltas),
            "delta_median": sorted(deltas)[len(deltas) // 2],
            "frac_positive": sum(d > 0 for d in deltas) / len(deltas),
            "s_target_mean": sum(s_tgt_all) / len(deltas),
            "s_source_mean": sum(s_src_all) / len(deltas),
            "cross_n": len(cross),
            "cross_delta_mean": sum(cross) / len(cross) if cross else 0.0,
            "cross_frac_positive": sum(d > 0 for d in cross) / len(cross) if cross else 0.0,
        }
        summary["arms"][tag] = res
        print(tag, json.dumps(res))

    Path(args.out).write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
