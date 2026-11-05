"""Calibrate Delta_style with reference points so the sign can be interpreted.

S(x, s) = max cos(DINO(x), DINO(ref)) over the first `--max-refs` refs of style s.
Reference points: identity (source image), VAE round-trip, perfect style (target
ref image), random other style (non-target ref), and the four arms.
The source image is excluded from its own style pool (leave-one-out) because it
is itself one of the first refs; the leaky variant is reported for contrast.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

import sys
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


def stats(deltas):
    n = len(deltas)
    mean = sum(deltas) / n
    var = sum((d - mean) ** 2 for d in deltas) / (n - 1)
    se = math.sqrt(var / n)
    return {"n": n, "mean": mean, "se": se, "frac_positive": sum(d > 0 for d in deltas) / n}


def paired(a, b):
    """paired a - b with standard error and t statistic."""
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    mean = sum(d) / n
    var = sum((x - mean) ** 2 for x in d) / (n - 1)
    se = math.sqrt(var / n) if var > 0 else 0.0
    t = mean / se if se > 0 else float("inf")
    return {"n": n, "mean": mean, "se": se, "t": t,
            "frac_positive": sum(x > 0 for x in d) / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", default="eval/p2_val75")
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vae-root", default="/root/calib/vae_rt")
    ap.add_argument("--cache-dir", default="/mnt/i/Latent_Style/eval_cache/hf")
    ap.add_argument("--max-refs", type=int, default=30)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-vae", action="store_true")
    ap.add_argument("--vae-dtype", choices=["fp16", "fp32"], default="fp16",
                    help="round-trip precision; fp16 matches the canonical TRT decoder and fits 12GB")
    args = ap.parse_args()
    dev = args.device

    model = load_dino("facebook/dinov2-small", dev, args.cache_dir, False)
    tdir = Path(args.test_dir)
    styles = sorted(p.name for p in tdir.iterdir() if p.is_dir())
    pools = {}
    for s in styles:
        files = sorted(p for p in (tdir / s).iterdir() if p.suffix.lower() in IMAGE_EXTS)
        pools[s] = files[:args.max_refs]
    ref_feat = {s: encode(pools[s], model, dev) for s in styles}

    rows = list(csv.DictReader(open(Path(args.arms[0].split("=", 1)[1]) / "metrics.csv")))
    srcs = [r["src_image"] for r in rows]
    tgt = [r["tgt_style"] for r in rows]
    src_style = [p.split("__")[0] for p in srcs]

    def src_name(p):
        return p

    def src_path(p):
        return tdir / p.split("__", 1)[0] / p

    def S_set(feats, src_style, tgt, loo: bool):
        """(S_target, S_source) per pair; loo drops the source image from its own pool."""
        s_t, s_s = [], []
        for f, ss, tt, src in zip(feats, src_style, tgt, srcs):
            name = src_name(src)
            excl = None
            if loo:
                for i, p in enumerate(pools[ss]):
                    if p.name == name:
                        excl = i
                        break
            sims_s = (f @ ref_feat[ss].T)
            if excl is not None:
                sims_s = torch.cat([sims_s[:excl], sims_s[excl + 1:]])
            s_s.append(float(sims_s.max()))
            s_t.append(float((f @ ref_feat[tt].T).max()))
        return s_t, s_s

    def delta(feats, loo: bool):
        s_t, s_s = S_set(feats, src_style, tgt, loo)
        return [a - b for a, b in zip(s_t, s_s)]

    out = {"style_pools": {s: [p.name for p in pools[s][:3]] for s in styles},
           "pools_count": {s: len(pools[s]) for s in styles},
           "arms": {}, "reference_points": {}}

    # identity: the source image itself, no model involved
    id_feats = encode([src_path(s) for s in srcs], model, dev)
    d_id = delta(id_feats, True)
    d_id_leaky = delta(id_feats, False)
    out["reference_points"]["identity"] = stats(d_id)
    out["reference_points"]["identity_with_selfleak"] = stats(d_id_leaky)
    # leakage proof
    leak_count = sum(1 for ss, src in zip(src_style, srcs)
                     if any(p.name == src_name(src) for p in pools[ss]))
    out["reference_points"]["sources_present_in_own_pool"] = f"{leak_count}/{len(srcs)}"

    # upper bound: the target style reference image itself
    up_feats = encode([pools[tt][0] for tt in tgt], model, dev)
    d_up = delta(up_feats, True)
    out["reference_points"]["target_ref_upper"] = stats(d_up)
    # lower bound: a reference image from a style that is neither source nor target
    other = []
    for ss, tt in zip(src_style, tgt):
        other.append(next(s for s in styles if s not in (ss, tt)))
    low_feats = encode([pools[o][0] for o in other], model, dev)
    d_low = delta(low_feats, True)
    out["reference_points"]["other_ref_lower"] = stats(d_low)

    # VAE round-trip on the sources (separates VAE loss from style transfer)
    if not args.skip_vae:
        try:
            from utils.inference import load_vae, encode_image
        except Exception:
            load_vae = encode_image = None
        if load_vae is not None:
            import numpy as np
            vae = load_vae(device=dev, model_id="ema", cache_dir=args.cache_dir,
                           compile_dtype=torch.float16 if args.vae_dtype == "fp16" else torch.float32)
            vr = Path(args.vae_root)
            vr.mkdir(parents=True, exist_ok=True)
            made = []
            uniq = sorted(set(srcs))
            for src in uniq:
                dst = vr / (Path(src).stem + ".png")
                if not dst.exists():
                    img = Image.open(src_path(src)).convert("RGB")
                    x = torch.from_numpy(np.array(img)).permute(2, 0, 1)[None] / 127.5 - 1.0
                    lat = encode_image(vae, x, dev)
                    with torch.no_grad():
                        dec = vae.decode(lat / vae.config.scaling_factor, return_dict=False)[0]
                    dec = (dec.clamp(-1, 1) + 1) * 127.5
                    Image.fromarray(dec[0].permute(1, 2, 0).byte().cpu().numpy()).save(dst)
            rt = {s: vr / (Path(s).stem + ".png") for s in uniq}
            vf = encode([rt[s] for s in srcs], model, dev)
            out["reference_points"]["vae_roundtrip"] = stats(delta(vf, True))
            del vae
            torch.cuda.empty_cache()

    id_d = d_id
    for spec in args.arms:
        tag, d = spec.split("=", 1)
        gen = [Path(d) / r["gen_image"] for r in list(csv.DictReader(open(Path(d) / "metrics.csv")))]
        gf = encode(gen, model, dev)
        d_arm = delta(gf, True)
        out["arms"][tag] = {
            "delta_loo": stats(d_arm),
            "delta_with_selfleak": stats(delta(gf, False)),
            "increment_vs_identity": paired(d_arm, id_d),
            "increment_vs_identity_leaky": paired(delta(gf, False), d_id_leaky),
        }
        print(tag, json.dumps(out["arms"][tag]["delta_loo"]),
              "inc:", json.dumps(out["arms"][tag]["increment_vs_identity"]))

    for k, v in out["reference_points"].items():
        print("REF", k, json.dumps(v))
    Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
