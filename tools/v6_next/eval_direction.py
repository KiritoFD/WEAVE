"""P0.3: geometric direction diagnosis, replacing the pseudo "direction purity".

The previous report used |dT|/|dS|, a ratio of two SCALAR prototype-similarity changes, and
described it as "only 30% of the displacement points at the target, 70% is wrong". That inference
does not follow: a ratio of scalar similarities is not a projection and carries no directional
information.

This measures the real thing. With frozen CSD embeddings f (unit-norm), per image:

    df   = f(D(z_out)) - f(D(z_src))          the actual feature displacement
    d    = normalize(mu_t - mu_s)              the source->target prototype direction
    cos_align = <df, d> / (|df| |d|)           TRUE geometric alignment, in [-1, 1]

plus the quantities that separate the three failure modes the plan names:

    source retention   argmax(f_out) == s, and <f_out, mu_s>
    target acquisition argmax(f_out) == t, and <f_out, mu_t>
    near-but-not-crossed  argmax != t but m_target > -delta

and
    m_target = <f_out, mu_t> - max_{k != t} <f_out, mu_k>
    m_source = <f_out, mu_s> - max_{k != s} <f_out, mu_k>
    max_cos  = max_k <f_out, mu_k>
    knn_t    = mean cosine distance from f_out to its K nearest REAL target-style features

The kNN term needs individual real features, not just prototypes, so they are computed once from
the training pool and cached.

Usage
    python3 -m tools.v6_next.eval_direction --ckpt runs/v6/L11_k1024_dim16/L11_k1024_dim16.pt
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

from tools.v6.canonicalize import ContentCanonicalizer
from tools.v6.models_v6 import C3UNet
from tools.v6.train_eval_v6 import (STYLES, V5_ROOT, V6_ROOT, csd_model, csd_prototypes,
                                    decode_trt, load_trt, load_vae, _load_eval_image_tensor)

CACHE = Path("/mnt/i/Latent_Style/pseudo_gt_24h/orig_mean_fp16")
FILES = ["00_Early_Renaissance.pt", "01_Impressionism.pt", "02_Minimalism.pt",
         "03_Rococo.pt", "04_Ukiyo_e.pt"]
OUT = ROOT / "runs/v6_next"


def real_target_features(csd, vae, trt, dev, per_style: int = 50) -> dict:
    """Individual frozen CSD features of REAL target-style images, for kNN.

    Uses training-pool latents (the same source the prototypes come from) decoded through the
    frozen VAE, so the feature space matches the evaluation pipeline exactly.
    """
    cache = OUT / "probes" / f"real_csd_features_k{per_style}.pt"
    if cache.exists():
        return torch.load(cache, map_location="cpu", weights_only=False)
    cache.parent.mkdir(parents=True, exist_ok=True)
    feats = {}
    with torch.inference_mode():
        for sid, (s, f) in enumerate(zip(STYLES, FILES)):
            d = torch.load(CACHE / f, map_location="cpu", weights_only=False)
            lat = d["latents"].float()[100:100 + per_style].to(dev)
            img = decode_trt(trt, vae, lat, batch=8)
            u8 = img.mul(255.0).round().clamp(0, 255).to(torch.uint8).cpu().permute(0, 2, 3, 1).numpy()
            pils = [Image.fromarray(u8[i]) for i in range(len(u8))]
            F = csd_model.feats_pils(csd, pils, batch=16, device=dev)
            F = F / np.linalg.norm(F, axis=1, keepdims=True)
            feats[s] = torch.from_numpy(F.astype(np.float32))
            print(f"[dir] real features {s:20s} {F.shape}", flush=True)
    torch.save(feats, cache)
    return feats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/v6/L11_k1024_dim16/L11_k1024_dim16.pt")
    ap.add_argument("--tag", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--knn", type=int, default=5)
    ap.add_argument("--out", default="")
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
    cmode = a.get("canon_mode", "c1_gentle")
    canon = ContentCanonicalizer(torch.load(V6_ROOT / "cache" / "canonicalizer_stats.pt",
                                           map_location="cpu", weights_only=False)).to(dev)

    tasks = json.loads((V5_ROOT / "v5_eval200_tasks.json").read_text(encoding="utf-8"))
    if args.limit:
        tasks = tasks[: args.limit]
    lat = torch.load(V5_ROOT / "cache" / "eval_src_latents.pt", map_location="cpu")
    stems = sorted({t["src_stem"] for t in tasks})
    idx = {s: i for i, s in enumerate(stems)}
    z_all = torch.stack([lat[s] for s in stems]).to(dev).float()
    src_idx = torch.tensor([idx[t["src_stem"]] for t in tasks], device=dev)
    tgt = torch.tensor([t["tgt_style_id"] for t in tasks], device=dev, dtype=torch.long)

    P = np.asarray(csd_prototypes(), dtype=np.float32)
    P = P / np.linalg.norm(P, axis=1, keepdims=True)

    csd, _, _ = csd_model.load_csd(dev)
    vae = load_vae(dev)
    trt = load_trt(vae, dev)
    realF = real_target_features(csd, vae, trt, dev)

    # source CSD features (post round-trip), matching the eval protocol
    with torch.inference_mode():
        img_s = decode_trt(trt, vae, z_all, batch=8)
        u8s = img_s.mul(255.0).round().clamp(0, 255).to(torch.uint8).cpu().permute(0, 2, 3, 1).numpy()
        Fs = csd_model.feats_pils(csd, [Image.fromarray(u8s[i]) for i in range(len(u8s))],
                                  batch=16, device=dev)
        Fs = Fs / np.linalg.norm(Fs, axis=1, keepdims=True)

        z_src = z_all[src_idx]
        c = canon(z_src, mode=cmode, augment=False)
        z_out = model(c, 0.0, c, tgt)
        img_o = decode_trt(trt, vae, z_out, batch=8)
        u8o = img_o.mul(255.0).round().clamp(0, 255).to(torch.uint8).cpu().permute(0, 2, 3, 1).numpy()
        Fo = csd_model.feats_pils(csd, [Image.fromarray(u8o[i]) for i in range(len(u8o))],
                                  batch=16, device=dev)
        Fo = Fo / np.linalg.norm(Fo, axis=1, keepdims=True)

    rows = []
    for i, t in enumerate(tasks):
        ti, si = t["tgt_style_id"], t["src_style_id"]
        fs = Fs[idx[t["src_stem"]]].astype(np.float32)
        fo = Fo[i].astype(np.float32)
        df = fo - fs
        d = P[ti] - P[si]
        dn = np.linalg.norm(d)
        cos_align = float(df @ d / (np.linalg.norm(df) * dn + 1e-12)) if dn > 1e-9 else 0.0
        sc_o = fo @ P.T
        m_t = float(sc_o[ti] - np.delete(sc_o, ti).max())
        m_s = float(sc_o[si] - np.delete(sc_o, si).max())
        # kNN cosine distance to the REAL target-style cloud
        R = realF[STYLES[ti]].numpy()
        dk = 1.0 - (R @ fo)
        knn = float(np.sort(dk)[: args.knn].mean())
        rows.append({
            "task_id": t["task_id"], "src_stem": t["src_stem"],
            "src_style": t["src_style"], "tgt_style": t["tgt_style"],
            "cos_align": cos_align, "df_norm": float(np.linalg.norm(df)),
            "m_target": m_t, "m_source": m_s, "max_cos": float(sc_o.max()),
            "knn_target": knn,
            "argmax": STYLES[int(sc_o.argmax())],
            "src_retained": int(sc_o.argmax() == si),
            "tgt_acquired": int(sc_o.argmax() == ti),
            "near_not_crossed": int(sc_o.argmax() != ti and m_t > -0.10),
        })

    def agg(key):
        return float(np.mean([r[key] for r in rows]))

    per_t = {}
    for s in STYLES:
        sub = [r for r in rows if r["tgt_style"] == s]
        if not sub:
            continue
        per_t[s] = {k: float(np.mean([r[k] for r in sub]))
                    for k in ("cos_align", "m_target", "knn_target",
                              "tgt_acquired", "src_retained")}
        per_t[s]["near_not_crossed"] = float(np.mean([r["near_not_crossed"] for r in sub]))

    res = {
        "ckpt": args.ckpt, "tag": args.tag, "n": len(rows),
        "cos_align_mean": agg("cos_align"),
        "cos_align_p10": float(np.percentile([r["cos_align"] for r in rows], 10)),
        "m_target_mean": agg("m_target"),
        "m_source_mean": agg("m_source"),
        "max_cos_mean": agg("max_cos"),
        "knn_target_mean": agg("knn_target"),
        "df_norm_mean": agg("df_norm"),
        "tgt_acquired": agg("tgt_acquired"),
        "src_retained": agg("src_retained"),
        "near_not_crossed": agg("near_not_crossed"),
        "per_target": per_t,
        "rows": rows,
    }

    print(f"\n[dir] {args.ckpt}  n={res['n']}")
    print(f"  GEOMETRIC alignment cos(df, mu_t-mu_s) = {res['cos_align_mean']:+.4f} "
          f"(p10 {res['cos_align_p10']:+.4f})   <- replaces the pseudo 'purity'")
    print(f"  |df| = {res['df_norm_mean']:.4f}")
    print(f"  m_target = {res['m_target_mean']:+.4f}   m_source = {res['m_source_mean']:+.4f}")
    print(f"  max_cos  = {res['max_cos_mean']:.4f}   kNN->real target = {res['knn_target_mean']:.4f}")
    print(f"  target acquired {res['tgt_acquired']:.3f}   source retained {res['src_retained']:.3f}"
          f"   near-but-not-crossed {res['near_not_crossed']:.3f}")
    print(f"\n  {'target':22s}{'cos_align':>11s}{'m_target':>10s}{'knn':>8s}"
          f"{'acquired':>10s}{'retained':>10s}")
    for s, v in per_t.items():
        print(f"  {s:22s}{v['cos_align']:+11.4f}{v['m_target']:+10.4f}"
              f"{v['knn_target']:8.4f}{v['tgt_acquired']:10.3f}{v['src_retained']:10.3f}")

    out = ROOT / (args.out or f"runs/v6_next/probes/direction_{args.tag or 'ckpt'}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
