"""Score the Q2A+Oracle1 generated images (750 tasks) from PNG on disk.

Reads runs/sd0_s10/q2a_oracle_images/<variant>/{cross,same}/*.png plus the
shared sources/ round-trips, and computes per-task metrics using the SAME
feature extractors as the historical ranked eval (CSD ViT-L, DINOv2-base,
CLIP ViT-B/32, AlexNet LPIPS).

Reports:
  - cross600 / same150 / all750 for every variant
  - CSD Top-1, CSD Margin, dT, dS, DINO-C, DINO-S, CLIP-S, LPIPS
  - per-target 5-style breakdown (cross600)
  - paired source-clustered bootstrap (c0/c1/old_s10 vs sdvae) with duplicate
    cluster draws preserved
  - ranking of all (variant) rows by each metric

Run on 4090 (needs GPU for features):
  python -m tools.vae_suite.score_q2a_oracle_images
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path("/root/Workspace/WEAVE_VAE")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "12h"))

OUT = ROOT / "runs/sd0_s10/q2a_oracle_images"
STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
HF_CACHE = str(ROOT / "eval_cache/hf/hub")
NM = [0.48145466, 0.4578275, 0.40821073]
NS = [0.26862954, 0.26130258, 0.27577711]


def cluster_boot_idx(clusters, n_boot, seed):
    uniq, inv = np.unique(clusters, return_inverse=True)
    by = [np.where(inv == i)[0] for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        yield np.concatenate([by[c] for c in pick])


def paired_ci(a, b, clusters, n_boot=2000, seed=0):
    d = np.asarray(a, float) - np.asarray(b, float)
    ms = np.array([d[i].mean() for i in cluster_boot_idx(clusters, n_boot, seed)])
    lo, hi = float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5))
    return {"mean": float(d.mean()), "ci": [lo, hi],
            "excludes_zero": bool(lo > 0 or hi < 0)}


def load_pil(p):
    return Image.open(p).convert("RGB")


def csd_feats(csd, pils, dev, bs=16):
    import csd_model
    f = csd_model.feats_pils(csd, pils, batch=bs, device=dev)
    return f / np.linalg.norm(f, axis=1, keepdims=True)


def dino_feats(dino, pils, dev, bs=16):
    out = []
    nm = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    ns = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    with torch.no_grad():
        for i in range(0, len(pils), bs):
            x = torch.stack([torch.from_numpy(
                np.asarray(p.resize((224, 224)), np.uint8))
                .permute(2, 0, 1).float().div_(255.0) for p in pils[i:i + bs]]).to(dev)
            f = dino(pixel_values=(x - nm) / ns).last_hidden_state[:, 0].float()
            out.append(F.normalize(f, dim=-1).cpu().numpy())
            del x, f
            torch.cuda.empty_cache()
    return np.concatenate(out)


def clip_feats(m, proc, pils, dev, bs=16):
    out = []
    with torch.no_grad():
        for i in range(0, len(pils), bs):
            pvs = [proc(images=p, return_tensors="pt")["pixel_values"][0]
                   for p in pils[i:i + bs]]
            x = torch.stack(pvs).to(dev)
            f = m.get_image_features(pixel_values=x).float()
            out.append(F.normalize(f, dim=-1).cpu().numpy())
            del x, f
            torch.cuda.empty_cache()
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="sdvae,old_s10,c0,c1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bs", type=int, default=16)
    args = ap.parse_args()
    dev = args.device

    import csd_model, lpips
    from transformers import AutoModel, CLIPModel, CLIPProcessor
    csd, _, _ = csd_model.load_csd(dev); csd = csd.float().eval()
    for p in csd.parameters():
        p.requires_grad = False
    lp = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
    for p in lp.parameters():
        p.requires_grad = False
    dino = AutoModel.from_pretrained("facebook/dinov2-base",
                                     local_files_only=True).to(dev).eval()
    clipm = CLIPModel.from_pretrained("openai/clip-vit-base-patch32",
                                      local_files_only=True,
                                      cache_dir=HF_CACHE).to(dev).eval()
    clipp = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32",
                                          local_files_only=True,
                                          cache_dir=HF_CACHE)
    proto_d = torch.load(ROOT / "deliver/testset/refs/csd_proto.pt",
                         map_location="cpu", weights_only=False)
    proto = np.stack([np.asarray(proto_d[s], np.float32) for s in STYLES])
    proto = proto / np.linalg.norm(proto, axis=1, keepdims=True)
    clp = None
    rf = ROOT / "deliver/vae_ranked_eval/vae_style_reference_subset.pt"
    if rf.exists():
        _r = torch.load(rf, map_location="cpu", weights_only=False)
        _cp = _r.get("clip_prototypes", {})
        if all(k in _cp for k in STYLES):
            clp = np.stack([np.asarray(_cp[k], np.float32) for k in STYLES])
            clp = clp / np.linalg.norm(clp, axis=1, keepdims=True)

    # source round-trip features (shared reference, computed ONCE)
    src_dir = OUT / "sources"
    src_stems = sorted(p.stem for p in src_dir.glob("*.png"))
    src_idx = {s: i for i, s in enumerate(src_stems)}
    src_pils = [load_pil(src_dir / f"{s}.png") for s in src_stems]
    print(f"[score] {len(src_pils)} source round-trips", flush=True)
    Fs = csd_feats(csd, src_pils, dev, args.bs)
    Ds = dino_feats(dino, src_pils, dev, args.bs)
    Cs = clip_feats(clipm, clipp, src_pils, dev, args.bs)
    src_t = torch.stack([torch.from_numpy(
        np.asarray(p, np.uint8)).permute(2, 0, 1).float().div_(127.5).sub_(1.0)
        for p in src_pils])
    del src_pils

    results = {}
    per_task = {}
    variants = [v.strip() for v in args.variants.split(",")]
    for v in variants:
        mf = json.loads((OUT / v / "manifest.json").read_text())
        rows = mf["rows"]
        print(f"\n[score] {v}: {len(rows)} tasks", flush=True)
        pils = [load_pil(OUT / v / r["file"]) for r in rows]
        Fv = csd_feats(csd, pils, dev, args.bs)
        Dv = dino_feats(dino, pils, dev, args.bs)
        Cv = clip_feats(clipm, clipp, pils, dev, args.bs)
        del pils
        gen_t = torch.stack([torch.from_numpy(
            np.asarray(load_pil(OUT / v / r["file"]), np.uint8))
            .permute(2, 0, 1).float().div_(127.5).sub_(1.0) for r in rows])
        # LPIPS in batches (vs source round-trip)
        lpv = []
        with torch.no_grad():
            for i in range(0, len(rows), 8):
                g = gen_t[i:i + 8].to(dev)
                s = torch.stack([src_t[src_idx[rows[j]["source_stem"]]]
                                 for j in range(i, min(i + 8, len(rows)))]).to(dev)
                lpv.append(lp(g, s).flatten().cpu().numpy())
                del g, s
                torch.cuda.empty_cache()
        lpv = np.concatenate(lpv)
        del gen_t
        torch.cuda.empty_cache()

        recs = []
        for j, r in enumerate(rows):
            si = src_idx[r["source_stem"]]
            ti, soi = r["target_style_id"], r["source_style_id"]
            sc = Fv[j] @ proto.T
            top1 = int(int(np.argmax(sc)) == ti)
            margin = float(sc[ti] - np.max(np.delete(sc, ti)))
            dino_c = float(Dv[j] @ Ds[si])
            clip_s = float(Cv[j] @ Cs[si])
            clip_sel = float("nan")
            if clp is not None:
                dy = Cv[j] @ clp.T
                dx = Cs[si] @ clp.T
                dd = dy - dx
                clip_sel = float(dd[ti] - np.max(np.delete(dd, ti)))
            recs.append({
                "task_id": r["task_id"], "is_cross": r["is_cross"],
                "source_stem": r["source_stem"], "target_style": STYLES[ti],
                "target_style_id": ti, "source_style_id": soi,
                "top1": top1, "margin": margin,
                "dT": float(Fv[j] @ proto[ti] - Fs[si] @ proto[ti]),
                "dS": float(Fv[j] @ proto[soi] - Fs[si] @ proto[soi]),
                "dino_s": dino_c, "dino_c": dino_c, "clip_s": clip_s,
                "clip_target_selectivity": clip_sel,
                "lpips": float(lpv[j]),
            })
        per_task[v] = recs

        def agg(rs):
            g = lambda k: float(np.nanmean([x[k] for x in rs]))
            return {k: round(g(k), 5) for k in
                    ("top1", "margin", "dT", "dS", "dino_s", "dino_c",
                     "clip_s", "clip_target_selectivity", "lpips")}
        cross = [x for x in recs if x["is_cross"]]
        same = [x for x in recs if not x["is_cross"]]
        pt = defaultdict(list)
        for x in cross:
            pt[x["target_style"]].append(x)
        results[v] = {
            "cross600": agg(cross), "same150": agg(same), "all750": agg(recs),
            "per_target_cross600": {s: agg(rs) for s, rs in pt.items()},
            "n_cross": len(cross), "n_same": len(same),
        }
        print(f"  cross600 top1={results[v]['cross600']['top1']:.4f} "
              f"margin={results[v]['cross600']['margin']:+.4f} "
              f"dino_c={results[v]['cross600']['dino_c']:.4f} "
              f"lpips={results[v]['cross600']['lpips']:.4f}", flush=True)
        del Fv, Dv, Cv
        torch.cuda.empty_cache()

    # paired bootstrap on cross600: every variant vs sdvae
    boot = {}
    if "sdvae" in per_task:
        base = {x["task_id"]: x for x in per_task["sdvae"] if x["is_cross"]}
        common = sorted(base)
        clusters = np.array([base[k]["source_stem"] for k in common])
        for v in variants:
            if v == "sdvae":
                continue
            other = {x["task_id"]: x for x in per_task[v] if x["is_cross"]}
            for m in ("top1", "margin", "dino_s", "clip_s", "dino_c", "lpips",
                      "clip_target_selectivity"):
                a = [other[k][m] for k in common]
                b = [base[k][m] for k in common]
                boot[f"{v}|{m}"] = paired_ci(a, b, clusters, seed=0)

    # rankings by metric (cross600)
    rank = {}
    for m in ("top1", "margin", "dT", "dS", "dino_s", "clip_s", "dino_c",
              "lpips", "clip_target_selectivity"):
        rank[m] = [{"variant": v, "value": results[v]["cross600"][m]}
                   for v in sorted(variants,
                                   key=lambda v: -results[v]["cross600"][m])]

    out = {
        "protocol": "Q2A+Oracle1, 750 frozen tasks, images scored from PNG",
        "reference": "sdvae source round-trips (sources/*.png)",
        "variants": variants,
        "results": results, "paired_bootstrap_vs_sdvae_cross600": boot,
        "ranking_cross600": rank,
        "per_task_records": {v: len(r) for v, r in per_task.items()},
    }
    (OUT / "scores.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    # also save compact per-task CSV-friendly records for later metrics
    (OUT / "per_task.json").write_text(json.dumps(per_task), encoding="utf-8")
    print(f"\n[Saved] {OUT / 'scores.json'} + per_task.json", flush=True)
    print(json.dumps(rank["top1"], indent=2), flush=True)


if __name__ == "__main__":
    main()
