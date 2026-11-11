"""Score the Q2A+Oracle1 images under the docs/baseline paper_rgb_v1 protocol.

Metrics follow tools/vae_swap_eval/style_metrics.py and tools/evaluation_v2:
  DINO-S      = max cosine(gen DINOv2-small CLS, 30 target-style reference CLS)
  CLIP-S      = cosine(gen CLIP ViT-B/32, normalized mean of 30 target refs)
  Style Affinity = (DINO-S + CLIP-S)/2        <-- leaderboard primary metric
  DINO-C      = cosine(gen DINO CLS, source round-trip DINO CLS)  [paper_rgb_v1]
  LPIPS       = AlexNet LPIPS(gen, source round-trip)
  CSD Top-1 / Margin = against deliver/testset/refs/csd_proto.pt

Reports cross600 only (the leaderboard uses 600 pure cross-style tasks), plus
same150 as a no-drift check, and inserts the new decoders into the 32-method
UNIFIED_FULL_RANKING.md table.

Run on 4090:
  python -m tools.vae_suite.score_baseline_protocol
"""
from __future__ import annotations

import io
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
REFS = ROOT / "deliver/vae_ranked_eval/vae_style_reference_subset.pt"
CSD_PROTO = ROOT / "deliver/testset/refs/csd_proto.pt"
HF = ROOT / "eval_cache/hf/hub"
STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]


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


def to_tensor(p, resize=None):
    im = Image.open(p).convert("RGB")
    if resize:
        im = im.resize(resize, Image.BICUBIC)
    return torch.from_numpy(np.asarray(im, np.uint8)).permute(2, 0, 1).float().div_(127.5).sub_(1.0)


def pil(p):
    return Image.open(p).convert("RGB")


def dino_cls(model, pils, dev, bs=16):
    """DINOv2-small CLS via tools.pseudo_gt.matching.encode_dino.

    This is EXACTLY the baseline path (DINO_TRANSFORM on PIL, fp16 autocast),
    so DINO-S is comparable with the leaderboard numbers.
    """
    from tools.pseudo_gt.matching import encode_dino
    out = []
    with torch.no_grad():
        for i in range(0, len(pils), bs):
            cls, _ = encode_dino(model, pils[i:i + bs], batch=bs, device=dev)
            out.append(F.normalize(cls.float(), dim=-1).cpu().numpy())
    return np.concatenate(out)


def clip_cls(model, proc, pils, dev, bs=16):
    out = []
    with torch.no_grad():
        for i in range(0, len(pils), bs):
            pvs = [proc(images=p, return_tensors="pt")["pixel_values"][0]
                   for p in pils[i:i + bs]]
            x = torch.stack(pvs).to(dev)
            f = model.get_image_features(pixel_values=x).float()
            out.append(F.normalize(f, dim=-1).cpu().numpy())
    return np.concatenate(out)


def csd_feats(csd, pils, dev, bs=16):
    import csd_model
    f = csd_model.feats_pils(csd, pils, batch=bs, device=dev)
    return f / np.linalg.norm(f, axis=1, keepdims=True)


def main():
    import csd_model, lpips
    from transformers import AutoModel, CLIPModel, CLIPProcessor
    dev = "cuda"

    refs = torch.load(REFS, map_location="cpu", weights_only=False)
    ref_dino = [F.normalize(refs["styles"][s]["dino_cls"].float(), dim=-1).numpy()
                for s in STYLES]                       # (30,384) each
    proto_clip = F.normalize(torch.stack(
        [refs["clip_prototypes"][s].float() for s in STYLES]), dim=-1).numpy()
    proto_csd = np.stack([np.asarray(csd_proto_load()[s], np.float32)
                          for s in STYLES])
    proto_csd = proto_csd / np.linalg.norm(proto_csd, axis=1, keepdims=True)

    dino = AutoModel.from_pretrained("facebook/dinov2-small",
                                     local_files_only=True,
                                     cache_dir=str(HF)).to(dev).eval()
    dino.requires_grad_(False)
    clipm = CLIPModel.from_pretrained("openai/clip-vit-base-patch32",
                                      local_files_only=True,
                                      cache_dir=str(HF)).to(dev).eval()
    clipp = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32",
                                          local_files_only=True,
                                          cache_dir=str(HF))
    clipm.requires_grad_(False)
    csd, _, _ = csd_model.load_csd(dev); csd = csd.float().eval()
    for p in csd.parameters():
        p.requires_grad = False
    lp = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
    for p in lp.parameters():
        p.requires_grad = False

    # ---- source round-trips (paper_rgb_v1 reference) ----
    src_dir = OUT / "sources"
    stems = sorted(p.stem for p in src_dir.glob("*.png"))
    src_idx = {s: i for i, s in enumerate(stems)}
    src_pils = [pil(src_dir / f"{s}.png") for s in stems]
    src_t = [to_tensor(src_dir / f"{s}.png") for s in stems]  # for LPIPS only
    Ds = dino_cls(dino, src_pils, dev)
    Cs = clip_cls(clipm, clipp, src_pils, dev)
    Fs = csd_feats(csd, src_pils, dev)
    print(f"[baseline-protocol] {len(stems)} source refs", flush=True)

    per_task = {}
    results = {}
    for v in ("sdvae", "old_s10", "c0", "c1"):
        mf = json.loads((OUT / v / "manifest.json").read_text())
        rows = mf["rows"]
        paths = [OUT / v / r["file"] for r in rows]
        pils = [pil(p) for p in paths]
        gt = [to_tensor(p) for p in paths]   # for LPIPS only
        Dg = dino_cls(dino, pils, dev)
        Cg = clip_cls(clipm, clipp, pils, dev)
        Fg = csd_feats(csd, pils, dev)
        # DINO-S: max cosine to 30 target refs, for ALL 5 styles -> (n,5)
        dino_scores = np.stack([(Dg @ r.T).max(axis=1) for r in ref_dino], axis=1)
        clip_scores = Cg @ proto_clip.T
        # LPIPS vs source round-trip
        lpv = []
        with torch.no_grad():
            for i in range(0, len(rows), 8):
                g = torch.stack(gt[i:i + 8]).to(dev)
                s = torch.stack([src_t[src_idx[rows[j]["source_stem"]]]
                                 for j in range(i, min(i + 8, len(rows)))]).to(dev)
                lpv.append(lp(g, s).flatten().cpu().numpy())
        lpv = np.concatenate(lpv)
        recs = []
        for j, r in enumerate(rows):
            ti = r["target_style_id"]
            si = src_idx[r["source_stem"]]
            sc = Fg[j] @ proto_csd.T
            din_s = float(dino_scores[j, ti])       # TARGET style max-ref cosine
            clip_s = float(clip_scores[j, ti])      # TARGET style prototype
            recs.append({
                "task_id": r["task_id"], "is_cross": r["is_cross"],
                "source_stem": r["source_stem"],
                "target_style": STYLES[ti], "target_style_id": ti,
                "dino_s": din_s, "clip_s": clip_s,
                "style_affinity": (din_s + clip_s) / 2.0,
                "dino_c": float(Dg[j] @ Ds[si]),
                "lpips": float(lpv[j]),
                "csd_top1": int(int(np.argmax(sc)) == ti),
                "csd_margin": float(sc[ti] - np.max(np.delete(sc, ti))),
                "dino_scores_all5": [float(x) for x in dino_scores[j]],
                "clip_scores_all5": [float(x) for x in clip_scores[j]],
            })
        per_task[v] = recs

        def agg(rs):
            g = lambda k: float(np.nanmean([x[k] for x in rs]))
            return {k: round(g(k), 5) for k in
                    ("dino_s", "clip_s", "style_affinity", "dino_c", "lpips",
                     "csd_top1", "csd_margin")}
        cross = [x for x in recs if x["is_cross"]]
        same = [x for x in recs if not x["is_cross"]]
        pt = defaultdict(list)
        for x in cross:
            pt[x["target_style"]].append(x)
        results[v] = {"cross600": agg(cross), "same150": agg(same),
                      "all750": agg(recs),
                      "per_target_cross600": {s: agg(rs) for s, rs in pt.items()}}
        c = results[v]["cross600"]
        print(f"  {v:<9} affinity={c['style_affinity']:.4f} dino_s={c['dino_s']:.4f} "
              f"clip_s={c['clip_s']:.4f} dino_c={c['dino_c']:.4f} "
              f"lpips={c['lpips']:.4f} top1={c['csd_top1']:.4f}", flush=True)
        del Dg, Cg, Fg, pils, gt
        torch.cuda.empty_cache()

    # ---- paired bootstrap vs sdvae (cross600) ----
    boot = {}
    base = {x["task_id"]: x for x in per_task["sdvae"] if x["is_cross"]}
    common = sorted(base)
    clusters = np.array([base[k]["source_stem"] for k in common])
    for v in ("old_s10", "c0", "c1"):
        other = {x["task_id"]: x for x in per_task[v] if x["is_cross"]}
        for m in ("style_affinity", "dino_s", "clip_s", "dino_c", "lpips",
                  "csd_top1", "csd_margin"):
            boot[f"{v}|{m}"] = paired_ci(
                [other[k][m] for k in common], [base[k][m] for k in common],
                clusters, seed=0)

    # ---- parse the 32-method leaderboard table ----
    lb = ROOT / "docs/baseline/UNIFIED_FULL_RANKING.md"
    if not lb.exists():
        lb = OUT / "UNIFIED_FULL_RANKING.md"      # uploaded copy
    board = parse_leaderboard(lb)
    out = {
        "protocol": "paper_rgb_v1, cross600, docs/baseline 32-method leaderboard",
        "metric_definitions": {
            "dino_s": "max cosine to 30 frozen target-style DINOv2-small CLS refs",
            "clip_s": "cosine to normalized mean of 30 target-style CLIP refs",
            "style_affinity": "(DINO-S + CLIP-S)/2  [leaderboard primary]",
            "dino_c": "gen vs source round-trip DINO CLS cosine",
            "lpips": "AlexNet LPIPS(gen, source round-trip)",
        },
        "new_decoders": results,
        "paired_bootstrap_vs_sdvae_cross600": boot,
        "leaderboard_baseline_methods": board,
    }
    (OUT / "scores_baseline_protocol.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    (OUT / "per_task_baseline_protocol.json").write_text(
        json.dumps(per_task), encoding="utf-8")
    print(f"[Saved] {OUT / 'scores_baseline_protocol.json'}", flush=True)


def csd_proto_load():
    return torch.load(CSD_PROTO, map_location="cpu", weights_only=False)


def parse_leaderboard(path):
    """Parse section 1 (Style Affinity) table of UNIFIED_FULL_RANKING.md."""
    if not path.exists():
        return []
    rows = []
    in_sec1 = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## 一、Style Affinity"):
            in_sec1 = True
            continue
        if in_sec1 and line.startswith("## 二、"):
            break
        if not in_sec1 or not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 10 or not cells[0].isdigit():
            continue
        try:
            rows.append({
                "rank": int(cells[0]),
                "method": cells[1].replace("**", ""),
                "category": cells[2], "vae": cells[3],
                "style_affinity": float(cells[4].replace("**", "")),
                "dino_s": float(cells[5]), "clip_s": float(cells[6]),
                "dino_c": float(cells[7]), "lpips": float(cells[8]),
                "csd_top1": float(cells[9].replace("%", "")) / 100.0,
            })
        except ValueError:
            continue
    return rows


if __name__ == "__main__":
    main()
