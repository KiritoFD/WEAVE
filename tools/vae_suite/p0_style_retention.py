"""P0: Style information retention through VAE roundtrip.

Question: does x -> E -> z -> D -> x_hat preserve artistic style information?

Measures CSD style features on original x, B0 reconstruction, A0 reconstruction
for the SAME 2,048 val images and the SAME pairs, reporting:
  Cross-Artist AUROC, Content-Matched AUROC, Same-Genre Hard AUROC,
  kNN@1/5/10 style purity, and per-image cos(CSD(x), CSD(x_hat)).

Verdict:
  - If discriminability largely survives -> style info is in the latent; focus on reorganization.
  - If it drops sharply -> Encoder/Decoder already lost style; a reversible T cannot recover it.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "/root/Workspace/WEAVE_VAE")
sys.path.insert(0, "/root/Workspace/WEAVE_VAE/tools/12h")

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from transformers import AutoModel

from tools.vae_suite.dataset import RawMemmapWikiArtDataset
import csd_model

OUT = Path("/root/Workspace/WEAVE_VAE/runs_vae/v2_diagnosis")
MANIFEST = Path("/root/Workspace/data/wikiart_512_raw/wikiart_512_raw_manifest.json")
B0_DIR = "/root/Workspace/xy/DiT/data/pretrained/sd-vae-ft-ema"
A0_DIR = "/root/Workspace/WEAVE_VAE/runs_vae/A0/final_diffusers_vae"

NORM_MEAN = [0.48145466, 0.4578275, 0.40821073]
NORM_STD = [0.26862954, 0.26130258, 0.27577711]
DINO_MEAN = [0.485, 0.456, 0.406]
DINO_STD = [0.229, 0.224, 0.225]


def csd_style(csd, img, nm, ns):
    x01 = (img.clamp(-1, 1) + 1.0) / 2.0
    x224 = F.interpolate(x01, (224, 224), mode="bicubic", align_corners=False)
    _, _, s = csd((x224 - nm) / ns)
    return s.float()


def dino_feat(m, img, device):
    x01 = (img.clamp(-1, 1) + 1.0) / 2.0
    x224 = F.interpolate(x01, (224, 224), mode="bicubic", align_corners=False)
    nm = torch.tensor(DINO_MEAN, device=device).view(1, 3, 1, 1)
    ns = torch.tensor(DINO_STD, device=device).view(1, 3, 1, 1)
    o = m(pixel_values=(x224 - nm) / ns)
    return o.last_hidden_state[:, 0].float()


def build_pairs(y, a, g, seed=42):
    rng = np.random.default_rng(seed)
    N = len(y)
    same, cross, diff, hard = [], [], [], []
    for i in range(N):
        sc = np.where(y == y[i])[0]; sc = sc[sc != i]
        if len(sc):
            same.append((i, int(rng.choice(sc))))
            if a[i] >= 0:
                ca = sc[a[sc] != a[i]]
                if len(ca):
                    cross.append((i, int(rng.choice(ca))))
        dc = np.where(y != y[i])[0]
        if len(dc):
            diff.append((i, int(rng.choice(dc))))
            if g[i] >= 0:
                hc = dc[g[dc] == g[i]]
                if len(hc):
                    hard.append((i, int(rng.choice(hc))))
    return same, cross, diff, hard


def build_cm(dino, y, a, g, cap=1000, seed=42):
    rng = np.random.default_rng(seed)
    pos, neg, gaps = [], [], []
    for i in rng.permutation(len(y)):
        if len(pos) >= cap:
            break
        pc = np.where(y == y[i])[0]; pc = pc[pc != i]
        if a[i] >= 0:
            pc = pc[a[pc] != a[i]]
        if len(pc) == 0:
            continue
        nc = np.where((y != y[i]) & (g == g[i]))[0]
        if len(nc) < 2:
            continue
        if len(pc) > 12: pc = rng.choice(pc, 12, replace=False)
        if len(nc) > 24: nc = rng.choice(nc, 24, replace=False)
        sp = dino[pc] @ dino[i]; sn = dino[nc] @ dino[i]
        gap = np.abs(sp[:, None] - sn[None, :])
        pi, ni = np.unravel_index(np.argmin(gap), gap.shape)
        pos.append((int(i), int(pc[pi]))); neg.append((int(i), int(nc[ni])))
        gaps.append(float(gap[pi, ni]))
    return pos, neg, np.array(gaps)


def auroc(feats, pos, neg, n_boot=200, seed=42, cap=800):
    n = min(len(pos), len(neg))
    if n < 10:
        return None
    n = min(n, cap)
    sp = np.array(pos[:n]); sn = np.array(neg[:n])
    a1 = (feats[sp[:, 0]] * feats[sp[:, 1]]).sum(1)
    a0 = (feats[sn[:, 0]] * feats[sn[:, 1]]).sum(1)
    lbs = np.concatenate([np.ones(n), np.zeros(n)])
    try:
        auc = float(roc_auc_score(lbs, np.concatenate([a1, a0])) * 100)
    except ValueError:
        return None
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        pi = rng.integers(0, n, n); ni = rng.integers(0, n, n)
        try:
            boot.append(roc_auc_score(lbs, np.concatenate([a1[pi], a0[ni]])) * 100)
        except ValueError:
            pass
    lo, hi = np.percentile(boot, [2.5, 97.5]) if boot else (auc, auc)
    return {"auroc": round(auc, 2), "lo": round(float(lo), 2), "hi": round(float(hi), 2), "n": int(n)}


def knn_purity(feats, styles, ks=(1, 5, 10), block=256):
    N = len(styles)
    hits = {k: 0 for k in ks}
    for st in range(0, N, block):
        en = min(st + block, N)
        sim = feats[st:en] @ feats.T
        for r in range(en - st):
            sim[r, st + r] = -np.inf
        order = np.argsort(-sim, axis=1)
        for r in range(en - st):
            for k in ks:
                if styles[st + r] in styles[order[r, :k]]:
                    hits[k] += 1
    return {f"knn{k}": round(hits[k] / N * 100, 2) for k in ks}


def norm_rows(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def main():
    device = "cuda"
    ds = RawMemmapWikiArtDataset(MANIFEST, is_training=False)
    y, a, g = [], [], []
    for i in range(len(ds)):
        rec = ds.records_by_idx[ds.indices[i]]
        y.append(rec["style_id"]); a.append(rec.get("artist_id", -1)); g.append(rec.get("genre_id", -1))
    y = np.array(y); a = np.array(a); g = np.array(g)

    same, cross, diff, hard = build_pairs(y, a, g)

    print("[DINO] for content matching...", flush=True)
    dino_m = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval()
    dl = DataLoader(ds, batch_size=8, shuffle=False)
    df = []
    with torch.no_grad():
        for i, b in enumerate(dl):
            df.append(F.normalize(dino_feat(dino_m, b["image"].to(device), device), dim=-1).cpu())
            if (i + 1) % 40 == 0:
                print(f"  {i+1}/{len(dl)}", flush=True)
    del dino_m
    torch.cuda.empty_cache()
    dino = norm_rows(torch.cat(df).numpy())
    cm_pos, cm_neg, cm_gap = build_cm(dino, y, a, g)
    print(f"[Pairs] cross={len(cross)} hard={len(hard)} cm={len(cm_pos)} gap={cm_gap.mean():.4f}", flush=True)

    csd, _, _ = csd_model.load_csd(device)
    csd = csd.float().eval()
    for p in csd.parameters():
        p.requires_grad = False
    nm = torch.tensor(NORM_MEAN, device=device).view(1, 3, 1, 1)
    ns = torch.tensor(NORM_STD, device=device).view(1, 3, 1, 1)

    feats = {}
    cos_x_hat = {}

    # original
    print("\n[CSD] original images...", flush=True)
    f = []
    dl = DataLoader(ds, batch_size=16, shuffle=False)
    with torch.no_grad():
        for i, b in enumerate(dl):
            f.append(csd_style(csd, b["image"].to(device), nm, ns).cpu())
            if (i + 1) % 40 == 0:
                print(f"  {i+1}/{len(dl)}", flush=True)
    f = torch.cat(f).numpy()
    feats["original"] = norm_rows(f)
    f_raw = feats["original"]

    for tag, path in (("B0", B0_DIR), ("A0", A0_DIR)):
        print(f"\n[{tag}] encode+decode+CSD ...", flush=True)
        vae = AutoencoderKL.from_pretrained(path).to(device).eval()
        rec_feats, cosines = [], []
        with torch.no_grad():
            for i, b in enumerate(dl):
                x = b["image"].to(device)
                post = vae.encode(x).latent_dist
                xhat = vae.decode(post.mean).sample
                fx = csd_style(csd, x, nm, ns)
                fh = csd_style(csd, xhat, nm, ns)
                rec_feats.append(fh.cpu())
                cosines.append(F.cosine_similarity(fx, fh, dim=-1).cpu())
                if (i + 1) % 40 == 0:
                    print(f"  {i+1}/{len(dl)}", flush=True)
        fr = torch.cat(rec_feats).numpy()
        feats[tag] = norm_rows(fr)
        cos_x_hat[tag] = torch.cat(cosines).numpy()
        del vae
        torch.cuda.empty_cache()

    results = {}
    for tag in ("original", "B0", "A0"):
        Fm = feats[tag]
        r = {
            "cross_artist": auroc(Fm, cross, diff),
            "same_genre_hard": auroc(Fm, cross, hard),
            "content_matched": auroc(Fm, cm_pos, cm_neg),
            "knn": knn_purity(Fm, y),
        }
        results[tag] = r
        print(f"\n--- {tag} ---", flush=True)
        print(f"  CrossArtist={r['cross_artist']['auroc']} | Hard={r['same_genre_hard']['auroc']} "
              f"| CM={r['content_matched']['auroc']} | kNN={r['knn']}", flush=True)

    for tag in ("B0", "A0"):
        c = cos_x_hat[tag]
        results[tag]["csd_cos_to_original"] = {
            "mean": round(float(c.mean()), 4),
            "median": round(float(np.median(c)), 4),
            "p5": round(float(np.percentile(c, 5)), 4),
            "p95": round(float(np.percentile(c, 95)), 4),
        }
        print(f"  {tag} cos(CSD(x), CSD(x_hat)): mean={c.mean():.4f} "
              f"median={np.median(c):.4f} p5={np.percentile(c,5):.4f}", flush=True)

    results["pair_stats"] = {"cross": len(cross), "hard": len(hard),
                             "cm": len(cm_pos), "cm_gap": round(float(cm_gap.mean()), 5)}
    (OUT / "P0_style_retention.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n" + "=" * 92)
    print("P0: STYLE INFORMATION RETENTION THROUGH VAE ROUNDTRIP")
    print("=" * 92)
    print(f"{'repr':<12}{'CrossArtist':>13}{'Hard':>9}{'ContentMatch':>14}{'kNN@1':>8}{'kNN@5':>8}{'cos(x,xh)':>11}")
    print("-" * 92)
    for tag in ("original", "B0", "A0"):
        r = results[tag]
        cc = r.get("csd_cos_to_original", {}).get("mean", 1.0)
        print(f"{tag:<12}{r['cross_artist']['auroc']:>13.2f}{r['same_genre_hard']['auroc']:>9.2f}"
              f"{r['content_matched']['auroc']:>14.2f}{r['knn']['knn1']:>8.2f}{r['knn']['knn5']:>8.2f}"
              f"{cc:>11.4f}")
    print("=" * 92)

    o = results["original"]; 
    for tag in ("B0", "A0"):
        r = results[tag]
        print(f"{tag} retention: CrossArtist {r['cross_artist']['auroc']-o['cross_artist']['auroc']:+.2f}pp | "
              f"CM {r['content_matched']['auroc']-o['content_matched']['auroc']:+.2f}pp | "
              f"kNN@1 {r['knn']['knn1']-o['knn']['knn1']:+.2f}pp")
    print(f"\n[Saved] {OUT/'P0_style_retention.json'}")


if __name__ == "__main__":
    main()
