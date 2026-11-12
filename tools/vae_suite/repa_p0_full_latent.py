"""Style-REPA P0: Offline probe of candidate full-latent representations.

No training. Evaluates, on the 2,048 held-out validation images, which
parameter-free representation of the raw VAE latent best carries style signal
and could plausibly be steered toward the CSD teacher geometry.

Descriptors:
  Pool8x8            256d   current Style-REPA student baseline
  RawFlatten         16384d raw latent direct cosine
  PU2_Gram           16x16  PixelUnshuffle(2) + channel Gram
  PU4_Gram           64x64  PixelUnshuffle(4) + channel Gram  (main candidate)
  PU8_Gram           256x256 PixelUnshuffle(8) + channel Gram
  Gram3Scale         3-scale concatenation
  CSD_Style_CLS      768d   teacher upper reference (not a latent)

Metrics: Overall / Cross-Artist / Same-Genre-Hard / Content-Matched AUROC
         (paired bootstrap 95% CI), kNN@1/5/10, mAP@10,
         Spearman vs CSD style similarity matrix, scale invariance.

Output: full_latent_representation_probe.csv + .json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/Workspace/WEAVE_VAE")
sys.path.insert(0, "/root/Workspace/WEAVE_VAE/tools/12h")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from transformers import AutoModel

from tools.vae_suite.dataset import RawMemmapWikiArtDataset
import csd_model

OUT_DIR = Path("/root/Workspace/WEAVE_VAE/runs_vae/v2_diagnosis")
MANIFEST = Path("/root/Workspace/data/wikiart_512_raw/wikiart_512_raw_manifest.json")
B0_DIR = "/root/Workspace/xy/DiT/data/pretrained/sd-vae-ft-ema"
A0_DIR = "/root/Workspace/WEAVE_VAE/runs_vae/A0/final_diffusers_vae"
V2_CKPT = Path("/root/Workspace/WEAVE_VAE/runs_vae/V2/checkpoint_step_001800.pt")

NORM_MEAN = [0.48145466, 0.4578275, 0.40821073]
NORM_STD = [0.26862954, 0.26130258, 0.27577711]
DINO_MEAN = [0.485, 0.456, 0.406]
DINO_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------- descriptors
def d_pool8(z):
    return F.adaptive_avg_pool2d(z, (8, 8)).flatten()


def d_raw(z):
    return z.flatten()


def d_pu_gram(z, s):
    """PixelUnshuffle(s) then channel Gram matrix (upper triangle incl diag)."""
    u = F.pixel_unshuffle(z.unsqueeze(0), s).squeeze(0)   # (4*s^2, 64/s, 64/s)
    C, H, W = u.shape
    X = u.reshape(C, -1)
    G = (X @ X.t()) / float(H * W)
    iu = torch.triu_indices(C, C, device=G.device)
    return G[iu[0], iu[1]]


DESCRIPTORS = {
    "Pool8x8": lambda z: d_pool8(z),
    "RawFlatten": lambda z: d_raw(z),
    "PU2_Gram": lambda z: d_pu_gram(z, 2),
    "PU4_Gram": lambda z: d_pu_gram(z, 4),
    "PU8_Gram": lambda z: d_pu_gram(z, 8),
    "Gram3Scale": lambda z: torch.cat([F.normalize(d_pu_gram(z, 2), dim=0),
                                       F.normalize(d_pu_gram(z, 4), dim=0),
                                       F.normalize(d_pu_gram(z, 8), dim=0)]),
}


def normalize_rows(M):
    M = M.astype(np.float32)
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)


# ---------------------------------------------------------------- latents
def extract_latents(kind, dataset, device, batch_size=16):
    if kind == "B0":
        vae = AutoencoderKL.from_pretrained(B0_DIR)
    else:
        vae = AutoencoderKL.from_pretrained(A0_DIR)
        if kind == "V2":
            ck = torch.load(V2_CKPT, map_location="cpu", weights_only=False)
            vae.load_state_dict(ck.get("ema", ck["model"]))
    vae = vae.to(device).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    zs = []
    with torch.no_grad():
        for i, b in enumerate(loader):
            post = vae.encode(b["image"].to(device)).latent_dist
            zs.append(post.mean.float().cpu())
            if (i + 1) % 40 == 0:
                print(f"    [{kind}] {i+1}/{len(loader)}", flush=True)
    del vae
    torch.cuda.empty_cache()
    return torch.cat(zs)


def extract_csd(dataset, device, batch_size=16):
    csd, _, _ = csd_model.load_csd(device)
    nm = torch.tensor(NORM_MEAN, device=device).view(1, 3, 1, 1)
    ns = torch.tensor(NORM_STD, device=device).view(1, 3, 1, 1)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    feats = []
    with torch.no_grad():
        for i, b in enumerate(loader):
            x = b["image"].to(device)
            x01 = (x.clamp(-1, 1) + 1.0) / 2.0
            x224 = F.interpolate(x01, (224, 224), mode="bicubic", align_corners=False)
            xin = ((x224 - nm) / ns).half()
            _, _, sty = csd(xin)
            feats.append(sty.float().cpu())
            if (i + 1) % 40 == 0:
                print(f"    [CSD] {i+1}/{len(loader)}", flush=True)
    del csd
    torch.cuda.empty_cache()
    f = torch.cat(feats).numpy()
    return normalize_rows(f)


def extract_dino(dataset, device, batch_size=8):
    dino = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval()
    nm = torch.tensor(DINO_MEAN, device=device).view(1, 3, 1, 1)
    ns = torch.tensor(DINO_STD, device=device).view(1, 3, 1, 1)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    feats = []
    with torch.no_grad():
        for i, b in enumerate(loader):
            x = b["image"].to(device)
            x01 = (x.clamp(-1, 1) + 1.0) / 2.0
            x224 = F.interpolate(x01, (224, 224), mode="bicubic", align_corners=False)
            out = dino(pixel_values=(x224 - nm) / ns)
            feats.append(F.normalize(out.last_hidden_state[:, 0].float(), dim=-1).cpu())
            if (i + 1) % 40 == 0:
                print(f"    [DINO] {i+1}/{len(loader)}", flush=True)
    del dino
    torch.cuda.empty_cache()
    return normalize_rows(torch.cat(feats).numpy())


# ---------------------------------------------------------------- pairs
def build_pairs(styles, artists, genres, seed=42):
    rng = np.random.default_rng(seed)
    N = len(styles)
    same, cross, diff, hard = [], [], [], []
    for i in range(N):
        sc = np.where(styles == styles[i])[0]
        sc = sc[sc != i]
        if len(sc):
            same.append((i, int(rng.choice(sc))))
            if artists[i] >= 0:
                ca = sc[artists[sc] != artists[i]]
                if len(ca):
                    cross.append((i, int(rng.choice(ca))))
        dc = np.where(styles != styles[i])[0]
        if len(dc):
            diff.append((i, int(rng.choice(dc))))
            if genres[i] >= 0:
                hc = dc[genres[dc] == genres[i]]
                if len(hc):
                    hard.append((i, int(rng.choice(hc))))
    return same, cross, diff, hard


def build_content_matched(dino, styles, artists, genres, max_pairs=1000, seed=42):
    rng = np.random.default_rng(seed)
    N = len(styles)
    pos, neg, gaps = [], [], []
    for i in rng.permutation(N):
        if len(pos) >= max_pairs:
            break
        pc = np.where(styles == styles[i])[0]
        pc = pc[pc != i]
        if artists[i] >= 0:
            pc = pc[artists[pc] != artists[i]]
        if len(pc) == 0:
            continue
        nc = np.where((styles != styles[i]) & (genres == genres[i]))[0]
        if len(nc) < 2:
            continue
        if len(pc) > 12:
            pc = rng.choice(pc, 12, replace=False)
        if len(nc) > 24:
            nc = rng.choice(nc, 24, replace=False)
        sp = dino[pc] @ dino[i]
        sn = dino[nc] @ dino[i]
        gap = np.abs(sp[:, None] - sn[None, :])
        pi, ni = np.unravel_index(np.argmin(gap), gap.shape)
        pos.append((int(i), int(pc[pi])))
        neg.append((int(i), int(nc[ni])))
        gaps.append(float(gap[pi, ni]))
    return pos, neg, np.array(gaps)


# ---------------------------------------------------------------- metrics
def auroc_ci(feats, pos, neg, n_boot=200, seed=42, cap=800):
    n = min(len(pos), len(neg))
    if n < 10:
        return None
    n = min(n, cap)
    sp = np.array(pos[:n]); sn = np.array(neg[:n])
    sps = (feats[sp[:, 0]] * feats[sp[:, 1]]).sum(1)
    sns = (feats[sn[:, 0]] * feats[sn[:, 1]]).sum(1)
    lbs = np.concatenate([np.ones(n), np.zeros(n)])
    try:
        auc = float(roc_auc_score(lbs, np.concatenate([sps, sns])) * 100.0)
    except ValueError:
        return None
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        pi = rng.integers(0, n, n); ni = rng.integers(0, n, n)
        ss = np.concatenate([sps[pi], sns[ni]])
        if np.ptp(ss) == 0:
            continue
        bb = np.concatenate([np.ones(n), np.zeros(n)])
        try:
            boot.append(roc_auc_score(bb, ss) * 100.0)
        except ValueError:
            continue
    if not boot:
        lo, hi = auc, auc
    else:
        lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"auroc": round(auc, 2), "lo": round(float(lo), 2), "hi": round(float(hi), 2), "n": int(n)}


def knn_metrics(feats, styles, ks=(1, 5, 10), block=256):
    N = len(styles)
    hits = {k: 0 for k in ks}
    for start in range(0, N, block):
        end = min(start + block, N)
        sim = feats[start:end] @ feats.T
        for r in range(end - start):
            sim[r, start + r] = -np.inf
        order = np.argsort(-sim, axis=1)
        for r in range(end - start):
            for k in ks:
                if styles[start + r] in styles[order[r, :k]]:
                    hits[k] += 1
    return {f"knn{k}": round(hits[k] / N * 100.0, 2) for k in ks}


def map10(feats, styles, seed=42, n_q=200):
    rng = np.random.default_rng(seed)
    N = len(styles)
    idx = rng.permutation(N)
    q, g = idx[:n_q], idx[n_q:]
    aps = []
    gf, gs = feats[g], styles[g]
    for qi in q:
        sims = gf @ feats[qi]
        order = np.argsort(-sims)[:10]
        rel = (gs[order] == styles[qi]).astype(float)
        if rel.sum() == 0:
            aps.append(0.0); continue
        prec = np.cumsum(rel) / (np.arange(len(rel)) + 1)
        aps.append(float((prec * rel).sum() / rel.sum()))
    return round(float(np.mean(aps) * 100), 2)


def spearman_vs(ref_sim, sim):
    iu = np.triu_indices(sim.shape[0], k=1)
    a = ref_sim[iu].astype(np.float64)
    b = sim[iu].astype(np.float64)
    ra = np.argsort(np.argsort(a)).astype(np.float64); ra -= ra.mean()
    rb = np.argsort(np.argsort(b)).astype(np.float64); rb -= rb.mean()
    return round(float((ra @ rb) / (np.linalg.norm(ra) * np.linalg.norm(rb) + 1e-12)), 4)


def apply_batch(dfn, z_batch, chunk=128):
    """Apply a single-sample descriptor fn over a 4D latent batch."""
    out = []
    with torch.no_grad():
        for i in range(0, z_batch.shape[0], chunk):
            c = z_batch[i:i + chunk]
            out.append(torch.stack([dfn(z) for z in c]))
    return torch.cat(out)


def scale_invariance(dfn, z_sample, styles_sample, seed=42):
    """Does scaling the latent change the ranking? Compare AUROC at s=0.5/1.0/2.0."""
    rng = np.random.default_rng(seed)
    N = len(styles_sample)
    same, diff = [], []
    for i in range(N):
        sc = np.where(styles_sample == styles_sample[i])[0]
        sc = sc[sc != i]
        if len(sc):
            same.append((i, int(rng.choice(sc))))
        dc = np.where(styles_sample != styles_sample[i])[0]
        if len(dc):
            diff.append((i, int(rng.choice(dc))))
    out = {}
    for s in (0.5, 1.0, 2.0):
        f = normalize_rows(apply_batch(dfn, z_sample * s).numpy())
        r = auroc_ci(f, same, diff, n_boot=60)
        out[f"s{s}"] = r["auroc"] if r else None
    vals = [v for v in out.values() if v is not None]
    out["range"] = round(max(vals) - min(vals), 3) if vals else None
    return out


def main():
    device = "cuda"
    ds = RawMemmapWikiArtDataset(MANIFEST, is_training=False)
    N = len(ds)

    styles, artists, genres = [], [], []
    for i in range(N):
        rec = ds.records_by_idx[ds.indices[i]]
        styles.append(rec["style_id"]); artists.append(rec.get("artist_id", -1))
        genres.append(rec.get("genre_id", -1))
    styles = np.array(styles); artists = np.array(artists); genres = np.array(genres)

    print("\n[Pairs] building standard + content-matched pairs...", flush=True)
    same, cross, diff, hard = build_pairs(styles, artists, genres)
    print(f"  same={len(same)} cross={len(cross)} diff={len(diff)} hard={len(hard)}", flush=True)

    print("\n[DINO] content features for matched pairs...", flush=True)
    dino = extract_dino(ds, device)
    cm_pos, cm_neg, cm_gaps = build_content_matched(dino, styles, artists, genres)
    print(f"  content-matched pairs={len(cm_pos)} gap_mean={cm_gaps.mean():.4f}", flush=True)

    print("\n[CSD] teacher reference features...", flush=True)
    csd_f = extract_csd(ds, device)
    csd_sim = csd_f @ csd_f.T

    rows = []
    models = {}
    for kind in ("B0", "A0", "V2"):
        print(f"\n=== Latents: {kind} ===", flush=True)
        models[kind] = extract_latents(kind, ds, device)

    # small subsample for scale invariance
    sub_idx = np.random.default_rng(0).choice(N, 400, replace=False)
    sub_styles = styles[sub_idx]

    for kind, z_all in models.items():
        z_sub = z_all[sub_idx]
        for dname, dfn in DESCRIPTORS.items():
            print(f"\n--- {kind} / {dname} ---", flush=True)
            feats = normalize_rows(apply_batch(dfn, z_all).numpy())
            dim = feats.shape[1]

            r_o = auroc_ci(feats, same, diff)
            r_c = auroc_ci(feats, cross, diff)
            r_h = auroc_ci(feats, cross, hard) if len(hard) >= 10 else None
            r_m = auroc_ci(feats, cm_pos, cm_neg)
            knn = knn_metrics(feats, styles)
            mp = map10(feats, styles)
            sp = spearman_vs(csd_sim, feats @ feats.T)

            si = scale_invariance(dfn, z_sub, sub_styles)

            row = {
                "model": kind, "descriptor": dname, "dim": dim,
                "overall": r_o["auroc"], "overall_lo": r_o["lo"], "overall_hi": r_o["hi"],
                "cross": r_c["auroc"], "cross_lo": r_c["lo"], "cross_hi": r_c["hi"],
                "hard": r_h["auroc"] if r_h else None,
                "hard_lo": r_h["lo"] if r_h else None,
                "hard_hi": r_h["hi"] if r_h else None,
                "content_matched": r_m["auroc"] if r_m else None,
                "cm_lo": r_m["lo"] if r_m else None,
                "cm_hi": r_m["hi"] if r_m else None,
                "knn1": knn["knn1"], "knn5": knn["knn5"], "knn10": knn["knn10"],
                "map10": mp, "spearman_vs_csd": sp,
                "scale_auroc_range": si["range"],
            }
            rows.append(row)
            print(f"  dim={dim} O={row['overall']} C={row['cross']} H={row['hard']} "
                  f"CM={row['content_matched']} kNN1={row['knn1']} mAP={mp} "
                  f"sp_csd={sp} scaleRange={si['range']}", flush=True)

    # CSD reference row
    r_o = auroc_ci(csd_f, same, diff)
    r_c = auroc_ci(csd_f, cross, diff)
    r_h = auroc_ci(csd_f, cross, hard)
    r_m = auroc_ci(csd_f, cm_pos, cm_neg)
    knn = knn_metrics(csd_f, styles)
    mp = map10(csd_f, styles)
    rows.append({
        "model": "TEACHER", "descriptor": "CSD_Style_CLS", "dim": csd_f.shape[1],
        "overall": r_o["auroc"], "overall_lo": r_o["lo"], "overall_hi": r_o["hi"],
        "cross": r_c["auroc"], "cross_lo": r_c["lo"], "cross_hi": r_c["hi"],
        "hard": r_h["auroc"], "hard_lo": r_h["lo"], "hard_hi": r_h["hi"],
        "content_matched": r_m["auroc"] if r_m else None,
        "cm_lo": r_m["lo"] if r_m else None, "cm_hi": r_m["hi"] if r_m else None,
        "knn1": knn["knn1"], "knn5": knn["knn5"], "knn10": knn["knn10"],
        "map10": mp, "spearman_vs_csd": 1.0, "scale_auroc_range": 0.0,
    })

    # write CSV
    cols = ["model", "descriptor", "dim", "overall", "overall_lo", "overall_hi",
            "cross", "cross_lo", "cross_hi", "hard", "hard_lo", "hard_hi",
            "content_matched", "cm_lo", "cm_hi", "knn1", "knn5", "knn10",
            "map10", "spearman_vs_csd", "scale_auroc_range"]
    lines = [",".join(cols)]
    for r in rows:
        lines.append(",".join("" if r[c] is None else str(r[c]) for c in cols))
    (OUT_DIR / "full_latent_representation_probe.csv").write_text("\n".join(lines), encoding="utf-8")
    (OUT_DIR / "full_latent_representation_probe.json").write_text(
        json.dumps({"pair_stats": {"n_content_matched": len(cm_pos),
                                   "content_gap_mean": round(float(cm_gaps.mean()), 4)},
                    "rows": rows}, indent=2), encoding="utf-8")

    print("\n" + "=" * 108)
    print("FULL-LATENT REPRESENTATION PROBE")
    print("=" * 108)
    hdr = f"{'model':<8}{'descriptor':<14}{'dim':>8}{'Overall':>9}{'Cross':>8}{'Hard':>8}{'CM':>8}{'kNN1':>7}{'mAP10':>8}{'sp_CSD':>8}{'scaleR':>8}"
    print(hdr); print("-" * 108)
    for r in rows:
        print(f"{r['model']:<8}{r['descriptor']:<14}{r['dim']:>8}"
              f"{r['overall']:>9.2f}{r['cross']:>8.2f}"
              f"{(r['hard'] if r['hard'] is not None else 0):>8.2f}"
              f"{(r['content_matched'] if r['content_matched'] is not None else 0):>8.2f}"
              f"{r['knn1']:>7.2f}{r['map10']:>8.2f}{r['spearman_vs_csd']:>8.3f}"
              f"{(r['scale_auroc_range'] if r['scale_auroc_range'] is not None else 0):>8.2f}")
    print("=" * 108)
    print(f"\n[Saved] {OUT_DIR / 'full_latent_representation_probe.csv'}")


if __name__ == "__main__":
    main()
