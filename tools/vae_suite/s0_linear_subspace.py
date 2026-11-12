"""S0: CPU-only linear style subspace screening.

Compares four orthogonal projections P (k x 16384), all solved by linear algebra,
no neural training:

  P0_random      random orthonormal (control)
  P1_csd_pls     SVD of cross-covariance Z^T T  (CSD-PLS)
  P2_genre_pls   genre-residualized CSD-PLS
  P3_fisher      PCA -> shrinkage LDA on Style labels

Evaluates DIRECT retrieval with u = P (z - mu_z), no extra readout head.
Also measures cross-artist subspace stability R = ||P1 P2^T||_F^2 / k
(random expectation k / 16384).
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "/root/Workspace/WEAVE_VAE")
sys.path.insert(0, "/root/Workspace/WEAVE_VAE/tools/12h")

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

OUT = Path("/root/Workspace/WEAVE_VAE/runs_vae/v2_diagnosis")
CACHE = OUT / "p1_cache"
DIM = 16384
KS = [128, 256, 768]
KS_FISHER = [64, 128, 256]


# ---------------------------------------------------------------- data
def load_cache(n_train=20000):
    z_tr = torch.load(CACHE / f"train_z_{n_train}.pt", weights_only=False)
    t_tr = torch.load(CACHE / f"train_t_{n_train}.pt", weights_only=False)
    m_tr = torch.load(CACHE / f"train_meta_{n_train}.pt", weights_only=False)
    z_va = torch.load(CACHE / "val_z.pt", weights_only=False)
    t_va = torch.load(CACHE / "val_t.pt", weights_only=False)
    m_va = torch.load(CACHE / "val_meta.pt", weights_only=False)
    return z_tr, t_tr, m_tr, z_va, t_va, m_va


def to_flat(z):
    return F.pixel_unshuffle(z, 4).flatten(1).numpy().astype(np.float32)


# ---------------------------------------------------------------- projections
def random_P(k, seed=0):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(DIM, k, generator=g)
    Q, _ = torch.linalg.qr(A)
    return Q.t().numpy().astype(np.float32)          # (k, DIM)


def pls_P(Zc, Tc, k):
    """Top-k left singular vectors of cross-covariance (DIM x tdim)."""
    C = (Zc.T @ Tc) / Zc.shape[0]                    # (DIM, tdim)
    U, S, Vt = np.linalg.svd(C, full_matrices=False)
    return U[:, :k].T.astype(np.float32), S[:k]


def genre_residualize(M, g):
    out = M.copy()
    for gv in np.unique(g):
        m = (g == gv)
        if m.sum() > 1:
            out[m] -= M[m].mean(0, keepdims=True)
    return out


def fisher_P(Z, y, k, pca_dim=768, shrink=0.1, seed=0):
    """PCA reduce -> shrinkage LDA; returns (k, DIM)."""
    mu = Z.mean(0, keepdims=True)
    Zc = Z - mu
    # randomized PCA
    Zt = torch.from_numpy(Zc)
    U, S, V = torch.svd_lowrank(Zt, q=pca_dim, niter=4)
    V = V.numpy().astype(np.float32)                 # (DIM, pca_dim)
    Y = (Zc @ V).astype(np.float32)                  # (N, pca_dim)
    classes = np.unique(y)
    d = Y.shape[1]
    gmean = Y.mean(0)
    Sw = np.zeros((d, d), np.float64)
    Sb = np.zeros((d, d), np.float64)
    N = Y.shape[0]
    for c in classes:
        m = (y == c)
        nc = int(m.sum())
        if nc < 2:
            continue
        mc = Y[m].mean(0)
        Yc = Y[m] - mc
        Sw += Yc.T @ Yc
        diff = (mc - gmean)
        Sb += nc * np.outer(diff, diff)
    Sw /= N
    Sb /= N
    tr = np.trace(Sw) / d
    Sw = (1 - shrink) * Sw + shrink * tr * np.eye(d)
    # top-k eigenvectors of inv(Sw) Sb
    M = np.linalg.solve(Sw, Sb)
    w, W = np.linalg.eig(M)
    order = np.argsort(-np.real(w))[:k]
    W = np.real(W[:, order]).astype(np.float32)
    P = W.T @ V.T                                    # (k, DIM)
    # re-orthonormalize rows (LDA vectors are not orthogonal a priori)
    Q, _ = np.linalg.qr(P.T)
    return Q.T.astype(np.float32)


def orth_err(P):
    k = P.shape[0]
    E = P @ P.T - np.eye(k, dtype=np.float32)
    return float(np.abs(E).max())


# ---------------------------------------------------------------- metrics
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
    pos, neg = [], []
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
    return pos, neg


def auroc_pairs(feats, pos, neg, n_boot=200, seed=42, cap=800):
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
        en = min(st + N, N) if False else min(st + block, N)
        sim = feats[st:en] @ feats.T
        for r in range(en - st):
            sim[r, st + r] = -np.inf
        order = np.argsort(-sim, axis=1)
        for r in range(en - st):
            for k in ks:
                if styles[st + r] in styles[order[r, :k]]:
                    hits[k] += 1
    return {f"knn{k}": round(hits[k] / N * 100, 2) for k in ks}


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


def stability(P1, P2):
    k = P1.shape[0]
    return float((np.linalg.norm(P1 @ P2.T, "fro") ** 2) / k), k / DIM


# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    print("=== S0: CPU linear style-subspace screening ===", flush=True)
    z_tr, t_tr, m_tr, z_va, t_va, m_va = load_cache()
    Ztr = to_flat(z_tr); Zva = to_flat(z_va)
    Ttr = t_tr.numpy().astype(np.float32); Tva = t_va.numpy().astype(np.float32)
    Ttr /= (np.linalg.norm(Ttr, axis=1, keepdims=True) + 1e-8)
    Tva /= (np.linalg.norm(Tva, axis=1, keepdims=True) + 1e-8)
    y_tr = m_tr["style"].numpy(); a_tr = m_tr["artist"].numpy(); g_tr = m_tr["genre"].numpy()
    y_va = m_va["style"].numpy(); a_va = m_va["artist"].numpy(); g_va = m_va["genre"].numpy()
    print(f"[Data] train {Ztr.shape} val {Zva.shape}", flush=True)

    mu_z = Ztr.mean(0, keepdims=True)
    Ztr_c = Ztr - mu_z
    Ttr_c = Ttr - Ttr.mean(0, keepdims=True)

    # content-matched pairs from cached DINO? recompute cheaply via latent-free proxy:
    # use CSD content? No -> we need DINO. Reuse saved dino if present.
    fdin = CACHE / "val_dino.pt"
    if fdin.exists():
        dino = torch.load(fdin, weights_only=False).numpy()
    else:
        print("[DINO] extracting val content features (GPU) ...", flush=True)
        from transformers import AutoModel
        from diffusers import AutoencoderKL
        from torch.utils.data import DataLoader
        from tools.vae_suite.dataset import RawMemmapWikiArtDataset
        MANIFEST = Path("/root/Workspace/data/wikiart_512_raw/wikiart_512_raw_manifest.json")
        ds = RawMemmapWikiArtDataset(MANIFEST, is_training=False)
        dev = "cuda"
        dm = AutoModel.from_pretrained("facebook/dinov2-base").to(dev).eval()
        nm = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
        ns = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
        dl = DataLoader(ds, batch_size=8, shuffle=False)
        df = []
        with torch.no_grad():
            for i, b in enumerate(dl):
                x = b["image"].to(dev)
                x01 = (x.clamp(-1, 1) + 1) / 2
                x224 = F.interpolate(x01, (224, 224), mode="bicubic", align_corners=False)
                o = dm(pixel_values=(x224 - nm) / ns)
                df.append(F.normalize(o.last_hidden_state[:, 0].float(), dim=-1).cpu())
        dino = torch.cat(df).numpy()
        del dm
        torch.cuda.empty_cache()
        torch.save(torch.from_numpy(dino), fdin)
    dino = dino / (np.linalg.norm(dino, axis=1, keepdims=True) + 1e-8)

    same, cross, diff, hard = build_pairs(y_va, a_va, g_va)
    cm_pos, cm_neg = build_cm(dino, y_va, a_va, g_va)
    print(f"[Pairs] cross={len(cross)} hard={len(hard)} cm={len(cm_pos)}", flush=True)

    def evaluate(P, tag, results):
        u_tr = (Ztr_c @ P.T)
        u_va = (Zva - mu_z) @ P.T
        un = u_va / (np.linalg.norm(u_va, axis=1, keepdims=True) + 1e-8)
        r_cm = auroc_pairs(un, cm_pos, cm_neg)
        r_cross = auroc_pairs(un, cross, diff)
        r_hard = auroc_pairs(un, cross, hard)
        k = knn_purity(un, y_va)
        mp = map10(un, y_va)
        tn = Tva
        tcos = float((un * tn).sum(1).mean())
        results[tag] = {
            "k": int(P.shape[0]),
            "content_matched": r_cm, "cross_artist": r_cross,
            "same_genre_hard": r_hard, "knn": k, "map10": mp,
            "teacher_cos": round(tcos, 4),
            "orth_err": orth_err(P),
        }
        print(f"  [{tag:<22} k={P.shape[0]:>4}] CM={r_cm['auroc'] if r_cm else None} "
              f"[{r_cm['lo'] if r_cm else ''},{r_cm['hi'] if r_cm else ''}] "
              f"Cross={r_cross['auroc'] if r_cross else None} "
              f"Hard={r_hard['auroc'] if r_hard else None} "
              f"kNN1={k['knn1']} mAP={mp} tCos={tcos:.3f} orthErr={results[tag]['orth_err']:.1e}",
              flush=True)

    results = {}

    # ---------- baselines ----------
    print("\n--- baselines ---", flush=True)
    evaluate(np.eye(DIM, dtype=np.float32)[:768], "raw_u0_first768", results)

    # ---------- candidates ----------
    print("\n--- P0 random ---", flush=True)
    for k in KS:
        P = random_P(k, seed=0)
        evaluate(P, f"P0_random_k{k}", results)

    print("\n--- P1 CSD-PLS ---", flush=True)
    U_full, S_full = None, None
    for k in KS:
        P, S = pls_P(Ztr_c, Ttr_c, k)
        if U_full is None:
            U_full, S_full = P, S
        evaluate(P, f"P1_csdpls_k{k}", results)
        print(f"    singular values top1={S[0]:.4e} top{k}={S[-1]:.4e}", flush=True)

    print("\n--- P2 Genre-controlled PLS ---", flush=True)
    Zr = genre_residualize(Ztr_c, g_tr)
    Tr = genre_residualize(Ttr_c, g_tr)
    for k in KS:
        P, S = pls_P(Zr, Tr, k)
        evaluate(P, f"P2_genrepls_k{k}", results)

    print("\n--- P3 Supervised Fisher ---", flush=True)
    for k in KS_FISHER:
        P = fisher_P(Ztr, y_tr, k, pca_dim=768, shrink=0.1)
        evaluate(P, f"P3_fisher_k{k}", results)

    # ---------- cross-artist stability ----------
    print("\n--- subspace stability (artist-disjoint halves) ---", flush=True)
    arts = np.unique(a_tr[a_tr >= 0])
    rng = np.random.default_rng(3)
    rng.shuffle(arts)
    half = len(arts) // 2
    m1 = np.isin(a_tr, arts[:half]); m2 = np.isin(a_tr, arts[half:])
    print(f"  half sizes: {m1.sum()} / {m2.sum()}", flush=True)

    stab = {}
    for k in [128, 256]:
        try:
            Pa, _ = pls_P(Ztr_c[m1], Ttr_c[m1], k)
            Pb, _ = pls_P(Ztr_c[m2], Ttr_c[m2], k)
            R, exp = stability(Pa, Pb)
            stab[f"P1_csdpls_k{k}"] = {"R": round(R, 5), "random_expectation": round(exp, 5),
                                       "ratio": round(R / exp, 2)}
            print(f"  P1_csdpls_k{k}: R={R:.5f} (random {exp:.5f}, {R/exp:.2f}x)", flush=True)
        except Exception as e:
            print(f"  P1 k={k} failed: {e}", flush=True)

        try:
            Za = genre_residualize(Ztr_c[m1], g_tr[m1]); Ta = genre_residualize(Ttr_c[m1], g_tr[m1])
            Zb = genre_residualize(Ztr_c[m2], g_tr[m2]); Tb = genre_residualize(Ttr_c[m2], g_tr[m2])
            Pa, _ = pls_P(Za, Ta, k)
            Pb, _ = pls_P(Zb, Tb, k)
            R, exp = stability(Pa, Pb)
            stab[f"P2_genrepls_k{k}"] = {"R": round(R, 5), "random_expectation": round(exp, 5),
                                         "ratio": round(R / exp, 2)}
            print(f"  P2_genrepls_k{k}: R={R:.5f} (random {exp:.5f}, {R/exp:.2f}x)", flush=True)
        except Exception as e:
            print(f"  P2 k={k} failed: {e}", flush=True)

    for k in [64, 128]:
        try:
            Pa = fisher_P(Ztr[m1], y_tr[m1], k, pca_dim=512, shrink=0.1)
            Pb = fisher_P(Ztr[m2], y_tr[m2], k, pca_dim=512, shrink=0.1)
            R, exp = stability(Pa, Pb)
            stab[f"P3_fisher_k{k}"] = {"R": round(R, 5), "random_expectation": round(exp, 5),
                                       "ratio": round(R / exp, 2)}
            print(f"  P3_fisher_k{k}: R={R:.5f} (random {exp:.5f}, {R/exp:.2f}x)", flush=True)
        except Exception as e:
            print(f"  P3 k={k} failed: {e}", flush=True)

    results["_stability"] = stab
    results["_meta"] = {
        "n_train": int(Ztr.shape[0]), "n_val": int(Zva.shape[0]),
        "elapsed_sec": round(time.time() - t0, 1),
        "baselines": {"A0_raw_CM": 51.65, "SmallCNN_CM": 53.39,
                      "SmallCNN_Cross": 61.40, "CSD_CM": 59.62, "CSD_Cross": 74.27},
    }
    (OUT / "S0_linear_subspace.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    # ---------- summary ----------
    print("\n" + "=" * 100)
    print("S0 SUMMARY  (CM = Content-Matched AUROC; target >= 56.0, SmallCNN baseline 53.39)")
    print("=" * 100)
    print(f"{'candidate':<24}{'k':>5}{'CM':>8}{'95% CI':>16}{'Cross':>8}{'Hard':>8}{'kNN@1':>8}{'mAP10':>8}{'tCos':>7}")
    print("-" * 100)
    for tag, v in results.items():
        if tag.startswith("_"):
            continue
        cm = v["content_matched"]
        ci = "[{},{}]".format(cm["lo"], cm["hi"])
        print(f"{tag:<24}{v['k']:>5}{cm['auroc']:>8.2f}{ci:>16}"
              f"{v['cross_artist']['auroc']:>8.2f}{v['same_genre_hard']['auroc']:>8.2f}"
              f"{v['knn']['knn1']:>8.2f}{v['map10']:>8.2f}{v['teacher_cos']:>7.3f}")
    print("=" * 100)
    print(f"\n[Saved] {OUT/'S0_linear_subspace.json'}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
