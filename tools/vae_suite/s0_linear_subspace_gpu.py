"""S0 (GPU): linear style-subspace screening with cuBLAS/cuSOLVER acceleration.

Four orthogonal projections P (k x 16384), solved by linear algebra on GPU:
  P0_random, P1_csd_pls, P2_genre_pls, P3_fisher (PCA->shrinkage LDA)

Primary metric: DIRECT retrieval with u = P(z - mu_z), no readout head.
Also: cross-artist subspace stability R = ||P1 P2^T||_F^2 / k  (random = k/16384).
"""
from __future__ import annotations

import json
import sys
import time
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
DEV = "cuda"
KS = [128, 256, 768]
KS_FISHER = [64, 128, 256]


def load_cache(n_train=20000):
    z_tr = torch.load(CACHE / f"train_z_{n_train}.pt", weights_only=False)
    t_tr = torch.load(CACHE / f"train_t_{n_train}.pt", weights_only=False)
    m_tr = torch.load(CACHE / f"train_meta_{n_train}.pt", weights_only=False)
    z_va = torch.load(CACHE / "val_z.pt", weights_only=False)
    t_va = torch.load(CACHE / "val_t.pt", weights_only=False)
    m_va = torch.load(CACHE / "val_meta.pt", weights_only=False)
    return z_tr, t_tr, m_tr, z_va, t_va, m_va


def to_flat_gpu(z, device=DEV):
    return F.pixel_unshuffle(z, 4).flatten(1).to(device).float()


def random_P(k, device=DEV, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    A = torch.randn(DIM, k, generator=g).to(device)
    Q, _ = torch.linalg.qr(A)
    return Q.t().contiguous()


def pls_P(Zc, Tc, k):
    C = (Zc.t() @ Tc) / Zc.shape[0]                 # (DIM, tdim)
    U, S, Vh = torch.linalg.svd(C, full_matrices=False)
    return U[:, :k].t().contiguous(), S[:k]


def genre_residualize(M, g):
    out = M.clone()
    for gv in torch.unique(g):
        m = (g == gv)
        if int(m.sum()) > 1:
            out[m] -= M[m].mean(0, keepdim=True)
    return out


def fisher_P(Z, y, k, pca_dim=768, shrink=0.1):
    mu = Z.mean(0, keepdim=True)
    Zc = Z - mu
    U, S, V = torch.svd_lowrank(Zc, q=pca_dim, niter=4)
    V = V.contiguous()                              # (DIM, pca_dim)
    Y = Zc @ V                                      # (N, pca_dim)
    d = Y.shape[1]
    gmean = Y.mean(0)
    Sw = torch.zeros(d, d, device=Y.device)
    Sb = torch.zeros(d, d, device=Y.device)
    N = Y.shape[0]
    for c in torch.unique(y):
        m = (y == c)
        nc = int(m.sum())
        if nc < 2:
            continue
        Yc = Y[m]
        mc = Yc.mean(0)
        Zc_ = Yc - mc
        Sw += Zc_.t() @ Zc_
        diff = mc - gmean
        Sb += nc * torch.outer(diff, diff)
    Sw /= N; Sb /= N
    tr = torch.trace(Sw) / d
    Sw = (1 - shrink) * Sw + shrink * tr * torch.eye(d, device=Y.device)
    M = torch.linalg.solve(Sw, Sb)
    w, W = torch.linalg.eig(M)
    w = w.real
    order = torch.argsort(-w)[:k]
    W = W[:, order].real.float()
    P = W.t() @ V.t()
    Q, _ = torch.linalg.qr(P.t())
    return Q.t().contiguous()


def orth_err(P):
    k = P.shape[0]
    E = P @ P.t() - torch.eye(k, device=P.device)
    return float(E.abs().max().item())


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


def auroc_pairs(feats_t, pos, neg, n_boot=200, seed=42, cap=800):
    """feats_t: torch tensor (N,k) already L2-normalized on GPU."""
    n = min(len(pos), len(neg))
    if n < 10:
        return None
    n = min(n, cap)
    sp = torch.tensor(np.array(pos[:n]), device=feats_t.device)
    sn = torch.tensor(np.array(neg[:n]), device=feats_t.device)
    a1 = (feats_t[sp[:, 0]] * feats_t[sp[:, 1]]).sum(1).cpu().numpy()
    a0 = (feats_t[sn[:, 0]] * feats_t[sn[:, 1]]).sum(1).cpu().numpy()
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


def knn_map(feats_t, styles, ks=(1, 5, 10), n_q_map=200, seed=42):
    N = feats_t.shape[0]
    sim = feats_t @ feats_t.t()
    sim.fill_diagonal_(-float("inf"))
    order = torch.topk(sim, k=max(ks), dim=1).indices.cpu().numpy()
    st = np.asarray(styles)
    knn = {f"knn{k}": round(float((st[order[:, :k]] == st[:, None]).mean()) * 100, 2) for k in ks}

    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)
    q, g = idx[:n_q_map], idx[n_q_map:]
    gs = st[g]
    aps = []
    for qi in q:
        sims = (feats_t[g] @ feats_t[qi]).cpu().numpy()
        top = np.argsort(-sims)[:10]
        rel = (gs[top] == st[qi]).astype(float)
        if rel.sum() == 0:
            aps.append(0.0); continue
        prec = np.cumsum(rel) / (np.arange(len(rel)) + 1)
        aps.append(float((prec * rel).sum() / rel.sum()))
    return knn, round(float(np.mean(aps) * 100), 2)


def stability(P1, P2):
    k = P1.shape[0]
    return float((torch.linalg.norm(P1 @ P2.t(), "fro") ** 2).item() / k), k / DIM


def main():
    t0 = time.time()
    print(f"=== S0 (GPU) linear subspace screening | dev={torch.cuda.get_device_name(0)} ===", flush=True)
    z_tr, t_tr, m_tr, z_va, t_va, m_va = load_cache()

    Ztr = to_flat_gpu(z_tr); Zva = to_flat_gpu(z_va)
    Ttr = F.normalize(t_tr.to(DEV).float(), dim=-1); Tva = F.normalize(t_va.to(DEV).float(), dim=-1)
    y_tr = m_tr["style"].to(DEV); a_tr = m_tr["artist"]; g_tr = m_tr["genre"].to(DEV)
    y_va_np = m_va["style"].numpy(); a_va_np = m_va["artist"].numpy(); g_va_np = m_va["genre"].numpy()
    print(f"[Data] train {tuple(Ztr.shape)} val {tuple(Zva.shape)} | "
          f"VRAM {torch.cuda.memory_allocated()/1024**2:.0f}M", flush=True)

    mu_z = Ztr.mean(0, keepdim=True)
    Ztr_c = Ztr - mu_z
    Ttr_c = Ttr - Ttr.mean(0, keepdim=True)

    # DINO for content matching (cached)
    fdin = CACHE / "val_dino.pt"
    if fdin.exists():
        dino = torch.load(fdin, weights_only=False).numpy()
    else:
        print("[DINO] extracting ...", flush=True)
        from transformers import AutoModel
        from torch.utils.data import DataLoader
        from tools.vae_suite.dataset import RawMemmapWikiArtDataset
        MANIFEST = Path("/root/Workspace/data/wikiart_512_raw/wikiart_512_raw_manifest.json")
        ds = RawMemmapWikiArtDataset(MANIFEST, is_training=False)
        dm = AutoModel.from_pretrained("facebook/dinov2-base").to(DEV).eval()
        nm = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
        ns = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)
        dl = DataLoader(ds, batch_size=8, shuffle=False)
        df = []
        with torch.no_grad():
            for i, b in enumerate(dl):
                x = b["image"].to(DEV)
                x01 = (x.clamp(-1, 1) + 1) / 2
                x224 = F.interpolate(x01, (224, 224), mode="bicubic", align_corners=False)
                o = dm(pixel_values=(x224 - nm) / ns)
                df.append(F.normalize(o.last_hidden_state[:, 0].float(), dim=-1).cpu())
        dino = torch.cat(df).numpy()
        del dm
        torch.cuda.empty_cache()
        torch.save(torch.from_numpy(dino), fdin)
    dino = dino / (np.linalg.norm(dino, axis=1, keepdims=True) + 1e-8)

    same, cross, diff, hard = build_pairs(y_va_np, a_va_np, g_va_np)
    cm_pos, cm_neg = build_cm(dino, y_va_np, a_va_np, g_va_np)
    print(f"[Pairs] cross={len(cross)} hard={len(hard)} cm={len(cm_pos)}", flush=True)

    results = {}

    def evaluate(P, tag):
        u = (Zva - mu_z) @ P.t()
        un = F.normalize(u, dim=-1)
        r_cm = auroc_pairs(un, cm_pos, cm_neg)
        r_cross = auroc_pairs(un, cross, diff)
        r_hard = auroc_pairs(un, cross, hard)
        knn, mp = knn_map(un, y_va_np)
        tcos = float((un * Tva).sum(1).mean().item()) if un.shape[1] == Tva.shape[1] else None
        results[tag] = {
            "k": int(P.shape[0]), "content_matched": r_cm, "cross_artist": r_cross,
            "same_genre_hard": r_hard, "knn": knn, "map10": mp,
            "teacher_cos": None if tcos is None else round(tcos, 4), "orth_err": orth_err(P),
        }
        ts = "n/a" if tcos is None else f"{tcos:.3f}"
        print(f"  [{tag:<22} k={P.shape[0]:>4}] CM={r_cm['auroc'] if r_cm else None} "
              f"[{r_cm['lo'] if r_cm else ''},{r_cm['hi'] if r_cm else ''}] "
              f"Cross={r_cross['auroc'] if r_cross else None} "
              f"Hard={r_hard['auroc'] if r_hard else None} kNN1={knn['knn1']} "
              f"mAP={mp} tCos={ts} orthErr={results[tag]['orth_err']:.1e}", flush=True)

    print("\n--- baseline: raw first 768 dims ---", flush=True)
    evaluate(torch.eye(DIM, device=DEV)[:768], "raw_u0_first768")

    print("\n--- P0 random ---", flush=True)
    for k in KS:
        evaluate(random_P(k), f"P0_random_k{k}")

    print("\n--- P1 CSD-PLS ---", flush=True)
    for k in KS:
        P, S = pls_P(Ztr_c, Ttr_c, k)
        evaluate(P, f"P1_csdpls_k{k}")
        print(f"     sv: top={S[0].item():.3e} kth={S[-1].item():.3e}", flush=True)

    print("\n--- P2 genre-controlled PLS ---", flush=True)
    Zr = genre_residualize(Ztr_c, g_tr)
    Tr = genre_residualize(Ttr_c, g_tr)
    for k in KS:
        P, S = pls_P(Zr, Tr, k)
        evaluate(P, f"P2_genrepls_k{k}")

    print("\n--- P3 supervised Fisher ---", flush=True)
    for k in KS_FISHER:
        P = fisher_P(Ztr, y_tr, k, pca_dim=768, shrink=0.1)
        evaluate(P, f"P3_fisher_k{k}")

    # ---- stability ----
    print("\n--- cross-artist subspace stability ---", flush=True)
    arts = np.unique(a_tr.numpy()[a_tr.numpy() >= 0])
    rng = np.random.default_rng(3); rng.shuffle(arts)
    half = len(arts) // 2
    m1 = torch.tensor(np.isin(a_tr.numpy(), arts[:half]), device=DEV)
    m2 = torch.tensor(np.isin(a_tr.numpy(), arts[half:]), device=DEV)
    print(f"  halves: {int(m1.sum())} / {int(m2.sum())}", flush=True)

    stab = {}
    for k in [128, 256]:
        Pa, _ = pls_P(Ztr_c[m1], Ttr_c[m1], k)
        Pb, _ = pls_P(Ztr_c[m2], Ttr_c[m2], k)
        R, exp = stability(Pa, Pb)
        stab[f"P1_csdpls_k{k}"] = {"R": round(R, 5), "random": round(exp, 5), "ratio": round(R / exp, 2)}
        print(f"  P1_csdpls_k{k}: R={R:.5f} random={exp:.5f} -> {R/exp:.2f}x", flush=True)

        Za = genre_residualize(Ztr_c[m1], g_tr[m1]); Ta = genre_residualize(Ttr_c[m1], g_tr[m1])
        Zb = genre_residualize(Ztr_c[m2], g_tr[m2]); Tb = genre_residualize(Ttr_c[m2], g_tr[m2])
        Pa, _ = pls_P(Za, Ta, k); Pb, _ = pls_P(Zb, Tb, k)
        R, exp = stability(Pa, Pb)
        stab[f"P2_genrepls_k{k}"] = {"R": round(R, 5), "random": round(exp, 5), "ratio": round(R / exp, 2)}
        print(f"  P2_genrepls_k{k}: R={R:.5f} random={exp:.5f} -> {R/exp:.2f}x", flush=True)

    for k in [64, 128]:
        Pa = fisher_P(Ztr[m1], y_tr[m1], k, pca_dim=512, shrink=0.1)
        Pb = fisher_P(Ztr[m2], y_tr[m2], k, pca_dim=512, shrink=0.1)
        R, exp = stability(Pa, Pb)
        stab[f"P3_fisher_k{k}"] = {"R": round(R, 5), "random": round(exp, 5), "ratio": round(R / exp, 2)}
        print(f"  P3_fisher_k{k}: R={R:.5f} random={exp:.5f} -> {R/exp:.2f}x", flush=True)

    results["_stability"] = stab
    results["_meta"] = {
        "n_train": int(Ztr.shape[0]), "n_val": int(Zva.shape[0]),
        "elapsed_sec": round(time.time() - t0, 1),
        "baselines": {"A0_raw_CM": 51.65, "SmallCNN_CM": 53.39,
                      "SmallCNN_Cross": 61.40, "CSD_CM": 59.62, "CSD_Cross": 74.27},
    }
    (OUT / "S0_linear_subspace.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n" + "=" * 100)
    print("S0 SUMMARY (CM = Content-Matched AUROC; target >= 56.0; SmallCNN baseline 53.39)")
    print("=" * 100)
    print(f"{'candidate':<24}{'k':>5}{'CM':>8}{'95% CI':>16}{'Cross':>8}{'Hard':>8}{'kNN@1':>8}{'mAP10':>8}{'tCos':>7}")
    print("-" * 100)
    for tag, v in results.items():
        if tag.startswith("_"):
            continue
        cm = v["content_matched"]
        ci = "[{},{}]".format(cm["lo"], cm["hi"])
        tc = v.get("teacher_cos")
        ts = "n/a" if tc is None else f"{tc:.3f}"
        print(f"{tag:<24}{v['k']:>5}{cm['auroc']:>8.2f}{ci:>16}"
              f"{v['cross_artist']['auroc']:>8.2f}{v['same_genre_hard']['auroc']:>8.2f}"
              f"{v['knn']['knn1']:>8.2f}{v['map10']:>8.2f}{ts:>7}")
    print("=" * 100)
    print(f"\n[Saved] {OUT/'S0_linear_subspace.json'}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
