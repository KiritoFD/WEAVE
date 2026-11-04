"""S2: is style separability driven by *what is painted* (content)?

Uses the offline DINOv2-base token cache (224px, 16x16 patch tokens) as a
content descriptor that is independent of the 4-channel latent.

Tests
-----
1. content-neighbourhood style purity: for each query, take its top-k most
   DINO-content-similar images across all styles; how often same style?
2. style accuracy inside content clusters (KMeans k=25 on DINO tokens):
   train/test both restricted to one cluster.
3. author-held-out style classification on latent features (train on 70% of
   artists, test on unseen artists of the same styles).
4. semantic upper bound: style linear probe on DINO tokens themselves.
5. content-cluster-held-out style classification (latent features).

Output: runs/audit/s2_confound.json
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import audit_core as C


def load_dino_tokens(ds: C.Dataset) -> torch.Tensor:
    """[5,1000,768] mean-pooled DINO tokens, matched by file stem."""
    feats = []
    for s, style in enumerate(C.STYLES):
        d = C.DINO_CACHE / style
        idx = {p.stem: p for p in d.glob("*.pt")}
        rows = []
        for name in ds.names[s]:
            stem = Path(name.replace("\\", "/")).stem
            if stem not in idx:
                raise FileNotFoundError(f"missing DINO token for {stem}")
            tok = torch.load(idx[stem], map_location="cpu", weights_only=False)["tokens"].float()
            rows.append(tok.mean(0))
        feats.append(torch.stack(rows))
    return torch.stack(feats)


def kmeans(x: torch.Tensor, k: int, *, iters: int = 25, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    n = x.shape[0]
    cent = x[torch.randperm(n, generator=g)[:k]].clone()
    for _ in range(iters):
        d = torch.cdist(x, cent)
        lab = d.argmin(1)
        for c in range(k):
            m = lab == c
            if m.any():
                cent[c] = x[m].mean(0)
    return lab


def main() -> None:
    torch.manual_seed(0)
    ds = C.load_dataset()
    split = C.make_split(1000, n_ref=800, n_query=200, seed=0)
    dino = load_dino_tokens(ds)                       # [5,1000,768]
    dn = F.normalize(dino.reshape(5000, -1).float(), dim=-1)
    y = torch.arange(5).repeat_interleave(1000)

    # latent features used for the style probes (subband = best S1 representation)
    z = ds.latents.reshape(5000, 4, 64, 64)
    xlat = F.normalize(C.feat_subband(z).float(), dim=-1)

    ref_mask = torch.zeros(5, 1000, dtype=torch.bool)
    q_mask = torch.zeros(5, 1000, dtype=torch.bool)
    for s in range(5):
        ref_mask[s, split.ref[s]] = True
        q_mask[s, split.query[s]] = True
    ref_mask = ref_mask.reshape(-1)
    q_mask = q_mask.reshape(-1)

    out: dict = {}

    # 1. content-neighbourhood style purity (DINO content vs chance 0.2)
    cos = dn @ dn.T
    cos.fill_diagonal_(-2.0)
    topk = cos.topk(50, dim=1).indices
    purity = (y[topk] == y[:, None]).float().mean(1)
    out["content_neighbour_style_purity"] = {
        "mean": float(purity.mean()),
        "per_style": {C.SHORT[s]: float(purity[y == s].mean()) for s in range(5)},
        "chance": 0.2,
    }

    # 2. content clusters + style composition
    groups = kmeans(dn[ref_mask], 25, seed=0)          # fit on ref only
    # assign every image to nearest fitted corpus centroid
    cent = torch.stack([dn[ref_mask][groups == c].mean(0) for c in range(25)])
    lab = torch.cdist(dn, cent).argmin(1)
    comp = []
    for c in range(25):
        m = lab == c
        if int(m.sum()) < 20:
            continue
        cnt = [int(((y[m] == s)).sum()) for s in range(5)]
        comp.append({"cluster": c, "n": int(m.sum()), "counts": cnt,
                     "dominant_style": C.SHORT[int(torch.tensor(cnt).argmax())]})
    out["content_clusters"] = {
        "k": 25,
        "n_valid": len(comp),
        "dominant_style_share": float(torch.tensor([max(c["counts"]) / c["n"] for c in comp]).mean()),
        "per_cluster": comp,
    }

    # 3. within-cluster style accuracy (latent subband probe, trained per cluster)
    accs = []
    for c in range(25):
        tr = (lab == c) & ref_mask
        te = (lab == c) & q_mask
        ntr_styles = torch.unique(y[tr])
        nte_styles = torch.unique(y[te])
        if len(ntr_styles) < 3 or len(nte_styles) < 3 or int(te.sum()) < 15:
            continue
        pred, _ = C.linear_probe(xlat[tr], y[tr], xlat[te], k=5, iters=200)
        # evaluate only on styles present in training
        keep = torch.isin(y[te], ntr_styles)
        if int(keep.sum()) < 10:
            continue
        accs.append(float((pred[keep] == y[te][keep]).float().mean()))
    out["within_content_cluster_style_acc"] = {
        "n_clusters_used": len(accs),
        "mean": float(torch.tensor(accs).mean()) if accs else None,
        "chance_k5": 0.2,
        "values": accs,
    }

    # 4. semantic upper bound: linear probe on DINO tokens
    pred, _ = C.linear_probe(dn[ref_mask], y[ref_mask], dn[q_mask], k=5, iters=300)
    out["dino_style_probe"] = {
        "acc": float((pred == y[q_mask]).float().mean()),
        "per_style": {C.SHORT[s]: float((pred[y[q_mask] == s] == s).float().mean()) for s in range(5)},
    }

    # 5. latent subband probe (reference number for comparison)
    pred_lat, _ = C.linear_probe(xlat[ref_mask], y[ref_mask], xlat[q_mask], k=5, iters=300)
    out["latent_subband_probe"] = {
        "acc": float((pred_lat == y[q_mask]).float().mean()),
        "per_style": {C.SHORT[s]: float((pred_lat[y[q_mask] == s] == s).float().mean()) for s in range(5)},
    }

    # 6. author-held-out split (latent features)
    artist_ids = torch.zeros(5000, dtype=torch.long)
    for s in range(5):
        vocab = {a: i for i, a in enumerate(sorted(set(ds.artists[s])))}   # stable ids
        base = s * 1000
        for i, a in enumerate(ds.artists[s]):
            artist_ids[base + i] = vocab[a]
    g = torch.Generator().manual_seed(0)
    tr_m = torch.zeros(5000, dtype=torch.bool)
    for s in range(5):
        base = s * 1000
        ua = torch.unique(artist_ids[base:base + 1000])
        perm = ua[torch.randperm(ua.numel(), generator=g)]
        cut = max(1, int(0.7 * ua.numel()))
        tr_a = set(perm[:cut].tolist())
        for i in range(1000):
            tr_m[base + i] = int(artist_ids[base + i]) in tr_a
    te_m = ref_mask & ~tr_m
    pred, _ = C.linear_probe(xlat[tr_m], y[tr_m], xlat[te_m], k=5, iters=300)
    out["author_heldout_subband_probe"] = {
        "acc": float((pred == y[te_m]).float().mean()),
        "per_style": {C.SHORT[s]: float((pred[y[te_m] == s] == s).float().mean()) for s in range(5)},
        "n_train": int(tr_m.sum()), "n_test": int(te_m.sum()),
    }
    # same split, DINO upper bound
    pred, _ = C.linear_probe(dn[tr_m], y[tr_m], dn[te_m], k=5, iters=300)
    out["author_heldout_dino_probe"] = {
        "acc": float((pred == y[te_m]).float().mean()),
        "per_style": {C.SHORT[s]: float((pred[y[te_m] == s] == s).float().mean()) for s in range(5)},
    }

    # 7. content-cluster-held-out (latent + dino)
    ucl = torch.unique(lab)
    g2 = torch.Generator().manual_seed(0)
    hold = ucl[torch.randperm(ucl.numel(), generator=g2)[:int(0.3 * ucl.numel())]]
    te_m = torch.isin(lab, hold) & q_mask
    tr_m = ref_mask & ~torch.isin(lab, hold)
    pred, _ = C.linear_probe(xlat[tr_m], y[tr_m], xlat[te_m], k=5, iters=300)
    out["content_cluster_heldout_subband"] = {
        "acc": float((pred == y[te_m]).float().mean()), "n_test": int(te_m.sum()),
        "per_style": {C.SHORT[s]: float((pred[y[te_m] == s] == s).float().mean()) for s in range(5)},
    }
    pred, _ = C.linear_probe(dn[tr_m], y[tr_m], dn[te_m], k=5, iters=300)
    out["content_cluster_heldout_dino"] = {
        "acc": float((pred == y[te_m]).float().mean()),
        "per_style": {C.SHORT[s]: float((pred[y[te_m] == s] == s).float().mean()) for s in range(5)},
    }

    C.save_json("s2_confound.json", out)
    print({k: v for k, v in out.items() if not isinstance(v, dict) or len(str(v)) < 300})


if __name__ == "__main__":
    main()
