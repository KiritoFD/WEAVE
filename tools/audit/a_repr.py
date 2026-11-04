"""S1 + S3: does a latent representation carry real style information?

S1: intra/inter distances, 5-way classification, confusion matrix for nine
    frozen latent representations (raw / subbands / lowpass / patches /
    channel covariance / spectrum / local covariance).
S3: candidate *distances* (raw L2, moment, SWD over patch descriptors,
    MMD over global stats, covariance) scored by D_same / D_cross, top-1
    retrieval, one-vs-rest AUROC, and the hardest style pair.

All features are computed from the 4x64x64 latent only.  Gallery = ref split,
test = query split (disjoint).  Output: runs/audit/s1_repr.json, s3_distances.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import audit_core as C


def _bootstrap_groups(y: torch.Tensor, k: int = 5) -> torch.Tensor:
    """Content-free bootstrap unit fallback: per-style halves (10 groups)."""
    return y * 2 + (torch.arange(y.numel()) % 2)


def main() -> None:
    torch.manual_seed(0)
    ds = C.load_dataset()
    split = C.make_split(1000, n_ref=800, n_query=200, seed=0)
    z_ref = torch.stack([ds.latents[s][split.ref[s]] for s in range(5)])  # [5,800,4,64,64]
    z_q = torch.stack([ds.latents[s][split.query[s]] for s in range(5)])  # [5,200,...]
    y_ref = torch.arange(5).repeat_interleave(z_ref.shape[1])
    y_q = torch.arange(5).repeat_interleave(z_q.shape[1])

    out: dict = {"n_ref_per_style": 800, "n_query_per_style": 200, "representations": {}}

    for name, fn in C.REPRESENTATIONS.items():
        with torch.no_grad():
            fr = torch.cat([fn(z_ref[s]) for s in range(5)])       # [4000, D]
            fq = torch.cat([fn(z_q[s]) for s in range(5)])         # [1000, D]
        fr = fr.flatten(1).float()
        fq = fq.flatten(1).float()
        mu, sd = C.standardize(fr)
        tr = (fr - mu) / sd
        te = (fq - mu) / sd

        # pairwise distances -> intra / inter
        d_ref = torch.cdist(tr, tr)
        same = (y_ref[:, None] == y_ref[None, :])
        intra = d_ref[same & ~torch.eye(tr.shape[0], dtype=torch.bool)].mean()
        inter = d_ref[~same].mean()

        pred_nc, _ = C.nearest_centroid(fr, y_ref, fq, k=5)
        pred_lp, logits = C.linear_probe(fr, y_ref, fq, k=5)
        acc_nc = float((pred_nc == y_q).float().mean())
        acc_lp = float((pred_lp == y_q).float().mean())
        cm = C.confusion(pred_lp, y_q, 5)
        per_style = [float((pred_lp[y_q == c] == c).float().mean()) for c in range(5)]
        auroc = [C.auroc_ovr(logits, y_q, c) for c in range(5)]

        # effective rank (participation) of the standardised features
        s = torch.linalg.svdvals(tr)
        p = s.pow(2) / s.pow(2).sum()
        eff_rank = float(torch.exp(-(p * torch.log(p.clamp_min(1e-12))).sum()))

        out["representations"][name] = {
            "dim": int(fr.shape[1]),
            "intra_mean": float(intra),
            "inter_mean": float(inter),
            "inter_over_intra": float(inter / intra.clamp_min(1e-8)),
            "acc_nearest_centroid": acc_nc,
            "acc_linear_probe": acc_lp,
            "per_style_acc": dict(zip(C.SHORT, per_style)),
            "auroc_ovr": dict(zip(C.SHORT, auroc)),
            "confusion_linear_probe": cm,
            "effective_rank": eff_rank,
        }
        print(f"[S1] {name:12s} d={fr.shape[1]:4d} inter/intra={float(inter/intra):.3f} "
              f"acc_nc={acc_nc:.3f} acc_lp={acc_lp:.3f} effrank={eff_rank:.1f}")

    C.save_json("s1_repr.json", out)

    # ------------------------------------------------------------------ S3
    # candidate distances.  Descriptor sets from the same ref/query split.
    lib_patch = {s: C.feat_patch(z_ref[s]) for s in range(5)}         # [800,N,11]
    lib_glob = {s: C.feat_global(z_ref[s]) for s in range(5)}
    lib_cov = {s: C.feat_chan_cov(z_ref[s]) for s in range(5)}
    q_patch = {s: C.feat_patch(z_q[s]) for s in range(5)}
    q_glob = {s: C.feat_global(z_q[s]) for s in range(5)}

    # fixed projections for SWD, fitted once on ref descriptors (S11 measures variance)
    D = lib_patch[0].shape[-1]
    g = torch.Generator().manual_seed(0)
    proj = torch.randn(D, 64, generator=g)
    proj = proj / proj.norm(dim=0, keepdim=True)

    # standardisation for global-moment distances
    all_glob = torch.cat([lib_glob[s] for s in range(5)])
    gmu, gsd = C.standardize(all_glob)
    sub = torch.randperm(800, generator=torch.Generator().manual_seed(1))[:100]
    scale = gsd.mean()

    out3: dict = {"n_query_per_style": 200, "n_ref_per_style_sub": 100, "distances": {}}

    def per_query_global(metric) -> torch.Tensor:
        rows = []
        for i in range(5):
            for qi in range(200):
                a = q_glob[i][qi]
                row = [metric(a, lib_glob[j][sub]) for j in range(5)]
                rows.append(row)
        return torch.tensor(rows)

    # (a) standardised global-moment L2 (RMS in sigma units)
    def m_global(a, b):
        d = ((a - b) / gsd).pow(2)
        return float(d.mean().sqrt())
    M = per_query_global(m_global)
    out3["distances"]["global_moment_l2"] = _summarise_q(M, y_q, C.SHORT)

    # (b) SWD over patch descriptors (fixed projections)
    lib_flat = {j: lib_patch[j][sub].reshape(-1, D) for j in range(5)}
    M = torch.zeros(1000, 5)
    for i in range(5):
        n = q_patch[i].shape[1]
        for qi in range(200):
            a = q_patch[i][qi]
            M[i * 200 + qi] = torch.tensor(
                [float(C.swd2_proj(a, lib_flat[j], proj=proj)) for j in range(5)])
    out3["distances"]["latent_swd_patch"] = _summarise_q(M, y_q, C.SHORT)

    # (c) MMD over global stats
    def m_mmd(a, b):
        return float(C.mmd2(((a - gmu) / gsd).float().unsqueeze(0),
                            ((b - gmu) / gsd).float()))
    M = per_query_global(m_mmd)
    out3["distances"]["latent_mmd_global"] = _summarise_q(M, y_q, C.SHORT)

    # (d) channel covariance Frobenius (per-query cov vs style mean cov)
    cent = torch.stack([lib_cov[j][sub].mean(0) for j in range(5)])
    def m_cov(a, j):
        return float((C.feat_chan_cov(a) - cent[j]).pow(2).mean().sqrt())
    M = torch.zeros(1000, 5)
    for i in range(5):
        for qi in range(200):
            a = z_q[i][qi].unsqueeze(0)
            M[i * 200 + qi] = torch.tensor([m_cov(a, j) for j in range(5)])
    out3["distances"]["chan_cov"] = _summarise_q(M, y_q, C.SHORT)

    C.save_json("s3_distances.json", out3)


def _summarise_q(M: torch.Tensor, y_q: torch.Tensor, short) -> dict:
    """M: [1000, 5] per-query distance to each style set."""
    d_same = M[torch.arange(M.shape[0]), y_q]
    off = M.clone()
    off[torch.arange(M.shape[0]), y_q] = float("inf")
    d_cross = off.min(1).values
    pred = M.argmin(1)
    top1 = float((pred == y_q).float().mean())
    auroc = [C.auroc_ovr(-M, y_q, c) for c in range(5)]
    # hardest pair: highest rate of cross-style confusion
    pairs = {}
    for i in range(5):
        for j in range(5):
            if i != j:
                pairs[f"{short[i]}|{short[j]}"] = float(((pred[y_q == i] == j).float().mean()))
    return {
        "d_same_mean": float(d_same.mean()),
        "d_cross_min_mean": float(d_cross.mean()),
        "ratio_cross_over_same": float(d_cross.mean() / d_same.mean().clamp_min(1e-8)),
        "top1_retrieval": top1,
        "auroc_ovr": dict(zip(short, auroc)),
        "per_style_top1": {short[i]: float((pred[y_q == i] == i).float().mean()) for i in range(5)},
        "confusion": C.confusion(pred, y_q, 5),
        "hardest_pair": max(pairs.items(), key=lambda kv: kv[1]),
    }


if __name__ == "__main__":
    main()
