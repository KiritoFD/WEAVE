"""CSD Dual-Head Diagnostic: are style and content heads actually different?

Measures on 2048 validation images:
1. Per-image cosine between style vector and content vector (768-d each).
2. CKA and Spearman correlation between the two similarity matrices.
3. Error-sample overlap on Cross-Artist and Same-Genre-Hard pairs.
4. Whether the two heads' pairwise rankings agree.

Interpretation:
  - If per-image cosine ~1.0 and CKA ~1.0 -> likely implementation bug OR
    the shared backbone dominates and the two projection heads barely diverge.
  - If per-image cosine moderate but CKA ~1.0 -> heads differ per-image but
    preserve the same relational geometry (no disentanglement).
  - If CKA low but AUROCs similar -> genuinely different geometry that happens
    to score similarly.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/Workspace/WEAVE_VAE")
sys.path.insert(0, "/root/Workspace/WEAVE_VAE/tools/12h")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tools.vae_suite.dataset import RawMemmapWikiArtDataset
import csd_model

OUT = Path("/root/Workspace/WEAVE_VAE/runs_vae/v2_diagnosis/14_csd_dual_head.json")
MANIFEST = Path("/root/Workspace/data/wikiart_512_raw/wikiart_512_raw_manifest.json")
NORM_MEAN = [0.48145466, 0.4578275, 0.40821073]
NORM_STD = [0.26862954, 0.26130258, 0.27577711]


def linear_cka(X, Y):
    """Linear CKA between two (N, D) representations."""
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    num = np.linalg.norm(X.T @ Y, ord="fro") ** 2
    den = np.linalg.norm(X.T @ X, ord="fro") * np.linalg.norm(Y.T @ Y, ord="fro")
    return float(num / (den + 1e-12))


def spearman(a, b):
    """Spearman correlation between two flattened vectors."""
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    return float((ra @ rb) / (np.linalg.norm(ra) * np.linalg.norm(rb) + 1e-12))


def main():
    device = "cuda"
    print("=== CSD Dual-Head Diagnostic ===", flush=True)
    ds = RawMemmapWikiArtDataset(MANIFEST, is_training=False)
    loader = DataLoader(ds, batch_size=16, shuffle=False)

    csd, _, _ = csd_model.load_csd(device)
    csd.eval()

    nm = torch.tensor(NORM_MEAN, device=device).view(1, 3, 1, 1)
    ns = torch.tensor(NORM_STD, device=device).view(1, 3, 1, 1)

    sty, cnt, styles, artists, genres = [], [], [], [], []
    with torch.no_grad():
        for i, b in enumerate(loader):
            x = b["image"].to(device)
            x01 = (x.clamp(-1, 1) + 1.0) / 2.0
            x224 = F.interpolate(x01, (224, 224), mode="bicubic", align_corners=False)
            xin = ((x224 - nm) / ns).half()
            _, c, s = csd(xin)
            sty.append(s.float().cpu().numpy())
            cnt.append(c.float().cpu().numpy())
            styles.append(b["style_id"].numpy())
            artists.append(b["artist_id"].numpy())
            genres.append(b["genre_id"].numpy())
            if (i + 1) % 40 == 0:
                print(f"  [{i+1}/{len(loader)}]", flush=True)

    S = np.concatenate(sty)          # (N, 768) already L2-normalized by csd_model
    C = np.concatenate(cnt)
    y = np.concatenate(styles)
    a = np.concatenate(artists)
    g = np.concatenate(genres)
    N = len(y)
    print(f"[Data] N={N}, style dim={S.shape[1]}, content dim={C.shape[1]}", flush=True)

    # ---- 1. per-image cosine between heads ----
    per_img_cos = (S * C).sum(1)
    print(f"\n[1] Per-image cosine(style, content): "
          f"mean={per_img_cos.mean():.4f}, min={per_img_cos.min():.4f}, "
          f"max={per_img_cos.max():.4f}, std={per_img_cos.std():.4f}", flush=True)

    # raw norms
    print(f"    |style| mean={np.linalg.norm(S,axis=1).mean():.4f}, "
          f"|content| mean={np.linalg.norm(C,axis=1).mean():.4f}", flush=True)

    # ---- 2. CKA / Spearman between similarity matrices ----
    Ssim = S @ S.T
    Csim = C @ C.T
    iu = np.triu_indices(N, k=1)
    cka = linear_cka(S, C)
    sp_mat = spearman(Ssim[iu], Csim[iu])
    print(f"\n[2] CKA(style, content) = {cka:.4f}", flush=True)
    print(f"    Spearman(S_ij, C_ij) = {sp_mat:.4f}", flush=True)

    # top-k ranking agreement
    k = 10
    agree = []
    for i in range(min(500, N)):
        rs = np.argsort(-Ssim[i])[:k]
        rc = np.argsort(-Csim[i])[:k]
        agree.append(len(set(rs) & set(rc)) / k)
    top10_agree = float(np.mean(agree))
    print(f"    Top-10 neighbour overlap = {top10_agree*100:.1f}%", flush=True)

    # ---- 3. error overlap on hard pairs ----
    rng = np.random.default_rng(42)
    cross, diff, hard = [], [], []
    for i in range(N):
        sc = np.where(y == y[i])[0]
        sc = sc[sc != i]
        if len(sc) and a[i] >= 0:
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

    def pair_sims(M, pairs):
        p = np.array(pairs)
        return (M[p[:, 0]] * M[p[:, 1]]).sum(1)

    results = {}
    for name, pairs in [("cross_artist", cross), ("same_genre_hard", hard)]:
        n = min(len(pairs), len(diff))
        pp = pairs[:n]
        np_ = diff[:n]
        ss_pos = pair_sims(S, pp); ss_neg = pair_sims(S, np_)
        cc_pos = pair_sims(C, pp); cc_neg = pair_sims(C, np_)
        # an "error" = positive pair has lower sim than its matched negative
        err_s = ss_pos < ss_neg
        err_c = cc_pos < cc_neg
        overlap = float((err_s & err_c).mean())
        jacc = float((err_s & err_c).sum() / max((err_s | err_c).sum(), 1))
        results[name] = {
            "n_pairs": int(n),
            "style_error_rate": round(float(err_s.mean()), 4),
            "content_error_rate": round(float(err_c.mean()), 4),
            "error_overlap_rate": round(overlap, 4),
            "error_jaccard": round(jacc, 4),
            "mean_sim_pos_style": round(float(ss_pos.mean()), 4),
            "mean_sim_pos_content": round(float(cc_pos.mean()), 4),
        }
        print(f"\n[3] {name}: n={n} | style err={err_s.mean()*100:.1f}% "
              f"| content err={err_c.mean()*100:.1f}% | overlap={overlap*100:.1f}% "
              f"| Jaccard={jacc*100:.1f}%", flush=True)

    # ---- verdict ----
    if per_img_cos.mean() > 0.99 and cka > 0.99:
        verdict = "IDENTICAL: heads output near-identical vectors -> likely weight-mapping bug OR shared-backbone dominance"
    elif cka > 0.95:
        verdict = "NO DISENTANGLEMENT: heads differ per-image but relational geometry is nearly identical"
    elif cka > 0.85:
        verdict = "WEAK DISENTANGLEMENT: geometry largely shared"
    else:
        verdict = "GENUINE DISENTANGLEMENT: heads encode different relational geometry"

    out = {
        "n_images": N,
        "per_image_cosine": {
            "mean": round(float(per_img_cos.mean()), 4),
            "min": round(float(per_img_cos.min()), 4),
            "max": round(float(per_img_cos.max()), 4),
            "std": round(float(per_img_cos.std()), 4),
        },
        "cka_style_vs_content": round(cka, 4),
        "spearman_simmat": round(sp_mat, 4),
        "top10_neighbour_overlap": round(top10_agree, 4),
        "error_analysis": results,
        "verdict": verdict,
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print("VERDICT:", verdict)
    print("=" * 70)
    print(f"[Saved] {OUT}")


if __name__ == "__main__":
    main()
