"""WEAVE-VAE V2 Diagnosis: P1 Unified Geometry Evaluation.

Compares B0 (original SD-VAE), A0, and V2 checkpoints on the SAME 2048 val set,
SAME SWD descriptor, SAME query/pair sets.

Metrics:
- Overall AUROC (same vs diff style)
- Cross-Artist AUROC (same style, different artist as positive)
- Same-Genre Hard AUROC (diff style, same genre as negative)
- Style Retrieval mAP@10
- kNN@1/5/10
- Intra/Inter mean distance
- Relative Margin M_rel = (d_neg - d_pos) / (d_neg + d_pos + eps)
- Bootstrap 95% CI (clustered by source image)
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "/root/Workspace/WEAVE_VAE")

import numpy as np
import torch
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from tools.vae_suite.dataset import RawMemmapWikiArtDataset
from tools.vae_suite.swd_metric import LatentSWDDescriptor

OUT_DIR = Path("/root/Workspace/WEAVE_VAE/runs_vae/v2_diagnosis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST = Path("/root/Workspace/data/wikiart_512_raw/wikiart_512_raw_manifest.json")
CALIB = Path("/root/Workspace/WEAVE_VAE/swd_calibration.json")

MODELS = [
    ("B0_original", "/root/Workspace/xy/DiT/data/pretrained/sd-vae-ft-ema", None),
    ("A0_recon_kl", "/root/Workspace/WEAVE_VAE/runs_vae/A0/final_diffusers_vae", None),
    ("V2_200", "/root/Workspace/WEAVE_VAE/runs_vae/A0/final_diffusers_vae", "checkpoint_step_000200.pt"),
    ("V2_400", "/root/Workspace/WEAVE_VAE/runs_vae/A0/final_diffusers_vae", "checkpoint_step_000400.pt"),
    ("V2_800", "/root/Workspace/WEAVE_VAE/runs_vae/A0/final_diffusers_vae", "checkpoint_step_000800.pt"),
    ("V2_1200", "/root/Workspace/WEAVE_VAE/runs_vae/A0/final_diffusers_vae", "checkpoint_step_001200.pt"),
    ("V2_1600", "/root/Workspace/WEAVE_VAE/runs_vae/A0/final_diffusers_vae", "checkpoint_step_001600.pt"),
    ("V2_1800", "/root/Workspace/WEAVE_VAE/runs_vae/A0/final_diffusers_vae", "checkpoint_step_001800.pt"),
]

CKPT_DIR = Path("/root/Workspace/WEAVE_VAE/runs_vae/V2")


def build_descriptor():
    calib = json.loads(CALIB.read_text())
    desc = LatentSWDDescriptor()
    desc.set_calibration(
        torch.tensor(calib["chan_mean"]), torch.tensor(calib["chan_std"]),
        calib["scale_swd1"], calib["scale_swd2"], calib["scale_moment"],
        calib["median_inter_dist"],
    )
    return desc


def extract_latents(vae, dataset, num_batches=None):
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    dev = next(vae.parameters()).device
    z_all, y_all, a_all, g_all = [], [], [], []
    with torch.no_grad():
        for i, b in enumerate(loader):
            if num_batches is not None and i >= num_batches:
                break
            post = vae.encode(b["image"].to(dev)).latent_dist
            z_all.append(post.mean.float().cpu())
            y_all.append(b["style_id"])
            a_all.append(b["artist_id"])
            g_all.append(b["genre_id"])
    return (
        torch.cat(z_all),
        torch.cat(y_all).numpy(),
        torch.cat(a_all).numpy(),
        torch.cat(g_all).numpy(),
    )


def pairwise_dists(desc, q1, q2, mom, ia, ib, chunk=64):
    ds = []
    with torch.no_grad():
        for c in range(0, len(ia), chunk):
            ca, cb = ia[c:c+chunk], ib[c:c+chunk]
            d = desc.compute_pairwise_distance(q1[ca], q2[ca], mom[ca], q1[cb], q2[cb], mom[cb])
            ds.append(torch.diag(d))
    return torch.cat(ds).numpy()


def make_pairs(styles, artists, genres, seed=42):
    rng = np.random.default_rng(seed)
    N = len(styles)
    same, cross, diff, hard = [], [], [], []
    for i in range(N):
        sc = np.where(styles == styles[i])[0]
        sc = sc[sc != i]
        if len(sc):
            p = int(rng.choice(sc))
            same.append((i, p))
            ca = sc[artists[sc] != artists[i]] if artists[i] >= 0 else np.array([], dtype=int)
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


def desc_chunked(desc, z, chunk=192):
    """Compute SWD descriptors in chunks to avoid huge unfold tensors."""
    q1s, q2s, ms = [], [], []
    with torch.no_grad():
        for i in range(0, z.shape[0], chunk):
            a, b, c = desc(z[i:i + chunk])
            q1s.append(a); q2s.append(b); ms.append(c)
    return torch.cat(q1s), torch.cat(q2s), torch.cat(ms)


def diag_dists(desc, q1, q2, mom, ia, ib, chunk=48):
    """Memory-safe diagonal distance computation for index pairs."""
    out = []
    with torch.no_grad():
        for c in range(0, len(ia), chunk):
            ca, cb = ia[c:c + chunk], ib[c:c + chunk]
            d = desc.compute_pairwise_distance(q1[ca], q2[ca], mom[ca], q1[cb], q2[cb], mom[cb])
            out.append(torch.diag(d))
    return torch.cat(out).cpu().numpy()


def auroc_with_ci(desc, q1, q2, mom, pos_pairs, neg_pairs, n_boot=200, seed=42):
    n = min(len(pos_pairs), len(neg_pairs))
    if n < 10:
        return None
    n = min(n, 600)
    sp = np.array(pos_pairs[:n])
    dp = np.array(neg_pairs[:n])

    dpos = diag_dists(desc, q1, q2, mom, sp[:, 0], sp[:, 1])
    dneg = diag_dists(desc, q1, q2, mom, dp[:, 0], dp[:, 1])

    lbs = np.concatenate([np.ones(n), np.zeros(n)])
    d = np.concatenate([dpos, dneg])
    auc = float(roc_auc_score(lbs, -d) * 100.0)

    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        pi = rng.integers(0, n, n)
        ni = rng.integers(0, n, n)
        bb = np.concatenate([np.ones(n), np.zeros(n)])
        dd = np.concatenate([dpos[pi], dneg[ni]])
        boot.append(roc_auc_score(bb, -dd) * 100.0)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "auroc": round(auc, 2),
        "ci95_low": round(float(lo), 2),
        "ci95_high": round(float(hi), 2),
        "n_pairs": n,
        "mean_d_pos": round(float(dpos.mean()), 4),
        "mean_d_neg": round(float(dneg.mean()), 4),
        "rel_margin": round(float((dneg.mean() - dpos.mean()) / (dneg.mean() + dpos.mean() + 1e-8)), 4),
        "margin_violation_rate": round(float((dpos <= dneg).mean()), 4),
    }


def retrieval_map10(desc, q1, q2, mom, z_flat, styles, seed=42):
    """mAP@10 using SWD distance on a query/gallery split."""
    rng = np.random.default_rng(seed)
    N = len(styles)
    idx = rng.permutation(N)
    n_q = min(300, N // 4)
    q_ids, g_ids = idx[:n_q], idx[n_q:]
    aps = []
    for qi in q_ids:
        d = pairwise_dists(
            desc, q1, q2, mom,
            np.full(len(g_ids), qi), g_ids, chunk=128
        )
        order = np.argsort(d)[:10]
        rel = (styles[g_ids[order]] == styles[qi]).astype(float)
        if rel.sum() == 0:
            aps.append(0.0)
            continue
        prec = np.cumsum(rel) / (np.arange(len(rel)) + 1)
        aps.append(float((prec * rel).sum() / rel.sum()))
    return round(float(np.mean(aps) * 100.0), 2)


def knn_purity(styles, seed=42, ks=(1, 5, 10)):
    """kNN purity using 8x8 pooled latent cosine."""
    return None  # computed separately with pooled features


def main():
    desc = build_descriptor()
    dataset = RawMemmapWikiArtDataset(MANIFEST, is_training=False)
    print(f"[P1] Val dataset size: {len(dataset)}", flush=True)

    results = {}
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    for name, init_path, ckpt_name in MODELS:
        print(f"\n===== P1 Evaluating {name} =====", flush=True)
        vae = AutoencoderKL.from_pretrained(init_path).to(DEVICE).eval()
        if ckpt_name:
            ck = torch.load(CKPT_DIR / ckpt_name, map_location="cpu", weights_only=False)
            vae.load_state_dict(ck.get("ema", ck["model"]))
            vae.eval()

        z, styles, artists, genres = extract_latents(vae, dataset)
        del vae
        q1, q2, mom = desc_chunked(desc, z)
        same, cross, diff, hard = make_pairs(styles, artists, genres)

        r_overall = auroc_with_ci(desc, q1, q2, mom, same, diff)
        r_cross = auroc_with_ci(desc, q1, q2, mom, cross, diff)
        r_hard = auroc_with_ci(desc, q1, q2, mom, cross, hard) if len(hard) >= 10 else None

        # kNN purity on 8x8 pooled latent cosine
        import torch.nn as nn
        z_pool = nn.functional.adaptive_avg_pool2d(z, (8, 8)).flatten(1).numpy()
        z_pool = z_pool / (np.linalg.norm(z_pool, axis=1, keepdims=True) + 1e-8)
        sim = z_pool @ z_pool.T
        np.fill_diagonal(sim, -1)
        order = np.argsort(-sim, axis=1)
        knn = {}
        for k in (1, 5, 10):
            hits = (styles[order[:, :k]] == styles[:, None]).mean()
            knn[f"knn{k}"] = round(float(hits) * 100.0, 2)

        # Intra/inter on SWD distances (sample 300 anchors)
        z_flat = z.flatten(1)
        rng = np.random.default_rng(0)
        anchor_pool = rng.choice(len(styles), size=min(300, len(styles)), replace=False)
        intra, inter = [], []
        for i in anchor_pool:
            sc = np.where(styles == styles[i])[0]
            sc = sc[sc != i]
            if len(sc):
                j = int(rng.choice(sc))
                intra.append(diag_dists(desc, q1, q2, mom, np.array([i]), np.array([j]))[0])
            dc = np.where(styles != styles[i])[0]
            j = int(rng.choice(dc))
            inter.append(diag_dists(desc, q1, q2, mom, np.array([i]), np.array([j]))[0])

        res = {
            "n_samples": int(len(styles)),
            "latent_rms": round(float(z_flat.pow(2).mean().sqrt().item()), 4),
            "channel_std": [round(float(z[:, c].std().item()), 4) for c in range(4)],
            "overall": r_overall,
            "cross_artist": r_cross,
            "same_genre_hard": r_hard,
            "knn": knn,
            "mean_intra_swd": round(float(np.mean(intra)), 4),
            "mean_inter_swd": round(float(np.mean(inter)), 4),
        }
        results[name] = res

        print(f"  Overall: {r_overall['auroc']}% [{r_overall['ci95_low']}, {r_overall['ci95_high']}]", flush=True)
        if r_cross:
            print(f"  CrossArtist: {r_cross['auroc']}% [{r_cross['ci95_low']}, {r_cross['ci95_high']}] "
                  f"| relMargin={r_cross['rel_margin']}", flush=True)
        if r_hard:
            print(f"  SameGenreHard: {r_hard['auroc']}%", flush=True)
        print(f"  kNN: {knn} | RMS={res['latent_rms']} | intra={res['mean_intra_swd']} inter={res['mean_inter_swd']}", flush=True)

    (OUT_DIR / "02_unified_geometry.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    csv_lines = ["model,overall_auroc,overall_lo,overall_hi,cross_auroc,cross_lo,cross_hi,hard_auroc,knn1,knn5,knn10,rms,intra_swd,inter_swd,rel_margin"]
    for k, v in results.items():
        o = v["overall"] or {}
        c = v["cross_artist"] or {}
        h = v["same_genre_hard"] or {}
        csv_lines.append(
            f"{k},{o.get('auroc','')},{o.get('ci95_low','')},{o.get('ci95_high','')},"
            f"{c.get('auroc','')},{c.get('ci95_low','')},{c.get('ci95_high','')},"
            f"{h.get('auroc','')},{v['knn']['knn1']},{v['knn']['knn5']},{v['knn']['knn10']},"
            f"{v['latent_rms']},{v['mean_intra_swd']},{v['mean_inter_swd']},{c.get('rel_margin','')}"
        )
    (OUT_DIR / "02_unified_geometry.csv").write_text("\n".join(csv_lines), encoding="utf-8")

    print("\n===== P1 SUMMARY =====")
    print(f"{'Model':<12}{'Overall':>9}{'CrossArt':>10}{'Hard':>8}{'kNN1':>7}{'kNN5':>7}{'RMS':>8}{'RelM':>8}")
    for k, v in results.items():
        o = (v["overall"] or {}).get("auroc", 0)
        c = (v["cross_artist"] or {}).get("auroc", 0)
        h = (v["same_genre_hard"] or {}).get("auroc", 0)
        rm = (v["cross_artist"] or {}).get("rel_margin", 0)
        print(f"{k:<12}{o:>9.2f}{c:>10.2f}{h:>8.2f}{v['knn']['knn1']:>7.2f}{v['knn']['knn5']:>7.2f}{v['latent_rms']:>8.3f}{rm:>8.4f}")

    print(f"\n[Saved] {OUT_DIR / '02_unified_geometry.json'}")


if __name__ == "__main__":
    main()
