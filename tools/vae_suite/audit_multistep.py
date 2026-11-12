"""WEAVE-VAE V2: Fast Multi-Checkpoint Evaluation Audit.

Evaluates trained checkpoints (200, 400, 600, 800, 1000, 1200, 1400, 1600) on:
1. P1 Style Geometry:
   - Overall Style AUROC (SWD)
   - Cross-Artist Style AUROC (SWD)
   - kNN-5 Style Retrieval Accuracy
   - Effective Rank & Intra-Class Variance
2. Posterior Stability:
   - Median posterior sigma
   - Mean KL divergence
3. Save all to a single JSON with a summary table.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier

from tools.vae_suite.dataset import RawMemmapWikiArtDataset
from tools.vae_suite.swd_metric import LatentSWDDescriptor
from tools.vae_suite.losses import compute_kl_divergence


def compute_effective_rank(z: torch.Tensor) -> float:
    mat = z.flatten(1).float()
    mat = mat - mat.mean(dim=0, keepdim=True)
    _, s, _ = torch.svd(mat)
    s_norm = s / s.sum()
    entropy = -torch.sum(s_norm * torch.log(s_norm.clamp(min=1e-12))).item()
    return round(float(np.exp(entropy)), 2)


def evaluate_checkpoint(
    vae: AutoencoderKL,
    descriptor: LatentSWDDescriptor,
    loader: DataLoader,
    num_batches: int = 16,
) -> dict:
    sigmas, kl_vals, z_all, styles_all, artists_all = [], [], [], [], []
    with torch.no_grad():
        for b_idx, b in enumerate(loader):
            if b_idx >= num_batches:
                break
            post = vae.encode(b["image"]).latent_dist
            sigmas.append(post.std.reshape(-1))
            _, kl_v, _, _ = compute_kl_divergence(post)
            kl_vals.append(kl_v)
            z_all.append(post.mean.float())
            styles_all.append(b["style_id"])
            artists_all.append(b["artist_id"])

    sigmas_np = torch.cat(sigmas).numpy()
    p50 = float(np.percentile(sigmas_np, 50))
    mean_kl = float(np.mean(kl_vals))

    z_eval = torch.cat(z_all, dim=0)
    styles_eval = torch.cat(styles_all, dim=0).numpy()
    artists_eval = torch.cat(artists_all, dim=0).numpy()

    q1, q2, mom = descriptor(z_eval)
    rng = np.random.default_rng(42)
    N = len(styles_eval)
    same_pairs, diff_pairs, cross_pairs = [], [], []

    for i in range(N):
        s_i, a_i = styles_eval[i], artists_eval[i]
        same_c = np.where(styles_eval == s_i)[0]
        same_c = same_c[same_c != i]
        if len(same_c) > 0:
            p = int(rng.choice(same_c))
            same_pairs.append((i, p))
            if a_i >= 0 and artists_eval[p] >= 0 and a_i != artists_eval[p]:
                cross_pairs.append((i, p))
        diff_c = np.where(styles_eval != s_i)[0]
        if len(diff_c) > 0:
            diff_pairs.append((i, int(rng.choice(diff_c))))

    def auroc_for(pairs):
        if len(pairs) < 15:
            return 50.0
        n = min(len(pairs), len(diff_pairs))
        sp = np.array(pairs[:n])
        dp = np.array(diff_pairs[:n])
        ia = np.concatenate([sp[:, 0], dp[:, 0]])
        ib = np.concatenate([sp[:, 1], dp[:, 1]])
        lbs = np.concatenate([np.ones(n), np.zeros(n)])
        with torch.no_grad():
            ds = []
            for c in range(0, len(ia), 64):
                ca, cb = ia[c:c+64], ib[c:c+64]
                d = descriptor.compute_pairwise_distance(
                    q1[ca], q2[ca], mom[ca], q1[cb], q2[cb], mom[cb]
                )
                ds.append(torch.diag(d))
            dall = torch.cat(ds).numpy()
        return round(float(roc_auc_score(lbs, -dall) * 100.0), 2)

    overall = auroc_for(same_pairs)
    cross = auroc_for(cross_pairs)

    # kNN-5 on 8x8 pooled latent
    z_pool = nn.functional.adaptive_avg_pool2d(z_eval, (8, 8)).flatten(1).numpy()
    n_tr = int(0.75 * len(z_pool))
    idx = rng.permutation(len(z_pool))
    knn = KNeighborsClassifier(n_neighbors=5, metric="cosine")
    knn.fit(z_pool[idx[:n_tr]], styles_eval[idx[:n_tr]])
    pred = knn.predict(z_pool[idx[n_tr:]])
    knn_top1 = round(float((pred == styles_eval[idx[n_tr:]]).mean() * 100.0), 2)

    eff_rank = compute_effective_rank(z_eval)

    return {
        "overall_auroc": overall,
        "cross_artist_auroc": cross,
        "knn5_top1": knn_top1,
        "p50_sigma": round(p50, 6),
        "mean_kl": round(mean_kl, 2),
        "effective_rank": eff_rank,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="/root/Workspace/WEAVE_VAE/runs_vae/V2")
    ap.add_argument("--steps", default="200,400,600,800,1000,1200,1400,1600")
    ap.add_argument("--manifest", default="/root/Workspace/data/wikiart_512_raw/wikiart_512_raw_manifest.json")
    ap.add_argument("--calib-json", default="/root/Workspace/WEAVE_VAE/swd_calibration.json")
    ap.add_argument("--output", default="/root/Workspace/WEAVE_VAE/runs_vae/V2/multistep_audit.json")
    ap.add_argument("--num-batches", type=int, default=16)
    args = ap.parse_args()

    steps = [int(s) for s in args.steps.split(",")]
    calib = json.loads(Path(args.calib_json).read_text())
    descriptor = LatentSWDDescriptor()
    descriptor.set_calibration(
        torch.tensor(calib["chan_mean"]),
        torch.tensor(calib["chan_std"]),
        calib["scale_swd1"],
        calib["scale_swd2"],
        calib["scale_moment"],
        calib["median_inter_dist"],
    )

    dataset = RawMemmapWikiArtDataset(Path(args.manifest), is_training=False)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    vae = AutoencoderKL.from_pretrained(
        "/root/Workspace/WEAVE_VAE/runs_vae/A0/final_diffusers_vae"
    ).eval()

    results = {
        "A0_baseline": {
            "overall_auroc": 59.90,
            "cross_artist_auroc": 54.67,
            "knn5_top1": 10.40,
            "p50_sigma": None,
            "mean_kl": 17.50,
            "effective_rank": 1772.65,
        }
    }

    for st in steps:
        ckpt_path = Path(args.ckpt_dir) / f"checkpoint_step_{st:06d}.pt"
        if not ckpt_path.exists():
            print(f"[Skip] {ckpt_path} not found")
            continue
        print(f"\n=== Evaluating checkpoint step {st} ===", flush=True)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        vae.load_state_dict(ckpt.get("ema", ckpt["model"]))
        vae.eval()
        t0 = time.time()
        res = evaluate_checkpoint(vae, descriptor, loader, num_batches=args.num_batches)
        res["eval_sec"] = round(time.time() - t0, 1)
        print(
            f"  step {st}: Overall={res['overall_auroc']}% | CrossArtist={res['cross_artist_auroc']}% | "
            f"kNN5={res['knn5_top1']}% | p50_sigma={res['p50_sigma']:.6f} | KL={res['mean_kl']} | "
            f"Rank={res['effective_rank']} ({res['eval_sec']}s)",
            flush=True,
        )
        results[f"step_{st}"] = res

    out = Path(args.output)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[Saved] {out}")

    # Print summary table
    print("\n===== MULTI-STEP SUMMARY =====")
    print(f"{'Model':<14}{'Overall':>10}{'CrossArt':>10}{'kNN5':>8}{'KL':>8}{'Rank':>8}")
    for k, v in results.items():
        print(
            f"{k:<14}{v['overall_auroc']:>10.2f}{v['cross_artist_auroc']:>10.2f}"
            f"{v['knn5_top1']:>8.2f}{v['mean_kl']:>8.2f}{v['effective_rank']:>8.1f}"
        )


if __name__ == "__main__":
    main()
