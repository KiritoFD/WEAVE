"""WEAVE-VAE V2: Calibration of SWD & Moment Distance on A0 Model.

Performs:
1. Calculates channel-wise mean and std from A0 training latents.
2. Calibrates component scaling factors (scale_swd1, scale_swd2, scale_moment).
3. Computes median inter-class distance M_inter for final normalization.
4. Measures true baseline Style AUROC of A0 under this new metric (both overall & cross-artist).
5. Outputs swd_calibration.json.
"""
from __future__ import annotations

import argparse
import json
import os
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

from tools.vae_suite.dataset import RawMemmapWikiArtDataset
from tools.vae_suite.swd_metric import LatentSWDDescriptor


def calibrate_swd_metric(
    vae_path: str,
    manifest_path: str,
    output_path: str,
    device: str = "cuda",
    num_samples: int = 1500,
):
    print(f"=== [SWD Metric Calibration] Calibrating on A0 at {vae_path} ===", flush=True)

    dataset = RawMemmapWikiArtDataset(Path(manifest_path), is_training=True)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    vae = AutoencoderKL.from_pretrained(vae_path).to(device).eval()

    # Step 1: Extract 1500 latents from A0
    print(f"[Step 1/4] Extracting latents from {num_samples} training samples...", flush=True)
    z_list, styles_list, artists_list, genres_list = [], [], [], []
    count = 0

    with torch.inference_mode():
        for b in loader:
            x = b["image"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                mu = vae.encode(x).latent_dist.mean
            z_list.append(mu.float().cpu())
            styles_list.append(b["style_id"])
            artists_list.append(b["artist_id"])
            genres_list.append(b["genre_id"])
            count += x.shape[0]
            if count >= num_samples:
                break

    z_all = torch.cat(z_list, dim=0)[:num_samples] # (N, 4, 64, 64) on CPU
    styles = torch.cat(styles_list, dim=0)[:num_samples].cpu().numpy()
    artists = torch.cat(artists_list, dim=0)[:num_samples].cpu().numpy()
    genres = torch.cat(genres_list, dim=0)[:num_samples].cpu().numpy()

    # Step 2: Compute channel mean and std
    print("[Step 2/4] Computing channel normalization parameters...", flush=True)
    chan_mean = z_all.mean(dim=(0, 2, 3)).cpu() # (4,)
    chan_std = z_all.std(dim=(0, 2, 3)).cpu() # (4,)
    print(f"  Channel Mean: {chan_mean.tolist()}")
    print(f"  Channel Std : {chan_std.tolist()}")

    descriptor = LatentSWDDescriptor().to(device)
    descriptor.chan_mean.copy_(chan_mean.view(1, 4, 1, 1).to(device))
    descriptor.chan_std.copy_(chan_std.view(1, 4, 1, 1).to(device))

    # Step 3: Extract Q1, Q2, Moments on device
    print("[Step 3/4] Extracting descriptors and calibrating component scales...", flush=True)
    q1_list, q2_list, mom_list = [], [], []
    with torch.inference_mode():
        for i in range(0, z_all.shape[0], 128):
            z_chunk = z_all[i : i + 128].to(device)
            q1_c, q2_c, mom_c = descriptor(z_chunk)
            q1_list.append(q1_c.cpu())
            q2_list.append(q2_c.cpu())
            mom_list.append(mom_c.cpu())

    q1_all = torch.cat(q1_list, dim=0).to(device)
    q2_all = torch.cat(q2_list, dim=0).to(device)
    mom_all = torch.cat(mom_list, dim=0).to(device)

    # Sample 5000 random pairs to estimate scale_swd1, scale_swd2, scale_moment
    rng = np.random.default_rng(42)
    N = z_all.shape[0]
    p_idx1 = rng.integers(0, N, size=5000)
    p_idx2 = rng.integers(0, N, size=5000)
    valid = (p_idx1 != p_idx2)
    p_idx1, p_idx2 = p_idx1[valid], p_idx2[valid]

    diff1 = (q1_all[p_idx1] - q1_all[p_idx2]).pow(2).mean(dim=(-2, -1))
    diff2 = (q2_all[p_idx1] - q2_all[p_idx2]).pow(2).mean(dim=(-2, -1))
    diff_m = (mom_all[p_idx1] - mom_all[p_idx2]).pow(2).mean(dim=-1)

    scale_swd1 = float(diff1.mean().item())
    scale_swd2 = float(diff2.mean().item())
    scale_moment = float(diff_m.mean().item())

    # Raw combined distance
    d_raw = 0.4 * (diff1 / scale_swd1) + 0.4 * (diff2 / scale_swd2) + 0.2 * (diff_m / scale_moment)

    # Sample inter-class pairs (different styles) to get median inter distance
    diff_style_mask = (styles[p_idx1] != styles[p_idx2])
    median_inter_dist = float(torch.median(d_raw[diff_style_mask]).item())

    print(f"  Component Scales: SWD1={scale_swd1:.4f}, SWD2={scale_swd2:.4f}, Moment={scale_moment:.4f}")
    print(f"  Median Inter-Class Distance: {median_inter_dist:.4f}")

    descriptor.set_calibration(
        chan_mean, chan_std, scale_swd1, scale_swd2, scale_moment, median_inter_dist
    )

    # Step 4: Measure baseline Style AUROC of A0 with this new metric
    print("\n[Step 4/4] Measuring True Baseline Style AUROC on A0...", flush=True)
    # 1. Overall Style AUROC (same-style vs diff-style)
    # We construct 5000 same-style pairs and 5000 diff-style pairs
    same_pairs = []
    diff_pairs = []
    unseen_artist_same_pairs = []

    for i in range(N):
        s_i = styles[i]
        a_i = artists[i]
        same_cand = np.where(styles == s_i)[0]
        same_cand = same_cand[same_cand != i]
        if len(same_cand) > 0:
            pick = rng.choice(same_cand)
            same_pairs.append((i, pick))
            if a_i >= 0 and artists[pick] >= 0 and a_i != artists[pick]:
                unseen_artist_same_pairs.append((i, pick))

        diff_cand = np.where(styles != s_i)[0]
        if len(diff_cand) > 0:
            diff_pairs.append((i, rng.choice(diff_cand)))

    n_eval = min(len(same_pairs), len(diff_pairs))
    same_arr = np.array(same_pairs[:n_eval])
    diff_arr = np.array(diff_pairs[:n_eval])

    idx_a = np.concatenate([same_arr[:, 0], diff_arr[:, 0]])
    idx_b = np.concatenate([same_arr[:, 1], diff_arr[:, 1]])
    labels = np.concatenate([np.ones(n_eval), np.zeros(n_eval)])

    # Compute calibrated distances
    with torch.inference_mode():
        dists = []
        for c in range(0, len(idx_a), 256):
            c_a = idx_a[c : c + 256]
            c_b = idx_b[c : c + 256]
            d_batch = descriptor.compute_pairwise_distance(
                q1_all[c_a], q2_all[c_a], mom_all[c_a],
                q1_all[c_b], q2_all[c_b], mom_all[c_b]
            )
            # Take diagonal
            dists.append(torch.diag(d_batch).cpu())
        dists_all = torch.cat(dists).numpy()

    # Smaller distance means same style -> score = -distance
    overall_auroc = float(roc_auc_score(labels, -dists_all) * 100.0)
    print(f"  >>> A0 Overall Style AUROC (New SWD Metric): {overall_auroc:.2f}%")

    # Cross-artist same style vs diff style AUROC
    cross_artist_auroc = 50.0
    if len(unseen_artist_same_pairs) >= 50:
        n_ca = min(len(unseen_artist_same_pairs), len(diff_pairs))
        ca_same_arr = np.array(unseen_artist_same_pairs[:n_ca])
        ca_diff_arr = np.array(diff_pairs[:n_ca])
        ca_idx_a = np.concatenate([ca_same_arr[:, 0], ca_diff_arr[:, 0]])
        ca_idx_b = np.concatenate([ca_same_arr[:, 1], ca_diff_arr[:, 1]])
        ca_labels = np.concatenate([np.ones(n_ca), np.zeros(n_ca)])

        with torch.inference_mode():
            ca_dists = []
            for c in range(0, len(ca_idx_a), 256):
                c_a = ca_idx_a[c : c + 256]
                c_b = ca_idx_b[c : c + 256]
                d_b = descriptor.compute_pairwise_distance(
                    q1_all[c_a], q2_all[c_a], mom_all[c_a],
                    q1_all[c_b], q2_all[c_b], mom_all[c_b]
                )
                ca_dists.append(torch.diag(d_b).cpu())
            ca_dists_all = torch.cat(ca_dists).numpy()
        cross_artist_auroc = float(roc_auc_score(ca_labels, -ca_dists_all) * 100.0)
        print(f"  >>> A0 Cross-Artist Style AUROC (New SWD Metric): {cross_artist_auroc:.2f}%")

    out_data = {
        "calibrated_from_model": vae_path,
        "chan_mean": chan_mean.tolist(),
        "chan_std": chan_std.tolist(),
        "scale_swd1": scale_swd1,
        "scale_swd2": scale_swd2,
        "scale_moment": scale_moment,
        "median_inter_dist": median_inter_dist,
        "a0_baseline_auroc": {
            "overall_style_auroc": round(overall_auroc, 2),
            "cross_artist_style_auroc": round(cross_artist_auroc, 2),
        },
    }

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(out_data, indent=2), encoding="utf-8")
    print(f"[Done] Saved calibration and baseline AUROC to {out_p}")
    return out_data


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae-path", default="/root/Workspace/WEAVE_VAE/runs_vae/A0/final_diffusers_vae")
    ap.add_argument("--manifest", default="/root/Workspace/data/wikiart_512_raw/wikiart_512_raw_manifest.json")
    ap.add_argument("--output", default="/root/Workspace/WEAVE_VAE/swd_calibration.json")
    args = ap.parse_args()

    calibrate_swd_metric(args.vae_path, args.manifest, args.output)
