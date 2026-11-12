"""WEAVE Art-VAE Suite: Ultra-Fast Latent Style Separability & P1 Probe Diagnostic on CPU.

Specifically focuses on User's core question:
Did A0 and A1 fundamentally reorganize the latent space to improve style separability?

Evaluates across B0, A0, and A1 (strictly on CPU):
1. Style-Pair AUROC (Same-style vs Different-style discrimination in latent space, 10k pairs).
2. 5-NN Unsupervised Style Retrieval Accuracy (%).
3. Mean Intra-Class Variance (monitors geometric clustering / collapse).
4. Effective Rank of Latent Space (singular value spectrum).
5. P1 Independent Probes trained from scratch (136 classes):
   - Linear-Flat (256-d spatial pooled)
   - Linear-Stats (50-d with Gram & DWT)
   - Shallow-CNN (4x64x64)
   - Unseen Artist Generalization (Leave-25%-Artists-Out Top-1 and Macro-F1)
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

os.environ["OMP_NUM_THREADS"] = "32"
os.environ["MKL_NUM_THREADS"] = "32"

import numpy as np
import torch
import torch.nn as nn
torch.set_num_threads(32)

from diffusers import AutoencoderKL
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier

from tools.v6.canonicalize import dwt2_haar
from tools.vae_suite.dataset import RawMemmapWikiArtDataset


class LinearFlatProbe(nn.Module):
    def __init__(self, in_features: int = 256, num_classes: int = 136):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((8, 8))
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(self.pool(z).flatten(1))


class LinearStatsProbe(nn.Module):
    def __init__(self, in_features: int = 50, num_classes: int = 136):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B, C, H, W = z.shape
        m = z.mean(dim=(2, 3))
        s = z.std(dim=(2, 3)).clamp(min=1e-5)
        z_flat = z.view(B, C, -1)
        gram = torch.bmm(z_flat, z_flat.transpose(1, 2)) / float(H * W)
        tri_idx = torch.triu_indices(C, C)
        gram_tri = gram[:, tri_idx[0], tri_idx[1]] # (B, 10)

        ll, lh, hl, hh = dwt2_haar(z)
        dwt_stats = []
        for b in (ll, lh, hl, hh):
            dwt_stats.append(b.mean(dim=(2, 3)))
            dwt_stats.append(b.std(dim=(2, 3)).clamp(min=1e-5))
        feat = torch.cat([m, s, gram_tri] + dwt_stats, dim=1)
        return self.fc(feat)


class ShallowCNNProbe(nn.Module):
    def __init__(self, in_channels: int = 4, num_classes: int = 136):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, num_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def compute_effective_rank(z_tensor: torch.Tensor) -> float:
    mat = z_tensor.flatten(1).float()
    mat = mat - mat.mean(dim=0, keepdim=True)
    _, s, _ = torch.svd(mat)
    s_norm = s / s.sum()
    entropy = -torch.sum(s_norm * torch.log(s_norm.clamp(min=1e-12))).item()
    return round(float(math.exp(entropy)), 2)


def compute_style_pair_auroc(z_feats: torch.Tensor, styles: np.ndarray, n_pairs: int = 10000, seed: int = 42) -> float:
    rng = np.random.default_rng(seed)
    N = z_feats.shape[0]
    idx1 = rng.integers(0, N, size=n_pairs)
    idx2 = rng.integers(0, N, size=n_pairs)
    valid = (idx1 != idx2)
    idx1, idx2 = idx1[valid], idx2[valid]

    labels = (styles[idx1] == styles[idx2]).astype(int)
    f1 = nn.functional.normalize(z_feats[idx1], dim=-1)
    f2 = nn.functional.normalize(z_feats[idx2], dim=-1)
    sims = (f1 * f2).sum(dim=-1).numpy()

    if labels.sum() == 0 or (1 - labels).sum() == 0:
        return 50.0
    return round(float(roc_auc_score(labels, sims) * 100.0), 2)


def compute_intra_class_variance(z_feats: torch.Tensor, styles: np.ndarray) -> float:
    f = nn.functional.normalize(z_feats, dim=-1)
    unique_styles = np.unique(styles)
    vars_list = []
    for s in unique_styles:
        mask = (styles == s)
        if mask.sum() > 1:
            cls_feats = f[mask]
            cls_center = cls_feats.mean(dim=0, keepdim=True)
            var_s = (cls_feats - cls_center).pow(2).sum(dim=-1).mean().item()
            vars_list.append(var_s)
    return round(float(np.mean(vars_list)), 4) if vars_list else 0.0


def train_eval_probe(probe: nn.Module, z_tr, y_tr, z_ev, y_ev, num_epochs: int = 15, lr: float = 1e-3):
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    train_ds = torch.utils.data.TensorDataset(z_tr, y_tr)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    for epoch in range(num_epochs):
        probe.train()
        for z_b, y_b in train_loader:
            optimizer.zero_grad()
            logits = probe(z_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()

    probe.eval()
    val_ds = torch.utils.data.TensorDataset(z_ev, y_ev)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    preds_all = []
    targets_all = []
    top5_hits = 0

    with torch.no_grad():
        for z_b, y_b in val_loader:
            logits = probe(z_b)
            pred = logits.argmax(dim=1)
            preds_all.append(pred)
            targets_all.append(y_b)

            top5 = torch.topk(logits, k=min(5, logits.shape[1]), dim=1).indices
            for i in range(len(y_b)):
                if y_b[i] in top5[i]:
                    top5_hits += 1

    preds_arr = torch.cat(preds_all).numpy()
    targets_arr = torch.cat(targets_all).numpy()

    top1 = float((preds_arr == targets_arr).mean() * 100.0)
    top5 = float(top5_hits / len(targets_arr) * 100.0)
    f1 = float(f1_score(targets_arr, preds_arr, average="macro", zero_division=0) * 100.0)

    return {
        "Top1_Accuracy": round(top1, 2),
        "Top5_Accuracy": round(top5, 2),
        "Macro_F1": round(f1, 2),
    }


def run_unseen_artist_test(probe_cls, z_all: torch.Tensor, y_all: np.ndarray, artists_all: np.ndarray, num_classes: int) -> dict:
    valid_mask = (artists_all >= 0)
    if valid_mask.sum() < 200:
        return {"Top1": 0.0, "Macro_F1": 0.0}

    unique_artists = np.unique(artists_all[valid_mask])
    rng = np.random.default_rng(42)
    rng.shuffle(unique_artists)

    n_seen = int(0.75 * len(unique_artists))
    seen_artists = set(unique_artists[:n_seen])

    tr_mask = np.isin(artists_all, list(seen_artists))
    te_mask = ~tr_mask

    if tr_mask.sum() < 100 or te_mask.sum() < 50:
        return {"Top1": 0.0, "Macro_F1": 0.0}

    z_tr, y_tr = z_all[tr_mask], torch.from_numpy(y_all[tr_mask])
    z_te, y_te = z_all[te_mask], torch.from_numpy(y_all[te_mask])

    probe = probe_cls(num_classes=num_classes)
    res = train_eval_probe(probe, z_tr, y_tr, z_te, y_te, num_epochs=15)
    return {
        "Top1": res["Top1_Accuracy"],
        "Macro_F1": res["Macro_F1"],
    }


def extract_encoder_latents(vae: nn.Module, loader: DataLoader) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Pure Encoder forward pass on CPU - fast and lightweight."""
    z_all, y_all, ar_all = [], [], []
    with torch.inference_mode():
        for b_idx, batch in enumerate(loader):
            x = batch["image"] # (B, 3, 512, 512)
            # encoder + quant_conv
            h = vae.encoder(x)
            moments = vae.quant_conv(h)
            mu, _ = torch.chunk(moments, 2, dim=1)
            z_all.append(mu)
            y_all.append(batch["style_id"])
            ar_all.append(batch["artist_id"])
            if (b_idx + 1) % 40 == 0 or (b_idx + 1) == len(loader):
                print(f"    Encoded [{b_idx+1}/{len(loader)}] batches...", flush=True)

    return torch.cat(z_all, dim=0), torch.cat(y_all, dim=0).numpy(), torch.cat(ar_all, dim=0).numpy()


def evaluate_model_style_separability(
    model_name: str,
    vae_path: str,
    dataset: RawMemmapWikiArtDataset,
    loader: DataLoader,
    num_classes: int,
) -> dict:
    print(f"\n=== Evaluating Style Separability for: {model_name} ===", flush=True)
    t0 = time.time()

    print(f"  [1/3] Loading VAE Encoder from {vae_path} on CPU...", flush=True)
    vae = AutoencoderKL.from_pretrained(vae_path).to("cpu").eval()

    print("  [2/3] Extracting latent representations on 2048 validation images...", flush=True)
    z_all, y_all, ar_all = extract_encoder_latents(vae, loader)
    latent_std = float(z_all.std().item())
    latent_mean = float(z_all.mean().item())

    # Topology
    print("  [3/3] Computing Latent Topology & Training Probes (136 Classes)...", flush=True)
    eff_rank = compute_effective_rank(z_all)
    style_auroc = compute_style_pair_auroc(z_all.flatten(1), y_all, n_pairs=10000)
    intra_var = compute_intra_class_variance(z_all.flatten(1), y_all)

    # 5-NN Retrieval
    knn = KNeighborsClassifier(n_neighbors=5, metric="cosine")
    z_pool = nn.functional.adaptive_avg_pool2d(z_all, (8, 8)).flatten(1).numpy()
    rng = np.random.default_rng(42)
    indices = np.arange(len(z_all))
    rng.shuffle(indices)
    tr_idx, val_idx = indices[:1500], indices[1500:]

    knn.fit(z_pool[tr_idx], y_all[tr_idx])
    knn_preds = knn.predict(z_pool[val_idx])
    knn_top1 = float((knn_preds == y_all[val_idx]).mean() * 100.0)

    z_tr, y_tr = z_all[tr_idx], torch.from_numpy(y_all[tr_idx])
    z_ev, y_ev = z_all[val_idx], torch.from_numpy(y_all[val_idx])

    probes = {}
    for p_name, p_cls in (
        ("Linear_Flat", LinearFlatProbe),
        ("Linear_Stats", LinearStatsProbe),
        ("Shallow_CNN", ShallowCNNProbe),
    ):
        p_obj = p_cls(num_classes=num_classes)
        p_res = train_eval_probe(p_obj, z_tr, y_tr, z_ev, y_ev, num_epochs=15)
        unseen_res = run_unseen_artist_test(p_cls, z_all, y_all, ar_all, num_classes=num_classes)
        p_res["Unseen_Artist_Top1"] = unseen_res["Top1"]
        p_res["Unseen_Artist_Macro_F1"] = unseen_res["Macro_F1"]
        probes[p_name] = p_res
        print(f"    -> {p_name:12s}: Top-1={p_res['Top1_Accuracy']:5.2f}% | Top-5={p_res['Top5_Accuracy']:5.2f}% | F1={p_res['Macro_F1']:4.2f}% | Unseen-Artist Top-1={p_res['Unseen_Artist_Top1']:5.2f}%")

    eval_duration = round(time.time() - t0, 1)
    res = {
        "model_name": model_name,
        "vae_path": vae_path,
        "duration_sec": eval_duration,
        "latent_statistics": {
            "mean": round(latent_mean, 4),
            "std": round(latent_std, 4),
            "effective_rank": eff_rank,
        },
        "style_topology": {
            "style_pair_auroc": style_auroc,
            "kNN_5_retrieval_acc": round(knn_top1, 2),
            "mean_intra_class_variance": intra_var,
        },
        "probes": probes,
    }
    return res


def main():
    print("=== [Fast CPU Diagnostic] Comparing Latent Style Separability for B0, A0, A1 ===", flush=True)
    manifest_path = "/root/Workspace/data/wikiart_512_raw/wikiart_512_raw_manifest.json"
    dataset = RawMemmapWikiArtDataset(Path(manifest_path), is_training=False)
    num_classes = dataset.manifest["num_styles"]
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    models_to_test = [
        ("B0 (Original Baseline)", "/root/Workspace/xy/DiT/data/pretrained/sd-vae-ft-ema"),
        ("A0 (Recon+KL)", "/root/Workspace/WEAVE_VAE/runs_vae/A0/final_diffusers_vae"),
        ("A1 (A0+Style_CE)", "/root/Workspace/WEAVE_VAE/runs_vae/A1/final_diffusers_vae"),
    ]

    results = {}
    for m_name, m_path in models_to_test:
        if not Path(m_path).exists():
            print(f"[WARN] Skipping missing model {m_name} at {m_path}")
            continue
        res = evaluate_model_style_separability(m_name, m_path, dataset, loader, num_classes)
        results[m_name] = res

    out_file = Path("/root/Workspace/WEAVE_VAE/fast_style_diagnostic_results.json")
    out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[Saved] Wrote full diagnostic results to {out_file}", flush=True)


if __name__ == "__main__":
    main()
