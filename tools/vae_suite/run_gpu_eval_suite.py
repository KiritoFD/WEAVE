"""WEAVE Art-VAE Suite: Full GPU Evaluation Suite for B0, A0, A1, A2.

Runs on RTX 4090 with batch_size=32 and zero-decode RAM dataset:
1. P0 Hard Gate Metrics:
   - Reconstruction LPIPS (AlexNet GPU)
   - PSNR (dB)
   - SSIM
   - L1 error
   - CSD Art Style Cosine Fidelity (before vs after VAE roundtrip)
   - Latent Std & Variance Stability
2. P1 Style Separability & Topology across 136 Style Classes:
   - Linear-Flat Probe (256-d, trained on GPU)
   - Linear-Stats Probe (50-d with Gram & DWT, trained on GPU)
   - Shallow-CNN Probe (4x64x64, trained on GPU)
   - Unseen Artist Generalization (Leave-25%-Artists-Out Top-1 & Macro-F1)
   - Style-Pair Discrimination AUROC (10k pairs)
   - 5-NN Unsupervised Style Retrieval Accuracy (%)
   - Mean Intra-Class Variance (monitors over-clustering collapse)
   - Effective Rank of Latent Space
3. Output:
   - gpu_eval_summary.json
   - docs/v6/VAE_A0_A1_A2_FULL_REPORT.md
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
sys.path.insert(0, str(REPO_ROOT / "tools" / "12h"))

import numpy as np
import torch
import torch.nn as nn
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier

from tools.v6.canonicalize import dwt2_haar
from tools.vae_suite.dataset import RawMemmapWikiArtDataset
import csd_model
import lpips


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
        gram_tri = gram[:, tri_idx[0], tri_idx[1]]

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


def compute_psnr(x: torch.Tensor, y: torch.Tensor) -> float:
    x01 = (x.clamp(-1, 1) + 1.0) / 2.0
    y01 = (y.clamp(-1, 1) + 1.0) / 2.0
    mse = torch.mean((x01 - y01) ** 2).item()
    if mse <= 1e-10:
        return 50.0
    return float(10.0 * math.log10(1.0 / mse))


def compute_ssim_simple(x: torch.Tensor, y: torch.Tensor) -> float:
    x01 = (x.clamp(-1, 1) + 1.0) / 2.0
    y01 = (y.clamp(-1, 1) + 1.0) / 2.0
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2

    mu_x = nn.functional.avg_pool2d(x01, kernel_size=11, stride=1, padding=5)
    mu_y = nn.functional.avg_pool2d(y01, kernel_size=11, stride=1, padding=5)

    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x_sq = nn.functional.avg_pool2d(x01 * x01, kernel_size=11, stride=1, padding=5) - mu_x_sq
    sigma_y_sq = nn.functional.avg_pool2d(y01 * y01, kernel_size=11, stride=1, padding=5) - mu_y_sq
    sigma_xy = nn.functional.avg_pool2d(x01 * y01, kernel_size=11, stride=1, padding=5) - mu_xy

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))
    return float(ssim_map.mean().item())


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


def train_eval_probe_gpu(probe: nn.Module, z_tr, y_tr, z_ev, y_ev, num_epochs: int = 15, lr: float = 1e-3, device: str = "cuda"):
    probe = probe.to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    train_ds = torch.utils.data.TensorDataset(z_tr.to(device), y_tr.to(device))
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)

    for epoch in range(num_epochs):
        probe.train()
        for z_b, y_b in train_loader:
            optimizer.zero_grad()
            logits = probe(z_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()

    probe.eval()
    val_ds = torch.utils.data.TensorDataset(z_ev.to(device), y_ev.to(device))
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

    preds_all = []
    targets_all = []
    top5_hits = 0

    with torch.no_grad():
        for z_b, y_b in val_loader:
            logits = probe(z_b)
            pred = logits.argmax(dim=1).cpu()
            preds_all.append(pred)
            targets_all.append(y_b.cpu())

            top5 = torch.topk(logits, k=min(5, logits.shape[1]), dim=1).indices.cpu()
            for i in range(len(y_b)):
                if y_b[i].item() in top5[i]:
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


def run_unseen_artist_test_gpu(probe_cls, z_all: torch.Tensor, y_all: np.ndarray, artists_all: np.ndarray, num_classes: int, device: str = "cuda") -> dict:
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
    res = train_eval_probe_gpu(probe, z_tr, y_tr, z_te, y_te, num_epochs=15, device=device)
    return {
        "Top1": res["Top1_Accuracy"],
        "Macro_F1": res["Macro_F1"],
    }


def evaluate_model_on_gpu(
    model_name: str,
    vae_path: str,
    dataset: RawMemmapWikiArtDataset,
    lpips_fn: nn.Module,
    csd: nn.Module,
    b0_benchmark: dict,
    num_classes: int,
    device: str = "cuda",
    batch_size: int = 16,
) -> dict:
    print(f"\n=======================================================", flush=True)
    print(f"  GPU Evaluating Model: {model_name}", flush=True)
    print(f"  Path: {vae_path}", flush=True)
    print(f"=======================================================", flush=True)

    t0 = time.time()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    vae = AutoencoderKL.from_pretrained(vae_path).to(device).eval()

    # Phase 1: Reconstruction
    print(f"[Phase 1] Evaluating Reconstruction on {len(dataset)} Val Images...", flush=True)
    lpips_scores, psnr_scores, ssim_scores, l1_scores, csd_sim_scores = [], [], [], [], []
    latents_mu, style_labels, artist_labels = [], [], []

    norm_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
    norm_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)

    with torch.inference_mode():
        for b_idx, batch in enumerate(loader):
            x = batch["image"].to(device, non_blocking=True)
            y = batch["style_id"]
            ar = batch["artist_id"]

            posterior = vae.encode(x).latent_dist
            mu = posterior.mean
            latents_mu.append(mu.cpu())
            style_labels.append(y)
            artist_labels.append(ar)

            recon = vae.decode(mu).sample
            l1_scores.append(nn.functional.l1_loss(recon, x).item())
            lpips_scores.append(lpips_fn(recon.float(), x.float()).mean().item())
            psnr_scores.append(compute_psnr(recon, x))
            ssim_scores.append(compute_ssim_simple(recon, x))

            # CSD Art Style Cosine Fidelity
            recon_01 = (recon.clamp(-1, 1) + 1.0) / 2.0
            x_01 = (x.clamp(-1, 1) + 1.0) / 2.0
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                x_224 = nn.functional.interpolate(x_01, size=(224, 224), mode="bicubic", align_corners=False)
                recon_224 = nn.functional.interpolate(recon_01, size=(224, 224), mode="bicubic", align_corners=False)
                x_csd = (x_224 - norm_mean) / norm_std
                recon_csd = (recon_224 - norm_mean) / norm_std
                _, _, sty_orig = csd(x_csd.half())
                _, _, sty_recon = csd(recon_csd.half())
                csd_sim = torch.sum(sty_recon * sty_orig, dim=-1).mean().item()
            csd_sim_scores.append(csd_sim)

            if (b_idx + 1) % 40 == 0 or (b_idx + 1) == len(loader):
                print(f"  [{b_idx+1}/{len(loader)}] LPIPS={np.mean(lpips_scores):.4f} PSNR={np.mean(psnr_scores):.2f}dB SSIM={np.mean(ssim_scores):.4f} CSD_Sim={np.mean(csd_sim_scores):.4f}", flush=True)

    z_all = torch.cat(latents_mu, dim=0) # (2048, 4, 64, 64)
    y_all = torch.cat(style_labels, dim=0).numpy()
    ar_all = torch.cat(artist_labels, dim=0).numpy()

    # Phase 2: Topology
    print("\n[Phase 2] Latent Topology, AUROC & 5-NN Retrieval...", flush=True)
    latent_std = float(z_all.std().item())
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

    # Phase 3: P1 Style Probes on GPU
    print("\n[Phase 3] Training 3 Independent Style Probes on GPU (136 Classes)...", flush=True)
    z_tr, y_tr = z_all[tr_idx], torch.from_numpy(y_all[tr_idx])
    z_ev, y_ev = z_all[val_idx], torch.from_numpy(y_all[val_idx])

    probes = {}
    for p_name, p_cls in (
        ("Linear_Flat", LinearFlatProbe),
        ("Linear_Stats", LinearStatsProbe),
        ("Shallow_CNN", ShallowCNNProbe),
    ):
        p_obj = p_cls(num_classes=num_classes)
        p_res = train_eval_probe_gpu(p_obj, z_tr, y_tr, z_ev, y_ev, num_epochs=15, device=device)
        unseen_res = run_unseen_artist_test_gpu(p_cls, z_all, y_all, ar_all, num_classes=num_classes, device=device)
        p_res["Unseen_Artist_Top1"] = unseen_res["Top1"]
        p_res["Unseen_Artist_Macro_F1"] = unseen_res["Macro_F1"]
        probes[p_name] = p_res
        print(f"  -> {p_name:12s}: Top-1={p_res['Top1_Accuracy']:5.2f}% | Top-5={p_res['Top5_Accuracy']:5.2f}% | F1={p_res['Macro_F1']:4.2f}% | Unseen-Artist Top-1={p_res['Unseen_Artist_Top1']:5.2f}%")

    cur_lpips = float(np.mean(lpips_scores))
    cur_psnr = float(np.mean(psnr_scores))
    b0_lpips = b0_benchmark["P0_reconstruction_quality"]["LPIPS"]
    b0_psnr = b0_benchmark["P0_reconstruction_quality"]["PSNR_dB"]

    res = {
        "model_name": model_name,
        "vae_path": vae_path,
        "wall_sec": round(time.time() - t0, 1),
        "P0_reconstruction": {
            "LPIPS": round(cur_lpips, 4),
            "PSNR_dB": round(cur_psnr, 2),
            "SSIM": round(float(np.mean(ssim_scores)), 4),
            "L1": round(float(np.mean(l1_scores)), 4),
            "CSD_Style_Fidelity": round(float(np.mean(csd_sim_scores)), 4),
            "latent_std": round(latent_std, 4),
            "gate_lpips_pass": bool(cur_lpips <= 1.08 * b0_lpips),
            "gate_psnr_pass": bool(cur_psnr >= b0_psnr - 0.50),
            "gate_stability_pass": bool(latent_std > 0.50),
            "overall_p0_passed": bool(cur_lpips <= 1.08 * b0_lpips and cur_psnr >= b0_psnr - 0.50 and latent_std > 0.50),
        },
        "P1_style_separability": {
            "probes": probes,
            "style_pair_auroc": style_auroc,
            "kNN_5_retrieval": round(knn_top1, 2),
            "intra_class_variance": intra_var,
            "effective_rank": eff_rank,
        },
    }
    return res


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Full GPU Evaluation Suite on {torch.cuda.get_device_name(0)} ===", flush=True)

    manifest_path = "/root/Workspace/data/wikiart_512_raw/wikiart_512_raw_manifest.json"
    dataset = RawMemmapWikiArtDataset(Path(manifest_path), is_training=False)
    num_classes = dataset.manifest["num_styles"]
    print(f"[Dataset] Val size: {len(dataset)} | Number of classes: {num_classes}", flush=True)

    b0_path = "/root/Workspace/WEAVE_VAE/b0_benchmark_full.json"
    b0_data = json.loads(Path(b0_path).read_text(encoding="utf-8"))

    # Shared evaluation models
    lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    csd, _, _ = csd_model.load_csd(device)

    models_to_eval = [
        ("B0_Baseline", "/root/Workspace/xy/DiT/data/pretrained/sd-vae-ft-ema"),
        ("A0_Recon_KL", "/root/Workspace/WEAVE_VAE/runs_vae/A0/final_diffusers_vae"),
        ("A1_Style_CE", "/root/Workspace/WEAVE_VAE/runs_vae/A1/final_diffusers_vae"),
        ("A2_MoCo_SupCon", "/root/Workspace/WEAVE_VAE/runs_vae/A2/final_diffusers_vae"),
    ]

    all_results = {}
    for m_name, m_path in models_to_eval:
        if not Path(m_path).exists():
            print(f"[WARN] Path {m_path} does not exist, skipping.")
            continue
        eval_res = evaluate_model_on_gpu(
            m_name, m_path, dataset, lpips_fn, csd, b0_data, num_classes, device=device, batch_size=16
        )
        all_results[m_name] = eval_res

    out_file = Path("/root/Workspace/WEAVE_VAE/gpu_eval_summary.json")
    out_file.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[Saved] Wrote full evaluation summary to {out_file}", flush=True)


if __name__ == "__main__":
    main()
