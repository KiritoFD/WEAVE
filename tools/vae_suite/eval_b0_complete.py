"""WEAVE Art-VAE Suite: Comprehensive B0 Baseline Benchmark & Style Separability Suite.

Evaluates B0 (Original sd-vae-ft-ema) on the frozen 2048-image independent validation set across:
1. Reconstruction Quality:
   - LPIPS (AlexNet)
   - PSNR (dB)
   - SSIM
   - L1 reconstruction error
   - Posterior variance & stability
2. Style Separability across 136 WikiArt Style Classes (P1 Probes):
   - Linear-Flat Probe (8x8 spatial pooled latent, 256-d -> 136 classes)
   - Linear-Stats Probe (Mean, Std, DWT subband statistics, 40-d -> 136 classes)
   - Shallow-CNN Probe (2-layer ConvNet on raw 4x64x64 latent)
   - Metrics: Macro Top-1 (%), Top-5 (%), Macro-F1 (%)
3. Latent Geometry & Retrieval:
   - k-NN (k=5) Style Retrieval Accuracy (%)
   - Effective Rank (singular value entropy)
4. CSD Art Style Preservation:
   - CSD cosine fidelity before vs after VAE roundtrip
"""
from __future__ import annotations

import argparse
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
from sklearn.metrics import f1_score
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
        # mean (4) + std (4) + Gram upper-triangle (10) + DWT subband stats (32) = 50-d
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B, C, H, W = z.shape
        m = z.mean(dim=(2, 3))
        s = z.std(dim=(2, 3)).clamp(min=1e-5)
        # Gram matrix across 4 latent channels: (B, 4, 4)
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
    """Compute effective rank (entropy of singular values) of flattened latent matrix."""
    mat = z_tensor.flatten(1).float()
    mat = mat - mat.mean(dim=0, keepdim=True)
    _, s, _ = torch.svd(mat)
    s_norm = s / s.sum()
    entropy = -torch.sum(s_norm * torch.log(s_norm.clamp(min=1e-12))).item()
    eff_rank = float(math.exp(entropy))
    return round(eff_rank, 2)


def compute_style_pair_auroc(z_feats: torch.Tensor, styles: np.ndarray, n_pairs: int = 10000, seed: int = 42) -> float:
    """Compute AUROC for discriminating same-style vs different-style pairs using latent cosine similarity."""
    from sklearn.metrics import roc_auc_score
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
    """Compute mean intra-class variance on normalized features (monitors over-clustering collapse)."""
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


def run_unseen_artist_test(probe_cls, z_all: torch.Tensor, y_all: np.ndarray, artists_all: np.ndarray, num_classes: int, device: str = "cuda") -> dict:
    """Evaluate generalization on unseen artists: train probe on 75% artists, test on 25% unseen artists."""
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
    res = train_eval_probe(probe, z_tr, y_tr, z_te, y_te, num_epochs=15, device=device)
    return {
        "Top1": res["Top1_Accuracy"],
        "Macro_F1": res["Macro_F1"],
    }


def train_eval_probe(probe: nn.Module, z_train, y_train, z_val, y_val, num_epochs: int = 15, lr: float = 1e-3, device: str = "cuda"):
    probe = probe.to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    train_ds = torch.utils.data.TensorDataset(z_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    for epoch in range(num_epochs):
        probe.train()
        for z_b, y_b in train_loader:
            z_b, y_b = z_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = probe(z_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()

    probe.eval()
    val_ds = torch.utils.data.TensorDataset(z_val, y_val)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    preds_all = []
    targets_all = []
    top5_hits = 0

    with torch.no_grad():
        for z_b, y_b in val_loader:
            z_b = z_b.to(device)
            logits = probe(z_b)
            pred = logits.argmax(dim=1).cpu()
            preds_all.append(pred)
            targets_all.append(y_b)

            top5 = torch.topk(logits, k=min(5, logits.shape[1]), dim=1).indices.cpu()
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


def evaluate_b0_complete(
    vae_path: str,
    manifest_path: str,
    output_path: str,
    device: str = "cuda",
) -> dict:
    print(f"=== [B0 Baseline Benchmark] Comprehensive Evaluation of {vae_path} ===", flush=True)
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # 1. Dataset: Val subset from Memory-Mapped RAW Binary
    dataset = RawMemmapWikiArtDataset(Path(manifest_path), is_training=False)
    num_classes = dataset.manifest["num_styles"]
    print(f"[Dataset] Val set size: {len(dataset)} | Number of style classes: {num_classes}", flush=True)

    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    # 2. Load Models
    vae = AutoencoderKL.from_pretrained(vae_path).to(device).eval()
    lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    # Load CSD for art style fidelity
    csd, _, _ = csd_model.load_csd(device)

    # 3. Phase 1: Reconstruction & Latent Extraction
    print("\n--- Phase 1: Evaluating Reconstruction Quality & Extracting Latents ---", flush=True)
    t0 = time.time()

    lpips_scores, psnr_scores, ssim_scores, l1_scores, var_scores = [], [], [], [], []
    latents_mu, style_labels, artist_labels = [], [], []
    csd_sim_scores = []

    with torch.inference_mode():
        for b_idx, batch in enumerate(loader):
            x = batch["image"].to(device)
            y = batch["style_id"]
            ar = batch["artist_id"]

            posterior = vae.encode(x).latent_dist
            mu = posterior.mean
            var_scores.append(posterior.var.mean().item())

            latents_mu.append(mu.cpu())
            style_labels.append(y)
            artist_labels.append(ar)

            # Reconstruct from posterior mean
            recon = vae.decode(mu).sample

            # Metrics
            l1_scores.append(nn.functional.l1_loss(recon, x).item())
            lpips_scores.append(lpips_fn(recon.float(), x.float()).mean().item())
            psnr_scores.append(compute_psnr(recon, x))
            ssim_scores.append(compute_ssim_simple(recon, x))

            # CSD Art Style Cosine Fidelity before vs after reconstruction
            recon_01 = (recon.clamp(-1, 1) + 1.0) / 2.0
            x_01 = (x.clamp(-1, 1) + 1.0) / 2.0
            
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                x_224 = nn.functional.interpolate(x_01, size=(224, 224), mode="bicubic", align_corners=False)
                recon_224 = nn.functional.interpolate(recon_01, size=(224, 224), mode="bicubic", align_corners=False)
                norm_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
                norm_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
                x_csd = (x_224 - norm_mean) / norm_std
                recon_csd = (recon_224 - norm_mean) / norm_std
                _, _, sty_orig = csd(x_csd.half())
                _, _, sty_recon = csd(recon_csd.half())
                csd_sim = torch.sum(sty_recon * sty_orig, dim=-1).mean().item()
            csd_sim_scores.append(csd_sim)

            if (b_idx + 1) % 40 == 0 or (b_idx + 1) == len(loader):
                print(f"  [{b_idx+1}/{len(loader)}] LPIPS={np.mean(lpips_scores):.4f} PSNR={np.mean(psnr_scores):.2f}dB SSIM={np.mean(ssim_scores):.4f} CSD_Sim={np.mean(csd_sim_scores):.4f}", flush=True)

    recon_duration = time.time() - t0
    z_all = torch.cat(latents_mu, dim=0) # (2048, 4, 64, 64)
    y_all = torch.cat(style_labels, dim=0) # (2048,)
    artists_all = torch.cat(artist_labels, dim=0).numpy() # (2048,)

    # 4. Phase 2: Latent Geometry, AUROC & Unsupervised Retrieval
    print("\n--- Phase 2: Analyzing Latent Geometry, AUROC & Retrieval ---", flush=True)
    eff_rank = compute_effective_rank(z_all)
    print(f"  Effective Rank of Latent Space: {eff_rank} / {4*64*64}", flush=True)

    # Style Pair AUROC (Same vs Diff Style)
    style_auroc = compute_style_pair_auroc(z_all.flatten(1), y_all.numpy(), n_pairs=10000)
    print(f"  Style Pair Discrimination AUROC: {style_auroc:.2f}%", flush=True)

    # Mean Intra-Class Variance (monitors over-clustering collapse)
    intra_var = compute_intra_class_variance(z_all.flatten(1), y_all.numpy())
    print(f"  Mean Intra-Class Variance: {intra_var:.4f}", flush=True)

    # 5-NN Retrieval Accuracy on Style Classes
    knn = KNeighborsClassifier(n_neighbors=5, metric="cosine")
    z_pool = nn.functional.adaptive_avg_pool2d(z_all, (8, 8)).flatten(1).numpy()

    # Fixed 1500 train, 548 test split for probes and kNN
    rng = np.random.default_rng(42)
    indices = np.arange(len(z_all))
    rng.shuffle(indices)
    tr_idx, val_idx = indices[:1500], indices[1500:]

    knn.fit(z_pool[tr_idx], y_all[tr_idx].numpy())
    knn_preds = knn.predict(z_pool[val_idx])
    knn_top1 = float((knn_preds == y_all[val_idx].numpy()).mean() * 100.0)
    print(f"  5-NN Style Retrieval Accuracy: {knn_top1:.2f}%", flush=True)

    # 5. Phase 3: P1 Independent Probes & Unseen Artist Generalization
    print("\n--- Phase 3: Training 3 Independent Style Probes from Scratch ---", flush=True)
    z_tr, y_tr = z_all[tr_idx], y_all[tr_idx]
    z_ev, y_ev = z_all[val_idx], y_all[val_idx]

    probes_results = {}
    for name, p_cls in (
        ("Linear_Flat", LinearFlatProbe),
        ("Linear_Stats", LinearStatsProbe),
        ("Shallow_CNN", ShallowCNNProbe),
    ):
        t_pr = time.time()
        print(f"  [Probe: {name}] Training 15 epochs on {len(z_tr)} samples...", flush=True)
        probe = p_cls(num_classes=num_classes)
        p_res = train_eval_probe(probe, z_tr, y_tr, z_ev, y_ev, num_epochs=15, device=device)
        p_res["train_sec"] = round(time.time() - t_pr, 1)

        # Unseen Artist Generalization Test
        unseen_res = run_unseen_artist_test(p_cls, z_all, y_all.numpy(), artists_all, num_classes=num_classes, device=device)
        p_res["Unseen_Artist_Top1"] = unseen_res["Top1"]
        p_res["Unseen_Artist_Macro_F1"] = unseen_res["Macro_F1"]

        probes_results[name] = p_res
        print(f"    -> {name}: Top-1={p_res['Top1_Accuracy']}% | Top-5={p_res['Top5_Accuracy']}% | Macro-F1={p_res['Macro_F1']}% | Unseen-Artist Top-1={p_res['Unseen_Artist_Top1']}% ({p_res['train_sec']}s)")

    # 6. Aggregate Final Report
    report = {
        "model_name": "B0_Original_sd-vae-ft-ema",
        "vae_path": vae_path,
        "n_validation_images": len(dataset),
        "num_style_classes": num_classes,
        "eval_wall_sec": round(time.time() - t0, 1),
        "P0_reconstruction_quality": {
            "LPIPS": round(float(np.mean(lpips_scores)), 4),
            "PSNR_dB": round(float(np.mean(psnr_scores)), 2),
            "SSIM": round(float(np.mean(ssim_scores)), 4),
            "L1": round(float(np.mean(l1_scores)), 4),
            "CSD_Style_Fidelity": round(float(np.mean(csd_sim_scores)), 4),
            "mean_posterior_var": round(float(np.mean(var_scores)), 4),
        },
        "P1_style_separability": {
            "probes_136_classes": probes_results,
            "kNN_k5_retrieval_acc": round(knn_top1, 2),
            "style_pair_auroc": style_auroc,
            "mean_intra_class_variance": intra_var,
            "latent_effective_rank": eff_rank,
        },
    }

    out_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[Saved] Comprehensive B0 baseline report saved to {out_file}", flush=True)

    # Output Clean Markdown Table
    lines = []
    lines.append("# B0 原始 VAE 完整基准评测报告 (sd-vae-ft-ema)\n")
    lines.append("> 评估数据集：冻结的 2048 张独立 WikiArt 验证集；\n"
                 "> 风格类别：136 类官方 WikiArt 艺术流派；\n"
                 "> 评测包含：物理重建（LPIPS/PSNR/SSIM）、CSD画风保持、潜空间有效秩、5-NN检索与 3 种独立 Probe 风格可分性。\n")

    lines.append("## 一、P0 基础重建与物理保真度基线 (用于 A0/A1/A2 硬门槛对照)\n")
    lines.append("| 评测指标 | B0 实测基线数值 | 硬门槛要求 (A0/A1/A2 不得劣于) |")
    lines.append("|---|---:|---|")
    lines.append(f"| **LPIPS (AlexNet) $\\downarrow$** | **{report['P0_reconstruction_quality']['LPIPS']:.4f}** | $\\le {1.08*report['P0_reconstruction_quality']['LPIPS']:.4f}$ (不超 1.08 倍) |")
    lines.append(f"| **PSNR $\\uparrow$** | **{report['P0_reconstruction_quality']['PSNR_dB']:.2f} dB** | $\\ge {report['P0_reconstruction_quality']['PSNR_dB']-0.50:.2f} \\text{{ dB}}$ (不降超 0.5 dB) |")
    lines.append(f"| **SSIM $\\uparrow$** | **{report['P0_reconstruction_quality']['SSIM']:.4f}** | 保持高结构相似性 |")
    lines.append(f"| **CSD 风格保真度 $\\uparrow$** | **{report['P0_reconstruction_quality']['CSD_Style_Fidelity']:.4f}** | 编解码画风余弦保持率 |")
    lines.append(f"| **后验方差 (Posterior Var)** | **{report['P0_reconstruction_quality']['mean_posterior_var']:.4f}** | 防方差塌缩与数值 NaN |")

    lines.append("\n## 二、P1 风格可分性与表征质量基线 (136 类艺术画派)\n")
    lines.append("| 探测器 (Probe) / 表征评估 | 输入特征形式 | 136类 Top-1 (%) | 136类 Top-5 (%) | 136类 Macro-F1 (%) | 评价核心 |")
    lines.append("|---|---|---:|---:|---:|---|")
    lines.append(f"| **Linear-Flat** | 空间 $8\\times 8$ 池化 (256维) | **{probes_results['Linear_Flat']['Top1_Accuracy']:.2f}%** | **{probes_results['Linear_Flat']['Top5_Accuracy']:.2f}%** | **{probes_results['Linear_Flat']['Macro_F1']:.2f}%** | 线性空间可读性 |")
    lines.append(f"| **Linear-Stats** | 均值/方差/DWT统计 (40维) | **{probes_results['Linear_Stats']['Top1_Accuracy']:.2f}%** | **{probes_results['Linear_Stats']['Top5_Accuracy']:.2f}%** | **{probes_results['Linear_Stats']['Macro_F1']:.2f}%** | 全局风格统计量 |")
    lines.append(f"| **Shallow-CNN** | 原始 $4\\times 64\\times 64$ 潜变量 | **{probes_results['Shallow_CNN']['Top1_Accuracy']:.2f}%** | **{probes_results['Shallow_CNN']['Top5_Accuracy']:.2f}%** | **{probes_results['Shallow_CNN']['Macro_F1']:.2f}%** | 非线性潜空间信息 |")
    lines.append(f"| **5-NN 无监督检索** | 256维空间潜特征 | **{knn_top1:.2f}%** | — | — | 聚类纯度与拓扑结构 |")
    lines.append(f"| **有效奇异值秩 (Rank)** | 潜变量全维度 (16384维) | **{eff_rank}** | — | — | 潜空间流形表达自由度 |")

    md_path = DOCS_V6 / "B0_BASELINE_EVALUATION_REPORT.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[Done] Wrote markdown report to {md_path}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae-path", default="/root/Workspace/xy/DiT/data/pretrained/sd-vae-ft-ema")
    ap.add_argument("--manifest", default="/root/Workspace/data/wikiart_512_raw/wikiart_512_raw_manifest.json")
    ap.add_argument("--output", default="/root/Workspace/WEAVE_VAE/b0_benchmark_full.json")
    args = ap.parse_args()

    evaluate_b0_complete(args.vae_path, args.manifest, args.output)
