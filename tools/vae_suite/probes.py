"""WEAVE Art-VAE Suite: P1 Independent Style Probes Evaluation (137 Classes).

Trains 3 independent probes from scratch for each VAE model to test style separability:
1. Linear-Flat: Linear on spatial 8x8 pooled latent (256-d).
2. Linear-Stats: Linear on global mean, std, and DWT subband statistics (40-d).
3. Shallow-CNN: 2-layer ConvNet on raw 4x64x64 latent.

Outputs:
- 137-class Macro Top-1 Accuracy (%)
- Macro-F1 score
- Top-5 Accuracy (%)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score

from tools.v6.canonicalize import dwt2_haar
from tools.vae_suite.dataset import ValDataset2048

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


class LinearFlatProbe(nn.Module):
    def __init__(self, num_classes: int = 137):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((8, 8))
        self.fc = nn.Linear(4 * 8 * 8, num_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(self.pool(z).flatten(1))


class LinearStatsProbe(nn.Module):
    def __init__(self, num_classes: int = 136):
        super().__init__()
        # mean (4) + std (4) + Gram upper-triangle (10) + DWT subbands (32) = 50-d
        self.fc = nn.Linear(50, num_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B, C, H, W = z.shape
        m = z.mean(dim=(2, 3))
        s = z.std(dim=(2, 3)).clamp(min=1e-5)
        # Gram matrix: (B, 4, 4)
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
    def __init__(self, num_classes: int = 137):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
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


def extract_latents_for_dataset(vae: nn.Module, dataset: Dataset, device: str = "cuda", batch_size: int = 32):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    z_all, y_all, ar_all = [], [], []
    with torch.inference_mode():
        for b in loader:
            x = b["image"].to(device)
            mu = vae.encode(x).latent_dist.mean.cpu()
            z_all.append(mu)
            y_all.append(b["style_id"])
            ar_all.append(b.get("artist_id", torch.full_like(b["style_id"], -1)))
    return torch.cat(z_all, dim=0), torch.cat(y_all, dim=0), torch.cat(ar_all, dim=0).numpy()


def train_and_eval_single_probe(probe: nn.Module, z_train, y_train, z_val, y_val, num_epochs: int = 15, lr: float = 1e-3, device: str = "cuda"):
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

    top1_acc = float((preds_arr == targets_arr).mean() * 100.0)
    top5_acc = float(top5_hits / len(targets_arr) * 100.0)
    macro_f1 = float(f1_score(targets_arr, preds_arr, average="macro", zero_division=0) * 100.0)

    return {
        "Top1_Accuracy": round(top1_acc, 2),
        "Top5_Accuracy": round(top5_acc, 2),
        "Macro_F1": round(macro_f1, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae-path", required=True)
    ap.add_argument("--val-manifest", default="/root/Workspace/WEAVE_VAE/val_2048_manifest.json")
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== P1 Independent Style Probes Evaluation for {args.vae_path} ===", flush=True)

    val_dataset = ValDataset2048(Path(args.val_manifest))
    num_classes = 136

    vae = AutoencoderKL.from_pretrained(args.vae_path).to(device).eval()

    print("[Probes] Pre-extracting latents from 2048 validation set...", flush=True)
    z_val, y_val, ar_val = extract_latents_for_dataset(vae, val_dataset, device=device)

    # Split 2048 into 1500 train-probe and 548 eval-probe with fixed seed
    rng = np.random.default_rng(42)
    indices = np.arange(len(z_val))
    rng.shuffle(indices)
    tr_idx, val_idx = indices[:1500], indices[1500:]

    z_tr, y_tr = z_val[tr_idx], y_val[tr_idx]
    z_ev, y_ev = z_val[val_idx], y_val[val_idx]

    from tools.vae_suite.eval_b0_complete import run_unseen_artist_test

    results = {}
    for name, p_cls in (
        ("Linear_Flat", LinearFlatProbe),
        ("Linear_Stats", LinearStatsProbe),
        ("Shallow_CNN", ShallowCNNProbe),
    ):
        print(f"  [Training Probe] {name} (epochs=15, classes={num_classes})...", flush=True)
        probe = p_cls(num_classes=num_classes)
        p_res = train_and_eval_single_probe(probe, z_tr, y_tr, z_ev, y_ev, num_epochs=15, device=device)
        
        # Unseen artist generalization test
        unseen_res = run_unseen_artist_test(p_cls, z_val, y_val.numpy(), ar_val, num_classes=num_classes, device=device)
        p_res["Unseen_Artist_Top1"] = unseen_res["Top1"]
        p_res["Unseen_Artist_Macro_F1"] = unseen_res["Macro_F1"]

        results[name] = p_res
        print(f"    -> {name}: Top-1={p_res['Top1_Accuracy']}% | Top-5={p_res['Top5_Accuracy']}% | Macro-F1={p_res['Macro_F1']}% | Unseen-Artist Top-1={p_res['Unseen_Artist_Top1']}%")

    out_p = Path(args.output_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps({
        "model_path": args.vae_path,
        "probes": results,
    }, indent=2), encoding="utf-8")
    print(f"[Done] Saved probe evaluation results to {out_p}")


if __name__ == "__main__":
    main()
