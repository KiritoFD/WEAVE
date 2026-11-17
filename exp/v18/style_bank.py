"""V18 Style Sub-Mode Prototype Bank & Fast Pre-Decoded Structural Cache.

Strictly fit on the clean, deduplicated, 1:1:1:1:1 balanced training pool
(excluding all Test750 sources and all references_v1 paintings).
"""
from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn.functional as F

from . import OUT, STYLES, clean_balanced_pool, save_json, tasks_and_sources
from .structure import (
    extract_whitened_dino_layout,
    fit_latent_luma_proj,
    rgb_to_gray,
    rgb_to_structure_64,
)


def _kmeans_torch(X: torch.Tensor, k: int = 8, iters: int = 30, seed: int = 42) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic L2 K-means in PyTorch. X: (N, D) -> (centers (k, D), labels (N,))."""
    g = torch.Generator(device=X.device).manual_seed(seed)
    N = X.shape[0]
    perm = torch.randperm(N, generator=g, device=X.device)[:k]
    centers = X[perm].clone()
    labels = torch.zeros(N, dtype=torch.long, device=X.device)
    for _ in range(iters):
        dists = torch.cdist(X, centers)
        labels = dists.argmin(dim=1)
        for c in range(k):
            m = labels == c
            if m.any():
                centers[c] = X[m].mean(dim=0)
    return centers, labels


@torch.no_grad()
def build_submode_bank(z_train: torch.Tensor, sid_train: torch.Tensor, k_submodes: int = 8, device: str = "cuda") -> dict:
    """Extract standardized 48-d wavelet style vectors & fit K=8 sub-modes per style."""
    from tools.v6.canonicalize import extract_band_stats_features
    from src.infer import load_critic
    from exp.v8.guidance import target_margin

    raw_feats = []
    for st in range(0, len(z_train), 256):
        raw_feats.append(extract_band_stats_features(z_train[st:st + 256].to(device)).cpu())
    feats = torch.cat(raw_feats, dim=0).float()  # (N, 48)
    f_mean = feats.mean(dim=0, keepdim=True)
    f_std = feats.std(dim=0, keepdim=True).clamp_min(1e-4)
    feats_norm = F.normalize((feats - f_mean) / f_std, dim=-1)  # unit-sphere 48-d

    critic = load_critic(device)
    r1_margins = []
    for st in range(0, len(z_train), 128):
        zb = z_train[st:st + 128].to(device).float()
        sb = sid_train[st:st + 128].to(device)
        r1_margins.append(target_margin(critic, zb, sb).cpu())
    r1_all = torch.cat(r1_margins, dim=0)
    del critic
    torch.cuda.empty_cache()

    mean_protos = torch.zeros(5, 48)
    vivid_protos = torch.zeros(5, 48)
    submode_centers = torch.zeros(5, k_submodes, 48)
    sample_proto = torch.zeros_like(feats_norm)
    cluster_summary = {}

    for s in range(5):
        mask = sid_train == s
        Xs = feats_norm[mask].to(device)
        ms = r1_all[mask].to(device)
        centers, labels = _kmeans_torch(Xs, k=k_submodes, iters=35, seed=100 + s)
        centers = F.normalize(centers, dim=-1)
        submode_centers[s] = centers.cpu()
        mean_protos[s] = F.normalize(Xs.mean(dim=0), dim=-1).cpu()

        # Rank sub-modes by mean R1 target margin on training samples (requiring >=15 paintings)
        best_c, best_score = 0, -1e9
        c_scores = []
        for c in range(k_submodes):
            cm = labels == c
            cnt = int(cm.sum().item())
            sc = float(ms[cm].mean().item()) if cnt >= 15 else -1e9
            c_scores.append({"cluster": c, "count": cnt, "r1_margin": round(sc, 4)})
            if sc > best_score:
                best_score, best_c = sc, c
        vivid_protos[s] = centers[best_c].cpu()
        sample_proto[mask] = centers[labels].cpu()
        cluster_summary[STYLES[s]] = {"best_submode": best_c, "clusters": c_scores}

    save_json(OUT / "submode_bank_summary.json", cluster_summary)
    return {
        "feat_mean": f_mean,
        "feat_std": f_std,
        "mean_protos": mean_protos,        # (5, 48)
        "vivid_protos": vivid_protos,      # (5, 48)
        "submode_centers": submode_centers,# (5, K, 48)
        "sample_proto": sample_proto,      # (N_train, 48)
    }


@torch.no_grad()
def build_or_load_ccf_cache(
    pool_dir: str | Path,
    train_per_style: int = 880,
    hold_per_style: int = 60,
    device: str = "cuda",
) -> tuple[dict, dict]:
    """Build or load cached (c_64, dino_16, y_64, style_bank, luma_proj) + Test750 source structs."""
    cache_path = OUT / "cache/ccf_pool_cache.pt"
    test_cache_path = OUT / "cache/test750_struct_cache.pt"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists() and test_cache_path.exists():
        print(f"[V18 Cache] Reusing existing {cache_path.name} & {test_cache_path.name}", flush=True)
        return (
            torch.load(cache_path, map_location="cpu", weights_only=False),
            torch.load(test_cache_path, map_location="cpu", weights_only=False),
        )

    t0 = time.time()
    splits, manifest = clean_balanced_pool(pool_dir, train_per_style, hold_per_style)
    save_json(OUT / "cache/pool_manifest.json", manifest)

    from src.utils import load_s10_decoder, decode_s10
    from src.utils._local_eval_refs import load_dino_model

    dec = load_s10_decoder(device)
    dino = load_dino_model(device=device)

    for sp_name, sp in splits.items():
        z_all = sp["z"]
        c64_list, dino_list, y64_list = [], [], []
        for st in range(0, len(z_all), 32):
            zb = z_all[st:st + 32].to(device)
            rgb = decode_s10(dec, zb, batch=8).to(device).clamp(0.0, 1.0)
            c64 = rgb_to_structure_64(rgb).cpu().half()
            y64 = F.avg_pool2d(rgb_to_gray(rgb), kernel_size=8, stride=8).cpu().half()
            d16 = extract_whitened_dino_layout(dino, rgb, batch=16, device=device).cpu().half()
            c64_list.append(c64)
            y64_list.append(y64)
            dino_list.append(d16)
            if (st // 32) % 20 == 0:
                print(f"[V18 Cache] {sp_name} {st+len(zb)}/{len(z_all)} ({time.time()-t0:.0f}s)", flush=True)
        sp["c_64"] = torch.cat(c64_list, dim=0)
        sp["y_64"] = torch.cat(y64_list, dim=0)
        sp["dino_16"] = torch.cat(dino_list, dim=0)

    #Also precompute deterministic source structures for the 750 test tasks (150 unique sources)
    tasks, z_src_750, _ = tasks_and_sources()
    c64_t, d16_t = [], []
    for st in range(0, 750, 32):
        zb = z_src_750[st:st + 32].to(device)
        rgb = decode_s10(dec, zb, batch=8).to(device).clamp(0.0, 1.0)
        c64_t.append(rgb_to_structure_64(rgb).cpu().half())
        d16_t.append(extract_whitened_dino_layout(dino, rgb, batch=16, device=device).cpu().half())
    test_struct = {
        "c_64": torch.cat(c64_t, dim=0),
        "dino_16": torch.cat(d16_t, dim=0),
    }

    del dec, dino
    torch.cuda.empty_cache()

    bank = build_submode_bank(splits["train"]["z"], splits["train"]["sid"], k_submodes=8, device=device)
    w_luma, b_luma = fit_latent_luma_proj(splits["train"]["z"], splits["train"]["y_64"].float())

    full_cache = {
        "train": splits["train"],
        "holdout": splits["holdout"],
        "style_bank": bank,
        "luma_proj": {"w": w_luma, "b": b_luma},
        "wall_sec": round(time.time() - t0, 1),
    }
    torch.save(full_cache, cache_path)
    torch.save(test_struct, test_cache_path)
    print(f"[V18 Cache] Saved {cache_path.name} & {test_cache_path.name} in {time.time()-t0:.1f}s", flush=True)
    return full_cache, test_struct
