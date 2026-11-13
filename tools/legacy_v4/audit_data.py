"""P0 数据审计：latent 口径、划分/作者留出可行性、去重、内容簇、跨风格近邻、DINO 对照。

只产出报告，不改数据。DINOv2 只在这里作为「内容簇是否可信」的对照，绝不进训练。
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import (STYLE_NAMES, build_split, content_clusters, cross_style_topk,
                   duplicate_groups, latent_descriptor, load_bank, save_json)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/v4/p0")
    ap.add_argument("--dino-per-style", type=int, default=48)
    ap.add_argument("--vae-cache", default="/mnt/i/Latent_Style/eval_cache/hf")
    return ap.parse_args()


def latent_stats(bank) -> dict:
    out = {}
    for sid, name in enumerate(STYLE_NAMES[: bank.num_styles]):
        z = bank.latents[bank.style_ids == sid]
        out[name] = {
            "n": int(z.shape[0]),
            "mean": float(z.mean()),
            "std": float(z.std()),
            "per_channel_std": [round(float(v), 4) for v in z.std(dim=(0, 2, 3))],
            "per_image_std_min": float(z.flatten(1).std(1).min()),
            "per_image_std_median": float(z.flatten(1).std(1).median()),
        }
    allz = bank.latents
    out["_all"] = {
        "mean": float(allz.mean()), "std": float(allz.std()),
        "implied_raw_std_if_scaled_0.18215": float(allz.std() / 0.18215),
    }
    return out


def normalize_mutual_info(a: np.ndarray, b: np.ndarray) -> float:
    ka, kb = a.max() + 1, b.max() + 1
    n = len(a)
    mi = 0.0
    for i in range(ka):
        for j in range(kb):
            nij = np.sum((a == i) & (b == j))
            if nij == 0:
                continue
            ni, nj = np.sum(a == i), np.sum(b == j)
            mi += (nij / n) * math.log((nij * n) / (ni * nj))
    ha = -sum((np.sum(a == i) / n) * math.log(np.sum(a == i) / n) for i in range(ka) if np.sum(a == i))
    hb = -sum((np.sum(b == j) / n) * math.log(np.sum(b == j) / n) for j in range(kb) if np.sum(b == j))
    return float(mi / max(1e-8, 0.5 * (ha + hb)))


@torch.no_grad()
def dino_context(bank, *, per_style: int, cache_dir: str) -> dict:
    """解码少量真实 latent，用 DINOv2-small 做内容簇对照（含风格信息，仅作 sanity check）。"""
    import os
    os.environ.setdefault("HF_HOME", cache_dir)
    from transformers import AutoModel
    from utils.inference import decode_latent, load_vae

    rng = np.random.default_rng(0)
    picked = []
    for sid in range(bank.num_styles):
        idx = np.nonzero(bank.style_ids.numpy() == sid)[0]
        picked.extend(rng.choice(idx, size=min(per_style, len(idx)), replace=False).tolist())
    z = bank.latents[torch.tensor(picked)].cuda()
    vae = load_vae("cuda", model_id="ema", cache_dir=cache_dir)
    imgs = []
    for s in range(0, z.shape[0], 8):
        imgs.append(decode_latent(vae, z[s:s + 8], "cuda"))
    imgs = torch.cat(imgs, 0)
    model = AutoModel.from_pretrained("facebook/dinov2-small", local_files_only=True).cuda().eval()
    feats = []
    for s in range(0, imgs.shape[0], 16):
        out = model(pixel_values=imgs[s:s + 16] * 2.0 - 1.0)
        feats.append(out.last_hidden_state[:, 0].float())
    dino = torch.cat(feats, 0)
    dino = F.normalize(dino, dim=1)
    labels_dino = content_clusters(dino.cpu(), k=8, seed=0)
    desc = latent_descriptor(bank.latents[torch.tensor(picked)])
    labels_lat = content_clusters(desc, k=8, seed=0)
    sid = bank.style_ids[torch.tensor(picked)].numpy()
    purity = float(np.mean([np.bincount(labels_lat[labels_dino == c]).max() / max(1, (labels_dino == c).sum())
                            for c in np.unique(labels_dino)]))
    return {
        "subset": int(len(picked)),
        "nmi_dino_vs_latent_clusters": normalize_mutual_info(labels_dino, labels_lat),
        "latent_cluster_purity_vs_dino": purity,
        "latent_cluster_style_entropy": float(-sum(
            (np.sum(sid == s) / len(sid)) * math.log(np.sum(sid == s) / len(sid))
            for s in range(bank.num_styles) if np.sum(sid == s) > 0)),
        "dino_available": True,
    }


@torch.no_grad()
def vae_roundtrip(cache_dir: str, *, n: int = 8) -> dict:
    """测试图 512 编码-解码往返一致性（只验证口径，不参与训练）。"""
    from pathlib import Path as P
    from PIL import Image
    import numpy as np
    from utils.inference import decode_latent, encode_image, load_vae
    vae = load_vae("cuda", model_id="ema", cache_dir=cache_dir)
    paths = sorted(P("data/test").glob("*/*.jpg"))[:n]
    arrs = []
    for p in paths:
        img = Image.open(p).convert("RGB").resize((512, 512))
        arrs.append(torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float() / 255.0)
    x = torch.stack(arrs, 0).cuda() * 2.0 - 1.0
    lat = encode_image(vae, x, "cuda")
    rec = decode_latent(vae, lat, "cuda")
    mse = float(((rec * 2.0 - 1.0) - x).pow(2).mean())
    return {"n": int(n), "latent_std": float(lat.std()), "roundtrip_mse": mse,
            "latent_scale_used": float(getattr(vae.config, "scaling_factor", 0.18215))}


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bank = load_bank("data/train/.latent_cache/packed/packed")
    desc = latent_descriptor(bank.latents)
    dup_group, dup_pairs = duplicate_groups(desc)
    clusters = content_clusters(desc, k=20, seed=0)
    per_style_clusters = []
    sid = bank.style_ids.numpy()
    for s in range(bank.num_styles):
        per_style_clusters.append(int(len(np.unique(clusters[sid == s])))
                                  if (sid == s).any() else 0)
    neighbors = cross_style_topk(desc, bank.style_ids, k=5)
    nn_cos = []
    for i in range(0, 500):
        js = neighbors[i]
        nn_cos.append(float((desc[i:i + 1] * desc[js]).sum(1).mean()))
    train_idx, val_idx, split = build_split(bank, val_per_style=100, seed=0, dup_group=dup_group)
    report = {
        "latent_stats": latent_stats(bank),
        "dedup": {"threshold": 0.995, "pairs": int(dup_pairs), "groups": int(dup_group.max()) + 1},
        "clusters": {"k": 20, "per_style_unique": per_style_clusters},
        "cross_style_top5_mean_cos_sampled": float(np.mean(nn_cos)),
        "split": split,
        "split_sizes": {"train": int(len(train_idx)), "val": int(len(val_idx))},
        "vae_roundtrip": vae_roundtrip(args.vae_cache),
    }
    try:
        report["dino_context"] = dino_context(bank, per_style=args.dino_per_style,
                                              cache_dir=args.vae_cache)
    except Exception as exc:  # 审计失败不阻塞训练，但必须留下痕迹
        report["dino_context"] = {"error": f"{type(exc).__name__}: {exc}"}
    save_json(out / "data_report.json", report)
    torch.save({"dup_group": torch.from_numpy(dup_group), "cluster": torch.from_numpy(clusters),
                "neighbors": neighbors, "train_idx": torch.from_numpy(train_idx),
                "val_idx": torch.from_numpy(val_idx)}, out / "indices.pt")
    print(json.dumps(report, indent=2, ensure_ascii=False)[:4000])


if __name__ == "__main__":
    main()
