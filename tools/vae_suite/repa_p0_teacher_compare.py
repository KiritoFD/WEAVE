"""Frozen Teacher Cross-Evaluation: CSD vs ALADIN-ViT on identical protocol.

Same 2048 validation images, same pairs, same metrics as the P0 teacher test.
Also reports ALADIN's own dual-view consistency if applicable.

Usage:
  python repa_p0_teacher_compare.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/Workspace/WEAVE_VAE")
sys.path.insert(0, "/root/Workspace/WEAVE_VAE/tools/12h")
sys.path.insert(0, "/tmp/aladin_repo")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from tools.vae_suite.dataset import RawMemmapWikiArtDataset
from tools.vae_suite.repa_p0_teacher import (
    make_pairs, auroc_cosine, knn_purity, retrieval_map10,
)
import csd_model

OUT = Path("/root/Workspace/WEAVE_VAE/runs_vae/v2_diagnosis/15_teacher_compare.json")
MANIFEST = Path("/root/Workspace/data/wikiart_512_raw/wikiart_512_raw_manifest.json")
ALADIN_CKPT = "/tmp/aladin_repo/aladinVIT_bamfg_64.47.pt"

NORM_MEAN = [0.48145466, 0.4578275, 0.40821073]
NORM_STD = [0.26862954, 0.26130258, 0.27577711]


def build_aladin(device):
    from vit_networks import VisionTransformer
    import torch.nn as nn

    class AViT(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc_style = VisionTransformer()

        def encode(self, image):
            proj, _ = self.enc_style(image, just_style_code=True)
            return proj

    ck = torch.load(ALADIN_CKPT, map_location="cpu", weights_only=False)
    m = AViT()
    m.load_state_dict(ck)
    m.eval().to(device)
    return m


def extract_aladin(model, dataset, device, batch_size=16):
    """ALADIN expects [-1,1] normalized 224x224 (per official main.py)."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    feats = []
    with torch.no_grad():
        for i, b in enumerate(loader):
            x = b["image"].to(device)
            x224 = F.interpolate(x.clamp(-1, 1), (224, 224), mode="bicubic", align_corners=False)
            emb = model.encode(x224)
            feats.append(emb.float().cpu())
            if (i + 1) % 40 == 0:
                print(f"    ALADIN [{i+1}/{len(loader)}]", flush=True)
    f = torch.cat(feats).numpy()
    return f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)


def extract_csd(csd, dataset, device, batch_size=16):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    nm = torch.tensor(NORM_MEAN, device=device).view(1, 3, 1, 1)
    ns = torch.tensor(NORM_STD, device=device).view(1, 3, 1, 1)
    feats = []
    with torch.no_grad():
        for i, b in enumerate(loader):
            x = b["image"].to(device)
            x01 = (x.clamp(-1, 1) + 1.0) / 2.0
            x224 = F.interpolate(x01, (224, 224), mode="bicubic", align_corners=False)
            xin = ((x224 - nm) / ns).half()
            _, _, sty = csd(xin)
            feats.append(sty.float().cpu())
            if (i + 1) % 40 == 0:
                print(f"    CSD [{i+1}/{len(loader)}]", flush=True)
    f = torch.cat(feats).numpy()
    return f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)


def eval_desc(name, feats, styles, same, cross, diff, hard):
    print(f"\n--- {name} (dim={feats.shape[1]}) ---", flush=True)
    o = auroc_cosine(feats, same, diff)
    c = auroc_cosine(feats, cross, diff)
    h = auroc_cosine(feats, cross, hard) if len(hard) >= 10 else None
    k = knn_purity(feats, styles)
    m = retrieval_map10(feats, styles)
    print(f"  Overall    : {o['auroc']}% [{o['ci95_low']}, {o['ci95_high']}]", flush=True)
    print(f"  CrossArtist: {c['auroc']}% [{c['ci95_low']}, {c['ci95_high']}]", flush=True)
    if h:
        print(f"  SameGenreH : {h['auroc']}% [{h['ci95_low']}, {h['ci95_high']}]", flush=True)
    print(f"  kNN={k} | mAP@10={m}%", flush=True)
    return {"dim": int(feats.shape[1]), "overall": o, "cross_artist": c,
            "same_genre_hard": h, "knn": k, "map10": m}


def main():
    device = "cuda"
    ds = RawMemmapWikiArtDataset(MANIFEST, is_training=False)

    # labels from manifest directly (no need to encode)
    styles, artists, genres = [], [], []
    for i in range(len(ds)):
        rec = ds.records_by_idx[ds.indices[i]]
        styles.append(rec["style_id"])
        artists.append(rec.get("artist_id", -1))
        genres.append(rec.get("genre_id", -1))
    styles = np.array(styles); artists = np.array(artists); genres = np.array(genres)
    same, cross, diff, hard = make_pairs(styles, artists, genres)
    print(f"[Pairs] same={len(same)} cross={len(cross)} diff={len(diff)} hard={len(hard)}", flush=True)

    results = {}

    print("\n[1/2] CSD style head...", flush=True)
    csd, _, _ = csd_model.load_csd(device)
    csd_f = extract_csd(csd, ds, device)
    del csd
    torch.cuda.empty_cache()
    results["CSD_style"] = eval_desc("CSD style head", csd_f, styles, same, cross, diff, hard)

    print("\n[2/2] ALADIN-ViT...", flush=True)
    aladin = build_aladin(device)
    al_f = extract_aladin(aladin, ds, device)
    del aladin
    torch.cuda.empty_cache()
    results["ALADIN_ViT"] = eval_desc("ALADIN-ViT (bamfg)", al_f, styles, same, cross, diff, hard)

    # cross-model geometry agreement
    S = csd_f @ csd_f.T
    A = al_f @ al_f.T
    iu = np.triu_indices(len(styles), k=1)
    ra = np.argsort(np.argsort(S[iu])).astype(np.float64)
    rb = np.argsort(np.argsort(A[iu])).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    sp = float((ra @ rb) / (np.linalg.norm(ra) * np.linalg.norm(rb) + 1e-12))
    results["geometry_agreement"] = {"spearman_csd_vs_aladin": round(sp, 4)}
    print(f"\n[Geometry] Spearman(CSD simmat, ALADIN simmat) = {sp:.4f}", flush=True)

    # verdict table
    print("\n" + "=" * 74)
    print(f"{'Teacher':<16}{'Overall':>10}{'CrossArt':>10}{'SameGenreH':>12}{'kNN@1':>8}{'mAP@10':>9}")
    print("=" * 74)
    for k, v in results.items():
        if k == "geometry_agreement":
            continue
        h = v["same_genre_hard"]["auroc"] if v["same_genre_hard"] else 0
        print(f"{k:<16}{v['overall']['auroc']:>10.2f}{v['cross_artist']['auroc']:>10.2f}"
              f"{h:>12.2f}{v['knn']['knn1']:>8.2f}{v['map10']:>9.2f}")
    print("=" * 74)

    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[Saved] {OUT}")


if __name__ == "__main__":
    main()
