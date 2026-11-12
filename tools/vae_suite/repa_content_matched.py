"""Content-Matched Cross-Artist AUROC: stricter protocol to rule out content shortcut.

For each anchor i:
  positives = same Style, different Artist
  negatives = different Style, SAME Genre
  Content matching: pick the (pos, neg) pair whose DINOv2 content similarity
  to the anchor is closest to each other (minimal content gap).

Content proxy = frozen DINOv2 (independent of CSD, avoids evaluation self-selection).

Reports per descriptor: CSD style, ALADIN, A0 latent (8x8 pooled), REPA checkpoints.
Also reports the *content gap* distribution to prove matching worked.
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
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from transformers import AutoModel

from tools.vae_suite.dataset import RawMemmapWikiArtDataset
import csd_model

OUT = Path("/root/Workspace/WEAVE_VAE/runs_vae/v2_diagnosis/16_content_matched.json")
MANIFEST = Path("/root/Workspace/data/wikiart_512_raw/wikiart_512_raw_manifest.json")
A0_DIR = "/root/Workspace/WEAVE_VAE/runs_vae/A0/final_diffusers_vae"
REPA_DIR = Path("/root/Workspace/WEAVE_VAE/runs_vae/REPA")
ALADIN_CKPT = "/tmp/aladin_repo/aladinVIT_bamfg_64.47.pt"
DINO_ID = "facebook/dinov2-base"

NORM_MEAN = [0.48145466, 0.4578275, 0.40821073]
NORM_STD = [0.26862954, 0.26130258, 0.27577711]
DINO_MEAN = [0.485, 0.456, 0.406]
DINO_STD = [0.229, 0.224, 0.225]


def extract_dino(dataset, device, batch_size=8):
    print(f"[DINOv2] loading {DINO_ID}...", flush=True)
    dino = AutoModel.from_pretrained(DINO_ID).to(device).eval()
    nm = torch.tensor(DINO_MEAN, device=device).view(1, 3, 1, 1)
    ns = torch.tensor(DINO_STD, device=device).view(1, 3, 1, 1)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    feats = []
    with torch.no_grad():
        for i, b in enumerate(loader):
            x = b["image"].to(device)
            x01 = (x.clamp(-1, 1) + 1.0) / 2.0
            x224 = F.interpolate(x01, (224, 224), mode="bicubic", align_corners=False)
            xin = (x224 - nm) / ns
            out = dino(pixel_values=xin)
            cls = out.last_hidden_state[:, 0]
            feats.append(F.normalize(cls.float(), dim=-1).cpu())
            if (i + 1) % 40 == 0:
                print(f"  DINO [{i+1}/{len(loader)}]", flush=True)
    del dino
    torch.cuda.empty_cache()
    return torch.cat(feats).numpy()


def extract_csd(dataset, device, batch_size=16):
    csd, _, _ = csd_model.load_csd(device)
    nm = torch.tensor(NORM_MEAN, device=device).view(1, 3, 1, 1)
    ns = torch.tensor(NORM_STD, device=device).view(1, 3, 1, 1)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    feats = []
    with torch.no_grad():
        for i, b in enumerate(loader):
            x = b["image"].to(device)
            x01 = (x.clamp(-1, 1) + 1.0) / 2.0
            x224 = F.interpolate(x01, (224, 224), mode="bicubic", align_corners=False)
            xin = ((x224 - nm) / ns).half()
            _, _, sty = csd(xin)
            feats.append(F.normalize(sty.float(), dim=-1).cpu())
            if (i + 1) % 40 == 0:
                print(f"  CSD [{i+1}/{len(loader)}]", flush=True)
    del csd
    torch.cuda.empty_cache()
    return torch.cat(feats).numpy()


def extract_aladin(dataset, device, batch_size=16):
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
    m = AViT(); m.load_state_dict(ck); m.eval().to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    feats = []
    with torch.no_grad():
        for i, b in enumerate(loader):
            x = b["image"].to(device)
            x224 = F.interpolate(x.clamp(-1, 1), (224, 224), mode="bicubic", align_corners=False)
            feats.append(F.normalize(m.encode(x224).float(), dim=-1).cpu())
            if (i + 1) % 40 == 0:
                print(f"  ALADIN [{i+1}/{len(loader)}]", flush=True)
    del m
    torch.cuda.empty_cache()
    return torch.cat(feats).numpy()


def extract_vae_pooled(vae_path, ckpt, dataset, device, batch_size=16, whiten=None):
    vae = AutoencoderKL.from_pretrained(vae_path).to(device).eval()
    if ckpt is not None:
        ck_data = torch.load(ckpt, map_location="cpu", weights_only=False)
        vae.load_state_dict(ck_data.get("ema", ck_data["model"]))
        vae.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    feats = []
    with torch.no_grad():
        for i, b in enumerate(loader):
            post = vae.encode(b["image"].to(device)).latent_dist
            pooled = F.adaptive_avg_pool2d(post.mean.float(), (8, 8)).flatten(1)
            if whiten is not None:
                pooled = (pooled - whiten["mean"].to(device)) / whiten["std"].to(device)
            feats.append(F.normalize(pooled, dim=-1).cpu())
            if (i + 1) % 40 == 0:
                print(f"  VAE [{i+1}/{len(loader)}]", flush=True)
    del vae
    torch.cuda.empty_cache()
    return torch.cat(feats).numpy()


def build_content_matched_pairs(dino, styles, artists, genres, max_pairs=1200, seed=42):
    """For each anchor pick (pos, neg) with minimal DINOv2 content-similarity gap."""
    rng = np.random.default_rng(seed)
    N = len(styles)
    pos_list, neg_list, gaps = [], [], []

    order = rng.permutation(N)
    for i in order:
        if len(pos_list) >= max_pairs:
            break
        pc = np.where((styles == styles[i]))[0]
        pc = pc[pc != i]
        if artists[i] >= 0:
            pc = pc[artists[pc] != artists[i]]
        if len(pc) == 0:
            continue
        nc = np.where((styles != styles[i]) & (genres == genres[i]))[0]
        if len(nc) < 2:
            continue
        if len(pc) > 12:
            pc = rng.choice(pc, 12, replace=False)
        if len(nc) > 24:
            nc = rng.choice(nc, 24, replace=False)

        sim_p = dino[pc] @ dino[i]        # (P,)
        sim_n = dino[nc] @ dino[i]        # (K,)
        # find (p, n) pair with closest content similarity to the anchor
        gap = np.abs(sim_p[:, None] - sim_n[None, :])
        pi, ni = np.unravel_index(np.argmin(gap), gap.shape)
        pos_list.append((int(i), int(pc[pi])))
        neg_list.append((int(i), int(nc[ni])))
        gaps.append(float(gap[pi, ni]))

    return pos_list, neg_list, np.array(gaps)


def auroc_pairs(feats, pos, neg, n_boot=200, seed=42):
    n = min(len(pos), len(neg))
    if n < 10:
        return None
    sp = np.array(pos[:n]); sn = np.array(neg[:n])
    sp_sim = (feats[sp[:, 0]] * feats[sp[:, 1]]).sum(1)
    sn_sim = (feats[sn[:, 0]] * feats[sn[:, 1]]).sum(1)
    lbs = np.concatenate([np.ones(n), np.zeros(n)])
    sims = np.concatenate([sp_sim, sn_sim])
    auc = float(roc_auc_score(lbs, sims) * 100.0)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        pi = rng.integers(0, n, n); ni = rng.integers(0, n, n)
        bb = np.concatenate([np.ones(n), np.zeros(n)])
        boot.append(roc_auc_score(bb, np.concatenate([sp_sim[pi], sn_sim[ni]])) * 100.0)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"auroc": round(auc, 2), "ci95_low": round(float(lo), 2),
            "ci95_high": round(float(hi), 2), "n_pairs": int(n)}


def main():
    device = "cuda"
    ds = RawMemmapWikiArtDataset(MANIFEST, is_training=False)

    styles, artists, genres = [], [], []
    for i in range(len(ds)):
        rec = ds.records_by_idx[ds.indices[i]]
        styles.append(rec["style_id"])
        artists.append(rec.get("artist_id", -1))
        genres.append(rec.get("genre_id", -1))
    styles = np.array(styles); artists = np.array(artists); genres = np.array(genres)

    print("\n[1/5] DINOv2 content features...", flush=True)
    dino = extract_dino(ds, device)

    pos, neg, gaps = build_content_matched_pairs(dino, styles, artists, genres)
    print(f"\n[Pairs] content-matched pairs: {len(pos)}", flush=True)
    print(f"  content-sim gap: mean={gaps.mean():.4f}, median={np.median(gaps):.4f}, "
          f"p95={np.percentile(gaps,95):.4f}", flush=True)

    results = {"pair_stats": {
        "n_pairs": len(pos),
        "content_gap_mean": round(float(gaps.mean()), 4),
        "content_gap_median": round(float(np.median(gaps)), 4),
        "content_gap_p95": round(float(np.percentile(gaps, 95)), 4),
    }}

    print("\n[2/5] CSD style...", flush=True)
    csd_f = extract_csd(ds, device)
    results["CSD_style"] = auroc_pairs(csd_f, pos, neg)
    print(f"  CSD: {results['CSD_style']}", flush=True)

    print("\n[3/5] ALADIN...", flush=True)
    al_f = extract_aladin(ds, device)
    results["ALADIN"] = auroc_pairs(al_f, pos, neg)
    print(f"  ALADIN: {results['ALADIN']}", flush=True)

    print("\n[4/5] A0 latent (8x8 pooled)...", flush=True)
    a0_f = extract_vae_pooled(A0_DIR, None, ds, device)
    results["A0_latent"] = auroc_pairs(a0_f, pos, neg)
    print(f"  A0: {results['A0_latent']}", flush=True)

    print("\n[5/5] REPA checkpoints...", flush=True)
    ws = torch.load(REPA_DIR / "whiten_stats.pt", map_location="cpu", weights_only=False)
    for ck in sorted(REPA_DIR.glob("checkpoint_step_*.pt")):
        step = int(ck.stem.split("_")[-1])
        print(f"  -- REPA step {step} --", flush=True)
        f = extract_vae_pooled(A0_DIR, ck, ds, device, whiten=ws)
        results[f"REPA_{step}"] = auroc_pairs(f, pos, neg)
        print(f"     {results[f'REPA_{step}']}", flush=True)

    # DINOv2 itself as content control (should be near chance under content matching)
    results["DINOv2_content_control"] = auroc_pairs(dino, pos, neg)
    print(f"\n[Control] DINOv2 content: {results['DINOv2_content_control']}", flush=True)

    print("\n" + "=" * 78)
    print("CONTENT-MATCHED CROSS-ARTIST AUROC")
    print("=" * 78)
    print(f"{'Descriptor':<26}{'AUROC':>9}{'CI95':>20}{'n':>7}")
    print("-" * 78)
    for k, v in results.items():
        if k == "pair_stats" or v is None:
            continue
        ci = "[{}, {}]".format(v["ci95_low"], v["ci95_high"])
        print(f"{k:<26}{v['auroc']:>9.2f}{ci:>20}{v['n_pairs']:>7}")
    print("=" * 78)

    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[Saved] {OUT}")


if __name__ == "__main__":
    main()
