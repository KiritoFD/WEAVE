"""Style-REPA: Positive/Negative Relation-Alignment Decomposition.

Diagnoses whether the 3 same-style pairs are swamped by the 12 cross-style pairs
inside a 6-image microbatch relation matrix.

For a fixed held-out batch construction (3 styles x 2 artists), computes:
  L_pos = mean over same-style off-diagonal pairs of (S_ij - T_ij)^2   [3 pairs]
  L_neg = mean over cross-style off-diagonal pairs of (S_ij - T_ij)^2  [12 pairs]
  L_all = mean over all off-diagonal pairs

Evaluated on A0 and every REPA checkpoint, on a FIXED held-out set of batches
(no training-time randomness), so curves are comparable.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, "/root/Workspace/WEAVE_VAE")
sys.path.insert(0, "/root/Workspace/WEAVE_VAE/tools/12h")

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader

from tools.vae_suite.dataset import RawMemmapWikiArtDataset
import csd_model

OUT = Path("/root/Workspace/WEAVE_VAE/runs_vae/v2_diagnosis/17_pos_neg_decomp.json")
MANIFEST = Path("/root/Workspace/data/wikiart_512_raw/wikiart_512_raw_manifest.json")
A0_DIR = "/root/Workspace/WEAVE_VAE/runs_vae/A0/final_diffusers_vae"
REPA_DIR = Path("/root/Workspace/WEAVE_VAE/runs_vae/REPA")

NORM_MEAN = [0.48145466, 0.4578275, 0.40821073]
NORM_STD = [0.26862954, 0.26130258, 0.27577711]

N_BATCHES = 60       # 60 batches x 6 images = 360 images, held-out
N_STYLES = 3
N_PER_STYLE = 2


def build_fixed_batches(dataset, seed=1234):
    """Deterministic 3-style x 2-artist batches from the validation set."""
    from collections import defaultdict
    sa = defaultdict(lambda: defaultdict(list))
    for pos, raw in enumerate(dataset.indices):
        rec = dataset.records_by_idx[raw]
        sa[rec["style_id"]][rec.get("artist_id", -1)].append(pos)
    usable = sorted([s for s, am in sa.items()
                     if len([a for a, v in am.items() if a >= 0 and v]) >= 2])

    rng = random.Random(seed)
    batches = []
    for _ in range(N_BATCHES):
        sts = rng.sample(usable, N_STYLES)
        b = []
        for s in sts:
            am = sa[s]
            arts = [a for a, v in am.items() if a >= 0 and v]
            rng.shuffle(arts)
            for a in arts[:N_PER_STYLE]:
                b.append(rng.choice(am[a]))
        if len(b) == N_STYLES * N_PER_STYLE:
            batches.append(b)
    return batches


def batch_styles(dataset, batch):
    return [dataset.records_by_idx[dataset.indices[p]]["style_id"] for p in batch]


def relation_losses(s_feat, t_feat, styles):
    """Split relation-alignment loss into same-style (pos) and cross-style (neg)."""
    S = s_feat @ s_feat.t()
    T = t_feat @ t_feat.t()
    B = S.shape[0]
    st = torch.tensor(styles, device=S.device)
    same = (st[:, None] == st[None, :])
    eye = torch.eye(B, dtype=torch.bool, device=S.device)
    pos_mask = same & ~eye
    neg_mask = (~same) & ~eye
    diff2 = (S - T) ** 2
    l_pos = diff2[pos_mask].mean() if pos_mask.any() else torch.tensor(0.0, device=S.device)
    l_neg = diff2[neg_mask].mean() if neg_mask.any() else torch.tensor(0.0, device=S.device)
    l_all = diff2[~eye].mean()
    return float(l_pos), float(l_neg), float(l_all), int(pos_mask.sum()), int(neg_mask.sum())


def eval_model(vae, whiten, batches, dataset, csd, device):
    nm = torch.tensor(NORM_MEAN, device=device).view(1, 3, 1, 1)
    ns = torch.tensor(NORM_STD, device=device).view(1, 3, 1, 1)
    pos_all, neg_all, all_all = [], [], []
    npos = nneg = 0
    with torch.no_grad():
        for b in batches:
            x = torch.stack([dataset[p]["image"] for p in b]).to(device)
            styles = batch_styles(dataset, b)
            post = vae.encode(x).latent_dist
            pooled = F.adaptive_avg_pool2d(post.mean.float(), (8, 8)).flatten(1)
            if whiten is not None:
                pooled = (pooled - whiten["mean"].to(device)) / whiten["std"].to(device)
            s = F.normalize(pooled, dim=-1)

            x01 = (x.clamp(-1, 1) + 1.0) / 2.0
            x224 = F.interpolate(x01, (224, 224), mode="bicubic", align_corners=False)
            xin = ((x224 - nm) / ns).half()
            _, _, sty = csd(xin)
            t = F.normalize(sty.float(), dim=-1)

            lp, ln, la, cp, cn = relation_losses(s, t, styles)
            pos_all.append(lp); neg_all.append(ln); all_all.append(la)
            npos, nneg = cp, cn

    return {
        "L_pos": round(float(np.mean(pos_all)), 6),
        "L_neg": round(float(np.mean(neg_all)), 6),
        "L_all": round(float(np.mean(all_all)), 6),
        "ratio_pos_over_neg": round(float(np.mean(pos_all) / max(np.mean(neg_all), 1e-9)), 4),
        "pairs_per_batch": {"pos": npos, "neg": nneg},
        "pos_share_of_pairs": round(npos / (npos + nneg), 4),
    }


def main():
    device = "cuda"
    ds = RawMemmapWikiArtDataset(MANIFEST, is_training=False)
    batches = build_fixed_batches(ds)
    print(f"[Batches] {len(batches)} fixed held-out batches of {N_STYLES*N_PER_STYLE} images", flush=True)

    csd, _, _ = csd_model.load_csd(device)
    results = {}

    print("\n-- A0 (baseline) --", flush=True)
    vae = AutoencoderKL.from_pretrained(A0_DIR).to(device).eval()
    results["A0"] = eval_model(vae, None, batches, ds, csd, device)
    print(f"   {results['A0']}", flush=True)
    del vae
    torch.cuda.empty_cache()

    ws = torch.load(REPA_DIR / "whiten_stats.pt", map_location="cpu", weights_only=False)
    for ck in sorted(REPA_DIR.glob("checkpoint_step_*.pt")):
        step = int(ck.stem.split("_")[-1])
        print(f"\n-- REPA step {step} --", flush=True)
        vae = AutoencoderKL.from_pretrained(A0_DIR).to(device).eval()
        ck_data = torch.load(ck, map_location="cpu", weights_only=False)
        vae.load_state_dict(ck_data.get("ema", ck_data["model"]))
        vae.eval()
        results[f"REPA_{step}"] = eval_model(vae, ws, batches, ds, csd, device)
        print(f"   {results[f'REPA_{step}']}", flush=True)
        del vae
        torch.cuda.empty_cache()

    print("\n" + "=" * 86)
    print("POS / NEG RELATION-ALIGNMENT DECOMPOSITION (fixed held-out batches)")
    print("=" * 86)
    print(f"{'Model':<14}{'L_pos':>10}{'L_neg':>10}{'L_all':>10}{'pos/neg':>10}{'pos%':>8}")
    print("-" * 86)
    for k, v in results.items():
        print(f"{k:<14}{v['L_pos']:>10.5f}{v['L_neg']:>10.5f}{v['L_all']:>10.5f}"
              f"{v['ratio_pos_over_neg']:>10.3f}{v['pos_share_of_pairs']*100:>8.1f}")
    print("=" * 86)

    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[Saved] {OUT}")


if __name__ == "__main__":
    main()
