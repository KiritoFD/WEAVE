"""WEAVE-VAE Style-REPA: P0 Teacher Qualification Test.

Verifies whether frozen CSD (ViT-L) is a qualified style teacher on our
136-class WikiArt validation set, using EXACTLY the same image pairs and
positive/negative definitions as the SWD diagnosis (P1).

Metrics per descriptor:
  - Overall Style AUROC (same vs diff style)
  - Cross-Artist AUROC (same style, different artist)
  - Same-Genre Hard AUROC (different style, SAME genre) <- the decisive one
  - kNN@1/5/10 style purity
  - Style retrieval mAP@10

Descriptors compared:
  - CSD style head feature (teacher candidate)
  - CSD content head feature (control)
  - SWD on B0 latent (current failed metric, reference)
  - Raw latent GAP (naive baseline)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/Workspace/WEAVE_VAE")
sys.path.insert(0, "/root/Workspace/WEAVE_VAE/tools/12h")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from tools.vae_suite.dataset import RawMemmapWikiArtDataset
from tools.vae_suite.swd_metric import LatentSWDDescriptor
import csd_model

OUT_DIR = Path("/root/Workspace/WEAVE_VAE/runs_vae/v2_diagnosis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST = Path("/root/Workspace/data/wikiart_512_raw/wikiart_512_raw_manifest.json")
CALIB = Path("/root/Workspace/WEAVE_VAE/swd_calibration.json")
B0_DIR = "/root/Workspace/xy/DiT/data/pretrained/sd-vae-ft-ema"

NORM_MEAN = [0.48145466, 0.4578275, 0.40821073]
NORM_STD = [0.26862954, 0.26130258, 0.27577711]


def make_pairs(styles, artists, genres, seed=42):
    rng = np.random.default_rng(seed)
    N = len(styles)
    same, cross, diff, hard = [], [], [], []
    for i in range(N):
        sc = np.where(styles == styles[i])[0]
        sc = sc[sc != i]
        if len(sc):
            p = int(rng.choice(sc))
            same.append((i, p))
            if artists[i] >= 0:
                ca = sc[artists[sc] != artists[i]]
                if len(ca):
                    cross.append((i, int(rng.choice(ca))))
        dc = np.where(styles != styles[i])[0]
        if len(dc):
            diff.append((i, int(rng.choice(dc))))
            if genres[i] >= 0:
                hc = dc[genres[dc] == genres[i]]
                if len(hc):
                    hard.append((i, int(rng.choice(hc))))
    return same, cross, diff, hard


def auroc_cosine(feats, pos_pairs, neg_pairs, n_boot=200, seed=42):
    """feats: (N, D) L2-normalized numpy array. Cosine similarity."""
    n = min(len(pos_pairs), len(neg_pairs))
    if n < 10:
        return None
    n = min(n, 800)
    sp, dp = np.array(pos_pairs[:n]), np.array(neg_pairs[:n])
    sim_pos = (feats[sp[:, 0]] * feats[sp[:, 1]]).sum(1)
    sim_neg = (feats[dp[:, 0]] * feats[dp[:, 1]]).sum(1)

    lbs = np.concatenate([np.ones(n), np.zeros(n)])
    sims = np.concatenate([sim_pos, sim_neg])
    auc = float(roc_auc_score(lbs, sims) * 100.0)

    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        pi = rng.integers(0, n, n)
        ni = rng.integers(0, n, n)
        bb = np.concatenate([np.ones(n), np.zeros(n)])
        ss = np.concatenate([sim_pos[pi], sim_neg[ni]])
        boot.append(roc_auc_score(bb, ss) * 100.0)
    lo, hi = np.percentile(boot, [2.5, 97.5])

    return {
        "auroc": round(auc, 2),
        "ci95_low": round(float(lo), 2),
        "ci95_high": round(float(hi), 2),
        "n_pairs": n,
        "mean_sim_pos": round(float(sim_pos.mean()), 4),
        "mean_sim_neg": round(float(sim_neg.mean()), 4),
        "rel_margin": round(float((sim_pos.mean() - sim_neg.mean()) /
                                  (abs(sim_pos.mean()) + abs(sim_neg.mean()) + 1e-8)), 4),
    }


def knn_purity(feats, styles, ks=(1, 5, 10)):
    sim = feats @ feats.T
    np.fill_diagonal(sim, -np.inf)
    order = np.argsort(-sim, axis=1)
    out = {}
    for k in ks:
        out[f"knn{k}"] = round(float((styles[order[:, :k]] == styles[:, None]).mean()) * 100.0, 2)
    return out


def retrieval_map10(feats, styles, seed=42, n_q=300):
    rng = np.random.default_rng(seed)
    N = len(styles)
    idx = rng.permutation(N)
    q_ids, g_ids = idx[:min(n_q, N // 4)], idx[min(n_q, N // 4):]
    aps = []
    g_feats = feats[g_ids]
    g_styles = styles[g_ids]
    for qi in q_ids:
        sims = g_feats @ feats[qi]
        order = np.argsort(-sims)[:10]
        rel = (g_styles[order] == styles[qi]).astype(float)
        if rel.sum() == 0:
            aps.append(0.0)
            continue
        prec = np.cumsum(rel) / (np.arange(len(rel)) + 1)
        aps.append(float((prec * rel).sum() / rel.sum()))
    return round(float(np.mean(aps) * 100.0), 2)


def extract_csd_features(csd, dataset, device, batch_size=16):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    nm = torch.tensor(NORM_MEAN, device=device).view(1, 3, 1, 1)
    ns = torch.tensor(NORM_STD, device=device).view(1, 3, 1, 1)

    style_feats, content_feats = [], []
    with torch.no_grad():
        for i, b in enumerate(loader):
            x = b["image"].to(device)
            x01 = (x.clamp(-1, 1) + 1.0) / 2.0
            x224 = F.interpolate(x01, (224, 224), mode="bicubic", align_corners=False)
            xin = ((x224 - nm) / ns).half()
            _, cont, sty = csd(xin)
            style_feats.append(sty.float().cpu())
            content_feats.append(cont.float().cpu())
            if (i + 1) % 30 == 0:
                print(f"    CSD [{i+1}/{len(loader)}]", flush=True)
    return torch.cat(style_feats).numpy(), torch.cat(content_feats).numpy()


def extract_b0_latents(dataset, device, batch_size=16):
    vae = AutoencoderKL.from_pretrained(B0_DIR).to(device).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    zs, ys, ars, ges = [], [], [], []
    with torch.no_grad():
        for i, b in enumerate(loader):
            post = vae.encode(b["image"].to(device)).latent_dist
            zs.append(post.mean.float().cpu())
            ys.append(b["style_id"])
            ars.append(b["artist_id"])
            ges.append(b["genre_id"])
    del vae
    torch.cuda.empty_cache()
    return (torch.cat(zs), torch.cat(ys).numpy(),
            torch.cat(ars).numpy(), torch.cat(ges).numpy())


def evaluate_descriptor(name, feats, styles, same, cross, diff, hard, use_cosine=True):
    print(f"\n--- {name} (dim={feats.shape[1]}) ---", flush=True)
    r_overall = auroc_cosine(feats, same, diff)
    r_cross = auroc_cosine(feats, cross, diff)
    r_hard = auroc_cosine(feats, cross, hard) if len(hard) >= 10 else None
    knn = knn_purity(feats, styles)
    map10 = retrieval_map10(feats, styles)

    print(f"  Overall   : {r_overall['auroc']}% [{r_overall['ci95_low']}, {r_overall['ci95_high']}]", flush=True)
    print(f"  CrossArt  : {r_cross['auroc']}% [{r_cross['ci95_low']}, {r_cross['ci95_high']}] relM={r_cross['rel_margin']}", flush=True)
    if r_hard:
        print(f"  SameGenreH: {r_hard['auroc']}% [{r_hard['ci95_low']}, {r_hard['ci95_high']}]", flush=True)
    print(f"  kNN={knn} | mAP@10={map10}%", flush=True)

    return {
        "dim": int(feats.shape[1]),
        "overall": r_overall,
        "cross_artist": r_cross,
        "same_genre_hard": r_hard,
        "knn": knn,
        "map10": map10,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=== Style-REPA P0: Teacher Qualification Test ===", flush=True)

    dataset = RawMemmapWikiArtDataset(MANIFEST, is_training=False)
    print(f"[Data] Val set: {len(dataset)} images", flush=True)

    # ---- B0 latents (also gives us labels) ----
    print("\n[1/3] Extracting B0 latents...", flush=True)
    z, styles, artists, genres = extract_b0_latents(dataset, device)
    same, cross, diff, hard = make_pairs(styles, artists, genres)
    print(f"  pairs: same={len(same)} cross={len(cross)} diff={len(diff)} hard={len(hard)}", flush=True)

    # ---- CSD features ----
    print("\n[2/3] Extracting frozen CSD features...", flush=True)
    csd, _, _ = csd_model.load_csd(device)
    csd_style, csd_content = extract_csd_features(csd, dataset, device)

    # L2 normalize
    csd_style_n = csd_style / (np.linalg.norm(csd_style, axis=1, keepdims=True) + 1e-8)
    csd_content_n = csd_content / (np.linalg.norm(csd_content, axis=1, keepdims=True) + 1e-8)

    results = {}
    results["CSD_style_head"] = evaluate_descriptor(
        "CSD style head (teacher candidate)", csd_style_n, styles, same, cross, diff, hard)
    results["CSD_content_head"] = evaluate_descriptor(
        "CSD content head (control)", csd_content_n, styles, same, cross, diff, hard)

    # ---- SWD on B0 latent (reference, current failed metric) ----
    print("\n[3/3] Computing SWD reference on B0 latent...", flush=True)
    calib = json.loads(CALIB.read_text())
    desc = LatentSWDDescriptor()
    desc.set_calibration(
        torch.tensor(calib["chan_mean"]), torch.tensor(calib["chan_std"]),
        calib["scale_swd1"], calib["scale_swd2"], calib["scale_moment"],
        calib["median_inter_dist"],
    )

    def swd_auroc(pos_pairs, neg_pairs, chunk=192):
        n = min(len(pos_pairs), len(neg_pairs))
        if n < 10:
            return None
        n = min(n, 600)
        q1, q2, mom = [], [], []
        with torch.no_grad():
            for i in range(0, z.shape[0], chunk):
                a, b, c = desc(z[i:i + chunk])
                q1.append(a); q2.append(b); mom.append(c)
        q1, q2, mom = torch.cat(q1), torch.cat(q2), torch.cat(mom)

        def dists(pairs):
            sp = np.array(pairs[:n])
            out = []
            with torch.no_grad():
                for c0 in range(0, len(sp), 48):
                    ca, cb = sp[c0:c0 + 48, 0], sp[c0:c0 + 48, 1]
                    d = desc.compute_pairwise_distance(
                        q1[ca], q2[ca], mom[ca], q1[cb], q2[cb], mom[cb])
                    out.append(torch.diag(d))
            return torch.cat(out).cpu().numpy()

        dpos, dneg = dists(pos_pairs), dists(neg_pairs)
        lbs = np.concatenate([np.ones(n), np.zeros(n)])
        dd = np.concatenate([dpos, dneg])
        auc = float(roc_auc_score(lbs, -dd) * 100.0)
        rng = np.random.default_rng(42)
        boot = []
        for _ in range(200):
            pi = rng.integers(0, n, n); ni = rng.integers(0, n, n)
            bb = np.concatenate([np.ones(n), np.zeros(n)])
            boot.append(roc_auc_score(bb, -np.concatenate([dpos[pi], dneg[ni]])) * 100.0)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        return {"auroc": round(auc, 2), "ci95_low": round(float(lo), 2),
                "ci95_high": round(float(hi), 2), "n_pairs": n}

    swd_res = {
        "dim": "4x64x64 (SWD descriptor)",
        "overall": swd_auroc(same, diff),
        "cross_artist": swd_auroc(cross, diff),
        "same_genre_hard": swd_auroc(cross, hard) if len(hard) >= 10 else None,
    }
    print(f"  SWD Overall   : {swd_res['overall']['auroc']}%", flush=True)
    print(f"  SWD CrossArt  : {swd_res['cross_artist']['auroc']}%", flush=True)
    if swd_res["same_genre_hard"]:
        print(f"  SWD SameGenreH: {swd_res['same_genre_hard']['auroc']}%", flush=True)
    results["SWD_B0_latent_reference"] = swd_res

    # ---- Save ----
    out = OUT_DIR / "13_teacher_qualification.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # ---- Verdict ----
    csd_cross = results["CSD_style_head"]["cross_artist"]["auroc"]
    csd_hard = results["CSD_style_head"]["same_genre_hard"]["auroc"]
    swd_cross = swd_res["cross_artist"]["auroc"]
    swd_hard = swd_res["same_genre_hard"]["auroc"] if swd_res["same_genre_hard"] else 0

    print("\n" + "=" * 70)
    print("TEACHER QUALIFICATION VERDICT")
    print("=" * 70)
    print(f"{'Metric':<22}{'SWD (B0)':>12}{'CSD style':>12}{'Delta':>10}")
    print(f"{'Cross-Artist AUROC':<22}{swd_cross:>12.2f}{csd_cross:>12.2f}{csd_cross-swd_cross:>+10.2f}")
    print(f"{'Same-Genre Hard':<22}{swd_hard:>12.2f}{csd_hard:>12.2f}{csd_hard-swd_hard:>+10.2f}")

    passed = (csd_cross >= swd_cross + 5.0) and (csd_hard >= 60.0)
    print(f"\nPASS criteria: CrossArtist >= SWD+5.0 AND SameGenreHard >= 60.0")
    print(f"VERDICT: {'PASS - CSD is a qualified teacher' if passed else 'FAIL - CSD insufficient as teacher'}")
    results["verdict"] = {"passed": bool(passed),
                          "csd_cross": csd_cross, "csd_hard": csd_hard,
                          "swd_cross": swd_cross, "swd_hard": swd_hard}
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[Saved] {out}")


if __name__ == "__main__":
    main()
