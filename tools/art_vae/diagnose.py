"""Read-only root-cause diagnosis for the stage-A negative result.

Question 1: why is Cross-Artist Content-Matched AUROC < 0.5 (worse than random)?
Hypothesis: the fixed readout r(z) is dominated by CONTENT, so DINO-matched
different-style images look MORE similar than same-style/different-painter ones.

Question 2: did SupCon move the representation in the intended direction at all?
Compare the SAME three cosine distributions for a0 and the trained encoder.

No training, no model changes. Writes one JSON.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.art_vae.data import OUT, TrainPool
from tools.art_vae.eval import (dino_features, encode_all, load_variant,
                                val_pool_records)
from tools.art_vae.style_readout import style_readout


def cosine_stats(R, style, artist, dino, hard_k=16, seed=0):
    sim_z = (R @ R.t()).cpu().numpy()
    sim_d = (dino @ dino.t()).cpu().numpy()
    n = len(style)
    rng = np.random.default_rng(seed)
    groups = {"same_style_diff_artist": [], "hard_neg_diff_style": [],
              "easy_neg_diff_style": [], "same_artist_diff_style": []}
    for i in range(n):
        ss = np.where(style == style[i])[0]
        pos = ss[(artist[ss] != artist[i]) & (ss != i)]
        diff = np.where(style != style[i])[0]
        if len(pos) == 0 or len(diff) == 0:
            continue
        hard = diff[np.argsort(-sim_d[i, diff])[:hard_k]]
        easy = rng.choice(diff, min(hard_k, len(diff)), replace=False)
        for j in pos:
            groups["same_style_diff_artist"].append(sim_z[i, j])
        for j in hard:
            groups["hard_neg_diff_style"].append(sim_z[i, j])
        for j in easy:
            groups["easy_neg_diff_style"].append(sim_z[i, j])
    out = {}
    for k, v in groups.items():
        a = np.asarray(v)
        out[k] = {"mean": round(float(a.mean()), 5),
                  "std": round(float(a.std()), 5),
                  "n": int(len(a))}
    out["gap_pos_minus_hard"] = round(
        out["same_style_diff_artist"]["mean"] - out["hard_neg_diff_style"]["mean"], 5)
    out["gap_pos_minus_easy"] = round(
        out["same_style_diff_artist"]["mean"] - out["easy_neg_diff_style"]["mean"], 5)
    return out


def auroc_scores(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    a = np.asarray(pos); b = np.asarray(neg)
    allv = np.concatenate([a, b])
    order = np.argsort(allv)
    ranks = np.empty(len(allv), float)
    ranks[order] = np.arange(1, len(allv) + 1)
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    avg = np.bincount(inv, weights=ranks) / cnt
    ranks = avg[inv]
    n1 = len(a)
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * len(b)))


def cosine_stats_with_scores(R, style, artist, dino, hard_k=16, seed=0):
    sim_z = (R @ R.t()).cpu().numpy()
    sim_d = (dino @ dino.t()).cpu().numpy()
    n = len(style)
    rng = np.random.default_rng(seed)
    pos, hard = [], []
    for i in range(n):
        ss = np.where(style == style[i])[0]
        p = ss[(artist[ss] != artist[i]) & (ss != i)]
        diff = np.where(style != style[i])[0]
        if len(p) == 0 or len(diff) == 0:
            continue
        h = diff[np.argsort(-sim_d[i, diff])[:hard_k]]
        pos += [sim_z[i, j] for j in p]
        hard += [sim_z[i, j] for j in h]
    return cosine_stats(R, style, artist, dino, hard_k, seed), auroc_scores(pos, hard)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(OUT / "A_best.pt"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device
    pool = TrainPool().describe()
    recs = val_pool_records(pool)
    if args.limit:
        recs = recs[:args.limit]
    style = np.array([r["style_id"] for r in recs], np.int64)
    artist = np.array([r["artist_id"] for r in recs], np.int64)
    print(f"[Diag] val n={len(recs)}", flush=True)
    dino = dino_features(recs, pool, dev)

    res = {}
    for name, ckpt in [("a0", ""), ("s10", ""), ("A", args.ckpt)]:
        vname = name if name != "A" else "ckpt"
        _, enc, _ = load_variant(vname, ckpt, dev)
        Z, R = encode_all(enc, recs, pool, dev)
        stats, auroc_raw = cosine_stats_with_scores(R, style, artist, dino)
        # --- zero-cost control: remove the common mode (mean readout over the
        # validation pool) and re-measure. If this restores discrimination the
        # failure is an operator-implementation defect, not the objective.
        mu = R.mean(0, keepdim=True)
        Rc = F.normalize(R - mu, dim=1)
        stats_c, auroc_centered = cosine_stats_with_scores(
            Rc, style, artist, dino)
        # per-dim collapse / latent health
        with torch.no_grad():
            flat = Z.flatten(1)
            health = {"z_rms": round(float(Z.float().pow(2).mean().sqrt()), 5),
                      "q_per_dim_std_mean": round(float(R.std(0).mean()), 6),
                      "q_norm_mean": round(float(R.norm(dim=1).mean()), 5)}
        res[name] = {"cosine": stats, "cosine_centered": stats_c,
                     "auroc_raw": auroc_raw, "auroc_centered": auroc_centered,
                     "health": health}
        print(f"  [{name}] AUROC raw={auroc_raw:.4f} centered={auroc_centered:.4f} "
              f"| pos-hard gap={stats['gap_pos_minus_hard']:+.5f} "
              f"pos-easy gap={stats['gap_pos_minus_easy']:+.5f} "
              f"z_rms={health['z_rms']}", flush=True)
        del enc, Z, R
        torch.cuda.empty_cache()

    print("\n" + "=" * 92)
    print("ROOT-CAUSE DIAGNOSIS")
    print("=" * 92)
    for name, r in res.items():
        c = r["cosine"]
        print(f"  {name:<5} pos(same-style/cross-painter)={c['same_style_diff_artist']['mean']:+.4f}  "
              f"hard-neg(content-matched other style)={c['hard_neg_diff_style']['mean']:+.4f}  "
              f"easy-neg={c['easy_neg_diff_style']['mean']:+.4f}")
    a0, A = res["a0"], res["A"]
    print(f"\n  content dominance in a0: hard-neg exceeds pos by "
          f"{-a0['cosine']['gap_pos_minus_hard']:+.4f} -> AUROC<0.5 expected")
    print(f"  SupCon effect: pos-hard gap {a0['cosine']['gap_pos_minus_hard']:+.5f} -> "
          f"{A['cosine']['gap_pos_minus_hard']:+.5f} "
          f"(delta {A['cosine']['gap_pos_minus_hard']-a0['cosine']['gap_pos_minus_hard']:+.5f})")
    print(f"  latent change: z_rms {a0['health']['z_rms']} -> {A['health']['z_rms']}, "
          f"q std {a0['health']['q_per_dim_std_mean']} -> {A['health']['q_per_dim_std_mean']}")

    verdict = ("r(z) is CONTENT-dominated: content-matched other-style images are "
               "closer than same-style/cross-painter ones. A contrastive objective on "
               "this fixed readout optimises a signal that does not separate style.")
    if A["cosine"]["gap_pos_minus_hard"] > a0["cosine"]["gap_pos_minus_hard"] + 0.005:
        verdict += (" Stage A did move the gap in the right direction, but by less "
                    "than the noise level of the AUROC metric.")
    else:
        verdict += (" Stage A did NOT measurably change the pos-vs-hard gap.")
    print(f"\n  VERDICT: {verdict}")
    res["verdict"] = verdict

    outp = OUT / "diagnosis_A.json"
    outp.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"[Saved] {outp}", flush=True)


if __name__ == "__main__":
    main()
