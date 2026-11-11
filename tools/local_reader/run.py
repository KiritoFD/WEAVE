"""Train + evaluate the Local Style Reader (and the CNN control) on FROZEN latents.

    python -m tools.local_reader.run --steps 1500

Training : multi-positive SupCon, logical batch 64, cached latent tensors only.
Eval    : frozen 2048 holdout, content-matched hard negatives.

Reported:
  1. Cross-Artist Content-Matched AUROC  [target >= 0.55]
     - full 2048 holdout AND the strict 768-image painter-disjoint subset
  2. 5-shot retrieval                    [target > 0.0708 = flat-latent baseline]
  3. Unseen-school retrieval             [target >= 0.129]
  4. Paired cluster bootstrap of the PER-SOURCE discrimination gap
     (pos_minus_hard), not the raw cosine score. Raw scores are dominated by a
     common mode: both positives and hard negatives rise together, so a paired
     comparison of raw cosines reports a large but meaningless gain.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.art_vae.losses import (TAU, gamma_for_dim, supcon,
                                  supcon_relation_weights, vicreg_var)
from tools.art_vae.style_readout import style_readout
from tools.local_reader.data import FrozenLatents, SupConBatchSampler, batch_meta
from tools.local_reader.readers import D_OUT, build, n_params

OUT = ROOT / "runs" / "local_reader"
OUT.mkdir(parents=True, exist_ok=True)
DINO_MEAN = [0.485, 0.456, 0.406]
DINO_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------- statistics
def cluster_boot_idx(clusters, n_boot, seed):
    """Resample WHOLE clusters WITH replacement, preserving multiplicity.

    tools/v6_24h/stats.py uses np.isin(cid, pick), which collapses duplicate
    draws and understates variance. Here each drawn cluster's rows are
    concatenated, so a cluster drawn twice contributes twice.
    """
    uniq, inv = np.unique(clusters, return_inverse=True)
    by = [np.where(inv == i)[0] for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        yield np.concatenate([by[c] for c in pick])


def paired_cluster_ci(a, b, clusters, n_boot=2000, seed=0):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    ms = np.array([d[idx].mean() for idx in cluster_boot_idx(clusters, n_boot, seed)])
    lo, hi = float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5))
    return {"mean": float(d.mean()), "ci": [lo, hi],
            "excludes_zero": bool(lo > 0 or hi < 0), "n": int(len(d))}


def auroc_scores(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    a, b = np.asarray(pos, float), np.asarray(neg, float)
    allv = np.concatenate([a, b])
    order = np.argsort(allv)
    ranks = np.empty(len(allv), float)
    ranks[order] = np.arange(1, len(allv) + 1)
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    ranks = (np.bincount(inv, weights=ranks) / cnt)[inv]
    n1 = len(a)
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * len(b)))


# ---------------------------------------------------------------- eval helpers
@torch.no_grad()
def encode_all(reader, data, dev, hold=False, bs=64):
    n = len(data.hold_artist) if hold else data.pool.n
    out = []
    for i in range(0, n, bs):
        z = data.latent(np.arange(i, min(i + bs, n)), hold=hold).to(dev)
        out.append(reader(z).float().cpu())
        del z
        torch.cuda.empty_cache()
    return torch.cat(out)


@torch.no_grad()
def holdout_dino(data, dev, bs=32):
    from PIL import Image
    from transformers import AutoModel
    m = AutoModel.from_pretrained("facebook/dinov2-base",
                                  local_files_only=True).to(dev).eval()
    nm = torch.tensor(DINO_MEAN, device=dev).view(1, 3, 1, 1)
    ns = torch.tensor(DINO_STD, device=dev).view(1, 3, 1, 1)
    pos_of_raw = {int(r): i for i, r in enumerate(data.raw_hold.tolist())}
    ordered = sorted(data.hold_records, key=lambda r: pos_of_raw[r["raw_index"]])
    out = []
    for i in range(0, len(ordered), bs):
        ims = []
        for r in ordered[i:i + bs]:
            arr = np.ascontiguousarray(data.pool.mmap[r["raw_index"]])
            ims.append(torch.from_numpy(np.asarray(Image.fromarray(arr), np.uint8))
                       .permute(2, 0, 1).float())
        x = torch.stack(ims).to(dev).div_(127.5).sub_(1.0)
        x01 = (x.clamp(-1, 1) + 1) / 2
        x224 = F.interpolate(x01, (224, 224), mode="bicubic", align_corners=False)
        f = F.normalize(m(pixel_values=(x224 - nm) / ns)
                        .last_hidden_state[:, 0].float(), dim=-1)
        out.append(f.cpu())
        del x, x01, x224, f
        torch.cuda.empty_cache()
    return torch.cat(out)


def cross_artist_auroc(R, style, artist, dino, hard_k=16, subset=None):
    """Content-matched hard negatives, same-style/different-painter positives.

    Returns (auroc, pos, neg, pos_owner, gaps, gap_owner). `gaps[i]` is the
    per-source (mean positive sim - mean hard-negative sim); gap_owner is the
    source index. Paired bootstrap uses GAPS, clustered by source image.
    """
    sim = (R @ R.t()).cpu().numpy()
    sd = (dino @ dino.t()).cpu().numpy()
    idxs = np.arange(len(style)) if subset is None else np.asarray(subset)
    pos, neg, owner, gaps, gap_owner = [], [], [], [], []
    for i in idxs:
        same = np.where((style == style[i]) & (artist != artist[i]))[0]
        diff = np.where(style != style[i])[0]
        if len(same) == 0 or len(diff) == 0:
            continue
        hard = diff[np.argsort(-sd[i, diff])[:hard_k]]
        sp = np.array([sim[i, j] for j in same])
        sn = np.array([sim[i, j] for j in hard])
        pos.append(sp); neg.append(sn)
        owner.append(np.full(len(sp), i, np.int64))
        gaps.append(float(sp.mean() - sn.mean())); gap_owner.append(i)
    P = np.concatenate(pos) if pos else np.array([])
    N = np.concatenate(neg) if neg else np.array([])
    O = np.concatenate(owner) if owner else np.array([], np.int64)
    return (auroc_scores(P, N), P, N, O,
            np.asarray(gaps, float), np.asarray(gap_owner, np.int64))


def retrieval_acc(R, style, train_idx, test_idx):
    by = {}
    for i in train_idx:
        by.setdefault(int(style[i]), []).append(i)
    if len(by) < 2:
        return float("nan")
    keys = list(by)
    P = torch.stack([F.normalize(R[by[s]].mean(0), dim=0) for s in keys])
    correct = tot = 0
    for i in test_idx:
        if int(style[i]) not in by:
            continue
        correct += int(keys[int((P @ R[i]).argmax())] == style[i]); tot += 1
    return correct / max(tot, 1)


def few_shot(R, flat, style, n_shots=(5, 10), seed=0):
    """Fixed TEST pool across shot counts so 5/10-shot are comparable.

    Earlier code used v[k:k+20] as the test set, so changing k also changed the
    test pool (and small styles dropped out entirely), which made 10-shot look
    worse than 5-shot. Here the test pool is always v[20:40].
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(style)
    perm = {int(s): rng.permutation(np.where(style == s)[0]) for s in uniq}
    out = {}
    for k in n_shots:
        tr, te = [], []
        for s in uniq:
            v = perm[int(s)]
            tr += v[:k].tolist()
            te += v[20:40].tolist()
        out[str(k)] = {"reader": retrieval_acc(R, style, tr, te),
                       "flat": retrieval_acc(flat, style, tr, te),
                       "n_test": len(te)}
    return out


def linear_probe_curve(R, style, steps=150, seed=0):
    rng = np.random.default_rng(seed)
    uniq = np.unique(style)
    ymap = {int(s): j for j, s in enumerate(uniq)}
    idx = {int(s): np.where(style == s)[0] for s in uniq}
    tr, te = [], []
    for s in uniq:
        v = rng.permutation(idx[int(s)])
        tr += v[:5].tolist(); te += v[5:25].tolist()
    X = R[tr]; Y = torch.tensor([ymap[int(style[i])] for i in tr])
    Yte = np.array([ymap[int(style[i])] for i in te])
    clf = torch.nn.Linear(R.shape[1], len(uniq))
    opt = torch.optim.AdamW(clf.parameters(), lr=1e-2, weight_decay=1e-4)
    curve = []
    for st in range(steps):
        loss = F.cross_entropy(clf(X), Y)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if (st + 1) % 30 == 0:
            with torch.no_grad():
                curve.append(round(float((clf(R[te]).argmax(1).numpy() == Yte).mean()), 4))
    return curve


# ---------------------------------------------------------------- training
def train_reader(reader_name, data, dev, steps, lr, seed, tag):
    # Seed BEFORE construction: build() initialises weights from the global
    # torch RNG, so seeding afterwards leaves the initialisation dependent on
    # the process's random state (observed: local AUROC 0.4664 vs 0.4654 across
    # two identical runs). All artefacts must be reproducible.
    torch.manual_seed(seed); np.random.seed(seed)
    reader = build(reader_name).to(dev).train()
    opt = torch.optim.AdamW(reader.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 30) *
        (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps)))))
    sampler = SupConBatchSampler(data, seed=seed, batches_per_epoch=steps)
    gamma = gamma_for_dim(D_OUT) * 0.7
    t0 = time.time()
    losses = []
    print(f"\n=== TRAIN {tag}  params={n_params(reader):,}  steps={steps} ===", flush=True)
    for upd, idxs in enumerate(sampler, 1):
        meta = batch_meta(data, idxs)
        z = data.latent(meta["idx"]).to(dev)
        W = supcon_relation_weights(meta["style"].to(dev), meta["artist"].to(dev),
                                    meta["view"].to(dev),
                                    supervised=meta["supervised"].to(dev))
        q = reader(z)
        loss = supcon(q, W, TAU) + 0.1 * vicreg_var(q, gamma)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        losses.append(float(loss.detach()))
        if upd % 100 == 0 or upd == 1:
            el = time.time() - t0
            print(f"  [{tag}] {upd:5d}/{steps} | supcon={np.mean(losses[-100:]):.4f} "
                  f"| {el/upd*1000:.1f} ms/upd | {(steps-upd)*el/upd/60:.1f}min left",
                  flush=True)
        del z, q
        torch.cuda.empty_cache()
    torch.save({"model": reader.state_dict(), "tag": tag, "steps": steps,
                "seed": seed, "final_loss": float(np.mean(losses[-100:]))},
               OUT / f"{tag}_best.pt")
    print(f"  [{tag}] saved. wall={(time.time()-t0)/60:.1f}min "
          f"final_loss={np.mean(losses[-100:]):.4f}", flush=True)
    return reader.eval()


def evaluate(reader, data, dino_hold, flat_hold, dev, tag):
    print(f"\n--- EVAL {tag} ---", flush=True)
    with torch.no_grad():
        R = encode_all(reader, data, dev, hold=True)
    style, artist = data.hold_style, data.hold_artist
    auroc, pos, neg, owner, gaps, gap_owner = cross_artist_auroc(
        R, style, artist, dino_hold)
    pd_idx = np.where(data.hold_painter_disjoint)[0]
    auroc_pd = cross_artist_auroc(R, style, artist, dino_hold, subset=pd_idx)[0]
    fs = few_shot(R, flat_hold, style)
    rng = np.random.default_rng(0)
    us = np.unique(style[np.where(data.hold_unseen)[0]])
    tr, te = [], []
    for s in us:
        v = rng.permutation(np.where(style == s)[0])
        tr += v[:5].tolist(); te += v[5:15].tolist()
    unseen_retr = retrieval_acc(R, style, tr, te)
    curve = linear_probe_curve(R, style)
    res = {
        "auroc": round(auroc, 4),
        "auroc_painter_disjoint": round(auroc_pd, 4),
        "n_painter_disjoint": int(len(pd_idx)),
        "retrieval_5shot": round(fs["5"]["reader"], 4),
        "retrieval_5shot_flat": round(fs["5"]["flat"], 4),
        "retrieval_10shot": round(fs["10"]["reader"], 4),
        "unseen_retrieval": round(unseen_retr, 4),
        "linear_probe_curve": curve,
        "linear_probe_final": curve[-1] if curve else None,
        "param_count": n_params(reader),
        "mean_gap": round(float(gaps.mean()), 5),
    }
    print(f"  AUROC={auroc:.4f} (painter-disjoint {auroc_pd:.4f}, n={len(pd_idx)}) "
          f"mean_gap={gaps.mean():+.5f}", flush=True)
    print(f"  5-shot retr={fs['5']['reader']*100:.1f}% flat={fs['5']['flat']*100:.1f}% "
          f"10-shot={fs['10']['reader']*100:.1f}%  unseen={unseen_retr*100:.1f}% "
          f"probe_final={res['linear_probe_final']}", flush=True)
    (OUT / f"eval_{tag}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res, gaps, gap_owner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--readers", default="local,cnn")
    args = ap.parse_args()
    dev = args.device
    data = FrozenLatents().describe()
    dino_hold = holdout_dino(data, dev)
    flat_hold = []
    for i in range(0, len(data.hold_artist), 128):
        z = data.latent(np.arange(i, min(i + 128, len(data.hold_artist))), hold=True)
        flat_hold.append(F.normalize(z.flatten(1).float(), dim=1))
        del z
    flat_hold = torch.cat(flat_hold)
    print(f"[Ref] holdout DINO {tuple(dino_hold.shape)}, flat {tuple(flat_hold.shape)}",
          flush=True)

    # fixed 1340-d statistical readout: the FAILED baseline, recomputed on the
    # SAME holdout so paired comparison uses identical source enumeration.
    print("\n[FIXED BASELINE] frozen statistical readout (no training)", flush=True)
    Rf = []
    for i in range(0, len(data.hold_artist), 64):
        z = data.latent(np.arange(i, min(i + 64, len(data.hold_artist))), hold=True)
        Rf.append(style_readout(z)); del z
    Rf = torch.cat(Rf)
    auroc_fixed, _, _, _, gaps_fixed, gap_owner_fixed = cross_artist_auroc(
        Rf, data.hold_style, data.hold_artist, dino_hold)
    print(f"  fixed-readout AUROC = {auroc_fixed:.4f}  "
          f"mean_gap = {gaps_fixed.mean():+.5f}", flush=True)

    results, gaps_by, gap_owner = {}, {}, None
    for name in args.readers.split(","):
        name = name.strip()
        reader = train_reader(name, data, dev, args.steps, args.lr,
                              args.seed, name)
        res, gaps, owner = evaluate(reader, data, dino_hold, flat_hold, dev, name)
        assert len(gaps) == len(gaps_fixed) and np.array_equal(owner, gap_owner_fixed), \
            "source enumeration differs from fixed baseline"
        boot = paired_cluster_ci(gaps, gaps_fixed, owner, n_boot=2000, seed=11)
        res["paired_gap_vs_fixed"] = boot
        print(f"  paired gap vs fixed: mean={boot['mean']:+.5f} "
              f"CI=[{boot['ci'][0]:+.5f},{boot['ci'][1]:+.5f}] "
              f"excludes_zero={boot['excludes_zero']}", flush=True)
        results[name] = res
        gaps_by[name] = gaps
        gap_owner = owner
        del reader
        torch.cuda.empty_cache()

    if "local" in gaps_by and "cnn" in gaps_by:
        boot = paired_cluster_ci(gaps_by["local"], gaps_by["cnn"], gap_owner,
                                 n_boot=2000, seed=13)
        results["local"]["paired_gap_vs_cnn"] = boot
        print(f"\n[paired local-cnn gap] mean={boot['mean']:+.5f} "
              f"CI=[{boot['ci'][0]:+.5f},{boot['ci'][1]:+.5f}] "
              f"excludes_zero={boot['excludes_zero']}", flush=True)

    print("\n" + "=" * 92)
    print("GATE R")
    print("=" * 92)
    g = results.get("local", {})
    pb = g.get("paired_gap_vs_fixed", {})
    cond = {
        "auroc_ge_0.55": g.get("auroc", 0) >= 0.55,
        "retr5_gt_flat_latent_0.0708": g.get("retrieval_5shot", 0) > 0.0708,
        "unseen_ge_0.129": g.get("unseen_retrieval", 0) >= 0.129,
        "paired_bootstrap_excludes_zero": bool(pb.get("excludes_zero", False)),
        "content_matched_negatives": True,
    }
    print(f"  {'condition':<38}{'value':>12}{'result':>9}")
    valmap = {"auroc_ge_0.55": "auroc",
              "retr5_gt_flat_latent_0.0708": "retrieval_5shot",
              "unseen_ge_0.129": "unseen_retrieval"}
    for k, v in cond.items():
        val = g.get(valmap[k], v) if k in valmap else v
        print(f"  {k:<38}{str(val):>12}{'PASS' if v else 'FAIL':>9}")
    passed = all(cond.values())
    print(f"\n  GATE R: {'PASSED' if passed else 'FAILED'}", flush=True)
    (OUT / "gate_R.json").write_text(json.dumps({
        "gate_R": {"conditions": cond, "passed": bool(passed)},
        "readers": results,
        "protocol": {"steps": args.steps, "lr": args.lr, "seed": args.seed,
                     "held_painters": 250, "logical_batch": 64,
                     "fixed_reader_auroc": round(auroc_fixed, 4),
                     "fixed_reader_auroc_plan_baseline": 0.4535,
                     "flat_latent_5shot_this_run":
                         g.get("retrieval_5shot_flat"),
                     "flat_latent_5shot_plan_baseline": 0.0708,
                     "unseen_plan_baseline": 0.129}},
        indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[Saved] {OUT / 'gate_R.json'}", flush=True)


if __name__ == "__main__":
    main()
