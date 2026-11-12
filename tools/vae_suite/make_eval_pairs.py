"""Cache the 20-class eval set indices, labels and evaluation pairs (CPU only).

Used by in-training validation so the training loop does not rebuild pairs.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "/root/Workspace/WEAVE_VAE")

import numpy as np
import torch

OUT = Path("/root/Workspace/WEAVE_VAE/runs/vae_style_rank20")
MANIFEST = Path("/root/Workspace/data/wikiart_512_raw/wikiart_512_raw_manifest.json")


def build_eval_set(max_per_style=120):
    man = json.loads(MANIFEST.read_text())
    val_idx = set(man["val_indices"])
    keep = set(c["id"] for c in json.loads((OUT / "classes.json").read_text()))
    by_style = defaultdict(list)
    for r in man["records"]:
        if r["in_train"] or r["raw_index"] in val_idx or r["style_id"] not in keep:
            continue
        by_style[r["style_id"]].append(r["raw_index"])
    rng = np.random.default_rng(7)
    picks = []
    for s, lst in by_style.items():
        n = min(max_per_style, len(lst))
        picks += [int(x) for x in rng.choice(lst, n, replace=False)]
    picks = sorted(set(picks))
    recs = {r["raw_index"]: r for r in man["records"]}
    return picks, recs


def build_pairs(y, a, g, seed=42):
    rng = np.random.default_rng(seed)
    N = len(y)
    same, cross, diff, hard = [], [], [], []
    for i in range(N):
        sc = np.where(y == y[i])[0]; sc = sc[sc != i]
        if len(sc):
            same.append((i, int(rng.choice(sc))))
            if a[i] >= 0:
                ca = sc[a[sc] != a[i]]
                if len(ca):
                    cross.append((i, int(rng.choice(ca))))
        dc = np.where(y != y[i])[0]
        if len(dc):
            diff.append((i, int(rng.choice(dc))))
            if g[i] >= 0:
                hc = dc[g[dc] == g[i]]
                if len(hc):
                    hard.append((i, int(rng.choice(hc))))
    return same, cross, diff, hard


def build_cm(dino, y, a, g, cap=1200, seed=42):
    rng = np.random.default_rng(seed)
    pos, neg = [], []
    for i in rng.permutation(len(y)):
        if len(pos) >= cap:
            break
        pc = np.where(y == y[i])[0]; pc = pc[pc != i]
        if a[i] >= 0:
            pc = pc[a[pc] != a[i]]
        if len(pc) == 0:
            continue
        nc = np.where((y != y[i]) & (g == g[i]))[0]
        if len(nc) < 2:
            nc = np.where(y != y[i])[0]
        if len(nc) < 2:
            continue
        if len(pc) > 16: pc = rng.choice(pc, 16, replace=False)
        if len(nc) > 64: nc = rng.choice(nc, 64, replace=False)
        sp = dino[pc] @ dino[i]; sn = dino[nc] @ dino[i]
        gap = np.abs(sp[:, None] - sn[None, :])
        pi, ni = np.unravel_index(np.argmin(gap), gap.shape)
        pos.append((int(i), int(pc[pi]))); neg.append((int(i), int(nc[ni])))
    return pos, neg


def main():
    classes = json.loads((OUT / "classes.json").read_text())
    id2new = {c["id"]: i for i, c in enumerate(classes)}

    picks, recs = build_eval_set(120)
    y, a, g = [], [], []
    for r in picks:
        rr = recs[r]
        y.append(id2new[rr["style_id"]]); a.append(rr.get("artist_id", -1)); g.append(rr.get("genre_id", -1))
    y = np.array(y); a = np.array(a); g = np.array(g)
    print(f"[EvalSet] {len(picks)} images, {len(np.unique(y))} styles", flush=True)

    dino = torch.load(OUT / "eval_dino.pt", weights_only=False).numpy()
    dino = dino / (np.linalg.norm(dino, axis=1, keepdims=True) + 1e-8)

    same, cross, diff, hard = build_pairs(y, a, g)
    cm_pos, cm_neg = build_cm(dino, y, a, g)
    print(f"[Pairs] cross={len(cross)} hard={len(hard)} cm={len(cm_pos)}", flush=True)

    torch.save({
        "raw": picks, "y": y, "a": a, "g": g, "dino": dino,
        "same": same, "cross": cross, "diff": diff, "hard": hard,
        "cm_pos": cm_pos, "cm_neg": cm_neg,
    }, OUT / "eval_pairs.pt")
    print(f"[Saved] {OUT/'eval_pairs.pt'}", flush=True)


if __name__ == "__main__":
    main()
