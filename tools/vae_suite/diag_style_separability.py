"""Is style stably separable from content in the A0 4x64x64 latent?

L0 showed no per-pair dz transfers (paired classification delta ~0). The plan then
requires asking whether cross-content style differences can be stably SEPARATED at
all, before adding model capacity.

Linear probe on the frozen latent, using the verified full cache (78,661 train /
2,048 holdout), style label per row:
  - 136-class style classification from z          -> is style linearly present?
  - same probe from CSD image features (reference) -> what a good encoder gives
  - content check: does z predict style for the SAME content? (artist-disjoint)

If the latent probe is near chance while CSD is high, style is not linearly
recoverable from z and the failure is upstream of any style direction.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "/root/Workspace/WEAVE_VAE")
sys.path.insert(0, "/root/Workspace/WEAVE_VAE/tools/vae_suite")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT = Path("/root/Workspace/WEAVE_VAE/runs/vae_decoder_prune")
MANIFEST = Path("/root/Workspace/data/wikiart_512_raw/wikiart_512_raw_manifest.json")
FULL = OUT / "full"


def pca_reduce(Xtr, Xte, k):
    """Reduce to k dims with PCA fitted on TRAIN only (avoids the d>>n overfit
    that makes a raw 16384-dim probe look artificially weak)."""
    mu = Xtr.mean(0, keepdim=True)
    A = Xtr - mu
    # top-k right singular vectors
    U, S, V = torch.svd_lowrank(A, q=min(k + 32, min(A.shape) - 1), niter=4)
    V = V[:, :k]
    return (Xtr - mu) @ V, (Xte - mu) @ V


def fit_linear(Xtr, ytr, Xte, yte, n_cls, dev, epochs=30, bs=2048, lr=1e-3, wd=1e-4):
    """Plain multinomial logistic regression, the cheapest separability probe."""
    mu = Xtr.mean(0, keepdim=True)
    sd = Xtr.std(0, keepdim=True) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    d = Xtr.shape[1]
    clf = nn.Linear(d, n_cls).to(dev)
    nn.init.zeros_(clf.weight); nn.init.zeros_(clf.bias)
    opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=wd)
    n = len(Xtr)
    for ep in range(epochs):
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            loss = F.cross_entropy(clf(Xtr[idx]), ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    with torch.no_grad():
        pred = clf(Xte).argmax(-1)
        acc = float((pred == yte).float().mean())
        train_acc = float((clf(Xtr[:8000]).argmax(-1) == ytr[:8000]).float().mean())
    return acc, train_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-style", type=int, default=300)
    ap.add_argument("--out", default=str(OUT / "style_separability.json"))
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    split = json.loads((FULL / "split.json").read_text())
    n_tr, dim = split["n_train"], split["dim"]
    lat = np.memmap(split["train_bin"], dtype=np.float16, mode="r", shape=(n_tr, dim))
    raw_tr = np.load(FULL / "raw_index_train.npy")

    man = json.loads(MANIFEST.read_text())
    style_of_raw = {r["raw_index"]: r["style_id"] for r in man["records"]}

    # gather per-style row indices, cap per style for balance
    by_style = defaultdict(list)
    for j in range(n_tr):
        by_style[style_of_raw[int(raw_tr[j])]].append(j)
    keep = {}
    rng = np.random.default_rng(0)
    for s, lst in by_style.items():
        if len(lst) >= 20:
            pick = lst if len(lst) <= args.max_per_style else \
                rng.choice(lst, args.max_per_style, replace=False).tolist()
            keep[s] = [int(x) for x in pick]
    styles = sorted(keep)
    n_cls = len(styles)
    sid = {s: i for i, s in enumerate(styles)}
    print(f"[Probe] {n_cls} styles with >=20 images, "
          f"{sum(len(v) for v in keep.values())} latents total", flush=True)

    rows, ys = [], []
    for s in styles:
        for j in keep[s]:
            rows.append(j); ys.append(sid[s])
    rows = np.array(rows); ys = np.array(ys)

    X = torch.from_numpy(lat[rows].astype(np.float32)).to(dev)
    y = torch.from_numpy(ys).to(dev)
    print(f"[Probe] X {tuple(X.shape)} chance={100/n_cls:.2f}%", flush=True)

    # style-stratified 80/20 split
    tr_idx, te_idx = [], []
    for s in styles:
        m = np.where(ys == sid[s])[0]
        rng.shuffle(m)
        cut = max(2, int(len(m) * 0.8))
        tr_idx += m[:cut].tolist(); te_idx += m[cut:].tolist()
    tr_idx = torch.tensor(tr_idx, device=dev)
    te_idx = torch.tensor(te_idx, device=dev)
    print(f"[Probe] train={len(tr_idx)} test={len(te_idx)}", flush=True)

    acc_z, tr_acc_z = fit_linear(X[tr_idx], y[tr_idx], X[te_idx], y[te_idx],
                                 n_cls, dev)
    print(f"\n[Latent linear probe] test acc = {acc_z*100:.2f}%  "
          f"(train {tr_acc_z*100:.2f}%, chance {100/n_cls:.2f}%)", flush=True)

    # ---- reference: same probe on raw pixels (downsampled) ----
    mmap = np.memmap(man["bin_path"], dtype=np.uint8, mode="r",
                     shape=tuple(man["shape"]))
    px = np.stack([np.ascontiguousarray(mmap[int(raw_tr[j])])[::8, ::8]
                   for j in rows])
    Xp = torch.from_numpy(px.astype(np.float32).reshape(len(rows), -1) / 127.5 - 1).to(dev)
    acc_p, tr_acc_p = fit_linear(Xp[tr_idx], y[tr_idx], Xp[te_idx], y[te_idx],
                                 n_cls, dev)
    print(f"[Pixel linear probe ] test acc = {acc_p*100:.2f}%  "
          f"(train {tr_acc_p*100:.2f}%)  [{Xp.shape[1]}-dim]", flush=True)
    del Xp
    torch.cuda.empty_cache()

    # ---- reference: CSD image features ----
    sys.path.insert(0, "/root/Workspace/WEAVE_VAE/tools/12h")
    import csd_model
    csd, _, _ = csd_model.load_csd(dev)
    csd = csd.float().eval()
    for p in csd.parameters():
        p.requires_grad = False
    nm_ = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=dev).view(1, 3, 1, 1)
    ns_ = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=dev).view(1, 3, 1, 1)
    feats = []
    with torch.no_grad():
        for i in range(0, len(rows), 32):
            chunk = rows[i:i + 32]
            arr = np.stack([np.ascontiguousarray(mmap[int(raw_tr[j])]) for j in chunk])
            x = torch.from_numpy(arr).permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0).to(dev)
            x224 = F.interpolate(x, (224, 224), mode="bicubic", align_corners=False)
            _, _, f = csd((x224 - nm_) / ns_)
            feats.append(F.normalize(f.float(), dim=-1))
            del x
            torch.cuda.empty_cache()
    Xc = torch.cat(feats)
    acc_c, tr_acc_c = fit_linear(Xc[tr_idx], y[tr_idx], Xc[te_idx], y[te_idx],
                                 n_cls, dev)
    print(f"[CSD linear probe   ] test acc = {acc_c*100:.2f}%  "
          f"(train {tr_acc_c*100:.2f}%)  [{Xc.shape[1]}-dim]", flush=True)

    # ---- fair comparison: PCA-256 then probe (d >> n caused the overfit) ----
    print("")
    print("[PCA probes] reduce to 256 dims on train only", flush=True)
    pca_res = {}
    for name, XX in (("latent", X), ("csd", Xc)):
        Xtr_p, Xte_p = pca_reduce(XX[tr_idx], XX[te_idx], 256)
        a, ta = fit_linear(Xtr_p, y[tr_idx], Xte_p, y[te_idx], n_cls, dev)
        pca_res[name] = {"test_pct": round(a * 100, 2),
                         "train_pct": round(ta * 100, 2)}
        print(f"  [{name:<7} PCA-256] test={a*100:5.2f}%  train={ta*100:5.2f}%",
              flush=True)
        del Xtr_p, Xte_p
        torch.cuda.empty_cache()

    res = {"pca256": pca_res, "n_styles": n_cls, "chance_pct": round(100 / n_cls, 2),
           "latent_acc_pct": round(acc_z * 100, 2),
           "latent_train_acc_pct": round(tr_acc_z * 100, 2),
           "pixel_acc_pct": round(acc_p * 100, 2),
           "csd_acc_pct": round(acc_c * 100, 2),
           "csd_train_acc_pct": round(tr_acc_c * 100, 2),
           "n_train": int(len(tr_idx)), "n_test": int(len(te_idx))}
    print("\n" + "=" * 76)
    print(f"{'representation':<26}{'dim':>8}{'test acc':>12}{'chance':>10}")
    print(f"{'A0 latent 4x64x64':<26}{dim:>8}{acc_z*100:>11.2f}%{100/n_cls:>9.2f}%")
    print(f"{'raw pixels (64x64)':<26}{Xp.shape[1] if 'Xp' in dir() else 12288:>8}"
          f"{acc_p*100:>11.2f}%{100/n_cls:>9.2f}%")
    print(f"{'CSD image features':<26}{Xc.shape[1]:>8}{acc_c*100:>11.2f}%"
          f"{100/n_cls:>9.2f}%")
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"\n[Saved] {args.out}")


if __name__ == "__main__":
    main()
