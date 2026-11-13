"""P1: build the per-style (q, v) KDE banks and train the shared h_psi.

Pipeline (pre-registered §2-§4):
 1. painting-level train split -> extract (q, f) patch features on GPU chunks
 2. global standardization over the train split (q and f separately)
 3. train shared h_psi: q_std -> f_std (L2, cross-style, no style conditioning)
 4. residual v = f_std - h_psi(q_std), globally re-standardized
 5. color/texture column views re-standardized per block
 6. per-style subsample <= 200k patches (seed 0) -> runs/v10_local_4070/bank/kde_bank.pt

Run: python -m exp.v10.bank
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from .features import D_F, D_Q, HPsi, f_feats, patchify, q_feats
from .features import F_COLOR, F_TEXTURE
from .splits import POOL_DIR, STYLES, build_splits, check_integrity

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "runs/v10_local_4070"
PER_STYLE_CAP = 200_000
LATENT_CHUNK = 512          # latents per GPU chunk (~32k patches)
TRAIN_EPOCHS = 40
TRAIN_BATCH = 8192
TRAIN_MAX_PATCHES = 600_000
SEED = 0


def extract_style_feats(si: int, style: str, idx: torch.Tensor, device: str):
    """-> (Q_all fp16 cpu, F_all fp16 cpu, ids used) for the given painting subset."""
    d = torch.load(POOL_DIR / f"{si:02d}_{style}.pt",
                   map_location="cpu", weights_only=False)
    lat = d["latents"]
    ids = [str(x) for i, x in enumerate(d["ids"]) if i in set(idx.tolist())]
    sel = lat[idx]
    qs, fs = [], []
    for i in range(0, sel.shape[0], LATENT_CHUNK):
        z = sel[i:i + LATENT_CHUNK].float().to(device)
        p = patchify(z)
        qs.append(q_feats(p).half().cpu())
        fs.append(f_feats(p).half().cpu())
    return torch.cat(qs), torch.cat(fs), ids


def train_hpsi(q_all: torch.Tensor, f_all: torch.Tensor, device: str):
    g = torch.Generator().manual_seed(SEED)
    n = q_all.shape[0]
    if n > TRAIN_MAX_PATCHES:
        sel = torch.randperm(n, generator=g)[:TRAIN_MAX_PATCHES]
        q_tr, f_tr = q_all[sel], f_all[sel]
    else:
        q_tr, f_tr = q_all, f_all
    hpsi = HPsi().to(device)
    opt = torch.optim.Adam(hpsi.parameters(), lr=1e-3)
    n_tr = q_tr.shape[0]
    print(f"[hpsi] train patches {n_tr}, epochs {TRAIN_EPOCHS}", flush=True)
    t0 = time.time()
    for ep in range(TRAIN_EPOCHS):
        perm = torch.randperm(n_tr, generator=g)
        tot, nb = 0.0, 0
        for i in range(0, n_tr, TRAIN_BATCH):
            bidx = perm[i:i + TRAIN_BATCH]
            qb = q_tr[bidx].to(device)
            fb = f_tr[bidx].to(device)
            pred = hpsi(qb)
            loss = (pred - fb).square().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item(); nb += 1
        if ep % 10 == 0 or ep == TRAIN_EPOCHS - 1:
            print(f"  ep{ep} mse {tot / max(nb, 1):.4f} {time.time()-t0:.0f}s", flush=True)
    return hpsi


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "bank").mkdir(exist_ok=True)
    t0 = time.time()
    splits = build_splits()
    rep = check_integrity(splits)
    (OUT / "splits_report.json").write_text(json.dumps(rep, indent=1))
    print("split integrity:", json.dumps(rep), flush=True)

    # 1) extract features for the train split (holdout extracted on demand later)
    Q_raw, F_raw = {}, {}
    for si, s in enumerate(STYLES):
        q, f, _ = extract_style_feats(si, s, splits[s]["train_idx"], device)
        Q_raw[s], F_raw[s] = q, f
        print(f"[feat] {s}: {q.shape[0]} patches "
              f"({splits[s]['train_idx'].shape[0]} paintings) {time.time()-t0:.0f}s",
              flush=True)

    # 2) global standardization over the train split
    Q_all = torch.cat([Q_raw[s] for s in STYLES]).float()
    F_all = torch.cat([F_raw[s] for s in STYLES]).float()
    q_mean, q_std = Q_all.mean(0), Q_all.std(0).clamp_min(1e-6)
    f_mean, f_std = F_all.mean(0), F_all.std(0).clamp_min(1e-6)
    del Q_all, F_all

    # 3) shared h_psi on standardized space
    Q_std = {s: ((Q_raw[s].float() - q_mean) / q_std).half() for s in STYLES}
    F_std = {s: ((F_raw[s].float() - f_mean) / f_std).half() for s in STYLES}
    hpsi = train_hpsi(torch.cat([Q_std[s] for s in STYLES]).float(),
                      torch.cat([F_std[s] for s in STYLES]).float(), device)

    # 4) residual + 5) column views
    with torch.no_grad():
        R_std_all = []
        for s in STYLES:
            r = F_std[s].float() - hpsi(Q_std[s].float().to(device)).cpu()
            R_std_all.append(r)
        R_all = torch.cat(R_std_all)
        r_mean, r_std = R_all.mean(0), R_all.std(0).clamp_min(1e-6)
        del R_all, R_std_all
        Fc_all = torch.cat([(F_std[s][:, F_COLOR[0]:F_COLOR[1]].float()) for s in STYLES])
        Ft_all = torch.cat([(F_std[s][:, F_TEXTURE[0]:F_TEXTURE[1]].float()) for s in STYLES])
        c_mean, c_std = Fc_all.mean(0), Fc_all.std(0).clamp_min(1e-6)
        t_mean, t_std = Ft_all.mean(0), Ft_all.std(0).clamp_min(1e-6)
        del Fc_all, Ft_all

    # 6) per-style subsample + save
    g = torch.Generator().manual_seed(SEED)
    bank = {"styles": list(STYLES), "Q": {}, "V": {"f": {}, "res": {},
            "f_color": {}, "f_texture": {}},
            "stats": {"q_mean": q_mean, "q_std": q_std, "f_mean": f_mean,
                      "f_std": f_std, "r_mean": r_mean, "r_std": r_std,
                      "c_mean": c_mean, "c_std": c_std, "t_mean": t_mean,
                      "t_std": t_std},
            "meta": {"per_style_cap": PER_STYLE_CAP, "seed": SEED,
                     "hpsi_epochs": TRAIN_EPOCHS,
                     "train_paintings": {s: int(splits[s]["train_idx"].shape[0])
                                         for s in STYLES}}}
    for s in STYLES:
        n = Q_std[s].shape[0]
        take = torch.randperm(n, generator=g)[:min(n, PER_STYLE_CAP)]
        bank["Q"][s] = Q_std[s][take]
        fs = F_std[s][take]
        bank["V"]["f"][s] = fs
        with torch.no_grad():
            r = fs.float() - hpsi(Q_std[s][take].float().to(device)).cpu()
            bank["V"]["res"][s] = ((r - r_mean) / r_std).half()
            fc = fs[:, F_COLOR[0]:F_COLOR[1]].float()
            ft = fs[:, F_TEXTURE[0]:F_TEXTURE[1]].float()
            bank["V"]["f_color"][s] = ((fc - c_mean) / c_std).half()
            bank["V"]["f_texture"][s] = ((ft - t_mean) / t_std).half()
        print(f"[bank] {s}: {bank['Q'][s].shape[0]} patches kept", flush=True)
    torch.save({"hpsi": hpsi.state_dict()}, OUT / "bank/hpsi.pt")
    torch.save(bank, OUT / "bank/kde_bank.pt")
    print("saved", OUT / "bank/kde_bank.pt",
          (OUT / "bank/kde_bank.pt").stat().st_size, f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
