"""Style-VAE prep: build the fixed orthogonal style subspace P and all constants.

Steps
 1. Load cached A0 train latents + CSD teacher features.
 2. mu_z (train latent mean), t_bar (train teacher mean).
 3. CSD-PLS: SVD of cross-covariance Z_c^T T_c -> P0 = U[:, :768]^T   (768 x 16384)
 4. Orthogonal Procrustes: rotate coordinates so that P z aligns with teacher t.
 5. Style prototypes: mean normalized CSD feature per style (136 x 768).
 6. Save style_subspace.pt

Everything here is deterministic and frozen for the whole training run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/Workspace/WEAVE_VAE")

import numpy as np
import torch
import torch.nn.functional as F

OUT = Path("/root/Workspace/WEAVE_VAE/runs_vae/style_vae")
OUT.mkdir(parents=True, exist_ok=True)
CACHE = Path("/root/Workspace/WEAVE_VAE/runs_vae/v2_diagnosis/p1_cache")
DIM = 16384
K = 768
DEV = "cuda"


def main():
    n_train = 20000
    z = torch.load(CACHE / f"train_z_{n_train}.pt", weights_only=False)
    t = torch.load(CACHE / f"train_t_{n_train}.pt", weights_only=False)
    meta = torch.load(CACHE / f"train_meta_{n_train}.pt", weights_only=False)

    Z = F.pixel_unshuffle(z, 4).flatten(1).to(DEV).float()
    T = F.normalize(t.to(DEV).float(), dim=-1)
    y = meta["style"].to(DEV)
    print(f"[Data] Z={tuple(Z.shape)} T={tuple(T.shape)} styles={int(y.max())+1}", flush=True)

    mu_z = Z.mean(0, keepdim=True)
    t_bar = T.mean(0, keepdim=True)
    Zc = Z - mu_z
    Tc = T - t_bar

    # ---- CSD-PLS ----
    C = (Zc.t() @ Tc) / Zc.shape[0]
    U, S, Vh = torch.linalg.svd(C, full_matrices=False)
    P0 = U[:, :K].t().contiguous()                     # (K, DIM)
    print(f"[PLS] sv top={S[0].item():.4e} kth={S[K-1].item():.4e}", flush=True)
    print(f"[PLS] orth err = {(P0 @ P0.t() - torch.eye(K, device=DEV)).abs().max().item():.2e}", flush=True)

    # ---- Procrustes on an exactly orthonormal basis (float64 for accuracy) ----
    Qp, _ = torch.linalg.qr(P0.t())
    P = Qp.t().contiguous()                            # exactly orthonormal, same row span
    A = (P.double() @ Zc.double().t()).t()             # (N, K)
    M = A.t() @ Tc.double()                            # (K, K)
    Uu, Ss, Vv = torch.linalg.svd(M, full_matrices=False)
    R = (Uu @ Vv).float()                              # (K, K) orthogonal
    print(f"[Procrustes] R orth err = {(R @ R.t() - torch.eye(K, device=DEV)).abs().max().item():.2e}",
          flush=True)
    P = (R.t() @ P).contiguous()                       # rotate coordinates toward teacher
    print(f"[Final P] orth err = {(P @ P.t() - torch.eye(K, device=DEV)).abs().max().item():.2e}",
          flush=True)

    u = (P @ Zc.t()).t()
    cos = F.cosine_similarity(F.normalize(u, dim=-1), F.normalize(Tc, dim=-1), dim=-1)
    print(f"[Check] train cos(u, t-tbar): mean={cos.mean().item():.4f} "
          f"median={cos.median().item():.4f}", flush=True)

    # ---- style prototypes ----
    n_styles = int(y.max().item()) + 1
    proto = torch.zeros(n_styles, K, device=DEV)
    for s in range(n_styles):
        m = (y == s)
        if int(m.sum()) > 0:
            proto[s] = F.normalize(Tc[m].mean(0, keepdim=True), dim=-1)
    proto = F.normalize(proto, dim=-1)
    print(f"[Proto] {tuple(proto.shape)}, styles with samples: "
          f"{int((proto.abs().sum(1) > 0).sum())}", flush=True)

    torch.save({
        "P": P.cpu(),
        "mu_z": mu_z.cpu(),
        "t_bar": t_bar.cpu(),
        "prototypes": proto.cpu(),
        "n_styles": n_styles,
        "k": K,
        "dim": DIM,
        "train_cos_mean": float(cos.mean().item()),
    }, OUT / "style_subspace.pt")
    print(f"[Saved] {OUT/'style_subspace.pt'}", flush=True)


if __name__ == "__main__":
    main()
