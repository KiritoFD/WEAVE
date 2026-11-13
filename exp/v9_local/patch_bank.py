"""Build per-style local patch style bank from the real training pool.

Latents (B,4,64,64) -> 64 non-overlapping 8x8 patches -> Haar-ish stats
(LL 4x4 + 3 HF-dir std pooled 2x2 + channel mean/std) ~ D=36.
K=128 prototypes per style via GPU kmeans on a subsample. CPU/GPU agnostic;
saves runs/v9_local_4070/bank/style_bank.pt.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
PATCH = 8
K = 128
MAX_PATCHES_PER_STYLE = 120_000  # subsample cap for kmeans


def patch_feats(z: torch.Tensor) -> torch.Tensor:
    """(B,4,64,64) -> (B*64, D) differentiable patch descriptors."""
    b = z.shape[0]
    # view splits H->(H_out,H_in), W->(W_out,W_in); patch=(H_out,W_out)
    p = z.view(b, 4, 8, 8, 8, 8) \
         .permute(0, 2, 4, 1, 3, 5) \
         .reshape(b * 64, 4, PATCH, PATCH)
    # Haar on each 8x8 patch: reshape (4ch,4,2,4,2)
    x = p.reshape(-1, 4, 4, 2, 4, 2)
    ll = x.mean(dim=(3, 5))                       # (N,4,4,4) -> LL 4x4 per ch
    ll = ll.reshape(-1, 4 * 4 * 4)
    # horizontal/vertical/diagonal diffs on the 8x8 grid
    g = p  # (N,4,8,8)
    dh = (g[:, :, :, 1:] - g[:, :, :, :-1])       # (N,4,8,7)
    dv = (g[:, :, 1:, :] - g[:, :, :-1, :])       # (N,4,7,8)
    dd = (g[:, :, 1:, 1:] - g[:, :, :-1, :-1])    # (N,4,7,7)
    def pool_std(t):
        t = t.square().mean(1)                    # over channels -> (N,*,*)
        t = F.avg_pool2d(t.unsqueeze(1), 4).squeeze(1)  # rough 2x2-ish pooling
        return t.flatten(1)
    parts = [ll, pool_std(dh), pool_std(dv), pool_std(dd),
             p.mean(dim=(2, 3)), p.std(dim=(2, 3))]
    return torch.cat(parts, 1)


FEAT_DIM = None  # set after first call


def kmeans(x: torch.Tensor, k: int, iters: int = 30, seed: int = 0):
    g = torch.Generator(device=x.device).manual_seed(seed)
    idx = torch.randperm(x.shape[0], generator=g, device=x.device)[:k]
    c = x[idx].clone()
    for _ in range(iters):
        d = torch.cdist(x, c)
        a = d.argmin(1)
        for j in range(k):
            m = a == j
            if m.any():
                c[j] = x[m].mean(0)
    return c, a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(REPO / "weights_local/sd_pool_orig"))
    ap.add_argument("--out", default=str(REPO / "runs/v9_local_4070/bank/style_bank.pt"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--per-style", type=int, default=MAX_PATCHES_PER_STYLE)
    a = ap.parse_args()
    dev = a.device
    bank = {"styles": STYLES, "k": K, "protos": {}, "feat_dim": None,
            "mean": None, "std": None}
    t0 = time.time()
    # global normalisation stats from first pass over a subset
    all_feats = []
    for si, s in enumerate(STYLES):
        d = torch.load(Path(a.pool) / f"{si:02d}_{s}.pt",
                       map_location="cpu", weights_only=False)
        z = d["latents"].float()
        n = min(z.shape[0], a.per_style // 64)
        z = z[torch.randperm(z.shape[0])[:n]]
        with torch.no_grad():
            f = patch_feats(z.to(dev)).cpu()
        all_feats.append(f)
        print(s, "latents", z.shape[0], "patches", f.shape[0], flush=True)
    F_all = torch.cat(all_feats)
    mu, sd = F_all.mean(0), F_all.std(0).clamp_min(1e-6)
    bank["feat_dim"] = F_all.shape[1]
    bank["mean"], bank["std"] = mu, sd
    del F_all, all_feats
    for si, s in enumerate(STYLES):
        d = torch.load(Path(a.pool) / f"{si:02d}_{s}.pt",
                       map_location="cpu", weights_only=False)
        z = d["latents"].float()
        n = min(z.shape[0], a.per_style // 64)
        z = z[torch.randperm(z.shape[0])[:n]]
        with torch.no_grad():
            f = (patch_feats(z.to(dev)) - mu.to(dev)) / sd.to(dev)
            if f.shape[0] > a.per_style:
                f = f[torch.randperm(f.shape[0], device=dev)[:a.per_style]]
            proto, assign = kmeans(f, K, iters=25, seed=si)
        bank["protos"][s] = proto.cpu()
        print(s, "protos", proto.shape, f"{time.time()-t0:.0f}s", flush=True)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bank, out)
    print("saved", out, out.stat().st_size, f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
