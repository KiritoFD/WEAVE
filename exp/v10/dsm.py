"""P4: conditional denoising score matching network (plan §7).

D_theta(v~, q, t, sigma) -> v_hat, trained with standard DSM on train-split
patches (same standardized f space as variant B):

    L_DSM = E || D_theta(f + sigma*eps, q, t, sigma) - f ||^2
    score_theta(f~, q, t, sigma) = (D_theta - f~) / sigma^2

~0.5M params; the network is compared to the closed-form KDE on the SAME
experiment-A recovery protocol (exp_d_dsm.py). Pre-registered: keep KDE unless
DSM is strictly better.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import torch
import torch.nn as nn

from .features import D_F, D_Q, HPsi
from .splits import STYLES

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "runs/v10_local_4070"
SIGMAS = (0.1, 0.3, 0.5, 1.0)
HIDDEN = 448
TRAIN_MAX = 400_000
EPOCHS = 60
BATCH = 8192
SEED = 0
SIGMA_EVAL = 0.1   # smallest sigma for near-clean queries (standard practice)


class CondDenoiser(nn.Module):
    def __init__(self, d_f: int = D_F, d_q: int = D_Q, hidden: int = HIDDEN,
                 n_styles: int = 5, n_sigmas: int = len(SIGMAS)):
        super().__init__()
        self.t_emb = nn.Embedding(n_styles, 32)
        self.s_emb = nn.Embedding(n_sigmas, 32)
        d_in = d_f + d_q + 32 + 32
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, d_f))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, v_tilde, q, t_idx, s_idx):
        e_t = self.t_emb(t_idx)
        e_s = self.s_emb(s_idx)
        return self.net(torch.cat([v_tilde, q, e_t, e_s], dim=1))

    def score(self, v_tilde, q, t_idx, sigma: float):
        s_idx = torch.full((v_tilde.shape[0],), SIGMAS.index(sigma),
                           device=v_tilde.device, dtype=torch.long)
        return (self.forward(v_tilde, q, t_idx, s_idx) - v_tilde) / (sigma ** 2)


def train_dsm(device="cuda"):
    """Train on the train-split patches; reuses the bank's standardization."""
    raw = torch.load(OUT / "bank/kde_bank.pt", map_location="cpu", weights_only=False)
    stats = {k: v.to(device) for k, v in raw["stats"].items()}
    q_mean, q_std = stats["q_mean"], stats["q_std"]
    f_mean, f_std = stats["f_mean"], stats["f_std"]
    from .features import patchify, q_feats, f_feats
    from .splits import POOL_DIR, build_splits

    splits = build_splits()
    g = torch.Generator().manual_seed(SEED)
    model = CondDenoiser().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    t0 = time.time()
    for ep in range(EPOCHS):
        tot, nb = 0.0, 0
        for si, s in enumerate(STYLES):
            d = torch.load(POOL_DIR / f"{si:02d}_{s}.pt",
                           map_location="cpu", weights_only=False)
            sel = splits[s]["train_idx"]
            lat = d["latents"][sel]
            take = torch.randperm(lat.shape[0], generator=g)[:BATCH // 64]
            z = lat[take].float().to(device)
            p = patchify(z)
            with torch.no_grad():
                q = ((q_feats(p) - q_mean) / q_std)
                f = ((f_feats(p) - f_mean) / f_std)
            sigma = SIGMAS[torch.randint(0, len(SIGMAS), (1,), generator=g).item()]
            eps = torch.randn_like(f)
            f_tilde = f + sigma * eps
            t_idx = torch.full((f.shape[0],), si, device=device, dtype=torch.long)
            s_idx = torch.full((f.shape[0],), SIGMAS.index(sigma),
                               device=device, dtype=torch.long)
            pred = model(f_tilde, q, t_idx, s_idx)
            loss = (pred - f).square().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item(); nb += 1
        if ep % 10 == 0 or ep == EPOCHS - 1:
            print(f"[dsm] ep{ep} mse {tot / max(nb, 1):.4f} {time.time()-t0:.0f}s",
                  flush=True)
    (OUT / "bank").mkdir(exist_ok=True)
    torch.save(model.state_dict(), OUT / "bank/dsm.pt")
    print("saved", OUT / "bank/dsm.pt", f"{time.time()-t0:.0f}s")
    return model


def load_dsm(device="cuda"):
    model = CondDenoiser().to(device)
    ck = torch.load(OUT / "bank/dsm.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ck)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def dsm_direction(model, stats, hpsi, z: torch.Tensor, style_idx: int,
                  sigma: float = SIGMA_EVAL) -> torch.Tensor:
    """Per-sample normalized grad_z of sum_r [D*f - f^2/2] (score ascent).

    q path is stop-gradient (same convention as the KDE direction).
    """
    from .features import patchify, q_feats, f_feats
    q_mean, q_std = stats["q_mean"], stats["q_std"]
    f_mean, f_std = stats["f_mean"], stats["f_std"]
    b = z.shape[0]
    g = torch.zeros_like(z)
    for i in range(b):
        with torch.enable_grad():
            zi = z[i:i + 1].detach().clone().requires_grad_(True)
            with torch.no_grad():
                p0 = patchify(zi.detach())
                q = ((q_feats(p0) - q_mean) / q_std)
            p = patchify(zi)
            f = ((f_feats(p) - f_mean) / f_std)
            s_idx = torch.full((f.shape[0],), SIGMAS.index(sigma),
                               device=z.device, dtype=torch.long)
            t_idx = torch.full((f.shape[0],), style_idx,
                               device=z.device, dtype=torch.long)
            with torch.no_grad():
                # D must be detached: the score-ascent potential is
                # L(f) = D.detach()*f - f^2/2, whose gradient is D - f = sigma^2*score.
                # Letting autograd differentiate through D would add the Jacobian
                # term (dD/df)^T f and flip/distort the direction.
                D = model(f, q, t_idx, s_idx)
            L = (D * f - f.square() / 2).sum()
            gi = torch.autograd.grad(L, zi)[0]
        flat = gi.flatten(1)
        nrm = flat.norm(dim=1, keepdim=True).clamp_min(1e-12)
        g[i:i + 1] = (flat / nrm).view_as(zi)
    return g
