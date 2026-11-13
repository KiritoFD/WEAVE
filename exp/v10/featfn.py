"""Shared helpers: bank/hpsi loading, bank-space feature functions, holdout access."""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from .features import F_COLOR, F_TEXTURE, HPsi, f_feats, patchify, q_feats
from .kde import VARIANTS, StyleBank
from .splits import POOL_DIR, STYLES

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "runs/v10_local_4070"


def load_bank(device="cuda", bank_path=None):
    bank = StyleBank(bank_path or (OUT / "bank/kde_bank.pt"), device)
    ck = torch.load(OUT / "bank/hpsi.pt", map_location="cpu", weights_only=False)
    hpsi = HPsi().to(device)
    hpsi.load_state_dict(ck["hpsi"])
    hpsi.eval()
    for p in hpsi.parameters():
        p.requires_grad_(False)
    return bank, hpsi


def make_feat_fn(stats, hpsi, variant: str):
    """-> fn(z_leaf (B,4,64,64)) -> {"q": (B*64,84) std, "v": (B*64,Dv) bank-space}.

    Variant -> (space, view) via kde.VARIANTS:
      A -> (res, joint)   B -> (f, joint)   C -> (res, joint)
      B_color -> (f, color)   B_texture -> (f, texture)
    The q path is computed under no_grad (kde.py detaches it anyway); the v
    path keeps the autograd chain to z.
    """
    space, view, _cond = VARIANTS[variant]
    q_mean, q_std = stats["q_mean"], stats["q_std"]
    f_mean, f_std = stats["f_mean"], stats["f_std"]
    r_mean, r_std = stats["r_mean"], stats["r_std"]
    c_mean, c_std = stats["c_mean"], stats["c_std"]
    t_mean, t_std = stats["t_mean"], stats["t_std"]
    dev = q_mean.device

    def fn(z: torch.Tensor) -> dict:
        with torch.no_grad():
            p = patchify(z.detach())
            q = (q_feats(p) - q_mean) / q_std
        p = patchify(z)
        f = (f_feats(p) - f_mean) / f_std
        if space == "res":
            v = ((f - hpsi(q.to(dev))) - r_mean) / r_std
        elif view == "joint":
            v = f
        elif view == "color":
            v = (f[:, F_COLOR[0]:F_COLOR[1]] - c_mean) / c_std
        else:  # texture
            v = (f[:, F_TEXTURE[0]:F_TEXTURE[1]] - t_mean) / t_std
        return {"q": q.to(dev), "v": v}

    return fn


@torch.no_grad()
def holdout_latents(style_idx: int, n: int, seed: int = 0,
                    splits_path=None):
    """Deterministic holdout subset: (latents fp16 (m,4,64,64), ids, files)."""
    sp = torch.load(splits_path or (OUT / "splits.pt"), map_location="cpu",
                    weights_only=False)
    s = STYLES[style_idx]
    d = torch.load(POOL_DIR / f"{style_idx:02d}_{s}.pt",
                   map_location="cpu", weights_only=False)
    ho = sp[s]["holdout_idx"]
    g = torch.Generator().manual_seed(seed)
    take = ho[torch.randperm(ho.shape[0], generator=g)[:n]]
    take = torch.sort(take).values
    ids = [str(d["ids"][i]) for i in take.tolist()]
    files = [str(d["files"][i]) for i in take.tolist()]
    return d["latents"][take], ids, files


@torch.no_grad()
def pool_channel_stats(style_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    s = STYLES[style_idx]
    d = torch.load(POOL_DIR / f"{style_idx:02d}_{s}.pt",
                   map_location="cpu", weights_only=False)
    z = d["latents"].float()
    return z.mean(dim=(0, 2, 3)), z.std(dim=(0, 2, 3))


def local_image_path(rel_file: str) -> Path:
    """Pool 'files' are /mnt/i/... WSL paths; map to the local datasets copy."""
    return REPO / "datasets/wikiart_new_512" / Path(rel_file).parent.name / Path(rel_file).name
