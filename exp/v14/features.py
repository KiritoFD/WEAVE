"""V14 feature extractors: unified interface for the scale-scan experiment.

Each extractor takes an RGB tensor (3,512,512) or latent (4,64,64) and returns
a (N_cells, D) feature matrix where N_cells is the 8x8 grid (64 patches).
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as Fn
from PIL import Image

import numpy as np

REPO = Path(__file__).resolve().parents[2]
STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
POOL = REPO / "weights_local/sd_pool_orig"
IMG_ROOT = REPO / "datasets/wikiart_new_512"
S10_PATH = REPO / "deliver/checkpoints/runs/vae_decoder_prune/s10_p5_lowlr/decoder.pt"


def _h10(pid):
    import hashlib
    return int(hashlib.md5(pid.encode()).hexdigest()[:8], 16) % 10


def load_paintings(n_train=300, n_hold=150, seed=0):
    """-> {"train": [...], "holdout": [...]} each item = (style_idx, img_path, latent_idx)."""
    g = torch.Generator().manual_seed(seed)
    out = {"train": [], "holdout": []}
    for si, s in enumerate(STYLES):
        d = torch.load(POOL / f"{si:02d}_{s}.pt", map_location="cpu",
                       weights_only=False)
        tr, ho = [], []
        for i, pid in enumerate(d["ids"]):
            img = IMG_ROOT / s / Path(str(d["files"][i])).name
            if not img.exists():
                continue
            item = (si, str(img), i)
            (tr if _h10(str(pid)) < 9 else ho).append(item)
        take_tr = torch.randperm(len(tr), generator=g)[:n_train].tolist()
        take_ho = torch.randperm(len(ho), generator=g)[:n_hold].tolist()
        out["train"].extend([tr[i] for i in take_tr])
        out["holdout"].extend([ho[i] for i in take_ho])
        out.setdefault("latents", {})[s] = d["latents"]
    return out


def load_image(path, size=512):
    img = Image.open(path).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.BICUBIC)
    return torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255.0


# ============================ S0: V10 latent stats ============================
def s0_latent_stats(z: torch.Tensor) -> torch.Tensor:
    """(1,4,64,64) -> (64,34) V10 handcrafted features."""
    from exp.v10.features import f_feats, patchify
    return f_feats(patchify(z))


# ============================ S1: DWT tokens 8x8 =============================
def s1_dwt_tokens(z: torch.Tensor) -> torch.Tensor:
    """(1,4,64,64) -> (64,13) DWT tokens (H2_ds 9 + LL2_ds 3 + edge 1)."""
    d = _dwt_levels(z)
    h2 = Fn.adaptive_avg_pool2d(d["H2"], 8)                    # (1,9,8,8)
    ll2 = Fn.adaptive_avg_pool2d(d["LL2"], 8)                  # (1,3,8,8)
    e = Fn.adaptive_avg_pool2d(sobel_mag(d["LL2"].mean(1, keepdim=True)), 8)
    x = torch.cat([h2, ll2, e], 1)                             # (1,13,8,8)
    return x[0].permute(1, 0).reshape(-1, 13)                   # (64,13)


def _dwt_levels(z):
    from exp.v11b.bands import dwt2, sobel_mag
    d1 = dwt2(z)
    d2 = dwt2(d1["LL2"])
    e = sobel_mag(d2["LL2"].mean(1, keepdim=True))
    return {"H2": d2["H2"], "LL2": d2["LL2"], "H1": d1["H1"], "LL1": d1["LL1"],
            "edge2": e}


# ============================ S2: DWT large crop =============================
def s2_dwt_large(rgb512: torch.Tensor) -> torch.Tensor:
    """(1,3,512,512) -> (64,13) from 128px-crop DWT (larger receptive field)."""
    # downsample to 128, DWT, then pool to 8x8
    x = Fn.interpolate(rgb512, size=(128, 128), mode="bilinear",
                       align_corners=False)
    d = _dwt_levels(x)
    h2 = Fn.adaptive_avg_pool2d(d["H2"], 8)
    ll2 = Fn.adaptive_avg_pool2d(d["LL2"], 8)
    e = Fn.adaptive_avg_pool2d(sobel_mag(d["LL2"].mean(1, keepdim=True)), 8)
    x2 = torch.cat([h2, ll2, e], 1)
    return x2[0].permute(1, 0).reshape(-1, 13)


# ====================== S3/S4/S5: RGB CNN at multiple scales =================
_CNN = None


def _get_cnn(device):
    global _CNN
    if _CNN is None:
        layers = []
        cin = 3
        for cout, stride in ((32, 1), (64, 2), (64, 1), (128, 2), (128, 1)):
            layers.append(torch.nn.Conv2d(cin, cout, 3, stride=stride,
                                          padding=1, bias=False))
            layers.append(torch.nn.SiLU(inplace=True))
            cin = cout
        _CNN = torch.nn.Sequential(*layers).to(device).eval()
        for p in _CNN.parameters():
            p.requires_grad_(False)
    return _CNN


def rgb_cnn_features(rgb: torch.Tensor, device="cuda") -> torch.Tensor:
    """(1,3,S,S) -> (64,128) 8x8-cell CNN descriptors."""
    m = _get_cnn(device)
    x = rgb if rgb.dim() == 4 else rgb.unsqueeze(0)
    feat = m(x)                                                # (1,128,S',S')
    return Fn.adaptive_avg_pool2d(feat, 8).flatten(2).squeeze(0).T  # (64,128)


def s3_rgb64(rgb512: torch.Tensor, device="cuda") -> torch.Tensor:
    x = Fn.interpolate(rgb512, size=(64, 64), mode="bilinear",
                       align_corners=False)
    return rgb_cnn_features(x, device)


def s4_rgb128(rgb512: torch.Tensor, device="cuda") -> torch.Tensor:
    x = Fn.interpolate(rgb512, size=(128, 128), mode="bilinear",
                       align_corners=False)
    return rgb_cnn_features(x, device)


def s5_rgb224(rgb512: torch.Tensor, device="cuda") -> torch.Tensor:
    x = Fn.interpolate(rgb512, size=(224, 224), mode="bilinear",
                       align_corners=False)
    return rgb_cnn_features(x, device)


# ============================ S6: frozen DINOv2 ==============================
_DINO = None


def _dino(device):
    global _DINO
    if _DINO is None:
        _DINO = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14",
                               verbose=False).to(device).eval()
        for p in _DINO.parameters():
            p.requires_grad_(False)
    return _DINO


def s6_dino_tokens(rgb512: torch.Tensor, device="cuda") -> torch.Tensor:
    """(1,3,512,512) -> (64,384) DINOv2 patch tokens (224 crop, 16x16, 2x2 pool)."""
    m = _dino(device)
    x = Fn.interpolate(rgb512, size=(224, 224), mode="bilinear",
                       align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    x = (x - mean) / std
    with torch.no_grad():
        tok = m(x, return_dict=True).last_hidden_state[:, 1:, :]  # (1,256,384)
        tok = tok.transpose(1, 2).reshape(1, -1, 16, 16)
    return Fn.adaptive_avg_pool2d(tok, 8).flatten(2).squeeze(0).T  # (64,384)


# ============================ S7: frozen CSD =================================
_CSD = None


def _csd(device):
    global _CSD
    if _CSD is None:
        import sys
        sys.path.insert(0, str(REPO))
        from tools.v12h.csd_model import load_csd
        _CSD = load_csd(device=device)
    return _CSD


def s7_csd_cls(rgb512: torch.Tensor, device="cuda") -> torch.Tensor:
    """(1,3,512,512) -> (1, D) CSD CLS feature (painting-level reference)."""
    return torch.zeros(1, 1, device=device)  # placeholder; CSD needs special loader


# ============================ S8: multi-scale ================================
def s8_multiscale(rgb512: torch.Tensor, z: torch.Tensor, device="cuda") -> torch.Tensor:
    """Full image context + local DWT tokens."""
    local = s1_dwt_tokens(z)                                    # (64,13)
    global_feat = s6_dino_tokens(rgb512, device)                # (64,384)
    return torch.cat([local, global_feat], 1)                   # (64,397)


# ============================ grayscale ======================================
def to_grayscale(rgb: torch.Tensor) -> torch.Tensor:
    w = torch.tensor([0.299, 0.587, 0.114], device=rgb.device).view(1, 3, 1, 1)
    return (rgb * w).sum(1, keepdim=True).repeat(1, 3, 1, 1)
