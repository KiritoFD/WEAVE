"""V18 Gate 0: Strictly De-Styled Structural Representations & Leakage Audit.

Constructs S(x) = (C_64, P_16):
  1. C_64 (B, 4, 64, 64): Pure ITU-R BT.601 luminance-derived geometric maps
     [LCN_fine(k=5), LCN_coarse(k=13), SmoothSobel95, NormalizedLaplacian].
     100% of RGB chrominance / color-palette information is annihilated.
  2. P_16 (B, 384, 16, 16): Frozen DINOv2-small patch tokens extracted from
     3-channel GRAYSCALE [Y, Y, Y], standardized per-(image, channel) across
     the 16x16 grid (zero spatial mean, unit spatial std), then L2-normalized
     per spatial token. Global Gram/mean style signatures are identically zero.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from . import OUT, STYLES, save_json


def rgb_to_gray(rgb01: torch.Tensor) -> torch.Tensor:
    """rgb01: (B, 3, H, W) in [0, 1] -> luminance Y: (B, 1, H, W) in [0, 1]."""
    r, g, b = rgb01[:, 0:1], rgb01[:, 1:2], rgb01[:, 2:3]
    return (0.299 * r + 0.587 * g + 0.114 * b).clamp(0.0, 1.0)


def local_contrast_norm(y: torch.Tensor, kernel: int = 5, eps: float = 0.03) -> torch.Tensor:
    """Zero-mean, local-RMS-equalized structure map bounded in [-1, 1]."""
    pad = kernel // 2
    yp = F.pad(y, (pad, pad, pad, pad), mode="reflect")
    mean = F.avg_pool2d(yp, kernel, stride=1)
    diff = y - mean
    dp = F.pad(diff.square(), (pad, pad, pad, pad), mode="reflect")
    # Clamp BEFORE sqrt: saturated/flat regions (e.g. Minimalism or clamped z1_hat)
    # have dp == 0, whose backward d/dx sqrt(0) = +inf -> NaN.
    std = F.avg_pool2d(dp, kernel, stride=1).clamp_min(1e-8).sqrt()
    return (diff / (std + eps)).clamp(-3.0, 3.0) / 3.0


def _gaussian_blur_3x3(x: torch.Tensor) -> torch.Tensor:
    k = torch.tensor(
        [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3) / 16.0
    return F.conv2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), k)


def rgb_to_structure_64(rgb01: torch.Tensor) -> torch.Tensor:
    """Extract palette-free, stroke-amplitude-normalized 4-ch structure at 64x64."""
    y512 = rgb_to_gray(rgb01.float())
    # Pool to 128, smooth micro-impasto, then pool to 64
    y128 = F.avg_pool2d(y512, kernel_size=4, stride=4)
    y128_s = _gaussian_blur_3x3(y128)
    y64 = F.avg_pool2d(y128_s, kernel_size=2, stride=2)  # (B, 1, 64, 64)

    lcn_fine = local_contrast_norm(y64, kernel=5, eps=0.03)
    lcn_coarse = local_contrast_norm(y64, kernel=13, eps=0.04)

    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
                      device=y64.device, dtype=y64.dtype).view(1, 1, 3, 3)
    ky = kx.transpose(-1, -2).contiguous()
    yp = F.pad(y64, (1, 1, 1, 1), mode="reflect")
    gx = F.conv2d(yp, kx)
    gy = F.conv2d(yp, ky)
    mag = (gx.square() + gy.square() + 1e-8).sqrt()
    q95 = torch.quantile(mag.flatten(1), 0.95, dim=1).view(-1, 1, 1, 1).clamp_min(0.05)
    edge_norm = (mag / q95).clamp(0.0, 1.5) / 1.5 * 2.0 - 1.0  # [-1, 1]

    klap = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
                        device=y64.device, dtype=y64.dtype).view(1, 1, 3, 3)
    lap = F.conv2d(yp, klap)
    lap_rms = lap.flatten(1).square().mean(1).sqrt().view(-1, 1, 1, 1).clamp_min(0.02)
    lap_norm = (lap / (3.0 * lap_rms)).clamp(-1.0, 1.0)

    return torch.cat([lcn_fine, lcn_coarse, edge_norm, lap_norm], dim=1)


@torch.no_grad()
def extract_whitened_dino_layout(dino_model, rgb01: torch.Tensor, batch: int = 16, device: str = "cuda") -> torch.Tensor:
    """Extract grayscale-input, spatially-standardized DINOv2 patch layout (B, 384, 16, 16)."""
    from src.utils._local_eval_refs import encode_dino

    y = rgb_to_gray(rgb01.float())
    gray3 = y.repeat(1, 3, 1, 1)
    u8 = (gray3.cpu() * 255.0).round().clamp(0, 255).byte().permute(0, 2, 3, 1).numpy()
    imgs = [Image.fromarray(arr) for arr in u8]
    _, patch = encode_dino(dino_model, imgs, batch=batch, device=device)
    # patch: (B, 256, 384) -> (B, 384, 16, 16)
    p = patch.float().transpose(1, 2).reshape(len(imgs), 384, 16, 16)
    # Strictly strip per-(image, channel) global mean & std across the 16x16 spatial grid
    mean = p.mean(dim=(2, 3), keepdim=True)
    std = p.std(dim=(2, 3), keepdim=True, unbiased=False).clamp_min(1e-5)
    p_whiten = (p - mean) / std
    # Per-token L2 normalization across channels so amplitude carries no residual energy cue
    p_norm = p_whiten / p_whiten.norm(dim=1, keepdim=True).clamp_min(1e-6) * math.sqrt(384.0)
    # Re-center spatial mean to exact zero after token norm
    p_out = p_norm - p_norm.mean(dim=(2, 3), keepdim=True)
    return p_out


def augment_structure(c_64: torch.Tensor, gen: torch.Generator | None = None) -> torch.Tensor:
    """Training-time mild geometric/contrast jitter on C_64 to prevent micro-edge memorization."""
    B = c_64.shape[0]
    dev = c_64.device
    scale = (0.85 + 0.30 * torch.rand(B, 4, 1, 1, generator=gen).to(dev))
    out = c_64 * scale
    # With 40% probability, smooth C_64 slightly so fine brushstroke edges cannot be relied on
    blur_mask = (torch.rand(B, 1, 1, 1, generator=gen).to(dev) < 0.40)
    if blur_mask.any():
        blurred = torch.cat([_gaussian_blur_3x3(out[:, i:i+1]) for i in range(4)], dim=1)
        out = torch.where(blur_mask, blurred, out)
    noise = torch.randn(c_64.shape, generator=gen).to(dev) * 0.03
    return (out + noise).clamp(-1.2, 1.2)


def fit_latent_luma_proj(z_train: torch.Tensor, y64_train: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit linear projection (4,) + bias mapping 64x64 scaled latent z -> 64x64 luminance Y."""
    # Subsample pixels for fast exact least-squares
    X = z_train.permute(0, 2, 3, 1).reshape(-1, 4).float()
    Y = y64_train.permute(0, 2, 3, 1).reshape(-1, 1).float()
    step = max(1, X.shape[0] // 200_000)
    Xs = X[::step]
    Ys = Y[::step]
    ones = torch.ones(Xs.shape[0], 1)
    A = torch.cat([Xs, ones], dim=1)
    sol = torch.linalg.lstsq(A, Ys).solution.squeeze(1)
    return sol[:4], sol[4]


def project_latent_lcn(z: torch.Tensor, w_luma: torch.Tensor, b_luma: torch.Tensor) -> torch.Tensor:
    """Differentiable map from predicted clean latent z_hat -> LCN_fine(Y_64) for consistency loss."""
    y_hat = (z * w_luma.to(z.device).view(1, 4, 1, 1)).sum(dim=1, keepdim=True) + b_luma.to(z.device)
    return local_contrast_norm(y_hat.clamp(0.0, 1.0), kernel=5, eps=0.03)


def _train_linear_mlp_probe(X_tr: torch.Tensor, y_tr: torch.Tensor, X_ho: torch.Tensor, y_ho: torch.Tensor, epochs: int = 120) -> float:
    torch.manual_seed(42)
    d = X_tr.shape[1]
    net = nn.Sequential(
        nn.Linear(d, 128),
        nn.SiLU(),
        nn.Linear(128, 5),
    ).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-2)
    X_tr, y_tr = X_tr.cuda().float(), y_tr.cuda().long()
    X_ho, y_ho = X_ho.cuda().float(), y_ho.cuda().long()
    for _ in range(epochs):
        logits = net(X_tr)
        loss = F.cross_entropy(logits, y_tr)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = net(X_ho).argmax(dim=1)
        return float((pred == y_ho).float().mean().item())


def audit_leakage(cache_dict: dict) -> dict:
    """Gate 0: Quantify 5-way source-style leakage across representations on clean train->holdout."""
    from src.model import canonicalizer
    from tools.v6.canonicalize import extract_band_stats_features

    tr, ho = cache_dict["train"], cache_dict["holdout"]
    canon = canonicalizer("cuda")

    with torch.no_grad():
        f_raw_tr = extract_band_stats_features(tr["z"].cuda()).cpu()
        f_raw_ho = extract_band_stats_features(ho["z"].cuda()).cpu()

        c1_tr = canon(tr["z"].cuda(), mode="c1_gentle", augment=False)
        c1_ho = canon(ho["z"].cuda(), mode="c1_gentle", augment=False)
        f_c1_tr = extract_band_stats_features(c1_tr).cpu()
        f_c1_ho = extract_band_stats_features(c1_ho).cpu()

        f_c64_tr = extract_band_stats_features(tr["c_64"].float().cuda()).cpu()
        f_c64_ho = extract_band_stats_features(ho["c_64"].float().cuda()).cpu()

        # Global mean of whitened DINO tokens (mathematically ~0)
        f_dino_mean_tr = tr["dino_16"].float().mean(dim=(2, 3))
        f_dino_mean_ho = ho["dino_16"].float().mean(dim=(2, 3))

    del canon
    torch.cuda.empty_cache()

    acc_raw = _train_linear_mlp_probe(f_raw_tr, tr["sid"], f_raw_ho, ho["sid"])
    acc_c1 = _train_linear_mlp_probe(f_c1_tr, tr["sid"], f_c1_ho, ho["sid"])
    acc_c64 = _train_linear_mlp_probe(f_c64_tr, tr["sid"], f_c64_ho, ho["sid"])
    acc_dino_mean = _train_linear_mlp_probe(f_dino_mean_tr, tr["sid"], f_dino_mean_ho, ho["sid"])

    # Spatial layout correlation between C_64[0] (LCN_fine) and true luminance LCN
    lcn_true = local_contrast_norm(ho["y_64"].float(), kernel=5, eps=0.03)
    layout_corr = float(
        F.cosine_similarity(
            ho["c_64"][:, 0:1].float().flatten(1),
            lcn_true.flatten(1),
            dim=1,
        ).mean().item()
    )

    report = {
        "n_train": int(len(tr["sid"])),
        "n_holdout": int(len(ho["sid"])),
        "chance_level": 0.20,
        "style_probe_acc_5way": {
            "raw_latent_stats": round(acc_raw, 4),
            "v6_c1_gentle_stats": round(acc_c1, 4),
            "v18_c64_struct_stats": round(acc_c64, 4),
            "v18_whitened_dino_global_mean": round(acc_dino_mean, 4),
        },
        "dino_global_mean_max_abs": float(f_dino_mean_ho.abs().max().item()),
        "c64_layout_cosine_preserve": round(layout_corr, 4),
    }
    save_json(OUT / "gate0_leakage_audit.json", report)
    print("[V18 Gate0] Leakage Audit:", report["style_probe_acc_5way"], flush=True)
    return report
