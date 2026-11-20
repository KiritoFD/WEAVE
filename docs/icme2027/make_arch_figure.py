#!/usr/bin/env python3
"""Build Best-Paper Oral grade ICME 2027 PRISM architecture diagram.

Unified single-framework visual narrative:
    x_hat = D_* o T o (Pi_perp, Pi_parallel)(x_s, x_ref)

Pure Python + NumPy + Pillow + Matplotlib (strictly zero torch imports).
Outputs:
    figures/fig3_prism_architecture.{pdf,png,svg}
"""
from __future__ import annotations

import math
from pathlib import Path
import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.path import Path as MplPath
from matplotlib.patches import (
    Circle,
    Ellipse,
    FancyArrowPatch,
    FancyBboxPatch,
    PathPatch,
    Polygon,
    Rectangle,
)

ICME_DIR = Path(__file__).resolve().parent
REPO = ICME_DIR.parents[1]
FIG_DIR = ICME_DIR / "figures"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})


# =============================================================================
# 1. Pure-NumPy / PIL Real Artwork & Feature-Map Computations
# =============================================================================

def load_center_square(path: Path, size: int = 256, bias_x: float = 0.5, bias_y: float = 0.5) -> np.ndarray:
    """Load an image, take a smart aspect-preserving square crop, and resize."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    side = min(w, h)
    left = int(round((w - side) * np.clip(bias_x, 0.0, 1.0)))
    top = int(round((h - side) * np.clip(bias_y, 0.0, 1.0)))
    crop = img.crop((left, top, left + side, top + side)).resize((size, size), Image.Resampling.LANCZOS)
    return np.asarray(crop, dtype=np.uint8)


def compute_luminance_rgb(rgb_u8: np.ndarray) -> np.ndarray:
    """Exact hue-free perceptual luminance Y(x) = w_Y^T x rendered as crisp grayscale."""
    rgb = rgb_u8.astype(np.float32) / 255.0
    y = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    p2, p98 = np.percentile(y, [2.0, 98.0])
    yn = np.clip((y - p2) / max(p98 - p2, 1e-5), 0.0, 1.0)
    u8 = (yn * 255.0).astype(np.uint8)
    return np.stack([u8, u8, u8], axis=-1)


def _box_mean(arr: np.ndarray, k: int) -> np.ndarray:
    r = k // 2
    pad = np.pad(arr, r, mode="reflect")
    h, w = arr.shape
    acc = np.zeros_like(arr, dtype=np.float32)
    for di in range(k):
        for dj in range(k):
            acc += pad[di:di + h, dj:dj + w]
    return acc / float(k * k)


def compute_lcn_map(rgb_u8: np.ndarray, out_size: int = 192) -> np.ndarray:
    """Exact Eq. (2) multi-scale LCN_{5,13}(Y) + Sobel edge relief with crisp blueprint contrast."""
    rgb = rgb_u8.astype(np.float32) / 255.0
    y = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    y96 = np.asarray(
        Image.fromarray((y * 255.0).astype(np.uint8)).resize((96, 96), Image.Resampling.BILINEAR),
        dtype=np.float32,
    ) / 255.0

    d5 = y96 - _box_mean(y96, 5)
    s5 = np.sqrt(np.maximum(_box_mean(d5 * d5, 5), 1e-8)) + 0.020
    lcn5 = np.clip(d5 / s5, -3.0, 3.0) / 3.0

    d13 = y96 - _box_mean(y96, 13)
    s13 = np.sqrt(np.maximum(_box_mean(d13 * d13, 13), 1e-8)) + 0.030
    lcn13 = np.clip(d13 / s13, -3.0, 3.0) / 3.0

    py = np.pad(y96, 1, mode="reflect")
    gx = (py[1:-1, 2:] - py[1:-1, :-2]) * 0.5
    gy = (py[2:, 1:-1] - py[:-2, 1:-1]) * 0.5
    edge = np.sqrt(gx * gx + gy * gy)
    edge = np.clip(edge / max(float(np.percentile(edge, 95.0)), 1e-5), 0.0, 1.0)

    # High-contrast zero-DC structural relief (deep slate background + bright ice-cyan contours)
    val = np.clip(0.28 + 0.30 * np.abs(lcn5) + 0.18 * (lcn13 * 0.5 + 0.5) + 0.52 * edge, 0.0, 1.0)
    v_hi = np.asarray(
        Image.fromarray((val * 255.0).astype(np.uint8)).resize((out_size, out_size), Image.Resampling.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    r = np.clip(0.06 + 0.84 * (v_hi ** 1.35), 0.0, 1.0)
    g = np.clip(0.12 + 0.86 * (v_hi ** 1.10), 0.0, 1.0)
    b = np.clip(0.24 + 0.76 * (v_hi ** 0.85), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)


def compute_polar_zca_grid(rgb_u8: np.ndarray, grid: int = 16, cell_px: int = 14) -> tuple[np.ndarray, np.ndarray]:
    """Compute real 16x16 Polar-ZCA orthogonalized structural patch descriptors from grayscale Y(x)."""
    rgb = rgb_u8.astype(np.float32) / 255.0
    y = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    sz = grid * 16
    y_im = np.asarray(
        Image.fromarray((y * 255.0).astype(np.uint8)).resize((sz, sz), Image.Resampling.BILINEAR),
        dtype=np.float32,
    ) / 255.0

    hp = y_im - _box_mean(y_im, 15)
    py = np.pad(hp, 1, mode="reflect")
    gx = py[1:-1, 2:] - py[1:-1, :-2]
    gy = py[2:, 1:-1] - py[:-2, 1:-1]
    gd1 = py[2:, 2:] - py[:-2, :-2]
    gd2 = py[2:, :-2] - py[:-2, 2:]
    lap = py[:-2, 1:-1] + py[2:, 1:-1] + py[1:-1, :-2] + py[1:-1, 2:] - 4.0 * hp

    feats = []
    for gi in range(grid):
        for gj in range(grid):
            s0, s1 = gi * 16, (gi + 1) * 16
            t0, t1 = gj * 16, (gj + 1) * 16
            p_hp = hp[s0:s1, t0:t1]
            p_gx = gx[s0:s1, t0:t1]
            p_gy = gy[s0:s1, t0:t1]
            p_d1 = gd1[s0:s1, t0:t1]
            p_d2 = gd2[s0:s1, t0:t1]
            p_lp = lap[s0:s1, t0:t1]
            feats.append([
                float(np.mean(p_hp[:8, :8])),
                float(np.mean(p_hp[:8, 8:])),
                float(np.mean(p_hp[8:, :8])),
                float(np.mean(p_hp[8:, 8:])),
                float(np.std(p_hp)),
                float(np.mean(np.abs(p_gx))),
                float(np.mean(np.abs(p_gy))),
                float(np.mean(np.abs(p_d1))),
                float(np.mean(np.abs(p_d2))),
                float(np.mean(np.abs(p_lp))),
                float(np.mean(p_gx * p_gy)),
                float(np.percentile(p_hp, 85.0) - np.percentile(p_hp, 15.0)),
            ])

    P = np.asarray(feats, dtype=np.float64).T  # (D, 256)
    P_hat = (P - P.mean(axis=1, keepdims=True)) / (P.std(axis=1, keepdims=True) + 1e-6)
    cov = (P_hat @ P_hat.T) / P_hat.shape[1] + 1e-4 * np.eye(P_hat.shape[0])
    eigvals, eigvecs = np.linalg.eigh(cov)
    inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(np.maximum(eigvals, 1e-6))) @ eigvecs.T
    P_zca = inv_sqrt @ P_hat
    norms = np.linalg.norm(P_zca, axis=0, keepdims=True) + 1e-8
    P_sph = math.sqrt(P_zca.shape[0]) * (P_zca / norms)
    P_tilde = P_sph - P_sph.mean(axis=1, keepdims=True)

    _, s, vt = np.linalg.svd(P_tilde, full_matrices=False)
    coords = (vt[:3, :] * s[:3, None]).T
    for c in range(3):
        if np.mean(coords[:, c] * np.linspace(-1.0, 1.0, coords.shape[0])) < 0:
            coords[:, c] *= -1.0
        lo, hi = np.percentile(coords[:, c], [5.0, 95.0])
        coords[:, c] = np.clip((coords[:, c] - lo) / max(hi - lo, 1e-6), 0.0, 1.0)

    c0, c1, c2 = coords[:, 0], coords[:, 1], coords[:, 2]
    r_ch = 0.14 + 0.68 * c0 + 0.14 * c2
    g_ch = 0.20 + 0.62 * c1 + 0.16 * (1.0 - c0)
    b_ch = 0.36 + 0.56 * (1.0 - 0.55 * c0) + 0.08 * c2
    patch_rgb = np.clip(np.stack([r_ch, g_ch, b_ch], axis=-1), 0.0, 1.0).reshape(grid, grid, 3)

    out_px = grid * cell_px
    mosaic = np.zeros((out_px, out_px, 3), dtype=np.uint8)
    for gi in range(grid):
        for gj in range(grid):
            col = (patch_rgb[gi, gj] * 255.0).astype(np.uint8)
            mosaic[gi * cell_px:(gi + 1) * cell_px, gj * cell_px:(gj + 1) * cell_px] = col

    for k in range(1, grid):
        idx = k * cell_px
        mosaic[idx:idx + 1, :, :] = (mosaic[idx:idx + 1, :, :].astype(np.uint16) * 2 + 735) // 5
        mosaic[:, idx:idx + 1, :] = (mosaic[:, idx:idx + 1, :].astype(np.uint16) * 2 + 735) // 5

    return mosaic, P_tilde.astype(np.float32)


def srgb_to_oklab(rgb01: np.ndarray) -> np.ndarray:
    """Exact Bjorn Ottosson sRGB [0,1] -> Oklab (L, a, b)."""
    lin = np.where(rgb01 <= 0.04045, rgb01 / 12.92, ((rgb01 + 0.055) / 1.055) ** 2.4)
    r, g, b = lin[..., 0], lin[..., 1], lin[..., 2]
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l_ = np.cbrt(np.maximum(l, 1e-10))
    m_ = np.cbrt(np.maximum(m, 1e-10))
    s_ = np.cbrt(np.maximum(s, 1e-10))
    L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    b_out = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    return np.stack([L, a, b_out], axis=-1)


def extract_oklab_swatches(rgb_u8: np.ndarray, k: int = 6) -> list[tuple[float, float, float]]:
    """Deterministic farthest-point k-means in Oklab space to extract real artwork palette swatches."""
    small = np.asarray(
        Image.fromarray(rgb_u8).resize((48, 48), Image.Resampling.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    flat_rgb = small.reshape(-1, 3)
    flat_lab = srgb_to_oklab(flat_rgb)

    chroma = np.sqrt(flat_lab[:, 1] ** 2 + flat_lab[:, 2] ** 2)
    centers_idx = [int(np.argmax(chroma))]
    for _ in range(1, k):
        d2 = np.min(
            [np.sum((flat_lab - flat_lab[idx]) ** 2, axis=1) for idx in centers_idx],
            axis=0,
        )
        centers_idx.append(int(np.argmax(d2)))

    centers = flat_lab[centers_idx].copy()
    for _ in range(14):
        dists = np.sum((flat_lab[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels = np.argmin(dists, axis=1)
        for ci in range(k):
            mask = labels == ci
            if np.any(mask):
                centers[ci] = flat_lab[mask].mean(axis=0)

    dists = np.sum((flat_lab[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    labels = np.argmin(dists, axis=1)
    swatches = []
    for ci in range(k):
        mask = labels == ci
        rgb_mean = flat_rgb[mask].mean(axis=0) if np.any(mask) else flat_rgb[centers_idx[ci]]
        swatches.append((float(centers[ci][0]), tuple(np.clip(rgb_mean, 0.0, 1.0))))
    swatches.sort(key=lambda item: item[0])
    return [s[1] for s in swatches]


def compute_luma_chroma_maps(rgb_u8: np.ndarray, sz: int = 64) -> tuple[np.ndarray, list[np.ndarray]]:
    """Compute 1-D structural luminance map P_Y and 3-D chrominance maps P_C on 64x64 grid."""
    small = np.asarray(
        Image.fromarray(rgb_u8).resize((sz, sz), Image.Resampling.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    lab = srgb_to_oklab(small)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]

    Ln = np.clip((L - np.percentile(L, 3)) / max(np.percentile(L, 97) - np.percentile(L, 3), 1e-5), 0.0, 1.0)
    py_rgb = np.stack([
        0.10 + 0.82 * Ln,
        0.19 + 0.77 * Ln,
        0.31 + 0.67 * Ln,
    ], axis=-1)

    an = np.clip((a - np.percentile(a, 5)) / max(np.percentile(a, 95) - np.percentile(a, 5), 1e-5), 0.0, 1.0)
    bn = np.clip((b - np.percentile(b, 5)) / max(np.percentile(b, 95) - np.percentile(b, 5), 1e-5), 0.0, 1.0)
    ch = np.sqrt(a * a + b * b)
    cn = np.clip((ch - np.percentile(ch, 5)) / max(np.percentile(ch, 95) - np.percentile(ch, 5), 1e-5), 0.0, 1.0)

    pc1 = np.stack([0.22 + 0.75 * an, 0.58 - 0.34 * an, 0.66 - 0.44 * an], axis=-1)
    pc2 = np.stack([0.20 + 0.76 * bn, 0.28 + 0.58 * bn, 0.76 - 0.56 * bn], axis=-1)
    pc3 = np.stack([0.28 + 0.70 * cn, 0.22 + 0.56 * (cn ** 1.1), 0.44 + 0.32 * (1.0 - cn)], axis=-1)

    return (
        (np.clip(py_rgb, 0.0, 1.0) * 255.0).astype(np.uint8),
        [(np.clip(m, 0.0, 1.0) * 255.0).astype(np.uint8) for m in (pc1, pc2, pc3)],
    )


def compute_attention_heatmap(zca_s: np.ndarray, zca_r: np.ndarray, n: int = 12) -> np.ndarray:
    """Compute real Quotient-Keyed cosine similarity attention matrix softmax(Q K^T / tau)."""
    idx = np.linspace(0, zca_s.shape[1] - 1, n).astype(int)
    q = zca_s[:, idx].T
    k = zca_r[:, idx].T
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-6)
    k = k / (np.linalg.norm(k, axis=1, keepdims=True) + 1e-6)
    logits = (q @ k.T) / 0.26
    logits = logits - logits.max(axis=1, keepdims=True)
    att = np.exp(logits)
    return att / (att.sum(axis=1, keepdims=True) + 1e-8)


# =============================================================================
# 2. Vector Drawing Primitives (Illustrator-Grade Matplotlib Helpers)
# =============================================================================

def _hex_to_rgb(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def _tint(hex_col: str, factor: float) -> tuple[float, float, float]:
    r, g, b = _hex_to_rgb(hex_col)
    if factor >= 1.0:
        t = min(factor - 1.0, 1.0)
        return (r + (1.0 - r) * t, g + (1.0 - g) * t, b + (1.0 - b) * t)
    return (r * factor, g * factor, b * factor)


def card(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    fc: str = "#FFFFFF",
    ec: str = "#334155",
    lw: float = 1.35,
    rad: float = 0.08,
    ls: str = "-",
    shadow: bool = False,
    zorder: int = 2,
):
    if shadow:
        ax.add_patch(
            FancyBboxPatch(
                (x + 0.024, y - 0.026),
                w,
                h,
                boxstyle=f"round,pad=0.0,rounding_size={rad}",
                facecolor="#0F172A",
                edgecolor="none",
                alpha=0.075,
                zorder=zorder - 1,
            )
        )
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.0,rounding_size={rad}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        linestyle=ls,
        zorder=zorder,
    )
    ax.add_patch(p)
    return p


def stage_container(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    *,
    body_fc: str,
    border_col: str,
    header_fc: str,
    header_h: float = 0.26,
    rad: float = 0.10,
    title_fs: float = 8.6,
    zorder: int = 2,
):
    """Draw a full-height operator pillar with a clean docked header banner."""
    card(ax, x, y, w, h, fc=body_fc, ec=border_col, lw=1.5, rad=rad, shadow=True, zorder=zorder)
    hdr = FancyBboxPatch(
        (x + 0.01, y + h - header_h),
        w - 0.02,
        header_h - 0.005,
        boxstyle=f"round,pad=0.0,rounding_size={rad * 0.85}",
        facecolor=header_fc,
        edgecolor="none",
        zorder=zorder + 1,
    )
    ax.add_patch(hdr)
    ax.add_patch(
        Rectangle(
            (x + 0.01, y + h - header_h),
            w - 0.02,
            header_h * 0.48,
            facecolor=header_fc,
            edgecolor="none",
            zorder=zorder + 1,
        )
    )
    ax.plot(
        [x + 0.01, x + w - 0.01],
        [y + h - header_h, y + h - header_h],
        color=border_col,
        lw=1.0,
        zorder=zorder + 2,
    )
    ax.text(
        x + w / 2.0,
        y + h - header_h / 2.0 - 0.004,
        title,
        ha="center",
        va="center",
        fontsize=title_fs,
        weight="bold",
        color="#FFFFFF",
        zorder=zorder + 3,
    )


def framed_image(
    ax,
    img_arr: np.ndarray,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    ec: str = "#1E293B",
    lw: float = 1.25,
    shadow: bool = True,
    interp: str = "bilinear",
    zorder: int = 6,
):
    if shadow:
        ax.add_patch(
            Rectangle(
                (x + 0.024, y - 0.024),
                w,
                h,
                facecolor="#0F172A",
                edgecolor="none",
                alpha=0.12,
                zorder=zorder - 1,
            )
        )
    ax.imshow(img_arr, extent=(x, x + w, y, y + h), aspect="auto", interpolation=interp, zorder=zorder)
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=ec, linewidth=lw, zorder=zorder + 1))


def draw_3d_slab(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    dx: float = 0.08,
    dy: float = 0.06,
    base_col: str = "#60A5FA",
    ec: str = "#1E3A8A",
    lw: float = 0.95,
    face_img: np.ndarray | None = None,
    stripes: int = 0,
    alpha: float = 1.0,
    ls: str = "-",
    zorder: int = 6,
):
    """Draw an oblique 3D tensor slab (front face + lighter top face + darker right face)."""
    top_pts = [(x, y + h), (x + dx, y + h + dy), (x + w + dx, y + h + dy), (x + w, y + h)]
    side_pts = [(x + w, y), (x + w + dx, y + dy), (x + w + dx, y + h + dy), (x + w, y + h)]

    ax.add_patch(
        Polygon(
            top_pts,
            closed=True,
            facecolor=_tint(base_col, 1.34),
            edgecolor=ec,
            linewidth=lw * 0.88,
            linestyle=ls,
            alpha=alpha,
            zorder=zorder,
        )
    )
    ax.add_patch(
        Polygon(
            side_pts,
            closed=True,
            facecolor=_tint(base_col, 0.78),
            edgecolor=ec,
            linewidth=lw * 0.88,
            linestyle=ls,
            alpha=alpha,
            zorder=zorder,
        )
    )
    if face_img is not None:
        ax.imshow(
            face_img,
            extent=(x, x + w, y, y + h),
            aspect="auto",
            interpolation="bilinear",
            zorder=zorder + 1,
        )
        ax.add_patch(
            Rectangle((x, y), w, h, fill=False, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=zorder + 2)
        )
    else:
        ax.add_patch(
            Rectangle(
                (x, y),
                w,
                h,
                facecolor=base_col,
                edgecolor=ec,
                linewidth=lw,
                linestyle=ls,
                alpha=alpha,
                zorder=zorder + 1,
            )
        )
        if stripes > 1:
            for si in range(1, stripes):
                sy = y + h * (si / float(stripes))
                ax.plot([x + 0.01, x + w - 0.01], [sy, sy], color=_tint(ec, 1.25), lw=0.55, alpha=0.65, zorder=zorder + 2)


def draw_multislice_block(
    ax,
    x: float,
    y_mid: float,
    slice_w: float,
    h: float,
    n_slices: int,
    *,
    step_x: float = 0.055,
    base_col: str = "#60A5FA",
    ec: str = "#1E3A8A",
    zorder: int = 6,
):
    """Draw a stacked multi-slice 3D U-Net stage block."""
    for si in range(n_slices):
        col = _tint(base_col, 1.0 + 0.10 * (n_slices - 1 - si))
        hex_c = "#{:02x}{:02x}{:02x}".format(
            int(np.clip(col[0] * 255, 0, 255)),
            int(np.clip(col[1] * 255, 0, 255)),
            int(np.clip(col[2] * 255, 0, 255)),
        )
        draw_3d_slab(
            ax,
            x + si * step_x,
            y_mid - h / 2.0,
            slice_w,
            h,
            dx=0.045,
            dy=0.038,
            base_col=hex_c,
            ec=ec,
            lw=0.85,
            zorder=zorder + si * 2,
        )


def rounded_ortho_arrow(
    ax,
    pts: list[tuple[float, float]],
    *,
    col: str = "#1E293B",
    lw: float = 1.4,
    ls: str = "-",
    rad: float = 0.075,
    hs: float = 9.0,
    halo: bool = False,
    zorder: int = 12,
):
    """Draw an orthogonal connector with smooth Bezier rounded bends and crisp arrowhead."""
    if len(pts) == 2:
        if halo:
            ax.plot([pts[0][0], pts[1][0]], [pts[0][1], pts[1][1]], color="white", lw=lw + 2.4, zorder=zorder - 1)
        arr = FancyArrowPatch(
            pts[0],
            pts[1],
            arrowstyle=f"-|>,head_length={hs * 0.44},head_width={hs * 0.29}",
            connectionstyle="arc3,rad=0.0",
            color=col,
            linewidth=lw,
            linestyle=ls,
            zorder=zorder,
        )
        ax.add_patch(arr)
        return

    p_prev = np.array(pts[-2], dtype=float)
    p_end = np.array(pts[-1], dtype=float)
    seg_vec = p_end - p_prev
    seg_len = float(np.linalg.norm(seg_vec))
    trim = min(0.085, seg_len * 0.35)
    p_body_end = p_end - (seg_vec / max(seg_len, 1e-8)) * trim

    verts: list[tuple[float, float]] = [pts[0]]
    codes: list[int] = [MplPath.MOVETO]

    work_pts = list(pts[:-1]) + [tuple(p_body_end)]
    for i in range(1, len(work_pts) - 1):
        p0 = np.array(work_pts[i - 1], dtype=float)
        p1 = np.array(work_pts[i], dtype=float)
        p2 = np.array(work_pts[i + 1], dtype=float)
        v1 = p1 - p0
        v2 = p2 - p1
        l1 = float(np.linalg.norm(v1))
        l2 = float(np.linalg.norm(v2))
        r = min(rad, l1 * 0.45, l2 * 0.45)
        if r < 1e-4:
            verts.append(tuple(p1))
            codes.append(MplPath.LINETO)
        else:
            p_in = p1 - (v1 / l1) * r
            p_out = p1 + (v2 / l2) * r
            verts.append(tuple(p_in))
            codes.append(MplPath.LINETO)
            verts.extend([tuple(p1), tuple(p_out)])
            codes.extend([MplPath.CURVE3, MplPath.CURVE3])

    verts.append(work_pts[-1])
    codes.append(MplPath.LINETO)
    path = MplPath(verts, codes)

    if halo:
        ax.add_patch(
            PathPatch(path, fill=False, edgecolor="white", linewidth=lw + 2.6, capstyle="round", zorder=zorder - 1)
        )
    ax.add_patch(
        PathPatch(
            path,
            fill=False,
            edgecolor=col,
            linewidth=lw,
            linestyle=ls,
            capstyle="round",
            zorder=zorder,
        )
    )
    head_start = tuple(p_end - (seg_vec / max(seg_len, 1e-8)) * max(trim * 1.35, 0.05))
    arr = FancyArrowPatch(
        head_start,
        tuple(p_end),
        arrowstyle=f"-|>,head_length={hs * 0.44},head_width={hs * 0.29}",
        connectionstyle="arc3,rad=0.0",
        color=col,
        linewidth=lw,
        linestyle="-",
        zorder=zorder + 1,
    )
    ax.add_patch(arr)


def draw_sphere_manifold(ax, cx: float, cy: float, r: float, zorder: int = 6):
    """Draw a shaded 3D hypersphere S^127 with wireframe, core clusters, u_s, u_ref, and universal chord e."""
    n = 128
    yy, xx = np.mgrid[-1.0:1.0:complex(0, n), -1.0:1.0:complex(0, n)]
    rr2 = xx * xx + yy * yy
    mask = rr2 <= 1.0
    zz = np.sqrt(np.maximum(1.0 - rr2, 0.0))
    light = np.clip(-0.42 * xx + 0.48 * yy + 0.77 * zz, 0.0, 1.0)
    rgba = np.zeros((n, n, 4), dtype=np.float32)
    rgba[..., 0] = 0.89 + 0.11 * light
    rgba[..., 1] = 0.79 + 0.19 * light
    rgba[..., 2] = 0.63 + 0.33 * light
    rgba[..., 3] = np.where(mask, 1.0, 0.0)
    ax.imshow(rgba, extent=(cx - r, cx + r, cy - r, cy + r), aspect="auto", interpolation="bilinear", zorder=zorder)

    ax.add_patch(
        Ellipse(
            (cx, cy),
            2.0 * r,
            0.55 * r,
            fill=False,
            edgecolor="#B45309",
            linewidth=0.7,
            linestyle="--",
            alpha=0.55,
            zorder=zorder + 1,
        )
    )
    ax.add_patch(
        Ellipse(
            (cx, cy),
            0.60 * r,
            2.0 * r,
            fill=False,
            edgecolor="#B45309",
            linewidth=0.7,
            linestyle="--",
            alpha=0.45,
            zorder=zorder + 1,
        )
    )
    ax.add_patch(Circle((cx, cy), r, fill=False, edgecolor="#92400E", linewidth=1.15, zorder=zorder + 2))

    # Intra-train k-NN core cluster dots around Ukiyo-e target region
    rng = np.random.default_rng(7)
    u_ref = (cx + 0.44 * r, cy + 0.42 * r)
    for _ in range(7):
        dx_c, dy_c = rng.normal(0.0, 0.08 * r, size=2)
        ax.add_patch(
            Circle(
                (u_ref[0] + dx_c, u_ref[1] + dy_c),
                0.032 * r,
                facecolor="#FB7185",
                edgecolor="none",
                alpha=0.70,
                zorder=zorder + 2,
            )
        )

    u_s = (cx - 0.46 * r, cy - 0.34 * r)
    ax.add_patch(Circle(u_s, 0.095 * r, facecolor="#2563EB", edgecolor="#FFFFFF", linewidth=0.9, zorder=zorder + 4))
    ax.add_patch(Circle(u_ref, 0.105 * r, facecolor="#BE123C", edgecolor="#FFFFFF", linewidth=0.9, zorder=zorder + 4))

    arr = FancyArrowPatch(
        u_s,
        u_ref,
        arrowstyle="-|>,head_length=4.0,head_width=2.7",
        connectionstyle="arc3,rad=0.0",
        color="#BE123C",
        linewidth=1.7,
        zorder=zorder + 5,
    )
    ax.add_patch(arr)
    stroke = [pe.withStroke(linewidth=2.2, foreground="#FFFDF7")]
    ax.text(u_s[0] - 0.04, u_s[1] - 0.06, r"$u_s$", ha="right", va="top", fontsize=7.5, weight="bold", color="#1E3A8A", path_effects=stroke, zorder=zorder + 6)
    ax.text(u_ref[0] + 0.03, u_ref[1] + 0.05, r"$u_{\mathrm{ref}}$", ha="left", va="bottom", fontsize=7.5, weight="bold", color="#9F1239", path_effects=stroke, zorder=zorder + 6)
    ax.text(cx - 0.05, cy + 0.09, r"$\mathbf{e}$", ha="center", va="center", fontsize=8.5, weight="bold", color="#9F1239", path_effects=stroke, zorder=zorder + 6)


# =============================================================================
# 3. Main Architecture Assembly (Refinement Pass 2)
# =============================================================================

def build_architecture() -> None:
    src_path = REPO / "data/test/Early_Renaissance/Early_Renaissance__benozzo-gozzoli_madonna-and-child-1485.jpg"
    ref_path = REPO / "data/test/Ukiyo_e/Ukiyo_e__katsushika-hokusai_ono-shindon-in-the-suraga-province.jpg"
    out_path = (
        REPO
        / "docs/performance/images/PRISM_Flow_D5_750"
        / "Early_Renaissance__Early_Renaissance__benozzo-gozzoli_madonna-and-child-1485__Ukiyo_e.png"
    )

    src_img = load_center_square(src_path, size=256, bias_x=0.5, bias_y=0.45)
    ref_img = load_center_square(ref_path, size=256, bias_x=0.58, bias_y=0.50)
    out_img = load_center_square(out_path, size=256, bias_x=0.5, bias_y=0.45)

    # Real computed visual maps (100% pure NumPy/PIL)
    y_map = compute_luminance_rgb(src_img)
    lcn_map = compute_lcn_map(src_img, out_size=192)
    zca_src_img, zca_src_feat = compute_polar_zca_grid(src_img, grid=16, cell_px=14)
    zca_ref_img, zca_ref_feat = compute_polar_zca_grid(ref_img, grid=16, cell_px=10)

    src_swatches = extract_oklab_swatches(src_img, k=6)
    ref_swatches = extract_oklab_swatches(ref_img, k=6)
    py_map, pc_maps = compute_luma_chroma_maps(src_img, sz=64)
    _, out_pc_maps = compute_luma_chroma_maps(out_img, sz=64)
    att_map = compute_attention_heatmap(zca_src_feat, zca_ref_feat, n=12)

    src_zoom = src_img[38:158, 68:188]
    out_zoom = out_img[38:158, 68:188]

    W, H = 14.30, 5.20
    fig = plt.figure(figsize=(W, H), facecolor="white")
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], facecolor="white")
    ax.set_xlim(0.0, W)
    ax.set_ylim(0.0, H)
    ax.axis("off")

    # =========================================================================
    # 0. TOP UNIFIED FACTORIZATION BANNER (Blocking Source-Style Backflow)
    # =========================================================================
    ax.plot([0.18, 2.55], [5.03, 5.03], color="#CBD5E1", lw=1.0, zorder=1)
    ax.plot([11.75, 14.12], [5.03, 5.03], color="#CBD5E1", lw=1.0, zorder=1)
    card(ax, 2.62, 4.91, 9.06, 0.24, fc="#F1F5F9", ec="#64748B", lw=1.0, rad=0.09, zorder=3)
    ax.text(
        7.15,
        5.03,
        r"Blocking Source-Style Backflow:   Source Texture Leakage ($92.3\%\!\to\!20.0\%$)  $\rightarrow$  Invariant Layout $\Pi_\perp$  $\rightarrow$  Target Style Chord $e(x_s\!\to\!\tau)$  $\rightarrow$  Controlled Transport $\mathcal{T}$",
        ha="center",
        va="center",
        fontsize=8.05,
        weight="bold",
        color="#0F172A",
        zorder=4,
    )

    # =========================================================================
    # 1. LEFT INPUT PAIR COLUMN (x_s, x_ref)
    # =========================================================================
    ax.text(0.73, 4.68, r"Source Content $x_s$", ha="center", va="bottom", fontsize=8.8, weight="bold", color="#0F172A")
    framed_image(ax, src_img, 0.17, 3.50, 1.12, 1.12, ec="#1E3A8A", lw=1.35, zorder=6)
    card(ax, 0.19, 3.25, 1.08, 0.19, fc="#EFF6FF", ec="#93C5FD", lw=0.85, rad=0.05, zorder=4)
    ax.text(0.73, 3.345, r"Unlabeled $\cdot\;512^2$", ha="center", va="center", fontsize=7.15, color="#1E40AF", weight="bold", zorder=5)

    ax.text(0.73, 2.46, r"Target Ref. $x_{\mathrm{ref}}$", ha="center", va="bottom", fontsize=8.8, weight="bold", color="#7C2D12")
    framed_image(ax, ref_img, 0.17, 1.28, 1.12, 1.12, ec="#B45309", lw=1.35, zorder=6)
    card(ax, 0.15, 1.02, 1.16, 0.20, fc="#FFFBEB", ec="#FCD34D", lw=0.85, rad=0.05, zorder=4)
    ax.text(0.73, 1.12, r"1-Shot (or proto. $\bar{u}_{y_t}$)", ha="center", va="center", fontsize=7.05, color="#92400E", weight="bold", zorder=5)

    # =========================================================================
    # 2. OPERATOR I PILLAR: ORTHOGONAL QUOTIENT-FIBER DUALITY (Pi_perp, Pi_parallel)
    # =========================================================================
    stage_container(
        ax,
        1.64,
        0.84,
        3.78,
        4.00,
        r"Operator I · Orthogonal Quotient–Fiber Duality $(\Pi_\perp, \Pi_\parallel)$",
        body_fc="#F8FAFC",
        border_col="#334155",
        header_fc="#1E3A5F",
        header_h=0.25,
        title_fs=8.55,
    )

    # ---- Upper Sub-Panel: Pi_perp (Cool Slate/Blue Theme) ----
    card(ax, 1.74, 2.97, 3.58, 1.56, fc="#EFF6FF", ec="#2563EB", lw=1.25, rad=0.08, zorder=3)
    ax.text(
        3.53,
        4.41,
        r"$\Pi_\perp$ · Invariant Structural Quotient (Hue & Moment Free)",
        ha="center",
        va="center",
        fontsize=7.65,
        weight="bold",
        color="#1E3A8A",
        zorder=5,
    )

    m_y = 3.51
    m_s = 0.74
    # Map 1: Grayscale Luminance Y(x_s)
    framed_image(ax, y_map, 1.85, m_y, m_s, m_s, ec="#334155", lw=1.0, zorder=6)
    ax.text(2.22, 3.37, r"Luminance $Y(x_s)$", ha="center", va="center", fontsize=7.0, weight="bold", color="#0F172A", zorder=6)

    rounded_ortho_arrow(ax, [(2.62, m_y + m_s / 2), (2.92, m_y + m_s / 2)], col="#1D4ED8", lw=1.2, hs=7.6)

    # Map 2: Multi-Scale Contrast LCN(Y)
    framed_image(ax, lcn_map, 2.95, m_y, m_s, m_s, ec="#1D4ED8", lw=1.0, zorder=6)
    ax.text(3.32, 3.37, r"Contrast $\mathrm{LCN}(Y)$", ha="center", va="center", fontsize=7.0, weight="bold", color="#1E3A8A", zorder=6)

    rounded_ortho_arrow(ax, [(3.72, m_y + m_s / 2), (4.02, m_y + m_s / 2)], col="#1D4ED8", lw=1.2, hs=7.6)

    # Map 3: Polar-ZCA 16x16 Semantic Layout Grid (+ back-stacked x_ref token grid)
    framed_image(ax, zca_ref_img, 4.18, m_y + 0.06, m_s * 0.90, m_s * 0.90, ec="#64748B", lw=0.8, shadow=False, interp="nearest", zorder=5)
    framed_image(ax, zca_src_img, 4.05, m_y, m_s, m_s, ec="#1E3A8A", lw=1.1, interp="nearest", zorder=7)
    card(ax, 4.89, m_y + 0.50, 0.34, 0.17, fc="#DBEAFE", ec="#3B82F6", lw=0.7, rad=0.04, zorder=8)
    ax.text(5.06, m_y + 0.585, r"$Q, K$", ha="center", va="center", fontsize=6.5, weight="bold", color="#1E40AF", zorder=9)
    ax.text(4.50, 3.37, r"Polar-ZCA $16{\times}16\;\tilde{P}_{16}^{\perp}$", ha="center", va="center", fontsize=7.0, weight="bold", color="#1E3A8A", zorder=6)

    # Leakage Audit Badge
    card(ax, 1.84, 3.03, 3.38, 0.22, fc="#DBEAFE", ec="#2563EB", lw=0.9, rad=0.05, zorder=5)
    ax.text(
        3.53,
        3.14,
        r"1st & 2nd-Order Style Leak: $18.3\%$ ($\sim\!20\%$ Chance) $\cdot$ $0.994$ Layout",
        ha="center",
        va="center",
        fontsize=7.05,
        weight="bold",
        color="#1E3A8A",
        zorder=6,
    )

    # ---- Middle Latent Pass-Through: Frozen VAE Encoder E(x_s) -> z_s ----
    vae_pts = [(1.88, 2.61), (2.94, 2.65), (2.94, 2.86), (1.88, 2.90)]
    ax.add_patch(Polygon(vae_pts, closed=True, facecolor="#E0F2FE", edgecolor="#0284C7", linewidth=1.15, zorder=5))
    ax.text(2.41, 2.755, r"Frozen VAE $\mathcal{E}$", ha="center", va="center", fontsize=7.15, weight="bold", color="#0C4A6E", zorder=6)

    card(ax, 3.16, 2.62, 2.06, 0.27, fc="#F0F9FF", ec="#0284C7", lw=0.95, rad=0.05, zorder=5)
    ax.text(4.19, 2.755, r"Latent $z_s \to$ Anchor $z_0\;(64^2{\times}4)$", ha="center", va="center", fontsize=7.05, weight="bold", color="#075985", zorder=6)
    rounded_ortho_arrow(ax, [(2.94, 2.755), (3.15, 2.755)], col="#0284C7", lw=1.2, hs=7.2)

    # ---- Lower Sub-Panel: Pi_parallel (Warm Amber/Coral Theme) ----
    card(ax, 1.74, 0.91, 3.58, 1.63, fc="#FFFBEB", ec="#D97706", lw=1.25, rad=0.08, zorder=3)
    ax.text(
        3.53,
        2.42,
        r"$\Pi_\parallel$ · Complementary Style Fiber & Universal Chord",
        ha="center",
        va="center",
        fontsize=7.65,
        weight="bold",
        color="#92400E",
        zorder=5,
    )

    # Left Sub-Card: Real Oklab Palette Swatch Bars + Moments
    card(ax, 1.83, 1.24, 1.60, 1.06, fc="#FFFFFF", ec="#F59E0B", lw=0.95, rad=0.06, zorder=4)
    ax.text(2.63, 2.18, r"Oklab Swatches $\oplus\;(\mu_{\mathrm{sp}},\sigma_{\mathrm{sp}})$", ha="center", va="center", fontsize=6.75, weight="bold", color="#78350F", zorder=5)

    ax.text(1.89, 1.94, r"$x_s$", ha="left", va="center", fontsize=7.0, weight="bold", color="#1E3A8A", zorder=6)
    sw_x0, sw_w, sw_h = 2.11, 0.155, 0.145
    for idx, col_rgb in enumerate(src_swatches):
        ax.add_patch(
            Rectangle(
                (sw_x0 + idx * sw_w, 1.87),
                sw_w - 0.015,
                sw_h,
                facecolor=col_rgb,
                edgecolor="#334155",
                linewidth=0.5,
                zorder=6,
            )
        )
    for bi, bh in enumerate([0.07, 0.13, 0.10]):
        ax.add_patch(Rectangle((3.09 + bi * 0.072, 1.87), 0.048, bh, facecolor="#60A5FA", edgecolor="none", zorder=6))

    ax.text(1.87, 1.64, r"$x_{\mathrm{ref}}$", ha="left", va="center", fontsize=6.85, weight="bold", color="#9F1239", zorder=6)
    for idx, col_rgb in enumerate(ref_swatches):
        ax.add_patch(
            Rectangle(
                (sw_x0 + idx * sw_w, 1.57),
                sw_w - 0.015,
                sw_h,
                facecolor=col_rgb,
                edgecolor="#334155",
                linewidth=0.5,
                zorder=6,
            )
        )
    for bi, bh in enumerate([0.14, 0.08, 0.12]):
        ax.add_patch(Rectangle((3.09 + bi * 0.072, 1.57), 0.048, bh, facecolor="#FB7185", edgecolor="none", zorder=6))

    ax.text(2.63, 1.36, r"Fiber Probe: $97.3\%$ Style Acc.", ha="center", va="center", fontsize=6.8, weight="bold", color="#B45309", zorder=5)

    rounded_ortho_arrow(ax, [(3.44, 1.77), (3.64, 1.77)], col="#B45309", lw=1.2, hs=7.4)
    ax.text(3.54, 1.87, r"$\phi_\eta$", ha="center", va="bottom", fontsize=7.4, weight="bold", color="#92400E", zorder=7)

    # Right Sub-Card: Spherical Style Manifold S^127 + Universal Chord e
    card(ax, 3.65, 1.24, 1.58, 1.06, fc="#FFFDF7", ec="#D97706", lw=0.95, rad=0.06, zorder=4)
    draw_sphere_manifold(ax, cx=4.06, cy=1.74, r=0.29, zorder=6)
    ax.text(4.86, 2.08, r"$\mathbb{S}^{127}$ Sphere", ha="center", va="center", fontsize=7.1, weight="bold", color="#78350F", zorder=6)
    ax.text(4.86, 1.81, "Universal\nChord", ha="center", va="center", fontsize=6.85, weight="bold", color="#9F1239", zorder=6)
    ax.text(4.86, 1.44, r"$e = u_{\mathrm{ref}} - u_s$", ha="center", va="center", fontsize=7.35, weight="bold", color="#0F172A", zorder=6)

    # Exact Self-Identity Badge
    card(ax, 1.84, 0.96, 3.38, 0.22, fc="#FEF3C7", ec="#D97706", lw=0.9, rad=0.05, zorder=5)
    ax.text(
        3.53,
        1.07,
        r"Zero Source-ID $\cdot$ Exact Identity: $x_{\mathrm{ref}}=x_s \Rightarrow e\equiv\mathbf{0}\;\;(99.3\%\;\mathrm{Same150})$",
        ha="center",
        va="center",
        fontsize=6.95,
        weight="bold",
        color="#92400E",
        zorder=6,
    )

    # Input-to-Operator-I routing (zero crossings)
    rounded_ortho_arrow(ax, [(1.29, 3.88), (1.83, 3.88)], col="#1D4ED8", lw=1.4)
    rounded_ortho_arrow(ax, [(1.29, 3.66), (1.47, 3.66), (1.47, 2.755), (1.87, 2.755)], col="#0284C7", lw=1.3)
    ax.add_patch(Circle((1.47, 2.755), 0.03, facecolor="#0284C7", edgecolor="white", linewidth=0.55, zorder=14))
    rounded_ortho_arrow(ax, [(1.47, 2.755), (1.47, 1.94), (1.82, 1.94)], col="#D97706", lw=1.25)
    rounded_ortho_arrow(ax, [(1.29, 1.64), (1.82, 1.64)], col="#B45309", lw=1.4)

    # =========================================================================
    # 3. OPERATOR II PILLAR: QUOTIENT-GUIDED LATENT TRANSPORT T (R^4 = P_Y (+) P_C)
    # =========================================================================
    stage_container(
        ax,
        5.62,
        0.84,
        5.18,
        4.00,
        r"Operator II · Quotient-Guided Latent Transport $\mathcal{T}\;\;(\mathbb{R}^4 = P_Y \oplus P_C)$",
        body_fc="#F8FAFC",
        border_col="#334155",
        header_fc="#1E293B",
        header_h=0.25,
        title_fs=8.65,
    )

    # ---- Module II-1: 3D-Staggered Luma-Chroma Split (64x64x4) ----
    card(ax, 5.75, 1.63, 1.16, 2.48, fc="#FFFFFF", ec="#CBD5E1", lw=1.0, rad=0.07, zorder=3)
    ax.text(6.33, 3.99, r"$64{\times}64{\times}4$ Split", ha="center", va="center", fontsize=7.35, weight="bold", color="#0F172A", zorder=5)

    # 1-D Structural Luminance P_Y 3D slab
    draw_3d_slab(
        ax,
        5.99,
        3.24,
        0.52,
        0.52,
        dx=0.09,
        dy=0.065,
        base_col="#38BDF8",
        ec="#0369A1",
        lw=1.0,
        face_img=py_map,
        zorder=6,
    )
    ax.text(6.33, 3.08, r"1-D Luma $P_Y$", ha="center", va="center", fontsize=7.05, weight="bold", color="#0369A1", zorder=6)
    ax.text(6.33, 2.93, r"(tight $\rho_Y$)", ha="center", va="center", fontsize=6.6, color="#475569", zorder=6)

    ax.add_patch(Circle((6.33, 2.75), 0.082, facecolor="#F1F5F9", edgecolor="#475569", linewidth=0.85, zorder=6))
    ax.text(6.33, 2.75, r"$\oplus$", ha="center", va="center", fontsize=8.0, weight="bold", color="#1E293B", zorder=7)

    # 3-D Chrominance Subspace P_C (3 staggered 3D slabs)
    chroma_cols = ["#FB7185", "#FBBF24", "#FB923C"]
    chroma_ecs = ["#9F1239", "#92400E", "#C2410C"]
    for idx in (2, 1, 0):
        ox = 5.90 + idx * 0.105
        oy = 1.98 + idx * 0.08
        draw_3d_slab(
            ax,
            ox,
            oy,
            0.46,
            0.46,
            dx=0.065,
            dy=0.05,
            base_col=chroma_cols[idx],
            ec=chroma_ecs[idx],
            lw=0.9,
            face_img=pc_maps[idx],
            zorder=6 + (2 - idx) * 2,
        )
    ax.text(6.33, 1.83, r"3-D Chroma $P_C$", ha="center", va="center", fontsize=7.05, weight="bold", color="#C2410C", zorder=6)
    ax.text(6.33, 1.69, r"(wide $\rho_C \gg \rho_Y$)", ha="center", va="center", fontsize=6.6, color="#9A3412", zorder=6)

    # ---- Module II-2: Multi-Slice 3D U-Net / 16x16 Semantic Bottleneck Hourglass ----
    y_hg = 2.755
    # Down-Blocks: 64^2 (2-slice) -> 32^2 (3-slice)
    ax.text(7.35, 3.83, "Latent\nStem", ha="center", va="center", fontsize=7.05, weight="bold", color="#1E3A8A", zorder=6)
    draw_multislice_block(ax, 7.08, y_hg, slice_w=0.11, h=1.18, n_slices=2, step_x=0.055, base_col="#60A5FA", ec="#1D4ED8", zorder=6)
    ax.text(7.18, y_hg - 0.71, r"$64^2$", ha="center", va="center", fontsize=6.85, weight="bold", color="#1E3A8A", zorder=6)

    draw_multislice_block(ax, 7.39, y_hg, slice_w=0.10, h=0.80, n_slices=3, step_x=0.05, base_col="#818CF8", ec="#4338CA", zorder=6)
    ax.text(7.51, y_hg - 0.52, r"$32^2$", ha="center", va="center", fontsize=6.85, weight="bold", color="#312E81", zorder=6)

    # Center 16x16 Semantic Bottleneck Card
    card(ax, 7.75, 1.69, 1.44, 2.28, fc="#EEF2FF", ec="#4338CA", lw=1.4, rad=0.08, shadow=True, zorder=4)
    ax.text(8.47, 3.83, r"$16{\times}16$ Bottleneck", ha="center", va="center", fontsize=7.85, weight="bold", color="#312E81", zorder=6)
    ax.text(8.47, 3.65, "Quotient-Keyed Router", ha="center", va="center", fontsize=7.0, weight="bold", color="#4338CA", zorder=6)

    # Visual Q x K^T attention heatmap + V / Chord interaction
    att_x, att_y, att_s = 7.99, 2.84, 0.55
    for qi in range(6):
        ax.add_patch(
            Rectangle(
                (att_x - 0.11, att_y + qi * (att_s / 6.0)),
                0.08,
                att_s / 6.0 - 0.01,
                facecolor="#3B82F6" if qi % 2 == 0 else "#60A5FA",
                edgecolor="#1E3A8A",
                linewidth=0.4,
                zorder=6,
            )
        )
    ax.text(att_x - 0.07, att_y + att_s + 0.05, r"$Q$", ha="center", va="bottom", fontsize=6.8, weight="bold", color="#1D4ED8", zorder=6)

    for ki in range(6):
        ax.add_patch(
            Rectangle(
                (att_x + ki * (att_s / 6.0), att_y + att_s + 0.03),
                att_s / 6.0 - 0.01,
                0.072,
                facecolor="#2563EB" if ki % 2 == 0 else "#93C5FD",
                edgecolor="#1E3A8A",
                linewidth=0.4,
                zorder=6,
            )
        )
    ax.text(att_x + att_s + 0.04, att_y + att_s + 0.065, r"$K^\top$", ha="left", va="center", fontsize=6.8, weight="bold", color="#1D4ED8", zorder=6)

    ax.imshow(att_map, extent=(att_x, att_x + att_s, att_y, att_y + att_s), cmap="YlGnBu", aspect="auto", interpolation="nearest", zorder=6)
    ax.add_patch(Rectangle((att_x, att_y), att_s, att_s, fill=False, edgecolor="#312E81", linewidth=0.95, zorder=7))

    ax.text(8.69, att_y + att_s / 2.0, r"$\otimes$", ha="center", va="center", fontsize=8.2, weight="bold", color="#4338CA", zorder=6)
    for vi, vcol in enumerate(["#F59E0B", "#FB7185", "#F97316", "#EA580C", "#D97706", "#BE123C"]):
        ax.add_patch(
            Rectangle(
                (8.83, att_y + vi * (att_s / 6.0)),
                0.15,
                att_s / 6.0 - 0.01,
                facecolor=vcol,
                edgecolor="#7C2D12",
                linewidth=0.45,
                zorder=6,
            )
        )
    ax.text(8.905, att_y + att_s + 0.045, r"$V, e$", ha="center", va="bottom", fontsize=6.8, weight="bold", color="#B45309", zorder=6)

    card(ax, 7.82, 2.05, 1.30, 0.65, fc="#FFFFFF", ec="#818CF8", lw=0.85, rad=0.05, zorder=5)
    ax.text(8.47, 2.54, r"$Q\!=\!\Pi_\perp(x_s),\;K\!=\!\Pi_\perp(x_{\mathrm{ref}})$", ha="center", va="center", fontsize=6.6, color="#1E293B", zorder=6)
    ax.text(8.47, 2.34, r"$V\!=\!\mathrm{Style}\;\oplus\;(\tilde{P}_{16}^{\perp}\!\otimes e)$", ha="center", va="center", fontsize=6.65, color="#312E81", weight="bold", zorder=6)
    ax.text(8.47, 2.15, r"$\to (\Gamma_{64}, B_{64})$ spatial mod.", ha="center", va="center", fontsize=6.45, color="#7C2D12", zorder=6)

    card(ax, 7.82, 1.75, 1.30, 0.23, fc="#DCFCE7", ec="#16A34A", lw=0.85, rad=0.05, zorder=5)
    ax.text(8.47, 1.865, r"No $64^2$ Source-Edge Lock!", ha="center", va="center", fontsize=6.7, weight="bold", color="#166534", zorder=6)

    # Up-Blocks: 32^2 (3-slice) -> 64^2 (2-slice) Brushstroke Synthesis Stages
    draw_multislice_block(ax, 9.27, y_hg, slice_w=0.10, h=0.80, n_slices=3, step_x=0.05, base_col="#FBBF24", ec="#B45309", zorder=6)
    ax.text(9.39, y_hg + 0.53, r"$32^2$", ha="center", va="center", fontsize=6.75, weight="bold", color="#92400E", zorder=6)

    draw_multislice_block(ax, 9.61, y_hg, slice_w=0.11, h=1.18, n_slices=2, step_x=0.055, base_col="#FB923C", ec="#C2410C", zorder=6)
    ax.text(9.71, y_hg + 0.72, r"$64^2$", ha="center", va="center", fontsize=6.75, weight="bold", color="#9A3412", zorder=6)
    ax.text(9.53, 3.83, "Brushstroke\nSynthesis", ha="center", va="center", fontsize=7.05, weight="bold", color="#9A3412", zorder=6)

    # Explicit AdaGN / Routed Modulation bar feeding both 32^2 & 64^2 synthesis blocks
    card(ax, 9.24, 1.69, 0.61, 0.27, fc="#FFF7ED", ec="#EA580C", lw=0.85, rad=0.045, zorder=5)
    ax.text(9.545, 1.825, r"AdaGN $(\Gamma, B)$", ha="center", va="center", fontsize=6.1, weight="bold", color="#9A3412", zorder=6)
    rounded_ortho_arrow(ax, [(9.39, 1.96), (9.39, y_hg - 0.42)], col="#D97706", lw=1.0, hs=5.6)
    rounded_ortho_arrow(ax, [(9.69, 1.96), (9.69, y_hg - 0.60)], col="#C2410C", lw=1.0, hs=5.6)

    # Hourglass horizontal links
    rounded_ortho_arrow(ax, [(6.91, y_hg), (7.07, y_hg)], col="#334155", lw=1.2, hs=6.8)
    rounded_ortho_arrow(ax, [(7.29, y_hg), (7.39, y_hg)], col="#4338CA", lw=1.1, hs=6.2)
    rounded_ortho_arrow(ax, [(7.65, y_hg), (7.74, y_hg)], col="#4338CA", lw=1.1, hs=6.2)
    rounded_ortho_arrow(ax, [(9.19, y_hg), (9.27, y_hg)], col="#B45309", lw=1.1, hs=6.2)
    rounded_ortho_arrow(ax, [(9.53, y_hg), (9.61, y_hg)], col="#C2410C", lw=1.1, hs=6.2)

    # ---- Module II-3: Continuous Zero-Chord Norm Gate & Residual Output ----
    ax.add_patch(Circle((10.11, y_hg), 0.105, facecolor="#FFF1F2", edgecolor="#BE123C", linewidth=1.15, zorder=7))
    ax.text(10.11, y_hg, r"$\times$", ha="center", va="center", fontsize=8.8, weight="bold", color="#9F1239", zorder=8)

    ax.add_patch(Circle((10.47, y_hg), 0.105, facecolor="#EFF6FF", edgecolor="#0369A1", linewidth=1.15, zorder=7))
    ax.text(10.47, y_hg, r"$+$", ha="center", va="center", fontsize=8.8, weight="bold", color="#0369A1", zorder=8)

    rounded_ortho_arrow(ax, [(9.84, y_hg), (9.99, y_hg)], col="#C2410C", lw=1.2, hs=6.8)
    rounded_ortho_arrow(ax, [(10.22, y_hg), (10.35, y_hg)], col="#BE123C", lw=1.2, hs=6.8)

    card(ax, 9.91, 1.69, 0.77, 0.64, fc="#FFF1F2", ec="#E11D48", lw=0.9, rad=0.05, zorder=5)
    ax.text(10.295, 2.18, "Norm Gate", ha="center", va="center", fontsize=6.75, weight="bold", color="#9F1239", zorder=6)
    ax.text(10.295, 1.90, r"$\lambda\tanh^2\!\frac{\|e\|}{\sigma_e}$", ha="center", va="center", fontsize=7.1, color="#881337", zorder=6)
    rounded_ortho_arrow(ax, [(10.11, 2.33), (10.11, y_hg - 0.11)], col="#BE123C", lw=1.15, hs=6.6)

    # Luma-Preserving Anchor bypass: exits right side of P_Y into x=6.98 corridor (zero overlap with 64x64x4 Split title!)
    rounded_ortho_arrow(
        ax,
        [(6.60, 3.53), (6.98, 3.53), (6.98, 4.19), (10.47, 4.19), (10.47, y_hg + 0.11)],
        col="#0284C7",
        lw=1.2,
        ls="--",
        rad=0.06,
        hs=7.4,
        halo=True,
        zorder=10,
    )
    ax.text(9.38, 4.22, r"Luma Anchor $z_0^\perp$ ($P_Y$ locked)", ha="center", va="bottom", fontsize=6.85, weight="bold", color="#0369A1", zorder=11)

    rounded_ortho_arrow(
        ax,
        [(4.79, 3.88), (5.50, 3.88), (5.50, 4.44), (8.12, 4.44), (8.12, 3.98)],
        col="#1D4ED8",
        lw=1.4,
        rad=0.065,
        hs=8.0,
        halo=True,
        zorder=13,
    )
    ax.text(6.82, 4.46, r"Invariant Layout $(C_{64}, Q_{32})$ · Source Texture Blocked", ha="center", va="bottom", fontsize=6.8, weight="bold", color="#1E40AF", zorder=14)

    # Middle Latent Highway: z_0 -> 64x64x4 Split
    rounded_ortho_arrow(ax, [(5.22, 2.755), (5.74, 2.755)], col="#0284C7", lw=1.4, halo=True, hs=8.0)

    # Bottom Chord Bus: dedicated clean lane at y = 1.37
    rounded_ortho_arrow(
        ax,
        [(5.23, 1.62), (5.50, 1.62), (5.50, 1.37), (8.47, 1.37), (8.47, 1.68)],
        col="#B45309",
        lw=1.35,
        rad=0.055,
        hs=7.6,
        halo=True,
    )
    ax.add_patch(Circle((8.47, 1.37), 0.028, facecolor="#B45309", edgecolor="white", linewidth=0.5, zorder=14))
    rounded_ortho_arrow(ax, [(8.47, 1.37), (9.545, 1.37), (9.545, 1.68)], col="#C2410C", lw=1.25, rad=0.055, hs=7.0)
    ax.add_patch(Circle((9.545, 1.37), 0.028, facecolor="#C2410C", edgecolor="white", linewidth=0.5, zorder=14))
    rounded_ortho_arrow(ax, [(9.545, 1.37), (10.295, 1.37), (10.295, 1.68)], col="#BE123C", lw=1.2, rad=0.055, hs=7.0)

    card(ax, 5.88, 1.285, 2.08, 0.175, fc="#FFFBEB", ec="#F59E0B", lw=0.75, rad=0.04, zorder=13)
    ax.text(6.92, 1.372, r"Target Style Chord $e(x_s\!\to\!\tau)$ (Zero Source-ID)", ha="center", va="center", fontsize=6.55, weight="bold", color="#92400E", zorder=14)

    # Unified Execution Bar inside Operator II (generous horizontal margins!)
    card(ax, 5.75, 0.91, 4.92, 0.30, fc="#FEFCE8", ec="#CA8A04", lw=0.95, rad=0.06, zorder=4)
    ax.text(
        8.21,
        1.06,
        r"1-Step Amortized (PRISM-1): $1.50\mathrm{M}\cdot 17\,\mathrm{ms}\;(58.7\,\mathrm{fps})$"
        r"   |   Bottleneck Flow: $134\,\mathrm{ms}\cdot 0.579$ DINO-S",
        ha="center",
        va="center",
        fontsize=6.85,
        weight="bold",
        color="#713F12",
        zorder=5,
    )

    # =========================================================================
    # 4. OPERATOR III PILLAR: SPECTRAL-BOTTLENECK DECODER D_* & OUTPUT x_hat
    # =========================================================================
    stage_container(
        ax,
        11.00,
        0.84,
        1.64,
        4.00,
        r"Op. III · PRISM-Dec $\mathcal{D}_\star$",
        body_fc="#FFF1F2",
        border_col="#BE123C",
        header_fc="#BE123C",
        header_h=0.25,
        title_fs=8.35,
    )

    ax.text(11.82, 4.41, r"$49.5\mathrm{M}\!\to\!\mathbf{10.7\mathrm{M}}$", ha="center", va="center", fontsize=7.95, weight="bold", color="#881337", zorder=6)
    ax.text(11.82, 4.21, r"$(-78.5\%\;\mathrm{Params},\;2.86\times\;\mathrm{Faster})$", ha="center", va="center", fontsize=6.6, weight="bold", color="#9F1239", zorder=6)

    # Bottom-aligned 4-stage expanding channel-pruned pyramid (base y = 2.12)
    stage_xs = [11.12, 11.46, 11.81, 12.16]
    full_hs = [1.05, 1.28, 1.54, 1.82]
    kept_hs = [0.46, 0.62, 0.82, 1.06]
    stage_lbls = [r"$64^2$", r"$128^2$", r"$256^2$", r"$512^2$"]
    y_base_dec = 2.12

    for sx, fh, kh, slbl in zip(stage_xs, full_hs, kept_hs, stage_lbls):
        ax.add_patch(
            Rectangle(
                (sx, y_base_dec + kh),
                0.22,
                fh - kh,
                facecolor="#E2E8F0",
                edgecolor="#94A3B8",
                linewidth=0.85,
                linestyle="--",
                alpha=0.88,
                zorder=4,
            )
        )
        draw_3d_slab(
            ax,
            sx,
            y_base_dec,
            0.21,
            kh,
            dx=0.046,
            dy=0.036,
            base_col="#FB7185",
            ec="#9F1239",
            lw=0.95,
            stripes=4,
            zorder=6,
        )
        ax.text(sx + 0.11, y_base_dec - 0.12, slbl, ha="center", va="center", fontsize=6.5, color="#475569", weight="bold", zorder=6)

    for i_st in range(len(stage_xs) - 1):
        mx = 0.5 * (stage_xs[i_st] + 0.23 + stage_xs[i_st + 1])
        ax.text(mx, y_base_dec - 0.12, r"$\rightarrow$", ha="center", va="center", fontsize=6.0, color="#9F1239", zorder=6)

    ax.text(11.58, 3.88, "Taylor-Pruned\nSD-VAE Filters", ha="center", va="center", fontsize=6.2, color="#64748B", style="italic", zorder=7)

    # Spectral Regularization Badge inside Op III
    card(ax, 11.09, 0.92, 1.46, 0.90, fc="#FFE4E6", ec="#E11D48", lw=1.0, rad=0.06, zorder=5)
    ax.text(11.82, 1.65, "Spectral Regularizer", ha="center", va="center", fontsize=7.1, weight="bold", color="#881337", zorder=6)
    ax.text(11.82, 1.40, "Suppresses Source\nTexture Retraction", ha="center", va="center", fontsize=6.6, color="#4C0519", zorder=6)
    ax.text(11.82, 1.07, r"$+3.17\mathrm{pp}$ Top-1 ($p\!<\!.01$)", ha="center", va="center", fontsize=6.9, weight="bold", color="#BE123C", zorder=6)

    # Stylized Output x_hat + 2.5x Zoom Comparison Column
    ax.text(13.54, 4.68, r"Stylized Output $\hat{x}$", ha="center", va="bottom", fontsize=8.9, weight="bold", color="#881337")
    ax.text(13.54, 4.49, r"$512{\times}512$ Ukiyo-e", ha="center", va="bottom", fontsize=7.3, weight="bold", color="#475569")
    framed_image(ax, out_img, 12.93, 3.20, 1.22, 1.22, ec="#BE123C", lw=1.5, zorder=6)
    ax.add_patch(Rectangle((12.93 + 0.32, 3.20 + 0.45), 0.56, 0.56, fill=False, edgecolor="#FACC15", linewidth=1.0, linestyle="--", zorder=8))

    card(ax, 12.90, 0.84, 1.27, 2.16, fc="#FFFDF9", ec="#CBD5E1", lw=1.05, rad=0.07, shadow=True, zorder=4)
    ax.text(13.535, 2.85, r"$2.5\times$ Structure & Stroke", ha="center", va="center", fontsize=7.0, weight="bold", color="#1E293B", zorder=5)

    framed_image(ax, src_zoom, 12.965, 2.00, 0.525, 0.68, ec="#334155", lw=0.9, shadow=False, zorder=6)
    ax.text(13.227, 1.87, r"Source $x_s$", ha="center", va="center", fontsize=6.55, color="#334155", weight="bold", zorder=6)

    framed_image(ax, out_zoom, 13.575, 2.00, 0.525, 0.68, ec="#BE123C", lw=1.05, shadow=False, zorder=6)
    ax.text(13.837, 1.87, r"PRISM $\hat{x}$", ha="center", va="center", fontsize=6.55, color="#9F1239", weight="bold", zorder=6)

    card(ax, 12.965, 0.93, 1.135, 0.80, fc="#FFF1F2", ec="#FDA4AF", lw=0.85, rad=0.05, zorder=5)
    ax.text(13.535, 1.56, "Zero Source-ID", ha="center", va="center", fontsize=6.8, weight="bold", color="#881337", zorder=6)
    ax.text(13.535, 1.33, r"$0.744$ DINO-C", ha="center", va="center", fontsize=6.85, weight="bold", color="#0F172A", zorder=6)
    ax.text(13.535, 1.10, r"$0.668$ Style Affinity", ha="center", va="center", fontsize=6.85, weight="bold", color="#BE123C", zorder=6)

    # Arrows: Op II -> Op III -> x_hat (centered cleanly in inter-pillar corridors with white border halos!)
    rounded_ortho_arrow(ax, [(10.58, y_hg), (10.89, y_hg), (10.89, 2.40), (11.11, 2.40)], col="#1E293B", lw=1.45, rad=0.05, halo=True, hs=8.0)
    ax.text(10.90, 2.86, r"$\hat{z}$", ha="center", va="bottom", fontsize=8.4, weight="bold", color="#0F172A", zorder=7)
    rounded_ortho_arrow(ax, [(12.42, 2.82), (12.78, 2.82), (12.78, 3.81), (12.92, 3.81)], col="#BE123C", lw=1.45, rad=0.06, halo=True, hs=8.4)

    # =========================================================================
    # 5. BOTTOM STRIP: OFFLINE TRAINING & DIFFERENTIABLE BF16 FEEDBACK
    # =========================================================================
    card(ax, 0.17, 0.08, 13.99, 0.56, fc="#FEFCE8", ec="#A16207", lw=1.15, rad=0.07, ls="--", zorder=2)

    card(ax, 0.27, 0.15, 2.18, 0.42, fc="#FEF3C7", ec="#D97706", lw=0.9, rad=0.05, zorder=4)
    ax.text(1.36, 0.42, "Offline Training Only", ha="center", va="center", fontsize=7.35, weight="bold", color="#78350F", zorder=5)
    ax.text(1.36, 0.24, "(0 ms Inference Cost)", ha="center", va="center", fontsize=6.55, color="#92400E", zorder=5)

    card(ax, 2.57, 0.15, 3.64, 0.42, fc="#FFFFFF", ec="#EAB308", lw=0.85, rad=0.05, zorder=4)
    ax.text(4.39, 0.43, r"1. Intra-Train $k$-NN Core & Anisotropic Trajectory Teacher", ha="center", va="center", fontsize=6.85, weight="bold", color="#0F172A", zorder=5)
    ax.text(4.39, 0.24, r"Filters boundary outliers; 2-step search ($\rho_Y\!\ll\!\rho_C$, $59.0\text{--}64.3\%$ Top-1)", ha="center", va="center", fontsize=6.45, color="#475569", zorder=5)

    card(ax, 6.33, 0.15, 3.84, 0.42, fc="#FFFFFF", ec="#EAB308", lw=0.85, rad=0.05, zorder=4)
    ax.text(8.25, 0.43, r"2. Un-Normalized VGG-16 + Oklab Moment Supervision", ha="center", va="center", fontsize=6.85, weight="bold", color="#0F172A", zorder=5)
    ax.text(8.25, 0.24, r"$\mathcal{L}_{\mathrm{VGG+Oklab}}(\mathcal{D}_\star(\hat{z})) + \mathcal{L}_{\mathrm{Huber}} + \lambda_{\mathrm{HF}}\mathcal{L}_{\mathrm{Haar}}$ (retains palette moments)", ha="center", va="center", fontsize=6.45, color="#475569", zorder=5)

    card(ax, 10.29, 0.15, 3.76, 0.42, fc="#FFFFFF", ec="#EAB308", lw=0.85, rad=0.05, zorder=4)
    ax.text(12.17, 0.43, r"3. 4-Stream Taylor Pruning & Fast BF16 Pixel Bridge", ha="center", va="center", fontsize=6.85, weight="bold", color="#0F172A", zorder=5)
    ax.text(12.17, 0.24, r"$\mathcal{I}(g)=|\sum W\,\partial\mathcal{L}_{\mathrm{rec}}/\partial W|\;\;(4.7\times\;\mathrm{leaner\;BF16\;backprop\;through}\;\mathcal{D}_\star)$", ha="center", va="center", fontsize=6.45, color="#475569", zorder=5)

    # Dashed BF16 gradient feedback arrow along dedicated y = 0.735 corridor
    rounded_ortho_arrow(
        ax,
        [(11.86, 0.84), (11.86, 0.735), (8.47, 0.735), (8.47, 0.84)],
        col="#BE123C",
        lw=1.25,
        ls="--",
        rad=0.04,
        hs=7.4,
    )
    card(ax, 9.15, 0.66, 2.08, 0.155, fc="#FFF1F2", ec="#FDA4AF", lw=0.75, rad=0.04, zorder=13)
    ax.text(
        10.19,
        0.737,
        r"BF16 Grad $\nabla_z \mathcal{L}_{\mathrm{VGG+Oklab}}(\mathcal{D}_\star(\hat{z}))$",
        ha="center",
        va="center",
        fontsize=6.45,
        weight="bold",
        color="#9F1239",
        zorder=14,
    )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(FIG_DIR / f"fig3_prism_architecture.{ext}", dpi=340 if ext == "png" else None)
    plt.close(fig)
    print("[Arch] Saved fig3_prism_architecture.{pdf,png,svg}")


if __name__ == "__main__":
    build_architecture()
