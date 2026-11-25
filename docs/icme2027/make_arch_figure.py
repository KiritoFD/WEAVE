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
    # 0. TOP UNIFIED BANNER (Selective Information Filtering)
    # =========================================================================
    ax.plot([0.18, 2.05], [5.03, 5.03], color="#CBD5E1", lw=1.0, zorder=1)
    ax.plot([12.25, 14.12], [5.03, 5.03], color="#CBD5E1", lw=1.0, zorder=1)
    card(ax, 2.12, 4.89, 10.06, 0.27, fc="#F1F5F9", ec="#475569", lw=1.1, rad=0.09, zorder=3)
    ax.text(
        7.15,
        5.025,
        r"CLEAVE Selective Information Filtering:   [×] Remove Source-Style Backflow   |   [+] Preserve Controllable Geometry & Target Brushwork",
        ha="center",
        va="center",
        fontsize=8.6,
        weight="bold",
        color="#0F172A",
        zorder=4,
    )

    # =========================================================================
    # 1. LEFT INPUT COLUMN (x_s, y_t, s)
    # =========================================================================
    ax.text(0.73, 4.66, r"Source $x_s$", ha="center", va="bottom", fontsize=9.8, weight="bold", color="#0F172A")
    framed_image(ax, src_img, 0.17, 3.48, 1.12, 1.12, ec="#1E3A8A", lw=1.35, zorder=6)
    ax.text(0.73, 3.30, r"$512{\times}512$", ha="center", va="center", fontsize=8.3, color="#1E40AF", weight="bold", zorder=5)

    ax.text(0.73, 2.46, r"Target $y_t,\;s$", ha="center", va="bottom", fontsize=9.8, weight="bold", color="#7C2D12")
    framed_image(ax, ref_img, 0.17, 1.28, 1.12, 1.12, ec="#B45309", lw=1.35, zorder=6)
    card(ax, 0.15, 0.96, 1.16, 0.24, fc="#FFFBEB", ec="#FCD34D", lw=0.9, rad=0.05, zorder=4)
    ax.text(0.73, 1.08, r"Knob $s\in[0,1.5]$", ha="center", va="center", fontsize=8.1, color="#92400E", weight="bold", zorder=5)

    # =========================================================================
    # 2. FILTER I PILLAR: CONDITIONING SPACE (Blue/Teal matching Fig. 4a)
    # =========================================================================
    stage_container(
        ax,
        1.64,
        0.84,
        3.78,
        4.00,
        r"Filter I · Conditioning (ConvNeXt-S3, $12.4\mathrm{M}$)",
        body_fc="#F8FAFC",
        border_col="#1E3A5F",
        header_fc="#1E3A5F",
        header_h=0.28,
        title_fs=9.2,
    )

    # Upper visual: Grayscale -> LCN -> Stiefel Q32
    m_y = 3.55
    m_s = 0.78
    framed_image(ax, y_map, 1.84, m_y, m_s, m_s, ec="#334155", lw=1.0, zorder=6)
    ax.text(2.23, 3.38, r"Grayscale $Y$", ha="center", va="center", fontsize=8.1, weight="bold", color="#0F172A", zorder=6)
    rounded_ortho_arrow(ax, [(2.65, m_y + m_s / 2), (2.98, m_y + m_s / 2)], col="#1D4ED8", lw=1.4, hs=8.0)

    framed_image(ax, lcn_map, 3.01, m_y, m_s, m_s, ec="#1D4ED8", lw=1.0, zorder=6)
    ax.text(3.40, 3.38, r"$C_{64},\;Y_{16}$", ha="center", va="center", fontsize=8.2, weight="bold", color="#1E3A8A", zorder=6)
    rounded_ortho_arrow(ax, [(3.82, m_y + m_s / 2), (4.15, m_y + m_s / 2)], col="#1D4ED8", lw=1.4, hs=8.0)

    framed_image(ax, zca_src_img, 4.18, m_y, m_s, m_s, ec="#1E3A8A", lw=1.15, interp="nearest", zorder=7)
    ax.text(4.57, 3.38, r"Stiefel $Q_{32}$", ha="center", va="center", fontsize=8.2, weight="bold", color="#1E3A8A", zorder=6)

    # Clean Red/Green Filter I Summary Badges
    card(ax, 1.80, 2.84, 3.46, 0.34, fc="#FEE2E2", ec="#DC2626", lw=1.05, rad=0.06, zorder=5)
    ax.text(3.53, 3.01, r"[×] Strip Source-Style Correlations ($91.3\%\to 20.7\%$)", ha="center", va="center", fontsize=8.0, weight="bold", color="#991B1B", zorder=6)

    card(ax, 1.80, 2.42, 3.46, 0.34, fc="#DCFCE7", ec="#16A34A", lw=1.05, rad=0.06, zorder=5)
    ax.text(3.53, 2.59, r"[+] Keep Semantic Layout ($Q_{32},\;C_{64},\;Y_{16}$)", ha="center", va="center", fontsize=8.0, weight="bold", color="#166534", zorder=6)

    # Lower visual: K=8 Sub-School Bank
    card(ax, 1.80, 0.96, 3.46, 1.32, fc="#FFFBEB", ec="#D97706", lw=1.2, rad=0.07, zorder=3)
    ax.text(3.53, 2.09, r"$K=8$ Sub-School Bank ($417\text{-D}$) & Scene Router", ha="center", va="center", fontsize=8.4, weight="bold", color="#92400E", zorder=5)

    sw_x0, sw_w, sw_h = 1.98, 0.25, 0.22
    for idx, col_rgb in enumerate(ref_swatches):
        ax.add_patch(Rectangle((sw_x0 + idx * sw_w, 1.54), sw_w - 0.025, sw_h, facecolor=col_rgb, edgecolor="#334155", linewidth=0.6, zorder=6))
    ax.text(2.70, 1.27, "Oklab + Haar + CNX", ha="center", va="center", fontsize=7.8, weight="bold", color="#78350F", zorder=5)

    rounded_ortho_arrow(ax, [(3.56, 1.65), (3.84, 1.65)], col="#B45309", lw=1.35, hs=7.8)
    draw_sphere_manifold(ax, cx=4.22, cy=1.60, r=0.28, zorder=6)
    ax.text(4.86, 1.60, r"Routed" "\n" r"$u_{y_t,k^\star}$", ha="center", va="center", fontsize=8.2, weight="bold", color="#9F1239", zorder=6)

    rounded_ortho_arrow(ax, [(1.29, 3.92), (1.82, 3.92)], col="#1D4ED8", lw=1.55)
    rounded_ortho_arrow(ax, [(1.29, 1.65), (1.79, 1.65)], col="#B45309", lw=1.55)

    # =========================================================================
    # 3. FILTER II PILLAR: TRAJECTORY SPACE (Slate/Crimson matching Fig. 4b)
    # =========================================================================
    stage_container(
        ax,
        5.62,
        0.84,
        5.18,
        4.00,
        r"Filter II · Trajectory (2-Tier CPU-Like P/E-Core $\mathcal{T}_{\mathrm{P/E}}$)",
        body_fc="#F8FAFC",
        border_col="#1E293B",
        header_fc="#1E293B",
        header_h=0.28,
        title_fs=9.2,
    )

    # ---- Top Tier (Teal): E-Core (1-NFE, 11.1M) ----
    card(ax, 5.76, 3.44, 4.18, 1.04, fc="#F0FDFA", ec="#0D9488", lw=1.45, rad=0.07, zorder=4)
    ax.text(7.85, 4.26, r"E-Core ($s \leq 0.75$ · 1-NFE, $11.1\mathrm{M}$)", ha="center", va="center", fontsize=8.9, weight="bold", color="#115E59", zorder=6)
    ax.text(7.85, 3.96, r"[+] Keep Controllable $64^2$ Geometry ($P_Y\!\oplus\!P_C$, exact $s=0$)", ha="center", va="center", fontsize=7.95, weight="bold", color="#166534", zorder=6)
    ax.text(7.85, 3.66, r"[+] Emit $1\,\mathrm{ms}$ Structural & Color Draft $z_{\mathrm{E}}$", ha="center", va="center", fontsize=7.95, weight="bold", color="#0F766E", zorder=6)

    # ---- Prominent E-Core -> P-Core Draft Arrow (Signature P/E Design!) ----
    rounded_ortho_arrow(
        ax,
        [(6.38, 3.44), (6.38, 2.04)],
        col="#0D9488",
        lw=2.6,
        hs=10.5,
        halo=True,
        zorder=11,
    )
    card(ax, 5.76, 2.50, 1.24, 0.56, fc="#CCFBF1", ec="#0F766E", lw=1.1, rad=0.06, zorder=12)
    ax.text(6.38, 2.78, "1 ms Draft $z_{\\mathrm{E}}$\n(Seeds P-Core)", ha="center", va="center", fontsize=7.55, weight="bold", color="#115E59", zorder=13)

    # ---- Center Simplified U-Net Bottleneck Diagram ----
    y_hg = 2.74
    draw_multislice_block(ax, 7.22, y_hg, slice_w=0.11, h=0.92, n_slices=2, step_x=0.055, base_col="#60A5FA", ec="#1D4ED8", zorder=6)
    ax.text(7.32, y_hg - 0.58, r"$64^2$", ha="center", va="center", fontsize=7.8, weight="bold", color="#1E3A8A", zorder=6)

    card(ax, 7.60, 2.22, 1.44, 1.04, fc="#EEF2FF", ec="#4338CA", lw=1.35, rad=0.07, shadow=True, zorder=4)
    ax.text(8.32, 3.06, r"$16{\times}16$ Bottleneck", ha="center", va="center", fontsize=8.3, weight="bold", color="#312E81", zorder=6)
    ax.imshow(att_map, extent=(7.74, 8.20, 2.38, 2.84), cmap="YlGnBu", aspect="auto", interpolation="nearest", zorder=6)
    ax.add_patch(Rectangle((7.74, 2.38), 0.46, 0.46, fill=False, edgecolor="#312E81", linewidth=0.9, zorder=7))
    ax.text(8.62, 2.61, r"$Q_{32}+Y_{16}$" "\n" r"Only!", ha="center", va="center", fontsize=7.8, weight="bold", color="#1D4ED8", zorder=6)

    draw_multislice_block(ax, 9.26, y_hg, slice_w=0.11, h=0.92, n_slices=2, step_x=0.055, base_col="#FB923C", ec="#C2410C", zorder=6)
    ax.text(9.36, y_hg - 0.58, r"$64^2$", ha="center", va="center", fontsize=7.8, weight="bold", color="#9A3412", zorder=6)

    rounded_ortho_arrow(ax, [(7.44, y_hg), (7.59, y_hg)], col="#4338CA", lw=1.3, hs=7.0)
    rounded_ortho_arrow(ax, [(9.05, y_hg), (9.24, y_hg)], col="#C2410C", lw=1.3, hs=7.0)

    # ---- Right Hybrid Director Switch (User-Budget Dispatch, Not Input MoE!) ----
    card(ax, 10.02, 1.64, 0.68, 2.20, fc="#FFF1F2", ec="#BE123C", lw=1.25, rad=0.07, zorder=6)
    ax.text(10.36, 3.56, "Hybrid\nDirector", ha="center", va="center", fontsize=7.9, weight="bold", color="#9F1239", zorder=7)
    ax.text(10.36, 2.90, "User\nKnob $s$", ha="center", va="center", fontsize=7.8, weight="bold", color="#0F172A", zorder=7)
    ax.text(10.36, 2.15, "2-Tier\nBudget\n(Not MoE)", ha="center", va="center", fontsize=7.0, weight="bold", color="#881337", zorder=7)

    rounded_ortho_arrow(ax, [(9.94, 3.94), (10.36, 3.94), (10.36, 3.85)], col="#0D9488", lw=1.45, hs=7.4)
    rounded_ortho_arrow(ax, [(9.94, 1.48), (10.36, 1.48), (10.36, 1.63)], col="#E11D48", lw=1.45, hs=7.4)

    # ---- Bottom Tier (Crimson): P-Core (6-Step Flow, 22.3M) ----
    card(ax, 5.76, 0.96, 4.18, 1.04, fc="#FFF1F2", ec="#E11D48", lw=1.45, rad=0.07, zorder=4)
    ax.text(7.85, 1.78, r"P-Core ($s \geq 0.80$ · 6-Step Flow, $22.3\mathrm{M}$)", ha="center", va="center", fontsize=8.9, weight="bold", color="#9F1239", zorder=6)
    ax.text(7.85, 1.48, r"[×] Drop $64^2/32^2$ Source Micro-Textures ($16{\times}16$ Only)", ha="center", va="center", fontsize=7.95, weight="bold", color="#991B1B", zorder=6)
    ax.text(7.85, 1.18, r"[+] Break Source-Lock via Pool C Reversed Pairs", ha="center", va="center", fontsize=7.95, weight="bold", color="#166534", zorder=6)

    # Inter-pillar arrows Filter I -> Filter II
    rounded_ortho_arrow(ax, [(5.42, 3.92), (5.75, 3.92)], col="#1D4ED8", lw=1.55, halo=True, hs=8.2)
    rounded_ortho_arrow(ax, [(5.42, 1.48), (5.75, 1.48)], col="#B45309", lw=1.55, halo=True, hs=8.2)

    # =========================================================================
    # 4. FILTER III PILLAR: DECODING SPACE (Crimson matching Fig. 4c)
    # =========================================================================
    stage_container(
        ax,
        11.00,
        0.84,
        1.64,
        4.00,
        r"Filter III · CLEAVE-Dec",
        body_fc="#FFF1F2",
        border_col="#BE123C",
        header_fc="#BE123C",
        header_h=0.28,
        title_fs=8.9,
    )

    ax.text(11.82, 4.35, r"$49.5\mathrm{M}\!\to\!10.7\mathrm{M}$", ha="center", va="center", fontsize=8.8, weight="bold", color="#881337", zorder=6)
    ax.text(11.82, 4.10, r"($-78.5\%$ Pruned)", ha="center", va="center", fontsize=7.6, weight="bold", color="#9F1239", zorder=6)

    stage_xs = [11.12, 11.46, 11.81, 12.16]
    full_hs = [0.92, 1.12, 1.34, 1.58]
    kept_hs = [0.40, 0.54, 0.72, 0.94]
    stage_lbls = [r"$64^2$", r"$128^2$", r"$256^2$", r"$512^2$"]
    y_base_dec = 2.34

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
        ax.text(sx + 0.11, y_base_dec - 0.14, slbl, ha="center", va="center", fontsize=7.1, color="#475569", weight="bold", zorder=6)

    card(ax, 11.08, 1.56, 1.48, 0.52, fc="#FEE2E2", ec="#DC2626", lw=1.05, rad=0.06, zorder=5)
    ax.text(11.82, 1.82, "[×] Remove Texture\nRetraction", ha="center", va="center", fontsize=7.65, weight="bold", color="#991B1B", zorder=6)

    card(ax, 11.08, 0.94, 1.48, 0.54, fc="#DCFCE7", ec="#16A34A", lw=1.05, rad=0.06, zorder=5)
    ax.text(11.82, 1.21, "[+] Keep Brushwork\n(+4.49pp, 3.15×)", ha="center", va="center", fontsize=7.55, weight="bold", color="#166534", zorder=6)

    # Output x_hat + 2.5x Zoom Detail
    ax.text(13.54, 4.66, r"Stylized $\hat{x}$", ha="center", va="bottom", fontsize=9.6, weight="bold", color="#881337")
    ax.text(13.54, 4.46, r"Ukiyo-e ($512^2$)", ha="center", va="bottom", fontsize=8.0, weight="bold", color="#475569")
    framed_image(ax, out_img, 12.93, 3.18, 1.22, 1.22, ec="#BE123C", lw=1.5, zorder=6)
    ax.add_patch(Rectangle((12.93 + 0.32, 3.18 + 0.45), 0.56, 0.56, fill=False, edgecolor="#FACC15", linewidth=1.0, linestyle="--", zorder=8))

    card(ax, 12.90, 0.84, 1.27, 2.16, fc="#FFFDF9", ec="#CBD5E1", lw=1.05, rad=0.07, shadow=True, zorder=4)
    ax.text(13.535, 2.84, r"$2.5\times$ Stroke Detail", ha="center", va="center", fontsize=7.8, weight="bold", color="#1E293B", zorder=5)

    framed_image(ax, src_zoom, 12.965, 1.98, 0.525, 0.68, ec="#334155", lw=0.9, shadow=False, zorder=6)
    ax.text(13.227, 1.83, r"Source", ha="center", va="center", fontsize=7.3, color="#334155", weight="bold", zorder=6)

    framed_image(ax, out_zoom, 13.575, 1.98, 0.525, 0.68, ec="#BE123C", lw=1.05, shadow=False, zorder=6)
    ax.text(13.837, 1.83, r"CLEAVE", ha="center", va="center", fontsize=7.3, color="#9F1239", weight="bold", zorder=6)

    card(ax, 12.965, 0.94, 1.135, 0.74, fc="#FFF1F2", ec="#FDA4AF", lw=0.9, rad=0.05, zorder=5)
    ax.text(13.535, 1.48, "18.2–33.1 img/s", ha="center", va="center", fontsize=7.45, weight="bold", color="#BE123C", zorder=6)
    ax.text(13.535, 1.16, "Cold RGB→RGB", ha="center", va="center", fontsize=7.2, weight="bold", color="#0F172A", zorder=6)

    # Arrows: Filter II -> Filter III -> x_hat
    rounded_ortho_arrow(ax, [(10.71, y_hg), (11.10, y_hg)], col="#1E293B", lw=1.55, halo=True, hs=8.4)
    ax.text(10.90, y_hg + 0.08, r"$\hat{z}$", ha="center", va="bottom", fontsize=9.2, weight="bold", color="#0F172A", zorder=7)
    rounded_ortho_arrow(ax, [(12.42, 2.82), (12.78, 2.82), (12.78, 3.79), (12.92, 3.79)], col="#BE123C", lw=1.55, rad=0.06, halo=True, hs=8.4)

    # =========================================================================
    # 5. COMPACT SINGLE-LINE OFFLINE TRAINING STRIP
    # =========================================================================
    card(ax, 0.17, 0.16, 13.99, 0.42, fc="#FEFCE8", ec="#A16207", lw=1.1, rad=0.07, ls="--", zorder=2)
    ax.text(
        7.16,
        0.37,
        r"Offline Training Only (0 ms Inference Cost):   Pool C In-Domain Reversed Real-Painting Pairs   ·   High-$\tau$ Pixel Patch-SWD & Taylor Channel Pruning",
        ha="center",
        va="center",
        fontsize=8.2,
        weight="bold",
        color="#78350F",
        zorder=5,
    )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(FIG_DIR / f"fig3_prism_architecture.{ext}", dpi=340 if ext == "png" else None)
    plt.close(fig)
    print("[Arch] Saved fig3_prism_architecture.{pdf,png,svg}")


if __name__ == "__main__":
    build_architecture()
