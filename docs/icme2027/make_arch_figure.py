#!/usr/bin/env python3
"""Build publication-grade ICME 2027 architecture diagram (modeled after aaai2027_v4/frame.png).

Visual grammar inherited & elevated from aaai2027_v4/frame.png:
  - Real artwork thumbnails: Source Content x_s, Target Style exemplar y_t, Stylized Output x_hat,
    plus a real computed 64x64 Luminance Local-Contrast (C_64) map.
  - Trapezoid Encoder (narrowing) and Pruned SLATE-Dec (expanding) in soft sky-blue.
  - Soft-cyan top Style Conditioning bar with cream sub-cards.
  - Warm-gray central solver container housing warm-orange (SLATE-1Step) and sage-green (SLATE-Pro).
  - Full-width pale-yellow bottom "Training & Pruning Only" strip with exact CM math equations.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, FancyArrowPatch, Circle

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


def _load_crop_square(path: Path, size: int = 256) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
    return np.asarray(img)


def _compute_lcn_preview(rgb_u8: np.ndarray) -> np.ndarray:
    rgb = rgb_u8.astype(np.float32) / 255.0
    y = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    # Downsample to 64x64 and compute 5x5 local contrast map
    y64 = np.asarray(Image.fromarray((y * 255).astype(np.uint8)).resize((64, 64), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    pad = np.pad(y64, 2, mode="reflect")
    mean = sum(pad[i:i+64, j:j+64] for i in range(5) for j in range(5)) / 25.0
    diff = y64 - mean
    d2 = np.pad(diff ** 2, 2, mode="reflect")
    std = np.sqrt(np.maximum(sum(d2[i:i+64, j:j+64] for i in range(5) for j in range(5)) / 25.0, 1e-8))
    lcn = np.clip(diff / (std + 0.03), -3.0, 3.0) / 6.0 + 0.5
    u8 = (lcn * 255).clip(0, 255).astype(np.uint8)
    return np.stack([u8, u8, u8], axis=-1)


def rbox(ax, x, y, w, h, *, fc="#FFFFFF", ec="#2C3E50", lw=1.4, rad=0.14, ls="-", zorder=2):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={rad}",
        facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=zorder,
    )
    ax.add_patch(p)
    return p


def trapezoid(ax, x, y_mid, w, h_left, h_right, *, fc="#D6EAF8", ec="#2E6B9E", lw=1.4, zorder=4):
    pts = [
        (x, y_mid - h_left / 2),
        (x + w, y_mid - h_right / 2),
        (x + w, y_mid + h_right / 2),
        (x, y_mid + h_left / 2),
    ]
    poly = Polygon(pts, closed=True, facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder)
    ax.add_patch(poly)
    return poly


def arrow(ax, x1, y1, x2, y2, *, col="#1C2833", lw=1.35, ls="-", rad=0.0, hs=10, zorder=8):
    conn = f"arc3,rad={rad}" if rad != 0.0 else "arc3,rad=0.0"
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=f"-|>,head_length={hs*0.45},head_width={hs*0.30}",
        connectionstyle=conn,
        color=col, linewidth=lw, linestyle=ls, zorder=zorder,
    )
    ax.add_patch(arr)


def ortho_arrow(ax, pts, *, col="#1C2833", lw=1.35, ls="-", hs=10, zorder=8):
    for i in range(len(pts) - 2):
        ax.plot([pts[i][0], pts[i+1][0]], [pts[i][1], pts[i+1][1]],
                color=col, lw=lw, ls=ls, solid_capstyle="round", zorder=zorder)
    arrow(ax, pts[-2][0], pts[-2][1], pts[-1][0], pts[-1][1], col=col, lw=lw, ls=ls, hs=hs, zorder=zorder)


def draw_thumb(ax, img_arr: np.ndarray, x: float, y: float, w: float, h: float, zorder: int = 5):
    ax.imshow(img_arr, extent=(x, x + w, y, y + h), aspect="auto", zorder=zorder, interpolation="bilinear")
    rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor="#1C2833", linewidth=1.2, zorder=zorder + 1)
    ax.add_patch(rect)


def build_architecture():
    src_path = REPO / "data/test/Early_Renaissance/Early_Renaissance__benozzo-gozzoli_madonna-and-child-1485.jpg"
    tgt_path = REPO / "data/test/Ukiyo_e/Ukiyo_e__hiroshige_hakone-kosuizu.jpg"
    out_path = REPO / "docs/performance/images/WEAVE_published_reval/Early_Renaissance_Early_Renaissance__benozzo-gozzoli_madonna-and-child-1485_to_Ukiyo_e.png"

    src_img = _load_crop_square(src_path, 256)
    tgt_img = _load_crop_square(tgt_path, 256)
    out_img = _load_crop_square(out_path, 256)
    lcn_img = _compute_lcn_preview(src_img)

    W, H = 18.0, 6.55
    fig = plt.figure(figsize=(W, H), facecolor="white")
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], facecolor="white")
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    # =========================================================================
    # 1. TOP BAR: Antisymmetric & Prior-Calibrated Style Conditioning (Cyan)
    # =========================================================================
    draw_thumb(ax, tgt_img, 0.25, 5.22, 1.08, 1.08)
    ax.text(0.79, 5.06, r"Target Style $y_t$", ha="center", va="top", fontsize=9.8, weight="bold", color="#182A42")

    rbox(ax, 1.75, 5.18, 13.75, 1.18, fc="#D4EEF7", ec="#1B6CA8", lw=1.6, rad=0.14, zorder=2)
    ax.text(15.32, 5.77, "Style\nConditioning", ha="right", va="center",
            fontsize=11.0, weight="bold", color="#0F3D63", zorder=5)

    # Three cream cards inside Top Bar
    rbox(ax, 1.95, 5.31, 3.85, 0.92, fc="#FEF9E7", ec="#D4AC0D", lw=1.2, rad=0.10, zorder=4)
    ax.text(3.875, 5.95, "Antisymmetric Style Chord", ha="center", va="center", fontsize=9.5, weight="bold", color="#182A42", zorder=5)
    ax.text(3.875, 5.56, r"$e_{s\to t} = u_{y_t} - u_{y_s}\;\;(e_{s\to s}\equiv \mathbf{0})$", ha="center", va="center", fontsize=9.3, color="#34495E", zorder=5)

    rbox(ax, 6.02, 5.31, 4.10, 0.92, fc="#FEF9E7", ec="#D4AC0D", lw=1.2, rad=0.10, zorder=4)
    ax.text(8.07, 5.95, "Holdout Prior Calibration", ha="center", va="center", fontsize=9.5, weight="bold", color="#182A42", zorder=5)
    ax.text(8.07, 5.56, r"$\tilde{f}(z)=f_\psi(z)-\hat{b}_{\mathrm{ho}}\;\;(\mathrm{rescues\ minority\ styles})$", ha="center", va="center", fontsize=9.0, color="#34495E", zorder=5)

    rbox(ax, 10.34, 5.31, 3.65, 0.92, fc="#FEF9E7", ec="#D4AC0D", lw=1.2, rad=0.10, zorder=4)
    ax.text(12.165, 5.95, "Local Patch-Energy Bank", ha="center", va="center", fontsize=9.5, weight="bold", color="#182A42", zorder=5)
    ax.text(12.165, 5.56, r"$\mathcal{E}_{\mathrm{loc}}(z, y_t)\;\;+\;\;\mathrm{SPD\ Metric\ } H_\omega$", ha="center", va="center", fontsize=9.1, color="#34495E", zorder=5)

    arrow(ax, 1.33, 5.76, 1.73, 5.76, col="#1B6CA8", lw=1.5)

    # =========================================================================
    # 2. LEFT INPUT & GATE-0 DECOUPLING (Quotient Projector + Canonical Anchor)
    # =========================================================================
    draw_thumb(ax, src_img, 0.22, 2.52, 1.14, 1.14)
    ax.text(0.79, 3.86, "Source Image", ha="center", va="bottom", fontsize=9.8, weight="bold", color="#182A42")
    ax.text(0.79, 3.68, r"$x_s \sim \mathbb{P}_{y_s}$", ha="center", va="bottom", fontsize=9.4, color="#34495E")

    # Upper-left: Operator I — Quotient Structure Projector Pi_\perp
    rbox(ax, 1.75, 3.22, 4.35, 1.72, fc="#E8F8F5", ec="#148F77", lw=1.5, rad=0.13, zorder=2)
    ax.text(3.925, 4.74, r"Operator I: Quotient Projector $\Pi_\perp(x_s)$",
            ha="center", va="center", fontsize=9.8, weight="bold", color="#0E6251", zorder=5)

    draw_thumb(ax, lcn_img, 1.90, 3.92, 0.62, 0.62, zorder=5)
    rbox(ax, 2.62, 3.94, 3.34, 0.58, fc="#FFFFFF", ec="#48C9B0", lw=1.0, rad=0.08, zorder=4)
    ax.text(4.29, 4.33, r"Luminance Contrast $C_{64}\in[-1,1]^{4\times 64^2}$", ha="center", va="center", fontsize=8.6, weight="bold", color="#182A42", zorder=5)
    ax.text(4.29, 4.08, r"$\mathrm{LCN}_{5,13}(Y)\oplus\mathrm{Sobel}\oplus\Delta Y$ (Hue-Free)", ha="center", va="center", fontsize=8.1, color="#566573", zorder=5)

    rbox(ax, 1.90, 3.32, 4.06, 0.52, fc="#FFFFFF", ec="#48C9B0", lw=1.0, rad=0.08, zorder=4)
    ax.text(3.93, 3.67, r"Whitened Grayscale DINOv2 $\tilde{P}_{16}\in\mathbb{R}^{384\times 16^2}$", ha="center", va="center", fontsize=8.6, weight="bold", color="#182A42", zorder=5)
    ax.text(3.93, 3.43, r"$\mu_{\mathrm{sp}}(\tilde{P}_{16})\equiv\mathbf{0}$  (5-way probe: $72.3\%\!\to\!18.3\%$)", ha="center", va="center", fontsize=8.0, color="#C0392B", weight="bold", zorder=5)

    # Lower-left: Frozen VAE Encoder + Haar Canonical Base Anchor
    trapezoid(ax, 1.75, 2.28, 1.18, 1.08, 0.68, fc="#D6EAF8", ec="#2E6B9E", lw=1.4, zorder=4)
    ax.text(2.34, 2.38, "Frozen VAE", ha="center", va="center", fontsize=8.5, weight="bold", color="#182A42", zorder=5)
    ax.text(2.34, 2.15, r"Encoder $\mathcal{E}$", ha="center", va="center", fontsize=8.3, color="#2C3E50", zorder=5)

    rbox(ax, 3.16, 1.82, 1.32, 0.92, fc="#EBF5FB", ec="#5DADE2", lw=1.2, rad=0.09, zorder=4)
    ax.text(3.82, 2.45, "Haar DWT", ha="center", va="center", fontsize=8.8, weight="bold", color="#182A42", zorder=5)
    ax.text(3.82, 2.18, r"Canon. $\mathcal{C}_1(z_s)$", ha="center", va="center", fontsize=8.4, color="#2C3E50", zorder=5)
    ax.text(3.82, 1.95, r"$(\ell, h_1, h_2, h_3)$", ha="center", va="center", fontsize=8.0, color="#566573", zorder=5)

    rbox(ax, 4.70, 1.82, 1.40, 0.92, fc="#FEF5E7", ec="#E67E22", lw=1.2, rad=0.09, zorder=4)
    ax.text(5.40, 2.46, "Base U-Net", ha="center", va="center", fontsize=8.8, weight="bold", color="#182A42", zorder=5)
    ax.text(5.40, 2.20, r"$(0.22\mathrm{M},\,1\mathrm{-step})$", ha="center", va="center", fontsize=8.0, color="#566573", zorder=5)
    ax.text(5.40, 1.95, r"Anchor $z_0$", ha="center", va="center", fontsize=8.7, weight="bold", color="#A04000", zorder=5)

    # Arrows from x_s into both branches
    ortho_arrow(ax, [(1.36, 3.25), (1.55, 3.25), (1.55, 4.08), (1.73, 4.08)], col="#148F77", lw=1.4)
    ortho_arrow(ax, [(1.36, 2.92), (1.55, 2.92), (1.55, 2.28), (1.73, 2.28)], col="#2E6B9E", lw=1.4)
    arrow(ax, 2.93, 2.28, 3.14, 2.28, col="#2E6B9E", lw=1.35)
    arrow(ax, 4.48, 2.28, 4.68, 2.28, col="#2E6B9E", lw=1.35)

    # =========================================================================
    # 3. CENTRAL MAIN CONTAINER: Dual-Mode Source-Lock-Free Latent Transport
    # =========================================================================
    rbox(ax, 6.42, 1.62, 8.32, 3.34, fc="#F2F4F6", ec="#2C3E50", lw=1.7, rad=0.15, zorder=2)
    ax.text(8.62, 4.77, r"Operator II: Antisymmetric Chord Transport $\mathcal{T}_{s\to t}$",
            ha="center", va="center", fontsize=10.5, weight="bold", color="#182A42", zorder=5)

    # ---- Upper Sub-Container: PRISM-1 (Warm Peach/Orange) ----
    rbox(ax, 6.64, 3.26, 7.90, 1.32, fc="#FDEBD0", ec="#CA6F1E", lw=1.5, rad=0.12, zorder=3)
    ax.text(6.78, 4.41, r"Mode A: PRISM-1 ($0.44\mathrm{M},\;28.5\,\mathrm{ms},\;1\mathrm{-step}$)",
            ha="left", va="center", fontsize=9.2, weight="bold", color="#873600", zorder=5)

    rbox(ax, 6.78, 3.40, 2.15, 0.80, fc="#FFFDF9", ec="#DC7633", lw=1.1, rad=0.08, zorder=4)
    ax.text(7.855, 3.96, "Multi-Scale Fusion Stem", ha="center", va="center", fontsize=8.5, weight="bold", color="#182A42", zorder=5)
    ax.text(7.855, 3.60, r"$\mathrm{Conv}([z_0,\mathcal{C}_1,C_{64}])+\mathrm{Up}(\tilde{P}_{16})$", ha="center", va="center", fontsize=8.1, color="#34495E", zorder=5)

    rbox(ax, 9.18, 3.40, 2.48, 0.80, fc="#FAD7A0", ec="#BA4A00", lw=1.2, rad=0.08, zorder=4)
    ax.text(10.42, 3.96, r"$4\times$ Width-80 AdaGN Blocks", ha="center", va="center", fontsize=8.6, weight="bold", color="#6E2C00", zorder=5)
    ax.text(10.42, 3.60, r"$h \leftarrow h + \Phi(h\cdot(1+\gamma(e))+\beta(e))$", ha="center", va="center", fontsize=8.0, color="#6E2C00", zorder=5)

    rbox(ax, 11.92, 3.40, 2.46, 0.80, fc="#FDEDEC", ec="#C0392B", lw=1.25, rad=0.08, zorder=4)
    ax.text(13.15, 3.96, "Diagonal Identity Gate", ha="center", va="center", fontsize=8.6, weight="bold", color="#922B21", zorder=5)
    ax.text(13.15, 3.59, r"$\hat{z} = z_0 + \mathbb{I}[y_t\neq y_s]\cdot\Delta_\theta$", ha="center", va="center", fontsize=8.5, weight="bold", color="#922B21", zorder=5)

    arrow(ax, 8.93, 3.80, 9.16, 3.80, col="#873600", lw=1.3)
    arrow(ax, 11.66, 3.80, 11.90, 3.80, col="#873600", lw=1.3)

    # ---- Lower Sub-Container: PRISM-Pro (Sage Green) ----
    rbox(ax, 6.64, 1.76, 7.90, 1.32, fc="#E8F6EF", ec="#1E8449", lw=1.5, rad=0.12, zorder=3)
    ax.text(6.78, 2.91, r"Mode B: PRISM-Pro (Prior-Calibrated Bidirectional SPD Solver, $M=3,\;38.0\%$ Top-1)",
            ha="left", va="center", fontsize=9.1, weight="bold", color="#145A32", zorder=5)

    rbox(ax, 6.78, 1.90, 2.52, 0.80, fc="#FFFFFF", ec="#27AE60", lw=1.1, rad=0.08, zorder=4)
    ax.text(8.04, 2.46, "Bidirectional Margin Grad", ha="center", va="center", fontsize=8.5, weight="bold", color="#182A42", zorder=5)
    ax.text(8.04, 2.10, r"$g_{s\to t}=\nabla_z[\tilde{m}(z,y_t)-\gamma\tilde{m}(z,y_s)]$", ha="center", va="center", fontsize=8.1, color="#145A32", zorder=5)

    rbox(ax, 9.54, 1.90, 2.38, 0.80, fc="#D4EFDF", ec="#1E8449", lw=1.15, rad=0.08, zorder=4)
    ax.text(10.73, 2.46, "SPD & HF Salience Mask", ha="center", va="center", fontsize=8.5, weight="bold", color="#145A32", zorder=5)
    ax.text(10.73, 2.10, r"$d = M_{\mathrm{HF}}\odot\mathcal{N}_{\mathrm{RMS}}(H_\omega g_{s\to t})$", ha="center", va="center", fontsize=8.1, color="#145A32", zorder=5)

    rbox(ax, 12.16, 1.90, 2.22, 0.80, fc="#FFFFFF", ec="#27AE60", lw=1.1, rad=0.08, zorder=4)
    ax.text(13.27, 2.46, "Trust-Region Search", ha="center", va="center", fontsize=8.5, weight="bold", color="#182A42", zorder=5)
    ax.text(13.27, 2.10, r"$\|z-z_0\|_{\mathrm{RMS}}\leq\rho,\;\mathcal{C}\leq\epsilon$", ha="center", va="center", fontsize=8.1, color="#34495E", zorder=5)

    arrow(ax, 9.30, 2.30, 9.52, 2.30, col="#145A32", lw=1.3)
    arrow(ax, 11.92, 2.30, 12.14, 2.30, col="#145A32", lw=1.3)

    # Routing arrows into Central Solver (strictly non-overlapping corridors!)
    # 1) Pi_\perp -> SLATE-1Step stem
    arrow(ax, 6.10, 3.92, 6.76, 3.92, col="#148F77", lw=1.45)
    # 2) Anchor z_0 -> both SLATE-1Step & SLATE-Pro
    ortho_arrow(ax, [(6.10, 2.18), (6.26, 2.18), (6.26, 3.64), (6.76, 3.64)], col="#A04000", lw=1.35)
    arrow(ax, 6.10, 2.18, 6.76, 2.18, col="#A04000", lw=1.35)
    # 3) Top Style Conditioning -> AdaGN (drops at x=10.85, completely clear of Mode A title!)
    ortho_arrow(ax, [(3.875, 5.31), (3.875, 5.06), (10.95, 5.06), (10.95, 4.22)], col="#BA4A00", lw=1.35)
    # 4) Holdout Calibration -> SLATE-Pro via outer-left corridor x=6.51 (zero box crossing!)
    ortho_arrow(ax, [(8.07, 5.31), (8.07, 4.99), (6.52, 4.99), (6.52, 2.50), (6.76, 2.50)], col="#1B6CA8", lw=1.25, ls="--")
    # 5) Local Patch-Energy Bank -> SLATE-Pro via right corridor x=14.64
    ortho_arrow(ax, [(12.165, 5.31), (12.165, 5.04), (14.64, 5.04), (14.64, 2.30), (14.40, 2.30)], col="#1E8449", lw=1.2, ls="--")

    # =========================================================================
    # 4. RIGHT OUTPUT STAGE: Operator III — Pruned PRISM-Dec + Stylized x_hat
    # =========================================================================
    trapezoid(ax, 15.12, 3.08, 1.26, 0.76, 1.24, fc="#D6EAF8", ec="#C0392B", lw=1.65, zorder=4)
    ax.text(15.75, 3.30, "PRISM-Dec", ha="center", va="center", fontsize=9.0, weight="bold", color="#922B21", zorder=5)
    ax.text(15.75, 3.05, r"$10.7\mathrm{M}\;(-78.5\%)$", ha="center", va="center", fontsize=7.9, weight="bold", color="#182A42", zorder=5)
    ax.text(15.75, 2.83, r"$2.86\times\;\mathrm{faster}$", ha="center", va="center", fontsize=7.9, color="#1B6CA8", weight="bold", zorder=5)

    rbox(ax, 14.92, 1.78, 1.66, 0.44, fc="#FDEDEC", ec="#C0392B", lw=1.0, rad=0.07, zorder=4)
    ax.text(15.75, 2.00, r"Op. III: $+3.17\mathrm{pp}$ Top-1", ha="center", va="center", fontsize=7.8, weight="bold", color="#922B21", zorder=5)

    draw_thumb(ax, out_img, 16.64, 2.51, 1.14, 1.14)
    ax.text(17.21, 3.85, "Stylized Output", ha="center", va="bottom", fontsize=9.8, weight="bold", color="#182A42")
    ax.text(17.21, 3.67, r"$\hat{x}_{s\to t}$", ha="center", va="bottom", fontsize=9.8, color="#922B21")

    ortho_arrow(ax, [(14.38, 3.80), (14.90, 3.80), (14.90, 3.22), (15.10, 3.22)], col="#922B21", lw=1.45)
    ortho_arrow(ax, [(14.38, 2.30), (14.90, 2.30), (14.90, 2.94), (15.10, 2.94)], col="#1E8449", lw=1.35)
    arrow(ax, 16.38, 3.08, 16.62, 3.08, col="#182A42", lw=1.5)

    # =========================================================================
    # 5. BOTTOM STRIP: Offline Cross-Domain Distillation & Decoder Pruning
    # =========================================================================
    ax.text(0.26, 1.27, "Offline Amortization & Spectral-Bottleneck Pruning (Training Only)",
            ha="left", va="bottom", fontsize=10.2, weight="bold", color="#182A42")
    rbox(ax, 0.22, 0.16, 17.56, 1.04, fc="#FEF9E7", ec="#B7950B", lw=1.4, rad=0.12, zorder=2)

    rbox(ax, 0.42, 0.29, 3.75, 0.78, fc="#FFFFFF", ec="#D4AC0D", lw=1.05, rad=0.08, zorder=4)
    ax.text(2.295, 0.84, "1. Balanced Cross-Domain Pairs", ha="center", va="center", fontsize=8.8, weight="bold", color="#182A42", zorder=5)
    ax.text(2.295, 0.49, r"Generate $2{,}500$ clean $(z_0, z_{\mathrm{Pro}}^\star)$ via PRISM-Pro", ha="center", va="center", fontsize=8.2, color="#34495E", zorder=5)

    rbox(ax, 4.52, 0.29, 4.35, 0.78, fc="#FFFFFF", ec="#D4AC0D", lw=1.05, rad=0.08, zorder=4)
    ax.text(6.695, 0.84, "2. Skew-Symmetric Trajectory Loss", ha="center", va="center", fontsize=8.8, weight="bold", color="#182A42", zorder=5)
    ax.text(6.695, 0.49, r"$\mathcal{L}_{\mathrm{stu}}=\|\Delta_\theta - (z_{\mathrm{Pro}}^\star - z_0)\|_H + \lambda_{\mathrm{HF}}\|P_H(\hat{z}-z_{\mathrm{Pro}}^\star)\|_2^2$",
            ha="center", va="center", fontsize=8.2, color="#34495E", zorder=5)

    rbox(ax, 9.22, 0.29, 3.80, 0.78, fc="#FFFFFF", ec="#D4AC0D", lw=1.05, rad=0.08, zorder=4)
    ax.text(11.12, 0.84, "3. Exact Diagonal Fiber Guarantee", ha="center", va="center", fontsize=8.8, weight="bold", color="#182A42", zorder=5)
    ax.text(11.12, 0.49, r"$y_t=y_s \Rightarrow e_{s\to s}=\mathbf{0},\;\hat{z}_{s\to s}\equiv z_s\;(96.7\%\;\mathrm{Same150})$",
            ha="center", va="center", fontsize=8.1, color="#922B21", weight="bold", zorder=5)

    rbox(ax, 13.37, 0.29, 4.20, 0.78, fc="#FFFFFF", ec="#D4AC0D", lw=1.05, rad=0.08, zorder=4)
    ax.text(15.47, 0.84, "4. 4-Stream Taylor Decoder Pruning", ha="center", va="center", fontsize=8.8, weight="bold", color="#182A42", zorder=5)
    ax.text(15.47, 0.49, r"$\mathcal{I}(g)=|\sum W\,\partial\mathcal{L}_{\mathrm{rec}}/\partial W|\;\;(49.5\mathrm{M}\!\to\!10.7\mathrm{M})$",
            ha="center", va="center", fontsize=8.2, color="#34495E", zorder=5)

    arrow(ax, 4.17, 0.68, 4.50, 0.68, col="#7D6608", lw=1.3)
    arrow(ax, 8.87, 0.68, 9.20, 0.68, col="#7D6608", lw=1.3)
    arrow(ax, 13.02, 0.68, 13.35, 0.68, col="#7D6608", lw=1.3)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(FIG_DIR / f"fig3_prism_architecture.{ext}", dpi=320 if ext == "png" else None)
    plt.close(fig)
    print("[Arch] Saved fig3_prism_architecture.{pdf,png,svg}")


if __name__ == "__main__":
    build_architecture()
