#!/usr/bin/env python3
"""Build full-width Qualitative Visual Comparison Matrix (fig5_qualitative) for PRISM ICME 2027."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

BASE_JPG = REPO / "docs/performance/images/baselines_D5_512/jpg"
SG_DIR = REPO / "docs/baseline/StyleGallery"
PRISM1_DIR = REPO / "docs/performance/images/PRISM1_v2_750"
PRISM_PRO_DIR = REPO / "docs/performance/images/PRISM_Pro_T4_750"
PRISM_FLOW_DIR = REPO / "docs/performance/images/PRISM_Flow_D2_750"


def _load_sg_map() -> dict[str, Path]:
    out: dict[str, Path] = {}
    ex_p = SG_DIR / "execution.jsonl"
    if ex_p.exists():
        for line in ex_p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            out[r["task_id"]] = SG_DIR / "images" / Path(r["result"]).name
    return out


def _find_baseline_img(folder: Path, src_style: str, stem: str, tgt_style: str) -> Path:
    cands = [
        folder / f"{src_style}__{stem}__to__{tgt_style}.jpg",
        folder / f"{src_style}_{stem}_to_{tgt_style}.jpg",
        folder / f"{stem}__to__{tgt_style}.jpg",
        folder / f"{stem}_to_{tgt_style}.jpg",
    ]
    for c in cands:
        if c.exists():
            return c
    matches = list(folder.glob(f"*{stem.split('__')[-1]}*{tgt_style}*"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find {stem} -> {tgt_style} in {folder}")


def _load_rgb(path: Path, size: int = 288) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB").resize((size, size), Image.Resampling.LANCZOS))


def build_qualitative_grid():
    sg_map = _load_sg_map()
    rows_spec = [
        (
            "Early_Renaissance",
            "Early_Renaissance__benozzo-gozzoli_madonna-and-child-1485",
            "Ukiyo_e",
            "Early Ren.\n→ Ukiyo-e",
        ),
        (
            "Impressionism",
            "Impressionism__armand-guillaumin_valley-in-pontgibaud-1890",
            "Minimalism",
            "Impressionism\n→ Minimalism",
        ),
        (
            "Ukiyo_e",
            "Ukiyo_e__hiroshige_hakone-kosuizu",
            "Impressionism",
            "Ukiyo-e\n→ Impressionism",
        ),
    ]

    cols_spec = [
        ("Source", None, "Input $x_s$", False),
        ("StyleShot", "styleshot", "24.5 s · 0.36 D-C", False),
        ("StyleGallery", "stylegallery", "30.4 s · .560 D-S", False),
        ("StyTr$^2$", "stytr2", "3.0 s · .407 D-S", False),
        ("Seedream 4.5", "seedream", "2.0 s · .389 D-S", False),
        ("PRISM-1 (Ours)", "prism1", "17 ms · 44.3%", True),
        ("PRISM-Pro (Ours)", "prism_pro", "137 ms · 59.0%", True),
        ("PRISM-Flow (Ours)", "prism_flow", "134 ms · .567 D-S", True),
    ]

    n_rows, n_cols = len(rows_spec), len(cols_spec)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(15.4, 5.85),
        facecolor="white",
        gridspec_kw={"wspace": 0.045, "hspace": 0.055, "left": 0.068, "right": 0.995, "top": 0.885, "bottom": 0.015},
    )

    for r_i, (s_style, stem, t_style, row_lbl) in enumerate(rows_spec):
        src_p = REPO / "data/test" / s_style / f"{stem}.jpg"
        p1_p = PRISM1_DIR / f"{s_style}__{stem}__{t_style}.png"
        pro_p = PRISM_PRO_DIR / f"{s_style}__{stem}__{t_style}.png"
        flow_p = PRISM_FLOW_DIR / f"{s_style}__{stem}__{t_style}.png"
        tid_str = f"{s_style}/{stem}/{t_style}"

        for c_i, (col_title, key, sub_lbl, is_ours) in enumerate(cols_spec):
            ax = axes[r_i, c_i]
            if key is None:
                img_p = src_p
            elif key == "prism1":
                img_p = p1_p
            elif key == "prism_pro":
                img_p = pro_p
            elif key == "prism_flow":
                img_p = flow_p
            elif key == "stylegallery":
                img_p = sg_map[tid_str]
            else:
                img_p = _find_baseline_img(BASE_JPG / key, s_style, stem, t_style)

            ax.imshow(_load_rgb(img_p, 288))
            ax.set_xticks([])
            ax.set_yticks([])
            ec = "#C81D25" if is_ours else "#2C3E50"
            lw = 2.0 if is_ours else 0.8
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_edgecolor(ec)
                sp.set_linewidth(lw)

            if r_i == 0:
                ax.set_title(
                    f"{col_title}\n",
                    fontsize=9.6,
                    weight="bold",
                    color="#C81D25" if is_ours else "#182A42",
                    pad=3,
                )
                ax.text(
                    0.5, 1.03, sub_lbl,
                    transform=ax.transAxes, ha="center", va="bottom",
                    fontsize=7.8, color="#922B21" if is_ours else "#5D6D7E",
                    weight="bold" if is_ours else "normal",
                )
            if c_i == 0:
                ax.text(
                    -0.10, 0.5, row_lbl,
                    transform=ax.transAxes, ha="right", va="center",
                    fontsize=9.3, weight="bold", color="#182A42", rotation=90,
                )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(FIG_DIR / f"fig5_qualitative.{ext}", dpi=260 if ext == "png" else None)
    plt.close(fig)
    print("[Qual] Saved fig5_qualitative.{pdf,png,svg}")


if __name__ == "__main__":
    build_qualitative_grid()
