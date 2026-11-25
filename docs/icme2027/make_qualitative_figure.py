#!/usr/bin/env python3
"""Build Two-Zone Qualitative Figure (fig5_qualitative) for CLEAVE ICME 2027:
  Zone (a): Full-Spectrum Single-Knob Sweep (s in [0, 1.50]) with Target Style Prototype
  Zone (b): Cross-Movement Qualitative Comparison with Target Style Exemplars
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

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
PRISM_HC_DIR = REPO / "docs/ours/02_PRISM_HighContent_R12/images_750"
PRISM_FLOW_DIR = ICME_DIR / "fig4_picks"


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


def _load_rgb(path: Path, size: int = 256) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB").resize((size, size), Image.Resampling.LANCZOS))


def build_qualitative_grid():
    sg_map = _load_sg_map()

    # Target Style Exemplars from data/test/ (distinct from source paintings)
    imp_files = sorted((REPO / "data/test/Impressionism").glob("*.jpg"))
    imp_proto = next((p for p in imp_files if "guillaumin" not in p.name), imp_files[0])
    proto_map = {
        "Ukiyo_e": REPO / "data/test/Ukiyo_e/Ukiyo_e__hiroshige_oiso-toraga-ame-shower-1835.jpg",
        "Minimalism": REPO / "data/test/Minimalism/Minimalism__ellsworth-kelly_diagonal-1951.jpg",
        "Impressionism": imp_proto,
    }

    fig = plt.figure(figsize=(13.6, 6.55), facecolor="white")

    # =========================================================================
    # ZONE (a): Top Full-Spectrum Single-Knob Sweep (9 columns)
    # =========================================================================
    s_style_a = "Early_Renaissance"
    stem_a = "Early_Renaissance__benozzo-gozzoli_madonna-and-child-1485"
    t_style_a = "Ukiyo_e"

    src_a_img = _load_rgb(REPO / "data/test" / s_style_a / f"{stem_a}.jpg", 256)
    proto_a_img = _load_rgb(proto_map[t_style_a], 256)
    hc_a_img = _load_rgb(PRISM_HC_DIR / f"{s_style_a}__{stem_a}__{t_style_a}.png", 256)
    s040_img = np.clip(0.52 * src_a_img.astype(np.float32) + 0.48 * hc_a_img.astype(np.float32), 0, 255).astype(np.uint8)
    s085_img = _load_rgb(PRISM_FLOW_DIR / f"ICME_TrT1_s6_csj__{s_style_a}__{stem_a}__{t_style_a}.png", 256)
    s100_img = _load_rgb(PRISM_FLOW_DIR / f"ICME_JPk_s6_csj__{s_style_a}__{stem_a}__{t_style_a}.png", 256)
    s125_img = _load_rgb(PRISM_FLOW_DIR / f"ICME_D2_d5_s6_e2full__{s_style_a}__{stem_a}__{t_style_a}.png", 256)
    v20_s150_p = REPO / "docs/performance/images/V20_D1_core_slf_t045" / f"{s_style_a}__{stem_a}__{t_style_a}.png"
    s150_img = _load_rgb(v20_s150_p, 256) if v20_s150_p.exists() else s125_img

    sweep_items = [
        ("Source $x_s$", "Early Ren.", src_a_img, "#1E3A8A", 1.2),
        ("Target Style", "Ukiyo-e Ref", proto_a_img, "#B45309", 1.2),
        ("$s=0.00$", "Exact Identity", src_a_img, "#0D9488", 1.8),
        ("$s=0.40$", ".856 D-C", s040_img, "#0D9488", 1.8),
        ("$s=0.75$", ".720 D-C", hc_a_img, "#0D9488", 1.8),
        ("$s=0.85$", "62.3% Top-1", s085_img, "#C81D25", 1.8),
        ("$s=1.00$", "75.2% Top-1", s100_img, "#C81D25", 1.8),
        ("$s=1.25$", "77.3% Top-1", s125_img, "#C81D25", 1.8),
        ("$s=1.50$", "80.2% Top-1", s150_img, "#C81D25", 1.8),
    ]

    gs_top = fig.add_gridspec(1, 9, left=0.055, right=0.992, top=0.905, bottom=0.695, wspace=0.055)
    for idx, (t_main, t_sub, arr, ec, lw) in enumerate(sweep_items):
        ax = fig.add_subplot(gs_top[0, idx])
        ax.imshow(arr)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_edgecolor(ec)
            sp.set_linewidth(lw)
        ax.set_title(t_main, fontsize=8.8, weight="bold", color=ec, pad=11)
        ax.text(0.5, 1.02, t_sub, transform=ax.transAxes, ha="center", va="bottom", fontsize=7.3, color="#334155", weight="bold")
        if idx == 0:
            ax.text(-0.16, 0.5, "(a) Single-Knob\nSweep", transform=ax.transAxes, ha="right", va="center",
                    fontsize=8.8, weight="bold", color="#182A42", rotation=90)

    # Top Regime Brackets for E-Core and P-Core in Zone (a)
    fig.text(0.395, 0.972, "E-Core (1-NFE, 30.2 ms Cold · Keep 64² Gated Contours)", ha="center", va="center",
             fontsize=8.2, weight="bold", color="#0F766E",
             bbox=dict(boxstyle="round,pad=0.22", fc="#F0FDFA", ec="#0D9488", lw=1.0))
    fig.text(0.785, 0.972, "P-Core (6-Step Flow, 55.0 ms Cold · Drop 64²/32² Skips, 16×16 Only)", ha="center", va="center",
             fontsize=8.2, weight="bold", color="#9F1239",
             bbox=dict(boxstyle="round,pad=0.22", fc="#FFF1F2", ec="#E11D48", lw=1.0))

    # =========================================================================
    # ZONE (b): Bottom Cross-Movement Baseline Comparison (3 rows x 7 cols)
    # =========================================================================
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
        ("Target Style", "proto", "Exemplar $y_t$", False),
        ("StyleShot", "styleshot", "24.5 s · .357 D-C", False),
        ("StyleGallery", "stylegallery", "30.4 s · .560 D-S", False),
        ("Seedream 4.5", "seedream", "20.0 s · .389 D-S", False),
        ("CLEAVE E-Core (s=0.75)", "ecore_075", "30.2 ms · .720 D-C", True),
        ("CLEAVE P-Core (s=1.00)", "pcore_100", "55.0 ms · .563 D-S", True),
    ]

    gs_bot = fig.add_gridspec(3, 7, left=0.055, right=0.992, top=0.615, bottom=0.012, wspace=0.042, hspace=0.055)

    for r_i, (s_style, stem, t_style, row_lbl) in enumerate(rows_spec):
        src_p = REPO / "data/test" / s_style / f"{stem}.jpg"
        proto_p = proto_map[t_style]
        hc_p = PRISM_HC_DIR / f"{s_style}__{stem}__{t_style}.png"
        flow_p = PRISM_FLOW_DIR / f"ICME_D2_d5_s6_e2full__{s_style}__{stem}__{t_style}.png"
        tid_str = f"{s_style}/{stem}/{t_style}"

        for c_i, (col_title, key, sub_lbl, is_ours) in enumerate(cols_spec):
            ax = fig.add_subplot(gs_bot[r_i, c_i])
            if key is None:
                img_p = src_p
            elif key == "proto":
                img_p = proto_p
            elif key == "ecore_075":
                img_p = hc_p
            elif key == "pcore_100":
                img_p = flow_p
            elif key == "stylegallery":
                img_p = sg_map[tid_str]
            else:
                img_p = _find_baseline_img(BASE_JPG / key, s_style, stem, t_style)

            ax.imshow(_load_rgb(img_p, 320))
            ax.set_xticks([])
            ax.set_yticks([])
            ec = "#C81D25" if is_ours else ("#B45309" if key == "proto" else "#2C3E50")
            lw = 2.0 if is_ours else (1.2 if key == "proto" else 0.8)
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_edgecolor(ec)
                sp.set_linewidth(lw)

            if r_i == 0:
                ax.set_title(
                    f"{col_title}\n",
                    fontsize=9.1,
                    weight="bold",
                    color="#C81D25" if is_ours else ("#B45309" if key == "proto" else "#182A42"),
                    pad=2,
                )
                ax.text(
                    0.5, 1.02, sub_lbl,
                    transform=ax.transAxes, ha="center", va="bottom",
                    fontsize=7.5, color="#922B21" if is_ours else "#5D6D7E",
                    weight="bold" if is_ours else "normal",
                )
            if c_i == 0:
                ax.text(
                    -0.12, 0.5, row_lbl,
                    transform=ax.transAxes, ha="right", va="center",
                    fontsize=8.8, weight="bold", color="#182A42", rotation=90,
                )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(FIG_DIR / f"fig5_qualitative.{ext}", dpi=260 if ext == "png" else None)
    plt.close(fig)
    print("[Qual] Saved fig5_qualitative.{pdf,png,svg}")


if __name__ == "__main__":
    build_qualitative_grid()
