"""Regenerate fig_pixel_vs_latent_one_column.pdf with a third curve in panel (b).

Panel (a): closed-form affine fit (Pixel vs VAE latent, binary) — unchanged.
Panel (b): convergence of a TinyUNet matched network under three coordinate
spaces — Pixel, VAE latent, VAE latent + spectral (Haar-routed) target.
The spectral target uses LL from content and LH/HL/HH from style
(idwt2_haar(ll_c, lh_s, hl_s, hh_s)); the velocity field is a 4-level
TinyUNet with spatial receptive field, identical across all three spaces.

Output is written under rebuttal_exps/output/ so the user can copy it back
to WEAVE/aaai2027_v4/ to replace the old figure.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


WEAVE_ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = WEAVE_ROOT / "experiments" / "rebuttal" / "unseparated_latent"
MAPPING_JSON = EXP_DIR / "mapping_complexity_results" / "results.json"

# TinyUNet 3-way runs (pixel / latent / latent_spectral), 3 seeds x 30 epochs.
TINYUNET_JSON = (
    WEAVE_ROOT.parent / "SchrodingerBridge" / "results" / "rebuttal"
    / "paper_figure3" / "tinyunet_3way_runs.json"
)

OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    mappings = json.loads(MAPPING_JSON.read_text(encoding="utf-8"))
    runs = json.loads(TINYUNET_JSON.read_text(encoding="utf-8"))

    # ---- Panel (a): closed-form affine fit, Pixel vs VAE latent (binary). ----
    mapping_names = mappings["mapping_order"]
    count = int(next(iter(mappings["results"].values()))["count"])
    pixel_values = [mappings["results"][n]["pixel_mean"] for n in mapping_names]
    latent_values = [mappings["results"][n]["latent_mean"] for n in mapping_names]
    pixel_errors = [mappings["results"][n]["pixel_std"] / np.sqrt(count) for n in mapping_names]
    latent_errors = [mappings["results"][n]["latent_std"] / np.sqrt(count) for n in mapping_names]

    # ---- Panel (b): 3-way TinyUNet convergence. ----
    line_colors = {
        "pixel": "#4C78A8",
        "latent": "#D6452F",
        "latent_spectral": "#70AD47",
    }
    line_labels = {
        "pixel": "Pixel",
        "latent": "VAE latent",
        "latent_spectral": "Latent + spectral",
    }
    curves_by_space: dict[str, dict] = {}
    for run in runs:
        space = run["space"]
        epochs = np.asarray([row["epoch"] for row in run["history"]])
        values = np.asarray([row["val_nmse"] for row in run["history"]])
        if space not in curves_by_space:
            curves_by_space[space] = {"epochs": epochs, "curves": []}
        curves_by_space[space]["curves"].append(values)
    for space in curves_by_space:
        curves_by_space[space]["curves"] = np.asarray(curves_by_space[space]["curves"])

    max_epoch = int(max(c["epochs"][-1] for c in curves_by_space.values()))

    # ---- Compose one-column figure. ----
    with plt.rc_context({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        # Author Kit requires at least 9 pt for text inside figures.
        "font.size": 9.0,
        "axes.titlesize": 9.0,
        "axes.labelsize": 9.0,
        "xtick.labelsize": 9.0,
        "ytick.labelsize": 9.0,
        "legend.fontsize": 9.0,
        "lines.linewidth": 1.15,
    }):
        fig, axes = plt.subplots(1, 2, figsize=(3.45, 2.55))
        x = np.arange(len(mapping_names))
        compact_width = 0.36
        axes[0].bar(x - compact_width / 2, pixel_values, compact_width,
                    yerr=pixel_errors, capsize=1.5, color="#4C78A8", label="Pixel")
        axes[0].bar(x + compact_width / 2, latent_values, compact_width,
                    yerr=latent_errors, capsize=1.5, color="#D6452F", label="VAE latent")
        axes[0].set_xticks(x, mapping_names, rotation=28, ha="right")
        axes[0].set_ylim(0, 0.42)
        axes[0].set_ylabel("Endpoint NMSE")
        axes[0].set_title("(a) Linear fit")

        for space in ("pixel", "latent", "latent_spectral"):
            if space not in curves_by_space:
                continue
            epochs = curves_by_space[space]["epochs"]
            curves = curves_by_space[space]["curves"]
            mean = curves.mean(0)
            std = curves.std(0)
            axes[1].plot(epochs, mean, color=line_colors[space], label=line_labels[space])
            axes[1].fill_between(epochs, mean - std, mean + std,
                                 color=line_colors[space], alpha=0.16)
        axes[1].axhline(1.0, color="#333333", linestyle=(0, (4, 3)), linewidth=0.7)
        axes[1].set_xlim(0, max_epoch)
        axes[1].set_xticks([0, max_epoch // 2, max_epoch])
        axes[1].set_ylim(0, 1.05)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Validation NMSE")
        axes[1].set_title("(b) Convergence")
        handles_a, labels_a = axes[0].get_legend_handles_labels()
        handles_b, labels_b = axes[1].get_legend_handles_labels()
        right_order = ["VAE latent", "Pixel", "Latent + spectral"]
        right_lookup = dict(zip(labels_b, handles_b))
        handles_b = [right_lookup[label] for label in right_order]
        labels_b = right_order
        fig.legend(handles_a, labels_a, loc="lower center", bbox_to_anchor=(0.29, 0.055),
                   ncol=1, frameon=False, handlelength=1.2,
                   handletextpad=0.35, columnspacing=0.8,
                   borderpad=0.0, labelspacing=0.25)
        fig.legend(handles_b, labels_b, loc="lower center", bbox_to_anchor=(0.76, 0.035),
                   ncol=1, frameon=False, handlelength=1.2,
                   handletextpad=0.35, columnspacing=0.8,
                   borderpad=0.0, labelspacing=0.18)
        fig.subplots_adjust(left=0.14, right=0.98, bottom=0.43, top=0.84, wspace=0.55)

        out_pdf = OUTPUT_DIR / "fig_pixel_vs_latent_one_column.pdf"
        out_png = OUTPUT_DIR / "fig_pixel_vs_latent_one_column.png"
        fig.savefig(out_pdf)
        fig.savefig(out_png, dpi=400)
        plt.close(fig)

    print(f"[INFO] Wrote: {out_pdf}")
    print(f"[INFO] Wrote: {out_png}")
    print("\nFinal-epoch validation NMSE (mean over 3 seeds):")
    for space in ("pixel", "latent", "latent_spectral"):
        curves = curves_by_space[space]["curves"]
        final = float(curves[:, -1].mean())
        print(f"  {space:16s}: {final:.4f}")


if __name__ == "__main__":
    main()
