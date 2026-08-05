from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "docs" / "exp" / "signal_level" / "results.json"
GRADIENTS = HERE.parent / "docs" / "exp" / "signal_level" / "gradient_hegemony.json"


def main() -> None:
    payload = json.loads(RESULTS.read_text(encoding="utf-8"))
    gradient_payload = json.loads(GRADIENTS.read_text(encoding="utf-8"))
    bands = ["LL", "LH", "HL", "HH"]
    energy = payload["experiment_a_energy"]["subbands"]
    style = payload["experiment_c_fstat"]["subbands"]

    means = np.asarray([energy[band]["mean"] for band in bands]) * 100
    p5 = np.asarray([energy[band]["p5"] for band in bands]) * 100
    p95 = np.asarray([energy[band]["p95"] for band in bands]) * 100
    eta2 = np.asarray([style[band]["eta_squared"] for band in bands])
    errors = np.vstack((means - p5, p95 - means))
    gradient_rows = {row["label"]: row for row in gradient_payload["results"]}
    weighted = gradient_rows["w03_weave"]
    uniform = gradient_rows["w10_global_mse"]

    colors = ["#34495E", "#2EAD64", "#3288BD", "#C83A2B"]
    with plt.rc_context({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "font.size": 6.2,
        "axes.titlesize": 6.8,
        "axes.labelsize": 6.4,
        "xtick.labelsize": 6.0,
        "ytick.labelsize": 5.8,
        "axes.linewidth": 0.65,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }):
        fig, axes = plt.subplots(2, 2, figsize=(3.45, 2.72))
        axes = axes.ravel()
        x = np.arange(len(bands))

        axes[0].bar(x, means, yerr=errors, color=colors, width=0.68,
                    capsize=1.7, edgecolor="white", linewidth=0.4)
        axes[0].set_xticks(x, bands)
        axes[0].set_ylim(0, 100)
        axes[0].set_ylabel("Energy share (%)")
        axes[0].set_title("(a) Latent energy")

        bars = axes[1].bar(x, eta2, color=colors, width=0.68,
                           edgecolor="white", linewidth=0.4)
        axes[1].set_xticks(x, bands)
        axes[1].set_ylim(0, 0.56)
        axes[1].set_ylabel(r"Style effect size $\eta^2$")
        axes[1].set_title("(b) Style separability")
        for bar, value in zip(bars, eta2):
            axes[1].text(bar.get_x() + bar.get_width() / 2, value + 0.014,
                         f"{value:.2f}", ha="center", va="bottom", fontsize=5.3)

        group_x = np.arange(2)
        width = 0.34
        loss_weighted = [weighted["internal_probe_loss_ll"], weighted["internal_probe_loss_hf"]]
        loss_uniform = [uniform["internal_probe_loss_ll"], uniform["internal_probe_loss_hf"]]
        axes[2].bar(group_x - width / 2, loss_weighted, width, color="#4C78A8",
                    label=r"$\lambda_{LL}=0.3$")
        axes[2].bar(group_x + width / 2, loss_uniform, width, color="#E45756",
                    label=r"$\lambda_{LL}=1.0$")
        axes[2].set_yscale("log")
        axes[2].set_ylim(1e-9, 10)
        axes[2].set_xticks(group_x, ["LL", "HF"])
        axes[2].set_ylabel("Band loss")
        axes[2].set_title("(c) Terminal residual")
        axes[2].legend(frameon=False, fontsize=4.8, handlelength=1.0,
                       borderaxespad=0.2, loc="upper left")

        grad_weighted = [weighted["internal_probe_shared_ll_grad_norm"],
                         weighted["internal_probe_shared_hf_grad_norm"]]
        grad_uniform = [uniform["internal_probe_shared_ll_grad_norm"],
                        uniform["internal_probe_shared_hf_grad_norm"]]
        axes[3].bar(group_x - width / 2, grad_weighted, width, color="#4C78A8")
        axes[3].bar(group_x + width / 2, grad_uniform, width, color="#E45756")
        axes[3].set_yscale("log")
        axes[3].set_ylim(1e-8, 10)
        axes[3].set_xticks(group_x, ["LL", "HF"])
        axes[3].set_ylabel(r"Shared $\|g\|_2$")
        axes[3].set_title("(d) Terminal gradient")

        for ax in axes:
            ax.grid(axis="y", color="#D9D9D9", linewidth=0.4, alpha=0.75)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(pad=1)

        fig.subplots_adjust(left=0.13, right=0.985, bottom=0.09, top=0.91,
                            wspace=0.52, hspace=0.62)
        fig.savefig(HERE / "fig_signal_level.pdf")
        fig.savefig(HERE / "fig_signal_level.png", dpi=400)
        plt.close(fig)


if __name__ == "__main__":
    main()
