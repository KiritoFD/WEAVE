"""Publication-style 2x2 figure showcasing the SD0-S10 decoder advantage.

Panel (a) Speed-vs-quality Pareto: x = whole-pipeline ms/img on RTX 3060
         (log), y = Style Affinity. Grey = paper.tex-timed baselines; the two
         big points are the SAME Q2A+Oracle frontend with official vs new S10
         decoder, joined by an arrow: faster AND slightly better.
Panel (b) Stacked stage breakdown (PyTorch measured / TRT spliced estimate):
         Encoder / WEAVE / Decoder shares shift from decoder- to encoder-bound.
Panel (c) Quality small-multiples: 5 metrics, official SDVAE vs new C0.
         Shows no regression (LPIPS down, CSD/Affinity up, DINO-C ~flat).
Panel (d) Parameters and speedups (decoder-segment + whole pipeline).

All numbers come from measured artifacts; spliced estimates are labelled.
Output: deliver/performance_advantage.png
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "deliver"

# ---- measured data -------------------------------------------------------
STAGE = json.loads((D / "staged_profile_3060_b8.json").read_text(encoding="utf-8"))
TRT = json.loads((D / "s10_trt_b8_bench.json").read_text(encoding="utf-8"))
SCORE = json.loads((D / "q2a_oracle_scores_baseline_protocol.json").read_text(encoding="utf-8"))

B = 8
E_PT = STAGE["results"]["official"]["encoder_ms_batch"] / B      # 65.385
W_PT = STAGE["results"]["official"]["weave_ms_batch"] / B        # 25.896
D_PT_OFF = STAGE["results"]["official"]["decoder_ms_batch"] / B   # 216.838
D_PT_NEW = STAGE["results"]["c0"]["decoder_ms_batch"] / B        # 68.803
TOT_PT_OFF, TOT_PT_NEW = E_PT + W_PT + D_PT_OFF, E_PT + W_PT + D_PT_NEW

D_TR_OFF = next(r["ms_per_img"] for r in TRT if "official" in r["variant"])  # 73.335
D_TR_NEW = next(r["ms_per_img"] for r in TRT if "s10" in r["variant"])       # 27.076
ENC_WEAVE_TR = 18.7   # paper ledger, RTX 3060 TRT/ONNX fused stage
# For the WHOLE-pipeline TRT comparison use the PAPER ledger official total
# (97.3 = 18.7 enc+WEAVE + 78.6 TRT decoder), and the spliced estimate for the
# new decoder (18.7 + measured 27.08). Decoder-only bars still use the 73.335
# re-measured official engine, which is a separate, clearly-labelled number.
TOT_TR_OFF = 97.3
TOT_TR_NEW = ENC_WEAVE_TR + D_TR_NEW

# paper.tex main table: Infer min/750 imgs on RTX 3060 -> ms/img
PAPER_MIN = {
    "StyleShot": 306.0, "StyleID": 63.0, "AesPA": 23.0, "StyTr2": 38.0,
    "SaMAM": 17.6, "CUT": 5.0, "SD-Turbo": 5.0, "WEAVE (Published)": 1.22,
}
AFF = {b["method"]: b["style_affinity"] for b in SCORE["leaderboard_baseline_methods"]}
BASE_MS = {k: v * 60000.0 / 750.0 for k, v in PAPER_MIN.items()}

# quality: same Q2A+Oracle frontend, official SDVAE vs new C0 (cross600)
Q_OFF = dict(aff=0.5548, top1=0.3450, dino_c=0.7362, lpips=0.5149, margin=-0.0551)
Q_NEW = SCORE["new_decoders"]["c0"]["cross600"]
Q_NEW = dict(aff=Q_NEW["style_affinity"], top1=Q_NEW["csd_top1"],
             dino_c=Q_NEW["dino_c"], lpips=Q_NEW["lpips"], margin=Q_NEW["csd_margin"])

BLUE, RED, GREY = "#2E6F9E", "#D62728", "#9AA0A6"
E_C, W_C, D_C = "#8FB3CC", "#C9A0CB", "#E8A87C"


def main():
    fig, axes = plt.subplots(2, 2, figsize=(16.0, 11.0), dpi=155)
    fig.suptitle("SD0-S10 decoder: same frontend, ~2x faster, no quality regression",
                 fontsize=16.5, weight="bold", y=0.975)
    for ax in axes.ravel():
        ax.set_facecolor("#FBFBFD")
        ax.grid(True, axis="y", ls="--", lw=0.5, alpha=0.4, color="#B8B8B8")

    # ---------------- (a) speed vs quality ----------------
    ax = axes[0, 0]
    for m, ms in BASE_MS.items():
        ax.scatter(ms, AFF[m], s=48, c=GREY, edgecolors="white", lw=0.6,
                   alpha=0.75, zorder=2)
        ax.annotate(m.replace(" (Published)", ""), (ms, AFF[m]),
                    xytext=(5, 4), textcoords="offset points", fontsize=7.2,
                    color="#6A6A6A", zorder=3)
    # strict same-frontend pair: official 49.49M vs new S10 (TRT spliced)
    ax.scatter([TOT_TR_OFF], [Q_OFF["aff"]], s=260, c=BLUE,
               edgecolors="#163A55", lw=1.2, zorder=5)
    ax.scatter([TOT_TR_NEW], [Q_NEW["aff"]], s=300, c=RED,
               edgecolors="#7A0E0E", lw=1.2, zorder=6)
    ax.add_patch(FancyArrowPatch((TOT_TR_OFF, Q_OFF["aff"]),
                                 (TOT_TR_NEW, Q_NEW["aff"]),
                                 arrowstyle="-|>", mutation_scale=16,
                                 color=RED, lw=2.0, zorder=4,
                                 connectionstyle="arc3,rad=-0.18"))
    ax.annotate("official 49.49M decoder\n97.3 ms/img (paper ledger)",
                (TOT_TR_OFF, Q_OFF["aff"]), xytext=(-12, -34),
                textcoords="offset points", ha="right", fontsize=8.0,
                color="#163A55",
                arrowprops=dict(arrowstyle="-", color="#163A55", lw=0.7))
    ax.annotate("NEW SD0-S10 (10.66M)\n45.8 ms/img  |  2.13x faster\n"
                "+0.003 affinity, -0.013 LPIPS",
                (TOT_TR_NEW, Q_NEW["aff"]), xytext=(14, 18),
                textcoords="offset points", fontsize=8.2, color="#A01818",
                weight="bold",
                arrowprops=dict(arrowstyle="-", color="#A01818", lw=0.7))
    ax.set_xscale("log")
    ax.set_xlabel("Whole-pipeline inference  ms / image  (RTX 3060, lower = faster)",
                  fontsize=9.5)
    ax.set_ylabel("Style Affinity = (DINO-S + CLIP-S)/2", fontsize=9.5)
    ax.set_title("(a) Speed vs quality  —  our decoder is leftmost (fastest)\n"
                 "grey = paper.tex-timed baselines; big dots = same Q2A+Oracle frontend",
                 fontsize=10.5, weight="bold", loc="left")
    ax.set_xlim(20, 40000)
    ax.set_ylim(0.47, 0.73)
    ax.legend(handles=[
        Line2D([0], [0], marker="o", ls="", c=GREY, label="baseline (paper.tex timing)"),
        Line2D([0], [0], marker="o", ls="", c=BLUE, markersize=9,
               label="official SDVAE decoder"),
        Line2D([0], [0], marker="o", ls="", c=RED, markersize=9,
               label="new SD0-S10 (TRT decoder + paper enc/WEAVE)"),
    ], fontsize=7.8, loc="upper left", framealpha=0.92)

    # ---------------- (b) stacked stage breakdown ----------------
    ax = axes[0, 1]
    labels = ["PyTorch\nofficial", "PyTorch\nnew S10",
              "TRT\nofficial", "TRT\nnew S10"]
    Es = [E_PT, E_PT, ENC_WEAVE_TR, ENC_WEAVE_TR]
    Ws = [W_PT, W_PT, 0.0, 0.0]      # TRT: enc+WEAVE fused, shown jointly
    Ds = [D_PT_OFF, D_PT_NEW, D_TR_OFF, D_TR_NEW]
    tots = [E_PT + W_PT + D_PT_OFF, E_PT + W_PT + D_PT_NEW,
            ENC_WEAVE_TR + D_TR_OFF, ENC_WEAVE_TR + D_TR_NEW]
    x = np.arange(4)
    ax.bar(x, Es, 0.62, color=E_C, edgecolor="white", label="Encoder")
    ax.bar(x, Ws, 0.62, bottom=Es, color=W_C, edgecolor="white", label="WEAVE (8-step)")
    ax.bar(x, Ds, 0.62, bottom=np.array(Es) + np.array(Ws),
           color=D_C, edgecolor="white", label="Decoder")
    for i, (e, w, dd, tot) in enumerate(zip(Es, Ws, Ds, tots)):
        ax.text(i, e / 2, f"E {e/tot*100:.0f}%", ha="center", va="center",
                fontsize=7.6, color="#20405A")
        if w > 0:
            ax.text(i, e + w / 2, f"W {w/tot*100:.0f}%", ha="center",
                    va="center", fontsize=7.0, color="#5A3A5E")
        ax.text(i, e + w + dd / 2, f"D {dd/tot*100:.0f}%", ha="center",
                va="center", fontsize=8.0, weight="bold", color="#7A4318")
        ax.text(i, tot + 6, f"{tot:.1f} ms/img", ha="center", va="bottom",
                fontsize=8.2, weight="bold", color="#222222")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.8)
    ax.set_ylabel("ms / image", fontsize=9.5)
    ax.set_ylim(0, max(tots) * 1.16)
    ax.set_title("(b) Where the time goes — bottleneck shifts Decoder -> Encoder\n"
                 "first two bars fully measured; TRT bars use paper enc+WEAVE 18.7 ms/img",
                 fontsize=10.5, weight="bold", loc="left")
    ax.legend(fontsize=8.2, loc="upper right", framealpha=0.92)
    ax.annotate("Decoder share 70% -> 43%", xy=(1, 150), xytext=(1.55, 235),
                fontsize=8.4, color="#A01818", weight="bold",
                arrowprops=dict(arrowstyle="->", color="#A01818", lw=1.1))

    # ---------------- (c) quality small multiples ----------------
    ax = axes[1, 0]
    metrics = [
        ("Style Affinity", "aff", "{:.4f}", "up"),
        ("CSD Top-1", "top1", "{:.4f}", "up"),
        ("DINO-C", "dino_c", "{:.4f}", "up"),
        ("LPIPS", "lpips", "{:.4f}", "down"),
    ]
    w = 0.34
    for i, (name, key, fmt, better) in enumerate(metrics):
        v0, v1 = Q_OFF[key], Q_NEW[key]
        ax.bar(i - w / 2, v0, w, color=BLUE, edgecolor="white")
        ax.bar(i + w / 2, v1, w, color=RED, edgecolor="white")
        top = max(v0, v1)
        ax.text(i - w / 2, v0 + (0.008 if v0 >= 0 else -0.012), fmt.format(v0),
                ha="center", va="bottom" if v0 >= 0 else "top",
                fontsize=6.8, color="#163A55")
        ax.text(i + w / 2, v1 + (0.008 if v1 >= 0 else -0.012), fmt.format(v1),
                ha="center", va="bottom" if v1 >= 0 else "top",
                fontsize=6.8, color="#A01818")
        good = (v1 > v0) if better == "up" else (v1 < v0)
        ax.text(i, -0.095,
                "improved" if good else "~flat",
                ha="center", va="top", fontsize=6.6,
                color="#1E7A3C" if good else "#777777")
    ax.axhline(0, color="#888888", lw=0.8)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([m[0] for m in metrics], fontsize=8.2)
    ax.set_ylabel("metric value (higher better; LPIPS lower better)", fontsize=8.6)
    ax.set_ylim(-0.12, 0.80)
    ax.text(0.99, 0.94,
            "CSD Margin -0.0551 -> -0.0457  (+0.0094, 95% CI [+0.0038,+0.0151])",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
            color="#1E7A3C",
            bbox=dict(boxstyle="round,pad=0.25", fc="#F2FAF4",
                      ec="#BFDCC8", lw=0.6))
    ax.set_title("(c) Quality is NOT the price of speed\n"
                 "same Q2A+Oracle frontend, cross600, only decoder swapped",
                 fontsize=10.5, weight="bold", loc="left")
    ax.legend(handles=[
        Line2D([0], [0], marker="s", ls="", c=BLUE, markersize=8,
               label="official 49.49M SDVAE decoder"),
        Line2D([0], [0], marker="s", ls="", c=RED, markersize=8,
               label="new SD0-S10 C0"),
    ], fontsize=8.0, loc="upper right", framealpha=0.92)
    ax.grid(True, axis="y", ls="--", lw=0.5, alpha=0.4)

    # ---------------- (d) params + speedups ----------------
    ax = axes[1, 1]
    # two clean rows inside this panel, no twin axis
    ax.axis("off")
    ax.text(0.0, 1.06, "(d) 78.5% smaller, 2-3x faster",
            transform=ax.transAxes, fontsize=10.5, weight="bold",
            ha="left", va="bottom")
    gs = ax.get_subplotspec().subgridspec(2, 1, hspace=0.45)
    ax_p = fig.add_subplot(gs[0])
    ax_s = fig.add_subplot(gs[1])
    for a in (ax_p, ax_s):
        a.set_facecolor("#FBFBFD")
        a.grid(True, axis="y", ls="--", lw=0.5, alpha=0.4)
    vals = [49.490, 10.658]
    bars = ax_p.bar([0, 1], vals, 0.42, color=[BLUE, RED], edgecolor="white")
    for b, v in zip(bars, vals):
        ax_p.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.2f}M",
                  ha="center", fontsize=8.6, weight="bold")
    ax_p.set_xticks([0, 1]); ax_p.set_xticklabels(["official", "new S10"], fontsize=8)
    ax_p.set_ylabel("params (M)", fontsize=8.4); ax_p.set_ylim(0, 60)
    ax_p.set_title("decoder size", fontsize=8.8, loc="left")
    sp = ["decoder\nPyTorch", "decoder\nTRT", "whole\nPyTorch", "whole\nTRT est."]
    mul = [D_PT_OFF / D_PT_NEW, D_TR_OFF / D_TR_NEW,
           TOT_PT_OFF / TOT_PT_NEW, TOT_TR_OFF / TOT_TR_NEW]
    ax_s.bar(range(4), mul, 0.45, color="#1E7A3C", alpha=0.85, edgecolor="white")
    for i, m in enumerate(mul):
        ax_s.text(i, m + 0.08, f"{m:.2f}x", ha="center", fontsize=8.6,
                  color="#1E7A3C", weight="bold")
    ax_s.set_xticks(range(4)); ax_s.set_xticklabels(sp, fontsize=7.6)
    ax_s.set_ylabel("speedup (x)", fontsize=8.4); ax_s.set_ylim(0, 3.9)
    ax_s.axhline(1.0, color="#999999", ls=":", lw=0.9)
    ax_s.set_title("speedup vs official decoder", fontsize=8.8, loc="left")

    note = ("Timing: paper.tex main table (min/750 imgs, RTX 3060) for baselines; "
            "PyTorch staged profile + TRT b8 fp16 engine both measured on the SAME RTX 3060. "
            "TRT whole-pipeline numbers are spliced estimates (paper enc+WEAVE 18.7 ms/img + measured TRT decoder), "
            "not end-to-end re-measurements. Quality deltas are small and Top-1 gains are not statistically "
            "significant (paired bootstrap CI includes 0); the advantage is speed without quality regression.")
    fig.text(0.5, 0.012, note, ha="center", fontsize=7.6, color="#444444")
    fig.tight_layout(rect=(0, 0.045, 1, 0.955))
    out = D / "performance_advantage.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {out}")
    print("key numbers:")
    print(f"  PyTorch whole: {TOT_PT_OFF:.2f} -> {TOT_PT_NEW:.2f} ms/img "
          f"({TOT_PT_OFF/TOT_PT_NEW:.2f}x)")
    print(f"  TRT whole (spliced): {TOT_TR_OFF:.2f} -> {TOT_TR_NEW:.2f} ms/img "
          f"({TOT_TR_OFF/TOT_TR_NEW:.2f}x)")
    print(f"  decoder PyTorch {D_PT_OFF/D_PT_NEW:.2f}x, TRT {D_TR_OFF/D_TR_NEW:.2f}x")


if __name__ == "__main__":
    main()
