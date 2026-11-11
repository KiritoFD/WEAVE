"""Content-vs-style bubble charts; bubble size = inference speed.

X = content preservation (DINO-C)
Y = style, two definitions:
    (a) CSD Target Top-1
    (b) original DINO+CLIP Style Affinity = (DINO-S + CLIP-S)/2
Bubble area ~ log(inference speed in img/s): faster => larger.

Only methods with a SAME-PROTOCOL timing are shown: the 9 leaderboard baselines
that map to aaai2027_v4/paper.tex main table Train/Infer column (minutes per
750 outputs on one RTX 3060), plus the two new SD0-S10 decoders (TRT decoder
measured + paper enc+WEAVE spliced estimate). Ours/V6 rows without reported
timing and the low-content outliers SaMST / WCT are excluded per request.

Outputs: deliver/bubble_csd.png, deliver/bubble_affinity.png
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
BOARD = ROOT / "deliver/q2a_oracle_scores_baseline_protocol.json"
TRT = ROOT / "deliver/s10_trt_b8_bench.json"
OUT = ROOT / "deliver"

# paper.tex main table: Infer minutes / 750 images / RTX 3060.
# SaMST and WCT deliberately excluded (low-content outliers).
PAPER_INFER_MIN = {
    "StyleShot": 306.0, "StyleID": 63.0, "AesPA": 23.0,
    "StyTr2": 38.0, "SaMAM": 17.6, "CUT": 5.0,
    "SD-Turbo": 5.0,
    "WEAVE (Published)": 1.22,
}
ENC_WEAVE_MS = 18.7   # paper ledger: encoder+WEAVE 18.7 ms/img on RTX 3060
SHORT = {"WEAVE (Published)": "WEAVE"}

# manual label offsets for crowded clusters: (dx, dy, ha)
OFF = {
    "WEAVE": (-9, 12, "right"), "CUT": (9, 9, "left"),
    "SaMAM": (10, -15, "left"), "SD-Turbo": (-9, 11, "right"),
    "StyTr2": (-9, 12, "right"), "WCT": (8, 8, "left"),
    "SD0-S10 C0": (9, 13, "left"), "SD0-S10 C1": (9, -16, "left"),
}


def ms_per_img(min750):
    return min750 * 60000.0 / 750.0


def main():
    d = json.loads(BOARD.read_text(encoding="utf-8"))
    trt = json.loads(TRT.read_text(encoding="utf-8"))
    s10_ms = next(r["ms_per_img"] for r in trt if "s10" in r["variant"])
    new_total_ms = ENC_WEAVE_MS + s10_ms
    new_speed = 1000.0 / new_total_ms

    points = []
    for b in d["leaderboard_baseline_methods"]:
        m = PAPER_INFER_MIN.get(b["method"])
        if m is None:
            continue
        points.append({
            "short": SHORT.get(b["method"], b["method"]),
            "x": b["dino_c"], "csd": b["csd_top1"], "aff": b["style_affinity"],
            "speed": 1000.0 / ms_per_img(m), "kind": "baseline",
        })
    for v, tag in (("c0", "SD0-S10 C0"), ("c1", "SD0-S10 C1")):
        c = d["new_decoders"][v]["cross600"]
        points.append({
            "short": tag, "x": c["dino_c"], "csd": c["csd_top1"],
            "aff": c["style_affinity"], "speed": new_speed, "kind": "new",
        })

    # radius (pt) strictly PROPORTIONAL to speed in img/s.
    # matplotlib scatter `s` is AREA in pt^2, so s = pi * (C*speed)^2.
    C_PT_PER_IMGS = 1.6

    def size(sp):
        return math.pi * (C_PT_PER_IMGS * sp) ** 2

    base = [p for p in points if p["kind"] == "baseline"]
    news = [p for p in points if p["kind"] == "new"]

    for ykey, ylabel, fname, title in (
        ("csd", "CSD Target Top-1  (style)", "bubble_csd.png",
         "Content vs Style (CSD)  |  bubble = inference speed"),
        ("aff", "Style Affinity = (DINO-S + CLIP-S)/2", "bubble_affinity.png",
         "Content vs Style (DINO+CLIP)  |  bubble = inference speed"),
    ):
        fig, ax = plt.subplots(figsize=(14.5, 8.2), dpi=150)
        ax.set_facecolor("#FAFAFA")
        ax.grid(True, ls="--", lw=0.5, alpha=0.45, color="#B0B0B0")

        ax.scatter([p["x"] for p in base], [p[ykey] for p in base],
                   s=[size(p["speed"]) for p in base], c="#2E6F9E", marker="o",
                   edgecolors="#163A55", linewidths=0.9, alpha=0.78, zorder=3,
                   label="Baselines w/ paper.tex Train/Infer (RTX 3060, 750 imgs)")
        ax.scatter([p["x"] for p in news], [p[ykey] for p in news],
                   s=[size(p["speed"]) for p in news], c="#D62728",
                   marker="o", edgecolors="#7A0E0E", linewidths=0.9,
                   alpha=0.62, zorder=5,
                   label="New SD0-S10 (TRT decoder measured; total = spliced est.)")

        for p in base + news:
            if p["short"] == "SD0-S10 C1":
                continue   # coincides with C0; noted in the corner box
            dx, dy, ha = OFF.get(p["short"], (6, 6, "left"))
            ax.annotate(p["short"], (p["x"], p[ykey]),
                        xytext=(dx, dy), textcoords="offset points", ha=ha,
                        fontsize=8.0, alpha=0.94, zorder=6,
                        color="#B01818" if p["kind"] == "new" else "#20303A")

        ax.set_xlabel("Content preservation  DINO-C  (higher = more content kept)",
                      fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=13, weight="bold", pad=12)

        # IDT--TGT sandwich: the two FLOORS bracket the valid transfer region.
        #   TGT (target-style reference) content floor : DINO-C = 0.220 -> vertical
        #   IDT (identity, no transfer) style floor    : horizontal
        # Values from aaai2027_v4/paper.tex main table (TGT D-C, IDT row) and
        # docs/baseline/UNIFIED_FULL_RANKING.md (Identity row, same protocol).
        x_tgt = 0.220
        y_idt = 0.005 if ykey == "csd" else 0.4902
        idt_lbl = ("IDT: CSD Top-1 = 0.005 (style floor)" if ykey == "csd"
                   else "IDT: Style Affinity = 0.4902 (style floor)")
        ax.axvline(x_tgt, color="#6E6E6E", ls=":", lw=1.4, zorder=1)
        ax.axhline(y_idt, color="#6E6E6E", ls=":", lw=1.4, zorder=1)
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        ax.set_xlim(min(xmin, x_tgt - 0.02), xmax)
        ax.set_ylim(min(ymin, y_idt - 0.02), ymax)
        ax.text(x_tgt + 0.006, ymax - 0.02, "TGT: DINO-C = 0.220 (content floor)",
                rotation=90, va="top", ha="left", fontsize=7.6,
                color="#555555", zorder=2)
        ax.text(xmin + 0.012, y_idt + 0.008, idt_lbl,
                va="bottom", ha="left", fontsize=7.6,
                color="#555555", zorder=2)

        ax.text(0.99, 0.015,
                "C0 / C1 nearly overlap\n(same topology & speed)",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7.4,
                color="#B01818", zorder=7,
                bbox=dict(boxstyle="round,pad=0.3", fc="#FFF4F4",
                          ec="#E0B0B0", lw=0.6))
        from matplotlib.lines import Line2D
        hs, ls = ax.get_legend_handles_labels()
        hs.append(Line2D([0], [0], color="#6E6E6E", ls=":", lw=1.4,
                         label="IDT / TGT anchor floors"))
        ls.append("IDT / TGT anchor floors")
        leg = ax.legend(hs, ls, loc="upper left", bbox_to_anchor=(1.012, 1.0),
                        fontsize=8.6, framealpha=0.95, scatterpoints=1,
                        borderpad=0.8, labelspacing=1.0, handletextpad=1.0)
        leg.set_zorder(10)

        note = ("Bubble RADIUS is strictly proportional to inference speed (img/s, linear; "
                "radius = 1.6 pt per img/s); faster = larger, slow methods are tiny by construction. "
                "Timing from aaai2027_v4/paper.tex main table: Infer minutes / 750 outputs / RTX 3060. "
                "Only methods with same-protocol timing are shown; SaMST / WCT excluded as low-content outliers. "
                "New SD0-S10 total = TRT decoder 27.08 ms/img + paper enc+WEAVE 18.7 ms/img "
                "= 45.78 ms/img (spliced estimate, not end-to-end measured).")
        fig.text(0.5, 0.012, note, ha="center", fontsize=7.4, color="#444444")
        fig.tight_layout(rect=(0, 0.065, 0.83, 1))
        out = OUT / fname
        fig.savefig(out, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"[saved] {out}", flush=True)

    print("\nPlotted points (speed img/s | ms/img | DINO-C | CSD | Affinity):")
    for p in sorted(points, key=lambda z: -z["speed"]):
        print("  %-14s %8.2f  %8.2f  %.4f  %.4f  %.4f" % (
            p["short"], p["speed"], 1000 / p["speed"], p["x"], p["csd"], p["aff"]))


if __name__ == "__main__":
    main()
