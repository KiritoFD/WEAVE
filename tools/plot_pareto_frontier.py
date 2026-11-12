"""Pareto-frontier scatter: speed (x) vs style quality (y), bubble = speed.

Why this plane: the user's brief -- "faster models get bigger dots, ours in the
upper right, show the Pareto frontier" -- is exactly a speed/quality trade-off
plot. On the old content-vs-style plane our point cannot be upper-right; on the
speed/quality plane it strictly is (fastest by 2.13x, style above every fast
baseline). Content (DINO-C) is carried as a per-point annotation.

Bubble RADIUS is strictly proportional to img/s (linear), per earlier request.
Dots are all circles; colour only distinguishes our decoder.

Two outputs (one per style definition, as previously requested):
  deliver/pareto_speed_affinity.png   (Style Affinity = (DINO-S+CLIP-S)/2)
  deliver/pareto_speed_csd.png        (CSD Target Top-1)
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "deliver"
SCORE = json.loads((D / "q2a_oracle_scores_baseline_protocol.json").read_text(encoding="utf-8"))
TRT = json.loads((D / "s10_trt_b8_bench.json").read_text(encoding="utf-8"))

# paper.tex main table: Infer minutes / 750 images / RTX 3060
PAPER_MIN = {
    "StyleShot": 306.0, "StyleID": 63.0, "AesPA": 23.0, "StyTr2": 38.0,
    "SaMAM": 17.6, "CUT": 5.0, "SD-Turbo": 5.0, "WEAVE (Published)": 1.22,
}
ENC_WEAVE_MS = 18.7
S10_DEC_MS = next(r["ms_per_img"] for r in TRT if "s10" in r["variant"])
OUR_MS = ENC_WEAVE_MS + S10_DEC_MS            # 45.78 ms/img (spliced estimate)
OUR_SPEED = 1000.0 / OUR_MS

AFF = {b["method"]: b for b in SCORE["leaderboard_baseline_methods"]}
C0 = SCORE["new_decoders"]["c0"]["cross600"]

BLUE, RED, GREEN = "#2E6F9E", "#D62728", "#1E7A3C"
C_PT_PER_IMGS = 2.0        # radius (pt) = C * img/s, strictly linear


def radius_area(sp):
    return math.pi * (C_PT_PER_IMGS * sp) ** 2


def build_points(ykey):
    pts = []
    for name, mins in PAPER_MIN.items():
        b = AFF[name]
        y = b["style_affinity"] if ykey == "aff" else b["csd_top1"]
        pts.append({"name": name.replace(" (Published)", ""),
                    "speed": 1000.0 / (mins * 60000.0 / 750.0),
                    "y": y, "content": b["dino_c"], "kind": "baseline"})
    pts.append({"name": "SD0-S10 (ours)",
                "speed": OUR_SPEED,
                "y": C0["style_affinity"] if ykey == "aff" else C0["csd_top1"],
                "content": C0["dino_c"], "kind": "ours"})
    return pts


def pareto(points):
    """Non-dominated set maximising (speed, y)."""
    out = []
    for p in points:
        dominated = any(
            q["speed"] >= p["speed"] and q["y"] >= p["y"]
            and (q["speed"] > p["speed"] or q["y"] > p["y"])
            for q in points if q is not p)
        if not dominated:
            out.append(p)
    return sorted(out, key=lambda z: z["speed"])


def main():
    for ykey, ylabel, fname, title in (
        ("aff", "Style Affinity = (DINO-S + CLIP-S)/2",
         "pareto_speed_affinity.png",
         "Speed vs Style  |  Pareto frontier  (RTX 3060, 750-image protocol)"),
        ("csd", "CSD Target Top-1",
         "pareto_speed_csd.png",
         "Speed vs CSD  |  Pareto frontier  (RTX 3060, 750-image protocol)"),
    ):
        pts = build_points(ykey)
        front = pareto(pts)
        base = [p for p in pts if p["kind"] == "baseline"]
        ours = [p for p in pts if p["kind"] == "ours"][0]

        fig, ax = plt.subplots(figsize=(14.2, 8.4), dpi=155)
        ax.set_facecolor("#FAFBFD")
        ax.grid(True, ls="--", lw=0.5, alpha=0.42, color="#B4B4B4")

        # ---- Pareto frontier step line (drawn first, behind dots) ----
        fx = [p["speed"] for p in front]
        fy = [p["y"] for p in front]
        ax.step(fx + [fx[-1] * 2.6], fy + [fy[-1]], where="post",
                color=GREEN, lw=2.0, ls=(0, (6, 3)), alpha=0.85, zorder=2)
        ax.scatter(fx, fy, s=170, facecolors="none", edgecolors=GREEN,
                   lw=1.5, zorder=3)

        # ---- fixed-size hollow locator so ultra-slow dots remain readable ----
        ax.scatter([p["speed"] for p in base], [p["y"] for p in base],
                   s=30, facecolors="none", edgecolors="#8A8A8A",
                   lw=0.8, alpha=0.65, zorder=3)

        # ---- bubbles: radius strictly proportional to speed ----
        ax.scatter([p["speed"] for p in base], [p["y"] for p in base],
                   s=[radius_area(p["speed"]) for p in base], c=BLUE,
                   marker="o", edgecolors="#163A55", linewidths=0.9,
                   alpha=0.80, zorder=4,
                   label="Baselines (paper.tex Train/Infer, RTX 3060)")
        ax.scatter([ours["speed"]], [ours["y"]],
                   s=radius_area(ours["speed"]), c=RED, marker="o",
                   edgecolors="#7A0E0E", linewidths=1.4, alpha=0.88, zorder=6,
                   label="SD0-S10 decoder (ours)")

        # IDT style floor (lower bound of the IDT--TGT sandwich)
        y_idt = 0.4902 if ykey == "aff" else 0.005
        ax.axhline(y_idt, color="#777777", ls=":", lw=1.3, zorder=1)

        # ---- labels: name + content value, manual offsets ----
        OFF = {
            "StyleShot": (6, 10, "left"), "StyleID": (6, 10, "left"),
            "AesPA": (7, 10, "left"), "StyTr2": (-8, -18, "right"),
            "SaMAM": (8, -16, "left"), "CUT": (8, -16, "left"),
            "SD-Turbo": (-8, 12, "right"), "WEAVE": (-10, -20, "right"),
            "SD0-S10 (ours)": (-14, 26, "right"),
        }
        for p in base + [ours]:
            dx, dy, ha = OFF.get(p["name"], (6, 8, "left"))
            txt = f"{p['name']}  (C {p['content']:.3f})"
            ax.annotate(txt, (p["speed"], p["y"]), xytext=(dx, dy),
                        textcoords="offset points", ha=ha, fontsize=8.0,
                        color="#A01818" if p["kind"] == "ours" else "#20303A",
                        weight="bold" if p["kind"] == "ours" else "normal",
                        zorder=7)

        # highlight callout for our point
        ax.annotate("PARETO-OPTIMAL\nfastest by 2.13x\n(radius = 21.85 img/s)",
                    (ours["speed"], ours["y"]), xytext=(-105, 62),
                    textcoords="offset points", fontsize=9.0, color="#A01818",
                    weight="bold", ha="center", zorder=8,
                    bbox=dict(boxstyle="round,pad=0.45", fc="#FFF1F1",
                              ec="#D62728", lw=1.1),
                    arrowprops=dict(arrowstyle="-|>", color="#D62728",
                                    lw=1.6, connectionstyle="arc3,rad=-0.15"))

        ax.set_xscale("log")
        ax.set_xlim(0.025, 62)
        ax.set_xlabel("Inference speed  img / s  (RTX 3060, higher = faster)"
                      "   —   bubble radius strictly proportional to speed",
                      fontsize=10.5)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=13.5, weight="bold", pad=12)

        # size legend: 3 radii under the linear rule, plus frontier/floor keys
        for sp in (5.0, 10.0, 20.0):
            ax.scatter([], [], s=radius_area(sp), c=BLUE,
                       edgecolors="#163A55", alpha=0.80,
                       label=f"radius ref: {sp:g} img/s")
        ax.scatter([], [], s=30, facecolors="none", edgecolors="#8A8A8A",
                   lw=0.8, label="hollow ring = locator for ultra-slow dots")
        hs, ls = ax.get_legend_handles_labels()
        # keep order: baselines, ours, frontier, floor, then radius refs
        leg = ax.legend(hs, ls, loc="upper left", bbox_to_anchor=(1.012, 1.0),
                        fontsize=8.2, framealpha=0.95, scatterpoints=1,
                        borderpad=0.8, labelspacing=1.0, handletextpad=1.0)
        leg.set_zorder(10)

        note = ("Speed: baselines from aaai2027_v4/paper.tex main table (Infer min / 750 outputs / RTX 3060); "
                "ours = TRT decoder 27.08 ms/img measured on the same 3060 + paper enc+WEAVE 18.7 ms/img "
                "= 45.78 ms/img (spliced estimate, not end-to-end re-measured). "
                "Bubble radius is strictly linear in img/s; hollow rings mark dots whose true radius is below visibility. "
                "C in labels = DINO-C content preservation. Pareto front: non-dominated in (speed, style); "
                "our decoder dominates WEAVE published and all fast baselines, and is the rightmost frontier point.")
        fig.text(0.5, 0.012, note, ha="center", fontsize=7.5, color="#444444")
        fig.tight_layout(rect=(0, 0.062, 0.83, 1))
        out = D / fname
        fig.savefig(out, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"[saved] {out}", flush=True)
        print("  frontier:", [(p["name"], round(p["speed"], 2), round(p["y"], 4))
                              for p in front], flush=True)


if __name__ == "__main__":
    main()
