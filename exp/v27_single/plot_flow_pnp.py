"""Plot the measured Flow-PnP trade-offs from results.json."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.eval.pareto import frontier_indices, is_public_literature


LEDGER = Path(__file__).resolve().parents[2] / "docs/performance/metrics.csv"


def literature_points():
    with LEDGER.open(encoding="utf-8-sig", newline="") as f:
        return [
            {"name": r["arm"], "dino_c": float(r["dino_c"]), "dino_s": float(r["dino_s"]),
             "csd_top1": float(r["target_top1"])}
            for r in csv.DictReader(f)
            if is_public_literature(r["arm"]) and r["protocol"] == "q2a_oracle_cross600"
            and r["n_cross"] == "600" and r["is_diagnostic"].lower() == "false"
            and all(r[k] for k in ("dino_c", "dino_s", "target_top1"))
        ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--prism", type=Path, help="Overlay a one-step PRISM variant and earlier WEAVE points.")
    parser.add_argument("--editor-label", default="PRISM-1+")
    args = parser.parse_args()
    rows = list(json.loads(args.results.read_text(encoding="utf-8"))["arms"].values())
    literature = literature_points()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.25), layout="constrained")
    palette = {"structure": ("#245794", "o", "Structure only"),
               "joint": ("#ab4827", "s", "Structure + style")}
    archived = [
        ("WEAVE", .8079, .4003, .1670),
        ("PRISM-1", .7440, .4431, .4433),
        ("R12 γ=.55", .7723, .4564, .4250),
        ("V23 HiC", .6299, .5330, .5883),
    ]
    if args.prism:
        prism = json.loads(args.prism.read_text(encoding="utf-8"))["arms"]
        rows.extend(dict(a, mode="prism", s=a["strength"]) for a in prism.values()
                    if a["strength"] > 0 and not a.get("wrong", False))
        palette["prism"] = ("#267057", "D", f"{args.editor_label} · 1 NFE")
    for ax, (metric, title, scale) in zip(
        axes, (("dino_s", "DINO-S ↑", 1), ("csd_top1", "CSD Top-1 (%) ↑", 100))
    ):
        public = [(r["dino_c"], scale*r[metric]) for r in literature]
        front = frontier_indices(public)
        if front:
            ax.scatter(*zip(*public), marker="o", s=17, facecolors="none", edgecolors="#565964", zorder=2)
            ax.plot(*zip(*(public[i] for i in front)), color="#565964", lw=1.25,
                    ls="--", marker="o", ms=3, zorder=2, label="External literature Pareto")
            offsets = {"Hist2Style": (-5, -14), "SaMAM": (-5, -15), "CUT": (5, -7),
                       "StyleShot": (5, -11), "StyleGallery": (-6, 8)}
            for i in front:
                name = literature[i]["name"]
                dx, dy = offsets.get(name, (4, 5))
                ax.annotate(name, public[i], xytext=(dx, dy), textcoords="offset points",
                            color="#565964", fontsize=7, ha="right" if dx < 0 else "left")
        for mode, (color, marker, label) in palette.items():
            group = sorted((r for r in rows if r["mode"] == mode), key=lambda a: a["s"])
            ax.plot([a["dino_c"] for a in group], [scale*a[metric] for a in group], marker=marker,
                    color=color, alpha=.87, lw=1.4, ms=5, zorder=3, label=label)
            for i, row in enumerate(group):
                if args.prism and mode != "prism":
                    continue
                if mode == "prism" and row["s"] not in (.2, .25, .4, .45, .8, 1., 1.15):
                    continue
                offset = ((4, 5) if (mode == "joint") != (i % 2 == 0) else (4, -12))
                if mode == "prism":
                    offset = {.4: (-20, -12), .45: (7, 6), 1.0: (-14, 8), 1.15: (-16, -13)}.get(row["s"], offset)
                mark = f"{row['s']:g}" if mode == "prism" else f"{row['s']:.1f}"
                ax.annotate(mark, (row["dino_c"], row[metric]*scale),
                            xytext=offset, textcoords="offset points", color=color, fontsize=8)
        if args.prism:
            for i, (name, dc, ds, csd) in enumerate(archived):
                score = ds if metric == "dino_s" else 100*csd
                ax.scatter(dc, score, marker="x", c="#776275", s=27, zorder=4,
                           label="Earlier WEAVE runs" if i == 0 else None)
                offset = {"V23 HiC": (4, 8), "PRISM-1": (-7, -15),
                          "R12 γ=.55": (7, -13), "WEAVE": (-6, 7)}[name]
                ax.annotate(name, (dc, score), xytext=offset, textcoords="offset points",
                            color="#776275", fontsize=7, ha="right" if offset[0] < 0 else "left")
        ax.grid(alpha=.18, zorder=0)
        ax.set(xlabel="DINO-C ↑", ylabel=title)
        ax.margins(x=.12, y=.17)
    axes[0].legend(loc="best", frameon=False, fontsize=8)
    title = "Single-model trade-offs  ·  Cross600" if args.prism else "Flow-PnP  ·  ICME Q32 6k  ·  Cross600"
    fig.suptitle(title, fontsize=13, fontweight="bold")
    dest = args.prism if args.prism else args.results
    for ext in ("png", "svg"):
        path = dest.with_name("pareto." + ext)
        fig.savefig(path, dpi=220, facecolor="white")
        if ext == "svg":
            lines = path.read_text(encoding="utf-8").splitlines()
            path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
