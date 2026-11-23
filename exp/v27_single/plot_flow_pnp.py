"""Plot the measured Flow-PnP trade-offs from results.json."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def nondominated(rows, metric):
    return sorted(
        (a for a in rows if not any(
            b["dino_c"] >= a["dino_c"] and b[metric] >= a[metric]
            and (b["dino_c"] > a["dino_c"] or b[metric] > a[metric])
            for b in rows
        )),
        key=lambda a: a["dino_c"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--prism", type=Path, help="Overlay a one-step PRISM variant and archived reference points.")
    parser.add_argument("--editor-label", default="PRISM-1+")
    args = parser.parse_args()
    rows = list(json.loads(args.results.read_text(encoding="utf-8"))["arms"].values())
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.25), layout="constrained")
    palette = {"structure": ("#245794", "o", "Structure only"),
               "joint": ("#ab4827", "s", "Structure + style")}
    archived = [
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
        front = nondominated(rows, metric)
        ax.plot([r["dino_c"] for r in front], [scale*r[metric] for r in front],
                color="#2d3540", lw=1.1, ls="--", alpha=.65, zorder=2,
                label="Front of new runs" if args.prism else "Measured Pareto front")
        for mode, (color, marker, label) in palette.items():
            group = sorted((r for r in rows if r["mode"] == mode), key=lambda a: a["s"])
            ax.plot([a["dino_c"] for a in group], [scale*a[metric] for a in group], marker=marker,
                    color=color, alpha=.87, lw=1.4, ms=5, zorder=3, label=label)
            for i, row in enumerate(group):
                if args.prism and mode != "prism":
                    continue
                offset = ((4, 5) if (mode == "joint") != (i % 2 == 0) else (4, -12))
                if mode == "prism" and row["s"] == .4:
                    offset = (-20, -12)
                mark = f"{row['s']:g}" if mode == "prism" else f"{row['s']:.1f}"
                ax.annotate(mark, (row["dino_c"], row[metric]*scale),
                            xytext=offset, textcoords="offset points", color=color, fontsize=8)
        if args.prism:
            for name, dc, ds, csd in archived:
                score = ds if metric == "dino_s" else 100*csd
                ax.scatter(dc, score, marker="x", c="#494955", s=27, zorder=4)
                align = "left" if name == "V23 HiC" else "right"
                offset = (3, 6) if align == "left" else (-4, -14 if name == "R12 γ=.55" else 5)
                ax.annotate(name, (dc, score), xytext=offset, textcoords="offset points",
                            color="#494955", fontsize=7, ha=align)
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
