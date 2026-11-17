"""Pareto scatter: published baselines + WEAVE/V6 + V8 arms + V15, dino_s x dino_c.

Copied from archives/_tmp_20261114/make_pareto_baseline.py (the script that
produced docs/v8/pareto_baseline.png) and extended:
- baselines += our Hist2Style / StyleGallery 750-task reproductions
- ours    += V15 Group A probe cells (runs/v15/eval/W_*.json)
- idt / tgt reference floor lines (same convention as
  docs/v8/plot_v8_style_content_speed.py)
- two outputs: _raw (auto limits) and _zoom (left/bottom cropped)

Baselines parsed from docs/baseline/UNIFIED_FULL_RANKING.md section 一
(same Cross600 protocol: tasks_750 600 cross tasks). V8 arms from
docs/v8/results.csv.
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RANK = REPO / "docs/baseline/UNIFIED_FULL_RANKING.md"
CSV = REPO / "docs/v8/results.csv"
V15_GLOB = REPO / "runs/v15/eval/W_*.json"

# reproduced external baselines (cross600, our eval pipeline)
REPRO = [("Hist2Style", 0.3643, 0.9083), ("StyleGallery", 0.5604, 0.6113)]

# reference floors (docs/v8/plot_v8_style_content_speed.py convention)
IDT_STYLE = 0.3633    # Identity row dino_s, UNIFIED_FULL_RANKING
TGT_CONTENT = 0.2190  # tgt reference as output, dino_c mean

LEVEL_COLORS = {0.0: "#4C78A8", 0.4: "#F58518", 0.7: "#54A24B"}


def parse_rankings():
    txt = RANK.read_text(encoding="utf-8")
    sec = txt.split("## 一、")[1].split("## 二、")[0]
    rows = []
    for line in sec.splitlines():
        if not line.startswith("|") or "---" in line or "排名" in line:
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 10:
            continue
        name = re.sub(r"\*\*", "", cells[1])
        cat = cells[2]
        try:
            rows.append({
                "name": name, "category": cat,
                "dino_s": float(cells[5]), "clip_s": float(cells[6]),
                "dino_c": float(cells[7]), "lpips": float(cells[8]),
            })
        except ValueError:
            continue
    return pd.DataFrame(rows)


def load_v15():
    rows = []
    for f in sorted(V15_GLOB.parent.glob(V15_GLOB.name)):
        for r in json.loads(f.read_text(encoding="utf-8"))["rows"]:
            if r["cross"]:
                rows.append({"arm": f.stem, "level": r["level"],
                             "dino_s": r["dino_s"], "dino_c": r["dino_c"]})
    df = pd.DataFrame(rows)
    return df.groupby(["arm", "level"], as_index=False).mean(numeric_only=True)


def pareto(df, x="dino_s", y="dino_c"):
    pts = df[[x, y]].values
    keep = []
    for i, (xi, yi) in enumerate(pts):
        dom = any(xj >= xi and yj >= yi and (xj > xi or yj > yi)
                  for j, (xj, yj) in enumerate(pts) if j != i)
        if not dom:
            keep.append(i)
    return sorted(keep, key=lambda i: df[x].values[i])


def render(tag, xlim=None, ylim=None):
    base = parse_rankings()
    repro = pd.DataFrame(
        [{"name": n, "category": "Baseline", "dino_s": s, "dino_c": c,
          "clip_s": float("nan"), "lpips": float("nan")}
         for n, s, c in REPRO])
    base = pd.concat([base, repro], ignore_index=True)

    v8 = pd.read_csv(CSV)
    v8 = v8[v8.split == "cross"][["arm", "dino_s", "dino_c", "lpips"]].rename(
        columns={"arm": "name"})
    v8["category"] = "V8"
    v8["clip_s"] = float("nan")

    v15 = load_v15().rename(columns={"arm": "name"})
    v15["name"] = "V15:" + v15["name"] + "@t" + v15["level"].map(lambda v: f"{v:g}")
    v15["category"] = "V15"
    v15["clip_s"] = float("nan")
    v15["lpips"] = float("nan")
    v15["arm"] = v15["name"].str.split(":").str[1].str.rsplit("@").str[0]

    df = pd.concat([base, v8, v15], ignore_index=True)
    is_pub = df.category.isin(["Baseline", "Ours"])
    front_pub = pareto(df[is_pub])
    pub_idx = df[is_pub].index

    fig, ax = plt.subplots(figsize=(11, 7.5))
    b = df[is_pub]
    v8d = df[df.category == "V8"]
    v15d = df[df.category == "V15"]
    ax.scatter(b.dino_s, b.dino_c, c="#999999", s=55, alpha=0.8,
               label="published baselines / V6-V9 + repro", zorder=2)
    ax.scatter(v8d.dino_s, v8d.dino_c, c="#d62728", s=60, alpha=0.85,
               label="V8 arms (this work)", zorder=3)
    ax.scatter(v15d.dino_s, v15d.dino_c, c="#7b2cbf", s=55, alpha=0.8,
               marker="D", label="V15 Group A (this work)", zorder=3)

    fp = df.loc[sorted(pub_idx[i] for i in front_pub)].sort_values("dino_s")
    ax.plot(fp.dino_s, fp.dino_c, "k--", lw=1.4, zorder=1,
            label="baseline Pareto front")

    # idt / tgt reference floors: x=dino_s, y=dino_c
    ax.axvline(IDT_STYLE, color="#C0392B", lw=1.2, ls=(0, (6, 4)),
               alpha=.8, zorder=1)
    ax.text(IDT_STYLE, 1.01, "idt style floor", fontsize=8,
            color="#C0392B", ha="center", va="bottom", style="italic")
    ax.axhline(TGT_CONTENT, color="#C0392B", lw=1.2, ls=(0, (6, 4)),
               alpha=.8, zorder=1)
    x_left = (xlim[0] + 0.008) if xlim else 0.155
    ax.text(x_left, TGT_CONTENT, "tgt content floor", fontsize=8,
            color="#C0392B", ha="left", va="center", style="italic")

    # label baseline frontier vertices + key ours points
    for _, r in fp.iterrows():
        ax.annotate(r["name"].replace("**", ""), (r.dino_s, r.dino_c),
                    fontsize=7, xytext=(3, 3), textcoords="offset points")
    key = ["L1.b4.r15", "L1.b4.r25", "L1.topk", "L2.e", "I2", "A2b",
           "N2C.e", "V11B.H2.t035"]
    for _, r in v8d[v8d.name.isin(key)].iterrows():
        ax.annotate(r["name"], (r.dino_s, r.dino_c), fontsize=8,
                    fontweight="bold", xytext=(0, -13),
                    textcoords="offset points", ha="center", color="#d62728")
    # label V15 dim96 cells
    for _, r in v15d[v15d.arm.str.contains("dim96")].iterrows():
        ax.annotate(r["arm"].replace("W_", "").replace("_", "·"),
                    (r.dino_s, r.dino_c), fontsize=6.5, color="#7b2cbf",
                    xytext=(4, 4), textcoords="offset points", alpha=.85)

    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xlabel("DINO-S (style similarity) ↑")
    ax.set_ylabel("DINO-C (content preservation) ↑")
    ax.set_title(f"Style–content Pareto: published baselines vs V8 + V15 "
                 f"(Cross600){' — zoom' if tag == '_zoom' else ''}")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()

    for ext in ("png", "svg", "pdf"):
        fig.savefig(HERE / f"dino_scatter_v15{tag}.{ext}",
                    bbox_inches="tight", pad_inches=.15,
                    dpi=200 if ext == "png" else None)
    plt.close(fig)
    print(f"[{tag}] baseline front:", list(fp["name"]))


def main():
    render("_raw")
    render("_zoom", xlim=(0.30, 0.62), ylim=(0.20, 1.05))
    print("saved docs/v15/dino_scatter_v15_{raw,zoom}.{png,svg,pdf}")


if __name__ == "__main__":
    main()
