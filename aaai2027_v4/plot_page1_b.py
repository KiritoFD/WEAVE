"""Build the first-page ArtFID figure (b) for the AAAI 2027 draft.

Split out from plot_page1_summary.py so each page-1 panel has its own script.
Generates: fig_page1_b.png  and  fig_page1_b.svg (draggable labels, preview only).
Used by paper.tex as Figure 2.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR
FIG_DATA = OUT_DIR / "fig_data"

plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 10,
        "legend.fontsize": 8.4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
        "figure.dpi": 300,
    }
)


def point(
    name: str,
    dino_s: float,
    clip_s: float,
    lpips: float,
    group: str,
    *,
    label: bool = False,
    display: str | None = None,
    train_min: float | None = None,
) -> dict:
    avg = 0.5 * (dino_s + clip_s)
    return {
        "name": name,
        "display": display or name,
        "dino_s": dino_s,
        "clip_s": clip_s,
        "avg": avg,
        "lpips": lpips,
        "x": 1.0 - lpips,
        "group": group,
        "label": label,
        "train_min": train_min,
    }


# All D5-512 methods with both style affinity (mean of DINO-S and CLIP-S; DINO-S from
# fig_data/dino_main.json, CLIP-S from the main table) and target-pooled ArtFID
# (recomputed 2026-07-23 from results/_artfid_details.json, n=750). TGT is kept as an
# optional content anchor (pass --with-tgt to include it in the panel).
STYLE_ARTFID = [
    {"name": "IDT",          "style": 0.5 * (0.4185 + 0.6399), "artfid": 216.5097, "color": "#111111"},
    {"name": "WEAVE",        "style": 0.5 * (0.4918 + 0.7128), "artfid": 295.2679, "color": "#D6452F"},
    {"name": "SaMam",        "style": 0.5 * (0.4771 + 0.7025), "artfid": 297.3159, "color": "#3B82C4"},
    {"name": "Seedream",     "style": 0.5 * (0.4864 + 0.7200), "artfid": 310.9730, "color": "#8464B3"},
    {"name": "CUT",          "style": 0.5 * (0.4709 + 0.7140), "artfid": 314.36,   "color": "#B0B7C3"},
    {"name": "Z-STAR",       "style": 0.5 * (0.4488 + 0.7840), "artfid": 332.9112, "color": "#7DD3A8"},
    {"name": "StyleID",      "style": 0.5 * (0.5477 + 0.8220), "artfid": 322.87,   "color": "#7DD3A8"},
    {"name": "StyleAligned", "style": 0.5 * (0.6751 + 0.7800), "artfid": 368.6293, "color": "#F28E2B"},
    {"name": "StyleShot",    "style": 0.5 * (0.5630 + 0.7870), "artfid": 370.33,   "color": "#7DD3A8"},
    {"name": "AesPA",        "style": 0.5 * (0.5200 + 0.7240), "artfid": 376.21,   "color": "#B0B7C3"},
    {"name": "StyTR-2",      "style": 0.5 * (0.5010 + 0.7090), "artfid": 382.65,   "color": "#B0B7C3"},
    {"name": "TGT",          "style": 0.5 * (1.0000 + 0.8550), "artfid": 545.6990, "color": "#6B7280"},
]


def pearson_corr(xs: list[float], ys: list[float]) -> float:
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_den = math.sqrt(sum((x - x_mean) ** 2 for x in xs))
    y_den = math.sqrt(sum((y - y_mean) ** 2 for y in ys))
    return num / (x_den * y_den)


def build_style_artfid(ax: plt.Axes, *, include_tgt: bool = False) -> None:
    rows = STYLE_ARTFID if include_tgt else [item for item in STYLE_ARTFID if item["name"] != "TGT"]
    points = []
    for item in rows:
        marker = "X" if item["name"] == "TGT" else "o"
        size = 62 if item["name"] == "WEAVE" else 42
        edge = "#7F1F10" if item["name"] == "WEAVE" else "white"
        h = ax.scatter(
            item["style"],
            item["artfid"],
            s=size,
            marker=marker,
            facecolor=item["color"],
            edgecolor=edge,
            linewidth=1.0,
            zorder=3,
        )
        points.append(h)

    # Manual label placement with leader lines. Each label is pushed off its data
    # point along a fixed direction (so text never sits on top of the marker) and a
    # thin gray line links it back. Directions are chosen to fan dense clusters out.
    LABEL_OFFSETS = {
        "IDT": (-28, -6),
        "WEAVE": (8, -18),
        "SaMam": (-34, 4),
        "Seedream": (34, 2),
        "CUT": (34, -12),
        "Z-STAR": (-6, 17),
        "StyleID": (8, 16),
        "StyleAligned": (28, 4),
        "StyleShot": (2, 16),
        "AesPA": (-2, -17),
        "StyTR-2": (-28, -15),
        "TGT": (0, 15),
    }
    from matplotlib.transforms import offset_copy

    for item in rows:
        label = "SaMam" if item["name"] == "SaMam" else item["name"]
        dx, dy = LABEL_OFFSETS[item["name"]]
        dy = -dy  # mirror the vertical label offset with the inverted y-axis
        trans = offset_copy(ax.transData, fig=ax.figure, x=dx, y=dy, units="points")
        ax.text(
            item["style"],
            item["artfid"],
            label,
            transform=trans,
            ha="left" if dx >= 0 else "right",
            va="bottom" if dy >= 0 else "top",
            fontsize=6.6,
            color="#2B2F38",
            zorder=5,
            gid="draggable",
        )

    # Correlation uses IDT and the evaluated methods; random TGT is shown only
    # as an anchor because its reference image is part of the target pool.
    corr_rows = [item for item in rows if item["name"] != "TGT"]
    style_values = [item["style"] for item in corr_rows]
    artfid_values = [item["artfid"] for item in corr_rows]
    r_quality = pearson_corr(
        style_values,
        [-value for value in artfid_values],
    )
    style_mean = sum(style_values) / len(style_values)
    artfid_mean = sum(artfid_values) / len(artfid_values)
    trend_slope = sum(
        (style - style_mean) * (artfid - artfid_mean)
        for style, artfid in zip(style_values, artfid_values)
    ) / sum((style - style_mean) ** 2 for style in style_values)
    trend_intercept = artfid_mean - trend_slope * style_mean
    trend_x = [min(style_values), max(style_values)]
    ax.plot(
        trend_x,
        [trend_slope * style + trend_intercept for style in trend_x],
        color="#6B7280",
        linewidth=1.15,
        linestyle=(0, (4, 3)),
        alpha=0.8,
        zorder=1.5,
    )
    ax.text(
        0.97, 0.95,
        rf"$R^2$ = {r_quality ** 2:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.0,
        color="#6B7280",
        gid="draggable",
    )
    ax.text(
        0.97,
        0.05,
        rf"$r(\mathrm{{style}},-\mathrm{{ArtFID}})={r_quality:.2f}$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.0,
        color="#5B6270",
        gid="draggable",
    )
    all_x = [item["style"] for item in rows]
    all_y = [item["artfid"] for item in rows]
    x_pad = 0.03
    y_pad = 22
    ax.set_xlim(min(all_x) - x_pad, max(all_x) + x_pad)
    # Invert the y-axis so the lowest ArtFID (IDT, the optimal point) sits at the top.
    ax.set_ylim(max(all_y) + y_pad, min(all_y) - y_pad)
    ax.set_xlabel("Mean style affinity")
    ax.set_ylabel("ArtFID (lower is better)")
    ax.set_title("ArtFID vs. Style Affinity", fontweight="bold", pad=3)
    ax.grid(axis="both", color="#D6D9DF", alpha=0.55, linewidth=0.6)


def _inject_drag_script(svg_path: "Path") -> None:
    """Make every <text gid=\"method-label\"> draggable with the mouse in a browser.

    The figure is saved as plain SVG (each label is its own <text> element); we
    append a small inline script that lets the user click-drag those labels to
    fine-tune positions that the static layout could not resolve."""
    DRAG_JS = r"""<script type="text/javascript"><![CDATA[
(function () {
  function bind(svg) {
    var texts = svg.querySelectorAll('text');
    for (var i = 0; i < texts.length; i++) {
      var t = texts[i];
      t.style.cursor = "move";
      t.addEventListener("mousedown", startDrag);
    }
  }
  function getXY(t) {
    if (t.getAttribute("x") !== null) {
      return { x: parseFloat(t.getAttribute("x")), y: parseFloat(t.getAttribute("y")) };
    }
    var tr = t.getAttribute("transform") || "";
    var m = /translate\(\s*([-\\d.]+)[ ,]+([-\\d.]+)\s*\)/.exec(tr);
    if (m) return { x: parseFloat(m[1]), y: parseFloat(m[2]) };
    return { x: 0, y: 0 };
  }
  function setXY(t, x, y) {
    if (t.getAttribute("x") !== null) {
      t.setAttribute("x", x); t.setAttribute("y", y);
    } else {
      t.setAttribute("transform", "translate(" + x + "," + y + ")");
    }
  }
  function startDrag(e) {
    e.preventDefault();
    var t = e.target;
    var svg = t.ownerSVGElement;
    var m = t.getScreenCTM();
    var pt = svg.createSVGPoint();
    pt.x = e.clientX; pt.y = e.clientY;
    var loc = pt.matrixTransform(m.inverse());
    var base = getXY(t);
    var ox = loc.x - base.x, oy = loc.y - base.y;
    function move(ev) {
      var p2 = svg.createSVGPoint();
      p2.x = ev.clientX; p2.y = ev.clientY;
      var l2 = p2.matrixTransform(m.inverse());
      setXY(t, l2.x - ox, l2.y - oy);
    }
    function up() {
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", up);
    }
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
  }
  var svgs = document.getElementsByTagName("svg");
  for (var s = 0; s < svgs.length; s++) bind(svgs[s]);
})();
]]></script>
"""
    svg = svg_path.read_text(encoding="utf-8")
    if "</svg>" in svg and "<text" in svg:
        svg = svg.replace("</svg>", DRAG_JS + "\n</svg>", 1)
        svg_path.write_text(svg, encoding="utf-8")


def save_single_panel(
    kind: str,
    fname: str,
    figsize,
    *,
    with_tgt: bool = False,
    drop_ylabel: bool = True,
    title: str | None = None,
) -> None:
    """Save one panel as a standalone single-column figure (ArtFID only)."""
    if kind != "artfid":
        raise ValueError(f"plot_page1_b.py only builds the ArtFID panel, got kind={kind!r}")
    times_rc = {
        "font.family": "Times New Roman",
        "mathtext.fontset": "stix",
    }
    with plt.rc_context(times_rc):
        fig, ax = plt.subplots(figsize=figsize)
        build_style_artfid(ax, include_tgt=with_tgt)
        ax.set_title(title or "ArtFID vs. Style Affinity", fontweight="bold", pad=3)
        if drop_ylabel:
            ax.set_ylabel("")
        ax.tick_params(labelsize=9)
        ax.xaxis.label.set_fontsize(9)
        ax.title.set_fontsize(10)
        ax.title.set_fontfamily("Times New Roman")
        ax.xaxis.label.set_fontfamily("Times New Roman")
        for artist in list(ax.texts) + list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
            artist.set_fontsize(9)
            artist.set_fontfamily("Times New Roman")
        fig.savefig(OUT_DIR / fname, dpi=300, bbox_inches="tight")
        if kind == "artfid":
            import matplotlib as mpl

            mpl.rcParams["svg.fonttype"] = "none"
            svg_path = OUT_DIR / (Path(fname).stem + ".svg")
            fig.savefig(svg_path, format="svg", bbox_inches="tight")
            _inject_drag_script(svg_path)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-tgt", action="store_true", help="include the TGT anchor in the ArtFID panel")
    args = parser.parse_args()

    # Paper panel (b): single-column, no Y-axis text label (axis described in caption).
    save_single_panel(
        "artfid",
        "fig_page1_b.png",
        (3.7, 2.18),
        with_tgt=args.with_tgt,
        drop_ylabel=True,
        title="ArtFID vs. Style Affinity",
    )


if __name__ == "__main__":
    main()
