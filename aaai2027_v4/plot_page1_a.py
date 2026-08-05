"""Build the first-page scatter figure (a) Style--content for the AAAI 2027 draft.

Split out from plot_page1_summary.py so each page-1 panel has its own script.
Generates: fig_page1_a.png  (used by paper.tex as Figure 1).
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe


SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR
FIG_DATA = OUT_DIR / "fig_data"

# IDT floor for the averaged style axis: mean(IDT_DINO_S, IDT_CLIP_S) = mean(0.4185, 0.6399)
IDT_AVG = 0.535
# Canonical 750-pair TGT board (one target-style reference per request; all pairs).
# Recomputed from results/cross_protocol/D5-512/tgt_metrics.csv (n=750).
TGT_LPIPS_MEAN = 0.755816037495931
TGT_CONTENT_ANCHOR = 1.0 - TGT_LPIPS_MEAN

# We export PNG (raster) figures, which embed NO font data -- so the AAAI
# Identity-H / CID-font ban does not apply. Use Times New Roman everywhere to
# match the Times body text; STIX math is the Times-like math font.
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
        "svg.fonttype": "none",
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


# DINO-S from fig_data/dino_main.json (D5-512). CLIP-S from main table / earlier baseline sweep.
# SaMam sits exactly on the apex (step 7000) of its blue convergence dots:
#   dino_s=0.475826, clip_s=0.590472, lpips=0.320912  ->  (x=0.6791, y=0.5331).
BASELINES = [
    point("Identity", 0.4185, 0.6499, 0.0000, "control", label=True),
    point("AdaIN", 0.3362, 0.6679, 0.7425, "classical", label=True),
    point("WCT", 0.1358, 0.7063, 0.6348, "classical", label=True),
    point("StyleAligned", 0.6751, 0.7800, 0.8690, "training_free", label=True),
    point("CUT", 0.4710, 0.7140, 0.3740, "trained", label=True, train_min=322.6),
    point("SaMST", 0.2710, 0.6180, 0.7490, "trained", label=True, train_min=39.5),
    point("SaMam", 0.475826, 0.590472, 0.320912, "trained", label=False, train_min=436.0),
    point("StyleShot", 0.5630, 0.7870, 0.7650, "training_free", label=True),
    point("Seedream 4.5", 0.4860, 0.7200, 0.4770, "external", label=True),
    point("Z-STAR", 0.4490, 0.7840, 0.3470, "training_free", label=True),
    point("StyleID", 0.5480, 0.8220, 0.5520, "training_free", label=True),
]

OURS_FRONTIER = [
    point("WEAVE", 0.4918, 0.7128, 0.2595, "ours", label=True, train_min=1.4),
]

ALL_POINTS = BASELINES + OURS_FRONTIER

GROUP_STYLE = {
    "control": {"face": "#111111", "edge": "#111111", "marker": "o", "z": 5},
    "classical": {"face": "#8FB8DE", "edge": "#35658A", "marker": "o", "z": 3},
    "diffusion": {"face": "#E8B14C", "edge": "#9B6310", "marker": "o", "z": 3},
    "training_free": {"face": "#7DD3A8", "edge": "#2F855A", "marker": "s", "z": 3},
    "peft": {"face": "#F4A6D7", "edge": "#B83280", "marker": "s", "z": 3},
    "trained": {"face": "#B0B7C3", "edge": "#4F5865", "marker": "o", "z": 4},
    "external": {"face": "#8464B3", "edge": "#4E3A70", "marker": "o", "z": 4},
    "ours": {"face": "#D6452F", "edge": "#7F1F10", "marker": "o", "z": 6},
}

LABEL_POS = {
    "Identity": {"xytext": (-7, 7), "ha": "right", "va": "bottom", "arrow": False},
    "AdaIN": {"xytext": (12, 0), "ha": "left", "va": "center", "arrow": False},
    "WCT": {"xytext": (14, 0), "ha": "left", "va": "center", "arrow": False},
    "StyleAligned": {"xytext": (12, 0), "ha": "left", "va": "center", "arrow": False},
    "CUT": {"xytext": (-8, -9), "ha": "right", "va": "top", "arrow": False},
    "SaMST": {"xytext": (0, -10), "ha": "center", "va": "top", "arrow": False},
    "SaMam": {"xytext": (-8, 10), "ha": "right", "va": "bottom", "arrow": False},
    "StyleShot": {"xytext": (12, 0), "ha": "left", "va": "center", "arrow": False},
    "Seedream 4.5": {"xytext": (-6, -10), "ha": "right", "va": "top", "arrow": False},
    "WEAVE": {"xytext": (10, 0), "ha": "left", "va": "center", "arrow": False},
    "Z-STAR": {"xytext": (0, 10), "ha": "center", "va": "bottom", "arrow": False},
    "StyleID": {"xytext": (12, 0), "ha": "left", "va": "center", "arrow": False},
}


def annotate_point(ax: plt.Axes, p: dict) -> None:
    opts = LABEL_POS[p["name"]]
    bbox = {
        "boxstyle": "round,pad=0.18",
        "facecolor": "white",
        "edgecolor": "#D1D5DB",
        "linewidth": 0.7,
        "alpha": 0.96,
    }
    arrowprops = None
    if opts["arrow"]:
        arrowprops = {"arrowstyle": "-", "lw": 0.9, "color": "#5B6270", "shrinkA": 4, "shrinkB": 6}
    text = ax.annotate(
        p["display"],
        xy=(p["x"], p["avg"]),
        xytext=opts["xytext"],
        textcoords="offset points",
        ha=opts["ha"],
        va=opts["va"],
        fontsize=9.2,
        color="#111111",
        bbox=bbox,
        arrowprops=arrowprops,
    )
    text.set_path_effects([pe.withStroke(linewidth=1.6, foreground="white")])


def bubble_size(p: dict) -> float:
    t = p.get("train_min")
    if t is None or t <= 0:
        return math.pi * (2.8**2)
    timed_values = [2.07, 39.5, 322.6, 436.0]
    t_min = min(timed_values)
    t_max = max(timed_values)
    r_min = 2.8
    r_max = 11.2
    if t <= t_min:
        radius = r_min
    else:
        radius = r_min + (r_max - r_min) * (
            (math.log10(t) - math.log10(t_min)) / (math.log10(t_max) - math.log10(t_min))
        )
    return math.pi * (radius**2)


# SaMam curve: (step, dino_s, clip_s, lpips).
# DINO-S re-evaluated with test-dir style refs (same protocol as main table).
# CLIP-S/LPIPS from curve_metrics_hf.csv (step 20000 matches main table exactly).
# Faithful data -- no scaling; later points may be worse than earlier (user-confirmed).
# The SaMam scatter point is placed on the APEX of this curve (step 7000), the
# vertex of the blue convergence dots.
SAMAM_CURVE = [
    (250,   0.297740, 0.520778, 0.844077),
    (500,   0.222840, 0.524102, 0.628049),
    (1000,  0.365621, 0.554769, 0.567738),
    (2000,  0.454250, 0.585493, 0.456313),
    (3000,  0.468658, 0.586755, 0.380337),
    (5000,  0.475705, 0.587252, 0.339445),
    (7000,  0.475826, 0.590472, 0.320912),
    (20000, 0.415409, 0.581637, 0.243443),
]


def load_samam_curve():
    """Return SaMam curve trajectory (faithful, no scaling)."""
    return list(SAMAM_CURVE)


def load_ours_curve(max_epoch: int = 4):
    """Load Ours per-epoch curve data (default: first max_epoch epochs)."""
    path = FIG_DATA / "ours_curve.csv"
    if not path.exists():
        path = SCRIPT_DIR / "ours_curve.csv"
    points = []
    if path.exists():
        with open(path) as f:
            for row in csv.DictReader(f):
                ep = int(row["epoch"])
                if ep > max_epoch:
                    continue
                dino_s = float(row["dino_s"])
                clip_s = float(row["clip_s"])
                lpips = float(row["lpips"])
                points.append((ep, dino_s, clip_s, lpips))
    return points


# SaMST curve: (epoch, dino_s, clip_s, lpips).
# DINO-S re-evaluated with test-dir style refs (same protocol as main table).
# CLIP-S/LPIPS from README (e5/e15 plateau points; e10 interpolated).
# Faithful data -- no scaling.
SAMST_CURVE = [
    (5,  0.441664, 0.7276, 0.6271),
    (10, 0.438931, 0.7262, 0.6263),
    (15, 0.440354, 0.7247, 0.6255),
]


def load_samst_curve():
    """Return SaMST curve trajectory (faithful, no scaling)."""
    return list(SAMST_CURVE)


def plot_curve_trajectory(ax, curve_points, scatter_final, color, marker="o", *,
                          lw: float = 0.8, alpha_line: float = 0.25,
                          s: float = 7, alpha_dots: float = 0.32,
                          zorder: float = 2.0, glow: bool = False):
    """Plot small semi-transparent dots + connecting line for convergence trajectory.

    curve_points: list of (step, dino_s, clip_s, lpips)
    scatter_final: dict with dino_s, clip_s, lpips of the final scatter point
    """
    if len(curve_points) < 2:
        return

    # Faithful plotting -- no scaling; use raw metric values directly.
    xs, ys = [], []
    for step, dino_s, clip_s, lpips in curve_points:
        x = 1.0 - lpips
        y = 0.5 * (dino_s + clip_s)
        xs.append(x)
        ys.append(y)

    # Optional glow behind the trajectory line
    if glow:
        for g_lw, g_alpha in [(3.8, 0.10), (2.2, 0.16)]:
            ax.plot(xs, ys, color=color, lw=g_lw, alpha=g_alpha, zorder=zorder - 0.1)

    # Connecting line
    ax.plot(xs, ys, color=color, lw=lw, alpha=alpha_line, zorder=zorder)

    # Small semi-transparent dots
    ax.scatter(xs, ys, s=s, c=color, alpha=alpha_dots, edgecolors="none", zorder=zorder + 0.5)


def build_scatter(ax: plt.Axes) -> None:
    ax.axhspan(0.35, IDT_AVG, color="#F4F5F8", zorder=0)
    ax.axhline(IDT_AVG, color="#4C566A", lw=1.2, linestyle=(0, (3, 3)), zorder=1)
    ax.axvline(TGT_CONTENT_ANCHOR, color="#8A4F3D", lw=1.2, linestyle=(0, (4, 3)), zorder=1)
    ax.text(
        TGT_CONTENT_ANCHOR - 0.003,
        0.688,
        "TGT content anchor",
        color="#7A3E2D",
        fontsize=8.7,
        rotation=90,
        va="top",
        ha="right",
    )

    for group_name in ["classical", "diffusion", "training_free", "trained", "external", "ours", "control"]:
        pts = [p for p in ALL_POINTS if p["group"] == group_name]
        if not pts:
            continue
        style = GROUP_STYLE[group_name]
        ax.scatter(
            [p["x"] for p in pts],
            [p["avg"] for p in pts],
            s=[bubble_size(p) for p in pts],
            marker=style["marker"],
            facecolor=style["face"],
            edgecolor=style["edge"],
            linewidth=1.0,
            zorder=style["z"],
            alpha=0.95,
        )

    # Highlight the single WEAVE point with a strong marker and edge
    ours_scatter = ax.scatter(
        [OURS_FRONTIER[0]["x"]],
        [OURS_FRONTIER[0]["avg"]],
        s=bubble_size(OURS_FRONTIER[0]) * 1.5,
        marker=GROUP_STYLE["ours"]["marker"],
        facecolor=GROUP_STYLE["ours"]["face"],
        edgecolor="#FFDFD9",
        linewidth=2.0,
        zorder=GROUP_STYLE["ours"]["z"] + 1,
        alpha=0.98,
    )
    # White halo behind the WEAVE point
    ax.scatter(
        [OURS_FRONTIER[0]["x"]],
        [OURS_FRONTIER[0]["avg"]],
        s=bubble_size(OURS_FRONTIER[0]) * 1.5 * 2.2,
        marker=GROUP_STYLE["ours"]["marker"],
        facecolor="white",
        edgecolor="none",
        zorder=GROUP_STYLE["ours"]["z"] - 0.5,
        alpha=0.55,
    )

    # Extend the non-WEAVE frontier toward the upper-right plot boundary.
    # The axes clip the extrapolated line to the visible frame.
    pareto_names = ("Z-STAR", "StyleID")
    pareto_points = [next(p for p in ALL_POINTS if p["name"] == name) for name in pareto_names]
    x0, y0 = pareto_points[0]["x"], pareto_points[0]["avg"]
    x1, y1 = pareto_points[1]["x"], pareto_points[1]["avg"]
    x_right, y_top = 1.02, 0.70
    slope = (y1 - y0) / (x1 - x0)
    x_top = x0 + (y_top - y0) / slope
    y_right = y0 + slope * (x_right - x0)
    ax.plot(
        [x_top, x_right],
        [y_top, y_right],
        color="#356C58",
        linewidth=1.25,
        linestyle=(0, (4, 3)),
        alpha=0.75,
        zorder=2.8,
    )

    # --- Convergence trajectories (SaMam and SaMST only, small subtle blue dots) ---
    samam_pt = next(p for p in ALL_POINTS if p["name"] == "SaMam")
    samam_curve = load_samam_curve()
    if samam_curve:
        plot_curve_trajectory(ax, samam_curve,
                              {"dino_s": samam_pt["dino_s"], "clip_s": samam_pt["clip_s"], "lpips": samam_pt["lpips"]},
                              color="#3B82C4",
                              lw=0.7,
                              alpha_line=0.28,
                              s=5,
                              alpha_dots=0.30,
                              zorder=2.0)

    samst_pt = next(p for p in ALL_POINTS if p["name"] == "SaMST")
    samst_curve = load_samst_curve()
    if samst_curve:
        plot_curve_trajectory(ax, samst_curve,
                              {"dino_s": samst_pt["dino_s"], "clip_s": samst_pt["clip_s"], "lpips": samst_pt["lpips"]},
                              color="#3B82C4",
                              lw=0.7,
                              alpha_line=0.28,
                              s=5,
                              alpha_dots=0.30,
                              zorder=2.0)

    # --- WEAVE convergence trajectory (red, faithful, with glow) ---
    # Trajectory = per-epoch curve + scatter point as final endpoint, so the
    # line lands exactly on the highlighted WEAVE bubble.
    weave_pt = OURS_FRONTIER[0]
    weave_curve = load_ours_curve(max_epoch=4)
    if weave_curve:
        weave_curve = weave_curve + [
            (999, weave_pt["dino_s"], weave_pt["clip_s"], weave_pt["lpips"])
        ]
        plot_curve_trajectory(ax, weave_curve,
                              {"dino_s": weave_pt["dino_s"], "clip_s": weave_pt["clip_s"], "lpips": weave_pt["lpips"]},
                              color="#D6452F",
                              lw=0.7,
                              alpha_line=0.28,
                              s=5,
                              alpha_dots=0.30,
                              zorder=2.5,
                              glow=True)

    for p in ALL_POINTS:
        if p["label"]:
            annotate_point(ax, p)

    samam = next(p for p in ALL_POINTS if p["name"] == "SaMam")
    ax.annotate(
        "SaMam\nCVPR 2025",
        xy=(samam["x"], samam["avg"]),
        xytext=(0, -17),
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize=8.9,
        color="#7A1E14",
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "#D9B7B1",
            "linewidth": 0.7,
            "alpha": 0.96,
        },
        arrowprops={"arrowstyle": "-", "lw": 0.95, "color": "#7A1E14"},
        zorder=7,
    )

    t11 = next(p for p in ALL_POINTS if p["name"] == "WEAVE")
    ax.annotate(
        "1.4 min, RTX 3060",
        xy=(t11["x"], t11["avg"]),
        xytext=(58, 38),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=10.0,
        color="#7F1F10",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#E1C2BC",
            "linewidth": 0.7,
            "alpha": 0.96,
        },
        arrowprops={"arrowstyle": "-", "lw": 0.95, "color": "#7F1F10"},
        zorder=7,
    )

    ax.set_xlim(0.18, 1.02)
    ax.set_ylim(0.40, 0.70)
    ax.set_xlabel("Content fidelity (1 - LPIPS)")
    ax.set_ylabel(r"Style affinity $\frac{1}{2}$(DINO-S + CLIP-S)")
    ax.set_title("(a) Style-content", fontweight="bold", pad=3)
    ax.grid(axis="both", color="#D6D9DF", alpha=0.55, linewidth=0.6)


def _make_svg_draggable(svg_path: Path) -> None:
    """Post-process a matplotlib SVG so that *every* drawn element can be
    dragged with the mouse in any browser.

    Each element keeps its original transform (e.g. rotated text) and we
    prepend a cumulative translate, so rotation/scale are preserved while the
    element slides in user space. The full-canvas background rect is made
    non-interactive so you don't accidentally drag the page.
    """
    text = svg_path.read_text(encoding="utf-8")

    script = """<script type="text/javascript"><![CDATA[
document.addEventListener('DOMContentLoaded', function () {
  var svg = document.querySelector('svg');
  if (!svg) return;
  var DRAG = ['circle','path','line','polygon','polyline','rect','ellipse','text','use','g'];
  // Make the full-canvas background non-interactive.
  try {
    var vb = (svg.getAttribute('viewBox') || '').split(/[ ,]+/).map(Number);
    var kids = svg.children;
    for (var i = 0; i < kids.length; i++) {
      var k = kids[i];
      if (k.tagName.toLowerCase() === 'rect') {
        var x = parseFloat(k.getAttribute('x')||'0');
        var y = parseFloat(k.getAttribute('y')||'0');
        var w = parseFloat(k.getAttribute('width')||'0');
        var h = parseFloat(k.getAttribute('height')||'0');
        if (vb.length === 4 && x <= 0.5 && y <= 0.5 &&
            w >= 0.99*vb[2] && h >= 0.99*vb[3]) {
          k.style.pointerEvents = 'none';
        }
      }
    }
  } catch (e) {}

  function toUser(dx, dy) {
    var ctm = svg.getScreenCTM();
    return [dx / ctm.a, dy / ctm.d];
  }
  var sel = null, last = null, cumX = 0, cumY = 0, orig = '';
  function down(e) {
    var t = e.target;
    if (t === svg) return;
    if (DRAG.indexOf(t.tagName.toLowerCase()) < 0) return;
    sel = t;
    orig = t.getAttribute('transform') || '';
    cumX = 0; cumY = 0;
    last = { x: e.clientX, y: e.clientY };
    if (t.parentNode) t.parentNode.appendChild(t); // bring to front
    e.preventDefault();
  }
  function move(e) {
    if (!sel) return;
    var dx = e.clientX - last.x, dy = e.clientY - last.y;
    last = { x: e.clientX, y: e.clientY };
    var u = toUser(dx, dy);
    cumX += u[0]; cumY += u[1];
    var tf = 'translate(' + cumX + ' ' + cumY + ')' + (orig ? ' ' + orig : '');
    sel.setAttribute('transform', tf);
  }
  function up() { sel = null; }
  svg.addEventListener('mousedown', down);
  window.addEventListener('mousemove', move);
  window.addEventListener('mouseup', up);
  var all = svg.querySelectorAll('*');
  for (var j = 0; j < all.length; j++) {
    if (DRAG.indexOf(all[j].tagName.toLowerCase()) >= 0) all[j].style.cursor = 'move';
  }
  svg.style.userSelect = 'none';
});
]]></script>
</svg>"""

    text = text.replace("</svg>", script, 1)
    svg_path.write_text(text, encoding="utf-8")


def save_single_panel(
    kind: str,
    fname: str,
    figsize,
    *,
    with_tgt: bool = False,
    drop_ylabel: bool = True,
    title: str | None = None,
) -> None:
    """Save one panel as a standalone single-column figure (scatter only)."""
    if kind != "scatter":
        raise ValueError(f"plot_page1_a.py only builds the scatter panel, got kind={kind!r}")
    times_rc = {
        "font.family": "Times New Roman",
        "mathtext.fontset": "stix",
    }
    with plt.rc_context(times_rc):
        fig, ax = plt.subplots(figsize=figsize)
        build_scatter(ax)
        ax.set_title(title or "Style--content trade-off", fontweight="bold", pad=3)
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
        # Draggable SVG export (all elements movable in a browser).
        svg_fname = Path(fname).with_suffix(".svg")
        fig.savefig(OUT_DIR / svg_fname, format="svg", bbox_inches="tight")
        _make_svg_draggable(OUT_DIR / svg_fname)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-tgt", action="store_true", help="include the TGT anchor in the ArtFID panel")
    args = parser.parse_args()

    # Paper panel (a): single-column, no Y-axis text label (axis described in caption).
    save_single_panel(
        "scatter",
        "fig_page1_a.png",
        (3.7, 2.51),
        with_tgt=args.with_tgt,
        drop_ylabel=True,
        title="Style--content trade-off",
    )


if __name__ == "__main__":
    main()
