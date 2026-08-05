"""Metric-block radar chart using the PAPER MAIN TABLE data (Table 1, paper.tex).

Source of truth = aaai2027_v4/paper.tex, Table "tab:main" (lines 330-343).
14 axes = 12 metric axes (4 metrics x 3 datasets) + 2 speed axes:
  DINO-S(D5,R5,P2A) -> CLIP-S(D5,R5,P2A) -> DINO-C(D5,R5,P2A) -> 1-LPIPS(D5,R5,P2A)
  -> Infer speed(750img) -> Train speed(min)
MUSIQ is intentionally omitted. Each metric family is min-max normalized once across all
reported methods and all three datasets, so the best reported value reaches the outer ring
and D5/P2A/R5 remain on the same scale. Train / Infer speed axes apply the same min-max
rule after log10(1 + seconds), with smaller times better. Training-free methods leave a GAP
on the Train-speed axis (not drawn, no cross-gap fill). Cells without DINO data are gaps
(NaN).

Color & visual hierarchy:
  - BOLD (thick, drawn on top): Ours(WEAVE), Seedream 4.5, Z-STAR, StyleAligned
  - FAINT (thin, low alpha): everything else (classical / secondary baselines)

Prints weakness analysis of WEAVE vs tier-A baselines.
"""
import json
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
import io

# Force stdout to UTF-8 for Windows GBK console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Times New Roman"
plt.rcParams["mathtext.it"] = "Times New Roman:italic"
plt.rcParams["mathtext.bf"] = "Times New Roman:bold"

# Self-contained: paths resolve relative to this script (which lives in
# aaai2027_v4); bundled inputs are read from ./fig_data so the paper folder
# stands alone and can regenerate every figure.
SCRIPT_DIR = Path(__file__).resolve().parent
OUT = SCRIPT_DIR / "supplement_figures" / "fig_radar_full.pdf"
OUT_PNG = SCRIPT_DIR / "supplement_figures" / "fig_radar_full.png"
DINO_JSON = SCRIPT_DIR / "fig_data" / "dino_main.json"

DATA = {
    # tuple = (CLIP-S, LPIPS(raw, lower=better), DINO-C, DINO-S); MUSIQ dropped.
    # CLIP-S/LPIPS from paper tab:main; DINO-C/DINO-S from local DINOv2 (state/dino/dino_main.json).
    # Per-axis empirical boundary: IDT on style axes, TGT on content axes.
    "IDT-TGT bound":  [(0.640, 0.756, 0.220, 0.419), (0.663, 0.800, 0.169, 0.416), (0.731, 0.747, 0.230, 0.433)],
    "SD-Turbo":       [(0.693, 0.003, 0.922, 0.484), (0.674, 0.603, 0.279, 0.341), (0.767, 0.449, 0.538, 0.505)],
    "StyleAligned":   [(0.780, 0.869, 0.239, 0.675), (0.768, 0.786, 0.310, 0.612), (0.824, 0.829, 0.315, 0.649)],
    "Z-STAR":         [(0.784, 0.347, 0.549, 0.449), (0.786, 0.332, 0.552, 0.498), (0.822, 0.384, 0.526, 0.514)],
    "StyleShot":      [(0.787, 0.765, 0.377, 0.563), (0.774, 0.713, 0.335, 0.552), (0.787, 0.795, 0.380, 0.484)],
    "CUT":            [(0.714, 0.374, 0.795, 0.471), (0.754, 0.494, 0.7020, 0.5387), (0.710, 0.620, 0.795, 0.441)],
    "SaMST":          [(0.618, 0.749, 0.145, 0.271), (0.709, 0.398, 0.838, 0.501), (0.667, 0.612, 0.615, 0.461)],
    "SaMam":          [(0.702, 0.248, 0.812, 0.477), (0.677, 0.205, 0.870, 0.505), (0.712, 0.227, 0.925, 0.503)],
    "Latent-WCT":     [(0.673, 0.441, 0.559, 0.362), (0.738, 0.585, 0.386, 0.338), (0.710, 0.444, 0.553, 0.414)],
    "Seedream 4.5":   [(0.720, 0.477, 0.739, 0.486), (0.752, 0.227, 0.785, 0.517), (0.742, 0.486, 0.725, 0.504)],
    "Ours (WEAVE)":   [(0.7128, 0.2595, 0.8102, 0.4918), (0.6681, 0.3116, 0.8612, 0.4801), (0.7747, 0.2895, 0.7717, 0.5226)],
}
DATASETS = ["D5-512", "P2A-256", "R5-WikiArt"]

# Train / Infer times from paper tab:main (D5 context, single value per method).
# Converted to SECONDS. math.nan = training-free ("free") or closed API ("--"/"API").
TRAIN_SEC = {
    "IDT-TGT bound":  math.nan,        # reference only
    "SD-Turbo":       math.nan,        # free
    "StyleAligned":   math.nan,        # free
    "Z-STAR":         math.nan,        # free
    "StyleShot":      math.nan,        # free
    "CUT":            322.6 * 60.0,    # 322.6 min
    "SaMST":          39.5  * 60.0,    # 39.5 min
    "SaMam":          436.0 * 60.0,    # 436.0 min
    "Latent-WCT":     math.nan,        # free
    "Seedream 4.5":   math.nan,        # API
    "Ours (WEAVE)":   82.8,            # 1.38 min, internal stop at epoch 4
}
INFER_SEC = {
    "IDT-TGT bound":  math.nan,        # reference only
    "SD-Turbo":       303.0,           # 5.1 m  (0.404 s/img x750, RTX3060 measured)
    "StyleAligned":   4635.0,          # 77 m   (6.18 s/img x750, RTX3060 measured)
    "Z-STAR":         10800.0,         # ~3 h   ESTIMATED (30-step SD1.5 + dual-latent reweight;
                                       #          OOMs at 512^2 on 12GB -> step-count scaling)
    "StyleShot":      18472.0,         # 5.1 h  (24.63 s/img x750, RTX3060 measured)
    "CUT":            300.0,           # 5 m    (not re-measured on 3060; prior value)
    "SaMST":          10.0 * 60.0,     # 10 m   (not re-measured on 3060; prior value)
    "SaMam":          17.6 * 60.0,     # 17.6 m (not re-measured on 3060; prior value)
    "Latent-WCT":     math.nan,        # free / not drawn on speed axis
    "Seedream 4.5":   math.nan,        # --
    "Ours (WEAVE)":   73.0,            # ONNX/TRT pipeline, 750 images, 97.3 ms/img
}

# radar method name -> local results/<ds>/<dir> name (for DINO lookup)
METHOD_DIR = {
    "IDT-TGT bound": "identity",
    "SD-Turbo": "sdturbo", "StyleAligned": "stylealigned",
    "Z-STAR": "zstar", "StyleShot": "styleshot", "CUT": "cut", "SaMST": "samst",
    "SaMam": "samam", "Latent-WCT": "latent_wct", "Seedream 4.5": "seedream", "Ours (WEAVE)": "weave",
}

# Load DINO (may be partial)
DINO = {}
if DINO_JSON.exists():
    try:
        DINO = json.loads(DINO_JSON.read_text(encoding="utf-8"))
    except Exception as e:
        print("WARN: could not read DINO json:", e)

# Three-tier visual hierarchy (plot line + legend text):
#   Tier-1 (thickest): Ours (WEAVE)
#   Tier-2 (secondary): Seedream 4.5, Z-STAR
#   Tier-3 (tertiary):  StyleAligned, SaMam
# Everything else is FAINT (thin line, low alpha) so the key competitors stand out.
TIER1 = {
    "Ours (WEAVE)": ("#D62728", 2.2, 0.16, 1.00, 14),
}
TIER2 = {
    "Seedream 4.5": ("#1565C0", 1.8, 0.14, 0.96, 13),
    "Z-STAR":       ("#2CA02C", 1.3, 0.11, 0.94, 12),
}
TIER3 = {
    "StyleAligned": ("#FF7F0E", 1.3, 0.11, 0.92, 11),
    "SaMam":        ("#BCBD22", 0.9, 0.08, 0.86, 10),
}
FAINT = {
    "StyleShot":    ("#9467BD", 0.60, 0.025, 0.38, 4),
    "CUT":          ("#E377C2", 0.60, 0.025, 0.38, 4),
    "SaMST":        ("#17BECF", 0.60, 0.02, 0.34, 3),
    "SD-Turbo":     ("#AEC7E8", 0.55, 0.02, 0.30, 3),
    "Latent-WCT":   ("#8C564B", 0.70, 0.03, 0.58, 5),
    "IDT-TGT bound":("#000000", 0.85, 0.04, 0.70, 3),
}
TIERS = {}
TIERS.update(FAINT)
TIERS.update(TIER3)
TIERS.update(TIER2)
TIERS.update(TIER1)

# BOLD set used by the weakness analysis (all emphasized competitors).
BOLD = {}
BOLD.update(TIER1); BOLD.update(TIER2); BOLD.update(TIER3)

# Legend tier of each method (drives per-entry font weight / color in the legend).
TIER_OF = {}
TIER_OF.update({k: "t1" for k in TIER1})
TIER_OF.update({k: "t2" for k in TIER2})
TIER_OF.update({k: "t3" for k in TIER3})

# The mixed IDT--TGT boundary is drawn as a black dashed reference.
LINESTYLE = {"IDT-TGT bound": "--"}

LEGEND_ORDER = ["Ours (WEAVE)", "Seedream 4.5", "Z-STAR", "StyleAligned", "SaMam",
                "IP-Adapter", "Latent-WCT", "StyleShot", "CUT", "SaMST", "SD-Turbo", "IDT-TGT bound"]

# Year / venue for each method (best-effort; AMiner token unavailable -> from public record).
META = {
    "IDT-TGT bound":  ("ref",        "reference"),
    "SD-Turbo":       ("2023",       "Sauer et al."),
    "IP-Adapter":     ("2023",       "Ye et al."),
    "StyleAligned":   ("CVPR 2024",  "Hertz et al."),
    "Z-STAR":         ("CVPR 2024",  "Deng et al."),
    "StyleShot":      ("2024",       "Gao et al."),
    "CUT":            ("ECCV 2020",  "Park et al."),
    "SaMST":          ("ACCV 2024",  "Liu et al."),
    "SaMam":          ("CVPR 2025",  "Liu et al."),
    "Latent-WCT":     ("ref",        "classical"),
    "Seedream 4.5":   ("2025",       "ByteDance"),
    "Ours (WEAVE)":   ("",           "this paper"),
}

# Metric groups (5 metrics x 3 datasets = 15 axes). kind -> how to compute.
METRICS = [
    ("DINO-S$\\uparrow$",    "dino_style"),
    ("CLIP-S$\\uparrow$",   "clip"),
    ("DINO-C$\\uparrow$",    "dino_content"),
    ("1$-$LPIPS$\\uparrow$", "lpips"),
]

# Per-metric dataset display order. All datasets for the same metric stay contiguous.
# R5 and P2A are swapped from the table order so the 256px P2A axes move later/lower.
DS_ORDER = {"dino_style": [0, 2, 1], "clip": [0, 2, 1], "dino_content": [0, 2, 1], "lpips": [0, 2, 1]}
def _ds_idx(kind):
    return DS_ORDER.get(kind, list(range(len(DATASETS))))

# Infer BEFORE Train: most training-free methods have an Infer value but Train=NaN, so putting
# Infer adjacent to the metric block lets their polygon connect through Infer, leaving the gap on
# the trailing Train axis instead of breaking the line mid-way.
TIME_AXES = [
    ("Infer speed$\\uparrow$\n750img", INFER_SEC),
    ("Train speed$\\uparrow$\nmin",  TRAIN_SEC),
]

def _time_score(sec):
    """Log-time score before standard min-max scaling; smaller time is better."""
    if not isinstance(sec, float) or math.isnan(sec):
        return math.nan          # training-free / API -> gap
    return -math.log10(1.0 + max(sec, 0.0))

_SHORT_METRIC = {"DINO-S$\\uparrow$": "D-S", "CLIP-S$\\uparrow$": "C-S",
                 "DINO-C$\\uparrow$": "D-C", "1$-$LPIPS$\\uparrow$": "1-LP"}
_SHORT_DS = {"D5-512": "D5", "P2A-256": "P2A", "R5-WikiArt": "R5"}
AXIS_LABELS = [f"{_SHORT_METRIC[gname]}\n{_SHORT_DS[DATASETS[d]]}"
               for gname, kind in METRICS for d in _ds_idx(kind)] + ["Infer\n750", "Train\nmin"]


def _dino_val(method, kind, d_idx):
    ds = DATASETS[d_idx]
    dirname = METHOD_DIR.get(method)
    if dirname is None:
        return math.nan
    cell = DINO.get(f"{ds}|{dirname}")
    if not cell:
        return math.nan
    v = cell.get(kind)
    return float(v) if isinstance(v, (int, float)) else math.nan


def raw_axes(method):
    """Unnormalized comparison values; LPIPS is inverted and time is log-scored.
    DATA tuple = (CLIP-S, LPIPS(raw, lower is better), DINO-C, DINO-S)."""
    triples = DATA[method]
    out = []
    for _, kind in METRICS:
        for d in _ds_idx(kind):
            if kind == "clip":
                out.append(triples[d][0])
            elif kind == "lpips":
                out.append(1.0 - triples[d][1])
            elif kind == "dino_content":
                out.append(triples[d][2])
            elif kind == "dino_style":
                out.append(triples[d][3])
    for _, secdict in TIME_AXES:
        out.append(_time_score(secdict.get(method)))
    return out


AXIS_GROUPS = [kind for _, kind in METRICS for _ in _ds_idx(kind)] + ["infer_speed", "train_speed"]


def _range(values):
    valid = [v for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
    return min(valid), max(valid)


# A metric has one shared range across D5/P2A/R5. This is a standard min-max
# normalization, not a per-axis knee or a rank transform.
GROUP_RANGES = {}
for group in set(AXIS_GROUPS):
    positions = [i for i, axis_group in enumerate(AXIS_GROUPS) if axis_group == group]
    GROUP_RANGES[group] = _range([raw_axes(name)[i] for name in DATA for i in positions])


def _minmax(value, bounds):
    if not isinstance(value, (int, float)) or math.isnan(value):
        return math.nan
    lo, hi = bounds
    if hi <= lo:
        return 1.0
    return (value - lo) / (hi - lo)


def to_axes(method):
    values = raw_axes(method)
    return [_minmax(value, GROUP_RANGES[group]) for value, group in zip(values, AXIS_GROUPS)]


def contiguous_runs(mask):
    """Maximal runs of True in a boolean list (no wrap-around)."""
    n = len(mask); runs = []; i = 0
    while i < n:
        if mask[i]:
            s = i
            while i < n and mask[i]:
                i += 1
            runs.append((s, i - 1))
        else:
            i += 1
    return runs


N = len(AXIS_LABELS)
PLOT_ORDER = [13, 12, 5, 3, 4, 0, 1, 2, 6, 7, 8, 9, 10, 11]
PLOT_AXIS_LABELS = [AXIS_LABELS[i] for i in PLOT_ORDER]
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]
THETA_OFFSET_DEG = (180.0 + 180.0 / N) % 360.0
THETA_OFFSET_RAD = np.deg2rad(THETA_OFFSET_DEG)

# Wider figure; the polar axes is squeezed into a LEFT-anchored, horizontally
# squashed box so its right-side axis labels and speed axes clear the
# dedicated legend/caption column on the right.
fig = plt.figure(figsize=(7.0, 4.4))
ax = fig.add_axes([0.05, 0.08, 0.56, 0.84], polar=True)
ax.set_theta_offset(THETA_OFFSET_RAD)
ax.set_theta_direction(-1)

for name in DATA:
    vals_raw_order = to_axes(name)
    vals = [vals_raw_order[i] for i in PLOT_ORDER]
    c, lw, af, al, zo = TIERS[name]
    mask = [not (isinstance(v, float) and math.isnan(v)) for v in vals]
    runs = contiguous_runs(mask)
    labeled = False
    for (s, e) in runs:
        # Fan-to-center fill: no cross-gap chord, no shadow drawn across missing axes.
        if e - s + 1 >= 2:
            th = [angles[s]] + angles[s:e+1] + [angles[e]]
            rv = [0.0] + vals[s:e+1] + [0.0]
            ax.fill(th, rv, color=c, alpha=af, zorder=zo)
        seg_a = angles[s:e+1]
        seg_v = vals[s:e+1]
        if s == 0 and e == N - 1:           # full circle -> close the loop
            seg_a = seg_a + [seg_a[0]]
            seg_v = seg_v + [seg_v[0]]
        ax.plot(seg_a, seg_v, color=c, linewidth=lw, alpha=al, zorder=zo,
                linestyle=LINESTYLE.get(name, "-"),
                label=(name if not labeled else None))
        labeled = True
    va = [angles[i] for i in range(N) if mask[i]]
    vv = [vals[i] for i in range(N) if mask[i]]
    if va:
        ax.scatter(va, vv, color=c, s=7, alpha=min(1.0, al + 0.25),
                   zorder=zo + 1, edgecolors="none")

# ---- axis labels drawn OUTSIDE the circle so they never overlap the spokes ----
ax.set_xticks(angles[:-1])
ax.set_xticklabels([])                       # labels drawn manually, radially outside
ax.set_ylim(0, 1)
ax.set_yticks([])
ax.set_yticklabels([])
ax.grid(True, alpha=0.35)
ax.tick_params(axis="y", length=0, pad=6)

_LABEL_R = 1.085   # radius (in data units); ~half the previous 0.17 gap from the outer ring
for ang, lab in zip(angles[:-1], PLOT_AXIS_LABELS):
    deg = (THETA_OFFSET_DEG - np.degrees(ang)) % 360
    if 45 <= deg < 135:
        ha, va = "center", "bottom"
    elif 135 <= deg < 225:
        ha, va = "right", "center"
    elif 225 <= deg < 315:
        ha, va = "center", "top"
    else:
        ha, va = "left", "center"
    ax.text(ang, _LABEL_R, lab, ha=ha, va=va,
            fontsize=9.5, fontweight="normal", color="#1a1a1a")

# ----- compact legend below the radar -----
handles, labels = ax.get_legend_handles_labels()
ordered_hl = []
for n in LEGEND_ORDER:
    for h, l in zip(handles, labels):
        if l == n:
            ordered_hl.append((h, n))
            break
oh, onames = list(zip(*ordered_hl)) if ordered_hl else ([], [])
_LEGEND_NAMES = {
    "Ours (WEAVE)": "WEAVE",
    "IDT-TGT bound": "IDT--TGT",
    "Seedream 4.5": "Seedream",
    "StyleAligned": "StyleAligned",
    "Latent-WCT": "Latent-WCT",
}
olabels = [_LEGEND_NAMES.get(n, n) for n in onames]

# Legend sits in the right margin (dedicated column, clear of the squashed radar);
# three-tier font weight mirrors the plot hierarchy.
leg = fig.legend(oh, olabels, loc="center left", bbox_to_anchor=(0.68, 0.50),
                 ncol=1, fontsize=9.5, frameon=False, handlelength=1.2,
                 handletextpad=0.25, labelspacing=0.18, columnspacing=0.0)
# Keep legend samples proportional to the plot line hierarchy.
for ln, n in zip(leg.get_lines(), onames):
    ln.set_linewidth(TIERS[n][1])
# Keep every legend label at the same visual weight.
for txt, n in zip(leg.get_texts(), onames):
    txt.set_fontweight("normal")
    txt.set_color("black")
    txt.set_fontsize(9.5)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight", pad_inches=0.01)
fig.savefig(OUT_PNG, dpi=400, bbox_inches="tight", pad_inches=0.01)
print("WROTE %s" % OUT)
print("DINO cells available: %d" % len(DINO))


# ======== WEAVE weakness analysis ============================================
OURS = "Ours (WEAVE)"
ours_flat = raw_axes(OURS)    # weakness analysis uses RAW values, not the v/v_max figure
# axis order (incl. per-metric dataset swap) -> (metric_index, dataset_index)
axis_specs = [(mi, d) for mi, (g, kind) in enumerate(METRICS) for d in _ds_idx(kind)]
axis_names = [f"{METRICS[mi][0]}({DATASETS[d]})" for mi, d in axis_specs]
metric_per_axis = [METRICS[mi][0] for mi, d in axis_specs]
ax_ds = [DATASETS[d] for mi, d in axis_specs]

print("\n" + "=" * 76)
print("WEAVE weakness analysis (per axis, vs tier-A baselines)")
print("=" * 76)
for i, aname in enumerate(axis_names):
    ov = ours_flat[i]
    if isinstance(ov, float) and math.isnan(ov):
        print(f"\n  [{i+1}] {aname:<20} WEAVE=NaN (no DINO data)")
        continue
    worse = []
    for name in BOLD:
        if name == OURS:
            continue
        tv = raw_axes(name)[i]
        if isinstance(tv, float) and math.isnan(tv):
            continue
        gap = ov - tv
        if gap < -0.02:
            worse.append((gap, name, metric_per_axis[i], ax_ds[i], ov, tv))
    worse.sort(key=lambda x: x[0])
    print(f"\n  [{i+1}] {aname:<20} WEAVE={ov:.3f}")
    if not worse:
        print("         [OK] no meaningful deficit vs tier-A baselines")
    else:
        for gap, nm, rn, ds_val, ro, rt in worse:
            print(f"         [!] loses to {nm:<14} by {abs(gap):.3f}  ({rn}={rt:.3f} vs ours {ro:.3f})")

print("\n--- Summary ---")
for m_idx, (metric, _) in enumerate(METRICS):
    losses_per_ds = []
    for d_idx, ds in enumerate(DATASETS):
        aidx = next(i for i, (mi, d) in enumerate(axis_specs) if mi == m_idx and d == d_idx)
        count = sum(1 for nm in BOLD if nm != OURS
                    and not (isinstance(raw_axes(nm)[aidx], float) and math.isnan(raw_axes(nm)[aidx]))
                    and raw_axes(nm)[aidx] > ours_flat[aidx] + 0.02)
        if count:
            losses_per_ds.append(f"{ds}(-{count})")
    if losses_per_ds:
        print(f"  * {metric}: weak on {', '.join(losses_per_ds)}")

print("\nNote: LPIPS column in paper uses down-arrow (lower is better). "
      "On radar we plot 1-LPIPS so higher = always better. DINO-C/DINO-S higher = better.\n"
      "Metric families use one shared min-max range across all reported methods and datasets; "
      "the best reported value reaches the outer ring.\n"
      "The weakness analysis above uses raw values.\n"
      "Train/Infer speed axes use min-max normalization after -log10(1+t), with t in seconds. Missing "
      "(training-free / API) cells are gaps; no fill crosses them.")
