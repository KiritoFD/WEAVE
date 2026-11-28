"""Compare Cross600 and All750 using cached scores, without model inference."""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Annotation
from matplotlib.transforms import Bbox
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.eval.pareto import frontier_indices, is_public_literature

OUT = Path(__file__).resolve().parent
FIGURES = OUT / "figures"
SNAPSHOT = OUT / "comparison.json"
BASELINE_SNAPSHOT = ROOT / "docs/baseline/evaluation/paper_rgb_v1_aggregates.json"
SCOPES = ("cross600", "all750")
CORE_COLORS = {"P": "#d54f38", "E": "#138657"}
ENVELOPE_COLOR = "#263746"
BLUE = "#3f77aa"
PURPLE = "#8051b0"
NAMES = (
    "AdaIN", "CUT", "SaMST", "SaMAM", "SD-Turbo", "StyleID",
    "StyleShot", "StyTr2", "AesPA", "Seedream 4.5",
)
QWEN_SUMMARY = ROOT / "docs/archive/historical/v28/qwen_turbo_cross600_summary.json"
QWEN_EVAL = ROOT / "docs/baseline/QwenImage21Turbo/eval/eval.json"
IDENTITY_DINO_S = {"cross600": 0.36329345845617356, "all750": 0.41159399808198216}
IDENTITY_CSD_TOP1 = {"cross600": 0.5, "all750": 20.0}
TARGET_COPY_DINO_C = 0.2190


def log_composite(dc: float, ds: float, t1_pct: float) -> float:
    """Balanced log-composite score: log(DINO-C) + 0.5 * (log(DINO-S) + log(CSD Top-1))."""
    return float(
        np.log(max(1e-4, float(dc)))
        + 0.5 * (np.log(max(1e-4, float(ds))) + np.log(max(1e-4, float(t1_pct) / 100.0)))
    )


IDENTITY_LOG_COMP = {
    "cross600": log_composite(1.0, IDENTITY_DINO_S["cross600"], IDENTITY_CSD_TOP1["cross600"]),
    "all750": log_composite(1.0, IDENTITY_DINO_S["all750"], IDENTITY_CSD_TOP1["all750"]),
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aggregate(rows, scope):
    selected = [row for row in rows if scope == "all750" or row["is_cross"]]
    if scope == "same150":
        selected = [row for row in rows if not row["is_cross"]]
    dc = float(np.mean([row["dino_c"] for row in selected]))
    ds = float(np.mean([row["dino_s"] for row in selected]))
    t1 = 100.0 * float(np.mean([
        row.get("top1", row.get("target_top1")) for row in selected
    ]))
    return {
        "n": len(selected),
        "dc": dc,
        "ds": ds,
        "t1": t1,
        "slog": log_composite(dc, ds, t1),
    }


def latencies():
    source = ROOT / "exp/v32_nodino/plot_v32_pareto.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "EXTERNAL_BASELINES"
            for target in statement.targets
        ):
            return {
                item[0].split(" (")[0].replace("AesPA-Net", "AesPA"): 1.0 / item[4]
                for item in ast.literal_eval(statement.value)
            }
    raise ValueError("External latency metadata not found")


def collect():
    methods = read(ROOT / "runs/v6/evaluator_v2_all_results.json")["methods_evaluated"]
    speeds = latencies()
    baselines = []
    for name in NAMES:
        pack = methods[name]["paper_rgb_v1"]
        scopes = {}
        for scope in ("cross600", "same150", "all750"):
            row = pack[scope]
            dc, ds, t1 = row["DINO_C"], row["DINO_S"], row["Target_Top1"] * 100
            scopes[scope] = {
                "n": row["n"], "dc": dc, "ds": ds,
                "t1": t1, "slog": log_composite(dc, ds, t1),
            }
        baselines.append({
            "name": name, "scopes": scopes, "ips": speeds.get(name),
            "literature": is_public_literature(name),
            "source": "runs/v6/evaluator_v2_all_results.json",
        })
    for name, relative in (
        ("StyleGallery", "docs/baseline/StyleGallery/eval_stylegallery.json"),
        ("Hist2Style", "docs/baseline/Hist2Style/eval/eval.json"),
    ):
        path = ROOT / relative
        rows = read(path)["rows"]
        baselines.append({
            "name": name, "scopes": {scope: aggregate(rows, scope)
                                      for scope in ("cross600", "same150", "all750")},
            "ips": speeds.get(name), "literature": True, "source": relative,
            "sha256": digest(path),
        })
    qwen = read(QWEN_EVAL)
    qwen_summary = read(QWEN_SUMMARY)
    qwen_rows = qwen["rows"]
    qwen_task_ids = {row["task_id"] for row in qwen_rows}
    assert qwen["n_total"] == 750 and qwen["n_cross"] == 600 and qwen["n_same"] == 150
    assert len(qwen_rows) == len(qwen_task_ids) == 750
    baselines.append({
        "name": "Qwen-Image-2.1-Turbo",
        "scopes": {scope: aggregate(qwen_rows, scope) for scope in ("cross600", "same150", "all750")},
        "ips": qwen_summary["timing"]["img_per_sec"],
        "literature": False,
        "source": "docs/baseline/QwenImage21Turbo/eval/eval.json",
        "sha256": digest(QWEN_EVAL),
        "hardware": qwen_summary["hardware"],
    })
    for point in baselines:
        for metric in ("dc", "ds", "t1"):
            cross, same, total = [point["scopes"][scope]
                                  for scope in ("cross600", "same150", "all750")]
            assert cross["n"] + same["n"] == total["n"]
            expected = (cross[metric] * cross["n"] + same[metric] * same["n"]) / total["n"]
            assert abs(expected - total[metric]) < 1e-6, (point["name"], metric)
    write(BASELINE_SNAPSHOT, {
        "manifest": "docs/baseline/README.md", "protocol": "paper_rgb_v1",
        "source_summary_sha256": digest(ROOT / "runs/v6/evaluator_v2_all_results.json"),
        "methods": baselines,
    })
    source = ROOT / "runs/cross_all750_comparison/cleave_rows.json"
    ours = []
    for record in read(source):
        arm = record["arm"]
        if arm == "P9_E_A0_l000":
            continue
        assert len(record["rows"]) == 750
        assert len({row["task_id"] for row in record["rows"]}) == 750
        ours.append({
            "name": arm, "branch": "P" if arm.startswith("P8") else "E",
            "knob": arm.rsplit("_", 1)[1],
            "scopes": {scope: aggregate(record["rows"], scope)
                       for scope in ("cross600", "same150", "all750")},
            "ips": 14.3 if arm.startswith("P8") else 33.1,
            "source": record["source"], "sha256": record["sha256"],
        })
    payload = {
        "protocol": "same cached outputs; Cross600 vs Cross600+Same150",
        "ours": ours, "baselines": baselines,
        "speed_note": "Historical reported latency metadata, unchanged across scopes. "
                      "Ours: RTX4090 loaded RGB-to-RGB B16; external timing setups differ. "
                      "Not process-cold or a controlled hardware speedup comparison.",
    }
    write(SNAPSHOT, payload)
    return payload


def points(data, scope, key):
    selected = []
    for point in data[key]:
        scope_values = point["scopes"][scope]
        if scope_values["n"]:
            row = {**point, **scope_values}
            if "slog" not in row:
                row["slog"] = log_composite(row["dc"], row["ds"], row["t1"])
            selected.append(row)
    return selected


def front(points_in, xkey, ykey):
    return [points_in[index] for index in frontier_indices([
        (point[xkey], point[ykey]) for point in points_in
    ])]


def interpolate_best(ours, content, metric):
    candidates = []
    for branch in ("E", "P"):
        curve = sorted([point for point in ours if point["branch"] == branch], key=lambda point: point["dc"])
        if curve[0]["dc"] <= content <= curve[-1]["dc"]:
            candidates.append(float(np.interp(content, [point["dc"] for point in curve],
                                              [point[metric] for point in curve])))
    return max(candidates) if candidates else None


def summarize(data):
    result = {}
    for scope in SCOPES:
        ours = points(data, scope, "ours")
        baselines = points(data, scope, "baselines")
        sg = next(point for point in baselines if point["name"] == "StyleGallery")
        matched = {metric: interpolate_best(ours, sg["dc"], metric) for metric in ("ds", "t1")}
        result[scope] = {
            "stylegallery": {key: sg[key] for key in ("n", "dc", "ds", "t1")},
            "ours_at_stylegallery_content_interpolated": matched,
            "delta_s_at_stylegallery_content": matched["ds"] - sg["ds"],
            "delta_top1_pp_at_stylegallery_content": matched["t1"] - sg["t1"],
            "dominated_baselines": {},
        }
        for metric in ("ds", "t1"):
            result[scope]["dominated_baselines"][metric] = [
                base["name"] for base in baselines if base["literature"] and any(
                    point["dc"] >= base["dc"] and point[metric] >= base[metric]
                    and (point["dc"] > base["dc"] or point[metric] > base[metric]) for point in ours
                )
            ]
    return result


def _auto_bounds(values: list[float], *, log: bool = False, pad_lo: float = 0.06, pad_hi: float = 0.08) -> tuple[float, float]:
    lo, hi = min(values), max(values)
    if log:
        llo, lhi = np.log10(lo), np.log10(hi)
        span = max(1e-3, lhi - llo)
        return float(10 ** (llo - pad_lo * span)), float(10 ** (lhi + pad_hi * span))
    span = max(1e-6, hi - lo)
    return float(lo - pad_lo * span), float(hi + pad_hi * span)


def _place_labels(figure, axes):
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    for axis in axes:
        labels = [text for text in axis.texts if isinstance(text, Annotation)
                  and text.xycoords == "data" and text.get_text()]
        occupied = [text.get_window_extent(renderer).padded(2) for text in axis.texts
                    if text not in labels and text.get_visible() and text.get_text()]
        legend = axis.get_legend()
        if legend is not None:
            occupied.append(legend.get_window_extent(renderer).padded(2))
        markers = []
        for collection in axis.collections:
            for coordinate in collection.get_offsets():
                pixel = collection.get_offset_transform().transform(coordinate)
                markers.append(Bbox.from_extents(pixel[0] - 5, pixel[1] - 5,
                                                pixel[0] + 5, pixel[1] + 5))
        bounds = axis.get_window_extent(renderer)
        for label in labels:
            original = label.get_position()
            candidates = [original]
            for distance in (7, 14, 22, 32, 45, 60, 78):
                candidates.extend((horizontal * distance, vertical * distance)
                                  for horizontal, vertical in ((1, 0), (-1, 0), (0, 1), (0, -1),
                                                               (1, 1), (-1, 1), (1, -1), (-1, -1)))
            best = None
            for offset in candidates:
                label.set_position(offset)
                label.set_horizontalalignment("left" if offset[0] > 0 else "right"
                                              if offset[0] < 0 else "center")
                label.set_verticalalignment("bottom" if offset[1] >= 0 else "top")
                box = label.get_window_extent(renderer).padded(2)
                outside = max(0, bounds.x0 + 2 - box.x0) + max(0, box.x1 - bounds.x1 + 2)
                outside += max(0, bounds.y0 + 2 - box.y0) + max(0, box.y1 - bounds.y1 + 2)
                collisions = sum(box.overlaps(block) for block in occupied)
                marker_hits = sum(box.overlaps(marker) for marker in markers)
                score = outside * 10000 + collisions * 100000 + marker_hits * 3000
                score += float(np.hypot(*offset)) + .2 * float(np.hypot(
                    offset[0] - original[0], offset[1] - original[1]))
                if best is None or score < best[0]:
                    best = (score, offset, box, label.get_ha(), label.get_va())
            _, offset, box, horizontal_alignment, vertical_alignment = best
            label.set_position(offset)
            label.set_ha(horizontal_alignment)
            label.set_va(vertical_alignment)
            occupied.append(box)
            if np.hypot(*offset) > 18:
                axis.annotate("", xy=label.xy, xytext=offset, textcoords="offset points",
                              arrowprops={"arrowstyle": "-", "color": label.get_color(),
                                          "linewidth": .55, "alpha": .7, "shrinkA": 2,
                                          "shrinkB": 5}, zorder=4)
        overlaps = sum(left.overlaps(right) for index, left in enumerate(occupied)
                       for right in occupied[index + 1:])
        print(f"Label layout: panel {axes.index(axis) + 1}, overlapping boxes={overlaps}")


def draw(data, scope):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False})
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.9))
    ours = points(data, scope, "ours")
    baselines = points(data, scope, "baselines")
    layout = (("dc", "ds", "DINO-C", "DINO-S", "(a) Content vs style similarity"),
              ("dc", "t1", "DINO-C", "CSD Top-1 (%)", "(b) Content vs style specificity"),
              ("ips", "slog", "Reported throughput (img/s, log)",
               "Log-composite score", "(c) Throughput vs balanced score"))
    for panel, (axis, (xkey, ykey, xlabel, ylabel, title)) in enumerate(zip(axes, layout)):
        available = [point for point in baselines if point.get(xkey) is not None]
        off_scale = [point for point in available if panel == 0 and point["name"] == "SaMST"]
        available = [point for point in available if point not in off_scale]
        displayed_ours = ours if panel < 2 else [
            max((point for point in ours if point["branch"] == branch), key=lambda point: point[ykey])
            for branch in CORE_COLORS
        ]
        public = [point for point in available if point["literature"]]
        edge = front(public, xkey, ykey)
        axis.plot([point[xkey] for point in edge], [point[ykey] for point in edge],
                  "--", color="#7b8a9a", linewidth=1.4, zorder=1)
        best = front(displayed_ours, xkey, ykey)
        for branch, color in CORE_COLORS.items():
            curve = sorted([point for point in displayed_ours if point["branch"] == branch],
                           key=lambda point: int(point["knob"][1:]))
            if panel < 2:
                axis.plot([point[xkey] for point in curve], [point[ykey] for point in curve],
                          color=color, linewidth=2.0, marker="o" if branch == "P" else "D",
                          markersize=3.8, markeredgecolor="white", markeredgewidth=.5, zorder=3)
        axis.plot([point[xkey] for point in best], [point[ykey] for point in best],
                  color=ENVELOPE_COLOR, linewidth=1.5, linestyle=(0, (4, 3)), zorder=4)
        if panel < 2:
            if panel == 0:
                identity_floor = IDENTITY_DINO_S[scope]
                axis.axhline(identity_floor, color="#c23b3b", linewidth=1.1,
                             linestyle=(0, (5, 3)), alpha=.8, zorder=2)
                axis.annotate(f"idt-style {identity_floor:.3f}",
                              xy=(.99, identity_floor), xycoords=("axes fraction", "data"),
                              xytext=(-2, 5), textcoords="offset points", ha="right",
                              fontsize=7.5, color="#a83232")
            else:
                identity_csd = IDENTITY_CSD_TOP1[scope]
                axis.axhline(identity_csd, color="#c23b3b", linewidth=1.1,
                             linestyle=(0, (5, 3)), alpha=.8, zorder=2)
                axis.annotate(f"idt-CSD {identity_csd:.1f}%",
                              xy=(0.43, identity_csd), xycoords=("data", "data"),
                              xytext=(0, 4), textcoords="offset points", ha="center",
                              va="bottom", fontsize=7.5, color="#a83232")
            axis.axvline(TARGET_COPY_DINO_C, color="#c23b3b", linewidth=1.1,
                         linestyle=(0, (5, 3)), alpha=.8, zorder=2)
            axis.annotate("tgt-content .219",
                          xy=(TARGET_COPY_DINO_C, .85), xycoords=("data", "axes fraction"),
                          xytext=(4, -3), textcoords="offset points", rotation=90,
                          va="top", fontsize=7.2, color="#a83232")
        for point in available:
            special = point["name"] == "StyleGallery"
            commercial = point["name"] == "Seedream 4.5"
            qwen = point["name"] == "Qwen-Image-2.1-Turbo"
            color = PURPLE if special or commercial else "#b45f06" if qwen else BLUE
            axis.scatter(point[xkey], point[ykey], s=125 if special else 67 if qwen else 39,
                         marker="*" if special else "D" if commercial or qwen else "s",
                         facecolors=color, edgecolors=color, linewidths=.7, zorder=5)
            label = point["name"]
            if label == "AesPA":
                label = "AesPA-Net"
            if point["n"] != (600 if scope == "cross600" else 750):
                label += "†"
            offset = (5, 6)
            if label.startswith("StyleID"):
                offset = (5, -13)
            if label.startswith("Seedream"):
                offset = (-5, -14)
            if label.startswith("StyTr2"):
                offset = (-5, 7)
            if label in ("CUT", "SaMAM", "SD-Turbo", "Hist2Style"):
                offset = (-5, {"CUT": -14, "SaMAM": 6, "SD-Turbo": 7, "Hist2Style": -15}[label])
            if panel == 0:
                offset = {"CUT": (-6, -18), "Hist2Style": (-5, -31),
                          "SaMAM": (-7, 9), "AesPA-Net": (5, -14),
                          "StyTr2": (-9, 7), "StyleGallery": (6, 12),
                          "Qwen-Image-2.1-Turbo": (-6, -12)}.get(label, offset)
            if panel == 1:
                offset = {"CUT": (-6, -18), "Hist2Style": (-6, 11),
                          "SaMAM": (-8, 6), "SD-Turbo": (-5, -12),
                          "StyleGallery": (-5, -15),
                          "Qwen-Image-2.1-Turbo": (6, -5)}.get(label, offset)
            if panel == 2 and label in ("StyleGallery", "Qwen-Image-2.1-Turbo", "Seedream 4.5",
                                         "StyTr2", "StyleShot", "StyleID", "AesPA-Net",
                                         "Hist2Style", "CUT", "SaMAM"):
                panel2_offs = {
                    "StyleGallery": (3, 8),
                    "Qwen-Image-2.1-Turbo": (6, 2),
                    "Seedream 4.5": (5, -8),
                    "StyleShot": (5, -8),
                    "StyTr2": (5, 5),
                    "StyleID": (5, -7),
                    "AesPA-Net": (5, 5),
                    "Hist2Style": (-5, 5),
                    "CUT": (6, -3),
                    "SaMAM": (0, -11),
                }
                offset = panel2_offs[label]
            elif panel == 2:
                offset = (5, 4)
            ha_lbl = "center" if (panel == 2 and offset[0] == 0) else ("right" if offset[0] < 0 else "left")
            axis.annotate(label, (point[xkey], point[ykey]), xytext=offset,
                          textcoords="offset points", ha=ha_lbl,
                          fontsize=8.1, color=color, fontweight="bold" if special else "normal")
        if panel == 2:
            axis.axhline(IDENTITY_LOG_COMP[scope], color="#c23b3b", linewidth=1.1,
                         linestyle=(0, (5, 3)), alpha=.75, zorder=2)
            if IDENTITY_LOG_COMP[scope] > -2.65:
                axis.annotate(f"idt-score {IDENTITY_LOG_COMP[scope]:.2f}",
                              xy=(0.74, IDENTITY_LOG_COMP[scope]), xycoords=("axes fraction", "data"),
                              xytext=(-2, 3), textcoords="offset points", ha="right", va="bottom",
                              fontsize=7.3, color="#a83232")
        for point in best:
            axis.scatter(point[xkey], point[ykey], s=36, marker="o" if point["branch"] == "P" else "D",
                         c=CORE_COLORS[point["branch"]], edgecolors="white", linewidths=.65, zorder=6)
        for point in displayed_ours:
            label_knobs = ("s085", "s100", "s150", "l025", "l100")
            if panel == 2 or point["knob"] in label_knobs:
                amount = int(point["knob"][1:]) / 100
                label = f"P s={amount:.2f}" if point["branch"] == "P" else f"E λ={amount:.2f}"
                if panel == 2:
                    label += " (best)"
                offset = (8, 13) if point["branch"] == "E" else (6, 7)
                if point["knob"] == "s150" or panel == 0 and point["knob"] == "s100":
                    offset = (-7, -14)
                if panel == 2:
                    offset = (-6, 5) if point["branch"] == "P" else (6, -3)
                axis.annotate(label, (point[xkey], point[ykey]), xytext=offset,
                              textcoords="offset points", fontsize=8.1, color=CORE_COLORS[point["branch"]],
                              ha="right" if offset[0] < 0 else "left", fontweight="bold")
        axis.text(.025, .975, f"{scope.capitalize()} · {title}" if panel == 0 else title,
                  transform=axis.transAxes, fontsize=10, fontweight="bold", va="top")
        axis.set_xlabel(xlabel + " ↑", fontweight="bold")
        axis.set_ylabel((ylabel + " ↑") if "↑" not in ylabel else ylabel, fontweight="bold")
        axis.grid(alpha=.23, linestyle="--", linewidth=.6)
        if panel == 2:
            axis.set_xscale("log")
            all_x = [p[xkey] for p in available] + [p[xkey] for p in displayed_ours]
            all_y = [p[ykey] for p in available] + [p[ykey] for p in displayed_ours]
            if IDENTITY_LOG_COMP[scope] >= min(all_y) - 0.15:
                all_y.append(IDENTITY_LOG_COMP[scope])
            axis.set_xlim(*_auto_bounds(all_x, log=True, pad_lo=0.07, pad_hi=0.09))
            axis.set_ylim(*_auto_bounds(all_y, log=False, pad_lo=0.08, pad_hi=0.18))
        else:
            all_x = [p[xkey] for p in available] + [p[xkey] for p in ours] + [TARGET_COPY_DINO_C]
            all_y = [p[ykey] for p in available] + [p[ykey] for p in ours]
            if panel == 0:
                all_y.append(IDENTITY_DINO_S[scope])
            else:
                all_y.append(IDENTITY_CSD_TOP1[scope])
            axis.set_xlim(*_auto_bounds(all_x, log=False, pad_lo=0.05, pad_hi=0.06))
            axis.set_ylim(*_auto_bounds(all_y, log=False, pad_lo=0.07, pad_hi=0.18))
        if off_scale:
            point = off_scale[0]
            axis.scatter(-.045, -.05, transform=axis.transAxes, marker="s", s=33,
                         color=BLUE, clip_on=False, zorder=5)
            axis.annotate(f"SaMST off-scale\nC={point['dc']:.3f}, S={point['ds']:.3f}",
                          xy=(-.045, -.05), xycoords="axes fraction", xytext=(8, -2),
                          textcoords="offset points", fontsize=7, color=BLUE,
                          ha="left", va="top", annotation_clip=False)
    handles = [Line2D([], [], color=CORE_COLORS["P"], marker="o", lw=2, label="P-Core"),
               Line2D([], [], color=CORE_COLORS["E"], marker="D", lw=2, label="E-Core"),
               Line2D([], [], color=ENVELOPE_COLOR, ls=(0, (4, 3)), lw=1.5, label="Joint frontier"),
               Line2D([], [], color="#7b8a9a", ls="--", lw=1.4, label="Literature front")]
    axes[0].legend(handles=handles, ncol=2, loc="upper right", bbox_to_anchor=(.99, .9),
                   frameon=False, fontsize=8, handlelength=2)
    note = "† StyleShot 596/600 · Seedream 576/600" if scope == "cross600" else "† StyleShot 745/750 · Seedream 720/750"
    axes[1].text(.025, .02, note, transform=axes[1].transAxes, fontsize=7.2, color="#526071")
    axes[2].text(.025, .02, "Reported timings; hardware / batches differ", transform=axes[2].transAxes,
                 fontsize=7.2, color="#526071")
    figure.subplots_adjust(left=.055, right=.99, top=.99, bottom=.19, wspace=.31)
    _place_labels(figure, list(axes))
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("svg", "png"):
        path = FIGURES / f"{scope}_triptych.{suffix}"
        figure.savefig(path, dpi=190, facecolor="white")
        if suffix == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    plt.close(figure)


def main():
    data = collect() if "--refresh" in sys.argv or not SNAPSHOT.exists() else read(SNAPSHOT)
    data["comparison"] = summarize(data)
    write(SNAPSHOT, data)
    for scope in SCOPES:
        draw(data, scope)
    html = '''<!doctype html><html><head><meta charset="utf-8"><style>
    body{margin:0;font-family:var(--font-sans,sans-serif);color:var(--foreground)}
    svg{display:block;width:100%;height:auto;margin-bottom:12px}
    </style></head><body><img src="cross600_triptych.svg">
    <img src="all750_triptych.svg"></body></html>'''
    for scope in SCOPES:
        svg = (FIGURES / f"{scope}_triptych.svg").read_text(encoding="utf-8")
        html = html.replace(f'<img src="{scope}_triptych.svg">', svg[svg.index("<svg"):])
    (FIGURES / "comparison.html").write_text(html, encoding="utf-8")
    csv_path = OUT / "metrics.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("name", "group", "scope", "n", "dc", "ds", "t1", "slog", "ips"))
        writer.writeheader()
        for group in ("ours", "baselines"):
            for point in data[group]:
                for scope in ("cross600", "same150", "all750"):
                    writer.writerow({"name": point["name"], "group": group, "scope": scope,
                                     **point["scopes"][scope], "ips": point.get("ips")})
    print(json.dumps(data["comparison"], ensure_ascii=False, indent=2))
    print("Wrote", FIGURES)


if __name__ == "__main__":
    main()
