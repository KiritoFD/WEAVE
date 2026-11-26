"""Generate consolidated Pareto frontier visualizations (SVG + interactive HTML) for CLEAVE V32–V34+.

Reads verified metrics strictly from disk:
  - docs/performance/metrics.csv (public literature + baseline points via src.eval.pareto)
  - runs/v32_nodino/v32_final_ledger.json (or embedded fallback when ledger is remote)
  - docs/v33/data/p2_report.json, docs/v33/data/p2_cfg_probe_report.json
  - docs/v34/data/*.json (Stages A, B, C, D, C2+A3 control, Holdout-750)
  - Any future docs/v*/data/*.json registered via series config

Usage:
  python docs/performance/plot_cleave_pareto.py
"""
from __future__ import annotations

import csv
import html
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.eval.pareto import frontier_indices, is_public_literature

FIG_DIR = REPO / "docs/performance/figures"

# Verified V32 ConvNeXt Zero-DINO ledger points (fallback if runs/v32_nodino is only on remote 4090)
V32_FALLBACK_SPECTRUM = [
    {"label": "ECore_l075", "dino_c": 0.7204, "dino_s": 0.4671, "csd_top1": 0.4517},
    {"label": "PCore_s060", "dino_c": 0.8140, "dino_s": 0.4210, "csd_top1": 0.3150},
    {"label": "PCore_s075", "dino_c": 0.7049, "dino_s": 0.4947, "csd_top1": 0.4717},
    {"label": "PCore_s075_cal_c10", "dino_c": 0.6585, "dino_s": 0.5087, "csd_top1": 0.5233},
    {"label": "PCore_s085", "dino_c": 0.6440, "dino_s": 0.5208, "csd_top1": 0.5450},
    {"label": "PCore_s085_c10", "dino_c": 0.6010, "dino_s": 0.5266, "csd_top1": 0.6233},
    {"label": "PCore_s093_cal_c10", "dino_c": 0.5524, "dino_s": 0.5446, "csd_top1": 0.6567},
    {"label": "PCore_s100", "dino_c": 0.4896, "dino_s": 0.5628, "csd_top1": 0.7517},
    {"label": "PCore_s125", "dino_c": 0.4171, "dino_s": 0.5814, "csd_top1": 0.7733},
    {"label": "PCore_s135", "dino_c": 0.3772, "dino_s": 0.5919, "csd_top1": 0.7733},
    {"label": "PCore_s150", "dino_c": 0.2805, "dino_s": 0.6054, "csd_top1": 0.8017},
]


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_curve(levels: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    pts = []
    for k, m in levels.items():
        if "wrong" in k:
            continue
        pts.append({
            "label": k,
            "dino_c": round(float(m["dino_c"]), 4),
            "dino_s": round(float(m["dino_s"]), 4),
            "csd_top1": round(float(m["csd_top1"]), 4),
            "lpips": round(float(m.get("lpips", 0.0)), 4),
            "same150_top1": round(float(m.get("same150_top1", 0.0)), 4),
        })
    return sorted(pts, key=lambda p: p["dino_c"])


def collect_pareto_dataset() -> dict[str, Any]:
    """Assemble external literature, V32, V33, and V34 Cross600 + Holdout points."""
    # 1. Public literature & baseline reference points from metrics.csv
    ext_points = []
    other_refs = []
    csv_p = REPO / "docs/performance/metrics.csv"
    with csv_p.open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            arm = r["arm"].strip()
            base = arm.split("_")[0].split(" ")[0]
            try:
                dc = float(r["dino_c"])
                ds = float(r["dino_s"])
                top1 = float(r.get("target_top1") or 0.0)
            except ValueError:
                continue
            if not (0.15 <= dc <= 0.92 and 0.25 <= ds <= 0.70):
                continue
            pt = {"label": arm, "dino_c": round(dc, 4), "dino_s": round(ds, 4), "csd_top1": round(top1, 4)}
            if is_public_literature(arm) or is_public_literature(base) or "StyleGallery" in arm:
                ext_points.append(pt)
            elif arm in ("Seedream 4.5", "WEAVE (Published)"):
                other_refs.append(pt)

    ext_points.sort(key=lambda p: p["dino_c"])
    ext_front_idx = frontier_indices([(p["dino_c"], p["dino_s"]) for p in ext_points])
    ext_frontier = [ext_points[i] for i in ext_front_idx]

    # 2. V32 baseline curve
    v32_ledger = _load_json(REPO / "runs/v32_nodino/v32_final_ledger.json")
    if v32_ledger and "spectrum" in v32_ledger:
        v32_pts = _extract_curve({k: v for k, v in v32_ledger["spectrum"].items() if k.startswith("PCore_")})
    else:
        v32_pts = [p for p in V32_FALLBACK_SPECTRUM if p["label"].startswith("PCore_")]

    # 3. V33 P2_S2 + CFG probe
    p2_rep = _load_json(REPO / "docs/v33/data/p2_report.json")
    v33_pts = _extract_curve(p2_rep["arms"]["P2_S2_hi32_pert30"]["levels"]) if p2_rep else []

    # 4. V34 Stages A, B, C, D + C2+A3 control
    sa = _load_json(REPO / "docs/v34/data/stage_a_report.json")
    sb = _load_json(REPO / "docs/v34/data/stage_b_report.json")
    sc = _load_json(REPO / "docs/v34/data/stage_c_report.json")
    sd = _load_json(REPO / "docs/v34/data/stage_d_report.json")
    c2a3 = _load_json(REPO / "docs/v34/data/c2_a3_control_report.json")

    series: list[dict[str, Any]] = [
        {"id": "ext_front", "name": "External Literature Frontier", "group": "external", "style": "dashed", "color": "#64748b", "points": ext_frontier},
        {"id": "v32", "name": "V32 PCore_NoDINO (22M)", "group": "baseline", "style": "solid", "color": "#3b82f6", "points": v32_pts},
    ]
    if v33_pts:
        series.append({"id": "v33_s2", "name": "V33 P2_S2 (22M+7M frozen)", "group": "v33", "style": "dashed", "color": "#8b5cf6", "points": v33_pts})

    # V34 Stage A zero-train startpoint points (A0, A1, A2, A3 at s085 & s100)
    if sa:
        for mode, col in (("A0", "#94a3b8"), ("A1", "#14b8a6"), ("A2", "#f59e0b"), ("A3", "#ec4899")):
            pts = []
            for kl in ("s100", "s085"):
                m = sa["arms"].get(f"StageA_{mode}_{kl}")
                if m:
                    pts.append({"label": f"{mode}_{kl}", "dino_c": round(float(m["dino_c"]), 4), "dino_s": round(float(m["dino_s"]), 4), "csd_top1": round(float(m["csd_top1"]), 4), "min_s": round(float(m["per_tgt_dino_s"]["Min"]), 4)})
            if pts:
                series.append({"id": f"v34_stage_{mode.lower()}", "name": f"V34 Stage A ({mode})", "group": "stage_a", "style": "dotted", "color": col, "points": sorted(pts, key=lambda p: p["dino_c"])})

    # V34 Stage B continuous conditioning arms
    if sb:
        for arm_k, label, col in (
            ("B0_k8_discrete", "V34 B0 (K=8 Discrete)", "#0ea5e9"),
            ("B1_inst_417", "V34 B1 (Inst 417-D)", "#f97316"),
            ("B2_inst_tex1101", "V34 B2 (Inst 1101-D+Tex)", "#ef4444"),
        ):
            if arm_k in sb["arms"]:
                series.append({"id": f"v34_{arm_k}", "name": label, "group": "stage_b", "style": "solid", "color": col, "points": _extract_curve(sb["arms"][arm_k]["levels"])})

    # V34 Stage C 2x2 arms
    if sc:
        for arm_k, label, col in (
            ("C0_22m_q32", "V34 C0 (22M + Q32)", "#06b6d4"),
            ("C1_22m_s2_joint", "V34 C1 (22M+S2 Joint)", "#a855f7"),
            ("C2_44m_q32", "V34 C2 (44M + Q32)", "#10b981"),
            ("C3_44m_s2_joint", "V34 C3 (44M+S2 Joint)", "#c084fc"),
        ):
            if arm_k in sc["arms"]:
                series.append({"id": f"v34_{arm_k}", "name": label, "group": "stage_c", "style": "solid", "color": col, "points": _extract_curve(sc["arms"][arm_k]["levels"])})

    # V34 Stage D & C2+A3 winner frontier
    if sd:
        series.append({"id": "v34_d_merged", "name": "V34 D_merged (44M+A3 Bridge)", "group": "stage_d", "style": "dashed", "color": "#eab308", "points": _extract_curve(sd["test750_levels"])})
    if c2a3:
        series.append({"id": "v34_winner", "name": "V34 Winner (44M C2 + A3 Start)", "group": "winner", "style": "solid", "color": "#22c55e", "width": 3.2, "points": _extract_curve(c2a3["levels"])})

    # Holdout-750 blind split comparison
    holdout_pts = []
    if sd and "holdout_750" in sd:
        for k, v in sd["holdout_750"].items():
            c = v["cross600"]
            holdout_pts.append({"label": k, "dino_c": round(c["dino_c"], 4), "dino_s": round(c["dino_s"], 4), "csd_top1": round(c["csd_top1"], 4), "lpips": round(c["lpips"], 4), "same150_top1": round(v["same150"]["csd_top1"], 4)})
    if c2a3 and "holdout" in c2a3:
        for k, v in c2a3["holdout"].items():
            c = v["cross600"]
            holdout_pts.append({"label": k, "dino_c": round(c["dino_c"], 4), "dino_s": round(c["dino_s"], 4), "csd_top1": round(c["csd_top1"], 4), "lpips": round(c["lpips"], 4), "same150_top1": round(v["same150"]["csd_top1"], 4)})

    return {
        "external_scatter": ext_points,
        "other_refs": other_refs,
        "series": series,
        "holdout": holdout_pts,
    }


def build_svg(data: dict[str, Any], *, metric_y: str = "dino_s") -> str:
    """Render clean static SVG for docs/performance/figures/cleave_pareto_latest.svg."""
    w, h = 860, 520
    pad_l, pad_r, pad_t, pad_b = 68, 220, 38, 56
    pw, ph = w - pad_l - pad_r, h - pad_t - pad_b
    x0, x1 = 0.25, 0.85
    if metric_y == "dino_s":
        y0, y1 = 0.36, 0.62
        y_label = "DINO-S (Style Similarity)"
    else:
        y0, y1 = 0.25, 0.95
        y_label = "CSD Top-1 Accuracy"

    def sx(v: float) -> float:
        return pad_l + (v - x0) / (x1 - x0) * pw

    def sy(v: float) -> float:
        return pad_t + ph - (v - y0) / (y1 - y0) * ph

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" style="background:#0f172a;font-family:system-ui,sans-serif">',
        f'<rect x="{pad_l}" y="{pad_t}" width="{pw}" height="{ph}" fill="#111827" stroke="#334155" stroke-width="1"/>',
        f'<text x="{pad_l}" y="22" fill="#f8fafc" font-size="14" font-weight="700">CLEAVE V32 -> V34 Pareto Frontier (Cross600, Zero-DINO, S10)</text>',
    ]
    for tick in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        px = sx(tick)
        lines.append(f'<line x1="{px:.1f}" y1="{pad_t}" x2="{px:.1f}" y2="{pad_t+ph}" stroke="#1e293b" stroke-width="1"/>')
        lines.append(f'<text x="{px:.1f}" y="{pad_t+ph+18}" fill="#94a3b8" font-size="11" text-anchor="middle">{tick:.1f}</text>')
    y_ticks = [0.40, 0.45, 0.50, 0.55, 0.60] if metric_y == "dino_s" else [0.30, 0.45, 0.60, 0.75, 0.90]
    for tick in y_ticks:
        py = sy(tick)
        lines.append(f'<line x1="{pad_l}" y1="{py:.1f}" x2="{pad_l+pw}" y2="{py:.1f}" stroke="#1e293b" stroke-width="1"/>')
        lbl = f"{tick*100:.0f}%" if metric_y == "csd_top1" else f"{tick:.2f}"
        lines.append(f'<text x="{pad_l-8}" y="{py+4:.1f}" fill="#94a3b8" font-size="11" text-anchor="end">{lbl}</text>')

    lines.append(f'<text x="{pad_l+pw/2:.1f}" y="{h-12}" fill="#cbd5e1" font-size="12" text-anchor="middle">DINO-C (Content Preservation)</text>')
    lines.append(f'<text x="18" y="{pad_t+ph/2:.1f}" fill="#cbd5e1" font-size="12" text-anchor="middle" transform="rotate(-90 18 {pad_t+ph/2:.1f})">{y_label}</text>')

    # External scatter
    for p in data["external_scatter"]:
        cx, cy = sx(p["dino_c"]), sy(p[metric_y])
        if pad_l <= cx <= pad_l + pw and pad_t <= cy <= pad_t + ph:
            lines.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4.2" fill="#94a3b8" opacity="0.8"/>')
            lines.append(f'<text x="{cx+6:.1f}" y="{cy-5:.1f}" fill="#cbd5e1" font-size="10">{html.escape(p["label"])}</text>')

    # Selected key curves in static SVG so it stays legible without clutter
    keep_ids = {"ext_front", "v32", "v33_s2", "v34_stage_a3", "v34_B2_inst_tex1101", "v34_C2_44m_q32", "v34_d_merged", "v34_winner"}
    ly = pad_t + 10
    for s in data["series"]:
        if s["id"] not in keep_ids or not s["points"]:
            continue
        pts = [p for p in s["points"] if x0 - 0.02 <= p["dino_c"] <= x1 + 0.02]
        if not pts:
            continue
        d_attr = " ".join(f"{'M' if i == 0 else 'L'}{sx(p['dino_c']):.1f},{sy(p[metric_y]):.1f}" for i, p in enumerate(pts))
        dash = ' stroke-dasharray="5,4"' if s["style"] == "dashed" else (' stroke-dasharray="2,3"' if s["style"] == "dotted" else "")
        sw = s.get("width", 2.0)
        lines.append(f'<path d="{d_attr}" fill="none" stroke="{s["color"]}" stroke-width="{sw}"{dash}/>')
        for p in pts:
            lines.append(f'<circle cx="{sx(p["dino_c"]):.1f}" cy="{sy(p[metric_y]):.1f}" r="{sw+1.4:.1f}" fill="{s["color"]}"/>')
        lines.append(f'<line x1="{pad_l+pw+14}" y1="{ly}" x2="{pad_l+pw+32}" y2="{ly}" stroke="{s["color"]}" stroke-width="{sw}"{dash}/>')
        lines.append(f'<text x="{pad_l+pw+38}" y="{ly+4}" fill="#e2e8f0" font-size="10.5">{html.escape(s["name"])}</text>')
        ly += 22

    lines.append("</svg>")
    return "\n".join(lines)


def build_interactive_html(data: dict[str, Any]) -> str:
    """Self-contained theme-aware HTML for html_preview / html_render."""
    payload_json = json.dumps(data, ensure_ascii=False)
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  font-family: var(--font-sans, system-ui, sans-serif);
  color: var(--foreground, #e2e8f0);
  padding: 0;
  margin: 0;
}}
.wrap {{ width: 100%; display: flex; flex-direction: column; gap: 10px; }}
.toolbar {{
  display: flex; flex-wrap: wrap; gap: 6px; align-items: center; justify-content: space-between;
}}
.tabs {{ display: flex; gap: 6px; flex-wrap: wrap; }}
.btn {{
  border: 1px solid var(--border, #334155);
  background: var(--secondary, #1e293b);
  color: var(--foreground, #e2e8f0);
  border-radius: var(--radius, 8px);
  padding: 5px 10px;
  font-size: 12px;
  cursor: pointer;
}}
.btn.active {{
  background: var(--primary, #3b82f6);
  color: var(--primary-foreground, #ffffff);
  border-color: var(--primary, #3b82f6);
  font-weight: 600;
}}
.card {{
  background: var(--card, #111827);
  border: 1px solid var(--border, #334155);
  border-radius: var(--radius, 8px);
  padding: 12px 14px;
}}
.legend {{
  display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px;
}}
.pill {{
  display: inline-flex; align-items: center; gap: 5px;
  padding: 3px 8px; border-radius: 999px; font-size: 11px;
  border: 1px solid var(--border, #334155);
  background: var(--secondary, #1e293b);
  color: var(--foreground, #e2e8f0);
  cursor: pointer; opacity: 1;
}}
.pill.off {{ opacity: 0.35; }}
.dot {{ width: 9px; height: 9px; border-radius: 50%; display: inline-block; }}
.info-bar {{
  font-family: var(--font-mono, monospace);
  font-size: 11px;
  color: var(--muted-foreground, #94a3b8);
  padding: 6px 10px;
  background: var(--muted, #0f172a);
  border-radius: var(--radius, 6px);
  min-height: 28px;
}}
svg {{ width: 100%; height: 390px; display: block; }}
.grid-line {{ stroke: var(--border, #334155); stroke-opacity: 0.45; stroke-width: 1; }}
.axis-text {{ fill: var(--muted-foreground, #94a3b8); font-size: 10.5px; }}
.ext-label {{ fill: var(--muted-foreground, #cbd5e1); font-size: 9.8px; paint-order: stroke; stroke: var(--card, #111827); stroke-width: 3px; stroke-linejoin: round; }}
</style>
</head>
<body>
<div class="wrap">
  <div class="toolbar">
    <div class="tabs">
      <button class="btn active" data-mode="dino_s">Cross600: DINO-C vs DINO-S</button>
      <button class="btn" data-mode="csd_top1">Cross600: DINO-C vs CSD Top-1</button>
      <button class="btn" data-mode="holdout">Independent Holdout-750</button>
    </div>
    <div class="tabs">
      <button class="btn active" data-preset="main">Main Frontiers</button>
      <button class="btn" data-preset="all">Show All V32-V34</button>
      <button class="btn" data-preset="ablation">Stage A/B/C Detail</button>
    </div>
  </div>
  <div class="card">
    <svg id="chart" viewBox="0 0 700 390" preserveAspectRatio="xMidYMid meet"></svg>
    <div class="info-bar" id="info">Hover any point for exact DINO-C / DINO-S / CSD Top-1 / LPIPS readout. Click legend pills to toggle curves.</div>
    <div class="legend" id="legend"></div>
  </div>
</div>
<script>
const DATA = {payload_json};
let curMode = "dino_s";
const PRESETS = {{
  main: ["ext_front", "v32", "v33_s2", "v34_stage_a3", "v34_C2_44m_q32", "v34_d_merged", "v34_winner"],
  all: DATA.series.map(s => s.id),
  ablation: ["v32", "v34_stage_a0", "v34_stage_a1", "v34_stage_a2", "v34_stage_a3", "v34_B0_k8_discrete", "v34_B1_inst_417", "v34_B2_inst_tex1101", "v34_C0_22m_q32", "v34_C1_22m_s2_joint", "v34_C2_44m_q32", "v34_C3_44m_s2_joint", "v34_winner"]
}};
const enabled = new Set(PRESETS.main);

const svg = document.getElementById("chart");
const info = document.getElementById("info");
const legendEl = document.getElementById("legend");

function el(tag, attrs, text) {{
  const n = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs || {{}})) n.setAttribute(k, v);
  if (text !== undefined) n.textContent = text;
  return n;
}}

function renderLegend() {{
  legendEl.innerHTML = "";
  if (curMode === "holdout") {{
    legendEl.innerHTML = '<span style="font-size:11px;color:var(--muted-foreground,#94a3b8)">Holdout-750: 150 disjoint unseen sources x 5 target styles scored against 150 disjoint unseen holdout references (Zero-DINO + S10).</span>';
    return;
  }}
  for (const s of DATA.series) {{
    const btn = document.createElement("span");
    btn.className = "pill" + (enabled.has(s.id) ? "" : " off");
    btn.innerHTML = `<span class="dot" style="background:${{s.color}}"></span>${{s.name}}`;
    btn.onclick = () => {{
      if (enabled.has(s.id)) enabled.delete(s.id); else enabled.add(s.id);
      render();
    }};
    legendEl.appendChild(btn);
  }}
}}

function render() {{
  renderLegend();
  svg.innerHTML = "";
  const W = 700, H = 390, pl = 52, pr = 18, pt = 18, pb = 38;
  const pw = W - pl - pr, ph = H - pt - pb;

  let x0 = 0.27, x1 = 0.85, y0 = 0.36, y1 = 0.62, yKey = "dino_s", yTitle = "DINO-S (Style Similarity ↑)";
  if (curMode === "csd_top1") {{
    y0 = 0.25; y1 = 0.96; yKey = "csd_top1"; yTitle = "CSD Top-1 Accuracy ↑";
  }} else if (curMode === "holdout") {{
    x0 = 0.52; x1 = 0.74; y0 = 0.49; y1 = 0.58; yKey = "dino_s"; yTitle = "Holdout-750 DINO-S ↑";
  }}

  const sx = v => pl + ((v - x0) / (x1 - x0)) * pw;
  const sy = v => pt + ph - ((v - y0) / (y1 - y0)) * ph;

  svg.appendChild(el("rect", {{x: pl, y: pt, width: pw, height: ph, fill: "none", stroke: "var(--border, #334155)"}}));
  const xTicks = curMode === "holdout" ? [0.55, 0.60, 0.65, 0.70] : [0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
  for (const t of xTicks) {{
    const px = sx(t);
    svg.appendChild(el("line", {{x1: px, y1: pt, x2: px, y2: pt + ph, class: "grid-line"}}));
    svg.appendChild(el("text", {{x: px, y: pt + ph + 16, "text-anchor": "middle", class: "axis-text"}}, t.toFixed(2)));
  }}
  const yTicks = curMode === "csd_top1" ? [0.30, 0.45, 0.60, 0.75, 0.90] : (curMode === "holdout" ? [0.50, 0.52, 0.54, 0.56] : [0.40, 0.45, 0.50, 0.55, 0.60]);
  for (const t of yTicks) {{
    const py = sy(t);
    svg.appendChild(el("line", {{x1: pl, y1: py, x2: pl + pw, y2: py, class: "grid-line"}}));
    const lbl = curMode === "csd_top1" ? `${{(t*100).toFixed(0)}}%` : t.toFixed(2);
    svg.appendChild(el("text", {{x: pl - 6, y: py + 3.5, "text-anchor": "end", class: "axis-text"}}, lbl));
  }}
  svg.appendChild(el("text", {{x: pl + pw / 2, y: H - 6, "text-anchor": "middle", class: "axis-text"}}, "DINO-C (Content Preservation ↑)"));
  svg.appendChild(el("text", {{x: 14, y: pt + ph / 2, "text-anchor": "middle", class: "axis-text", transform: `rotate(-90 14 ${{pt+ph/2}})`}}, yTitle));

  if (curMode === "holdout") {{
    const colors = {{V32_s085: "#3b82f6", V32_s100: "#3b82f6", D_merged_s085: "#eab308", D_merged_s100: "#eab308", C2_A3_s085: "#22c55e", C2_A3_s100: "#22c55e"}};
    for (const pair of [["V32_s100", "V32_s085", "#3b82f6"], ["D_merged_s100", "D_merged_s085", "#eab308"], ["C2_A3_s100", "C2_A3_s085", "#22c55e"]]) {{
      const p0 = DATA.holdout.find(p => p.label === pair[0]);
      const p1 = DATA.holdout.find(p => p.label === pair[1]);
      if (p0 && p1) {{
        svg.appendChild(el("line", {{x1: sx(p0.dino_c), y1: sy(p0.dino_s), x2: sx(p1.dino_c), y2: sy(p1.dino_s), stroke: pair[2], "stroke-width": 2.6}}));
      }}
    }}
    for (const p of DATA.holdout) {{
      const cx = sx(p.dino_c), cy = sy(p.dino_s), col = colors[p.label] || "#22c55e";
      const c = el("circle", {{cx, cy, r: 6, fill: col, stroke: "#0f172a", "stroke-width": 1.2}});
      c.onmouseenter = () => {{ info.textContent = `[Holdout-750] ${{p.label}} | C=${{p.dino_c.toFixed(4)}} S=${{p.dino_s.toFixed(4)}} Top1=${{(p.csd_top1*100).toFixed(2)}}% LP=${{p.lpips.toFixed(4)}} Same=${{(p.same150_top1*100).toFixed(1)}}%`; }};
      svg.appendChild(c);
      svg.appendChild(el("text", {{x: cx + 8, y: cy + 4, class: "axis-text", style: `fill:${{col}};font-weight:600`}}, `${{p.label}} (${{(p.csd_top1*100).toFixed(1)}}%)`));
    }}
    return;
  }}

  // External scatter
  for (const p of DATA.external_scatter) {{
    const cx = sx(p.dino_c), cy = sy(p[yKey]);
    if (cx < pl || cx > pl + pw || cy < pt || cy > pt + ph) continue;
    const c = el("circle", {{cx, cy, r: 4, fill: "#94a3b8", opacity: 0.85}});
    c.onmouseenter = () => {{ info.textContent = `[External] ${{p.label}} | DINO-C=${{p.dino_c.toFixed(4)}} DINO-S=${{p.dino_s.toFixed(4)}} Top1=${{(p.csd_top1*100).toFixed(1)}}%`; }};
    svg.appendChild(c);
    const below = p.label === "StyleShot" || p.label === "CUT";
    svg.appendChild(el("text", {{x: cx + 5, y: cy + (below ? 12 : -5), class: "ext-label"}}, p.label));
  }}

  // Series curves
  for (const s of DATA.series) {{
    if (!enabled.has(s.id)) continue;
    const pts = s.points.filter(p => p.dino_c >= x0 - 0.01 && p.dino_c <= x1 + 0.01);
    if (!pts.length) continue;
    const d = pts.map((p, i) => `${{i ? "L" : "M"}}${{sx(p.dino_c).toFixed(1)}},${{sy(p[yKey]).toFixed(1)}}`).join(" ");
    const dash = s.style === "dashed" ? "5,4" : (s.style === "dotted" ? "2,3" : "");
    const sw = s.width || 2.1;
    const pathAttrs = {{d, fill: "none", stroke: s.color, "stroke-width": sw}};
    if (dash) pathAttrs["stroke-dasharray"] = dash;
    svg.appendChild(el("path", pathAttrs));
    for (const p of pts) {{
      const cx = sx(p.dino_c), cy = sy(p[yKey]);
      if (cy < pt - 2 || cy > pt + ph + 2) continue;
      const c = el("circle", {{cx, cy, r: sw + 1.8, fill: s.color, stroke: "var(--card, #111827)", "stroke-width": 1}});
      c.onmouseenter = () => {{
        const lp = p.lpips ? ` LP=${{p.lpips.toFixed(4)}}` : "";
        const sm = p.same150_top1 ? ` Same=${{(p.same150_top1*100).toFixed(1)}}%` : "";
        info.textContent = `${{s.name}} [${{p.label}}] | C=${{p.dino_c.toFixed(4)}} S=${{p.dino_s.toFixed(4)}} Top1=${{(p.csd_top1*100).toFixed(2)}}%${{lp}}${{sm}}`;
      }};
      svg.appendChild(c);
    }}
  }}
}}

document.querySelectorAll("[data-mode]").forEach(b => b.onclick = () => {{
  document.querySelectorAll("[data-mode]").forEach(x => x.classList.remove("active"));
  b.classList.add("active");
  curMode = b.dataset.mode;
  render();
}});
document.querySelectorAll("[data-preset]").forEach(b => b.onclick = () => {{
  document.querySelectorAll("[data-preset]").forEach(x => x.classList.remove("active"));
  b.classList.add("active");
  enabled.clear();
  for (const id of PRESETS[b.dataset.preset]) enabled.add(id);
  render();
}});
render();
</script>
</body>
</html>"""


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = collect_pareto_dataset()
    (FIG_DIR / "cleave_pareto_latest.svg").write_text(build_svg(data, metric_y="dino_s"), encoding="utf-8")
    (FIG_DIR / "cleave_pareto_top1_latest.svg").write_text(build_svg(data, metric_y="csd_top1"), encoding="utf-8")
    (FIG_DIR / "cleave_pareto_latest.html").write_text(build_interactive_html(data), encoding="utf-8")
    print(f"Wrote Pareto SVG+HTML to {FIG_DIR} ({len(data['series'])} curves, {len(data['holdout'])} holdout pts)")


if __name__ == "__main__":
    main()
