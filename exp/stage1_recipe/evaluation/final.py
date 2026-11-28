"""CPU-only complete-split report, adoption gates and auditable Pareto export."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from exp.stage1_recipe.config import ROOT, read, resolve, write
from exp.stage1_recipe.evaluation.selection import acceptance, matched_ci, same_ci, curve, interpolate


def aggregate(path, split):
    import numpy as np
    d = read(path)
    if split == "core750":
        assert (d["n_total"], d["n_cross"], d["n_same"]) == (750, 600, 150)
        assert len({r["task_id"] for r in d["rows"]}) == 750
        rows = [r for r in d["rows"] if r["is_cross"]]
        top = "target_top1"
    else:
        rows = d["rows"]
        assert len(rows) == {"artist_ood": 1000, "movement_ood": 500}[split]
        top = "csd_2way_follow"
    return {"dc": float(np.mean([r["dino_c"] for r in rows])),
            "ds": float(np.mean([r["dino_s"] for r in rows])),
            "t1": 100 * float(np.mean([r[top] for r in rows])), "n": len(rows)}


def finish(config_path: Path, run: Path):
    config, index = read(config_path), read(run / "formal_index.json")
    ev = config["evaluation"]
    baseline = {x["split"]: [Path(p) for p in x["paths"]] for x in index["baseline"]}
    assert set(baseline) == {"core750", "artist_ood", "movement_ood"}
    report = {"selection_frozen_before_test": index["selection_frozen_before_test"], "candidates": [],
              "baseline_interpolated": [], "missing_review_is_not_a_pass": True}
    points = []
    base_curve, _ = curve(baseline["core750"], "core750")
    for c in ev["content_anchors"]:
        point = interpolate(base_curve.mean(1), c)
        report["baseline_interpolated"].append({"content": c, "s": float(point[0]) if point is not None else None,
                                                "top1_pp": float(100*point[1]) if point is not None else None,
                                                "source": "current_exact_protocol_baseline"})
    for split, paths in baseline.items():
        for path in paths:
            points.append({"route": "V36 P+A3", "split": split, **aggregate(path, split), "source": str(path)})
    for case in index["candidates"]:
        paths = {x["split"]: [Path(p) for p in x["paths"]] for x in case["splits"]}
        cis = {split: matched_ci(paths[split], baseline[split], scope=split,
                                anchors=ev["content_anchors"], draws=ev["bootstrap"],
                                minimum_valid=ev["minimum_valid_bootstrap"]) for split in baseline}
        same = [same_ci(a, b, ev["bootstrap"]) for a, b in zip(paths["core750"], baseline["core750"])]
        review_path = run / "reviews" / (case["name"] + ".json")
        visual = read(review_path) if review_path.exists() else {}
        probe_path = run / "probes" / case["name"] / "probe.json"
        baseline_probe_path = run / "probes/baseline_P_A3/probe.json"
        evidence = visual.get("evidence", [])
        visual["evidence_validated"] = bool(probe_path.exists() and baseline_probe_path.exists() and visual.get("reviewer") and
            visual.get("notes") and len(evidence) >= 2 and all(resolve(x).is_file() for x in evidence) and
            visual.get("probe_signature") == read(probe_path)["provenance"]["signature"] and
            visual.get("baseline_probe_signature") == read(baseline_probe_path)["provenance"]["signature"])
        result = acceptance(cis["core750"], same, cis, visual, ev)
        report["candidates"].append({"name": case["name"], "matched_ci": cis, "same150_ci": same,
                                      "visual_evidence": visual, "decision": result})
        for split, pp in paths.items():
            for p in pp:
                points.append({"route": case["name"], "split": split, **aggregate(p, split), "source": str(p)})
        for p in case.get("e_only", []):
            points.append({"route": case["name"] + " E-only", "split": "core750",
                           **aggregate(Path(p), "core750"), "source": p})
    report["no_supported_candidate"] = not bool(report["candidates"])
    report["baseline_retained"] = not any(x["decision"]["adopt"] for x in report["candidates"])
    write(run / "final_report.json", report)
    write(run / "pareto_points.json", {"points": points, "source_index": str(run / "formal_index.json")})
    lines = ["# CLEAVE 第一阶段实际结果", "", "正式评测与采用决策分开；缺少视觉证据不会自动升级。", "",
             "| 模型 | 数据集 | N | DINO-C | DINO-S | Top-1 / OOD 2-way |", "|---|---|---:|---:|---:|---:|"]
    for p in points:
        lines.append(f"| {p['route']} | {p['split']} | {p['n']} | {p['dc']:.4f} | {p['ds']:.4f} | {p['t1']:.2f}% |")
    lines += ["", "## 采用判定", "", "```json", json.dumps(report, ensure_ascii=False, indent=2), "```", "",
              "Seen-Dev实例参考DINO-S、正式Core750目标库DINO-S与OOD实例指标分开；不跨数据集求前沿。",
              "本报告不自动替换生产权重，不补造Qwen或旧F1/F6 LPIPS；样本/灰度/同色盘证据见reviews及对应输出。"]
    (run / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    render(run / "pareto_points.json")


def render(source: Path):
    """Only verified complete split results are eligible for plotting."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.eval.pareto import frontier_indices
    data = read(source)
    points = data["points"]
    if not points:
        raise ValueError("No measured points; do not manufacture a Stage1 curve")
    for p in points:
        if p["n"] != {"core750": 600, "artist_ood": 1000, "movement_ood": 500}[p["split"]]:
            raise ValueError("Incomplete formal point")
    fig, axs = plt.subplots(3, 2, figsize=(11, 12), constrained_layout=True)
    frozen = read(ROOT / "docs/performance/protocol_comparison/comparison.json")
    external = frozen["baselines"]
    for row, split in enumerate(("core750", "artist_ood", "movement_ood")):
        for col, metric in enumerate(("ds", "t1")):
            ax = axs[row, col]
            if split == "core750":
                eligible = [x for x in external if x["literature"] and x["scopes"]["cross600"]["n"] == 600]
                xy = [(x["scopes"]["cross600"]["dc"], x["scopes"]["cross600"][metric]) for x in eligible]
                if xy:
                    ax.scatter(*zip(*xy), s=15, color="gray", alpha=.6, label="External literature, complete Cross600")
                    frontier = [xy[i] for i in frontier_indices(xy)]
                    frontier.sort()
                    ax.plot(*zip(*frontier), "--", color="gray", alpha=.6)
                for branch in ("E", "P"):
                    closed = sorted([x["scopes"]["cross600"] for x in frozen["ours"] if x["branch"] == branch],
                                    key=lambda x: x["dc"])
                    ax.plot([p["dc"] for p in closed], [p[metric] for p in closed], ":", color="teal", alpha=.6,
                            label="Frozen closed-set " + branch + " (different target setting)")
            subset = [p for p in points if p["split"] == split]
            for route in sorted({p["route"] for p in subset}):
                pp = sorted([p for p in subset if p["route"] == route], key=lambda p: p["dc"])
                ax.plot([p["dc"] for p in pp], [p[metric] for p in pp], "o-", label=route, ms=4)
                idx = frontier_indices([(p["dc"], p[metric]) for p in pp])
                nondom = [pp[i] for i in idx]
                ax.scatter([p["dc"] for p in nondom], [p[metric] for p in nondom], s=45, facecolors="none")
            ax.set(xlabel="DINO-C ↑", ylabel="DINO-S ↑" if metric == "ds" else (
                "CSD Top-1 (%) ↑" if split == "core750" else "CSD instance 2-way (%) ↑"), title=split)
            ax.grid(alpha=.18)
            ax.legend(fontsize=7)
    out = source.parent
    fig.savefig(out / "stage1_pareto.svg")
    fig.savefig(out / "stage1_pareto.png", dpi=160)
    plt.close(fig)
    svg = (out / "stage1_pareto.svg").read_text(encoding="utf-8")
    payload = json.dumps(points, ensure_ascii=False).replace("<", "\\u003c")
    html = '''<!doctype html><meta charset="utf-8"><style>
body{margin:0;font-family:var(--font-sans,sans-serif);color:var(--foreground)}
svg{max-width:100%;height:auto}select{margin:10px}pre{white-space:pre-wrap;font-size:12px}
</style><label>查看实测点 <select id="pick"></select></label><pre id="detail"></pre>''' + svg + f'''
<script>const points={payload};const pick=document.getElementById('pick');
points.forEach((p,i)=>{{const o=document.createElement('option');o.value=i;o.textContent=p.route+' · '+p.split+' · C='+p.dc.toFixed(4);pick.append(o)}});
function show(){{document.getElementById('detail').textContent=JSON.stringify(points[pick.value||0],null,2)}}pick.onchange=show;show();</script>'''
    (out / "stage1_pareto.html").write_text(html, encoding="utf-8")
    figures = ROOT / "docs/performance/figures"
    (figures / "cleave_pareto_latest.svg").write_text(svg, encoding="utf-8")
    (figures / "cleave_pareto_latest.html").write_text(html, encoding="utf-8")
    # Preserve exact data behind the latest figure instead of inferring curves from an image.
    write(ROOT / "docs/performance/data/stage1_recipe_points.json", data)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path)
    p.add_argument("--run-dir", type=Path)
    p.add_argument("--plot", type=Path)
    a = p.parse_args()
    if a.plot:
        render(a.plot)
    elif a.config and a.run_dir:
        finish(a.config, a.run_dir)
    else:
        p.error("Use --plot or --config with --run-dir")


if __name__ == "__main__":
    main()
