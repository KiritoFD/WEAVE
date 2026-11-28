"""Summarize verified counts and paired source-cluster confidence intervals."""
import json
import re
from pathlib import Path
import numpy as np
from exp.v36_art_foundation import MEAS_DIR, ROOT
from exp.v37_editing import OUT, SEED, atomic_json, read_manifest


def cluster_curve(paths, stems):
    curves = []
    for path in paths:
        if not path.exists() and "/runs/" in path.as_posix():
            path = ROOT / "runs" / path.as_posix().split("/runs/", 1)[1]
        d = json.loads(path.read_text(encoding="utf-8"))
        assert (d["n_total"], d["n_cross"], d["n_same"]) == (750, 600, 150)
        by_source = {}
        for r in d["rows"]:
            if r["is_cross"]:
                by_source.setdefault(r["src_stem"], []).append([r["dino_c"], r["dino_s"], float(r["target_top1"])])
        assert set(by_source) == set(stems)
        curves.append(np.array([np.mean(by_source[s], axis=0) for s in stems]))
    return np.stack(curves)


def interpolate(curve, c):
    order = np.argsort(curve[:, 0]); curve = curve[order]
    if not curve[0, 0] <= c <= curve[-1, 0]:
        return None
    return np.array([np.interp(c, curve[:, 0], curve[:, j]) for j in (1, 2)])


def confidence_intervals(rows, draws=2000):
    baseline_paths = sorted(MEAS_DIR.glob("v36_main_p2_I2_t*/eval.json"))
    # Wrong-control dirs are excluded by the exact numeric suffix.
    baseline_paths = [p for p in baseline_paths if re.fullmatch(r"v36_main_p2_I2_t\d+", p.parent.name)]
    first = json.loads(baseline_paths[0].read_text(encoding="utf-8"))
    stems = sorted({r["src_stem"] for r in first["rows"] if r["is_cross"]})
    base = cluster_curve(baseline_paths, stems)
    out = []
    rng = np.random.default_rng(SEED + 500)
    boot = rng.integers(0, len(stems), (draws, len(stems)))
    for route in sorted({r["route"] for r in rows}):
        paths = [Path(r["measurement_dir"]) / "eval.json" for r in rows if r["route"] == route and r["tau"] is not None]
        arr = cluster_curve(paths, stems)
        for c in (.60, .65, .70, .75):
            aa, bb = interpolate(arr.mean(1), c), interpolate(base.mean(1), c)
            if aa is None or bb is None:
                out.append({"route": route, "content": c, "status": "outside_observed_bracket"}); continue
            delta = aa - bb
            bs = []
            for idx in boot:
                a, b = interpolate(arr[:, idx].mean(1), c), interpolate(base[:, idx].mean(1), c)
                if a is not None and b is not None: bs.append(a - b)
            ci = np.quantile(bs, [.025, .975], axis=0)
            out.append({"route": route, "content": c, "interpolated_dino_s": aa[0],
                        "interpolated_top1": aa[1], "delta_dino_s": delta[0], "delta_top1_pp": delta[1] * 100,
                        "dino_s_ci95": ci[:, 0].tolist(), "top1_pp_ci95": (ci[:, 1] * 100).tolist(),
                        "valid_bootstrap": len(bs), "draws": draws, "source_clusters": len(stems),
                        "status": "paired_cluster_bootstrap_with_reinterpolation"})
    atomic_json(OUT / "matched_content_ci.json", out)
    return out


def holdout_intervals(rows, draws=2000):
    result = []
    for split in ("artist_ood", "movement_ood"):
        subset = [r for r in rows if r["split"] == split]
        def curve(route):
            return np.stack([np.stack([r["per_source"][k] for k in ("dino_c", "dino_s", "csd_2way_follow")], axis=-1)
                             for r in subset if r["route"] == route])
        base = curve("P_A3")
        n = base.shape[1]
        boot = np.random.default_rng(SEED + n).integers(0, n, (draws, n))
        for route in sorted({r["route"] for r in subset} - {"P_A3"}):
            arr = curve(route)
            for c in (.60, .65, .70, .75):
                aa, bb = interpolate(arr.mean(1), c), interpolate(base.mean(1), c)
                if aa is None or bb is None: continue
                vals = []
                for idx in boot:
                    a, b = interpolate(arr[:, idx].mean(1), c), interpolate(base[:, idx].mean(1), c)
                    if a is not None and b is not None: vals.append(a - b)
                ci = np.quantile(vals, [.025, .975], axis=0)
                result.append({"split": split, "route": route, "n": n, "content": c,
                               "delta_dino_s": float(aa[0] - bb[0]), "dino_s_ci95": ci[:, 0].tolist(),
                               "delta_2way_pp": float((aa[1] - bb[1]) * 100),
                               "two_way_pp_ci95": (ci[:, 1] * 100).tolist(),
                               "valid_bootstrap": len(vals), "draws": draws})
    atomic_json(OUT / "holdout_matched_ci.json", result)
    return result


def main():
    rows = json.loads((OUT / "official_core750.json").read_text(encoding="utf-8"))
    holds = json.loads((OUT / "holdouts/summary.json").read_text(encoding="utf-8"))
    ci = confidence_intervals(rows)
    hold_ci = holdout_intervals(holds)
    manifest = read_manifest()
    stages = [json.loads(p.read_text()) for p in sorted((OUT / "stages").glob("*.json"))]
    report = {"manifest": manifest, "core750": rows, "holdouts": holds, "matched_content": ci, "holdout_matched_content": hold_ci,
              "stages": stages, "ecore_states": {r: json.loads((OUT / f"{r}_state.json").read_text()) for r in ("E0", "E1")},
              "p4_decision": json.loads((OUT / "p4_decision.json").read_text()),
              "caveats": ["Interpolated workpoints are estimates between observed configurations",
                          "E0 and E1 retain target-Q32 shortcut in Task A; E1 inverse Task B uses proxy-derived Q32",
                          "Same-artist does not establish identical painting technique",
                          "Oracle closed-set E-Core is diagnostic only",
                          "Reverse proxy supervision may also teach content repair; inspect fixed examples",
                          "No inference speed claim without a separate measured timing"]}
    atomic_json(OUT / "final_report.json", report)
    lines = ["# V37 开放域编辑实验结果", "", f"固定窗口：{manifest['started']} 至 {manifest['deadline']}（UTC）。",
             "", "正式基线仍为 V36 P-Core + Reference-A3；是否替换必须结合下表、置信区间与固定笔触样例判断。",
             "", "## 完整 test750 / Cross600", "", "| 模型 | λ | τ | DINO-C | DINO-S | Top-1 | LPIPS | Same150 Top-1 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        c = r["cross600"]
        lines.append(f"| {r['route']} | {r['lambda']} | {r['tau']} | {c['dino_c']:.4f} | {c['dino_s']:.4f} | {c['csd_top1']*100:.2f}% | {c['lpips']:.4f} | {r['same150']['csd_top1']*100:.2f}% |")
    lines += ["", "## 等内容比较（线性插值，不是额外实测点）", "", "| 模型 | C | ΔS | ΔS 95% CI | ΔTop-1 pp |", "|---|---:|---:|---|---:|"]
    for r in ci:
        if "delta_dino_s" not in r: continue
        lines.append(f"| {r['route']} | {r['content']:.2f} | {r['delta_dino_s']:+.4f} | {r['dino_s_ci95']} | {r['delta_top1_pp']:+.2f} |")
    lines += ["", "## 独立 Holdout（实例参考指标，不是核心五类 Top-1）", "", "| 模型 | 数据 | N | τ | DINO-C | DINO-S | CSD 2-way |", "|---|---|---:|---:|---:|---:|---:|"]
    for r in holds:
        lines.append(f"| {r['route']} | {r['split']} | {r['n']} | {r['tau']} | {r['dino_c']:.4f} | {r['dino_s']:.4f} | {r['csd_2way_follow']*100:.2f}% |")
    lines += ["", "## 限制", ""] + ["- " + s for s in report["caveats"]]
    dest = ROOT / "docs/one-shot/V37_RESULTS.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
