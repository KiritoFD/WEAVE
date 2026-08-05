"""Recompute cross metrics for every ablation board in submission_pack and update paper table."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ABL = ROOT / "results/cross_protocol/submission_pack/boards/ablation"
OUT = ROOT / "results/cross_protocol/aux_tables"


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def normalize(r):
    rr = dict(r)
    if "dino_s" in rr and "dino_style" not in rr:
        rr["dino_style"] = rr["dino_s"]
    if "dino_c" in rr and "dino_content" not in rr:
        rr["dino_content"] = rr["dino_c"]
    return rr


def aggregate_metrics(metrics_path: Path, dino_summary: Path | None = None) -> dict:
    rows = [normalize(r) for r in csv.DictReader(metrics_path.open(encoding="utf-8-sig"))]
    if not rows:
        return {}
    cross = [r for r in rows if r.get("src_style") != r.get("tgt_style")]
    keys = [k for k in ["clip_style", "content_lpips", "dino_style", "dino_content"] if k in rows[0]]

    def m(rs, k):
        xs = []
        for r in rs:
            try:
                xs.append(float(r[k]))
            except Exception:
                pass
        return mean(xs)

    out = {
        "n_all": len(rows),
        "n_cross": len(cross),
        "source_metrics": str(metrics_path),
    }
    for k in keys:
        out[f"all_{k}"] = m(rows, k)
        out[f"cross_{k}"] = m(cross, k)

    if dino_summary and dino_summary.is_file():
        d = json.loads(dino_summary.read_text(encoding="utf-8"))
        # prefer dino_summary all/off when present (canonical DINO)
        if d.get("all_dino_s") is not None:
            out["all_dino_style"] = d["all_dino_s"]
        if d.get("off_dino_s") is not None:
            out["cross_dino_style"] = d["off_dino_s"]
        if d.get("all_dino_c") is not None:
            out["all_dino_content"] = d["all_dino_c"]
        if d.get("off_dino_c") is not None:
            out["cross_dino_content"] = d["off_dino_c"]
        if d.get("all_clip_s") is not None and out.get("all_clip_style") is None:
            out["all_clip_style"] = d["all_clip_s"]
        if d.get("all_lpips") is not None and out.get("all_content_lpips") is None:
            out["all_content_lpips"] = d["all_lpips"]
        out["source_dino_summary"] = str(dino_summary)
    return out


# Map paper ablation rows to pack board names (prefer paper-matching sources)
PAPER_MAP = {
    "WEAVE": ["weave_main", "canonical"],
    "w/o AdaIN": ["d01_adain_0", "a01_wo_endpoint_adain", "wo_endpoint_adain"],
    "final-only AdaIN": ["final_only_adain"],
    "w/o HF conditioning": [],  # may still be missing
    "lambda_LL=1.0": ["wll_10", "b02_wll_20"],  # closest available; label carefully
    "direct endpoint": [],
    "learned HH head": ["b08_whh_4"],  # HH weight variant, not exact paper row
    "zero-velocity (pure analytic)": ["zero_velocity", "a03_wo_flow"],
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for d in sorted(ABL.iterdir()):
        if not d.is_dir():
            continue
        metrics = d / "metrics.csv"
        if not metrics.is_file():
            print("skip no metrics", d.name)
            continue
        dino_sum = d / "dino_summary_cross.json"
        if not dino_sum.is_file():
            dino_sum = d / "dino_summary.json"
        a = aggregate_metrics(metrics, dino_sum)
        n_img = sum(
            1
            for p in (d / "images").rglob("*")
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ) if (d / "images").exists() else 0
        a["n_images"] = n_img
        results[d.name] = a
        print(
            f"{d.name:28s} n={a.get('n_cross')} "
            f"DS_c={a.get('cross_dino_style')} CS_c={a.get('cross_clip_style')} "
            f"LP_c={a.get('cross_content_lpips')} DC_c={a.get('cross_dino_content')} imgs={n_img}"
        )

    # paper-facing selection
    paper_rows = {}
    # WEAVE: use weave_main cross
    if "weave_main" in results:
        paper_rows["WEAVE"] = results["weave_main"]
    # final-only
    if "final_only_adain" in results:
        paper_rows["final-only AdaIN"] = results["final_only_adain"]
    # zero velocity
    if "zero_velocity" in results:
        paper_rows["zero-velocity (pure analytic)"] = results["zero_velocity"]
    # w/o AdaIN: d01_adain_0 (inference adain=0) preferred
    if "d01_adain_0" in results:
        paper_rows["w/o AdaIN"] = results["d01_adain_0"]
        paper_rows["w/o AdaIN"]["note"] = "from ablation_v2/d01_adain_0 (endpoint AdaIN scale 0)"
    elif "a01_wo_endpoint_adain" in results:
        paper_rows["w/o AdaIN"] = results["a01_wo_endpoint_adain"]
    # lambda variants
    if "wll_01" in results:
        paper_rows["lambda_LL~0.1 (wll_01)"] = results["wll_01"]
    if "b01_wll_0" in results:
        paper_rows["lambda_LL=0 (b01)"] = results["b01_wll_0"]
    if "b02_wll_20" in results:
        paper_rows["lambda_LL high (b02_wll_20)"] = results["b02_wll_20"]
    # HH
    if "b07_whh_0" in results:
        paper_rows["HH weight 0"] = results["b07_whh_0"]
    if "b08_whh_4" in results:
        paper_rows["HH weight 4 / learned HH proxy"] = results["b08_whh_4"]
    # destructive
    for k in ["wo_endpoint_adain", "wo_flow", "wo_wavelet", "wo_spectral_ode", "wo_asg"]:
        if k in results:
            paper_rows[k] = results[k]

    payload = {
        "all_ablation_boards": results,
        "paper_facing": paper_rows,
        "protocol": "cross = mean over src_style != tgt_style",
    }
    (OUT / "ablation_pack_cross.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # markdown
    lines = [
        "# Ablation boards (submission pack) — all vs cross",
        "",
        "Each board has ~750 generated images under `submission_pack/boards/ablation/<name>/images`.",
        "",
        "| Board | n_cross | DINO-S all | DINO-S cross | CLIP-S all | CLIP-S cross | LPIPS all | LPIPS cross | DINO-C all | DINO-C cross | n_images |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def f(v):
        return f"{v:.4f}" if isinstance(v, float) else "—"

    for name, a in sorted(results.items()):
        lines.append(
            f"| {name} | {a.get('n_cross','—')} | {f(a.get('all_dino_style'))} | {f(a.get('cross_dino_style'))} | "
            f"{f(a.get('all_clip_style'))} | {f(a.get('cross_clip_style'))} | {f(a.get('all_content_lpips'))} | "
            f"{f(a.get('cross_content_lpips'))} | {f(a.get('all_dino_content'))} | {f(a.get('cross_dino_content'))} | "
            f"{a.get('n_images',0)} |"
        )
    (OUT / "ABLATION_PACK_CROSS.md").write_text("\n".join(lines), encoding="utf-8")
    print("WROTE", OUT / "ablation_pack_cross.json")
    print("WROTE", OUT / "ABLATION_PACK_CROSS.md")

    # print paper-facing for tex update
    print("\n=== PAPER FACING CROSS ===")
    for name, a in paper_rows.items():
        print(
            f"{name}: DS={f(a.get('cross_dino_style'))} CS={f(a.get('cross_clip_style'))} "
            f"LP={f(a.get('cross_content_lpips'))} DC={f(a.get('cross_dino_content'))}"
        )


if __name__ == "__main__":
    main()
