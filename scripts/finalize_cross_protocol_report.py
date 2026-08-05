"""Merge existing all/cross metrics CSVs into a clean report.

Prefers:
- GPU re-eval CSVs under results/cross_protocol/
- prior per-sample CSVs under eval_outputs/ when present
- marks content metrics unreliable when DINO-C > 0.95 and LPIPS < 0.05
  (source-content mismatch failure mode) except Identity.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "cross_protocol"


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def load_csv(path: Path):
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    # normalize keys
    out = []
    for r in rows:
        rr = dict(r)
        if "dino_s" in rr and "dino_style" not in rr:
            rr["dino_style"] = rr["dino_s"]
        if "dino_c" in rr and "dino_content" not in rr:
            rr["dino_content"] = rr["dino_c"]
        out.append(rr)
    return out


def agg(rows, keys):
    cross = [r for r in rows if r.get("src_style") != r.get("tgt_style")]
    res = {"n_all": len(rows), "n_cross": len(cross)}
    for k in keys:
        if k not in rows[0]:
            continue
        def vals(rs):
            xs = []
            for r in rs:
                try:
                    xs.append(float(r[k]))
                except Exception:
                    pass
            return xs
        va, vc = vals(rows), vals(cross)
        res[f"all_{k}"] = mean(va)
        res[f"cross_{k}"] = mean(vc)
    # reliability flag for content metrics
    dc = res.get("all_dino_content")
    lp = res.get("all_content_lpips")
    name = rows[0].get("src_style", "")
    if dc is not None and lp is not None and dc > 0.95 and lp < 0.05:
        res["content_unreliable"] = True
    else:
        res["content_unreliable"] = False
    return res


def collect_board(board_dir: Path, display_map: dict, extra_csvs: dict):
    summary = {}
    keys = ["clip_style", "content_lpips", "dino_style", "dino_content"]
    # load all method metrics in board_dir
    for p in sorted(board_dir.glob("*_metrics.csv")):
        folder = p.name.replace("_metrics.csv", "")
        rows = load_csv(p)
        if not rows:
            continue
        a = agg(rows, keys)
        a["display"] = display_map.get(folder, folder)
        a["folder"] = folder
        a["source"] = str(p)
        summary[folder] = a
    # extras override if better (e.g. full eval_outputs)
    for folder, path in extra_csvs.items():
        path = Path(path)
        if not path.is_file():
            continue
        rows = load_csv(path)
        a = agg(rows, keys)
        a["display"] = display_map.get(folder, folder)
        a["folder"] = folder
        a["source"] = str(path)
        # only override if we don't already have gpu metrics, or extra has more metrics
        if folder not in summary or summary[folder].get("all_dino_style") is None:
            summary[folder] = a
        else:
            # keep existing but if existing content unreliable and extra not, prefer extra
            if summary[folder].get("content_unreliable") and not a.get("content_unreliable"):
                summary[folder] = a
    return summary


def md_table(title, summary):
    lines = [f"## {title}", ""]
    lines.append(
        "| Method | n_all | n_cross | CLIP-S all | CLIP-S cross | LPIPS all | LPIPS cross | DINO-S all | DINO-S cross | DINO-C all | DINO-C cross | note |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")

    def f(v):
        return f"{v:.4f}" if isinstance(v, float) else "—"

    # order similar to paper
    order = [
        "identity", "tgt", "sdturbo", "stylealigned", "zstar", "styleshot", "styleid",
        "cut", "samst", "samam", "stytr2", "aespa", "seedream", "wct", "weave", "weave_oriented_e4",
    ]
    items = []
    for k in order:
        if k in summary:
            items.append((k, summary[k]))
    for k, v in summary.items():
        if k not in order:
            items.append((k, v))

    for folder, a in items:
        note = []
        if a.get("content_unreliable") and folder != "identity":
            note.append("content metrics suspect (near-copy DINO-C/LPIPS; source pairing likely wrong)")
        if folder == "weave_oriented_e4":
            note.append("paper D5 main checkpoint candidate")
        if folder == "weave" and title.startswith("D5"):
            note.append("eval_outputs / alternate ckpt")
        lines.append(
            f"| {a.get('display', folder)} | {a.get('n_all','—')} | {a.get('n_cross','—')} | "
            f"{f(a.get('all_clip_style'))} | {f(a.get('cross_clip_style'))} | "
            f"{f(a.get('all_content_lpips'))} | {f(a.get('cross_content_lpips'))} | "
            f"{f(a.get('all_dino_style'))} | {f(a.get('cross_dino_style'))} | "
            f"{f(a.get('all_dino_content'))} | {f(a.get('cross_dino_content'))} | "
            f"{'; '.join(note) if note else '—'} |"
        )
    lines.append("")
    return lines


def main():
    d5_map = {
        "identity": "Identity",
        "tgt": "Target Style",
        "sdturbo": "SD-Turbo",
        "stylealigned": "StyleAligned",
        "zstar": "Z-STAR",
        "styleshot": "StyleShot",
        "styleid": "StyleID",
        "cut": "CUT",
        "samst": "SaMST",
        "samam": "SaMam",
        "stytr2": "StyTR-2",
        "aespa": "AesPA-Net",
        "seedream": "Seedream 4.5",
        "wct": "Latent-WCT / WCT",
        "weave": "WEAVE",
        "weave_oriented_e4": "WEAVE (oriented e4)",
    }
    p2a_map = dict(d5_map)

    d5_extra = {
        "cut": ROOT / "eval_outputs/d5_750_baselines/cut/metrics.csv",
        "samam": ROOT / "eval_outputs/d5_750_baselines/samam/metrics.csv",
        "aespa": ROOT / "eval_outputs/d5_750_baselines/aespa/metrics.csv",
        "stytr2": ROOT / "eval_outputs/d5_750_baselines/stytr2/metrics.csv",
        "seedream": ROOT / "eval_outputs/d5_750_baselines/seedream/metrics.csv",
        "weave": ROOT / "eval_outputs/d5_750_baselines/weave/metrics.csv",
    }

    d5 = collect_board(OUT / "D5-512", d5_map, d5_extra)
    p2a = collect_board(OUT / "P2A-256", p2a_map, {})

    # Prefer oriented e4 as primary WEAVE on D5 for paper-aligned DINO-S
    if "weave_oriented_e4" in d5:
        d5["weave_main"] = dict(d5["weave_oriented_e4"])
        d5["weave_main"]["display"] = "WEAVE (main/oriented e4)"
        d5["weave_main"]["folder"] = "weave_main"

    lines = []
    lines.append("# All vs Cross Protocol Metrics (D5-512 & P2A-256)")
    lines.append("")
    lines.append("Generated by local GPU evaluation + re-aggregation (`scripts/recompute_cross_protocol.py`, `scripts/finalize_cross_protocol_report.py`).")
    lines.append("")
    lines.append("## Protocol")
    lines.append("")
    lines.append("| Protocol | Definition |")
    lines.append("|---|---|")
    lines.append("| **all** | Mean over all ordered content–style pairs, including identity diagonal `src_style == tgt_style`. |")
    lines.append("| **cross** | Mean over off-diagonal pairs only (`src_style != tgt_style`). |")
    lines.append("")
    lines.append("Metrics: **DINO-S** = max CLS cosine to target-style ref pool; **DINO-C** = CLS cosine to source; **CLIP-S** / **LPIPS** as in paper.")
    lines.append("")
    lines.append("### Why cross?")
    lines.append("")
    lines.append("Diagonal pairs are self-style reconstructions. Including them inflates style scores (DINO-S/CLIP-S) for methods that preserve identity and makes method ranking less about *transfer*. Cross is the transfer-centric board.")
    lines.append("")
    lines.extend(md_table("D5-512", d5))
    lines.extend(md_table("P2A-256", p2a))

    lines.append("## Headline comparison (cross only)")
    lines.append("")
    lines.append("### D5-512 (cross, n≈600)")
    lines.append("")
    lines.append("| Method | DINO-S↑ | CLIP-S↑ | LPIPS↓ | DINO-C↑ |")
    lines.append("|---|---:|---:|---:|---:|")
    for key in ["identity", "weave_oriented_e4", "samam", "cut", "stytr2", "aespa", "seedream", "styleid", "styleshot"]:
        if key not in d5:
            continue
        a = d5[key]
        def f(k):
            v = a.get(k)
            return f"{v:.4f}" if isinstance(v, float) else "—"
        note = " †content suspect" if a.get("content_unreliable") and key != "identity" else ""
        lines.append(
            f"| {a['display']}{note} | {f('cross_dino_style')} | {f('cross_clip_style')} | {f('cross_content_lpips')} | {f('cross_dino_content')} |"
        )

    lines.append("")
    lines.append("### P2A-256 (cross, n≈600)")
    lines.append("")
    lines.append("| Method | DINO-S↑ | CLIP-S↑ | LPIPS↓ | DINO-C↑ |")
    lines.append("|---|---:|---:|---:|---:|")
    for key in ["identity", "weave", "samam", "samst", "cut", "stytr2", "aespa", "seedream", "stylealigned", "sdturbo", "zstar", "styleid"]:
        if key not in p2a:
            continue
        a = p2a[key]
        def f(k):
            v = a.get(k)
            return f"{v:.4f}" if isinstance(v, float) else "—"
        lines.append(
            f"| {a['display']} | {f('cross_dino_style')} | {f('cross_clip_style')} | {f('cross_content_lpips')} | {f('cross_dino_content')} |"
        )

    lines.append("")
    lines.append("## Notes / caveats")
    lines.append("")
    lines.append("1. **Paper Table 1 used all** averages. Cross systematically lowers style scores (diagonal self-style removed).")
    lines.append("2. **WEAVE main on D5**: use `weave_oriented_e4` (matches paper DINO-S≈0.492 all). The `eval_outputs/.../weave` row is an alternate checkpoint.")
    lines.append("3. **D5 StyleAligned / Z-STAR** were re-evaluated with sources from `archives/datasets_legacy_20260721/wikiart20_test` (content stems match that split, not `data/test`).")
    lines.append("4. **P2A StyleShot**: only 150/750 gens matched resolvable sources under strict matching; treat that row with care.")
    lines.append("5. Identity **cross DINO-S** is the mean style affinity of content images to *other* styles (not a transfer method).")
    lines.append("6. SD-Turbo D5 near-identity LPIPS is consistent with the paper main table (strength≈identity regime).")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append("- Per-sample CSVs: `results/cross_protocol/D5-512/*_metrics.csv`, `results/cross_protocol/P2A-256/*_metrics.csv`")
    lines.append("- Machine-readable summary: `results/cross_protocol/all_summary.json`")
    lines.append("- Script: `scripts/recompute_cross_protocol.py`")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "CROSS_PROTOCOL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "all_summary.json").write_text(
        json.dumps({"D5-512": d5, "P2A-256": p2a}, indent=2), encoding="utf-8"
    )
    print("WROTE", OUT / "CROSS_PROTOCOL_REPORT.md")
    # print quick headline
    print("\n=== D5 cross headline ===")
    for k in ["identity", "weave_oriented_e4", "samam", "cut", "styleid"]:
        if k in d5:
            a = d5[k]
            print(f"{a['display']}: DINO-S {a.get('cross_dino_style'):.4f} CLIP-S {a.get('cross_clip_style'):.4f} LPIPS {a.get('cross_content_lpips')} DINO-C {a.get('cross_dino_content')}")


if __name__ == "__main__":
    main()
