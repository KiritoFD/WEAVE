"""Recompute ablation / Other5 / plugin tables under cross protocol and pack for submission."""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "cross_protocol" / "aux_tables"
PACK = ROOT / "results" / "cross_protocol" / "submission_pack"


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def load_rows(path: Path):
    return list(csv.DictReader(path.open(encoding="utf-8-sig")))


def normalize_row(r: dict) -> dict:
    rr = dict(r)
    if "dino_s" in rr and "dino_style" not in rr:
        rr["dino_style"] = rr["dino_s"]
    if "dino_c" in rr and "dino_content" not in rr:
        rr["dino_content"] = rr["dino_c"]
    return rr


def aggregate(rows: list[dict], keys: list[str]) -> dict:
    rows = [normalize_row(r) for r in rows]
    all_r = rows
    cross = [r for r in all_r if r.get("src_style") != r.get("tgt_style")]

    def m(rs, k):
        xs = []
        for r in rs:
            if k not in r:
                continue
            try:
                xs.append(float(r[k]))
            except Exception:
                pass
        return mean(xs)

    out = {"n_all": len(all_r), "n_cross": len(cross)}
    for k in keys:
        if all_r and k in all_r[0]:
            out[f"all_{k}"] = m(all_r, k)
            out[f"cross_{k}"] = m(cross, k)
    return out


def merge_dino_into_metrics(metrics_path: Path, dino_path: Path) -> list[dict]:
    mrows = [normalize_row(r) for r in load_rows(metrics_path)]
    if not dino_path.is_file():
        return mrows
    drows = [normalize_row(r) for r in load_rows(dino_path)]
    # index by (src,tgt,gen) or (src,tgt,src_image)
    dmap = {}
    for r in drows:
        key = (r.get("src_style"), r.get("tgt_style"), r.get("gen_image") or r.get("src_image"))
        dmap[key] = r
        key2 = (r.get("src_style"), r.get("tgt_style"), r.get("src_image"))
        dmap[key2] = r
    out = []
    for r in mrows:
        key = (r.get("src_style"), r.get("tgt_style"), r.get("gen_image") or r.get("src_image"))
        dr = dmap.get(key) or dmap.get((r.get("src_style"), r.get("tgt_style"), r.get("src_image")))
        rr = dict(r)
        if dr:
            if "dino_style" in dr:
                rr["dino_style"] = dr["dino_style"]
            if "dino_content" in dr:
                rr["dino_content"] = dr["dino_content"]
        out.append(rr)
    return out


def from_dino_summary(path: Path, metrics_path: Path | None = None) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    out = {
        "n_all": d.get("n_all"),
        "n_cross": d.get("n_off_diagonal"),
        "all_dino_style": d.get("all_dino_s"),
        "cross_dino_style": d.get("off_dino_s"),
        "all_dino_content": d.get("all_dino_c"),
        "cross_dino_content": d.get("off_dino_c"),
        "all_clip_style": d.get("all_clip_s"),
        "all_content_lpips": d.get("all_lpips"),
        "source_dino_summary": str(path),
    }
    if metrics_path and metrics_path.is_file():
        rows = [normalize_row(r) for r in load_rows(metrics_path)]
        # if dino columns missing, only recompute clip/lpips cross
        keys = [k for k in ["clip_style", "content_lpips", "dino_style", "dino_content"] if k in rows[0]]
        a = aggregate(rows, keys)
        out.update({k: v for k, v in a.items() if v is not None})
        # prefer dino off from summary if present
        if d.get("off_dino_s") is not None:
            out["cross_dino_style"] = d["off_dino_s"]
        if d.get("off_dino_c") is not None:
            out["cross_dino_content"] = d["off_dino_c"]
        if d.get("all_dino_s") is not None:
            out["all_dino_style"] = d["all_dino_s"]
        if d.get("all_dino_c") is not None:
            out["all_dino_content"] = d["all_dino_c"]
        out["source_metrics"] = str(metrics_path)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    PACK.mkdir(parents=True, exist_ok=True)

    ledger = {}

    # ---- Ablation rows with known sources ----
    ablations = {
        "WEAVE": {
            "dino": ROOT / "aaai_submission/final_v6/eval_outputs/d5_750/dino_summary.json",
            "metrics": ROOT / "runs/submission/canonical_oriented_epoch4/paper_eval/epoch_0004/metrics.csv",
        },
        "final-only AdaIN": {
            "dino": ROOT / "eval_outputs/d5_750_endpoint_only/dino_summary.json",
            "metrics": ROOT / "eval_outputs/d5_750_endpoint_only/metrics.csv",
        },
        "zero-velocity (pure analytic)": {
            "dino": ROOT / "aaai_submission/final_v6/eval_outputs/d5_750_zero_velocity/dino_summary.json",
            "metrics": ROOT / "aaai_submission/final_v6/eval_outputs/d5_750_zero_velocity/metrics.csv",
        },
        # also keep oriented_e4 from main cross protocol as alternative WEAVE
        "WEAVE (oriented e4 reval)": {
            "metrics": ROOT / "results/cross_protocol/D5-512/weave_oriented_e4_metrics.csv",
        },
    }

    abl_out = {}
    for name, src in ablations.items():
        if "dino" in src and src["dino"].is_file():
            a = from_dino_summary(src["dino"], src.get("metrics"))
        elif src.get("metrics") and Path(src["metrics"]).is_file():
            rows = load_rows(Path(src["metrics"]))
            keys = [k for k in ["clip_style", "content_lpips", "dino_style", "dino_content"] if k in rows[0]]
            a = aggregate(rows, keys)
            a["source_metrics"] = str(src["metrics"])
        else:
            a = {"error": "missing"}
        abl_out[name] = a
        print(name, {k: a.get(k) for k in a if k.startswith(("all_", "cross_", "n_"))})

    # ---- Other5 ----
    other5 = {}
    for name, folder in [
        ("WEAVE", "ours"),
        ("SaMam", "samam"),
        ("SaMST", "samst"),
        ("Identity", "idt"),
    ]:
        base = ROOT / "exp/other5_eval" / folder
        metrics = base / "metrics.csv"
        dino_csv = base / "dino_metrics.csv"
        dino_sum = base / "dino_summary.json"
        if dino_sum.is_file() and metrics.is_file():
            # merge dino csv for per-sample if available
            if dino_csv.is_file():
                rows = merge_dino_into_metrics(metrics, dino_csv)
                keys = [k for k in ["clip_style", "content_lpips", "dino_style", "dino_content"] if any(k in r for r in rows)]
                # ensure keys from either
                keys = []
                for k in ["clip_style", "content_lpips", "dino_style", "dino_content"]:
                    if any(k in r for r in rows):
                        keys.append(k)
                a = aggregate(rows, keys)
                # override with summary off if present
                d = json.loads(dino_sum.read_text(encoding="utf-8"))
                a["all_dino_style"] = d.get("all_dino_s", a.get("all_dino_style"))
                a["cross_dino_style"] = d.get("off_dino_s", a.get("cross_dino_style"))
                a["all_dino_content"] = d.get("all_dino_c", a.get("all_dino_content"))
                a["cross_dino_content"] = d.get("off_dino_c", a.get("cross_dino_content"))
                a["source_dino_summary"] = str(dino_sum)
                a["source_metrics"] = str(metrics)
            else:
                a = from_dino_summary(dino_sum, metrics)
            other5[name] = a
            print("Other5", name, {k: a.get(k) for k in a if k.startswith(("all_", "cross_", "n_"))})
        elif metrics.is_file():
            rows = load_rows(metrics)
            keys = [k for k in ["clip_style", "content_lpips", "dino_style", "dino_content"] if k in rows[0]]
            other5[name] = aggregate(rows, keys)
            other5[name]["source_metrics"] = str(metrics)
            print("Other5", name, other5[name])

    # ---- Plugin (already 600 cross pairs in paper; re-verify) ----
    plugin = {}
    for name, folder in [
        ("Base", "sd15_vanilla"),
        ("+ wavelet alignment", "sd15_weave"),
    ]:
        metrics = ROOT / "exp/plugin_sd15" / folder / "eval" / "metrics.csv"
        rows = [normalize_row(r) for r in load_rows(metrics)]
        keys = [k for k in ["clip_style", "content_lpips", "dino_style", "dino_content"] if k in rows[0]]
        # plugin board may already be transfer-only; report both
        a = aggregate(rows, keys)
        # if no identity pairs, all == cross
        a["note"] = "pair set may already exclude identity"
        a["source_metrics"] = str(metrics)
        plugin[name] = a
        print("Plugin", name, {k: a.get(k) for k in a if k.startswith(("all_", "cross_", "n_"))})

    # Paper-reported ablation rows without per-sample sources remain as all-only placeholders
    missing_ablation = {
        "w/o AdaIN": {"paper_all": {"dino_style": 0.4821, "clip_style": 0.7005, "content_lpips": 0.2258, "dino_content": 0.8455}, "status": "missing_per_sample"},
        "w/o HF conditioning": {"paper_all": {"dino_style": 0.4825, "clip_style": 0.7157, "content_lpips": 0.2837, "dino_content": 0.8125}, "status": "missing_per_sample"},
        "lambda_LL=1.0": {"paper_all": {"dino_style": 0.4910, "clip_style": 0.7180, "content_lpips": 0.2626, "dino_content": 0.7874}, "status": "missing_per_sample"},
        "direct endpoint": {"paper_all": {"dino_style": 0.4894, "clip_style": 0.7172, "content_lpips": 0.3138, "dino_content": 0.7966}, "status": "missing_per_sample"},
        "learned HH head": {"paper_all": {"dino_style": 0.4930, "clip_style": 0.7164, "content_lpips": 0.2670, "dino_content": 0.8061}, "status": "missing_per_sample"},
    }
    for k, v in missing_ablation.items():
        abl_out[k] = v

    # VAE port: paper all-only, no per-sample found
    vae_port = {
        "SD1.5 EMA (main)": {"paper_all": {"dino_style": 0.492, "clip_style": 0.713, "content_lpips": 0.259, "dino_content": 0.810}, "status": "use_main_cross_from_weave"},
        "FLUX.1": {"paper_all": {"dino_style": 0.489, "clip_style": 0.713, "content_lpips": 0.256, "dino_content": 0.780}, "status": "missing_per_sample"},
        "FLUX.2": {"paper_all": {"dino_style": 0.492, "clip_style": 0.717, "content_lpips": 0.239, "dino_content": 0.873}, "status": "missing_per_sample"},
        "Qwen-Image": {"paper_all": {"dino_style": 0.478, "clip_style": 0.714, "content_lpips": 0.263, "dino_content": 0.825}, "status": "missing_per_sample"},
    }
    # attach main weave cross as SD1.5 EMA cross estimate
    if "WEAVE" in abl_out:
        vae_port["SD1.5 EMA (main)"]["cross"] = {
            "dino_style": abl_out["WEAVE"].get("cross_dino_style"),
            "clip_style": abl_out["WEAVE"].get("cross_clip_style"),
            "content_lpips": abl_out["WEAVE"].get("cross_content_lpips"),
            "dino_content": abl_out["WEAVE"].get("cross_dino_content"),
        }

    ledger = {
        "protocol": "cross = mean over src_style != tgt_style",
        "ablation": abl_out,
        "other5": other5,
        "plugin": plugin,
        "vae_port": vae_port,
    }
    (OUT / "aux_tables_cross.json").write_text(json.dumps(ledger, indent=2), encoding="utf-8")

    # Markdown report
    lines = [
        "# Auxiliary tables under cross protocol",
        "",
        "Sources re-aggregated from existing per-sample metrics / dino_summary `off_*` fields.",
        "",
        "## Ablation (D5)",
        "",
        "| Variant | n_cross | DINO-S all | DINO-S cross | CLIP-S all | CLIP-S cross | LPIPS all | LPIPS cross | DINO-C all | DINO-C cross | status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    def f(v):
        return f"{v:.4f}" if isinstance(v, float) else "—"

    for name, a in abl_out.items():
        if a.get("status") == "missing_per_sample":
            pa = a["paper_all"]
            lines.append(
                f"| {name} | — | {pa['dino_style']:.4f} | — | {pa['clip_style']:.4f} | — | {pa['content_lpips']:.4f} | — | {pa['dino_content']:.4f} | — | missing per-sample |"
            )
            continue
        lines.append(
            f"| {name} | {a.get('n_cross','—')} | {f(a.get('all_dino_style'))} | {f(a.get('cross_dino_style'))} | "
            f"{f(a.get('all_clip_style'))} | {f(a.get('cross_clip_style'))} | {f(a.get('all_content_lpips'))} | "
            f"{f(a.get('cross_content_lpips'))} | {f(a.get('all_dino_content'))} | {f(a.get('cross_dino_content'))} | ok |"
        )

    lines += [
        "",
        "## Other5 zero-shot",
        "",
        "| Method | DINO-S all | DINO-S cross | CLIP-S all | CLIP-S cross | LPIPS all | LPIPS cross | DINO-C all | DINO-C cross |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, a in other5.items():
        lines.append(
            f"| {name} | {f(a.get('all_dino_style'))} | {f(a.get('cross_dino_style'))} | "
            f"{f(a.get('all_clip_style'))} | {f(a.get('cross_clip_style'))} | {f(a.get('all_content_lpips'))} | "
            f"{f(a.get('cross_content_lpips'))} | {f(a.get('all_dino_content'))} | {f(a.get('cross_dino_content'))} |"
        )

    lines += [
        "",
        "## Plugin SD1.5 (pair set already ~cross in paper)",
        "",
        "| Setting | n_all | n_cross | DINO-S | CLIP-S | LPIPS | DINO-C |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, a in plugin.items():
        # prefer cross if different else all
        lines.append(
            f"| {name} | {a.get('n_all')} | {a.get('n_cross')} | "
            f"{f(a.get('cross_dino_style') or a.get('all_dino_style'))} | "
            f"{f(a.get('cross_clip_style') or a.get('all_clip_style'))} | "
            f"{f(a.get('cross_content_lpips') or a.get('all_content_lpips'))} | "
            f"{f(a.get('cross_dino_content') or a.get('all_dino_content'))} |"
        )

    lines += [
        "",
        "## VAE portability",
        "",
        "Per-sample boards for FLUX/Qwen not found under local tree; only SD1.5 main has cross via WEAVE reval.",
        "",
        "| Frozen VAE | paper all DINO-S | cross DINO-S | status |",
        "|---|---:|---:|---|",
    ]
    for name, a in vae_port.items():
        pa = a.get("paper_all", {})
        cr = a.get("cross", {})
        lines.append(
            f"| {name} | {pa.get('dino_style','—')} | {f(cr.get('dino_style')) if cr else '—'} | {a.get('status')} |"
        )

    (OUT / "AUX_TABLES_CROSS.md").write_text("\n".join(lines), encoding="utf-8")
    print("WROTE", OUT / "AUX_TABLES_CROSS.md")

    # Pack submission-facing summary
    pack = {
        "main_table": "results/cross_protocol/CROSS_PROTOCOL_REPORT.md",
        "main_summary_json": "results/cross_protocol/all_summary.json",
        "aux_tables": "results/cross_protocol/aux_tables/AUX_TABLES_CROSS.md",
        "aux_json": "results/cross_protocol/aux_tables/aux_tables_cross.json",
        "d5_per_sample": "results/cross_protocol/D5-512/",
        "p2a_per_sample": "results/cross_protocol/P2A-256/",
        "paper_updates": {
            "ablation_cross_available": [k for k, v in abl_out.items() if v.get("cross_dino_style") is not None or v.get("cross_clip_style") is not None],
            "ablation_missing_per_sample": list(missing_ablation.keys()),
            "other5_cross_available": list(other5.keys()),
            "plugin_already_cross_pair_set": True,
            "vae_port_cross_only_sd15": True,
        },
    }
    (PACK / "INDEX.json").write_text(json.dumps(pack, indent=2), encoding="utf-8")
    # copy key reports into pack
    for src in [
        ROOT / "results/cross_protocol/CROSS_PROTOCOL_REPORT.md",
        ROOT / "results/cross_protocol/all_summary.json",
        OUT / "AUX_TABLES_CROSS.md",
        OUT / "aux_tables_cross.json",
    ]:
        if src.is_file():
            shutil.copy2(src, PACK / src.name)
    (PACK / "README.md").write_text(
        "\n".join(
            [
                "# Cross-protocol evaluation pack",
                "",
                "All metrics here use **cross** = mean over pairs with `src_style != tgt_style` (off-diagonal), unless noted.",
                "",
                "## Contents",
                "",
                "- `CROSS_PROTOCOL_REPORT.md` — main D5/P2A method board (all vs cross)",
                "- `all_summary.json` — machine-readable main board",
                "- `AUX_TABLES_CROSS.md` / `aux_tables_cross.json` — ablation, Other5, plugin, VAE",
                "- `INDEX.json` — path index",
                "",
                "## Gaps",
                "",
                "Some ablation variants and FLUX/Qwen VAE rows only have paper **all** numbers locally (no per-sample CSV). Re-run those evals with saved metrics.csv to complete cross.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print("WROTE pack", PACK)


if __name__ == "__main__":
    main()
