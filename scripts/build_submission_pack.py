"""Build submission pack: every table row -> 750-image board + metrics under one tree.

Layout:
  results/cross_protocol/submission_pack/
    MANIFEST.json
    README.md
    reports/          # markdown + json summaries
    boards/
      D5-512/<method>/{images, metrics.csv, summary.json}
      P2A-256/<method>/...
      ablation/<name>/...
      other5/<name>/...
      plugin/<name>/...

Images are junction-linked when local (no duplicate disk); metrics always copied.
Remote boards are listed in MANIFEST for pull via build_submission_pack_pull.bat.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "results" / "cross_protocol" / "submission_pack"
BOARDS = PACK / "boards"
REPORTS = PACK / "reports"

# Main-table methods (paper order)
D5_METHODS = {
    "identity": ROOT / "results/D5-512/identity",
    "tgt": ROOT / "results/D5-512/tgt",
    "sdturbo": ROOT / "results/D5-512/sdturbo",
    "stylealigned": ROOT / "results/D5-512/stylealigned",
    "zstar": ROOT / "results/D5-512/zstar",
    "styleshot": ROOT / "results/D5-512/styleshot",
    "styleid": ROOT / "results/D5-512/styleid",
    "cut": ROOT / "results/D5-512/cut",
    "samst": ROOT / "results/D5-512/samst",
    "samam": ROOT / "results/D5-512/samam",
    "stytr2": ROOT / "results/D5-512/stytr2",
    "aespa": ROOT / "results/D5-512/aespa",
    "seedream": ROOT / "results/D5-512/seedream",
    "wct": ROOT / "results/D5-512/wct",
    "weave": ROOT / "results/D5-512/weave_oriented_e4",
}

P2A_METHODS = {
    "identity": ROOT / "results/P256/identity",
    "tgt": ROOT / "results/P256/tgt",
    "sdturbo": ROOT / "results/P256/sdturbo",
    "stylealigned": ROOT / "results/P256/stylealigned",
    "zstar": ROOT / "results/P256/zstar",
    "styleshot": ROOT / "results/P256/styleshot",
    "styleid": ROOT / "results/P256/styleid",
    "cut": ROOT / "results/P256/cut",
    "samst": ROOT / "results/P256/samst",
    "samam": ROOT / "results/P256/samam",
    "stytr2": ROOT / "results/P256/stytr2",
    "aespa": ROOT / "results/P256/aespa",
    "seedream": ROOT / "results/P256/seedream",
    "wct": ROOT / "results/P256/wct",
    "weave": ROOT / "results/P256/weave",
}

# Paper ablation + known local evals
ABLATION_LOCAL = {
    "weave_main": {
        "images": ROOT / "results/D5-512/weave_oriented_e4",
        "metrics": ROOT / "results/cross_protocol/D5-512/weave_oriented_e4_metrics.csv",
        "dino_summary": ROOT / "aaai_submission/final_v6/eval_outputs/d5_750/dino_summary.json",
    },
    "final_only_adain": {
        "images": ROOT / "eval_outputs/d5_750_endpoint_only/images",
        "metrics": ROOT / "eval_outputs/d5_750_endpoint_only/metrics.csv",
        "dino_summary": ROOT / "eval_outputs/d5_750_endpoint_only/dino_summary.json",
    },
    "zero_velocity": {
        "images": ROOT / "aaai_submission/final_v6/eval_outputs/d5_750_zero_velocity/images",
        "metrics": ROOT / "aaai_submission/final_v6/eval_outputs/d5_750_zero_velocity/metrics.csv",
        "dino_summary": ROOT / "aaai_submission/final_v6/eval_outputs/d5_750_zero_velocity/dino_summary.json",
    },
}

OTHER5_LOCAL = {
    "weave": ROOT / "exp/other5_eval/ours",
    "samam": ROOT / "exp/other5_eval/samam",
    "samst": ROOT / "exp/other5_eval/samst",
    "identity": ROOT / "exp/other5_eval/idt",
}

PLUGIN_LOCAL = {
    "base": ROOT / "exp/plugin_sd15/sd15_vanilla/eval",
    "weave_plugin": ROOT / "exp/plugin_sd15/sd15_weave/eval",
}

# Remote ablation boards (paper-scale); pull script uses these paths
REMOTE_ABLATION_V2 = [
    "a01_wo_endpoint_adain",
    "a02_wo_cross_attn",
    "a03_wo_flow",
    "b01_wll_0",
    "b02_wll_20",
    "b07_whh_0",
    "b08_whh_4",
    "d01_adain_0",
    "d02_adain_05",
    "d03_adain_20",
    "d06_steps_1",
    "d07_steps_32",
]
REMOTE_ROOT = r"I:\Github\Latent_Style\WEAVE"


def count_images(d: Path) -> int:
    if not d.exists():
        return 0
    if d.is_file():
        return 0
    # prefer images/ subdir
    img_dir = d / "images" if (d / "images").is_dir() else d
    return sum(1 for p in img_dir.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"})


def ensure_junction(link: Path, target: Path) -> str:
    """Create directory junction link -> target. Returns status."""
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        # already linked?
        return "exists"
    if not target.exists():
        return "missing_target"
    # Windows junction
    cmd = f'cmd /c mklink /J "{link}" "{target}"'
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode == 0:
        return "junction"
    # fallback copy is too heavy; just record failure
    return f"fail:{r.stderr.strip() or r.stdout.strip()}"


def copy_if(src: Path, dst: Path) -> bool:
    if not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def find_metrics_for_method(board: str, method: str) -> Path | None:
    # prefer cross_protocol recomputed
    p = ROOT / "results/cross_protocol" / board / f"{method}_metrics.csv"
    if p.is_file():
        return p
    if method == "weave" and board == "D5-512":
        p2 = ROOT / "results/cross_protocol/D5-512/weave_oriented_e4_metrics.csv"
        if p2.is_file():
            return p2
    return None


def pack_method(board: str, method: str, image_src: Path, metrics_src: Path | None, extra: dict | None = None) -> dict:
    dest = BOARDS / board / method
    dest.mkdir(parents=True, exist_ok=True)
    entry = {
        "board": board,
        "method": method,
        "image_src": str(image_src) if image_src else None,
        "n_images": count_images(image_src) if image_src else 0,
        "metrics_src": str(metrics_src) if metrics_src else None,
        "link_status": None,
    }
    # images junction
    if image_src and image_src.exists():
        img_target = image_src / "images" if (image_src / "images").is_dir() else image_src
        entry["link_status"] = ensure_junction(dest / "images", img_target.resolve())
        entry["n_images"] = count_images(img_target)
    else:
        entry["link_status"] = "no_images"
    # metrics
    if metrics_src and metrics_src.is_file():
        copy_if(metrics_src, dest / "metrics.csv")
        entry["has_metrics"] = True
    else:
        # try under image_src
        for cand in ["metrics.csv", "eval/metrics.csv"]:
            if image_src and (image_src / cand).is_file():
                copy_if(image_src / cand, dest / "metrics.csv")
                entry["has_metrics"] = True
                entry["metrics_src"] = str(image_src / cand)
                break
        else:
            entry["has_metrics"] = False
    # dino / summary
    if image_src:
        for name in ["dino_summary.json", "summary.json", "dino_metrics.csv"]:
            for base in [image_src, image_src / "eval", image_src.parent]:
                f = base / name if base.is_dir() else None
                if f and f.is_file():
                    copy_if(f, dest / name)
    if extra:
        for k, p in extra.items():
            if isinstance(p, Path) and p.is_file():
                copy_if(p, dest / Path(p).name)
    return entry


def main():
    PACK.mkdir(parents=True, exist_ok=True)
    BOARDS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    manifest = {"boards": {}, "remote_to_pull": [], "protocol": "cross = src_style != tgt_style means in metrics where available"}

    # D5 main
    for method, src in D5_METHODS.items():
        metrics = find_metrics_for_method("D5-512", method)
        # also map weave metrics
        if method == "weave":
            metrics = ROOT / "results/cross_protocol/D5-512/weave_oriented_e4_metrics.csv"
        entry = pack_method("D5-512", method, src, metrics)
        manifest["boards"].setdefault("D5-512", {})[method] = entry
        print(f"D5 {method}: imgs={entry['n_images']} metrics={entry['has_metrics']} link={entry['link_status']}")

    # P2A main
    for method, src in P2A_METHODS.items():
        metrics = find_metrics_for_method("P2A-256", method)
        if method == "weave":
            metrics = ROOT / "results/cross_protocol/P2A-256/weave_metrics.csv"
        entry = pack_method("P2A-256", method, src, metrics)
        manifest["boards"].setdefault("P2A-256", {})[method] = entry
        print(f"P2A {method}: imgs={entry['n_images']} metrics={entry['has_metrics']} link={entry['link_status']}")

    # Ablation local
    for name, info in ABLATION_LOCAL.items():
        entry = pack_method(
            "ablation",
            name,
            info.get("images"),
            info.get("metrics"),
            {k: info[k] for k in ("dino_summary",) if k in info},
        )
        manifest["boards"].setdefault("ablation", {})[name] = entry
        print(f"ablation {name}: imgs={entry['n_images']} metrics={entry['has_metrics']} link={entry['link_status']}")

    # Other5
    for name, src in OTHER5_LOCAL.items():
        metrics = src / "metrics.csv" if (src / "metrics.csv").is_file() else None
        entry = pack_method("other5", name, src, metrics)
        # also copy dino
        if (src / "dino_metrics.csv").is_file():
            copy_if(src / "dino_metrics.csv", BOARDS / "other5" / name / "dino_metrics.csv")
        if (src / "dino_summary.json").is_file():
            copy_if(src / "dino_summary.json", BOARDS / "other5" / name / "dino_summary.json")
        manifest["boards"].setdefault("other5", {})[name] = entry
        print(f"other5 {name}: imgs={entry['n_images']} metrics={entry['has_metrics']}")

    # Plugin
    for name, src in PLUGIN_LOCAL.items():
        metrics = src / "metrics.csv" if (src / "metrics.csv").is_file() else None
        # images may be sibling
        img = src / "images" if (src / "images").is_dir() else src.parent / "images"
        entry = pack_method("plugin", name, img if img.exists() else src, metrics)
        manifest["boards"].setdefault("plugin", {})[name] = entry
        print(f"plugin {name}: imgs={entry['n_images']} metrics={entry['has_metrics']}")

    # Remote ablation list for pull
    for name in REMOTE_ABLATION_V2:
        # try both layouts
        remote_cands = [
            f"{REMOTE_ROOT}\\exp\\ablation_v2\\{name}\\eval",
            f"{REMOTE_ROOT}\\exp\\ablation_v2\\{name}",
        ]
        manifest["remote_to_pull"].append(
            {
                "name": name,
                "remote_candidates": remote_cands,
                "local_dest": str(BOARDS / "ablation" / name),
                "need": ["images", "metrics.csv", "dino_summary.json", "summary.json"],
            }
        )

    # Also remote: paper main if missing, zero_velocity images, etc.
    manifest["remote_to_pull"].append(
        {
            "name": "zero_velocity_images",
            "remote_candidates": [
                f"{REMOTE_ROOT}\\aaai_submission\\final_v6\\eval_outputs\\d5_750_zero_velocity\\images",
                f"{REMOTE_ROOT}\\eval_outputs\\d5_750_zero_velocity\\images",
            ],
            "local_dest": str(BOARDS / "ablation" / "zero_velocity" / "images"),
            "need": ["*.png"],
        }
    )

    # Copy reports
    for f in [
        ROOT / "results/cross_protocol/CROSS_PROTOCOL_REPORT.md",
        ROOT / "results/cross_protocol/all_summary.json",
        ROOT / "results/cross_protocol/aux_tables/AUX_TABLES_CROSS.md",
        ROOT / "results/cross_protocol/aux_tables/aux_tables_cross.json",
    ]:
        if f.is_file():
            shutil.copy2(f, REPORTS / f.name)

    # Inventory
    inventory = []
    for board, methods in manifest["boards"].items():
        for method, e in methods.items():
            inventory.append(
                {
                    "board": board,
                    "method": method,
                    "n_images": e.get("n_images", 0),
                    "has_metrics": e.get("has_metrics", False),
                    "link_status": e.get("link_status"),
                    "complete_750": e.get("n_images", 0) >= 700 and e.get("has_metrics", False),
                }
            )
    complete = sum(1 for x in inventory if x["complete_750"])
    incomplete = [x for x in inventory if not x["complete_750"]]

    manifest["inventory_summary"] = {
        "n_entries": len(inventory),
        "n_complete_with_metrics_and_images": complete,
        "incomplete": incomplete,
    }

    (PACK / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (PACK / "INVENTORY.csv").write_text(
        "board,method,n_images,has_metrics,link_status,complete_750\n"
        + "\n".join(
            f"{x['board']},{x['method']},{x['n_images']},{x['has_metrics']},{x['link_status']},{x['complete_750']}"
            for x in inventory
        ),
        encoding="utf-8",
    )

    readme = f"""# Submission pack (cross protocol)

Every **table row** should have a board folder under `boards/` with:
- `images/` — full generated set (typically 750; junction to source when local)
- `metrics.csv` — per-sample metrics (for cross re-aggregation)
- optional `dino_summary.json` / `summary.json`

## Status
- Entries: {len(inventory)}
- Complete (≈750 imgs + metrics): {complete}
- Incomplete: {len(incomplete)} (see INVENTORY.csv)

## Layout
```
boards/D5-512/<method>/images + metrics.csv
boards/P2A-256/<method>/...
boards/ablation/<variant>/...
boards/other5/<method>/...
boards/plugin/<name>/...
reports/  — all-vs-cross ledgers
```

## Pull remote boards
Incomplete ablations live on the evaluation machine. Run:

```
scripts\\pull_submission_pack_remote.bat
```

Then re-run cross aggregation:

```
python scripts/recompute_aux_tables_cross.py
python scripts/recompute_cross_protocol.py --reuse_only
```

## Cross protocol
Means for ranking tables: average over pairs with `src_style != tgt_style`.
"""
    (PACK / "README.md").write_text(readme, encoding="utf-8")
    print("\nPACK", PACK)
    print("complete", complete, "/", len(inventory))
    if incomplete:
        print("INCOMPLETE:")
        for x in incomplete:
            print(" ", x)


if __name__ == "__main__":
    main()
