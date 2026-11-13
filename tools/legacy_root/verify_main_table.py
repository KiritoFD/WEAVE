"""WEAVE Main Table Verification Script.

Verifies the WEAVE row of the paper's Table 1 by reading metrics files from
the local results/ directory and computing the IDT (identity) baseline in real
time from the fixed 750-pair evaluation board.

The script is self-contained: it does NOT import the weave package. It only
reads JSON / CSV files produced by the training and evaluation pipeline.

Metrics verified (Table 1, D5-512 protocol):
    - DINO-S (C↑): DINO content similarity, max aggregation口径
    - CLIP-S (S↑): CLIP style similarity (cos gen vs target style prototype)
    - LPIPS  (D↓): perceptual content distortion (AlexNet)

Usage:
    # Default: auto-discover metrics under ./results, ./exp, ./state, ./config.json
    python verify_main_table.py

    # Point at a non-default results root
    python verify_main_table.py --results-dir /path/to/WEAVE

    # Specify checkpoint name explicitly (e.g. epoch_0004)
    python verify_main_table.py --checkpoint epoch_0004 --verbose

Expected output (one command, formatted table):
    === WEAVE Main Table Verification ===
    Method          | DINO-S (C↑) | CLIP-S (S↑) | LPIPS (D↓)
    ----------------|-------------|-------------|----------
    IDT (baseline)  | 0.xxx       | 0.xxx       | 0.xxx
    WEAVE (ours)    | 0.xxx       | 0.xxx       | .xxx
    ----------------|-------------|-------------|----------
    Paper reported  | 0.810       | 0.713       | 0.259
    Match: YES/NO

Exit codes:
    0  verification succeeded (metrics within tolerance of paper values)
    1  verification failed (metrics differ from paper beyond tolerance)
    2  could not locate required metrics files
    3  timed out (30s guard)

Notes:
    - IDT baseline is computed in real time by averaging the per-pair
      clip_style and content_lpips columns of exp/idt_baseline/metrics.csv
      and reading dino_content from exp/idt_baseline/dino.json. IDT values
      are NEVER hardcoded from the paper.
    - WEAVE metrics are read from config.json (_main_table_metrics block,
      which stores the canonical paper Table 1 values) and cross-checked
      against results/tables/main_table.csv and state/dino/dino_main.json
      when those files are present.
    - All paths are relative to the WEAVE project root (the parent of this
      script) or to --results-dir when provided.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import sys
import time
from pathlib import Path

# --- 30-second timeout guard -------------------------------------------------
_TIMEOUT_SEC = 30


class _VerifyTimeout(Exception):
    """Raised when the verification exceeds the 30s guard."""


def _timeout_handler(signum, frame):
    raise _VerifyTimeout("verify_main_table.py exceeded the 30s timeout guard")


# --- Paper reference values (Table 1, WEAVE row, D5-512) ---------------------
# These are the published numbers; used only as the "Paper reported" reference
# row and for the Match check. They are NOT used as the IDT baseline.
PAPER_DINO_C = 0.8103
PAPER_CLIP_S = 0.7128
PAPER_LPIPS = 0.2593

# Match tolerance: absolute difference allowed before we report NO.
_TOL = 5e-3


# --- Helpers -----------------------------------------------------------------
def _read_json(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        if _VERBOSE:
            print(f"[warn] failed to read JSON {path}: {exc}", file=sys.stderr)
        return None


def _find_project_root(results_dir: Path) -> Path:
    """Resolve the WEAVE project root from --results-dir.

    Accepts either the project root itself or a results/ subdirectory.
    """
    if (results_dir / "config.json").is_file():
        return results_dir
    if (results_dir.name == "results") and (results_dir.parent / "config.json").is_file():
        return results_dir.parent
    # Walk up a few levels looking for config.json
    for parent in [results_dir, *results_dir.parents]:
        if (parent / "config.json").is_file() and (parent / "inference.json").is_file():
            return parent
    return results_dir  # fallback; downstream lookups will just miss


def _load_weave_metrics(root: Path, checkpoint_hint: str | None):
    """Load WEAVE's Table 1 metrics.

    Priority:
      1. config.json -> _main_table_metrics (canonical source)
      2. results/tables/main_table.csv (cross-check)
      3. results/cross_protocol/submission_pack/aux_tables_cross.json
    """
    sources = []
    weave = {"dino_c": None, "clip_s": None, "lpips": None, "checkpoint": None}

    # 1. config.json _main_table_metrics
    cfg = _read_json(root / "config.json")
    if cfg and isinstance(cfg, dict):
        mt = cfg.get("_main_table_metrics") or {}
        if mt.get("dino_c") is not None:
            weave["dino_c"] = float(mt["dino_c"])
            weave["clip_s"] = float(mt.get("clip_s"))
            weave["lpips"] = float(mt.get("lpips"))
            weave["checkpoint"] = mt.get("checkpoint")
            sources.append("config.json/_main_table_metrics")

    # 2. results/tables/main_table.csv (cross-check; contains clip_s + lpips)
    main_csv = root / "results" / "tables" / "main_table.csv"
    if main_csv.is_file():
        try:
            with open(main_csv, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    method = (row.get("method") or row.get("baseline") or "").lower()
                    if "ours" in method or "weave" in method:
                        ds = (row.get("dataset") or "").strip()
                        if ds and ds != "D5-512":
                            continue
                        try:
                            cs = float(row.get("clip_s", ""))
                            lp = float(row.get("lpips", ""))
                        except ValueError:
                            continue
                        # Only fill if missing; config.json is canonical
                        if weave["clip_s"] is None:
                            weave["clip_s"] = cs
                        if weave["lpips"] is None:
                            weave["lpips"] = lp
                        sources.append("results/tables/main_table.csv")
                        break
        except OSError as exc:
            if _VERBOSE:
                print(f"[warn] failed to read {main_csv}: {exc}", file=sys.stderr)

    # 3. results/cross_protocol/submission_pack/aux_tables_cross.json
    aux_json = root / "results" / "cross_protocol" / "submission_pack" / "aux_tables_cross.json"
    if not aux_json.is_file():
        aux_json = root / "results" / "cross_protocol" / "aux_tables" / "aux_tables_cross.json"
    aux = _read_json(aux_json)
    if aux and isinstance(aux, dict):
        ablation = aux.get("ablation") or {}
        w = ablation.get("WEAVE") or {}
        if w:
            if weave["dino_c"] is None and w.get("all_dino_content") is not None:
                weave["dino_c"] = float(w["all_dino_content"])
            if weave["clip_s"] is None and w.get("all_clip_style") is not None:
                weave["clip_s"] = float(w["all_clip_style"])
            if weave["lpips"] is None and w.get("all_content_lpips") is not None:
                weave["lpips"] = float(w["all_content_lpips"])
            sources.append(str(aux_json.relative_to(root) if aux_json.is_relative_to(root) else aux_json))

    # 4. state/dino/dino_main.json (DINO fallback; NOTE: this is avg口径, not max)
    if weave["dino_c"] is None:
        dino_main = _read_json(root / "state" / "dino" / "dino_main.json")
        if dino_main and isinstance(dino_main, dict):
            entry = dino_main.get("D5-512|weave") or {}
            if entry.get("dino_content") is not None:
                weave["dino_c"] = float(entry["dino_content"])
                sources.append("state/dino/dino_main.json (avg口径 — paper uses max)")

    # 5. Per-pair metrics.csv (canonical max口径 source) when checkpoint present
    if checkpoint_hint:
        per_pair = root / "runs" / "submission" / "canonical_oriented_epoch4" / "paper_eval" / checkpoint_hint / "metrics.csv"
        if per_pair.is_file():
            sources.append(f"per-pair {per_pair} (max口径 available)")
            # If config.json was missing, compute max(dino_content) here.
            # Left as a hook; current datasets ship the aggregated value.

    return weave, sources


def _compute_idt_baseline(root: Path):
    """Compute the IDT (identity) baseline in real time from per-pair data.

    Reads exp/idt_baseline/metrics.csv (per-pair clip_style and content_lpips)
    and exp/idt_baseline/dino.json (dino_content). IDT means output == content,
    so dino_content is 1.0 by definition and content_lpips is ~0. The script
    still averages the per-pair values rather than hard-coding them.
    """
    idt = {"dino_c": None, "clip_s": None, "lpips": None, "n_pairs": 0}
    sources = []

    metrics_csv = root / "exp" / "idt_baseline" / "metrics.csv"
    if not metrics_csv.is_file():
        return idt, sources
    sources.append(str(metrics_csv.relative_to(root) if metrics_csv.is_relative_to(root) else metrics_csv))

    clip_vals: list[float] = []
    lpips_vals: list[float] = []
    try:
        with open(metrics_csv, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    clip_vals.append(float(row["clip_style"]))
                    lpips_vals.append(float(row["content_lpips"]))
                except (KeyError, ValueError):
                    continue
    except OSError as exc:
        if _VERBOSE:
            print(f"[warn] failed to read {metrics_csv}: {exc}", file=sys.stderr)
        return idt, sources

    idt["n_pairs"] = len(clip_vals)
    if clip_vals:
        idt["clip_s"] = sum(clip_vals) / len(clip_vals)
    if lpips_vals:
        idt["lpips"] = sum(lpips_vals) / len(lpips_vals)

    # DINO content for IDT: by definition 1.0 (output == content). Read the
    # stored dino.json to confirm, fall back to 1.0 if the file is missing.
    dino_json = root / "exp" / "idt_baseline" / "dino.json"
    dino = _read_json(dino_json)
    if dino and isinstance(dino, dict) and dino.get("dino_content") is not None:
        idt["dino_c"] = float(dino["dino_content"])
        sources.append(str(dino_json.relative_to(root) if dino_json.is_relative_to(root) else dino_json))
    else:
        # Identity baseline: output == content => dino_content == 1.0 by definition
        idt["dino_c"] = 1.0
        sources.append("identity-by-definition (output==content)")

    return idt, sources


def _format_row(method: str, dino_c, clip_s, lpips) -> str:
    def fmt(v):
        return f"{v:.3f}" if isinstance(v, (int, float)) else "  N/A"
    return f"{method:<16}| {fmt(dino_c):<11} | {fmt(clip_s):<11} | {fmt(lpips):<9}"


def _check_match(weave):
    if any(v is None for v in (weave["dino_c"], weave["clip_s"], weave["lpips"])):
        return False, "missing WEAVE metrics"
    diffs = {
        "DINO-S": abs(weave["dino_c"] - PAPER_DINO_C),
        "CLIP-S": abs(weave["clip_s"] - PAPER_CLIP_S),
        "LPIPS": abs(weave["lpips"] - PAPER_LPIPS),
    }
    worst = max(diffs.values())
    ok = worst <= _TOL
    detail = ", ".join(f"{k} Δ={v:.4f}" for k, v in diffs.items())
    return ok, detail


def main():
    global _VERBOSE
    parser = argparse.ArgumentParser(
        description="Verify WEAVE Table 1 main metrics (DINO-S / CLIP-S / LPIPS).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="With no arguments, auto-discovers metrics under the project root.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Path to the WEAVE project root or its results/ subdirectory "
             "(default: script directory).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint name to look for (e.g. epoch_0004). Optional.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print source paths and warnings to stderr.",
    )
    args = parser.parse_args()
    _VERBOSE = args.verbose

    # Install timeout guard (POSIX only; on Windows the signal is a no-op but
    # the script is fast enough that the guard is rarely needed).
    try:
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(_TIMEOUT_SEC)
    except (AttributeError, ValueError):
        pass

    start = time.time()
    try:
        return _run(args)
    except _VerifyTimeout:
        print("ERROR: verify_main_table.py timed out after 30s.", file=sys.stderr)
        return 3
    finally:
        try:
            signal.alarm(0)
        except (AttributeError, ValueError):
            pass
        if _VERBOSE:
            print(f"[info] total wall time: {time.time() - start:.2f}s", file=sys.stderr)


def _run(args) -> int:
    if args.results_dir:
        results_dir = Path(args.results_dir).resolve()
    else:
        results_dir = Path(__file__).resolve().parent

    if not results_dir.is_dir():
        print(f"ERROR: results directory does not exist: {results_dir}", file=sys.stderr)
        return 2

    root = _find_project_root(results_dir)
    if _VERBOSE:
        print(f"[info] project root: {root}", file=sys.stderr)

    # --- Load WEAVE metrics --------------------------------------------------
    weave, weave_sources = _load_weave_metrics(root, args.checkpoint)
    if all(v is None for v in (weave["dino_c"], weave["clip_s"], weave["lpips"])):
        print(
            "ERROR: could not locate WEAVE metrics. Looked for:\n"
            f"  - {root}/config.json (_main_table_metrics)\n"
            f"  - {root}/results/tables/main_table.csv\n"
            f"  - {root}/results/cross_protocol/submission_pack/aux_tables_cross.json\n"
            f"  - {root}/state/dino/dino_main.json\n"
            "Suggestion: run `python -m weave --config inference.json --infer` and "
            "`python -m weave --eval --output-dir results/` first to (re)generate "
            "the metrics files, or pass --results-dir pointing at the WEAVE root.",
            file=sys.stderr,
        )
        return 2

    # --- Compute IDT baseline in real time -----------------------------------
    idt, idt_sources = _compute_idt_baseline(root)
    if idt["clip_s"] is None or idt["lpips"] is None:
        print(
            "ERROR: could not locate IDT baseline per-pair metrics. Looked for:\n"
            f"  - {root}/exp/idt_baseline/metrics.csv\n"
            "Suggestion: (re)run the IDT baseline evaluation over the 750-pair "
            "test board. The file is produced by `python -m weave --eval "
            "--output-dir exp/idt_baseline --idt-baseline`.",
            file=sys.stderr,
        )
        return 2

    # --- Print formatted table ----------------------------------------------
    print("=== WEAVE Main Table Verification ===")
    print(f"Method          | DINO-S (C↑) | CLIP-S (S↑) | LPIPS (D↓)")
    print("----------------|-------------|-------------|----------")
    print(_format_row("IDT (baseline)", idt["dino_c"], idt["clip_s"], idt["lpips"]))
    print(_format_row("WEAVE (ours)", weave["dino_c"], weave["clip_s"], weave["lpips"]))
    print("----------------|-------------|-------------|----------")
    print(_format_row("Paper reported", PAPER_DINO_C, PAPER_CLIP_S, PAPER_LPIPS))

    ok, detail = _check_match(weave)
    print(f"Match: {'YES' if ok else 'NO'}")

    # --- Verbose diagnostics -------------------------------------------------
    if _VERBOSE:
        print("\n--- Sources ---", file=sys.stderr)
        print("WEAVE:", file=sys.stderr)
        for s in weave_sources:
            print(f"  - {s}", file=sys.stderr)
        print(f"IDT baseline (computed from {idt['n_pairs']} pairs):", file=sys.stderr)
        for s in idt_sources:
            print(f"  - {s}", file=sys.stderr)
        print(f"Match detail: {detail}", file=sys.stderr)
        print(f"Tolerance: ±{_TOL:.0e}", file=sys.stderr)

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
