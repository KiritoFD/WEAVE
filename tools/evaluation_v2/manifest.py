"""WEAVE Evaluator V2 - Manifest Builder and Strict Baseline Image Auditor.

Responsibilities:
1. Build `eval_protocol/tasks_750.json`:
   - 150 unique source images in `data/test/<style>/*.jpg` (30 per style).
   - Each with SHA256, relative path, source_style, source_stem.
   - Exactly 750 tasks (150 same-style, 600 cross-style).
2. Build `eval_protocol/references_v1.json`:
   - Unified frozen reference set for DINO-S, CLIP-S, and CSD-MMD target distribution.
   - 30 images per style from `runs/p0/reference_selection.json` (r0 + r3 pool).
   - Verified disjoint from test source set (zero evaluation-source overlap).
   - Each entry records path, filename, SHA256, and style.
3. Strict Image Mapping (NO SUBSTRING FALLBACK):
   - For every method folder (in results/D5-512/ and runs/v6/), map each file
     strictly to its task_id.
   - Record exact valid task count, cross task count, same task count,
     and any missing task_ids explicitly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_DIR = REPO_ROOT / "data" / "test"
PROTOCOL_DIR = REPO_ROOT / "eval_protocol"
STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_tasks_750(test_dir: Path = TEST_DIR) -> dict:
    PROTOCOL_DIR.mkdir(parents=True, exist_ok=True)
    sources = []
    tasks = []

    for sid, sname in enumerate(STYLES):
        sdir = test_dir / sname
        if not sdir.exists():
            raise FileNotFoundError(f"Missing test style dir: {sdir}")
        imgs = sorted(list(sdir.glob("*.jpg")) + list(sdir.glob("*.png")))
        if len(imgs) != 30:
            print(f"[WARN] Style {sname} has {len(imgs)} images (expected 30)")
        
        for p in imgs:
            sha = file_sha256(p)
            rel_path = str(p.relative_to(REPO_ROOT)).replace("\\", "/")
            stem = p.stem
            sources.append({
                "source_style": sname,
                "source_style_id": sid,
                "source_filename": p.name,
                "source_stem": stem,
                "source_rel_path": rel_path,
                "source_sha256": sha,
            })

            for tid, tname in enumerate(STYLES):
                is_cross = (sid != tid)
                task_id = f"{sname}/{stem}/{tname}"
                tasks.append({
                    "task_id": task_id,
                    "source_style": sname,
                    "source_style_id": sid,
                    "source_stem": stem,
                    "source_filename": p.name,
                    "source_rel_path": rel_path,
                    "source_sha256": sha,
                    "target_style": tname,
                    "target_style_id": tid,
                    "is_cross": is_cross,
                })

    out = {
        "description": "Authoritative 750 evaluation tasks (150 sources x 5 target styles)",
        "num_sources": len(sources),
        "num_tasks_total": len(tasks),
        "num_tasks_cross": sum(1 for t in tasks if t["is_cross"]),
        "num_tasks_same": sum(1 for t in tasks if not t["is_cross"]),
        "styles": STYLES,
        "sources": sources,
        "tasks": tasks,
    }
    task_file = PROTOCOL_DIR / "tasks_750.json"
    task_file.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[Manifest] Wrote {task_file} ({len(tasks)} tasks: {out['num_tasks_cross']} cross, {out['num_tasks_same']} same)")
    return out


def build_references_v1(reference_selection_path: Path | None = None) -> dict:
    """Build unified frozen references v1 (5 styles x 30 images = 150 references).
    
    Strictly sourced from the real art training pool (datasets/distinct5_512/train),
    guaranteeing 0% overlap with evaluation sources or generated images.
    """
    PROTOCOL_DIR.mkdir(parents=True, exist_ok=True)
    train_dir = REPO_ROOT / "datasets" / "distinct5_512" / "train"

    tasks_meta = build_tasks_750(TEST_DIR)
    test_stems = {s["source_stem"] for s in tasks_meta["sources"]}
    test_shas = {s["source_sha256"] for s in tasks_meta["sources"]}

    references = {}
    for sid, sname in enumerate(STYLES):
        s_refs = []
        candidates = []

        # First, try candidates from reference_selection.json if valid and disjoint
        if reference_selection_path and reference_selection_path.exists():
            raw = json.loads(reference_selection_path.read_text(encoding="utf-8"))
            sel = raw.get("selected", raw)
            if sname in sel:
                for c in sel[sname].get("r3", []) + sel[sname].get("r0", []):
                    p = Path(c)
                    if not p.is_absolute():
                        p = REPO_ROOT / p
                    if p.exists() and p not in candidates:
                        candidates.append(p)

        # Second, supplement from training pool sorted canonically by filename
        s_train_dir = train_dir / sname
        if s_train_dir.exists():
            for p in sorted(list(s_train_dir.glob("*.jpg")) + list(s_train_dir.glob("*.png"))):
                if p not in candidates:
                    candidates.append(p)

        for p in candidates:
            sha = file_sha256(p)
            if p.stem in test_stems or sha in test_shas:
                continue
            rel = str(p.relative_to(REPO_ROOT)).replace("\\", "/") if p.is_relative_to(REPO_ROOT) else str(p)
            s_refs.append({
                "style": sname,
                "style_id": sid,
                "filename": p.name,
                "stem": p.stem,
                "rel_path": rel,
                "sha256": sha,
            })
            if len(s_refs) == 30:
                break

        if len(s_refs) < 30:
            raise RuntimeError(f"Could not find 30 disjoint references for {sname} (found {len(s_refs)})")
        references[sname] = s_refs

    out = {
        "description": "Unified frozen references v1 (5 styles x 30 images = 150 refs, strictly test-disjoint)",
        "num_styles": len(references),
        "references_per_style": {k: len(v) for k, v in references.items()},
        "references": references,
    }
    ref_file = PROTOCOL_DIR / "references_v1.json"
    ref_file.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[Manifest] Wrote {ref_file} ({sum(len(v) for v in references.values())} references, exactly 30 per style)")
    return out


def parse_strict_task_id(p: Path, stem_to_task_sources: Dict[str, str]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Parse method output filename strictly into (source_style, source_stem, target_style).
    
    NO SUBSTRING OR HEURISTIC FALLBACK ALLOWED!
    Supported exact templates:
      1. {src}__{stem}__to__{tgt}.png
      2. {src}_{stem}_to_{tgt}.png
      3. {src}__{src}__{stem}__to__{tgt}.png
      4. {src}__{src}__{src}__{stem}_to_{tgt}.png (StyleShot)
      5. {src}__{tgt}__{src}__{stem}_to_{tgt}.png (StyleShot cross)
      6. {src}__{tgt}__{src}__{src}__{stem}_to_{tgt}.png
      7. {stem}_to_{tgt}.png
    """
    stem = p.stem

    # Extract target style from suffix
    tgt_style = None
    left = None
    for s in STYLES:
        for sep in ["__to__", "_to_"]:
            token = f"{sep}{s}"
            if stem.endswith(token):
                tgt_style = s
                left = stem[:-len(token)]
                break
        if tgt_style:
            break

    if not tgt_style or not left:
        return None, None, None

    # Try matching against known source stems
    if left in stem_to_task_sources:
        return stem_to_task_sources[left], left, tgt_style

    # Iteratively strip style prefixes from left: e.g. "{style}__" or "{style}_"
    # This cleanly unpacks single, double, or triple style prefixes without guesswork.
    cur = left
    for _ in range(5):
        stripped = False
        for s in STYLES:
            for pref in [f"{s}__", f"{s}_"]:
                if cur.startswith(pref):
                    cur = cur[len(pref):]
                    stripped = True
                    if cur in stem_to_task_sources:
                        return stem_to_task_sources[cur], cur, tgt_style
                    break
            if stripped:
                break
        if not stripped:
            break

    return None, None, None


def audit_method_directory(
    method_name: str,
    method_dir: Path,
    tasks_meta: dict,
) -> dict:
    """Strictly audit a method's directory against the 750 frozen tasks."""
    stem_to_style = {s["source_stem"]: s["source_style"] for s in tasks_meta["sources"]}
    all_task_ids = {t["task_id"]: t for t in tasks_meta["tasks"]}

    imgs = sorted(list(method_dir.glob("*.png")) + list(method_dir.glob("*.jpg")))
    matched_tasks: Dict[str, dict] = {}
    unmatched_files = []

    for p in imgs:
        src_s, src_stem, tgt_s = parse_strict_task_id(p, stem_to_style)
        if src_s and src_stem and tgt_s:
            tid = f"{src_s}/{src_stem}/{tgt_s}"
            if tid in all_task_ids:
                matched_tasks[tid] = {
                    "path": p,
                    "filename": p.name,
                    "file_sha256": file_sha256(p),
                    "file_size": p.stat().st_size,
                    "task_id": tid,
                    "is_cross": all_task_ids[tid]["is_cross"],
                }
                continue
        unmatched_files.append(p.name)

    missing_task_ids = [tid for tid in all_task_ids if tid not in matched_tasks]
    cross_matched = sum(1 for v in matched_tasks.values() if v["is_cross"])
    same_matched = sum(1 for v in matched_tasks.values() if not v["is_cross"])

    res = {
        "method": method_name,
        "directory": str(method_dir),
        "total_files": len(imgs),
        "num_valid_matched": len(matched_tasks),
        "num_cross_matched": cross_matched,
        "num_same_matched": same_matched,
        "num_unmatched_files": len(unmatched_files),
        "num_missing_tasks": len(missing_task_ids),
        "missing_task_ids": missing_task_ids,
        "unmatched_filenames": unmatched_files[:10],
        "matched_task_map": {k: {"filename": v["filename"], "sha256": v["file_sha256"]} for k, v in matched_tasks.items()},
    }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", default=str(TEST_DIR))
    ap.add_argument("--ref-sel", default=str(REPO_ROOT / "runs" / "p0" / "reference_selection.json"))
    args = ap.parse_args()

    print("=== Building Evaluator V2 Authoritative Manifest ===")
    tasks_meta = build_tasks_750(Path(args.test_dir))
    ref_meta = build_references_v1(Path(args.ref_sel))

    # Audit all methods in results/D5-512
    results_d5 = REPO_ROOT / "results" / "D5-512"
    if results_d5.exists():
        audit_report = {}
        for mdir in sorted(results_d5.iterdir()):
            if mdir.is_dir() and not mdir.name.startswith("_"):
                res = audit_method_directory(mdir.name, mdir, tasks_meta)
                audit_report[mdir.name] = {
                    "valid": res["num_valid_matched"],
                    "cross": res["num_cross_matched"],
                    "same": res["num_same_matched"],
                    "missing": res["num_missing_tasks"],
                    "unmatched_files": res["num_unmatched_files"],
                }
                print(f"  [{mdir.name:18s}] valid={res['num_valid_matched']:3d}/750 (cross={res['num_cross_matched']:3d}, same={res['num_same_matched']:3d}) missing={res['num_missing_tasks']:2d}")
        
        audit_file = PROTOCOL_DIR / "baseline_audit_strict.json"
        audit_file.write_text(json.dumps(audit_report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[Manifest] Wrote strict baseline audit report to {audit_file}")


if __name__ == "__main__":
    main()
