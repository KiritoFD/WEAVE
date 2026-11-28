"""S0 CPU audit and explicit, generation-free LPIPS repair.

The default command reads files only: it does not import torch, inspect CUDA,
load a model, contact a network, or change an archived result. --out writes a
new CPU audit; --repair-lpips is the separate model-loading operation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
LIMITS = {"dino_c": (-.005, "lower"), "dino_s": (-.005, "lower"),
          "csd_top1": (-2., "lower"), "same150": (-2., "lower"),
          "lpips": (.005, "upper")}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def strict_ci_gate(comparison: dict) -> dict:
    """Re-evaluate archived CI, never infer a pass from a passing point mean."""
    checks, values = {}, {}
    for name, (threshold, side) in LIMITS.items():
        block, metric = ("same150", "csd_top1_pp") if name == "same150" else (
            "cross600", "csd_top1_pp" if name == "csd_top1" else name)
        val = comparison.get(block, {}).get(metric)
        if not val:
            checks[name], values[name] = "missing", None
            continue
        lo, hi = float(val["ci95_lo"]), float(val["ci95_hi"])
        if not np.isfinite([lo, hi]).all() or lo > hi:
            raise ValueError(f"Invalid CI for {name}: {lo}, {hi}")
        passed = lo >= threshold if side == "lower" else hi <= threshold
        rejected = hi < threshold if side == "lower" else lo > threshold
        checks[name] = "pass" if passed else ("reject" if rejected else "inconclusive")
        values[name] = {"delta": val.get("delta"), "ci95": [lo, hi],
                        "threshold": threshold, "bound": side}
    statuses = set(checks.values())
    status = "pass" if statuses == {"pass"} else (
        "reject" if "reject" in statuses else "inconclusive")
    return {"status": status, "non_inferiority_pass": status == "pass", "checks": checks,
            "values": values, "n_sources": comparison.get("n_sources"),
            "n_boot": comparison.get("n_boot"),
            "recorded_pass": comparison.get("non_inferiority_pass"),
            "differs_from_archive": (status == "pass") != comparison.get("non_inferiority_pass")}


def audit_fast5(root: Path) -> dict:
    path = root / "docs/performance/data/3060_profile_summary.json"
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    saved = read_json(path)["test750_quality_gate"]["paired_bootstrap"]
    profiles = {profile: {arm: strict_ci_gate(row) for arm, row in arms.items()}
                for profile, arms in saved.items()}
    return {"status": "rechecked_archived_source_cluster_ci", "source": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "profiles": profiles,
            "note": "Current eval_gate source uses CI bounds. Old archive may have used Same150 point means; "
                    "archive is preserved. No new bootstrap is claimed without original per-task rows."}


def repair_inventory(root: Path) -> dict:
    f1 = root / "runs/paper_audit/f1_artist_holdout"
    f6 = root / "runs/paper_audit/f6_p2a"
    spec = f1 / "artist_holdout_750_spec.pt"
    raw = sorted((f1 / "raw").glob("*/meta.json"))
    cache = f6 / "lpips_cache"
    sources = cache / "sources.pt"
    outputs = sorted(cache.glob("part_*.pt"))
    return {"f1": {"spec": str(spec), "spec_exists": spec.exists(),
                    "raw_meta": [str(p) for p in raw],
                    "ready": spec.exists() and bool(raw),
                    "missing": ([] if spec.exists() else [str(spec)]) + ([] if raw else [str(f1 / "raw/*/meta.json")])},
            "f6": {"sources": str(sources), "source_exists": sources.exists(),
                    "cached_outputs": [str(p) for p in outputs],
                    "ready": sources.exists() and bool(outputs),
                    "missing": ([] if sources.exists() else [str(sources)]) +
                               ([] if outputs else [str(cache / "part_*.pt")]),
                    "note": "Historical F6 stored only a few previews. It cannot be fully repaired from a summary "
                            "or six PNGs. New F6 runs cache actual outputs; --repair-spec can point to other existing RGBs."},
            "invalidated": {"metrics": ["F1 LPIPS", "F1 LPIPS CI", "F6 LPIPS", "F6 LPIPS CI"],
                            "reason": "uint8 source reference bypassed required /255 normalization",
                            "unaffected_by_this_bug": ["DINO", "CSD", "CLIP", "V37 formal CI", "3060 gates"],
                            "old_artifacts_modified": False}}


def audit_sources(root: Path, config: dict | None = None) -> dict:
    required = ["runs/v36_art_foundation/cache/v36_caches_ready.json",
                "runs/v36_art_foundation/cache/curation_manifest.json",
                "runs/v36_art_foundation/checkpoints/v36_main_p1_final.pt",
                "runs/v36_art_foundation/checkpoints/v36_main_p2_final.pt",
                "runs/v36_art_foundation/checkpoints/v36_ecore_latest.pt",
                "runs/v37_editing/matched_seen/summary.json",
                "runs/v36_art_foundation/measured/v36_main_p2_I2_tau_sweep.json",
                "docs/performance/protocol_comparison/comparison.json",
                "eval_protocol/tasks_750.json"]
    paths = (config or {}).get("paths", {})
    inventory = {p: {"exists": resolve(root, p).exists()} for p in required}
    configured = {name: {"path": str(value), "exists": resolve(root, value).exists()}
                  for name, value in paths.items() if isinstance(value, str)}
    return {"kind": "cpu_read_only_s0", "source_root": str(root), "assets": inventory,
            "configured_paths": configured, "fast5": audit_fast5(root),
            "lpips_repair": repair_inventory(root),
            "state": "prepared_only_no_gpu_measurements",
            "baseline_selection": "Seen-Dev evidence is required; never choose P2 solely by step count"}


def source_cluster_ci(values: list[float], sources: list[str], *, n_boot=2000, seed=20261128) -> dict:
    values, sources = np.asarray(values, dtype=float), np.asarray(sources, dtype=str)
    if not len(values) or values.shape != sources.shape or not np.isfinite(values).all():
        raise ValueError("Finite values and corresponding source IDs required")
    names, inv = np.unique(sources, return_inverse=True)
    sums = np.bincount(inv, weights=values, minlength=len(names))
    counts = np.bincount(inv, minlength=len(names))
    draws = np.random.default_rng(seed).integers(0, len(names), (n_boot, len(names)))
    bootstrap = sums[draws].sum(1) / counts[draws].sum(1)
    lo, hi = np.quantile(bootstrap, [.025, .975])
    return {"mean": float(values.mean()), "ci95": [float(lo), float(hi)],
            "n": len(values), "n_sources": len(names), "n_boot": n_boot, "seed": seed}


def validate_training_assets(root: Path, config: dict) -> dict:
    """Explicit CPU checkpoint/header check before allocating any training models."""
    cache = resolve(root, config["paths"]["cache_root"])
    needed = [resolve(root, config["paths"]["e_initial"]),
              *[resolve(root, p) for p in config["paths"]["p_candidates"]],
              *[cache / p for p in ("seen_train_index.pt", "v36_seen_train_priors.pt",
                                     "v36_caches_ready.json", "curation_manifest.json", "v36_eval_assets.pt")],
              root.parent / "eval_cache/torch/hub/checkpoints/vgg16-397923af.pth",
              root.parent / "eval_cache/torch/hub/checkpoints/convnext_tiny-983f1562.pth",
              root / "deliver/checkpoints/runs/vae_decoder_prune/s10_p5_lowlr/decoder.pt"]
    missing = [str(p) for p in needed if not p.is_file()]
    if missing:
        raise FileNotFoundError("Required training assets missing before GPU construction: " + ", ".join(missing))
    import torch
    e = torch.load(needed[0], map_location="cpu", weights_only=False, mmap=True)
    if int(e.get("step", -1)) != 4500 or "ecore" not in e:
        raise ValueError("E initialization must be step4500 online ecore, not EMA-only")
    del e
    for path in needed[1:3]:
        ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        if any(k not in ck for k in ("pcore_ema", "style_enc_ema")):
            raise ValueError(f"Missing V36 P/Style EMA in {path}")
        del ck
    meta = torch.load(cache / "seen_train_index.pt", map_location="cpu", weights_only=False, mmap=True)
    memmaps = [resolve(root, v) for k, v in meta.items() if k.endswith("_rel") and isinstance(v, str)]
    memmaps.extend(cache / "memmap" / name for name in (
        "seen_dev_z_gt.npy", "seen_dev_q32.npy", "seen_dev_y16.npy",
        "artist_ood_z_gt.npy", "artist_ood_q32.npy", "artist_ood_y16.npy",
        "movement_ood_z_gt.npy", "movement_ood_q32.npy", "movement_ood_y16.npy"))
    absent = [str(p) for p in memmaps if not p.is_file()]
    if absent:
        raise FileNotFoundError("Training memmaps missing: " + ", ".join(absent))
    ready = read_json(cache / "v36_caches_ready.json")
    for key in ("dev_info", "art_info", "mv_info"):
        for path in ready[key]["paths"].values():
            if isinstance(path, str) and not resolve(root, path).is_file():
                raise FileNotFoundError(f"{key} cached input missing: {path}")
    return {"status": "cpu_checkpoint_keys_and_cache_paths_verified", "e_online_step": 4500,
            "gpu_touched": False, "train_memmap_paths": len(memmaps),
            "not_a_gpu_forward_or_quality_test": True}


def _offline_lpips(device: str):
    """Load only local AlexNet/LPIPS weights; missing assets fail before construction."""
    import torch
    import lpips
    alex = Path(torch.hub.get_dir()) / "checkpoints/alexnet-owt-7be5be79.pth"
    weights = Path(lpips.__file__).parent / "weights/v0.1/alex.pth"
    missing = [str(p) for p in (alex, weights) if not p.is_file()]
    if missing:
        raise FileNotFoundError("LPIPS repair requires existing offline weights: " + ", ".join(missing))
    net = lpips.LPIPS(net="alex", version="0.1", model_path=str(weights), verbose=False)
    return net.to(device).eval()


def score_lpips_u8(net, generated_u8, reference_u8, *, device="cpu", batch=8):
    import torch
    if generated_u8.dtype != torch.uint8 or reference_u8.dtype != torch.uint8:
        raise ValueError("Repair inputs must be exact saved uint8 RGB, not unspecified floating ranges")
    if generated_u8.shape != reference_u8.shape or generated_u8.ndim != 4 or generated_u8.shape[1] != 3:
        raise ValueError("LPIPS inputs must be aligned Bx3xHxW")
    out = []
    with torch.inference_mode():
        for start in range(0, len(generated_u8), batch):
            g = generated_u8[start:start + batch].float().div(255).to(device)
            r = reference_u8[start:start + batch].float().div(255).to(device)
            out.extend(net(g * 2 - 1, r * 2 - 1).flatten().cpu().tolist())
    if not np.isfinite(out).all():
        raise ValueError("Nonfinite repaired LPIPS")
    return out


def _image_u8(path: Path):
    import torch
    from PIL import Image
    with Image.open(path) as im:
        # Match original F1/F6 RGB preprocessing, never synthesize a missing output.
        return torch.from_numpy(np.array(im.convert("RGB").resize((512, 512), Image.Resampling.BICUBIC))).permute(2, 0, 1)


def _manifest_jobs(root: Path, spec_path: Path):
    """Explicit existing-RGB fallback. spec.jobs[].tasks carry generated/reference paths."""
    import torch
    spec = read_json(spec_path)
    for job in spec["jobs"]:
        tasks = job["tasks"]
        generated = torch.stack([_image_u8(resolve(root, t["generated"])) for t in tasks])
        references = torch.stack([_image_u8(resolve(root, t["reference"])) for t in tasks])
        yield job["name"], tasks, generated, references, job.get("expected_tasks", len(tasks)), str(spec_path)


def _f1_jobs(root: Path):
    import torch
    base = root / "runs/paper_audit/f1_artist_holdout"
    spec = torch.load(base / "artist_holdout_750_spec.pt", map_location="cpu", weights_only=False)
    by_id = {t["task_id"]: t for t in spec["tasks"]}
    for meta_path in sorted((base / "raw").glob("*/meta.json")):
        meta = read_json(meta_path)
        images, ids = [], []
        for entry in meta["batches"]:
            payload = torch.load(meta_path.parent / entry["file"], map_location="cpu", weights_only=False)
            images.append(payload["rgb_u8"])
            ids.extend(payload["ids"])
        tasks = [{"task_id": i, "source_id": by_id[i]["source_stem"],
                  "target": by_id[i]["target_style"]} for i in ids]
        references = torch.stack([spec["src_u8"][by_id[i]["source_pos"]] for i in ids])
        yield meta["arm"], tasks, torch.cat(images), references, len(spec["tasks"]), str(meta_path)


def _f6_jobs(root: Path):
    import torch
    cache = root / "runs/paper_audit/f6_p2a/lpips_cache"
    source = torch.load(cache / "sources.pt", map_location="cpu", weights_only=False)
    for path in sorted(cache.glob("part_*.pt")):
        saved = torch.load(path, map_location="cpu", weights_only=False)
        tasks = saved["tasks"]
        references = torch.stack([source["src_u8"][t["source_pos"]] for t in tasks])
        yield path.stem, tasks, saved["rgb_u8"], references, saved["expected_tasks"], str(path)


def repair_lpips(root: Path, out: Path, *, target="all", spec_path=None, device="cpu", batch=8,
                 n_boot=2000, identity_tol=1e-6) -> dict:
    """Never generate. New rows/CI supplement the old invalid LPIPS; no old evidence is overwritten."""
    out = out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Choose an empty repair output directory; do not overwrite prior evidence")
    inv = repair_inventory(root)
    names = ("f1", "f6") if target == "all" else (target,)
    missing = {name: inv[name]["missing"] for name in names if not inv[name]["ready"]}
    if missing and spec_path is None:
        raise FileNotFoundError("Existing RGB/spec assets required, generation is disabled: " + json.dumps(missing))
    if spec_path is not None:
        jobs = _manifest_jobs(root, spec_path)
    else:
        import itertools
        jobs = itertools.chain.from_iterable(_f1_jobs(root) if name == "f1" else _f6_jobs(root) for name in names)
    net, summaries, all_rows = None, {}, {}
    for name, tasks, generated, references, expected, source in jobs:
        if len(tasks) != len(generated) or len(tasks) != len(references):
            raise ValueError(f"Unaligned tasks/images: {name}")
        ids = [str(t["task_id"]) for t in tasks]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate task IDs: {name}")
        if Path(name).name != name or not name:
            raise ValueError("Repair job name must be a plain filename component")
        if net is None:
            net = _offline_lpips(device)
        identity = score_lpips_u8(net, references[:min(batch, len(tasks))],
                                  references[:min(batch, len(tasks))], device=device, batch=batch)
        if not identity or max(abs(v) for v in identity) > identity_tol:
            raise RuntimeError(f"Identity LPIPS failed for {name}: {identity}")
        values = score_lpips_u8(net, generated, references, device=device, batch=batch)
        rows = [{**task, "lpips": value} for task, value in zip(tasks, values)]
        all_rows[name] = rows
        sources = [str(t["source_id"]) for t in tasks]
        summaries[name] = {"lpips": source_cluster_ci(values, sources, n_boot=n_boot),
                           "n_valid": len(tasks), "n_expected": expected, "complete": len(tasks) == expected,
                           "identity_max_abs": max(abs(v) for v in identity), "source": source}
        atomic_json(out / name / "lpips_rows.json", {"rows": rows, "summary": summaries[name],
                    "protocol": "saved uint8 RGB /255 to [-1,1]; LPIPS AlexNet v0.1"})
    if not summaries:
        raise FileNotFoundError("No actual saved output batches found; no LPIPS was computed")
    paired = {}
    # Restore relevant source-cluster pairwise LPIPS only; other metric CI remain unchanged.
    for a, b in (("F1_CLEAVE_PE_s085", "F1_CLEAVE_E_l075"),
                 ("F1_CLEAVE_PE_s085", "F1_CLEAVE_One_t50"),
                 ("F1_CLEAVE_PE_s100", "F1_CLEAVE_PE_s100_wrong"),
                 ("F1_CLEAVE_One_t50", "F1_CLEAVE_One_t50_wrong")):
        if a in all_rows and b in all_rows:
            right = {r["task_id"]: r for r in all_rows[b]}
            left = all_rows[a]
            if set(right) != {r["task_id"] for r in left}:
                raise ValueError(f"Paired CI requires identical task coverage: {a}, {b}")
            paired[a + "_minus_" + b] = source_cluster_ci(
                [r["lpips"] - right[r["task_id"]]["lpips"] for r in left],
                [str(r["source_id"]) for r in left], n_boot=n_boot)
    report = {"schema": "lpips_only_repair_v1", "summaries": summaries, "paired_lpips_ci": paired,
              "repaired_metrics": ["LPIPS", "LPIPS source-cluster CI"], "old_artifacts_modified": False,
              "invalidated_old_fields": inv["invalidated"],
              "non_lpips_metrics": "Preserved in original scoring artifacts; no re-extraction or generation performed"}
    atomic_json(out / "summary.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--out", type=Path, help="New CPU-audit JSON, or empty output directory for repair")
    parser.add_argument("--repair-lpips", action="store_true", help="Explicit model loading; never generates images")
    parser.add_argument("--validate-training-assets", action="store_true", help="CPU checkpoint keys/cache paths before GPU construction")
    parser.add_argument("--repair-target", choices=("f1", "f6", "all"), default="all")
    parser.add_argument("--repair-spec", type=Path, help="Existing-RGB jobs JSON; jobs[].tasks[] specify paths and IDs")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.repair_lpips:
        if not args.out:
            parser.error("--repair-lpips requires a new --out directory")
        report = repair_lpips(root, resolve(root, args.out), target=args.repair_target,
            spec_path=resolve(root, args.repair_spec) if args.repair_spec else None,
            device=args.device, batch=args.batch, n_boot=args.n_boot)
    else:
        report = audit_sources(root, read_json(resolve(root, args.config)) if args.config else None)
        if args.validate_training_assets:
            if not args.config:
                parser.error("--validate-training-assets needs --config")
            report["training_assets"] = validate_training_assets(root, read_json(resolve(root, args.config)))
        if args.out:
            atomic_json(resolve(root, args.out), report)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
