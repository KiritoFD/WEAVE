"""Build anonymous code+data package under subpack/ (no paper files).

Layout (all relative to package root):
  data/train          D5 latent cache (config data_root)
  data/test           D5 eval images (config test_image_dir)
  data/train_p2a      P2A latents (config data_root for P2A)
  data/test_p2a       P2A eval images
  data/images/d5/train    D5 train JPGs (for re-encode; not duplicated as latents)
  data/images/p2a256/train P2A train JPGs
  data/images/r5/{train,test}
  data/onnx_exported  VAE ONNX graphs
  assets/onnx_vae     same ONNX graphs (timing scripts)
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "subpack"
DATASET_ROOT = ROOT.parent / "Dataset"  # Latent_Style/Dataset

ABS_RE = re.compile(r"(?:[A-Za-z]:[\\/]|/mnt/|/home/)[^\s\"']+")
ARCHIVE_SUFFIX = {".zip", ".7z", ".rar", ".tar", ".gz", ".tgz"}


def cp(src: Path, dst: Path | None = None) -> None:
    if not src.exists():
        return
    if dst is None:
        dst = OUT / src.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)


def cp_glob(pattern: str, dst_dir: Path) -> None:
    for src in sorted(ROOT.glob(pattern)):
        if src.suffix.lower() in ARCHIVE_SUFFIX:
            continue
        cp(src, dst_dir / src.name)


def sanitize_text_files(root: Path) -> int:
    n = 0
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {
            ".json", ".md", ".txt", ".csv", ".py", ".tex",
        }:
            continue
        try:
            raw = path.read_text(encoding="utf-8")
        except Exception:
            continue
        cleaned = raw
        for a, b in [
            ("G:\\GitHub\\Latent_Style\\WEAVE\\", ""),
            ("G:/GitHub/Latent_Style/WEAVE/", ""),
            ("G:\\GitHub\\Latent_Style\\", ""),
            ("G:/GitHub/Latent_Style/", ""),
            ("I:/Github/Latent_Style/", ""),
            ("/mnt/i/Github/Latent_Style/", ""),
        ]:
            cleaned = cleaned.replace(a, b)
        cleaned = ABS_RE.sub(lambda m: Path(m.group(0).replace("\\", "/")).name, cleaned)
        if cleaned != raw:
            path.write_text(cleaned, encoding="utf-8")
            n += 1
    return n


def strip_archives(root: Path) -> int:
    n = 0
    for path in list(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in ARCHIVE_SUFFIX:
            path.unlink(missing_ok=True)
            n += 1
    return n


def fix_config_paths() -> None:
    """Force package-relative paths in shipped configs."""
    cfg_path = OUT / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        cfg.setdefault("data", {})
        cfg["data"]["data_root"] = "data/train"
        cfg["data"]["pairing_cache_path"] = "data/train/.latent_cache/prototype_pairing_top8.pt"
        cfg["data"]["latent_cache_dir"] = ".latent_cache/packed"
        cfg["data"]["latent_cache_mode"] = "packed"
        cfg.setdefault("training", {})
        cfg["training"]["test_image_dir"] = "data/test"
        cfg["training"]["full_eval_cache_dir"] = "runs/cache"
        cfg["training"]["full_eval_clip_hf_cache_dir"] = "runs/cache/hf"
        cfg.setdefault("checkpoint", {})
        cfg["checkpoint"]["save_dir"] = "runs/submission/canonical_oriented_epoch4"
        cfg_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")

    # oriented configs live under configs/; rewrite bases for package layout
    for name in ["hf_oriented_internal_early_stop.json", "hf_oriented_nohh.json", "hf_oriented_internal_probe.json"]:
        p = OUT / "configs" / name
        if not p.exists():
            continue
        obj = json.loads(p.read_text(encoding="utf-8"))
        if "checkpoint" in obj and "save_dir" in obj["checkpoint"]:
            obj["checkpoint"]["save_dir"] = f"runs/submission/{Path(obj['checkpoint']['save_dir']).name}"
        # package layout: configs/*.json inherit package-root config.json
        if name == "hf_oriented_nohh.json":
            obj["_base"] = "../config.json"
        elif name == "hf_oriented_internal_probe.json":
            obj["_base"] = "hf_oriented_nohh.json"
        elif name == "hf_oriented_internal_early_stop.json":
            obj["_base"] = "hf_oriented_internal_probe.json"
        p.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")

    # p2a helper config if present in repo root
    for name in ["config_p2a256_adain2.json", "config_r5.json"]:
        src = ROOT / name
        if not src.exists():
            continue
        obj = json.loads(src.read_text(encoding="utf-8"))
        if "data" in obj:
            if name.startswith("config_p2a"):
                obj["data"]["data_root"] = "data/train_p2a"
                obj["data"]["test_image_dir"] = "data/test_p2a"
            if name.startswith("config_r5"):
                obj["data"]["data_root"] = "data/images/r5/train"
                obj["data"]["test_image_dir"] = "data/images/r5/test"
        if "checkpoint" in obj and "save_dir" in obj["checkpoint"]:
            sd = str(obj["checkpoint"]["save_dir"])
            if ":" in sd or sd.startswith("/"):
                obj["checkpoint"]["save_dir"] = "runs/submission/" + Path(sd).name
        cp_content = json.dumps(obj, indent=2) + "\n"
        (OUT / name).write_text(cp_content, encoding="utf-8")


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()

    # === Core code only (no paper) ===
    for f in [
        "model.py", "flow.py", "run.py", "trainer.py", "blocks.py",
        "style.py", "wavelet.py", "config_schema.py", "internal_dynamics.py",
        "style_families.py", "__init__.py", "__main__.py",
    ]:
        cp(ROOT / f)

    cp(ROOT / "config.json")
    cp(ROOT / "inference.json")
    for name in ["hf_oriented_internal_early_stop.json", "hf_oriented_nohh.json", "hf_oriented_internal_probe.json"]:
        cp(ROOT / "experiments/architecture" / name, OUT / "configs" / name)

    cp_glob("utils/*.py", OUT / "utils")
    cp_glob("tests/*.py", OUT / "tests")
    # only essential tools (not entire tools dump with host paths)
    for name in [
        "prepare_submission_data.py",
        "encode_wikiart27_sd15.py",
    ]:
        cp(ROOT / "tools" / name, OUT / "tools" / name)
    speed_dir = ROOT / "experiments/inference_speed"
    if speed_dir.is_dir():
        for name in [
            "export_decoder.py", "run_pipeline_750.py", "run_end_to_end_750.py",
            "bench_inference_speed.py", "vae_encode_optimized.py",
            "vae_decode_optimized.py", "optimized_inference.py",
        ]:
            cp(speed_dir / name, OUT / "tools/inference_speed" / name)

    cp(ROOT / "requirements.txt")
    if (ROOT / "pytest.ini").exists():
        cp(ROOT / "pytest.ini")

    # === Checkpoint + eval outputs ===
    ckpt = ROOT / "runs/submission/canonical_oriented_epoch4/epoch_0004.pt"
    cp(ckpt, OUT / "checkpoints/weave_epoch_0004.pt")
    paper_eval = ROOT / "runs/submission/canonical_oriented_epoch4/paper_eval/epoch_0004"
    if paper_eval.is_dir():
        for f in ["summary.json", "dino_summary.json", "summary_grid.png"]:
            cp(paper_eval / f, OUT / "eval_outputs/d5_750" / f)
        imgs = paper_eval / "images"
        if imgs.is_dir():
            for img in sorted(imgs.glob("*.png")):
                cp(img, OUT / "eval_outputs/d5_750/images" / img.name)
    for f in ["paper_eval_epoch_metrics.json", "paper_eval_epoch_metrics.csv"]:
        cp(ROOT / "runs/submission/canonical_oriented_epoch4" / f, OUT / "eval_outputs/d5_750" / f)

    # === Unified data/ layout (no top-level datasets/, no duplicates) ===
    data = OUT / "data"
    data.mkdir(parents=True, exist_ok=True)

    # D5 latents (training default)
    cp(ROOT / "data/train", data / "train")

    # D5 eval images once
    d5_test = ROOT / "datasets/d5/test"
    if not d5_test.is_dir():
        d5_test = DATASET_ROOT / "distinct5_512/test"
    cp(d5_test, data / "test")

    # D5 train JPGs only under images/ (not also under train/)
    d5_train_img = DATASET_ROOT / "distinct5_512/train"
    if d5_train_img.is_dir():
        cp(d5_train_img, data / "images/d5/train")

    # P2A latents -> data/train_p2a (config)
    p2a_lat = DATASET_ROOT / "latent-256"
    if p2a_lat.is_dir():
        cp(p2a_lat, data / "train_p2a")
    # P2A eval images
    p2a_test = DATASET_ROOT / "legacy256_overfit50/test"
    if p2a_test.is_dir():
        cp(p2a_test, data / "test_p2a")
    # P2A train JPGs (separate from latents)
    p2a_train_img = DATASET_ROOT / "legacy256_overfit50/train"
    if p2a_train_img.is_dir():
        cp(p2a_train_img, data / "images/p2a256/train")

    # R5 boards (images only)
    r5 = ROOT / "datasets/r5"
    if r5.is_dir():
        for split in ["train", "test"]:
            if (r5 / split).is_dir():
                cp(r5 / split, data / "images/r5" / split)

    # optional small 256 board
    b256 = ROOT / "datasets/256/test"
    if b256.is_dir():
        cp(b256, data / "images/256/test")

    # ONNX (full set)
    onnx_src = ROOT / "data/onnx_exported"
    for name in ["encoder_b8.onnx", "decoder_b8.onnx", "encoder_b16.onnx", "decoder_b16.onnx"]:
        cp(onnx_src / name, data / "onnx_exported" / name)
        cp(onnx_src / name, OUT / "assets/onnx_vae" / name)

    # timing docs (no paper)
    cp(ROOT / "docs/model_probe/generation_only_timing_summary.json", OUT / "docs/generation_only_timing_summary.json")
    cp(ROOT / "docs/exp/onnx_inference/README.md", OUT / "docs/onnx_inference_README.md")
    cp(ROOT / "tools/prepare_submission_data.py", OUT / "tools/prepare_submission_data.py")

    strip_archives(OUT)
    fix_config_paths()
    sanitize_text_files(OUT)

    (data / "README.md").write_text(
        """# Data layout (relative paths only)

Training defaults (`config.json`) use:

| Config field | Path | Content |
|---|---|---|
| `data.data_root` | `data/train` | D5 latents + `.latent_cache/packed` |
| `training.test_image_dir` | `data/test` | D5 eval JPGs (150) |
| P2A `data_root` | `data/train_p2a` | P2A latents |
| P2A test | `data/test_p2a` | P2A eval JPGs |

Raw images for re-encoding / R5 (not used by default D5 train loop):

```
data/images/d5/train/       # D5 train JPGs (5k)
data/images/p2a256/train/   # P2A train JPGs
data/images/r5/{train,test}
data/onnx_exported/         # VAE ONNX (b8+b16)
```

There is **no** top-level `datasets/` tree and **no** zip archives.
Latents and JPGs are stored once each (different formats; not duplicates).
""",
        encoding="utf-8",
    )

    OUT.joinpath("README.md").write_text(
        """# WEAVE anonymous code package

Code, checkpoint, ONNX VAE graphs, and training/eval data.
**No paper TeX/PDF** (submit those separately).

## Quick start (relative paths, run from this directory)

```bash
pip install -r requirements.txt
# D5 train (uses data/train latents)
python run.py --config configs/hf_oriented_internal_early_stop.json
# or default config.json:
# python run.py --config config.json

# D5 eval
python -m utils.run_evaluation checkpoints/weave_epoch_0004.pt data/test eval_outputs/rerun \\
  --config_override inference.json --num_steps 8
```

## Contents

- Core: `model.py`, `run.py`, `trainer.py`, `flow.py`, `wavelet.py`, ...
- Configs: `config.json`, `inference.json`, `configs/hf_oriented_*.json`
- Checkpoint: `checkpoints/weave_epoch_0004.pt`
- Data: see `data/README.md`
- ONNX: `data/onnx_exported/` and `assets/onnx_vae/`
- Timing ledger: `docs/generation_only_timing_summary.json` (73.0 s / 750 pairs)

All storage paths in configs are package-relative. Launch with cwd = package root.
""",
        encoding="utf-8",
    )

    # summary
    total_files = 0
    total_bytes = 0
    dirs: dict[str, dict] = {}
    for p in OUT.rglob("*"):
        if not p.is_file():
            continue
        total_files += 1
        total_bytes += p.stat().st_size
        d = str(p.relative_to(OUT).parent).replace("\\", "/")
        dirs.setdefault(d, {"files": 0, "bytes": 0})
        dirs[d]["files"] += 1
        dirs[d]["bytes"] += p.stat().st_size
    print("-" * 70)
    for d, info in sorted(dirs.items()):
        if d == ".":
            continue
        # only print shallow dirs
        if d.count("/") <= 2:
            line = f"{d + '/':<50} {info['files']:>5} files, {info['bytes']/1024/1024:7.1f} MB"
            print(line.encode("ascii", "replace").decode("ascii"))
    print(f"\nTotal: {total_files} files, {total_bytes/1024/1024:.1f} MB")
    print("Archives stripped; paper files excluded; data/ unified.")


if __name__ == "__main__":
    main()
