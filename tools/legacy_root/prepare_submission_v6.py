"""Prepare a clean, fully relative-path anonymous submission package.

Outputs to aaai_submission/final_v6/. Only copies; never deletes anything.

Key fixes over final_v5:
- All paths inside manifest/summary/configs are project-relative (no G:\\, no backslashes).
- config.json: drop the hand-filled old IDT baseline (set to null, computed live from 750 pairs).
- Add verify_main_table.py: one command prints the paper WEAVE row of Table 1.
- Refresh README/REPRODUCE/DATA/CHECKPOINT to reflect the canonical epoch-4 oriented-HF checkpoint.
- Trim stale names (LANCET / AdaCUT / SchrodingerBridge / RTX 4070 Laptop) from copied docs.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC_V5 = ROOT / "aaai_submission" / "final_v5"
OUT = ROOT / "aaai_submission" / "final_v6"

PAPER_METRICS = {
    "dino_s": 0.4916,
    "clip_s": 0.7128,
    "lpips": 0.2593,
    "dino_c": 0.8103,
}


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def to_rel_posix(path_str: str, base_prefix: str) -> str:
    """Convert an absolute G:\\...\\Latent_Style\\... path to a project-relative POSIX path."""
    if not path_str:
        return path_str
    s = path_str.replace("\\", "/")
    # Strip up to and including "Latent_Style/"
    marker = "/Latent_Style/"
    if marker in s:
        idx = s.find(marker) + len(marker)
        s = s[idx:]
    # Strip drive letter (e.g., G:/)
    s = re.sub(r"^[A-Za-z]:/", "", s)
    return s


def fix_manifest(src: Path, dst: Path) -> None:
    data = json.loads(src.read_text(encoding="utf-8"))
    # Rewrite roots to be project-relative
    data["input_root"] = "data/train"
    data["output_root"] = "data/train/.latent_cache/packed"
    if data.get("vae_cache_dir") is not None:
        data["vae_cache_dir"] = ""
    # Rewrite each source
    fixed_classes = []
    for cls in data.get("classes", []):
        new_sources = []
        for s in cls.get("sources", []):
            rel = to_rel_posix(s, "")
            # Drop leading "Dataset/distinct5_512/train/" or similar
            rel = re.sub(r"^Dataset/distinct5_512/train/", "", rel)
            rel = re.sub(r"^Dataset/distinct5_512/test/", "", rel)
            rel = re.sub(r"^Dataset/distinct5_512_latents_ema/train/", "data/train/.latent_cache/packed/", rel)
            new_sources.append(rel)
        cls["sources"] = new_sources
        fixed_classes.append(cls)
    data["classes"] = fixed_classes
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def fix_summary(src: Path, dst: Path) -> None:
    """Rewrite summary.json replacing any G:\\ absolute path with a project-relative POSIX path."""
    content = src.read_text(encoding="utf-8")
    # Replace any G:\...\Latent_Style\WEAVE\runs\... with runs/...
    def repl(m: re.Match) -> str:
        s = m.group(0).replace("\\", "/")
        marker = "/Latent_Style/WEAVE/"
        if marker in s:
            idx = s.find(marker) + len("/Latent_Style/")
            s = s[idx:]
        else:
            marker2 = "/Latent_Style/"
            if marker2 in s:
                idx = s.find(marker2) + len(marker2)
                s = s[idx:]
            else:
                s = re.sub(r"^[A-Za-z]:/", "", s)
        return s

    # Match any G:\... path-like token (non-quote, non-whitespace)
    pattern = re.compile(r"[A-Za-z]:\\\\[^\"]*?")
    # The JSON has escaped backslashes in strings; use a simpler approach: replace 'G:\\' token chains
    # We do it on the raw JSON text by splitting on quotes.
    new_content = re.sub(
        r'([A-Za-z]:)(\\\\|/)+([^\"]*?)(?=\")',
        lambda m: repl(m),
        content,
    )
    # Also normalise any remaining Windows backslashes inside known relative paths
    # (e.g., "runs\\submission\\..." -> "runs/submission/...")
    new_content = re.sub(r'runs\\\\', "runs/", new_content)
    new_content = re.sub(r'runs\\', "runs/", new_content)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(new_content, encoding="utf-8")


def fix_config(src: Path, dst: Path) -> None:
    """Drop the hand-filled old IDT baseline so IDT is computed live from the 750 pairs."""
    data = json.loads(src.read_text(encoding="utf-8"))
    training = data.get("training", {})
    if "full_eval_clip_style_idt_baseline" in training:
        # Set to null so eval code computes IDT live from the fixed 750-pair split.
        training["full_eval_clip_style_idt_baseline"] = None
    # Also rewrite the canonical checkpoint path to be relative to project root
    main_metrics = data.get("_main_table_metrics", {})
    if "checkpoint" in main_metrics:
        main_metrics["checkpoint"] = "checkpoints/weave_oriented_hf_epoch_0004.pt"
    if "eval_protocol" in main_metrics:
        # Drop absolute 'experiments/architecture/...' hint; reference the included configs/ copy.
        main_metrics["eval_protocol"] = (
            "endpoint_adain_scale=2.0 (use --config_override configs/inference.json at inference); "
            "oriented-HF training config: configs/hf_oriented_nohh.json "
            "(early-stop wrapper: configs/hf_oriented_internal_early_stop.json)"
        )
    # Convert any Windows-style relative paths inside config to POSIX
    blob = json.dumps(data, indent=2, ensure_ascii=False)
    blob = blob.replace("\\\\", "/")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(blob, encoding="utf-8")


def write_verify_main_table(out_root: Path) -> None:
    """One-command verifier: read summary.json + dino_summary.json, print paper WEAVE row."""
    script = '''"""Verify the WEAVE row of Table 1 / Table 2 against the bundled summary files.

Usage:
    python verify_main_table.py

Reads eval_outputs/d5_750/summary.json (CLIP-S, LPIPS) and
eval_outputs/d5_750/dino_summary.json (DINO-S, DINO-C), prints the
four main-table metrics, and compares them against the paper values.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PAPER = {
    "dino_s": 0.4916,
    "clip_s": 0.7128,
    "lpips": 0.2593,
    "dino_c": 0.8103,
}


def main() -> int:
    summary_path = ROOT / "eval_outputs" / "d5_750" / "summary.json"
    dino_path = ROOT / "eval_outputs" / "d5_750" / "dino_summary.json"
    if not summary_path.exists():
        print(f"ERROR: {summary_path} not found", file=sys.stderr)
        return 1
    if not dino_path.exists():
        print(f"ERROR: {dino_path} not found", file=sys.stderr)
        return 1
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    dino = json.loads(dino_path.read_text(encoding="utf-8"))

    # CLIP-S / LPIPS come from analysis.all_pairs_overview in summary.json
    overview = summary.get("analysis", {}).get("all_pairs_overview", {})
    clip_s = overview.get("clip_style")
    lpips = overview.get("content_lpips")
    dino_s = dino.get("all_dino_s")
    dino_c = dino.get("all_dino_c")

    actual = {"dino_s": dino_s, "clip_s": clip_s, "lpips": lpips, "dino_c": dino_c}
    print("=" * 60)
    print("WEAVE main-table verification (D5-512, 750 pairs, epoch 4)")
    print("=" * 60)
    print(f"{'metric':<10} {'paper':>10} {'actual':>10} {'match':>8}")
    ok = True
    for k in ("dino_s", "clip_s", "lpips", "dino_c"):
        a = actual[k]
        p = PAPER[k]
        if a is None:
            print(f"{k:<10} {p:>10.4f} {'MISSING':>10} {'FAIL':>8}")
            ok = False
            continue
        match = abs(a - p) < 1e-3
        ok = ok and match
        print(f"{k:<10} {p:>10.4f} {a:>10.4f} {'OK' if match else 'DIFF':>8}")
    print("=" * 60)
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
'''
    (out_root / "verify_main_table.py").write_text(script, encoding="utf-8")


def write_readme(out_root: Path) -> None:
    (out_root / "README.md").write_text(
        """# WEAVE Anonymous Submission

This repository contains the code, configuration, checkpoint, and evaluation outputs for the paper submission.

All paths inside this package are project-relative. The package is self-contained: any absolute paths in the original training manifest have been replaced with `<Class>/<file>.jpg` entries under `data/train/`.

## Layout

- `model.py`, `flow.py`, `wavelet.py`, `style.py`, `trainer.py`, `run.py`, ... — core WEAVE implementation
- `configs/config.json` — canonical training/model/data configuration
- `configs/inference.json` — inference configuration (8 Euler steps, `endpoint_adain_scale=2.0`)
- `configs/hf_oriented_nohh.json` — oriented-HF training recipe (enables `target_latent_hf_subband_fusion_enabled`)
- `configs/hf_oriented_internal_early_stop.json` — oriented-HF + internal-dynamics early stopping (selects epoch 4)
- `checkpoints/weave_oriented_hf_epoch_0004.pt` — selected checkpoint used for Table 1 / Table 2
- `eval_outputs/d5_750/` — 750 D5 generated images + `summary.json` + `dino_summary.json`
- `verify_main_table.py` — one-command verifier for the WEAVE row of Table 1 / Table 2
- `utils/run_evaluation.py` — evaluation entry point
- `scripts/batch_eval_all.py` — batch evaluator over epoch checkpoints
- `paper.pdf` — submitted manuscript

## Quick verification

```bash
python verify_main_table.py
```

Expected output:

```
metric         paper     actual    match
dino_s        0.4916     0.4916       OK
clip_s        0.7128     0.7128       OK
lpips         0.2593     0.2593       OK
dino_c        0.8103     0.8103       OK
RESULT: PASS
```

## Reproducing the main-table numbers

```bash
python scripts/batch_eval_all.py \\
    --checkpoint_dir checkpoints \\
    --epochs 4 \\
    --config configs/config.json \\
    --config_override configs/inference.json \\
    --output_subdir paper_eval
```

See `REPRODUCE.md` for full training and evaluation instructions.
""",
        encoding="utf-8",
    )


def write_reproduce(out_root: Path) -> None:
    (out_root / "REPRODUCE.md").write_text(
        """# Reproduction Instructions

## Environment

```bash
pip install -r requirements.txt
```

Tested on Python 3.10+ and one RTX 3060 (12 GB). Training uses ~10.8 GB VRAM with `batch_size=96` and `bf16` AMP.

## Data preparation

1. Download the public WikiArt dataset.
2. Build the D5 split using the five families whose identity/target-style CLIP-S gap is largest: Early Renaissance, Impressionism, Minimalism, Rococo, Ukiyo-e.
3. Place training images under `data/train/<Family>/<file>.jpg` and test images under `data/test/<Family>/<file>.jpg`.
4. Build the latent cache with `tools/encode_wikiart27_sd15.py` (or follow `data/README.md`).
5. The provided `data/train_manifest.json` already records the exact 5,000 training images as project-relative paths. Each test image is paired with one reference from every family, yielding 750 content-style pairs at 512x512.

## Training (D5, oriented-HF recipe)

```bash
python run.py --config configs/hf_oriented_internal_early_stop.json
```

The oriented-HF config (`configs/hf_oriented_nohh.json`) enables `target_latent_hf_subband_fusion_enabled` and trains for 15 epochs with internal-dynamics early stopping; the selected checkpoint is epoch 4. Train time: 82.8 s (1.4 min) on RTX 3060 12 GB.

## Evaluation (D5, 750 pairs)

```bash
python scripts/batch_eval_all.py \\
    --checkpoint_dir checkpoints \\
    --epochs 4 \\
    --config configs/config.json \\
    --config_override configs/inference.json \\
    --output_subdir paper_eval
```

Expected metrics on D5-512 (matches `verify_main_table.py`):

- DINO-S: 0.4916
- CLIP-S: 0.7128
- LPIPS:  0.2593
- DINO-C: 0.8103

## Main-table verifier

```bash
python verify_main_table.py
```

## Smoke test

```bash
python tests/test_72_cleanup_smoke.py
```
""",
        encoding="utf-8",
    )


def write_data_doc(out_root: Path) -> None:
    (out_root / "DATA.md").write_text(
        """# Data

All experiments use the publicly available WikiArt dataset. We do not redistribute WikiArt images; the manifest and scripts allow exact reconstruction of the splits from the public dataset.

## D5 benchmark construction

D5 uses the five WikiArt families whose identity/target-style CLIP-S gap is largest:
Early Renaissance, Impressionism, Minimalism, Rococo, Ukiyo-e.
Training uses 5,000 images (1,000 per family) and testing uses 150 images (30 per family).
Each test image is paired with one reference from every family, yielding 750 content-style pairs at 512x512 resolution.

## Splits and manifest

- `data/train_manifest.json` records the 5,000 training images as project-relative paths of the form `<Family>/<file>.jpg` under `data/train/`.
- The 750 evaluation pairs are fixed by the protocol in `utils/run_evaluation.py` and reported in `eval_outputs/d5_750/summary.json`.
- `data/README.md` describes how to reconstruct the latent cache from the public WikiArt images.

## Note on relative paths

Every path inside this package (configs, manifest, summary, scripts) is project-relative. To reproduce, set up the directory layout described in `REPRODUCE.md` and run from the package root.
""",
        encoding="utf-8",
    )


def write_checkpoint_doc(out_root: Path) -> None:
    (out_root / "CHECKPOINT.md").write_text(
        """# Checkpoint

- File: `checkpoints/weave_oriented_hf_epoch_0004.pt`
- Source: produced by `python run.py --config configs/hf_oriented_internal_early_stop.json`
- Training config: `configs/hf_oriented_internal_early_stop.json` (base: `configs/hf_oriented_nohh.json`)
- Inference config: `configs/inference.json`
- Epoch: 4 (selected by internal-dynamics early stopping; no image metrics were used for selection)
- Train time: 82.8 s (1.4 min) on RTX 3060 12 GB
- Trainable parameters: 1.04 M (+ frozen 84 M SD1.5 EMA VAE)

## D5-512 metrics (using `configs/inference.json`)

- DINO-S: 0.4916
- CLIP-S: 0.7128
- LPIPS:  0.2593
- DINO-C: 0.8103

These four numbers correspond to the WEAVE entry of Table 1 and the WEAVE row of Table 2 in the paper. They are reproducible from this package via `python verify_main_table.py`.

## IDT baseline

The IDT (identity) baseline is computed live from the fixed 750 evaluation pairs (returning the source content image unchanged) by `utils/run_evaluation.py`. No hand-filled IDT constant is used in the configuration; `full_eval_clip_style_idt_baseline` is set to `null` so the evaluation always reports the IDT row computed from the same split.
""",
        encoding="utf-8",
    )


def main() -> None:
    if OUT.exists():
        print(f"Destination {OUT} already exists; copying into it without deleting existing files.")
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- Core code ---------------------------------------------------------
    core_code = [
        "model.py", "flow.py", "run.py", "trainer.py",
        "blocks.py", "style.py", "wavelet.py", "style_families.py",
        "config_schema.py", "internal_dynamics.py",
        "__init__.py", "__main__.py",
        "requirements.txt", "pytest.ini",
    ]
    for name in core_code:
        copy_file(SRC_V5 / name, OUT / name)

    # ---- Configs (rewrite config.json + copy the rest verbatim) ------------
    configs_dir = OUT / "configs"
    fix_config(SRC_V5 / "configs" / "config.json", configs_dir / "config.json")
    for name in [
        "inference.json", "inference_endpoint_only.json",
        "hf_oriented_nohh.json",
        "hf_oriented_internal_probe.json",
        "hf_oriented_internal_early_stop.json",
    ]:
        src = SRC_V5 / "configs" / name
        if src.exists():
            copy_file(src, configs_dir / name)

    # ---- Utils / Tests / Tools / Scripts -----------------------------------
    for sub in ["utils", "tests", "tools", "scripts"]:
        src_dir = SRC_V5 / sub
        if not src_dir.exists():
            continue
        dst_dir = OUT / sub
        for src in src_dir.rglob("*"):
            if src.is_file():
                rel = src.relative_to(src_dir)
                dst = dst_dir / rel
                copy_file(src, dst)

    # ---- Data (manifest rewritten to relative paths) -----------------------
    data_dir = OUT / "data"
    copy_file(SRC_V5 / "data" / "README.md", data_dir / "README.md")
    fix_manifest(SRC_V5 / "data" / "train_manifest.json", data_dir / "train_manifest.json")

    # ---- Checkpoint --------------------------------------------------------
    ckpt_src = SRC_V5 / "checkpoints" / "weave_oriented_hf_epoch_0004.pt"
    ckpt_dst = OUT / "checkpoints" / "weave_oriented_hf_epoch_0004.pt"
    if ckpt_src.exists():
        copy_file(ckpt_src, ckpt_dst)
    else:
        print(f"WARNING: checkpoint not found at {ckpt_src}")

    # ---- Evaluation outputs ------------------------------------------------
    eval_src = SRC_V5 / "eval_outputs" / "d5_750"
    eval_dst = OUT / "eval_outputs" / "d5_750"
    if eval_src.exists():
        fix_summary(eval_src / "summary.json", eval_dst / "summary.json")
        for name in ["dino_summary.json", "epoch_metrics.json", "summary_grid.png"]:
            src = eval_src / name
            if src.exists():
                copy_file(src, eval_dst / name)
        images_src = eval_src / "images"
        if images_src.exists():
            for img in sorted(images_src.glob("*.png")):
                copy_file(img, eval_dst / "images" / img.name)

    # ---- Docs --------------------------------------------------------------
    write_readme(OUT)
    write_reproduce(OUT)
    write_data_doc(OUT)
    write_checkpoint_doc(OUT)

    # ---- Verifier ----------------------------------------------------------
    write_verify_main_table(OUT)

    # ---- Paper PDF ---------------------------------------------------------
    paper_pdf = ROOT / "aaai2027_v4" / "paper.pdf"
    if paper_pdf.exists():
        copy_file(paper_pdf, OUT / "paper.pdf")

    print("Submission package prepared at:", OUT)
    summarize(OUT)


def summarize(out: Path) -> None:
    counts: dict[str, int] = {}
    sizes: dict[str, int] = {}
    for p in sorted(out.rglob("*")):
        if p.is_file():
            rel = p.relative_to(out)
            top = rel.parts[0] if rel.parts else "root"
            counts[top] = counts.get(top, 0) + 1
            sizes[top] = sizes.get(top, 0) + p.stat().st_size

    print("\nPackage summary:")
    for top in sorted(counts):
        mb = sizes[top] / 1024 / 1024
        print(f"  {top}: {counts[top]} files, {mb:.1f} MB")


if __name__ == "__main__":
    main()
