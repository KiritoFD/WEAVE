"""Prepare anonymous submission package under aaai_submission/final/.

This script only copies files; it never deletes anything from the source tree.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "aaai_submission" / "final"


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_glob(src_pattern: str, dst_dir: Path) -> None:
    for src in sorted(ROOT.glob(src_pattern)):
        dst = dst_dir / src.name
        copy_file(src, dst)


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
        copy_file(ROOT / name, OUT / name)

    # ---- Configs -----------------------------------------------------------
    configs_dir = OUT / "configs"
    for name in ["config.json", "inference.json", "inference_endpoint_only.json"]:
        copy_file(ROOT / name, configs_dir / name)

    # ---- Utils -------------------------------------------------------------
    utils_dir = OUT / "utils"
    copy_glob("utils/*.py", utils_dir)

    # ---- Tests -------------------------------------------------------------
    tests_dir = OUT / "tests"
    for name in ["test_72_cleanup_smoke.py", "_smoke_t26.py"]:
        copy_file(ROOT / "tests" / name, tests_dir / name)

    # ---- Data preparation scripts and manifests ----------------------------
    data_dir = OUT / "data"
    copy_file(ROOT / "data" / "README.md", data_dir / "README.md")
    if (ROOT / "data" / "train" / "manifest.json").exists():
        copy_file(ROOT / "data" / "train" / "manifest.json", data_dir / "train_manifest.json")
    if (ROOT / "tools" / "encode_wikiart27_sd15.py").exists():
        copy_file(ROOT / "tools" / "encode_wikiart27_sd15.py", OUT / "tools" / "encode_wikiart27_sd15.py")

    # ---- Checkpoint --------------------------------------------------------
    ckpt_src = ROOT / "runs" / "submission" / "canonical_oriented_epoch4" / "epoch_0004.pt"
    ckpt_dst = OUT / "checkpoints" / "weave_oriented_hf_epoch_0004.pt"
    if ckpt_src.exists():
        copy_file(ckpt_src, ckpt_dst)
    else:
        print(f"WARNING: checkpoint not found at {ckpt_src}")

    # ---- Oriented-HF training config --------------------------------------
    for cfg_name in [
        "experiments/architecture/hf_oriented_nohh.json",
        "experiments/architecture/hf_oriented_internal_probe.json",
        "experiments/architecture/hf_oriented_internal_early_stop.json",
    ]:
        src = ROOT / cfg_name
        if src.exists():
            copy_file(src, OUT / "configs" / Path(cfg_name).name)

    # ---- Evaluation outputs (750 D5 images + DINO) -------------------------
    eval_src = ROOT / "runs" / "submission" / "canonical_oriented_epoch4" / "paper_eval" / "epoch_0004"
    eval_dst = OUT / "eval_outputs" / "d5_750"
    if (eval_src / "summary.json").exists():
        copy_file(eval_src / "summary.json", eval_dst / "summary.json")
    if (eval_src / "dino_summary.json").exists():
        copy_file(eval_src / "dino_summary.json", eval_dst / "dino_summary.json")
    if (eval_src / "summary_grid.png").exists():
        copy_file(eval_src / "summary_grid.png", eval_dst / "summary_grid.png")
    # DINO epoch metrics
    dino_metrics = ROOT / "runs" / "submission" / "canonical_oriented_epoch4" / "paper_eval_epoch_metrics.json"
    if dino_metrics.exists():
        copy_file(dino_metrics, eval_dst / "epoch_metrics.json")
    images_src = eval_src / "images"
    if images_src.exists():
        for img in sorted(images_src.glob("*.png")):
            copy_file(img, eval_dst / "images" / img.name)

    # ---- Documentation -----------------------------------------------------
    write_docs(OUT)

    # ---- Paper + supplement PDFs ------------------------------------------
    paper_pdf = ROOT / "aaai2027_v4" / "paper.pdf"
    if paper_pdf.exists():
        copy_file(paper_pdf, OUT / "paper.pdf")
    supp_pdf = ROOT / "aaai2027_v4" / "supplement.pdf"
    if supp_pdf.exists():
        copy_file(supp_pdf, OUT / "supplement.pdf")
    # ---- Supplement source + figures --------------------------------------
    supp_fig_dir = OUT / "supplement_figures"
    for fig in sorted((ROOT / "aaai2027_v4" / "supplement_figures").glob("*")):
        if fig.is_file():
            copy_file(fig, supp_fig_dir / fig.name)
    for name in ["supplement.tex", "make_supplement_figures.py"]:
        copy_file(ROOT / "aaai2027_v4" / name, OUT / name)
    # ---- Paper source + figure data + scripts -----------------------------
    copy_file(ROOT / "aaai2027_v4" / "paper.tex", OUT / "paper.tex")
    fig_data_dir = OUT / "fig_data"
    for f in sorted((ROOT / "aaai2027_v4" / "fig_data").glob("*")):
        if f.is_file():
            copy_file(f, fig_data_dir / f.name)
    for name in [
        "plot_page1_summary.py", "plot_method_probes.py",
        "make_radar_metric_blocks.py", "gen_figures.ps1",
        "refs.bib", "aaai2027.sty",
    ]:
        src = ROOT / "aaai2027_v4" / name
        if src.exists():
            copy_file(src, OUT / name)

    print("Submission package prepared at:", OUT)
    summarize(OUT)


def write_docs(out: Path) -> None:
    (out / "README.md").write_text(
        """# WEAVE Anonymous Submission

This repository contains the code, configuration, checkpoint, and evaluation outputs for the paper submission.

## Quick start

```bash
pip install -r requirements.txt
python utils/run_evaluation.py \\
    --checkpoint checkpoints/weave_d5_epoch_0010.pt \\
    --output runs/d5_eval \\
    --config_override configs/inference.json
```

See `REPRODUCE.md` for full training and evaluation instructions.

## Contents

- `configs/` — training and inference configurations
- `checkpoints/` — selected D5 checkpoint
- `eval_outputs/d5_750/` — 750 generated images on D5 with summary metrics
- `utils/run_evaluation.py` — evaluation entry point
- `run.py` — training entry point
- `paper.pdf` — submitted manuscript
""",
        encoding="utf-8",
    )

    (out / "REPRODUCE.md").write_text(
        """# Reproduction Instructions

## Environment

```bash
pip install -r requirements.txt
```

Tested on Python 3.10+ and one RTX 3060 (12 GB).

## Data preparation

1. Download the WikiArt dataset.
2. Run the preprocessing script (or follow `data/README.md`) to build the latent cache and manifest.
3. D5 uses the five families whose identity/target CLIP-S gap is largest. Training uses 5,000 images (1,000 per family); testing uses 150 images (30 per family). Each test image is paired with one reference from every family, yielding 750 content-style pairs.

## Training (D5, oriented-HF recipe)

```bash
python run.py --config experiments/architecture/hf_oriented_internal_early_stop.json
```

The oriented-HF config enables `target_latent_hf_subband_fusion_enabled` and trains for 15 epochs with internal-dynamics early stopping; the selected checkpoint is epoch 4.

## Evaluation (D5, 750 pairs)

```bash
python scripts/batch_eval_all.py \\
    --checkpoint_dir checkpoints \\
    --epochs 4 \\
    --config configs/config.json \\
    --config_override configs/inference.json \\
    --output_subdir paper_eval
```

Expected metrics on D5-512:

- all_pairs CLIP-S: 0.7128
- all_pairs LPIPS:  0.2593
- DINO-S:           0.4916
- DINO-C:           0.8103

## Smoke test

```bash
python tests/test_72_cleanup_smoke.py
```
""",
        encoding="utf-8",
    )

    (out / "DATA.md").write_text(
        """# Data

All experiments use the publicly available WikiArt dataset.

## D5 benchmark construction

D5 uses the five WikiArt families whose identity/target-style CLIP-S gap is largest:
Early Renaissance, Impressionism, Minimalism, Rococo, and Ukiyo-e.
Training uses 5,000 images (1,000 per family) and testing uses 150 images (30 per family).
Each test image is paired with one reference from every family, yielding 750 content-style pairs at 512x512 resolution.

## Splits and manifest

- `data/train_manifest.json` records the training split.
- The evaluation pairs are fixed by the protocol in `utils/run_evaluation.py` and reported in `eval_outputs/d5_750/summary.json`.

## Note

We do not redistribute WikiArt images. Use the provided scripts and manifest to reconstruct the exact splits from the public dataset.
""",
        encoding="utf-8",
    )

    (out / "CHECKPOINT.md").write_text(
        """# Checkpoint

- File: `checkpoints/weave_oriented_hf_epoch_0004.pt`
- Source: `runs/submission/canonical_oriented_epoch4/epoch_0004.pt`
- Training config: `configs/hf_oriented_internal_early_stop.json` (base: `hf_oriented_nohh.json`)
- Inference config: `configs/inference.json`
- Epoch: 4 (selected by internal-dynamics early stopping)
- Train time: 82.8 s (1.4 min) on RTX 3060 12GB
- Trainable parameters: 1.04M (+ frozen 84M SD1.5 EMA VAE)

The D5 benchmark evaluates 150 test images (30 from each of five families) against one reference per family, producing 750 content-style pairs.

## D5-512 metrics (using `configs/inference.json`)

- all_pairs CLIP-S: 0.7128
- all_pairs LPIPS:  0.2593
- DINO-S:           0.4916
- DINO-C:           0.8103
- style_transfer_ability CLIP-S: 0.6825
- style_transfer_ability LPIPS:  0.2641

These correspond to the main-table WEAVE entry under D5-512.
""",
        encoding="utf-8",
    )


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
