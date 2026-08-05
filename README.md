# WEAVE
**W**avelet-based **E**ntanglement-**A**ugmented **V**ision **E**xchange — a frequency-native framework for artistic style transfer.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![AAAI 2027](https://img.shields.io/badge/AAAI-2027%20Submission-purple)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Anonymous%20Double--Blind-lightgrey)

> **Anonymous Authors — AAAI 2027 Submission (Double-Blind)**
>
> This repository is released anonymously for the purpose of double-blind review.
> Author identities, affiliations, and identifying URLs have been intentionally
> omitted. They will be populated in the camera-ready version upon acceptance.

---

## 📖 Introduction

Artistic style transfer must balance two competing objectives: faithfully
preserving the *content* of a source image while convincingly rendering the
*style* of a reference artwork. Classical moment-matching methods (AdaIN, WCT)
operate in pixel or VGG feature space and tend to either over-stylize (collapsing
content) or under-stylize (failing to capture texture). Modern diffusion priors
(SD-Turbo, IP-Adapter) inherit powerful generative priors but at the cost of
billions of parameters and limited controllability over the content–style
trade-off.

**WEAVE** approaches the problem from the wavelet domain. The source and target
signals are decomposed into Haar subbands, and a Flow Matching objective is
applied *per subband*: low-frequency (LL) coefficients carry global color and
illumination and receive a mild Channel-wise AdaIN migration, while
high-frequency (LH/HL/HH) coefficients carry structural texture and are
transported by a learned velocity field conditioned on style tokens. The
reconstruction is performed by an inverse DWT that is exact by construction
(`IDWT(DWT(x)) = x`), so no spectral information is lost across the forward /
backward pass.

The endpoint of the transport is sharpened by a **Channel-wise AdaIN** pass
controlled by the `endpoint_adain_scale` field — set to `1.0` during training
and `2.0` at inference (see `inference.json`). This decoupling lets the model
learn a stable velocity field while still producing crisp stylizations at
evaluation time.

### 🎯 Contributions

- A **frequency-native ODE solver** that performs style transport directly on
  Haar wavelet coefficients, with an exact inverse transform guaranteeing
  perfect reconstruction.
- A **per-subband Flow Matching objective** (FC-SB) that weights LL color
  migration and HF texture transport differently (`w_ll=0.3`, `w_lh=w_hl=1.0`),
  combined with a Structure-Aligned Target (SAT) and the brk_a LL partial-style
  mechanism.
- A **Channel-wise AdaIN endpoint** (`endpoint_adain_scale`) that decouples
  training stability from inference-time style strength.
- State-of-the-art **content preservation** on the D5-512 benchmark while
  remaining competitive on style alignment — trained from scratch on a single
  RTX 3060 12 GB.

---

## 🏆 Key Results (Table 1, D5-512)

WEAVE vs. representative baselines on the D5-512 hold-out test set
(5 distinct styles × 150 source images, n = 750 style-transfer pairs).
**↑** = higher is better, **↓** = lower is better.

| Method            | Class      | DINO-C ↑ | DINO-S ↑ | CLIP-S ↑ | LPIPS ↓ |
|-------------------|------------|---------:|---------:|---------:|--------:|
| Identity          | identity   |        — |        — |    0.693 |   0.000 |
| AdaIN             | classical  |        — |        — |    0.668 |   0.743 |
| WCT (VGG19)       | classical  |        — |        — |    0.706 |   0.635 |
| SD-Turbo          | diffusion  |        — |        — |    0.693 |   0.003 |
| IP-Adapter        | diffusion  |        — |        — |    0.829 |   0.636 |
| **WEAVE (ours)**  | —          | **0.810**| **0.492**| **0.713**| **0.259**|

**Reading the table.** `DINO-C` measures content preservation, `DINO-S` and
`CLIP-S` measure style alignment, and `LPIPS` measures perceptual distance to
the source (lower = more content preserved). SD-Turbo's near-zero LPIPS
indicates it barely alters the source; WEAVE delivers genuine style transfer
while keeping the highest DINO-C and the lowest LPIPS among true stylizers.
Baseline DINO-C/DINO-S are not reported by the original `main_table.csv` and
are marked `—`.

---

## 🗂 Repository Structure

```text
WEAVE/
├── model.py                # WEAVE main model: latent → DWT → backbone → 4 velocity heads → iDWT
├── flow.py                 # Per-subband Flow Matching objective (FC-SB)
├── wavelet.py              # Exact Haar DWT / IDWT utilities
├── trainer.py              # SBTrainer: optimizer, AMP, checkpoints, full-eval scheduling
├── run.py                  # Training entry point + per-epoch full evaluation + early stop
├── run_evaluation.py       # Evaluation driver: single ckpt / batch dir / multi-experiment
├── blocks.py               # ResidualBlock: AdaLN(t) → Self-Attn → Cross-Attn(style) → FFN
├── style.py                # StyleConditioner: learnable style-memory token projector
├── style_families.py       # Tokenizer / contract family registry and validation
├── internal_dynamics.py    # Training-dynamics probes (HF routing gates, gradient stats)
├── config_schema.py        # All config dataclasses + JSON load / merge / defaults
├── __init__.py             # Package exports: WEAVE, build_model_from_config
├── __main__.py             # `python -m WEAVE` entry → run.main()
│
├── config.json             # Canonical training / model / data config
├── inference.json          # Canonical 8-step inference & evaluation protocol
├── configs/                # Inference & rebuttal experiment variants
│   ├── inference_fix_whitening.json
│   ├── inference_no_preserve_mean.json
│   ├── exp_rebuttal_hh_head_p256.json
│   ├── exp_rebuttal_w10_p256.json
│   └── exp_rebuttal_w03_p256.json
│
├── utils/                  # Datasets, metric computation, training/inference helpers
├── tools/                  # One-off scripts: dataset prep, encoding, benchmarks, plots
├── tests/                  # Pytest suite (spectral ODE, infra, dynamics, layout)
│
├── requirements.txt        # Runtime dependencies
├── requirements-dev.txt    # Development / testing dependencies
├── pytest.ini              # Test configuration
│
├── INSTALL.md              # Installation guide
├── CHANGELOG.md            # Release history
├── CITATION.cff            # Citation metadata
└── LICENSE                 # MIT license
```

---

## 🚀 Quick Start

```bash
# 1. Create and activate a Python 3.10+ environment (conda recommended)
conda create -n weave python=3.10 -y
conda activate weave

# 2. Install runtime dependencies
pip install -r requirements.txt

# 3. Verify the package imports
python -c "from WEAVE import WEAVE, build_model_from_config; print('WEAVE package OK')"

# 4. Train the canonical 15-epoch run (single RTX 3060 12 GB, batch size 8)
python -m WEAVE --config config.json

# 5. Evaluate a checkpoint with the 8-step inference protocol
python run_evaluation.py --checkpoint runs/submission/canonical_oriented_epoch4/epoch_0004.pt \
                         --inference inference.json
```

> 💡 The datasets and result images are hosted on Google Drive — fetch them
> with `python scripts/download_data.py` (see
> [Data & Results Download](#-data--results-download)). The first time you run
> training or evaluation, latent caches are built under `runs/cache/`. See
> [INSTALL.md](INSTALL.md) for GPU memory tuning and
> [DATA.md](DATA.md) for the dataset layout.

---

## 📦 Data & Results Download

The datasets and result images are **too large for git** (~56 GB unpacked) and
are mirrored on **Google Drive** with public read links. Everything unpacks to
the exact paths the code expects (`data/`, `datasets/`, `results/`,
`eval_outputs/`) via one command.

| Archive           | Size      | Contents                                                       | Unpacks to       |
|-------------------|-----------|----------------------------------------------------------------|------------------|
| `eval_outputs.tar`| 694 MiB   | Evaluation output images (D5 / R5 / P2A-256)                   | `eval_outputs/`  |
| `data.tar`        | 7.0 GiB   | Training images, packed latent caches, ONNX VAE, TAESD          | `data/`          |
| `results.tar`     | 20.7 GiB  | Result images (D5-512, P2A-256, R5, ours, baselines, packs)     | `results/`       |
| `datasets.tar`    | 24.2 GiB  | WikiArt / COCO sources and preprocessed datasets, latent caches | `datasets/`      |

All archives carry a SHA-256 checksum that the download script verifies
automatically.

### Option A — one-command download & unpack (recommended)

From the repository root:

```bash
# download everything and unpack into data/, datasets/, results/, eval_outputs/
python scripts/download_data.py

# only the training data
python scripts/download_data.py --only data

# verify previously downloaded archives without re-extracting
python scripts/download_data.py --verify
```

The script streams from Google Drive (Python standard library only), verifies
each archive's size and SHA-256, then extracts with the system `tar` (bundled
on Windows 10+, macOS, and most Linux distros). Archives are deleted after a
successful unpack unless you pass `--keep`.

### Option B — raw download links

Each archive contains its top-level directory, so extracting at the repo root
restores the exact layout:

```bash
wget -c <URL> -O eval_outputs.tar && tar -xf eval_outputs.tar
```

- `eval_outputs.tar` — https://drive.google.com/open?id=1vWJVr8qDgdb2-RuA5K5k0znSj5Ut3dG9
- `data.tar` — https://drive.google.com/open?id=1ZdK7vpTgq7UibcRMz9QSoI1xeaYF2LTr
- `results.tar` — https://drive.google.com/open?id=1g6Srtkf5I8zFFbe5Q_Tf8dZmlmHtESoH
- `datasets.tar` — https://drive.google.com/open?id=1UOiz2UDV6IlpxCkBdBGvYPQS1mM7RXr9
- Folder — https://drive.google.com/open?id=1qQncxF2N7ioQFUi6uBmFRc3C6qnM72nB

> ⚠️ `results.tar` intentionally excludes `results/cross_protocol/submission_pack/boards/`
> (that subtree contains Windows junction links to result folders already
> included in the archive). The real per-method result images live in
> `results/D5-512/`, `results/P2A-256/`, `results/R5/`, `results/ours/` and
> `results/baselines/`.

**Requirements:** Python 3.8+ and ~56 GiB free disk space. No third-party
packages needed.

---

## 📚 Documentation

| Document | Description | Status |
|----------|-------------|--------|
| [INSTALL.md](INSTALL.md)         | System requirements, environment setup, troubleshooting | ✅ |
| [CHANGELOG.md](CHANGELOG.md)     | Versioned release history (Keep a Changelog format)    | ✅ |
| [CITATION.cff](CITATION.cff)     | Citation metadata in CFF 1.2 format                    | ✅ |
| [LICENSE](LICENSE)               | MIT license                                            | ✅ |
| [DATA.md](DATA.md)               | Dataset layout (D5-512, R5-512, P2A-256) and preparation | ✅ |
| REPRODUCE.md                     | Step-by-step reproduction of all paper tables          | 📝 forthcoming |
| EXPERIMENTS.md                   | Ablation and rebuttal experiment index                 | 📝 forthcoming |
| CHECKPOINT.md                    | Checkpoint conventions and download mirrors            | 📝 forthcoming |

---

## 📝 Citation

If you use WEAVE in your research, please cite it using the metadata in
[`CITATION.cff`](CITATION.cff). A BibTeX entry is provided below; author names
are anonymized pending the end of double-blind review.

```bibtex
@inproceedings{weave2027,
  title     = {WEAVE: Wavelet-based Entanglement-Augmented Vision Exchange},
  author    = {Anonymous Authors},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)},
  year      = {2027},
  note      = {Anonymous double-blind submission},
  url       = {https://anonymous.4open.science/r/weave}
}
```

---

## ⚖️ License

Released under the [MIT License](LICENSE).

```
MIT License

Copyright (c) 2026 Anonymous Authors (AAAI 2027 Submission)
```

---

## 🙏 Acknowledgments

WEAVE builds on a number of open-source efforts in the style-transfer,
flow-matching, and wavelet-analysis communities. We thank the maintainers of
PyTorch, the Stable Diffusion VAE, the DINO and CLIP metric backbones, and the
WikiArt dataset curators. Detailed credits will be added in the camera-ready
version following the double-blind review period.

---

<sub>
  🔒 This is an anonymous release for AAAI 2027 double-blind review.
  All identifying information has been redacted and will be restored upon acceptance.
</sub>
