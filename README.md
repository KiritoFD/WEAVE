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

## 🧭 Current research state (2026-11)

> The paragraph above describes the original AAAI-2027 submission snapshot.
> Research has since moved on; the paper-era flat modules referenced below now
> live in `tools/legacy_root/` (root `__init__.py`/`__main__.py` are thin
> compatibility shims). For the current research state, read in order:
>
> 1. [`AGENTS.md`](AGENTS.md) — decision protocol (2026-11-14): **all decoding uses the
>    S10 pruned VAE decoder** (10.66M params, −78.5%, 2.86× faster; measurably better
>    than the original decoder under the same protocol).
> 2. [`docs/HIGHLIGHTS.md`](docs/HIGHLIGHTS.md) — the two headline results: the S10
>    decoder and the **Q2A + Oracle** route/protocol (32-method leaderboard, Affinity
>    0.5587, CSD Top-1 36.3%, rank #4).
> 3. [`docs/README.md`](docs/README.md) — full documentation index (per-phase docs,
>    retrospective `docs/1111/`, leaderboards, negative results).
>
> Active code now lives in `src/` (facade) + `exp/v8`, `exp/v9_local` (experiment
> packages); historical code is archived under `tools/` and `archives/`.

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
├── src/                    # ACTIVE: facade layer (model/infer/eval/utils)
├── exp/                    # ACTIVE experiment packages: v8 (arm matrix), v9_local (V9-LD)
├── tools/                  # Historical code archive; PINNED subpackages used by src/
│   │                       # (see tools/README.md), incl. legacy_root/ = paper-era flat modules
│   │                       # (model.py, trainer.py, run.py, config_schema.py, ...)
├── scripts/                # Submission / reproduction scripts
├── tests/                  # Pytest suite (layout guards, spectral ODE, infra, dynamics)
├── configs → tools/configs_retired/   # retired experiment variants (moved 2026-11-14)
│
├── config.json             # Canonical training / model / data config
├── inference.json          # Canonical 8-step inference & evaluation protocol
├── paths.py                # Central path resolver (WEAVE_ROOT / asset roots)
│
├── eval_protocol/          # Frozen 750-task benchmark (Cross600 + Same150)
├── deliver/                # Delivery package incl. the S10 decoder checkpoints
├── runs/                   # Run outputs (mostly gitignored)
├── docs/                   # Documentation (start at docs/README.md)
├── archives/               # Cold storage (scratch, IDE workspaces, old outputs)
├── aaai2027_v4/            # LaTeX paper build
│
├── requirements.txt / requirements-dev.txt / pytest.ini
├── INSTALL.md / CHANGELOG.md / CITATION.cff / LICENSE
└── __init__.py / __main__.py   # compatibility shims → tools.legacy_root
```

---

## 🚀 Quick Start

```bash
# 1. Create and activate a Python 3.10+ environment (conda recommended)
conda create -n weave python=3.10 -y
conda activate weave

# 2. Install runtime dependencies
pip install -r requirements.txt

# 3. Verify the package imports (paper-era modules live in tools/legacy_root/)
python -c "import tools.legacy_root.model as m; print('WEAVE model OK:', m.WEAVE.__name__)"

# 4. Train the canonical 15-epoch run (single RTX 3060 12 GB, batch size 8)
python tools/legacy_root/run.py --config config.json

# 5. Evaluate a checkpoint with the 8-step inference protocol
python tools/legacy_root/run_evaluation.py --checkpoint runs/submission/canonical_oriented_epoch4/epoch_0004.pt \
                                           --inference inference.json
```

> 💡 The first time you run training or evaluation, latent caches will be built
> under `runs/cache/`. See [INSTALL.md](INSTALL.md) for GPU memory tuning and
> [docs/legacy_root/DATA.md](docs/legacy_root/DATA.md) for dataset layout.

---

## 📚 Documentation

| Document | Description | Status |
|----------|-------------|--------|
| [INSTALL.md](INSTALL.md)         | System requirements, environment setup, troubleshooting | ✅ |
| [CHANGELOG.md](CHANGELOG.md)     | Versioned release history (Keep a Changelog format)    | ✅ |
| [CITATION.cff](CITATION.cff)     | Citation metadata in CFF 1.2 format                    | ✅ |
| [LICENSE](LICENSE)               | MIT license                                            | ✅ |
| [REPRODUCE.md](REPRODUCE.md)     | Step-by-step reproduction of all paper tables          | ✅ |
| [docs/README.md](docs/README.md) | Current documentation index (research state, highlights)| ✅ |
| [docs/legacy_root/DATA.md](docs/legacy_root/DATA.md) | Dataset layout (D5-512, R5-512, P2A-256) and preparation | 📝 archived |
| [docs/legacy_root/EXPERIMENTS.md](docs/legacy_root/EXPERIMENTS.md) | Ablation and rebuttal experiment index | 📝 archived |
| [docs/legacy_root/CHECKPOINT.md](docs/legacy_root/CHECKPOINT.md) | Checkpoint conventions and download mirrors | 📝 archived |

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
