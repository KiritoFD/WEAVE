# Changelog

All notable changes to the WEAVE project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-07-25

Initial public release of WEAVE (Wavelet-based Entanglement-Augmented Vision
Exchange), prepared anonymously for AAAI 2027 double-blind review.

### Added

- **Core model (`model.py`)** — frequency-native ODE solver: latent → Haar DWT
  → shared backbone → four velocity heads (LL / LH / HL / HH) → exact iDWT
  synthesis. Includes `WEAVE`, `VelocityHead`, `StyleConditionedVelocityHead`,
  `StyleDeltaVelocityHead`, `StationaryTextureStatsEncoder`, and the
  `build_model_from_config` factory.
- **Flow Matching objective (`flow.py`)** — per-subband FC-SB Spectral ODE
  training target with weighted LL / LH / HL losses (`w_ll=0.3`,
  `w_lh=w_hl=1.0`), Structure-Aligned Target (SAT), and the brk_a LL
  partial-style mechanism.
- **Wavelet utilities (`wavelet.py`)** — exact Haar `dwt2_haar` / `idwt2_haar`
  with the `IDWT(DWT(x)) = x` guarantee, plus multi-level decomposition and
  subband-gamma helpers.
- **Training pipeline (`trainer.py`, `run.py`)** — `SBTrainer` with AMP
  (`bf16`), fused AdamW, gradient clipping, checkpointing, per-epoch full
  evaluation scheduling, and convergence-based early stopping. Canonical 15-epoch
  run validated on a single RTX 3060 12 GB with `batch_size=8`.
- **Evaluation pipeline (`run_evaluation.py`)** — single-checkpoint, batch
  directory, and multi-experiment aggregation modes; computes DINO-S, DINO-C,
  CLIP-S, and LPIPS.
- **Channel-wise AdaIN endpoint** — `endpoint_adain_scale` field decoupling
  training stability (`1.0` in `config.json`) from inference-time style strength
  (`2.0` in `inference.json`).
- **Style conditioning (`style.py`, `blocks.py`)** — learnable style-memory
  token projector and `ResidualBlock` with AdaLN(t) → Self-Attention →
  Cross-Attention(style) → FFN.
- **Config schema (`config_schema.py`)** — full dataclass schema (~400 fields
  across `model` / `bridge` / `training` / `data` / `checkpoint` / `full_eval` /
  `ablation`) with JSON load, merge, and inference-default resolution.
- **Canonical configs** — `config.json` (training), `inference.json` (8-step
  evaluation protocol), and 5 variants under `configs/` (inference whitening /
  mean-preservation fixes and three rebuttal experiment configs).
- **Dataset support** — D5-512 (Distinct 5 styles, 512 px, hold-out test),
  R5-512 (Random 5 styles), and P2A-256 (Photo-to-Art). Dataset preparation and
  VAE encoding tools under `tools/`.
- **Paper experiments** — main-table baselines (Identity, AdaIN, WCT-VGG19,
  SD-Turbo, IP-Adapter), the brk_a LL partial-style ablation, the
  `dino_s_break` ablation axis, cross-protocol reports, and VAE-port studies.
- **Utilities** — `utils/` (datasets, metric computation, training/inference
  helpers) and `tools/` (dataset prep, encoding, benchmarking, plotting).
- **Test suite** — `tests/` covering spectral ODE correctness, inference
  infrastructure, internal dynamics, training-log schema, and submission root
  layout.
- **Documentation** — `README.md`, `INSTALL.md`, `LICENSE`, `CITATION.cff`,
  and this `CHANGELOG.md`.

### Known Issues

- **No packaging metadata.** The project ships without `setup.py` /
  `pyproject.toml`; it is run from source. The package must be imported as
  `WEAVE` (uppercase). A `__version__` attribute is not yet exposed on the
  package — version is tracked in this file and in `CITATION.cff`.
- **ONNX / TensorRT VAE decoder unconfigured by default.**
  `full_eval.vae_onnx_decoder` is empty; users must export their own ONNX
  decoder to enable the accelerated backend. TensorRT support is opt-in.
- **DINO-C / DINO-S not reported for all baselines.** The shipped
  `results/tables/main_table.csv` reports CLIP-S, LPIPS, and MUSIQ for the
  classical / diffusion baselines; DINO-C and DINO-S are only populated for
  WEAVE. Re-running baselines through the unified DINO evaluation is on the
  roadmap.
- **Dataset download not automated.** D5-512, R5-512, and P2A-256 must be
  prepared manually using the scripts under `tools/` (e.g.
  `prepare_d5_512_clean.py`, `vae_encode_p2a256.py`, `vae_encode_r5.py`). A
  consolidated `DATA.md` with mirrors and MD5 manifests is forthcoming.
- **`r20` and `w5` training sets share ~11% of images by MD5.** Users combining
  these two sources should deduplicate before training to avoid leakage. See the
  dedup analysis referenced in the audit notes.
- **Windows path casing.** On case-sensitive filesystems (Linux), preserve the
  `WEAVE` directory casing. On Windows this is transparent, but scripts that
  hard-code lowercase `weave` will fail.
- **Forthcoming documents.** `DATA.md`, `REPRODUCE.md`, `EXPERIMENTS.md`, and
  `CHECKPOINT.md` are referenced from the README but not yet shipped in this
  release. They will be added in a subsequent patch release.
- **No multi-GPU training.** Only single-GPU training is supported. The
  canonical run fits on 12 GB VRAM; multi-GPU sharding is not implemented.

---

[1.0.0]: https://anonymous.4open.science/r/weave/releases/tag/v1.0.0
