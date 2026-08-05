# AAAI Reproducibility Checklist

Status: submission audit for `aaai2027_v4`  
Author-kit source: `AuthorKit27/AuthorKit27/ReproducibilityChecklist.tex`  
Filled checklist: `aaai2027_v4/ReproducibilityChecklist.tex`

This document records the evidence behind the filled AAAI checklist and the remaining
submission risks. It is an audit, not a claim that every item is already complete.

## Checklist Status

| Author-kit section | Status | Evidence / action |
|---|---|---|
| General paper structure | Yes | Conceptual pipeline, equations, architecture figure, limitations, and background citations are in `paper.tex` and `supplement.tex`. |
| Theoretical contributions | No | WEAVE uses mathematical definitions and design analysis, but makes no theorem-level contribution. The checklist's nested theory questions are not part of the paper's claim. |
| Dataset usage | Yes | Dataset motivation, board construction, pair counts, preprocessing contract, and logical paths are in `supplement.tex`, Sections `Datasets and Evaluation Boards` and `Data Preprocessing Contract`. |
| Computational experiments | Yes | Training, inference, metrics, ablations, timing, and probe protocols are documented in `paper.tex` and `supplement.tex`. |

## Detailed Audit

### General Paper Structure

- `[x]` Conceptual method outline: Haar decomposition, endpoint construction, velocity model, and inference route are described in the method and architecture figure.
- `[x]` Objective results are separated from hypotheses and limitations.
- `[x]` Haar, flow matching, spectral bias, VAE, and evaluation metrics have contextual citations or definitions.

### Theory

The paper does not claim a new theorem, proof, convergence guarantee, or general law about spectral bias. The affine-fiber and orthogonality material is a mathematical description of the implemented transform. Wording must continue to avoid `theorem`, `guarantee`, `proof`, `capacity exhaustion` as a proven mechanism, and universal claims about latent spaces.

### Dataset Usage

- `[x]` D5-512, P2A-256, and R5-WikiArt are motivated by resolution, cross-domain, and style-family coverage.
- `[x]` The supplement states board sizes, source-target construction, identity rows, style names, split logic, latent scale, and cache requirements.
- `[x]` Existing WikiArt and VAE sources are cited in the paper.
- `[ ]` Public release license and redistribution status are not yet recorded in the repository. Add a `LICENSE` file and a release note before marking the code/data release item `yes`.
- `[ ]` The released package must not contain private image data or machine-specific latent caches. Release manifests and download/preprocessing instructions are required instead.

### Computational Experiments

- `[x]` The D5 configuration records architecture, Haar mode, endpoint target, loss weights, optimizer, learning-rate schedule, batch size, precision, clipping, epochs, and stopping rule.
- `[x]` The inference configuration records eight solver steps, endpoint scale, batch sizes, VAE choice, and metric-cache policy.
- `[x]` Seeds are set in `run.py`; the main paper seed and probe stability checks are documented in `supplement.tex`.
- `[x]` DINO-S, CLIP-S, DINO-C, LPIPS, and ArtFID semantics and directions are documented. DINO-S is the primary style metric.
- `[x]` Runtime accounting separates training after latent caching, generation, VAE decoding, metric extraction, and serialization.
- `[partial]` The software environment names library ranges and GPU memory, but exact Python, OS, CUDA, driver, PyTorch, and diffusers versions should be frozen in a machine-readable environment file.
- `[partial]` Development sweeps are recorded in experiment ledgers, but the full tried range and selection criterion for every final hyperparameter are not summarized in one place.
- `[partial]` Source code exists in the repository, but it is not embedded as a literal code appendix. The release must provide the root entry points, preprocessing scripts, evaluation scripts, configs, and a clean manifest.
- `[no]` The paper does not claim statistical significance tests for the main table. It reports board means and selected stability intervals instead.
- `[no]` The main table does not consistently state the number of independent training runs for every baseline. This must be clarified in the supplement or explicitly marked as single-run evaluation where applicable.

## Required Before Submission

1. Add exact environment metadata: Python version, OS, CUDA toolkit, NVIDIA driver, PyTorch, torchvision, transformers, diffusers, VAE commit/model identifier, and GPU memory.
2. Add `LICENSE` and a release manifest. State which data are downloadable, which require user-side download, and which caches/checkpoints are excluded.
3. Add one canonical clean reproduction command from a fresh checkout, including preprocessing/cache construction, training, checkpoint selection, evaluation, and figure regeneration.
4. Add a table mapping every main-table row to: checkpoint/source, number of runs, hardware, evaluation board, and whether the value is reproduced or taken from the cited paper.
5. Summarize hyperparameter searches used for the final model: parameter, values tried, selection rule, and whether selection used external metrics or internal dynamics.
6. Audit all public paths and logs for absolute drive letters, SSH hosts, usernames, private caches, and generated outputs that should not be redistributed.
7. Compile the filled official checklist standalone with the author kit and verify that `NA` is accepted by the submission form for conditional dataset/theory items.
8. Rebuild the paper and supplement from the clean release directory, then verify that the generated PDFs contain no bookmarks, page numbers, headers, or forbidden links under the AAAI submission style.

## Recommended Evidence Map

| Reproduction task | Source of record |
|---|---|
| Train WEAVE from scratch | `run.py`, `config.json`, `experiments/architecture/hf_oriented_internal_early_stop.json` |
| Run fixed internal stopping | `trainer.py`, `docs/reproduction/internal_dynamics_early_stop.csv` |
| Run eight-step evaluation | `inference.json`, `scripts/batch_eval_all.py`, `utils/run_evaluation.py` |
| Rebuild paper figures | `aaai2027_v4/gen_figures.ps1` and figure scripts |
| Inspect method and metric details | `aaai2027_v4/supplement.tex` |
| Reproduce latent learnability diagnostic | `experiments/rebuttal/unseparated_latent/` and the Figure 3 source records |
| Audit historical experiments | `docs/713/` and `archives/README.md` |

## Current Assessment

The paper is substantively reproducible for an informed user with the dataset and local
environment already prepared. It is not yet release-grade reproducible from a clean public
checkout because environment pinning, release licensing, baseline run-count provenance, and
the complete hyperparameter-search ledger are incomplete. Those are the highest-priority
items before submission.
