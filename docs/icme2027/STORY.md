# ICME 2027 Story & Modular Architecture Blueprint (`STORY.md`)

## 1. Core Essence, Name & Title (`PRISM`)

- **Core Essence (方法主旨)**:
  Compact latent style transfer fails under standard training and scaling because **source-domain style stays entangled across three stages**—noisy/raw structural conditioning, one-sided target guidance, and an over-parameterized reconstruction decoder—creating **Source-Style Lock** (where scaling a self-paired flow $260\times$ actually *lowers* cross-style accuracy from $10.67\%$ to $6.67\%$). We factor conditional editing into three clean operators $\mathcal{D}_\star \circ \mathcal{T}_{s\to t} \circ \Pi_\perp$: **project** out chrominance and spatial channel moments ($\Pi_\perp$), transport along an **antisymmetric source-to-target chord** ($\mathcal{T}_{s\to t}$, amortized into a single millisecond-scale step with exact diagonal identity), and decode through a **pruned spectral-bottleneck renderer** ($\mathcal{D}_\star$).

- **Method Name — `PRISM`** (*Projected Invariant Style Mapping*):
  - Natural syllable blend (**Pr**ojected **I**nvariant **S**tyle **M**apping, just as `WEAVE` blended **W**av**e**let + **VAE** without forcing every letter!).
  - Effortless optical metaphor: a prism decomposes mixed input light into invariant structural geometry while dispersing cleanly toward the target chromatic/artistic mode.
  - Sub-variants (all controlled in one place via `macros.tex` and `CFG` in `make_*.py`):
    - **`PRISM-Pro`**: 3-step prior-calibrated bidirectional SPD tangent solver (`38.00%` CSD Top-1, `0.5674` Affinity, `115 ms/img` — strictly beating commercial Seedream 4.5 on **all 5** style & content metrics at $17\times$ higher speed).
    - **`PRISM-1`**: `0.66M` single-step antisymmetric chord network (`33.83%` CSD Top-1, `28.5 ms/img` = `35.1 img/s` on one RTX 3060, dropping to `1.33%` on counterfactual wrong labels = **`+32.50pp` causal specificity**).
    - **`PRISM-Dec`**: `10.66M` 4-stream Taylor-pruned spectral-bottleneck decoder (`-78.5%` params, `2.86×` faster, `+3.17pp` Top-1, $p<0.01$).

- **Title (Clean, Authoritative, Zero Gimmick)**:
  > **PRISM: Breaking Source-Style Lock for Fast Latent Style Transfer**

---

## 2. Multi-Layer Structural Design

1. **Introduction (Four-Movement Arc)**:
   - *Mvt. 1 (The Compress-Then-Edit Polarization)*: Why $1/8$-scale latent spaces should enable sub-megabyte millisecond style transfer, yet existing methods split between slow geometry-destroying diffusion (`StyleShot`, `StyleID`) and near-identity translators (`SaMam`, `CUT`, `SD-Turbo`).
   - *Mvt. 2 (Source-Style Lock Across Three Stages)*: Unifies three controlled empirical discoveries—(a) the **$260\times$ Self-Paired Scaling Paradox** ($49.3\%$ pure gen vs. $6.67\%$ SDEdit; $18.67\%$ $\tau_0=0$ brush-inversion ceiling), (b) the **Spectral-Bottleneck Decoder Paradox** (`+3.17pp` Top-1 and `2.86×` speedup from $78.5\%$ pruning), and (c) the **Specificity Mirage** (`ArtFID` $\rho=-0.16$ with style; post-hoc scalers scoring higher on wrong labels than true ones).
   - *Mvt. 3 (Three-Operator Factorization $\mathcal{D}_\star \circ \mathcal{T}_{s\to t} \circ \Pi_\perp$)*: Introduces `PRISM` as a principled operator decomposition rather than an ad-hoc module stack.
   - *Mvt. 4 (Contributions)*: Three crisp, high-altitude contributions with broad takeaways for latent generative editing, tokenizer design, and counterfactual evaluation.

2. **Theory & Methodology (Modular Operator Factorization)**:
   - **Sec. III-A (Diagnosis — Prop. 1)**: Posterior Gaussian concentration under self-paired partial denoising & stroke-cadence inversion under $\tau_0=0$ edge conditioning.
   - **Sec. III-B (Operator I: Quotient Structure Projection $\Pi_\perp$ — Prop. 2)**: Annihilates the product symmetry group $\mathcal{G}_{\mathrm{style}} = \ker(w_Y^\top) \times (\R^D \rtimes \R_+^D)$ via stabilized multi-scale luminance contrast $C_{64}$ and zero-mean spherical DINOv2 layout $\tilde{P}_{16} \in \mathcal{M}_0$ ($72.3\% \to 18.3\%$ style probe, $0.994$ layout cosine).
   - **Sec. III-C (Operator II: Antisymmetric Chord Transport $\mathcal{T}_{s\to t}$ — Prop. 3)**: Formulates constant-content style translation as a skew-symmetric operator ($\mathcal{T}_{s\to s}=\mathrm{id}$, $e_{t\to s}=-e_{s\to t}$), pairing our prior-calibrated bidirectional SPD teacher (`PRISM-Pro`) with single-step chord amortization (`PRISM-1`). *(Solver line-search grids cleanly placed in Sec. IV-A Implementation Details!)*
   - **Sec. III-D (Operator III: Spectral-Bottleneck Decoding $\mathcal{D}_\star$ — `PRISM-Dec`)**: Structured 4-stream Taylor pruning as a range-space regularizer suppressing source-texture retraction.

3. **Dual-Gate Evaluation Geometry (Sec. IV)**:
   - **Boundary Gate (`IDT–TGT` Corridor)** + **Causal Gate (Counterfactual Cyclic Label-Shift $\Delta_{\mathrm{causal}}$)**.

4. **Plug-and-Play Extensibility**:
   - Update `macros.tex` + `CFG` at top of `make_figures.py` / `make_arch_figure.py`, or swap only **Sec. III-C** if a stronger transport operator $\mathcal{T}_{s\to t}$ is discovered later; `build.sh` recompiles everything in 5 seconds.
