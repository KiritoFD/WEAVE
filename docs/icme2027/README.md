# ICME 2027 Paper — `PRISM: Breaking Source-Style Lock for Fast Latent Style Transfer`

## Directory Structure

| File | Purpose |
|---|---|
| `paper.tex` | Main 6-page `IEEEtran` double-column manuscript (`\mathcal{D}_\star \circ \mathcal{T}_{s\to t} \circ \Pi_\perp` factorization) |
| `macros.tex` | **Centralized plug-and-play macros** (`\method`, `\methodfast`, `\methodpro`, `\methoddec` + headline numbers) |
| `refs.bib` | Curated BibTeX database (with `IEEEtranBSTCTL` compact formatting) |
| `STORY.md` | Narrative, mathematical hierarchy, baseline strategy, and modular extension guide |
| `make_figures.py` | Generates `fig1_prism_pareto`, `fig2_three_paradoxes`, `fig4_specificity_frontier` (`.pdf/.png/.svg`) |
| `make_arch_figure.py` | Generates vector architecture diagram `fig3_prism_architecture` (`.pdf/.png/.svg`) |
| `build.sh` | One-command figure regeneration + `pdflatex`/`bibtex` build |
| `paper.pdf` | Compiled 6-page IEEE ICME submission-ready PDF |

## Quick Rebuild & Future Extension

```bash
bash docs/icme2027/build.sh
```

- **To rename the method or update headline numbers**: edit `macros.tex` and the top constants (`METHOD`, `PRO_NAME`, `FAST_NAME`, `DEC_NAME`) in `make_figures.py` / `make_arch_figure.py`.
- **To plug in a newly discovered transport operator $\mathcal{T}_{s\to t}$**: Operators I ($\Pi_\perp$), III ($\mathcal{D}_\star$), Propositions 1--2, and the Dual-Gate Evaluation remain unchanged; update only Section III-C and append a row to Tables I--III.
