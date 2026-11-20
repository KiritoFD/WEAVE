# ICME 2027 Paper — `PRISM: Breaking Source-Style Lock to Redefine the Content–Style–Speed Frontier`

## Submission Specification (ICME 2027 Compliance)

- **Title**: `PRISM: Breaking Source-Style Lock to Redefine the Content–Style–Speed Frontier`
- **Page Limit**: **Strictly 6 pages total** (including main text, 4 figures, 3 tables, and 37 references)
- **Abstract Length**: **145 words** (strictly within the 100–150 word requirement)
- **Anonymity**: Double-blind (`Anonymous ICME Submission`, zero self-identifying repository or author links)

## Directory Structure

| File | Purpose |
|---|---|
| `paper.tex` | Main 6-page `IEEEtran` double-column manuscript ($\mathcal{D}_\star \circ \mathcal{T}_{s\to\tau} \circ (\Pi_\perp,\Pi_\parallel)$ factorization) |
| `macros.tex` | Centralized macros (`\method`, `\methodfast`, `\methodhc`, `\methodflow`, `\methoddec` + headline metrics) |
| `refs.bib` | Curated BibTeX database (`IEEEtranBSTCTL` compact formatting, including `StyleGallery` CVPR 2026) |
| `STORY.md` | Narrative architecture, three attack propositions, three contributions, and rebuttal defense matrix |
| `make_figures.py` | Generates `fig1_prism_pareto`, `fig2_three_paradoxes`, `fig4_specificity_frontier` (`.pdf/.png/.svg`) |
| `make_arch_figure.py` | Generates vector architecture diagram `fig3_prism_architecture` (`.pdf/.png/.svg`) |
| `make_qualitative_figure.py` | Generates full-width qualitative comparison matrix `fig5_qualitative` (`.pdf/.png/.svg`) |
| `build.sh` | One-command figure regeneration + `pdflatex`/`bibtex` build |
| `paper.pdf` | Compiled 6-page IEEE ICME submission-ready PDF |

## Quick Rebuild

```bash
bash docs/icme2027/build.sh
```
