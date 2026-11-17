#!/usr/bin/env bash
set -euo pipefail
export PATH="/f/texlive/2026/bin/windows:$PATH"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
python make_figures.py
python make_arch_figure.py
python make_qualitative_figure.py
pdflatex -interaction=nonstopmode paper.tex > /dev/null
bibtex paper
pdflatex -interaction=nonstopmode paper.tex > /dev/null
pdflatex -interaction=nonstopmode paper.tex | grep -E "Overfull|Output written|Warning" || true
