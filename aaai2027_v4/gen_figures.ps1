# Regenerate every figure used by paper.tex from the bundled scripts + fig_data/.
# The paper folder is self-contained: no repo-external paths are required.
# Usage:  .\gen_figures.ps1
$ErrorActionPreference = "Stop"
$dir = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $dir
try {
    if (Test-Path .\gen_framework_figure.py) {
        python gen_framework_figure.py  # framework_sfm_main.png  (architecture diagram)
    } else {
        Write-Host "Skipping framework figure: gen_framework_figure.py not present."
    }
    if (Test-Path .\gen_teaser_figure.py) {
        python gen_teaser_figure.py     # fig_teaser_comparison.png
    } else {
        Write-Host "Skipping teaser figure: gen_teaser_figure.py not present."
    }
    python plot_page1_a.py              # fig_page1_a.png  (Figure 1, scatter; SaMam on convergence apex)
    python plot_page1_b.py              # fig_page1_b.png  (Figure 2, ArtFID)
    python plot_page1_summary.py        # fig_distinct5_page1_summary.pdf (side-by-side preview only)
    python make_radar_metric_blocks.py  # radar_metric_blocks_A_clip_dinos_robustbreak.png
    Write-Host "All figures regenerated."
} finally {
    Pop-Location
}
