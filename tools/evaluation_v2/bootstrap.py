"""WEAVE Evaluator V2 - Source-Clustered Stratified Paired Bootstrap.

Proper Statistical Methodology:
1. Identifies unique source images (30 per style, 150 total).
2. Performs stratified resampling of source images within each source style.
3. When a source image is resampled, ALL its cross-style tasks (4 tasks per source)
   are sampled together as an indivisible cluster, respecting intra-source correlation!
4. Applies identical cluster resamples to paired models A and B on their common tasks.
5. Computes:
   - mean_diff = mean(A - B)
   - CI95 = [2.5th percentile, 97.5th percentile]
   - p_boot = two-sided empirical bootstrap p-value
   - win_rate = fraction of bootstrap rounds where mean(A) > mean(B)
   - n_common = number of common tasks evaluated
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple
import numpy as np

STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]


def build_source_cluster_indices(rows: List[dict]) -> Dict[str, Dict[str, List[int]]]:
    """Group row indices by {source_style: {source_stem: [row_idx_0, row_idx_1, ...]}}."""
    clusters = {s: {} for s in STYLES}
    for idx, r in enumerate(rows):
        s_style = r["source_style"]
        s_stem = r["source_stem"]
        if s_style in clusters:
            if s_stem not in clusters[s_style]:
                clusters[s_style][s_stem] = []
            clusters[s_style][s_stem].append(idx)
    return clusters


def source_clustered_bootstrap_metric(
    rows: List[dict],
    metric_key: str,
    n_boot: int = 2000,
    seed: int = 42,
) -> Tuple[float, List[float]]:
    """Compute single-model mean and 95% CI via stratified source-clustered bootstrap."""
    if not rows:
        return 0.0, [0.0, 0.0]

    clusters = build_source_cluster_indices(rows)
    rng = np.random.default_rng(seed)

    boot_means = []
    vals = np.array([r[metric_key] for r in rows], dtype=np.float64)

    for _ in range(n_boot):
        sample_indices = []
        for s_style in STYLES:
            stems = list(clusters[s_style].keys())
            if not stems:
                continue
            n_stems = len(stems)
            picked_stems = rng.choice(stems, size=n_stems, replace=True)
            for stem in picked_stems:
                sample_indices.extend(clusters[s_style][stem])
        boot_means.append(vals[sample_indices].mean())

    boot_arr = np.array(boot_means)
    return float(vals.mean()), [float(np.percentile(boot_arr, 2.5)), float(np.percentile(boot_arr, 97.5))]


def paired_clustered_bootstrap_comparison(
    rows_a: List[dict],
    rows_b: List[dict],
    metric_key: str,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict:
    """Perform paired clustered bootstrap test between Model A and Model B on common tasks."""
    map_a = {r["task_id"]: r[metric_key] for r in rows_a}
    map_b = {r["task_id"]: r[metric_key] for r in rows_b}
    common_ids = sorted(list(set(map_a.keys()) & set(map_b.keys())))

    if not common_ids:
        return {
            "n_common": 0,
            "mean_diff": 0.0,
            "ci95": [0.0, 0.0],
            "win_rate": 0.5,
            "p_boot": 1.0,
        }

    # Build aligned common task lists
    aligned_a = np.array([map_a[tid] for tid in common_ids], dtype=np.float64)
    aligned_b = np.array([map_b[tid] for tid in common_ids], dtype=np.float64)
    diffs = aligned_a - aligned_b

    # Cluster mapping for common tasks
    clusters = {s: {} for s in STYLES}
    for idx, tid in enumerate(common_ids):
        # tid format: {source_style}/{source_stem}/{target_style}
        parts = tid.split("/")
        s_style, s_stem = parts[0], parts[1]
        if s_style in clusters:
            if s_stem not in clusters[s_style]:
                clusters[s_style][s_stem] = []
            clusters[s_style][s_stem].append(idx)

    rng = np.random.default_rng(seed)
    boot_diff_means = []

    for _ in range(n_boot):
        sample_indices = []
        for s_style in STYLES:
            stems = list(clusters[s_style].keys())
            if not stems:
                continue
            n_stems = len(stems)
            picked = rng.choice(stems, size=n_stems, replace=True)
            for stem in picked:
                sample_indices.extend(clusters[s_style][stem])
        boot_diff_means.append(diffs[sample_indices].mean())

    boot_diff_arr = np.array(boot_diff_means)
    mean_diff = float(diffs.mean())
    ci95 = [float(np.percentile(boot_diff_arr, 2.5)), float(np.percentile(boot_diff_arr, 97.5))]
    win_rate = float((boot_diff_arr > 0).mean())

    # Two-sided empirical bootstrap p-value: H0: diff == 0
    # Center bootstrap distribution at 0 and check tail probability of observing >= |mean_diff|
    centered = boot_diff_arr - mean_diff
    p_boot = float((np.abs(centered) >= np.abs(mean_diff)).mean())

    return {
        "n_common": len(common_ids),
        "mean_diff": round(mean_diff, 5),
        "ci95": [round(ci95[0], 5), round(ci95[1], 5)],
        "p_boot": round(p_boot, 5),
        "win_rate": round(win_rate, 4),
        "significant_at_05": bool(ci95[0] > 0 or ci95[1] < 0),
    }
