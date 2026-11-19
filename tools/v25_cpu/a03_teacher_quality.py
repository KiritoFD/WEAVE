"""A03: O2 teacher quality & selection-bias diagnosis (V25 Slot A, CPU-only).

Question: how much of the ``O3_label_free`` style weakness (t45 ``dino_s``
0.4666 vs ``O3`` 0.5245) is caused by the teacher selection pipeline biasing
toward near-identity outputs?  This module quantifies the attribution from
local artifacts only -- it never re-runs the teacher optimisation.

Evidence layers
---------------
1. Code level (static facts, recorded with file:line in the audit report):
   ``exp/v24/o2_ref_teacher_pairs.py`` ranks all candidates by ``mid_cos``
   (content retention vs the SOURCE) and keeps the Top-K (line 126).  The
   selection criterion is monotone-decreasing in edit magnitude, so Top-K
   systematically drops the strongest-edit pairs.
2. Data level: ``runs/v24/ref_teacher_pairs_2000.pt`` (kept-pair metadata,
   65 MB, under the 100 MB local-read limit) and the label-teacher contrast
   pack ``runs/v23/train_cross_pairs_1000.pt``.  The train pool is style
   blocked (880 per style, ``exp/v18/__init__.py:clean_balanced_pool``), so
   ``idx // 880`` recovers content/reference families without reading any
   label at teacher time (post-hoc analysis only).
3. Student level: ``runs/v25/r3_oneshot_minimal.json`` (O3 / O3_label_free /
   O3_self_only arms) + ``runs/v24/p0_o1_o3.json``.  ``self_only`` is an
   identity-teacher control: the label-free teacher's marginal style
   contribution over identity is ``lf - self``.

Known local limitation (BLOCKED item, see audit report): the 4400-row train
latent cache (``runs/v18/cache/ccf_pool_cache.pt``) lives on the GPU hosts,
so per-pair latent L2 / colour-change correlations cannot be computed here.
``pearson``/``spearman`` are provided ready-made for the remote export.

CPU-only (numpy + stdlib + ``torch.load(map_location="cpu")``); callers must
limit threads (``OMP_NUM_THREADS=2``).  No CUDA, no ssh, no git.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

PAIRS_V24 = REPO_ROOT / "runs/v24/ref_teacher_pairs_2000.pt"
PAIRS_V23 = REPO_ROOT / "runs/v23/train_cross_pairs_1000.pt"
R3_JSON = REPO_ROOT / "runs/v25/r3_oneshot_minimal.json"
O3_JSON = REPO_ROOT / "runs/v24/p0_o1_o3.json"

N_STYLES = 5
TRAIN_PER_STYLE = 880
STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]

#: C1 target band ceiling (docs/19_pull_together/02_mechanism_plan.md C.2):
#: quality screening wants ``mid_cos in [0.80, 0.90]``
C1_BAND = (0.80, 0.90)


# ---------------------------------------------------------------------------
# pure, unit-tested helpers
# ---------------------------------------------------------------------------
def rejected_mean_cos(mean_all: float, mean_kept: float, n_all: int, n_kept: int) -> float:
    """Exact mean ``mid_cos`` of the Top-K rejected candidates.

    Only ``mean_all`` / ``mean_kept`` scalars are saved in the packs, but the
    rejected mean is recoverable exactly because Top-K partitions the pool.
    """
    if n_all <= n_kept or n_kept <= 0:
        raise ValueError("need 0 < n_kept < n_all")
    return (mean_all * n_all - mean_kept * n_kept) / (n_all - n_kept)


def selection_share_of_excess(mean_all: float, mean_kept: float, band_hi: float = C1_BAND[1]) -> float:
    """Fraction of the excess ``(mean_kept - band_hi)`` that Top-K selection added.

    ``(mean_kept - mean_all) / (mean_kept - band_hi)``: how much of the gap
    between the shipped teacher and the C1 band ceiling is attributable to the
    Top-K selection itself (remainder = the optimisation objective).
    """
    excess = mean_kept - band_hi
    if excess <= 0:
        raise ValueError("kept mean is already inside the band; share undefined")
    return (mean_kept - mean_all) / excess


def kept_cos_stats(mid_cos: np.ndarray) -> dict:
    """Distribution summary of the kept ``mid_cos`` values.

    ``threshold`` = min(kept) is exact: Top-K keeps the largest ``n_kept``
    values, so the smallest kept value *is* the selection cut.
    """
    x = np.asarray(mid_cos, dtype=np.float64)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("mid_cos must be a non-empty 1-D array")
    q = np.quantile(x, [0.05, 0.25, 0.50, 0.75, 0.95])
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "min_threshold": float(x.min()),
        "max": float(x.max()),
        "q05": float(q[0]),
        "q25": float(q[1]),
        "q50": float(q[2]),
        "q75": float(q[3]),
        "q95": float(q[4]),
        "frac_gt_095": float((x > 0.95).mean()),
        "frac_gt_098": float((x > 0.98).mean()),
        "frac_in_c1_band": float(((x >= C1_BAND[0]) & (x <= C1_BAND[1])).mean()),
        "frac_below_band": float((x < C1_BAND[0]).mean()),
    }


def family_of(idx: int | np.ndarray, per_style: int = TRAIN_PER_STYLE) -> int | np.ndarray:
    """Family id of a train-pool index (style-blocked layout, 880 per style)."""
    arr = np.asarray(idx)
    out = arr // per_style
    if out.size and (out.min() < 0 or out.max() >= N_STYLES):
        raise ValueError("index outside the 5x880 train pool")
    return int(out) if out.ndim == 0 else out


def family_pair_matrix(
    idx_c: np.ndarray, idx_r: np.ndarray, per_style: int = TRAIN_PER_STYLE
) -> np.ndarray:
    """(n_styles x n_styles) counts of kept pairs, rows = content family."""
    fc = family_of(np.asarray(idx_c), per_style)
    fr = family_of(np.asarray(idx_r), per_style)
    if fc.shape != fr.shape:
        raise ValueError("idx_c / idx_r shape mismatch")
    mat = np.zeros((N_STYLES, N_STYLES), dtype=np.int64)
    np.add.at(mat, (fc, fr), 1)
    return mat


def same_family_fraction(idx_c: np.ndarray, idx_r: np.ndarray, per_style: int = TRAIN_PER_STYLE) -> float:
    """Fraction of pairs whose reference family equals the content family."""
    fc = family_of(np.asarray(idx_c), per_style)
    fr = family_of(np.asarray(idx_r), per_style)
    if fc.shape != fr.shape:
        raise ValueError("idx_c / idx_r shape mismatch")
    return float((fc == fr).mean())


def retention_estimates(kept_counts: np.ndarray, n_cand_total: int) -> dict:
    """Per-family retention rate under uniform random candidate sampling.

    The packs store only kept indices, so the rejected identities are unknown;
    under the uniform draw used by the teacher (``torch.randint`` over the
    balanced 4400 pool) each family supplies ``n_cand_total / n_styles``
    candidates in expectation.  Rates are therefore estimates; the binomial
    sd of the candidate count is propagated to a rate CI.
    """
    k = np.asarray(kept_counts, dtype=np.float64)
    if k.size != N_STYLES:
        raise ValueError("need one kept count per family")
    n_exp = n_cand_total / N_STYLES
    sd_cand = float(np.sqrt(n_cand_total * 0.2 * 0.8))  # Binomial(n, 1/5) sd
    rate = k / n_exp
    sd_rate = float(sd_cand / n_exp)
    return {
        "kept_counts": k.astype(int).tolist(),
        "expected_candidates_per_family": n_exp,
        "rate": rate.tolist(),
        "rate_sd": sd_rate,
        "rate_ci_lo": (rate - 1.96 * sd_rate).tolist(),
        "rate_ci_hi": (rate + 1.96 * sd_rate).tolist(),
    }


def _avg_ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties share the mean rank), 1-based."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=np.float64)
    sx = x[order]
    i = 0
    while i < x.size:
        j = i
        while j + 1 < x.size and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape or x.size < 2:
        raise ValueError("need matching arrays of length >= 2")
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape or x.size < 2:
        raise ValueError("need matching arrays of length >= 2")
    return pearson(_avg_ranks(x), _avg_ranks(y))


def attribution_chain(o3: dict, label_free: dict, self_only: dict) -> dict:
    """Quantify how far the identity-teacher control explains the style gap.

    All inputs are arm rows with ``dino_c``/``dino_s``/``csd_top1``.  Defines:
    - ``style_gap``: O3 minus label_free (the gap to explain);
    - ``identity_control_gap``: O3 minus self_only (what a *zero-signal*
      identity teacher already fails to close);
    - ``teacher_marginal``: label_free minus self_only (the label-free
      teacher's marginal contribution over identity, <= 0 means none);
    - ``share_at_or_below_identity``: fraction of the gap already present at
      the identity control (>= 1 means the teacher added no positive signal);
    - ``style_per_content``: dino_s bought per unit dino_c sacrificed vs the
      identity control (edit-direction quality; negative = edits that do not
      read as style).
    """
    out = {}
    for m in ("dino_c", "dino_s", "csd_top1"):
        gap = o3[m] - label_free[m]
        ctrl = o3[m] - self_only[m]
        marg = label_free[m] - self_only[m]
        out[m] = {
            "o3": o3[m],
            "label_free": label_free[m],
            "self_only": self_only[m],
            "gap_o3_minus_lf": gap,
            "identity_control_gap": ctrl,
            "teacher_marginal_over_identity": marg,
        }
        if gap != 0:
            out[m]["share_at_or_below_identity"] = ctrl / gap
    dc_s = self_only["dino_c"] - label_free["dino_c"]
    ds = label_free["dino_s"] - self_only["dino_s"]
    dc_o = self_only["dino_c"] - o3["dino_c"]
    ds_o = o3["dino_s"] - self_only["dino_s"]
    out["style_per_content"] = {
        "label_free_vs_self": (ds / dc_s) if dc_s > 0 else None,
        "o3_vs_self": (ds_o / dc_o) if dc_o > 0 else None,
    }
    return out


def c1_parameter_block() -> dict:
    """Dual-metric screening + three-tier teacher parameters for DEV/G2.

    Grounded in the measured distributions (see audit report section 5);
    thresholds are starting points to be re-calibrated on the 600-pair
    diagnostic subset per mechanism plan C.2.
    """
    return {
        "sampling": {
            "cross_family_only": True,
            "reason": "random reference draws admit same-family pairs (measured "
            "kept fraction reported by main()); same-family references cap the "
            "achievable style increment by construction",
            "stratify_directed_pairs": 20,
            "candidates_per_directed_pair": 120,
            "label_usage_decision": "choose one: (a) strict zero-label -- reject a "
            "reference whose 27D Oklab fiber cosine to the content exceeds 0.985 "
            "and stratify by fiber clustering k=5 (threshold needs calibration); "
            "(b) label-aware sampling only -- stratify on the style-blocked train "
            "pool layout (idx // 880), labels never enter any loss/target/eval; "
            "must be declared as sampling-label-aware in the experiment record",
        },
        "objective": {
            "w_c": {"T-conservative": 2.2, "T-medium": 1.2, "T-expressive": 0.7},
            "w_reg": 0.05,
            "note": "current w_c=2.2 puts ALL 2400 candidates above the C1 band "
            "(kept mean 0.9646, rejected mean 0.9426 >= band_hi 0.90): band "
            "filtering alone retains 0 pairs, so the calibration MUST change "
            "w_c, not only the selection rule",
        },
        "optimizer_budget": {
            "steps": {"T-conservative": 22, "T-medium": 40, "T-expressive": 48},
            "lr": 0.055,
            "note": "22 Adam steps at lr 0.055 cap the per-coordinate move at "
            "~1.2 latent-sd (sign-consistent upper bound); the expressive tier "
            "needs ~2x the budget to physically reach mid_cos 0.80-0.86",
        },
        "selection": {
            "dual_metric": "keep iff mid_cos in [0.80, 0.92] AND style_increment >= q25",
            "mid_cos_band": [0.80, 0.92],
            "style_increment": "SWD-to-reference improvement over the source "
            "baseline (requires per-pair export, see fields_to_export)",
            "per_family_retention_floor": 0.30,
        },
        "expected_mid_cos_band": {
            "T-conservative": [0.94, 0.97],
            "T-medium": [0.84, 0.91],
            "T-expressive": [0.78, 0.86],
        },
        "expected_retention_after_dual_filter": {
            "T-conservative": 0.83,
            "T-medium": [0.55, 0.70],
            "T-expressive": [0.40, 0.60],
        },
        "fields_to_export": [
            "mid_cos (kept + rejected)",
            "latent_l2 = ||z_fin - z_src||_2 per pair",
            "color_fiber_l2 = ||pal(z_fin) - pal(z_src)||_2",
            "swd_to_ref_before / swd_to_ref_after (style increment)",
            "idx_c / idx_r / sid_c / sid_r",
        ],
    }


# ---------------------------------------------------------------------------
# artifact loaders + report assembly
# ---------------------------------------------------------------------------
def _load_pack(path: Path) -> dict:
    import torch  # lazy: pure helpers stay torch-free

    if not path.exists():
        raise FileNotFoundError(f"missing local artifact: {path}")
    d = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "idx_c": np.asarray(d["idx_c"].numpy()),
        "idx_r": np.asarray(d["idx_r"].numpy()),
        "mid_cos": np.asarray(d["mid_cos"].numpy(), dtype=np.float64),
        "mean_all_cos": float(d["mean_all_cos"]),
        "mean_kept_cos": float(d["mean_kept_cos"]),
        "n_all_candidates": {"idx_c_shape": int(d["idx_c"].shape[0])},
    }


def _load_arm_rows() -> dict[str, tuple[dict, dict, dict]]:
    """(O3, label_free, self_only) rows per inference tier from the R3 json."""
    r3 = json.loads(R3_JSON.read_text(encoding="utf-8"))
    reused = r3["reused"]
    out: dict[str, tuple[dict, dict, dict]] = {}
    for tier in ("t45", "t55"):
        o3_key = f"O3_rgb_cnx_gated_{tier}"
        lf_key = f"O3_label_free_{tier}"
        if o3_key not in reused or lf_key not in reused:
            continue
        self_row = next(r for r in r3["self_only"] if r["arm"] == f"O3_self_only_{tier}")
        out[tier] = (reused[o3_key], reused[lf_key], self_row)
    return out


def composition_stats(mid_cos: np.ndarray, idx_c: np.ndarray, idx_r: np.ndarray) -> dict:
    """Does pair composition track edit magnitude inside the kept set?

    Reports same- vs cross-family ``mid_cos`` group stats, the point-biserial
    correlation, and per-content-family kept means.  Under the current
    conservative objective the groups are indistinguishable -- the objective
    compresses every composition to near-identity.
    """
    mc = np.asarray(mid_cos, dtype=np.float64)
    fc = family_of(np.asarray(idx_c))
    fr = family_of(np.asarray(idx_r))
    same = fc == fr
    cross = ~same
    by_fam = [float(mc[fc == s].mean()) for s in range(N_STYLES)]
    return {
        "n_same": int(same.sum()),
        "n_cross": int(cross.sum()),
        "mean_cos_same_family": float(mc[same].mean()),
        "mean_cos_cross_family": float(mc[cross].mean()),
        "welch_t_same_vs_cross": float(
            (mc[same].mean() - mc[cross].mean())
            / np.sqrt(mc[same].var() / same.sum() + mc[cross].var() / cross.sum())
        ),
        "point_biserial_r_same_family": pearson(same.astype(np.float64), mc),
        "mean_cos_by_content_family": dict(zip(STYLES, by_fam)),
    }


def analyze_pack(pack: dict, n_cand_total: int, label: str) -> dict:
    idx_c, idx_r, mid_cos = pack["idx_c"], pack["idx_r"], pack["mid_cos"]
    rej_mean = rejected_mean_cos(pack["mean_all_cos"], pack["mean_kept_cos"], n_cand_total, len(idx_c))
    mat = family_pair_matrix(idx_c, idx_r)
    kept_by_family = mat.sum(axis=1)
    # the share metric is only defined when the kept mean sits above the band
    # ceiling (V24 case); the V23 label teacher sits below it (opposite failure)
    if pack["mean_kept_cos"] > C1_BAND[1]:
        share = selection_share_of_excess(pack["mean_all_cos"], pack["mean_kept_cos"])
    else:
        share = None
    out = {
        "label": label,
        "n_kept": int(len(idx_c)),
        "n_candidates": int(n_cand_total),
        "mean_all_cos": pack["mean_all_cos"],
        "mean_kept_cos": pack["mean_kept_cos"],
        "rejected_mean_cos_exact": rej_mean,
        "selection_shift": pack["mean_kept_cos"] - pack["mean_all_cos"],
        "selection_share_of_excess_over_band_hi": share,
        "kept_cos_stats": kept_cos_stats(mid_cos),
        "same_family_fraction": same_family_fraction(idx_c, idx_r),
        "family_pair_matrix": mat.tolist(),
        "per_family_retention": retention_estimates(kept_by_family, n_cand_total),
    }
    return out


def main() -> int:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")

    report: dict = {"code_facts": {}, "packs": {}, "students": {}, "c1_parameter_block": c1_parameter_block()}

    # 1. V24 label-free reference teacher (kept 2000 of 2400)
    pack24 = _load_pack(PAIRS_V24)
    report["packs"]["v24_ref_teacher_label_free"] = analyze_pack(pack24, 2400, "V24 ref teacher (label-free)")

    # 2. V23 label-conditioned teacher (kept 1000 of 1500) as the amplitude contrast
    pack23 = _load_pack(PAIRS_V23)
    report["packs"]["v23_label_teacher"] = analyze_pack(pack23, 1500, "V23 label teacher (5-class prior)")
    report["packs"]["v23_label_teacher"]["metric_caveat"] = (
        "V23 mid_cos = ConvNeXt spatial s32/s16 cosine (exp/v23/build_cross_pairs_4090.py), "
        "V24 mid_cos = PatchSWD token s2/s3 cosine (exp/v24/o2_ref_teacher_pairs.py): both are "
        "ConvNeXt mid content retention vs source, same [0,1] direction, NOT bit-identical metrics"
    )

    # 3. student-level attribution chain (per available tier)
    arms_by_tier = _load_arm_rows()
    for tier, (o3_row, lf_row, self_row) in arms_by_tier.items():
        chain = attribution_chain(o3_row, lf_row, self_row)
        chain["teacher_mid_cos"] = {
            "self_only_identity": 1.0,
            "label_free_kept_mean": pack24["mean_kept_cos"],
            "label_teacher_kept_mean": pack23["mean_kept_cos"],
            "metric_caveat": "label teacher value uses the V23 metric variant (see packs caveat)",
        }
        report["students"][tier] = chain

    # 3b. does pair composition differentiate edit magnitude inside the kept set?
    report["packs"]["v24_ref_teacher_label_free"]["composition_stats"] = composition_stats(
        pack24["mid_cos"], pack24["idx_c"], pack24["idx_r"]
    )

    # 4. rejected-zone statement for the C1 band
    rej = report["packs"]["v24_ref_teacher_label_free"]
    report["code_facts"]["band_feasibility"] = {
        "c1_band": list(C1_BAND),
        "rejected_mean_cos": rej["rejected_mean_cos_exact"],
        "statement": "even the Top-K REJECTED 400 candidates average at/above the "
        "band ceiling: post-hoc selection alone cannot reach the C1 band; the "
        "objective (w_c) must change",
        "kept_frac_inside_band": rej["kept_cos_stats"]["frac_in_c1_band"],
    }

    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
