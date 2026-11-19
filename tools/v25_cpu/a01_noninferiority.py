"""A01: paired source-cluster bootstrap non-inferiority toolkit (V25 Slot A).

The statistical foundation is exclusively ``src/eval/metrics.py``
(``bootstrap_source_ci`` source-cluster resampling; ``paired_cluster_ci``
clustering logic). This module only adds extension wrappers: strict 600-task
pairing validation, one shared resampling-index matrix across metrics (to
preserve the joint bootstrap distribution), the pre-registered tolerance-band
three-state verdict, and eval.json loading. No statistical formula is
reinvented here.

Direction convention: every difference is ``simple - complex`` where *simple*
is the candidate-to-delete/simplify and *complex* the incumbent. The
pre-registered tolerance band (``docs/19_pull_together/01_convergence_plan.md``
section 2), in the ``simple - complex`` direction:

    dino_c >= -0.010   (higher is better)
    dino_s >= -0.005   (higher is better)
    lpips <= +0.010    (LOWER is better)

Three-state verdict (non-inferiority is NOT a significance test; CI crossing
zero is never used as a decision input):

    PASS          CI entirely inside the band -> complex component may be
                  deleted, subject to the worst-class Top-1 / same-blind guards.
    FAIL          complex significantly better beyond the band on some metric
                  -> keep the complex component.
    INCONCLUSIVE  CI overlaps a band edge without meeting either condition
                  -> default keep complex, flag as undecided.

CPU-only: numpy + stdlib. Callers must limit threads
(``OMP_NUM_THREADS=2 MKL_NUM_THREADS=2``). No CUDA, no torch API.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.eval.metrics import bootstrap_source_ci

REPO_ROOT = Path(__file__).resolve().parents[2]

#: pre-registered band metrics, in verdict priority order
BAND_METRICS = ("dino_c", "dino_s", "lpips")
#: (floor, ceiling) per metric in the simple-minus-complex direction; None = infinite
BAND = {"dino_c": (-0.010, None), "dino_s": (-0.005, None), "lpips": (None, 0.010)}

N_BOOT = 1500
SEED = 2026
CROSS_TASKS = 600
CROSS_SOURCES = 150
TASKS_PER_SOURCE = 4

#: plan section 2 delete-guard: worst single-class Top-1 drop must stay <= 3pp
WORST_CLASS_TOP1_DROP_MAX = 0.03
#: same-blind guard: same150 blind Top-1 must not degrade (0.5pp = 1 task of 150)
SAME_BLIND_TOP1_DROP_MAX = 0.005

VERDICT_PASS = "PASS"
VERDICT_FAIL = "FAIL"
VERDICT_INCONCLUSIVE = "INCONCLUSIVE"


# --------------------------------------------------------------------------- #
# eval.json loading and strict pairing
# --------------------------------------------------------------------------- #
def load_eval_rows(arm: str, version: str = "v25") -> list[dict]:
    """Load per-task rows from ``runs/<version>/measured/<arm>/eval.json``."""
    path = REPO_ROOT / "runs" / version / "measured" / arm / "eval.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing eval.json for arm {arm!r}: {path}")
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    rows = data.get("rows") if isinstance(data, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"eval.json for arm {arm!r} contains no task rows")
    return rows


def _finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _cross_by_task(rows: list[dict], name: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for row in rows:
        if not row.get("is_cross"):
            continue
        tid = row.get("task_id")
        if not tid:
            raise ValueError(f"{name}: cross row without task_id")
        if tid in out:
            raise ValueError(f"{name}: duplicate cross task_id {tid!r}")
        for metric in BAND_METRICS + ("target_top1",):
            if not _finite(row.get(metric)):
                raise ValueError(f"{name}: non-finite {metric} for task {tid!r}")
        out[tid] = row
    return out


def pair_cross_rows(
    rows_simple: list[dict],
    rows_complex: list[dict],
    *,
    simple_name: str = "simple",
    complex_name: str = "complex",
    expect_cross: int = CROSS_TASKS,
    expect_sources: int = CROSS_SOURCES,
    tasks_per_source: int = TASKS_PER_SOURCE,
) -> tuple[list[dict], list[dict], list[str]]:
    """Strictly pair two arms' cross tasks; raise on any count/set mismatch.

    Returns ``(rows_simple, rows_complex, clusters)`` aligned row-by-row in
    deterministic ``task_id`` order, ``clusters[i]`` being the source cluster
    (``src_stem``) of the i-th paired task.
    """
    a = _cross_by_task(rows_simple, simple_name)
    b = _cross_by_task(rows_complex, complex_name)
    for name, mapping in ((simple_name, a), (complex_name, b)):
        if len(mapping) != expect_cross:
            raise ValueError(
                f"{name}: expected {expect_cross} cross tasks, got {len(mapping)}"
            )
    only_simple = sorted(set(a) - set(b))
    only_complex = sorted(set(b) - set(a))
    if only_simple or only_complex:
        raise ValueError(
            "cross task_id sets differ between arms: "
            f"absent_in_{complex_name}={only_simple[:6]}... "
            f"absent_in_{simple_name}={only_complex[:6]}... "
            f"(n_absent_in_{complex_name}={len(only_simple)}, "
            f"n_absent_in_{simple_name}={len(only_complex)})"
        )
    ids = sorted(a)
    clusters = [str(a[tid]["src_stem"]) for tid in ids]
    for name, mapping in ((simple_name, a), (complex_name, b)):
        per_source: dict[str, int] = defaultdict(int)
        for tid in ids:
            per_source[str(mapping[tid]["src_stem"])] += 1
        if len(per_source) != expect_sources:
            raise ValueError(
                f"{name}: expected {expect_sources} source clusters, got {len(per_source)}"
            )
        bad = {s: n for s, n in per_source.items() if n != tasks_per_source}
        if bad:
            raise ValueError(
                f"{name}: cluster completeness violated (expected "
                f"{tasks_per_source} cross tasks per source): {dict(list(bad.items())[:6])}"
            )
    return [a[tid] for tid in ids], [b[tid] for tid in ids], clusters


# --------------------------------------------------------------------------- #
# shared resampling indices (foundation-compatible)
# --------------------------------------------------------------------------- #
def shared_bootstrap_draws(n_clusters: int, n_boot: int = N_BOOT, seed: int = SEED):
    """One resampling-index matrix shared by every metric of a comparison.

    Construction is bit-identical to the internal RNG of
    ``bootstrap_source_ci`` (``default_rng(seed).integers``), so passing these
    draws yields exactly the CI that ``draws=None`` with the same seed would
    produce -- while keeping each bootstrap replicate joint across metrics
    (the same re-drawn source set is applied to every metric).
    """
    return np.random.default_rng(seed).integers(0, n_clusters, size=(n_boot, n_clusters))


# --------------------------------------------------------------------------- #
# three-state non-inferiority verdict
# --------------------------------------------------------------------------- #
def noninferiority_verdict(ci_map: dict, band: dict | None = None) -> tuple[str, list[str]]:
    """Classify paired CIs (simple - complex) against the pre-registered band.

    ``ci_map``: ``{metric: (mean, lo, hi)}``; ``band``: ``{metric: (floor, ceiling)}``
    where ``None`` means infinite. Band edges are inclusive for PASS (``lo >=
    floor`` / ``hi <= ceiling``) and FAIL requires strictly beyond the far edge
    (``hi < floor`` for higher-is-better metrics, ``lo > ceiling`` for
    lower-is-better metrics such as LPIPS).
    """
    band = BAND if band is None else band
    if set(ci_map) != set(band):
        raise ValueError(
            f"verdict needs CIs for exactly the band metrics {sorted(band)}, got {sorted(ci_map)}"
        )
    reasons: list[str] = []
    in_band = True
    for metric, (floor, ceiling) in band.items():
        _, lo, hi = ci_map[metric]
        if floor is not None and lo < floor:
            in_band = False
        if ceiling is not None and hi > ceiling:
            in_band = False
    if in_band:
        return VERDICT_PASS, ["all paired CIs inside the pre-registered tolerance band"]
    for metric, (floor, ceiling) in band.items():
        _, lo, hi = ci_map[metric]
        if floor is not None and hi < floor:
            reasons.append(
                f"{metric}: CI upper {hi:.4f} < band floor {floor:.4f} "
                "(complex significantly better)"
            )
        if ceiling is not None and lo > ceiling:
            reasons.append(
                f"{metric}: CI lower {lo:.4f} > band ceiling {ceiling:.4f} "
                "(complex significantly better; LPIPS lower is better)"
            )
    if reasons:
        return VERDICT_FAIL, reasons
    return VERDICT_INCONCLUSIVE, [
        "CI overlaps a band edge without meeting the delete or the retain condition"
    ]


# --------------------------------------------------------------------------- #
# supplementary guards (plan section 2)
# --------------------------------------------------------------------------- #
def _guards(rows_simple: list[dict], rows_complex: list[dict]) -> dict:
    def per_class_top1(rows):
        agg: dict[str, list[float]] = defaultdict(list)
        for r in rows:
            if r.get("is_cross"):
                agg[r["tgt_style"]].append(float(r["target_top1"]))
        return {k: float(np.mean(v)) for k, v in agg.items()}

    def same_blind_top1(rows):
        vals = [float(r["target_top1"]) for r in rows if not r.get("is_cross")]
        return float(np.mean(vals)) if vals else None

    cs, cc = per_class_top1(rows_simple), per_class_top1(rows_complex)
    if set(cs) != set(cc):
        raise ValueError("per-target style sets differ between arms")
    drops = {k: cc[k] - cs[k] for k in cs}
    worst_style = max(drops, key=drops.get)
    worst_drop = drops[worst_style]
    s_same, c_same = same_blind_top1(rows_simple), same_blind_top1(rows_complex)
    same_drop = None if (s_same is None or c_same is None) else c_same - s_same
    return {
        "worst_class_style": worst_style,
        "worst_class_drop": worst_drop,
        "worst_class_ok": worst_drop <= WORST_CLASS_TOP1_DROP_MAX + 1e-12,
        "same_blind_top1_simple": s_same,
        "same_blind_top1_complex": c_same,
        "same_blind_drop": same_drop,
        "same_blind_ok": same_drop is None or same_drop <= SAME_BLIND_TOP1_DROP_MAX + 1e-12,
    }


# --------------------------------------------------------------------------- #
# full comparison pipeline
# --------------------------------------------------------------------------- #
def compare_arms(
    arm_simple: str,
    arm_complex: str,
    *,
    version_simple: str = "v25",
    version_complex: str = "v25",
    extra_metrics: tuple[str, ...] = ("target_top1",),
    n_boot: int = N_BOOT,
    seed: int = SEED,
    expect_cross: int = CROSS_TASKS,
    expect_sources: int = CROSS_SOURCES,
    tasks_per_source: int = TASKS_PER_SOURCE,
    rows_simple: list[dict] | None = None,
    rows_complex: list[dict] | None = None,
) -> dict:
    """Paired source-cluster bootstrap comparison of two measured arms.

    All metrics share one resampling-index matrix (joint distribution kept).
    Returns per-metric effect size + 95% CI, the three-state verdict on the
    pre-registered band, and the supplementary guards. ``rows_*`` allows
    injecting pre-loaded/synthetic rows instead of loading ``eval.json``.
    """
    rows_s = list(rows_simple) if rows_simple is not None else load_eval_rows(arm_simple, version_simple)
    rows_c = list(rows_complex) if rows_complex is not None else load_eval_rows(arm_complex, version_complex)
    aligned_s, aligned_c, clusters = pair_cross_rows(
        rows_s,
        rows_c,
        simple_name=arm_simple,
        complex_name=arm_complex,
        expect_cross=expect_cross,
        expect_sources=expect_sources,
        tasks_per_source=tasks_per_source,
    )
    n_clusters = len(set(clusters))
    draws = shared_bootstrap_draws(n_clusters, n_boot=n_boot, seed=seed)
    ci: dict[str, dict] = {}
    for metric in BAND_METRICS + tuple(extra_metrics):
        diffs = [float(x[metric]) - float(y[metric]) for x, y in zip(aligned_s, aligned_c)]
        res = bootstrap_source_ci(diffs, clusters, n_boot=n_boot, seed=seed, draws=draws)
        ci[metric] = {"mean": res["mean"], "lo": res["ci95"][0], "hi": res["ci95"][1]}
    verdict, reasons = noninferiority_verdict(
        {m: (ci[m]["mean"], ci[m]["lo"], ci[m]["hi"]) for m in BAND_METRICS}
    )
    guards = _guards(rows_s, rows_c)  # full row sets: guards need same150 + per-class coverage
    final = verdict
    if verdict == VERDICT_PASS and not (guards["worst_class_ok"] and guards["same_blind_ok"]):
        final = VERDICT_INCONCLUSIVE
        reasons = list(reasons)
        if not guards["worst_class_ok"]:
            reasons.append(
                f"guard violated: worst-class ({guards['worst_class_style']}) Top-1 drop "
                f"{guards['worst_class_drop'] * 100:.2f}pp > "
                f"{WORST_CLASS_TOP1_DROP_MAX * 100:.0f}pp"
            )
        if not guards["same_blind_ok"]:
            reasons.append(
                f"guard violated: same-blind Top-1 drop {guards['same_blind_drop'] * 100:.2f}pp"
            )
    return {
        "simple": arm_simple,
        "complex": arm_complex,
        "n_tasks": len(aligned_s),
        "n_clusters": n_clusters,
        "n_boot": n_boot,
        "seed": seed,
        "ci": ci,
        "verdict": verdict,
        "final": final,
        "reasons": reasons,
        "guards": guards,
    }


def format_comparison(res: dict, decision: str = "", confound: str = "") -> str:
    """Render one comparison as a markdown block (report paste-ready)."""
    lines = [
        f"### `{res['simple']}` vs `{res['complex']}`"
        + (f" -- {decision}" if decision else ""),
        "",
        f"n={res['n_tasks']} tasks / {res['n_clusters']} clusters, "
        f"n_boot={res['n_boot']}, seed={res['seed']}, direction = simple - complex",
        "",
        "| metric | mean | CI95 | in-band |",
        "|---|---:|---|---|",
    ]
    for metric in BAND_METRICS:
        floor, ceiling = BAND[metric]
        c = res["ci"][metric]
        lo_ok = floor is None or c["lo"] >= floor
        hi_ok = ceiling is None or c["hi"] <= ceiling
        band_txt = (">= %.3f" % floor) if floor is not None else ("<= +%.3f" % ceiling)
        ok = "yes" if (lo_ok and hi_ok) else "no"
        lines.append(
            f"| `{metric}` | {c['mean']:+.4f} | [{c['lo']:+.4f}, {c['hi']:+.4f}] "
            f"| {band_txt}: {ok} |"
        )
    for metric in res["ci"]:
        if metric in BAND_METRICS:
            continue
        c = res["ci"][metric]
        lines.append(
            f"| `{metric}` (suppl.) | {c['mean']:+.4f} | [{c['lo']:+.4f}, {c['hi']:+.4f}] | - |"
        )
    g = res["guards"]
    same = "-" if g["same_blind_drop"] is None else f"{g['same_blind_drop'] * 100:+.2f}pp"
    lines += [
        "",
        f"**verdict: {res['verdict']} -> final: {res['final']}**",
        "",
        f"- reasons: {'; '.join(res['reasons'])}",
        f"- guards: worst-class ({g['worst_class_style']}) Top-1 drop "
        f"{g['worst_class_drop'] * 100:+.2f}pp "
        f"({'ok' if g['worst_class_ok'] else 'VIOLATED'}); "
        f"same-blind Top-1 drop {same} "
        f"({'ok' if g['same_blind_ok'] else 'VIOLATED'})",
    ]
    if confound:
        lines.append(f"- confound: {confound}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Round 1 comparison plan (docs/19_pull_together/04_round1_results.md section 6)
# --------------------------------------------------------------------------- #
ROUND1_COMPARISONS = [
    {"simple": "P02_d6h_csj", "complex": "P02_d6h_full",
     "decision": "delete full CFG branch (40 -> 20 NFE)"},
    {"simple": "P14_d6h_fixed_g50", "complex": "P02_d6h_csj",
     "decision": "delete sigmoid density gate (fixed g=0.5)"},
    {"simple": "P02_d5_csj_ws42", "complex": "P02_d5_full",
     "decision": "delete full CFG branch at the D5 water level"},
    {"simple": "P13_ancp1_t040", "complex": "P02_d6h_csj",
     "decision": "substitute T4 anchor with PRISM-1 anchor",
     "confound": "tau_min 0.40 (simple) vs 0.48 (complex) not aligned; "
                 "water levels differ (dino_c 0.6691 vs 0.6782)"},
    {"simple": "V25_Tr_T1_d5csj", "complex": "V25_Tr_T0_d5csj",
     "decision": "recipe T1 (drop_c 30%) vs T0 at the D5 cs_j water level"},
    {"simple": "V25_Tr_T1_d6hcsj", "complex": "V25_Tr_T0_d6hcsj",
     "decision": "recipe T1 (drop_c 30%) vs T0 at the D6h cs_j water level"},
    {"simple": "B_d6_t066", "complex": "C_hyb_g70",
     "decision": "SENSITIVITY ONLY, not a decision input: nearest measured "
                 "single-model D6 arm vs nearest measured Hybrid arm",
     "confound": "water levels NOT aligned (dino_c 0.7007 vs 0.7129); "
                 "the pre-registered D6-vs-Hybrid question is at dino_c=0.73 "
                 "where no arm is measured -> stays undecided"},
]


def run_round1() -> None:
    """Run every locally computable Round 1 comparison and print report blocks."""
    for comp in ROUND1_COMPARISONS:
        res = compare_arms(
            comp["simple"], comp["complex"],
            version_simple=comp.get("version_simple", "v25"),
            version_complex=comp.get("version_complex", "v25"),
        )
        print(format_comparison(res, comp.get("decision", ""), comp.get("confound", "")))
        print()
    print("## Undecided entries (no computation, no extrapolated pairing)")
    print()
    print("- `D6 vs Hybrid @ dino_c=0.73`: UNDECIDED. Measured arms cover "
          "dino_c <= 0.7007 (B_d6_t066) and <= 0.7129 (C_hyb_g70); at 0.73 at "
          "least one arm has no per-task measurements. Mean extrapolation to "
          "fabricate paired samples is forbidden (agent2.md A01 acceptance); "
          "resolving it needs a GPU arm at tau_min ~ 0.72 reaching dino_c ~ 0.73.")
    print()
    print("- `V25_Tr_T3 vs V25_Tr_T0`: UNDECIDED (not computed). "
          "`runs/v25/measured/` contains no `V25_Tr_T3_*` eval.json yet "
          "(Round 2 training matrix not fully evaluated). Recompute with "
          "`compare_arms` once the GPU queue produces it.")
    print()
    print("- `V25_Tr_T2 vs V25_Tr_T0` was NOT in the assigned list; T2 arms "
          "exist and the same tool can score them on demand.")


if __name__ == "__main__":
    run_round1()
