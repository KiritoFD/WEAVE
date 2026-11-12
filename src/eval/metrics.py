"""Metric enrichment, target specificity, target-switch response and statistics.

Pure functions over per-task rows. No model, GPU or filesystem dependency.
Bootstrap resamples SOURCE clusters with replacement and keeps duplicate draws.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping, Sequence

import numpy as np

STYLES = ("Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e")
QUALITY = ("dino_s", "dino_c", "lpips", "target_top1", "target_margin", "dT",
           "G", "clip_s", "style_affinity", "extra_specificity")


def _styles(styles=None):
    values = tuple(STYLES if styles is None else styles)
    if len(values) != 5 or len(set(values)) != 5:
        raise ValueError("exactly five unique style names in score order required")
    return values


def enrich_rows(rows: Iterable[Mapping], styles=None, *, require_full=False,
                expected_sources: int = 150) -> list[dict]:
    """Add G, clip_s and style_affinity; validate the score vectors."""
    styles = _styles(styles)
    out, ids, per_source = [], set(), defaultdict(set)
    for row in rows:
        r = dict(row)
        tid = r.get("task_id")
        if tid is None or tid in ids:
            raise ValueError(f"missing/duplicate task_id: {tid!r}")
        ids.add(tid)
        sv = r.get("style_scores_5")
        cv = r.get("clip_scores_5")
        if not isinstance(sv, (list, tuple)) or len(sv) != 5:
            raise ValueError("style_scores_5 must have five entries")
        if not isinstance(cv, (list, tuple)) or len(cv) != 5:
            raise ValueError("clip_scores_5 must have five entries")
        ti = styles.index(r["tgt_style"]) if r["tgt_style"] in styles else None
        if ti is None:
            raise ValueError(f"unknown target style {r['tgt_style']!r}")
        others = [v for j, v in enumerate(sv) if j != ti]
        r["G"] = float(sv[ti] - np.mean(others))
        r["extra_specificity"] = [float(sv[ti] - v) for j, v in enumerate(sv) if j != ti]
        r["clip_s"] = float(cv[ti])
        r["style_affinity"] = float((r["dino_s"] + r["clip_s"]) / 2.0)
        if abs(float(sv[ti]) - float(r["dino_s"])) > 1e-6:
            raise ValueError("dino_s must equal the requested target's style score")
        out.append(r)
        per_source[r["src_stem"]].add(r["tgt_style"])
    if require_full:
        if len(out) != 750:
            raise ValueError(f"full protocol needs 750 rows, got {len(out)}")
        if len(per_source) != expected_sources:
            raise ValueError(f"expected {expected_sources} sources, got {len(per_source)}")
        for stem, targets in per_source.items():
            if len(targets) != 5:
                raise ValueError(f"source {stem!r} does not have five targets")
        cross = sum(bool(r["is_cross"]) for r in out)
        if cross != 600:
            raise ValueError(f"Cross600 expected, got {cross}")
    return out


def target_switch_response(rows: Sequence[Mapping], styles=None) -> dict:
    """I = mean_t S_t(y_t) - mean_{u != t} S_t(y_u), per source then averaged.

    A model that ignores the target and returns the same image for all four
    cross targets scores exactly 0.
    """
    styles = _styles(styles)
    groups, src_style = defaultdict(list), {}
    for r in rows:
        src_style[r["src_stem"]] = r["src_style"]
        if r["is_cross"]:
            groups[r["src_stem"]].append(r)
    per_source, per_source_target = {}, {}
    for stem, src in src_style.items():
        group = groups[stem]
        expected = set(styles) - {src}
        if len(group) != 4 or {r["tgt_style"] for r in group} != expected:
            raise ValueError(f"target-switch I needs exactly four cross outputs for {stem!r}")
        contrasts = {}
        for r in group:
            j = styles.index(r["tgt_style"])
            others = [u["style_scores_5"][j] for u in group if u["tgt_style"] != r["tgt_style"]]
            contrasts[r["tgt_style"]] = float(r["style_scores_5"][j] - np.mean(others))
        per_source_target[stem] = contrasts
        per_source[stem] = float(np.mean(list(contrasts.values())))
    return {"mean": float(np.mean(list(per_source.values()))) if per_source else None,
            "n_sources": len(per_source), "per_source": per_source,
            "per_source_target": per_source_target}


def bootstrap_source_ci(values, clusters, *, n_boot: int = 4000, seed: int = 0,
                        draws=None) -> dict:
    """Mean CI by resampling sources with replacement (duplicates preserved)."""
    values = np.asarray(values, float)
    clusters = np.asarray(clusters, str)
    if values.ndim != 1 or clusters.shape != values.shape or len(values) == 0:
        raise ValueError("non-empty 1-D values and clusters of equal length required")
    if not np.isfinite(values).all():
        raise ValueError("bootstrap values must be finite")
    names, inv = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inv, weights=values, minlength=len(names))
    counts = np.bincount(inv, minlength=len(names))
    if draws is None:
        draws = np.random.default_rng(seed).integers(0, len(names), size=(n_boot, len(names)))
    picked = sums[draws].sum(1) / counts[draws].sum(1).clip(min=1)
    lo, hi = np.percentile(picked, [2.5, 97.5])
    return {"mean": float(values.mean()), "ci95": [float(lo), float(hi)],
            "n": int(len(values)), "n_clusters": int(len(names)),
            "excludes_zero": bool(lo > 0 or hi < 0)}


def summarize_rows(rows, styles=None, *, require_full=False, n_boot: int = 4000,
                   seed: int = 0, expected_sources: int = 150) -> dict:
    styles = _styles(styles)
    rows = enrich_rows(rows, styles, require_full=require_full,
                       expected_sources=expected_sources)
    def block(group):
        if not group:
            return {"n": 0, "means": {}, "source_ci": {}}
        return {"n": len(group),
                "means": {k: (float(np.mean([r[k] for r in group if k in r]))
                              if any(k in r for r in group) else None) for k in QUALITY},
                "source_ci": {k: bootstrap_source_ci([r[k] for r in group if k in r],
                                                     [r["src_stem"] for r in group if k in r],
                                                     n_boot=n_boot, seed=seed)
                              for k in ("G", "dino_s", "dino_c", "lpips")
                              if all(k in r for r in group)}}
    cross = [r for r in rows if r["is_cross"]]
    same = [r for r in rows if not r["is_cross"]]
    per_target = {s: block([r for r in cross if r["tgt_style"] == s]) for s in styles}
    routes = {f"{a}->{b}": block([r for r in cross
                                  if r["src_style"] == a and r["tgt_style"] == b])
              for a in styles for b in styles if a != b}
    switch = {"mean": None, "per_source": {}}
    try:
        switch = target_switch_response(rows, styles)
    except ValueError:
        pass
    confusion = None
    if all("pred_class" in r for r in cross):
        confusion = np.zeros((5, 5), int)
        for r in cross:
            confusion[styles.index(r["tgt_style"]), int(r["pred_class"])] += 1
        confusion = confusion.tolist()
    return {"n_total": len(rows), "n_cross": len(cross), "n_same": len(same),
            "style_order": list(styles), "cross600": block(cross),
            "same150": block(same), "per_target": per_target, "routes": routes,
            "target_switch_I": switch, "csd_confusion_matrix": confusion,
            "hard_styles": {s: per_target[s] for s in ("Early_Renaissance", "Ukiyo_e")
                            if s in per_target},
            "visual_review": "PENDING",
            "note": "all statistics are in-development-set paired results on "
                    "tasks_750; no blind-test claim"}
