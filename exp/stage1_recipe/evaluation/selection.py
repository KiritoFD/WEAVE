"""CPU-only matched-content selection and paired source-cluster confidence bounds."""
from __future__ import annotations

import argparse
from pathlib import Path

from exp.stage1_recipe.config import read, write


def _records(path: Path, scope: str):
    d = read(path)
    rows = d["rows"]
    if scope == "core750":
        if len(rows) != 750 or (d["n_total"], d["n_cross"], d["n_same"]) != (750, 600, 150):
            raise ValueError(f"Incomplete formal task set: {path}")
        if len({r["task_id"] for r in rows}) != 750:
            raise ValueError("Duplicate task_id in formal evaluation")
        cross = [r for r in rows if r["is_cross"]]
        same = [r for r in rows if not r["is_cross"]]
        if len(cross) != 600 or len(same) != 150:
            raise ValueError("Declared Cross600/Same150 counts differ from task rows")
        grouped = {}
        for row in rows:
            grouped.setdefault(str(row["src_stem"]), []).append(row)
        if len(grouped) != 150 or any(len(v) != 5 or
                {int(r["tgt_style_id"]) for r in v} != set(range(5)) or
                sum(bool(r["is_cross"]) for r in v) != 4 for v in grouped.values()):
            raise ValueError("Formal evaluation needs five unique targets and four Cross tasks per source")
        rows = [r for r in rows if r["is_cross"]]
    else:
        expected = {"artist_ood": 1000, "movement_ood": 500}.get(scope, d.get("n"))
        if not expected or len(rows) != expected or d.get("n") != expected:
            raise ValueError(f"Incomplete {scope} row count")
        if len({str(r["source_id"]) for r in rows}) != expected or len({r["task_id"] for r in rows}) != expected:
            raise ValueError("Instance-reference source/task IDs must be unique")
        for row in rows:
            if any(row.get(k) is None for k in ("reference_id", "wrong_reference_id")) or str(
                    row["reference_id"]) == str(row["wrong_reference_id"]):
                raise ValueError("Instance evaluation needs distinct declared reference/wrong-reference IDs")
    return rows


def _pairing(rows, scope):
    if scope == "core750":
        return sorted((str(r["task_id"]), str(r["src_stem"]), int(r["tgt_style_id"]),
                       str(r.get("reference_id")), str(r.get("wrong_reference_id"))) for r in rows)
    return sorted((str(r["task_id"]), str(r["source_id"]), str(r["reference_id"]),
                   str(r["wrong_reference_id"])) for r in rows)


def _assert_paired(candidate_paths, baseline_paths, scope):
    if not candidate_paths or not baseline_paths:
        raise ValueError("Candidate and baseline curves are both required")
    if _pairing(_records(candidate_paths[0], scope), scope) != _pairing(_records(baseline_paths[0], scope), scope):
        raise ValueError("Candidate/control task, source, target or reference pairing differs")


def curve(paths: list[Path], scope: str):
    import numpy as np
    source_ids = None
    curves = []
    identity = None
    metric = "target_top1" if scope == "core750" else "csd_2way_follow"
    for path in paths:
        rows = _records(path, scope)
        current = _pairing(rows, scope)
        if identity is not None and current != identity:
            raise ValueError("Curve points must retain task/source/target/reference identity")
        identity = current
        grouped = {}
        for row in rows:
            key = str(row["src_stem"] if scope == "core750" else row["source_id"])
            grouped.setdefault(key, []).append([row["dino_c"], row["dino_s"], row[metric]])
        ids = sorted(grouped)
        if source_ids is not None and ids != source_ids:
            raise ValueError("Curve points must use exactly the same source clusters")
        source_ids = ids
        curves.append(np.array([np.mean(grouped[s], axis=0) for s in ids]))
    if not curves:
        raise ValueError("Empty observed curve")
    return np.stack(curves), source_ids


def interpolate(points, content):
    import numpy as np
    points = points[np.argsort(points[:, 0])]
    if len(points) < 2 or not points[0, 0] <= content <= points[-1, 0]:
        return None
    # Exact same C coordinates are merged, rather than selecting an arbitrary knob.
    cs = np.unique(points[:, 0])
    if len(cs) < 2:
        return None
    vals = np.stack([points[points[:, 0] == c, 1:].mean(0) for c in cs])
    return np.array([np.interp(content, cs, vals[:, j]) for j in range(vals.shape[1])])


def matched_ci(candidate_paths, baseline_paths, *, scope, anchors, draws=2000, seed=20263800,
               minimum_valid=1000):
    import numpy as np
    _assert_paired(candidate_paths, baseline_paths, scope)
    cand, ids = curve(candidate_paths, scope)
    base, bids = curve(baseline_paths, scope)
    if ids != bids:
        raise ValueError("Candidate and control source IDs differ")
    expected = {"core750": 150, "artist_ood": 1000, "movement_ood": 500}
    if scope in expected and len(ids) != expected[scope]:
        raise ValueError(f"Incomplete {scope}: {len(ids)} clusters")
    rng = np.random.default_rng(seed)
    boot = rng.integers(0, len(ids), (draws, len(ids)))
    result = []
    for c in anchors:
        a, b = interpolate(cand.mean(1), c), interpolate(base.mean(1), c)
        row = {"content": c, "scope": scope, "n_sources": len(ids), "draws": draws}
        if a is None or b is None:
            result.append({**row, "status": "outside_observed_bracket"})
            continue
        samples = []
        for idx in boot:
            aa, bb = interpolate(cand[:, idx].mean(1), c), interpolate(base[:, idx].mean(1), c)
            if aa is not None and bb is not None:
                samples.append(aa - bb)
        valid = len(samples)
        row.update(valid_draws=valid, delta_s=float(a[0] - b[0]), delta_top1_pp=float(100 * (a[1] - b[1])),
                   candidate_s=float(a[0]), baseline_s=float(b[0]))
        if valid < minimum_valid:
            row["status"] = "insufficient_valid_bootstrap"
        else:
            ci = np.quantile(samples, [.025, .975], axis=0)
            row.update(status="ok", s_ci95=ci[:, 0].tolist(), top1_pp_ci95=(100 * ci[:, 1]).tolist())
        result.append(row)
    return result


def seen_score(candidate_paths, baseline_paths, anchors):
    """Selection only on Seen-Dev instance metrics, never interpreted as formal Top-1."""
    import numpy as np
    _assert_paired(candidate_paths, baseline_paths, "seen_dev")
    a, ids = curve(candidate_paths, "seen_dev")
    b, bids = curve(baseline_paths, "seen_dev")
    if ids != bids:
        raise ValueError("Seen-Dev pairing mismatch")
    gains = []
    for c in anchors:
        aa, bb = interpolate(a.mean(1), c), interpolate(b.mean(1), c)
        if aa is not None and bb is not None:
            gains.append({"content": c, "delta_s": float(aa[0] - bb[0]),
                          "delta_2way_pp": float((aa[1] - bb[1]) * 100)})
    return {"matched": gains, "score": float(np.mean([r["delta_s"] for r in gains])) if gains else None,
            "eligible": len(gains) >= 2 and sum(r["delta_s"] >= .005 for r in gains) >= 2,
            "selection_scope": "seen_dev_instance_reference_only"}


def same_ci(candidate_path: Path, baseline_path: Path, draws=2000, seed=20263800):
    import numpy as np
    _assert_paired([candidate_path], [baseline_path], "core750")
    def values(p):
        d = read(p)
        assert (d["n_total"], d["n_cross"], d["n_same"]) == (750, 600, 150)
        return {r["task_id"]: float(r["target_top1"]) for r in d["rows"] if not r["is_cross"]}
    a, b = values(candidate_path), values(baseline_path)
    if len(a) != 150 or set(a) != set(b):
        raise ValueError("Same150 pairing incomplete")
    diff = np.array([a[k] - b[k] for k in sorted(a)]) * 100
    idx = np.random.default_rng(seed).integers(0, 150, (draws, 150))
    return {"delta_pp": float(diff.mean()), "ci95": np.quantile(diff[idx].mean(1), [.025, .975]).tolist()}


def acceptance(core, same, ood, visual, cfg):
    valid = [r for r in core if r["status"] == "ok"]
    style_wins = [r for r in valid if r["delta_s"] >= cfg["s_gain"] and r["s_ci95"][0] > 0]
    top_wins = [r for r in valid if r["delta_top1_pp"] >= cfg["top1_gain_pp"]
                and r["top1_pp_ci95"][0] > 0 and r["s_ci95"][0] >= cfg["s_noninferiority"]]
    same_pass = bool(same) and all(r["ci95"][0] >= cfg["same_noninferiority_pp"] for r in same)
    ood_pass = True
    for split in ("artist_ood", "movement_ood"):
        rs = [r for r in ood.get(split, []) if r["status"] == "ok"]
        # At least one bracketed content anchor per OOD split is required.
        ood_pass &= bool(rs) and all(r["s_ci95"][0] >= cfg["s_noninferiority"] and
                                    r["top1_pp_ci95"][0] >= cfg["ood_2way_noninferiority_pp"] for r in rs)
    visual_pass = bool(visual.get("evidence_validated")) and all(visual.get(k) is True for k in (
        "gray_style_improved", "color_matched_improved", "no_new_artifacts"))
    quality = len(style_wins) >= 2 or len(top_wins) >= 1
    return {"adopt": bool(quality and same_pass and ood_pass and visual_pass),
            "core_gain": quality, "same_noninferior": same_pass, "ood_noninferior": bool(ood_pass),
            "visual_supported": visual_pass, "style_winning_anchors": len(style_wins),
            "top1_winning_anchors": len(top_wins), "no_extrapolation": True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate", nargs="+", type=Path, required=True)
    p.add_argument("--baseline", nargs="+", type=Path, required=True)
    p.add_argument("--scope", choices=("core750", "artist_ood", "movement_ood", "seen_dev"), required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    write(a.output, matched_ci(a.candidate, a.baseline, scope=a.scope, anchors=[.60, .65, .70, .75]))


if __name__ == "__main__":
    main()
