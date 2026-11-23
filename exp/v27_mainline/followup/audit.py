"""Frozen-prototype CSLS sensitivity and filename-derived artist overlap audit."""

import json

import numpy as np
import torch

from exp.v18 import canonical_pid
from exp.v27_mainline import OUT, ROOT
from exp.v27_mainline.followup.frozen import RESULTS
from src.eval import paired_cluster_ci
from src.utils._local_eval_refs import csd_prototypes
from src.utils.runtime import save_json


def artist(stem):
    normalized = canonical_pid(stem)
    name, sep, _ = normalized.partition("_")
    return name if sep else None


def artists(stems):
    return {a for stem in stems if (a := artist(stem))}


def main():
    protocol = ROOT / "eval_protocol"
    tasks = json.loads((protocol / "tasks_750.json").read_text(encoding="utf-8"))["tasks"]
    refs = json.loads((protocol / "references_v1.json").read_text(encoding="utf-8"))["references"]
    refs_stems = [r["stem"] for rows in refs.values() for r in rows]
    source_stems = [t["source_stem"] for t in tasks]
    source_artists = artists(source_stems)
    ref_artists = artists(refs_stems)
    train = torch.load(ROOT / "runs/v18/cache/ccf_pool_cache.pt", map_location="cpu", weights_only=False)["train"]
    train_artists = artists(train["ids"])
    source = torch.load(ROOT / "runs/v6_r2/cache/tasks750_source_features.pt", map_location="cpu", weights_only=False)
    proto = csd_prototypes()
    background = np.stack([source["raw_csd"][stem] for stem in sorted(set(source_stems))])
    similarity = background @ proto.T
    report = {
        "artist_identity": "filename prefix heuristic; overlaps are candidates",
        "artist_overlap": {
            "test_source_artists": len(source_artists),
            "test_reference_artists": len(ref_artists),
            "train_artists": len(train_artists),
            "source_with_reference": sorted(source_artists & ref_artists),
            "source_with_train": sorted(source_artists & train_artists),
            "reference_with_train": sorted(ref_artists & train_artists),
            "cross_tasks_source_artist_unseen_train": sum(
                t["source_style_id"] != t["target_style_id"] and artist(t["source_stem"]) not in train_artists
                for t in tasks
            ),
        },
        "csls_background": "fixed 150 raw test sources; shared across arms, prototype ranking sensitivity only",
        "arms": {},
        "paired_deltas": {},
    }
    by_arm = {}
    for arm in sorted((OUT / "measured").glob("followup_*")):
        path = arm / "eval.json"
        if not path.exists():
            continue
        rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
        rows = [r for r in rows if r["is_cross"]]
        if len(rows) != 600:
            continue
        by_arm[arm.name] = rows
        x = np.array([r["csd_scores_5"] for r in rows])
        y = np.array([r["tgt_style_id"] for r in rows])
        mask = np.array([artist(r["src_stem"]) not in train_artists for r in rows])
        score = {"raw": float((x.argmax(1) == y).mean())}
        for k in (5, 10, 20):
            r_proto = np.sort(similarity, axis=0)[-k:].mean(0)
            # Query density cancels in the ranking for a fixed query.
            pred = (2 * x - r_proto).argmax(1)
            score[f"csls_k{k}"] = float((pred == y).mean())
            if k == 10:
                score["csls_k10_er_uki"] = {str(i): float((pred[y == i] == i).mean()) for i in (0, 4)}
        if mask.any():
            score["raw_source_artist_unseen_train"] = float((x.argmax(1)[mask] == y[mask]).mean())
        report["arms"][arm.name] = score
    comparisons = [
        ("followup_a_grid", "followup_a_global"),
        ("followup_a_start", "followup_a_global"),
        ("followup_a_clock", "followup_a_start"),
    ]
    for name in by_arm:
        if name.startswith("followup_b_aligned_"):
            comparisons.extend((name, name.replace("aligned", control))
                               for control in ("teacher", "shuffled"))
    for a, b in comparisons:
        if a not in by_arm or b not in by_arm:
            continue
        report["paired_deltas"][f"{a} minus {b}"] = {
            "cross600": paired_cluster_ci(by_arm[a], by_arm[b],
                                          metrics=("target_top1", "dino_c", "dino_s", "lpips")),
            "er_uki_240": paired_cluster_ci(
                [r for r in by_arm[a] if r["tgt_style_id"] in (0, 4)],
                [r for r in by_arm[b] if r["tgt_style_id"] in (0, 4)],
                metrics=("target_top1", "dino_c", "lpips"),
            ),
        }
    save_json(RESULTS / "evaluation_audit.json", report)


if __name__ == "__main__":
    main()
