"""Build a real-painting content-matched pair cache for WEAVE P3.

This creates ``z0/z1`` pairs where **both endpoints are real paintings**:
``z0`` is a source-style painting latent and ``z1`` is a DINO-content-matched
donor painting latent from the target style.  No pseudo-GT compositing and no
unpaired latent L2 is involved.

Inputs
------
- ``pseudo_gt_24h/retrieval/top8.json``: existing DINO retrieval table.
- ``pseudo_gt_24h/orig_mean_fp16_a0/*.pt``: A0-encoded real-painting latents.

Output
------
``pseudo_gt_24h/pairs/real_content_match`` with ``shard_*.pt`` and
``manifest.json`` compatible with ``utils.dataset.PairedLatentDataset``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.pseudo_gt.cache import SampleRecord, ShardWriter, verify_manifest  # noqa: E402

STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
STYLE_IDS = {s: i for i, s in enumerate(STYLES)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Build real-painting content-matched pair cache")
    ap.add_argument("--retrieval", default="pseudo_gt_24h/retrieval/top8.json")
    ap.add_argument("--latent-dir", default="pseudo_gt_24h/orig_mean_fp16_a0")
    ap.add_argument("--out-dir", default="pseudo_gt_24h/pairs/real_content_match")
    ap.add_argument("--per-route", type=int, default=100,
                    help="number of source->target pairs per directed style route")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--donor-rank", type=int, default=0,
                    help="0=best content match; use 1/2/... to choose later retrieval ranks")
    args = ap.parse_args()

    root = ROOT
    retrieval_path = root / args.retrieval
    latent_dir = root / args.latent_dir
    out_dir = root / args.out_dir
    if not retrieval_path.exists():
        raise FileNotFoundError(retrieval_path)
    if not latent_dir.exists():
        raise FileNotFoundError(latent_dir)

    retrieval = json.loads(retrieval_path.read_text(encoding="utf-8"))["styles"]
    latent_by_id: dict[str, torch.Tensor] = {}
    for path in sorted(latent_dir.glob("[0-9][0-9]_*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        ids = payload.get("ids") or []
        latents = payload.get("latents")
        if latents is None or len(ids) != latents.shape[0]:
            raise RuntimeError(f"bad latent pool file: {path}")
        for i, sid in enumerate(ids):
            latent_by_id[sid] = latents[i]
    print(f"[real-pairs] loaded {len(latent_by_id)} latents from {latent_dir.name}", flush=True)

    writer = ShardWriter(out_dir, shard_size=1000, dtype=torch.float16)
    completed = writer.completed_ids()
    summary = {}
    total_added = 0
    total_skipped = 0
    total_missing = 0
    rng = np.random.default_rng(args.seed)

    for src_style in STYLES:
        src_entry = retrieval.get(src_style)
        if not src_entry:
            continue
        source_ids = src_entry.get("source_ids", [])
        for tgt_style in STYLES:
            if tgt_style == src_style:
                continue
            donors_by_src = src_entry.get("by_target", {}).get(tgt_style, {}).get("donor_ids", [])
            scores_by_src = src_entry.get("by_target", {}).get(tgt_style, {}).get("donor_scores", [])
            if not donors_by_src:
                continue
            # Evenly cover the training pool instead of taking only the first N sources.
            pick_idx = np.unique(np.linspace(
                0, len(source_ids) - 1, min(args.per_route, len(source_ids))
            ).round().astype(int))
            added = skipped = missing = 0
            for k in pick_idx:
                src_id = source_ids[int(k)]
                if src_id not in latent_by_id:
                    missing += 1
                    continue
                donor_list = donors_by_src[int(k)] if int(k) < len(donors_by_src) else []
                score_list = scores_by_src[int(k)] if int(k) < len(scores_by_src) else []
                if not donor_list:
                    missing += 1
                    continue
                rank = min(args.donor_rank, len(donor_list) - 1)
                donor_id = donor_list[rank]
                score = float(score_list[rank]) if rank < len(score_list) else -1.0
                if donor_id not in latent_by_id:
                    missing += 1
                    continue
                sample_id = (
                    f"real_pair_content_cls_rank{rank}_{src_id}__{donor_id}__s{args.seed}"
                )
                if sample_id in completed:
                    skipped += 1
                    continue
                rec = SampleRecord(
                    sample_id=sample_id,
                    source_id=src_id,
                    donor_id=donor_id,
                    source_style=src_style,
                    target_style=tgt_style,
                    method="real_pair",
                    patch_spec="content_cls",
                    seed=args.seed,
                    match_score=score,
                    nominal_coverage=1.0,
                    effective_coverage=1.0,
                    z0_is_original=True,
                    z1_is_original=True,
                    band_energy={},
                    seam_grad=0.0,
                    effective_coverage_measured=1.0,
                    effective_coverage_weighted=1.0,
                )
                writer.add(
                    latent_by_id[src_id],
                    latent_by_id[donor_id],
                    rec,
                )
                completed.add(sample_id)
                added += 1
            summary[f"{src_style}->{tgt_style}"] = {
                "added": added, "skipped": skipped, "missing": missing,
            }
            total_added += added
            total_skipped += skipped
            total_missing += missing
            print(f"[real-pairs] {src_style}->{tgt_style}: "
                  f"added={added} skipped={skipped} missing={missing}", flush=True)
            writer.flush()

    writer.close()
    check = verify_manifest(out_dir)
    payload = {
        "out_dir": str(out_dir),
        "per_route": args.per_route,
        "donor_rank": args.donor_rank,
        "seed": args.seed,
        "total_added": total_added,
        "total_skipped": total_skipped,
        "total_missing": total_missing,
        "routes": summary,
        "verify": check,
    }
    (out_dir / "build_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"total_added": total_added, "total_skipped": total_skipped,
                      "total_missing": total_missing, "verify": check}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
