"""P0 dataset independence audit: prove val200 sources never appear in the latent training cache."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.v8.common import (  # noqa: E402
    STYLES, V8_ROOT, build_independent_val_tasks, ensure_dirs, load_d5_latents,
    save_json, test_source_stems, train_latent_stems,
)


def main() -> None:
    ensure_dirs()
    train, val, meta = load_d5_latents(device="cpu", heldout=100)
    train_ids = train_latent_stems()
    banned = test_source_stems()
    tasks = build_independent_val_tasks(per_style=10)

    val_stems = {t["source_stem"] for t in tasks}
    val_by_style = {}
    for t in tasks:
        val_by_style.setdefault(t["source_style"], set()).add(t["source_stem"])

    def variants(stem: str, style: str) -> set:
        short = stem.split("__", 1)[-1]
        return {stem, short, f"{style}__{short}", f"{style}__{stem}"}

    overlap = {}
    for s in STYLES:
        pool = train_ids[s]
        hit = set()
        for stem in val_by_style.get(s, set()):
            for v in variants(stem, s):
                if v in pool:
                    hit.add(v)
        overlap[s] = sorted(hit)

    # also record the heldout-100 slice used for internal validation
    heldout = {}
    for s in STYLES:
        heldout[s] = {
            "heldout_100_in_val_cache": 100,
            "train_cache_total": int(train[s].shape[0] + val[s].shape[0]),
            "train_used_for_training": int(train[s].shape[0]),
            "ids_pool_size": len(train_ids[s]),
        }

    out = {
        "vae": meta["vae"],
        "scaling_factor": meta["scaling_factor"],
        "frozen_test_source_stems": len(banned),
        "val200_tasks": len(tasks),
        "val200_unique_sources": len(val_stems),
        "val200_leaked_into_train_cache": {k: v for k, v in overlap.items()},
        "train_cache_sizes": heldout,
    }
    save_json(out, V8_ROOT / "p0_dataset_independence.json")

    total_leak = sum(len(v) for v in overlap.values())
    print(f"P0 dataset independence: val tasks={len(tasks)} sources={len(val_stems)} leaked={total_leak}")
    for s in STYLES:
        print(f"  {s}: cache_total={heldout[s]['train_cache_total']} train_used={heldout[s]['train_used_for_training']} "
              f"ids={heldout[s]['ids_pool_size']} val_sources={len(val_by_style.get(s, set()))} leak={len(overlap[s])}")
    assert total_leak == 0, overlap
    assert len(val_stems) > 0
    print("P0 DATASET INDEPENDENCE PASS")


if __name__ == "__main__":
    main()
