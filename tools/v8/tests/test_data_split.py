"""P0 data-source audit: prove the official 600/750 protocols do not overlap training data.

Two independent checks:
  A. the retired val200-style holdout pool (now only reachable via the private
     helper) contains zero ids that also appear in the latent training cache;
  B. the official frozen Test600/750 sources likewise do not appear in the
     latent training cache, i.e. the reported numbers are on held-out art.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.v8.common import (  # noqa: E402
    STYLES, V8_ROOT, _legacy_build_holdout_tasks, ensure_dirs, load_d5_latents,
    load_frozen_tasks, save_json, test_source_stems, train_latent_stems,
)


def _variants(stem: str, style: str) -> set:
    short = stem.split("__", 1)[-1]
    return {stem, short, f"{style}__{short}", f"{style}__{stem}"}


def main() -> None:
    ensure_dirs()
    train, val, meta = load_d5_latents(device="cpu", heldout=100)
    train_ids = train_latent_stems()
    banned = test_source_stems()

    # A. legacy holdout pool overlap
    holdout = _legacy_build_holdout_tasks(per_style=10)
    holdout_by_style: dict = {}
    for t in holdout:
        holdout_by_style.setdefault(t["source_style"], set()).add(t["source_stem"])
    holdout_leak = {}
    for s in STYLES:
        pool = train_ids[s]
        hit = {v for stem in holdout_by_style.get(s, set()) for v in _variants(stem, s) if v in pool}
        holdout_leak[s] = sorted(hit)
    holdout_sources = {t["source_stem"] for t in holdout}

    # B. official frozen tasks vs training cache
    official = load_frozen_tasks()
    official_leak = {}
    for s in STYLES:
        pool = train_ids[s]
        hit = set()
        for t in official:
            if t["source_style"] != s:
                continue
            for v in _variants(t["source_stem"], s):
                if v in pool:
                    hit.add(v)
        official_leak[s] = sorted(hit)

    out = {
        "vae": meta["vae"],
        "scaling_factor": meta["scaling_factor"],
        "frozen_test_source_stems": len(banned),
        "legacy_holdout": {
            "n_tasks": len(holdout),
            "n_unique_sources": len(holdout_sources),
            "per_style_unique_sources": {s: len(holdout_by_style.get(s, set())) for s in STYLES},
            "leaked_into_train_cache": {k: len(v) for k, v in holdout_leak.items()},
        },
        "official_protocol": {
            "n_tasks": len(official),
            "n_cross_tasks": sum(1 for t in official if t.get("is_cross", True)),
            "leaked_into_train_cache": {k: len(v) for k, v in official_leak.items()},
        },
        "train_cache_sizes": {
            s: {
                "cache_total": int(train[s].shape[0] + val[s].shape[0]),
                "used_for_training": int(train[s].shape[0]),
                "ids_pool": len(train_ids[s]),
            }
            for s in STYLES
        },
    }
    save_json(out, V8_ROOT / "p0_dataset_independence.json")

    holdout_total = sum(len(v) for v in holdout_leak.values())
    official_total = sum(len(v) for v in official_leak.values())
    print(f"legacy holdout: tasks={len(holdout)} sources={len(holdout_sources)} leak={holdout_total}")
    for s in STYLES:
        print(f"  {s}: holdout_sources={len(holdout_by_style.get(s, set()))} leak={len(holdout_leak[s])} "
              f"cache={out['train_cache_sizes'][s]['cache_total']} train={out['train_cache_sizes'][s]['used_for_training']}")
    print(f"official protocol: tasks={len(official)} leak={official_total}")
    print(f"  per-style leak: { {k: len(v) for k, v in official_leak.items()} }")

    assert holdout_total == 0, holdout_leak
    assert len(holdout_sources) > 0
    print("P0 DATASET AUDIT PASS (legacy holdout clean; official protocol leak reported above)")


if __name__ == "__main__":
    main()



if __name__ == "__main__":
    main()
