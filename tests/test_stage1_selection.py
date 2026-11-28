"""CPU-only guards for selected weights, frozen provenance and paired task identity."""
import copy
from pathlib import Path

import pytest

from exp.stage1_recipe.evaluation import generate as gen
from exp.stage1_recipe.evaluation import selection as sel
from exp.stage1_recipe.config import write


def test_explicit_online_and_ema_never_silently_fall_back():
    online, ema995, ema999 = {"w": 1}, {"w": 2}, {"w": 3}
    ck = {"online": online, "emas": {"0.995": ema995, "0.999": ema999}}
    assert gen._state(ck, "ecore", "online") is online
    assert gen._state(ck, "ecore", ".999") is ema999
    with pytest.raises(KeyError, match="absent"):
        gen._state(ck, "ecore", ".9995")
    with pytest.raises(KeyError, match="Explicit"):
        gen._state(ck, "ecore", None)
    assert gen._state({"ecore": online, "ecore_ema": ema995}, "ecore", None) is ema995
    assert gen._state({"ecore": online, "ecore_ema": ema995}, "ecore", "online") is online
    assert gen._state({"ecore_ema": ema995, "ema_decay": .995}, "ecore", ".995") is ema995


def test_provenance_changes_for_weights_config_seed_and_reference(tmp_path, monkeypatch):
    monkeypatch.setattr(gen, "ROOT", tmp_path)
    cache = tmp_path / "runs/v36_art_foundation/cache"
    cache.mkdir(parents=True)
    for name in ("v36_seen_train_priors.pt", "v36_caches_ready.json", "curation_manifest.json"):
        (cache / name).write_bytes(b"fixture")
    ck = tmp_path / "weights.pt"
    ck.write_bytes(b"first")
    config = {"paths": {"cache_root": str(cache)}, "evaluation": {"batch": 4, "steps": 6}}
    job = {"split": "seen_dev", "p_checkpoint": str(ck)}
    pairing = {"source_ids": [1], "reference_ids": [2], "wrong_reference_ids": [3]}
    initial = gen.generation_provenance(config, job, pairing, 42)
    gen.validate_provenance(initial, initial)
    assert initial == gen.generation_provenance(config, job, pairing, 42)
    assert gen.generation_provenance(config, job, pairing, 43)["signature"] != initial["signature"]
    changed = copy.deepcopy(config)
    changed["evaluation"]["batch"] = 8
    assert gen.generation_provenance(changed, job, pairing, 42)["signature"] != initial["signature"]
    assert gen.generation_provenance(config, job, {**pairing, "reference_ids": [4]}, 42)["signature"] != initial["signature"]
    ck.write_bytes(b"replacement weights")
    updated = gen.generation_provenance(config, job, pairing, 42)
    with pytest.raises(ValueError, match="Stale"):
        gen.validate_provenance(updated, initial)
    with pytest.raises(ValueError, match="canonical"):
        gen.require_canonical_cache({"paths": {"cache_root": str(tmp_path / "other_cache")}})


def test_formal_task_mapping_and_ood_duplicates_are_rejected(tmp_path):
    rows = [{"task_id": f"s{i}_t{t}", "src_stem": f"s{i}", "tgt_style_id": t,
             "is_cross": t != i % 5, "dino_c": .6, "dino_s": .5, "target_top1": 1}
            for i in range(150) for t in range(5)]
    base = {"n_total": 750, "n_cross": 600, "n_same": 150, "rows": rows}
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    write(a, base)
    changed = copy.deepcopy(base)
    changed["rows"][0]["task_id"], changed["rows"][1]["task_id"] = changed["rows"][1]["task_id"], changed["rows"][0]["task_id"]
    write(b, changed)
    with pytest.raises(ValueError, match="pairing"):
        sel._assert_paired([a], [b], "core750")
    ood = {"n": 1000, "rows": [{"task_id": str(i), "source_id": str(i),
            "reference_id": "r", "wrong_reference_id": "w"} for i in range(1000)]}
    ood["rows"][-1]["source_id"] = "0"
    write(b, ood)
    with pytest.raises(ValueError, match="unique"):
        sel._records(b, "artist_ood")


def test_seen_paired_ci_works_and_changed_reference_is_rejected(tmp_path):
    baseline, candidate = [], []
    for arm, paths, gain in (("base", baseline, 0.), ("candidate", candidate, .02)):
        for k, c in enumerate((.5, .8)):
            path = tmp_path / f"{arm}_{k}.json"
            rows = [{"task_id": f"seen_dev:{i}", "source_id": str(i), "reference_id": "r",
                     "wrong_reference_id": "w", "dino_c": c, "dino_s": .5 + gain,
                     "csd_2way_follow": .8} for i in range(6)]
            write(path, {"n": 6, "rows": rows})
            paths.append(path)
    result = sel.matched_ci(candidate, baseline, scope="seen_dev", anchors=[.6], draws=25, minimum_valid=10)
    assert result[0]["status"] == "ok"
    assert result[0]["delta_s"] == pytest.approx(.02)
    assert result[0]["s_ci95"][0] == pytest.approx(.02)
    changed = sel.read(candidate[0])
    changed["rows"][0]["reference_id"] = "other"
    write(candidate[0], changed)
    with pytest.raises(ValueError, match="pairing"):
        sel.seen_score(candidate, baseline, [.6])
