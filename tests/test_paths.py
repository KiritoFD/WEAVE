"""paths.resolve / resolve_config_path 的最小契约测试。"""
from __future__ import annotations

from pathlib import Path

import paths


def test_resolve_relative_joins_project_root() -> None:
    assert paths.resolve("data/train") == paths.WEAVE_ROOT / "data" / "train"
    assert paths.resolve(Path("runs/v4")) == paths.WEAVE_ROOT / "runs" / "v4"


def test_resolve_empty_returns_base() -> None:
    assert paths.resolve("") == paths.WEAVE_ROOT
    assert paths.resolve(None, base=paths.ASSET_ROOT) == paths.ASSET_ROOT


def test_resolve_keeps_absolute_path_unchanged(tmp_path: Path) -> None:
    absolute = tmp_path / "weights"
    assert paths.resolve(absolute) == absolute
    assert paths.resolve(str(absolute)) == absolute


def test_resolve_config_path_relative_is_project_root_based() -> None:
    assert paths.resolve_config_path("data/train") == paths.WEAVE_ROOT / "data" / "train"
    assert paths.resolve_config_path("datasets/distinct5_512/train") == (
        paths.WEAVE_ROOT / "datasets" / "distinct5_512" / "train"
    )


def test_resolve_config_path_empty_stays_unset() -> None:
    assert paths.resolve_config_path("") == ""
    assert paths.resolve_config_path(None) == ""


def test_resolve_dataset_path_keeps_dot_relative_legacy() -> None:
    assert paths.resolve_dataset_path(".latent_cache/packed") == ".latent_cache/packed"
    assert paths.resolve_dataset_path("data/train") == paths.WEAVE_ROOT / "data" / "train"
    assert paths.resolve_dataset_path("") == ""


def test_resolve_config_path_keeps_absolute_during_migration(tmp_path: Path) -> None:
    absolute = tmp_path / "legacy" / "model.pt"
    assert paths.resolve_config_path(absolute) == absolute
    assert paths.resolve_config_path(str(absolute)) == absolute
