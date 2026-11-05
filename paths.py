"""统一路径解析：代码根（WEAVE_ROOT）与资产根（WEAVE_ASSET_ROOT）。

约定：
- 代码/配置/产物里的相对路径，一律相对项目根 ``WEAVE_ROOT``；
- 数据集、权重、评测缓存等大资产允许放在项目根同级（``WEAVE_ASSET_ROOT``）；
- 环境变量优先，便于本地 Windows 与 3060/WSL 共用同一份代码。
"""
from __future__ import annotations

import os
from pathlib import Path

WEAVE_ROOT = Path(
    os.environ.get("WEAVE_ROOT", "").strip() or Path(__file__).resolve().parent
).resolve()
ASSET_ROOT = Path(
    os.environ.get("WEAVE_ASSET_ROOT", "").strip() or WEAVE_ROOT.parent
).resolve()


def project_root() -> Path:
    """代码仓库根目录。"""
    return WEAVE_ROOT


def asset_root() -> Path:
    """大资产根目录（默认是项目根的父目录）。"""
    return ASSET_ROOT


def resolve(raw: str | Path | None, *, base: Path = WEAVE_ROOT) -> Path:
    """相对路径拼到 ``base``；绝对路径原样返回；空值等价于 ``base``。"""
    text = str(raw or "").strip()
    if not text:
        return Path(base)
    path = Path(text)
    return path if path.is_absolute() else Path(base) / path


def resolve_config_path(raw: str | Path | None) -> Path | str:
    """配置字段专用：``@asset/...`` 相对资产根，其余相对 ``WEAVE_ROOT``。"""
    text = str(raw or "").strip()
    if not text:
        return ""
    if text.startswith("@asset/"):
        return ASSET_ROOT / text[len("@asset/") :]
    path = Path(text)
    return path if path.is_absolute() else WEAVE_ROOT / path


def resolve_dataset_path(raw: str | Path | None) -> Path | str:
    """数据集字段专用：以 ``.`` 开头的历史写法仍相对 ``data_root``，其余相对项目根。"""
    text = str(raw or "").strip()
    if not text:
        return ""
    if text.startswith("."):
        return text
    return resolve_config_path(text)


def rel(path: str | Path, *, base: Path = WEAVE_ROOT) -> str:
    """写成 POSIX 相对路径（用于 JSON / manifest / 日志）。"""
    return Path(path).resolve().relative_to(Path(base).resolve()).as_posix()


def as_str(path: str | Path) -> str:
    """按当前平台分隔符转字符串（用于 subprocess / 日志）。"""
    return str(Path(path))


def _env_root(name: str, default: Path) -> Path:
    value = os.environ.get(name, "").strip()
    return Path(value) if value else default


def data_root(*parts: str) -> Path:
    return _env_root("WEAVE_DATA_ROOT", WEAVE_ROOT / "data").joinpath(*parts)


def datasets_root(*parts: str) -> Path:
    return _env_root("WEAVE_DATASETS_ROOT", WEAVE_ROOT / "datasets").joinpath(*parts)


def runs_root(*parts: str) -> Path:
    return _env_root("WEAVE_RUNS_ROOT", WEAVE_ROOT / "runs").joinpath(*parts)


def results_root(*parts: str) -> Path:
    return _env_root("WEAVE_RESULTS_ROOT", WEAVE_ROOT / "results").joinpath(*parts)


def weights_root(*parts: str) -> Path:
    return _env_root("WEAVE_WEIGHTS_ROOT", WEAVE_ROOT / "weights").joinpath(*parts)


def eval_cache_root(*parts: str) -> Path:
    return _env_root("WEAVE_EVAL_CACHE_ROOT", ASSET_ROOT / "eval_cache").joinpath(*parts)


def dino_cache_root(*parts: str) -> Path:
    return _env_root("WEAVE_DINO_CACHE_ROOT", ASSET_ROOT / "dino_cache").joinpath(*parts)


def hf_cache_root(*parts: str) -> Path:
    return eval_cache_root("hf").joinpath(*parts)
