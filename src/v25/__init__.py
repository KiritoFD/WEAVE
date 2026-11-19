"""src.v25: V25 收敛期基础设施（当前仅 B03 血缘注册表；后续 B05/B06 由总控集成）。

纯 stdlib 门面：不 import torch/numpy，CPU/无 GPU 环境可直接使用。
血缘检查器见 ``artifact_registry``（agent2.md §四 B03）。
"""
from __future__ import annotations

from src.v25.artifact_registry import (
    ArtifactRecord,
    ArtifactRegistry,
    DEP_CKPT_SHA256,
    DEP_DECODER_VERSION,
    DEP_FEATURE_ENCODER_VERSION,
    DEP_INPUT_MANIFEST_SHA,
    DEP_PARAMS_HASH,
    FORMAT_REF_FEATS_V1_2D_NO_BATCH,
    FORMAT_REF_FEATS_V2_BATCH_DIM,
    REGISTRY_SCHEMA_VERSION,
    REPO_ROOT,
    S10_DECODER_RELPATH,
    STATUS_INVALID,
    STATUS_MISSING,
    STATUS_UNKNOWN,
    STATUS_VALID,
    UNKNOWN,
    ckpt_sha256_from_paths,
    file_sha256,
    normalize_dep_value,
    param_hash,
    resolve_git_sha,
    s10_decoder_version,
    utc_now,
)

__all__ = [
    "ArtifactRecord",
    "ArtifactRegistry",
    "DEP_CKPT_SHA256",
    "DEP_DECODER_VERSION",
    "DEP_FEATURE_ENCODER_VERSION",
    "DEP_INPUT_MANIFEST_SHA",
    "DEP_PARAMS_HASH",
    "FORMAT_REF_FEATS_V1_2D_NO_BATCH",
    "FORMAT_REF_FEATS_V2_BATCH_DIM",
    "REGISTRY_SCHEMA_VERSION",
    "REPO_ROOT",
    "S10_DECODER_RELPATH",
    "STATUS_INVALID",
    "STATUS_MISSING",
    "STATUS_UNKNOWN",
    "STATUS_VALID",
    "UNKNOWN",
    "ckpt_sha256_from_paths",
    "file_sha256",
    "normalize_dep_value",
    "param_hash",
    "resolve_git_sha",
    "s10_decoder_version",
    "utc_now",
]
