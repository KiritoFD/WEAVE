"""Offline cache for frozen DINOv2 features (infra patch P1).

Reads the directory layout written by ``tools/build_dino_feature_cache.py``::

    <root>/index.json
    <root>/f<res>_<layer>.npy      # [total_rows, tokens, channels], float16

Row layout for style ``s``, item ``i`` and flip flag ``f``::

    row = style_offsets[s] + i * stride + f      # stride = 2 if flip_rows else 1

Every lookup only reads the requested rows from a read-only ``np.memmap`` and
copies them to the device, so the cache never has to be resident in RAM.

The class is defensive on purpose: any incompatibility (missing directory,
different ``feature_res``, missing layer, unknown style id) disables the cache
and the trainer falls back to the online extractor path. Falling back is
bit-identical to the pre-patch behaviour, which is the hard requirement here.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

logger = logging.getLogger(__name__)

_NUMPY_DTYPES = {
    "float16": np.float16,
    "float32": np.float32,
    "fp16": np.float16,
    "fp32": np.float32,
}


class FrozenFeatureCache:
    """Read-only lookup of pre-computed extractor features."""

    def __init__(
        self,
        root: str | Path,
        *,
        feature_res: int,
        layers: Iterable[str],
        dtype: str = "float16",
    ) -> None:
        self.root = Path(root)
        self.feature_res = int(feature_res)
        self.layers = [str(name) for name in layers]
        self.dtype_name = str(dtype)
        self.enabled = False

        self._style_offsets: dict[str, int] = {}
        self._style_counts: dict[str, int] = {}
        self._stride = 1
        self._tokens = 0
        self._channels = 0
        self._side = 0
        self._files: dict[str, Path] = {}
        self._maps: dict[str, np.memmap] = {}

        if not self.root.is_dir():
            logger.warning("FrozenFeatureCache: %s is not a directory; online path stays active.", self.root)
            return
        index_path = self.root / "index.json"
        if not index_path.is_file():
            logger.warning("FrozenFeatureCache: missing %s; online path stays active.", index_path)
            return
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("FrozenFeatureCache: cannot read %s (%s); online path stays active.", index_path, exc)
            return

        cached_res = int(payload.get("feature_res", 0) or 0)
        if self.feature_res <= 0 or cached_res != self.feature_res:
            logger.warning(
                "FrozenFeatureCache: cache feature_res=%d does not match training feature_res=%d; "
                "online path stays active.",
                cached_res, self.feature_res,
            )
            return

        cached_layers = dict(payload.get("layers") or {})
        missing = [name for name in self.layers if name not in cached_layers]
        if missing:
            logger.warning(
                "FrozenFeatureCache: cache is missing layers %s (has %s); online path stays active.",
                missing, sorted(cached_layers),
            )
            return

        stored_dtype = str(payload.get("dtype", "float16")).lower()
        if stored_dtype not in _NUMPY_DTYPES:
            logger.warning("FrozenFeatureCache: unsupported cache dtype %r; online path stays active.", stored_dtype)
            return
        self._numpy_dtype = _NUMPY_DTYPES[stored_dtype]

        self._tokens = int(payload.get("tokens", 0) or 0)
        self._channels = int(payload.get("channels", 0) or 0)
        self._side = int(payload.get("side", 0) or 0)
        self._stride = 2 if bool(payload.get("flip_rows", False)) else 1

        for name, count in (payload.get("style_counts") or {}).items():
            self._style_counts[str(name)] = int(count)
        for name, offset in (payload.get("style_offsets") or {}).items():
            self._style_offsets[str(name)] = int(offset)

        for name in self.layers:
            rel = str(cached_layers[name])
            path = self.root / rel
            if not path.is_file():
                logger.warning("FrozenFeatureCache: missing layer file %s; online path stays active.", path)
                return
            self._files[name] = path

        self.enabled = True
        logger.info(
            "FrozenFeatureCache ready: %s (feature_res=%d, layers=%s, styles=%d, flip_rows=%s)",
            self.root, self.feature_res, self.layers, len(self._style_counts), self._stride == 2,
        )

    # ------------------------------------------------------------------
    # Row resolution
    # ------------------------------------------------------------------
    def rows(
        self,
        style_ids: Sequence[int] | torch.Tensor,
        item_indices: Sequence[int] | torch.Tensor,
        flip_flags: Sequence[float] | torch.Tensor | None,
        *,
        style_names: Sequence[str] | None = None,
    ) -> list[int] | None:
        """Return global cache rows, or ``None`` when any row is not cacheable."""
        if not self.enabled:
            return None
        style_ids = self._as_int_list(style_ids)
        item_indices = self._as_int_list(item_indices)
        flips = self._as_int_list(flip_flags) if flip_flags is not None else [0] * len(style_ids)
        names = list(style_names) if style_names is not None else sorted(self._style_counts)
        if len(style_ids) != len(item_indices) or len(item_indices) != len(flips):
            return None
        rows: list[int] = []
        for sid, idx, flip in zip(style_ids, item_indices, flips):
            name = names[sid] if 0 <= sid < len(names) else None
            if name is None or name not in self._style_offsets or name not in self._style_counts:
                return None
            if idx < 0 or idx >= self._style_counts[name]:
                return None
            rows.append(self._style_offsets[name] + idx * self._stride + (1 if flip else 0))
        return rows

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------
    def lookup(self, rows: Sequence[int], layers: Iterable[str], *, device: torch.device | str) -> dict[str, torch.Tensor]:
        """Read ``rows`` for the requested layers, reshaping to (B, C, side, side)."""
        if not self.enabled:
            raise RuntimeError("FrozenFeatureCache.lookup called while the cache is disabled")
        out: dict[str, torch.Tensor] = {}
        for name in layers:
            mm = self._map(name)
            array = np.asarray(mm[list(rows)])  # copy out of the memmap
            tensor = torch.from_numpy(array)
            b = tensor.shape[0]
            tensor = tensor.reshape(b, self._side, self._side, self._channels).permute(0, 3, 1, 2)
            out[name] = tensor.to(device=device, dtype=torch.float32).contiguous()
        return out

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _map(self, layer: str) -> np.memmap:
        mm = self._maps.get(layer)
        if mm is None:
            mm = np.memmap(
                self._files[layer],
                dtype=self._numpy_dtype,
                mode="r",
                shape=(self._total_rows(), self._tokens, self._channels),
            )
            self._maps[layer] = mm
        return mm

    def _total_rows(self) -> int:
        if not self._style_offsets:
            return 0
        return max(
            self._style_offsets[name] + count * self._stride
            for name, count in self._style_counts.items()
        )

    @staticmethod
    def _as_int_list(values: Sequence[int] | torch.Tensor) -> list[int]:
        if torch.is_tensor(values):
            return [int(v) for v in values.detach().to("cpu").view(-1).tolist()]
        return [int(v) for v in values]
