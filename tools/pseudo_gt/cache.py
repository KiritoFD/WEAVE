"""唯一 sample_id 索引与可续跑的 shard 写入（gt-practice §7）。"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import torch

SHARD_SIZE = 1000


@dataclass
class SampleRecord:
    sample_id: str
    source_id: str
    donor_id: str
    source_style: str
    target_style: str
    method: str
    patch_spec: str
    seed: int
    match_score: float
    nominal_coverage: float
    effective_coverage: float
    z0_is_original: bool
    z1_is_original: bool
    band_energy: Dict[str, float] = field(default_factory=dict)
    seam_grad: float = 0.0


def make_sample_id(method: str, patch_spec: str, source_id: str, donor_id: str, seed: int) -> str:
    return f"{method}_{patch_spec}_{source_id}__{donor_id}__s{seed}"


def _atomic_write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


class ShardWriter:
    """按 shard 写 ``z0/z1`` fp16 + manifest；已完成 shard 跳过（断点续跑）。"""

    def __init__(self, root: Path, *, shard_size: int = SHARD_SIZE, dtype=torch.float16):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.shard_size = int(shard_size)
        self.dtype = dtype
        self.manifest_path = self.root / "manifest.json"
        self._buf_z0: List[torch.Tensor] = []
        self._buf_z1: List[torch.Tensor] = []
        self._buf_rec: List[dict] = []
        self._shard_idx = self._count_existing_shards()
        self._records = self._load_manifest()

    # ---------- 续跑 ----------
    def _count_existing_shards(self) -> int:
        return len(sorted(self.root.glob("shard_*.pt")))

    def _load_manifest(self) -> List[dict]:
        if self.manifest_path.exists():
            with open(self.manifest_path, "r", encoding="utf-8") as fh:
                return json.load(fh).get("records", [])
        return []

    def completed_ids(self) -> set[str]:
        return {r["sample_id"] for r in self._records}

    def shard_exists(self, idx: int) -> bool:
        return (self.root / f"shard_{idx:05d}.pt").exists()

    # ---------- 写入 ----------
    def add(self, z0: torch.Tensor, z1: torch.Tensor, rec: SampleRecord) -> None:
        if z0.shape != z1.shape:
            raise ValueError(f"{rec.sample_id}: z0/z1 shape 不一致 {tuple(z0.shape)} vs {tuple(z1.shape)}")
        self._buf_z0.append(z0.detach().to("cpu", self.dtype))
        self._buf_z1.append(z1.detach().to("cpu", self.dtype))
        self._buf_rec.append(asdict(rec))
        if len(self._buf_z0) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self._buf_z0:
            return
        idx = self._shard_idx
        path = self.root / f"shard_{idx:05d}.pt"
        payload = {
            "z0": torch.stack(self._buf_z0),
            "z1": torch.stack(self._buf_z1),
            "meta": self._buf_rec,
        }
        torch.save(payload, path)
        self._records.extend(self._buf_rec)
        _atomic_write_json(
            self.manifest_path,
            {
                "count": len(self._records),
                "shards": self._shard_idx + 1,
                "dtype": str(self.dtype).replace("torch.", ""),
                "records": self._records,
            },
        )
        self._shard_idx += 1
        self._buf_z0, self._buf_z1, self._buf_rec = [], [], []

    def close(self) -> None:
        self.flush()


def load_shard(root: Path, idx: int) -> dict:
    return torch.load(Path(root) / f"shard_{idx:05d}.pt", map_location="cpu", weights_only=False)


def verify_manifest(root: Path) -> dict:
    """校验 manifest.count 与各 shard 内 meta 行数一致。"""
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    total = 0
    shards = sorted(root.glob("shard_*.pt"))
    for path in shards:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        total += len(payload["meta"])
    return {
        "manifest_count": manifest["count"],
        "shard_rows": total,
        "shards": len(shards),
        "ok": manifest["count"] == total == len(manifest["records"]),
    }
