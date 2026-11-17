"""Unified atomic RawWriter and multi-threaded lossless PNG exporter."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image
import torch

from .runtime import read_json, save_json


def save_png_batch_fast(
    u8_bchw: torch.Tensor,
    task_ids: list[str],
    out_dir: Path | str,
    *,
    max_workers: int = 8,
    compress_level: int = 1,
) -> int:
    """Save (B, 3, H, W) uint8 RGB tensors as lossless PNGs ~20x faster via C-zlib threading."""
    dst = Path(out_dir)
    dst.mkdir(parents=True, exist_ok=True)
    arrs = u8_bchw.detach().cpu().permute(0, 2, 3, 1).numpy()

    def _write_one(pair):
        tid_s, arr = pair
        fname = tid_s.replace("/", "__") + ".png"
        Image.fromarray(arr).save(dst / fname, format="PNG", compress_level=compress_level)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        list(ex.map(_write_one, zip(task_ids, arrs)))
    return len(task_ids)


def export_raw_pngs(raw_dir: Path | str, out_dir: Path | str, *, max_workers: int = 8) -> int:
    raw_p = Path(raw_dir)
    meta = read_json(raw_p / "meta.json")
    total = 0
    for b in meta["batches"]:
        payload = torch.load(raw_p / b["file"], map_location="cpu", weights_only=False)
        total += save_png_batch_fast(payload["rgb_u8"], list(payload["ids"]), out_dir, max_workers=max_workers)
    return total


class RawWriter:
    """Atomic batch writer compatible with src.eval.measure across V15-V19."""

    def __init__(self, arm_name: str, raw_root: Path | str, *, kind: str = "weave",
                 expected_tasks: int = 750, batch_digits: int = 5):
        self.arm_name = arm_name
        self.kind = kind
        self.expected_tasks = expected_tasks
        self.batch_digits = batch_digits
        self.path = Path(raw_root) / arm_name
        self.path.mkdir(parents=True, exist_ok=True)
        self.batches: list[dict] = []
        self.n = 0

    def append(self, ids: list[str], rgb01: torch.Tensor, z_final: torch.Tensor | None = None) -> None:
        fname = f"batch_{self.n:0{self.batch_digits}d}.pt"
        u8 = (rgb01.detach().cpu().float() * 255.0).round().clamp(0, 255).byte()
        payload = {"ids": list(ids), "rgb_u8": u8}
        if z_final is not None:
            payload["z_final"] = z_final.detach().cpu().half()
        tmp = self.path / (fname + ".tmp")
        torch.save(payload, tmp)
        tmp.replace(self.path / fname)
        self.batches.append({"offset": self.n, "file": fname, "ids": list(ids)})
        self.n += len(ids)

    def finish(self, **extra) -> Path:
        if self.expected_tasks and self.n != self.expected_tasks:
            raise ValueError(f"expected {self.expected_tasks} tasks, got {self.n}")
        all_ids = [tid for b in self.batches for tid in b["ids"]]
        if len(set(all_ids)) != len(all_ids):
            raise ValueError(f"duplicate task_ids in RawWriter({self.arm_name})")
        meta = {
            "kind": self.kind,
            "arm": self.arm_name,
            "n_tasks": self.n,
            "decoder": "S10",
            "batches": self.batches,
            **extra,
        }
        save_json(self.path / "meta.json", meta)
        return self.path
