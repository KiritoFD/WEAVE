"""Shared runtime helpers: atomic JSON IO, seeding, 3060-safe VRAM tiering & subprocess hygiene."""
from __future__ import annotations

import contextlib
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]


def utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read_json(path: Path | str):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save_json(path: Path | str, value) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False),
        encoding="utf-8",
    )
    tmp.replace(p)


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def bf16_autocast(device: str = "cuda"):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def vram_total_gb(device: str = "cuda") -> float:
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return 0.0
    idx = torch.device(device).index or 0
    return float(torch.cuda.get_device_properties(idx).total_memory) / (1024.0 ** 3)


def auto_vram_budget_gb(device: str = "cuda", *, total_gb_override: float | None = None) -> float:
    """Compute safe hardware-proportional peak VRAM budget (keeps 24GB 4090 ~21.8GB instead of 18.5GB)."""
    gb = vram_total_gb(device) if total_gb_override is None else float(total_gb_override)
    if gb <= 0.0:
        return 0.0
    if gb <= 9.0:
        return 7.2
    if gb <= 13.5:
        return 10.5
    return round(max(18.5, gb - 2.2), 1)


def auto_infer_batch(device: str = "cuda", *, total_gb_override: float | None = None) -> int:
    """Hardware-tiered latent trajectory inference batch size (saturates 24GB 4090 at 128)."""
    gb = vram_total_gb(device) if total_gb_override is None else float(total_gb_override)
    if gb <= 0.0 or gb <= 9.0:
        return 16
    if gb <= 13.5:
        return 32
    return 128


def auto_decode_batch(device: str = "cuda", *, total_gb_override: float | None = None) -> int:
    """Hardware-tiered 512px S10 decoder batch size (64 on 24GB 4090 avoids 512px UpBlock OOM at 128)."""
    gb = vram_total_gb(device) if total_gb_override is None else float(total_gb_override)
    if gb <= 0.0 or gb <= 9.0:
        return 16
    if gb <= 13.5:
        return 32
    return 64


def auto_referee_batch(device: str = "cuda", *, total_gb_override: float | None = None) -> int:
    """Hardware-tiered frozen FeatureStack batch size (saturates 24GB 4090 at 64)."""
    gb = vram_total_gb(device) if total_gb_override is None else float(total_gb_override)
    if gb <= 0.0 or gb <= 9.0:
        return 8
    if gb <= 13.5:
        return 16
    return 64


def tier_micro_accum(
    effective_batch: int,
    *,
    default_micro_24gb: int = 128,
    max_micro_12gb: int = 32,
    max_micro_8gb: int = 16,
    device: str = "cuda",
    total_gb_override: float | None = None,
) -> tuple[int, int]:
    """Auto-scale (micro, accum) for 24GB 4090 vs 12GB 3060 vs 8GB 4070LP keeping micro*accum == effective_batch."""
    gb = vram_total_gb(device) if total_gb_override is None else float(total_gb_override)
    if gb <= 0.0:
        cap = max_micro_8gb
    elif gb <= 9.0:
        cap = max_micro_8gb
    elif gb <= 13.5:
        cap = max_micro_12gb
    else:
        cap = default_micro_24gb
    micro = min(effective_batch, cap)
    while effective_batch % micro != 0 and micro > 1:
        micro -= 1
    return micro, effective_batch // micro


def sanitize_cuda_env(base_env: dict[str, str] | None = None) -> dict[str, str]:
    """Build clean UTF-8 subprocess env and strip disallowed expandable_segments flags."""
    env = dict(os.environ if base_env is None else base_env)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("MKL_NUM_THREADS", "4")
    for key in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
        if key in env:
            parts = [
                tok.strip()
                for tok in env[key].split(",")
                if tok.strip() and not tok.strip().startswith("expandable_segments")
            ]
            if parts:
                env[key] = ",".join(parts)
            else:
                env.pop(key, None)
    return env


def run_isolated_python(module: str, args: list[str], *, cwd: Path | None = None) -> int:
    """Spawn a clean isolated Python child process with sanitized CUDA & UTF-8 environment."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    cmd = [sys.executable, "-u", "-m", module, *args]
    proc = subprocess.run(cmd, cwd=cwd or REPO, env=sanitize_cuda_env())
    return int(proc.returncode)
