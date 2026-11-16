"""V17 shared runtime: paths, atomic IO, hashes, seeding, heartbeats.

Everything heavy is imported from the proven exp/v16 substrate:
pool_samples / tasks_and_sources / RawWriter / load_base / generate.
"""
from __future__ import annotations

import contextlib
import datetime
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/v17'
STYLES = ['Early_Renaissance', 'Impressionism', 'Minimalism', 'Rococo', 'Ukiyo_e']
TAU_EVAL = .7                       # cross-style main readout workpoint
EVAL_SEED = 20261185                # V16 B_eval noise seed (paired eps contract)
CORE_REPEAT_ARMS = ['S5', 'S6', 'H4', 'F1']   # preregistered seed-repeat pool


def utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=1,
                               default=str, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def sha(path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def autocast(device='cuda'):
    return (torch.autocast('cuda', dtype=torch.bfloat16)
            if str(device).startswith('cuda') else contextlib.nullcontext())


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def heartbeat(out_dir: Path, state: str, extra: dict | None = None):
    """progress.json heartbeat: pid, phase, gpu telemetry, utc."""
    payload = {'utc': utc(), 'pid': Path(__file__).name and __import__('os').getpid(),
               'state': state}
    try:
        import pynvml  # optional
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        payload['gpu_util'] = int(pynvml.nvmlDeviceGetUtilizationRates(h).gpu)
        payload['gpu_mem_mib'] = int(pynvml.nvmlDeviceGetMemoryInfo(h).used / 2 ** 20)
    except Exception:
        pass
    if extra:
        payload.update(extra)
    save(Path(out_dir) / 'progress.json', payload)


def cosine_lr(step: int, total: int, base_lr: float, warm: float = .05) -> float:
    return base_lr * (warm + (1 - warm) * .5 * (1 + math.cos(math.pi * step / total)))


def class_balanced_indices(sid: torch.Tensor, per_class: int, gen: torch.Generator):
    """Indices giving `per_class` samples per style, shuffled."""
    out = []
    for s in range(len(STYLES)):
        pool = (sid == s).nonzero(as_tuple=True)[0].tolist()
        if len(pool) < per_class:
            raise ValueError(f'class {s}: {len(pool)} < {per_class}')
        pick = [pool[i] for i in torch.randperm(len(pool), generator=gen)[:per_class]]
        out.extend(pick)
    g2 = torch.Generator().manual_seed(gen.initial_seed() + 1)
    order = torch.randperm(len(out), generator=g2).tolist()
    return torch.tensor([out[i] for i in order], dtype=torch.long)
