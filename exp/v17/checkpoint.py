"""V17 full-state checkpoints: model/optimizer/scheduler/RNG/sampler/scaler.

Contract (tests/v17/test_resume.py): 20 steps uninterrupted == 10 + resume + 10
on CPU; resume restores optimizer, scheduler, python/numpy/torch/cuda RNG,
dataloader-generator state and GradScaler, and never resets the lr schedule.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from .common import save


def snapshot_rng():
    return {'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state.get('cuda') is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def save_ckpt(path: Path, *, model, optimizer, scheduler=None, scaler=None,
              step: int, gen_state=None, extra: dict | None = None):
    payload = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler else None,
        'scaler': scaler.state_dict() if scaler else None,
        'step': step,
        'rng': snapshot_rng(),
        'gen_state': gen_state,
        'extra': extra or {},
    }
    tmp = Path(str(path) + '.tmp')
    torch.save(payload, tmp)
    tmp.replace(path)
    return payload


def load_ckpt(path: Path, *, model, optimizer=None, scheduler=None,
              scaler=None, map_location='cpu'):
    ck = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ck['model'])
    if optimizer is not None and ck.get('optimizer'):
        optimizer.load_state_dict(ck['optimizer'])
    if scheduler is not None and ck.get('scheduler'):
        scheduler.load_state_dict(ck['scheduler'])
    if scaler is not None and ck.get('scaler'):
        scaler.load_state_dict(ck['scaler'])
    restore_rng(ck['rng'])
    return ck


def find_resume(out_dir: Path, prefer: str = 'latest'):
    """Return newest ckpt path in out_dir (ckpt_latest.pt or ckpt_step_*.pt)."""
    cands = list(Path(out_dir).glob('ckpt_step_*.pt')) + \
        [Path(out_dir) / 'ckpt_latest.pt']
    cands = [c for c in cands if c.exists()]
    if not cands:
        return None
    if (Path(out_dir) / 'ckpt_latest.pt').exists() and prefer == 'latest':
        return Path(out_dir) / 'ckpt_latest.pt'
    return max(cands, key=lambda p: p.stat().st_mtime)
