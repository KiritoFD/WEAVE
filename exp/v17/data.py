"""V17 data integrity: canonical painting ids, balanced sampling, audits.

Reuses exp.v16.pool_samples (NFC + dedup + test/reference exclusion) and adds:
per-style balance audit, train/holdout disjointness, Test750 source independence,
and a persisted audit JSON under runs/v17/data_audit.json.
"""
from __future__ import annotations

import unicodedata
from pathlib import Path

import torch

from .common import OUT, STYLES, read, save, sha
from exp.v16 import pool_samples


def canonical_pid(name: str) -> str:
    return unicodedata.normalize('NFC', Path(str(name).replace('\\', '/')).stem
                                 .split('__')[-1]).casefold()


def balanced_pool(pool: str, train_each: int = 100, hold_each: int = 20,
                  seed: int = 20261117):
    """V16 split (all five styles, test/reference-excluded).  Raises if any
    style cannot supply its quota; the caller may lower quotas."""
    return pool_samples(pool, train_each=train_each, hold_each=hold_each,
                        seed=seed)


def audit(out_dir: Path, pool: str, source_manifest: dict) -> dict:
    """Persist the integrity audit for the queue gate."""
    tasks = read(Path(__file__).resolve().parents[2] /
                 'eval_protocol/tasks_750.json')['tasks']
    refs = read(Path(__file__).resolve().parents[2] /
                'eval_protocol/references_v1.json')['references']
    banned = {canonical_pid(t['source_stem']) for t in tasks}
    ref_ids = {canonical_pid(r['stem']) for rows in refs.values() for r in rows}
    train_ids = [canonical_pid(x) for x in source_manifest['train']['ids']]
    hold_ids = [canonical_pid(x) for x in source_manifest['holdout']['ids']]
    per_style = {s: sum(1 for x in train_ids
                        if x in _style_of(pool, s)) for s in STYLES}
    audit = {
        'utc_bootstrap': True,
        'train_n': len(train_ids), 'holdout_n': len(hold_ids),
        'train_holdout_disjoint': not (set(train_ids) & set(hold_ids)),
        'train_hits_test_or_ref': sorted(set(train_ids) & banned)[:20],
        'train_test_ref_overlap_count': len(set(train_ids) & banned),
        'train_ref_overlap_count': len(set(train_ids) & ref_ids),
        'train_per_style': per_style,
        'impressionism_fraction': per_style['Impressionism'] / max(len(train_ids), 1),
        'tasks_sha256': sha(Path(__file__).resolve().parents[2] /
                            'eval_protocol/tasks_750.json'),
        'balanced_training': False,   # preregistered fact: no rebalance in V17
    }
    save(Path(out_dir) / 'data_audit.json', audit)
    if audit['train_test_ref_overlap_count']:
        raise AssertionError('training pool leaked Test750/reference paintings')
    return audit


def _style_of(pool: str, style: str) -> set:
    sid = STYLES.index(style)
    d = torch.load(Path(pool) / f'{sid:02d}_{style}.pt', map_location='cpu',
                   weights_only=False)
    return {canonical_pid(str(x)) for x in d['ids']}


def balanced_view(sid: torch.Tensor, per_class: int, seed: int) -> torch.Tensor:
    """Class-balanced index view for residual-family training (P4 contract)."""
    from .common import class_balanced_indices
    return class_balanced_indices(sid, per_class,
                                  torch.Generator().manual_seed(seed))
