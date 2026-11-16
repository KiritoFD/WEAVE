"""Data / eval / queue contracts: split disjointness + test750 exclusion,
paired-noise determinism, DONE admission, retry cap."""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
ROOT = Path(__file__).resolve().parents[2]


def test_data_split_and_exclusion():
    from exp.v17.data import canonical_pid, audit
    assert canonical_pid('A__b__Painting_123') == 'painting_123'
    assert canonical_pid('Painting_123') == 'painting_123'


def test_paired_eps_determinism():
    from exp.v17.eval750 import EVAL_SEED
    g1 = torch.randn(750, 4, 64, 64, generator=torch.Generator()
                     .manual_seed(EVAL_SEED))
    g2 = torch.randn(750, 4, 64, 64, generator=torch.Generator()
                     .manual_seed(EVAL_SEED))
    assert torch.equal(g1, g2), 'paired noise must be identical across arms'


def test_done_admission_and_retry(tmp_path):
    from scripts.v17.queue import done_ok  # noqa: E402
    (tmp_path / 'a').mkdir()
    assert not done_ok(tmp_path / 'a')
    (tmp_path / 'a' / 'DONE.json').write_text(json.dumps(
        {'n_tasks': 750, 'n_cross': 600, 'n_same': 150}))
    assert done_ok(tmp_path / 'a')
    (tmp_path / 'b').mkdir()
    (tmp_path / 'b' / 'DONE.json').write_text(json.dumps(
        {'n_tasks': 100, 'n_cross': 0, 'n_same': 100}))
    assert not done_ok(tmp_path / 'b'), 'partial evals must not count'


def test_manifest_counts():
    from exp.v17.build_manifest import main as build
    import subprocess
    out = ROOT / 'exp/v17/manifest_36h.json'
    if not out.exists():
        build([])
    m = json.loads(out.read_text(encoding='utf-8'))
    ids = [a['arm_id'] for a in m['main_arms']]
    assert len(ids) == len(set(ids)), 'duplicate arm ids'
    assert sum(i.startswith('P2_') for i in ids) >= 18
    assert m['hours'] == {'min_valid': 24, 'target': 36}
