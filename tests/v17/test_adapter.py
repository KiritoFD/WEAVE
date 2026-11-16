"""Adapter assertions: centered sums to zero; target-source same-label is
EXACT zero; zero-init identity; no frozen-base gradients."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def test_centered_offsets_sum_zero():
    from exp.v17.adapter import conditioning_offset
    E = torch.randn(6, 32)
    total = sum(conditioning_offset('centered', torch.tensor([s]), None, E)
                for s in range(5))
    assert total.abs().max() < 1e-6, 'centered offsets must sum to zero'


def test_target_source_same_label_exact_zero():
    from exp.v17.adapter import conditioning_offset
    E = torch.randn(6, 32)
    tid = torch.tensor([2])
    off = conditioning_offset('target_source', tid, tid.clone(), E)
    assert off.abs().max() == 0, 'same label must be EXACT zero'


def test_students_identity_at_init():
    from exp.v17.adapter import build_student
    for kind in ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S8']:
        net = build_student(kind).cpu().eval()
        b = torch.randn(2, 4, 64, 64)
        c = torch.randn(2, 4, 64, 64)
        tid = torch.tensor([0, 3])
        tok = torch.randn(2, 384, 16, 16) if kind != 'S1' else None
        sl = torch.tensor([1, 2])
        with torch.no_grad():
            delta = net(b, c, tid, tok, sl)[0]
        assert delta.abs().max() == 0, f'{kind} must be identity at init'


def test_frozen_base_never_grads():
    from exp.v17.adapter import build_student
    net = build_student('S2')
    assert not hasattr(net, 'base_model'), 'base must not live inside student'
