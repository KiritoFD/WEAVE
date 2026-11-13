import sys
from pathlib import Path

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.legacy_root.internal_dynamics import (
    ConstantProbeReadingsError,
    InternalDynamicsState,
    InternalEarlyStopConfigError,
    check_probe_constancy,
    validate_internal_early_stop_model,
)


class _GateWithHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate = nn.Parameter(torch.tensor(0.05))


class _ModelWithoutHead(nn.Module):
    def __init__(self, *, with_head: bool) -> None:
        super().__init__()
        if with_head:
            self.target_latent_hf_subband_delta_lh = _GateWithHead()
            self.target_latent_hf_subband_delta_hl = _GateWithHead()


def test_internal_transition_requires_gate_reversal_and_relative_gradient_drop():
    state = InternalDynamicsState()
    rows = [
        (1, 0.173068, 1.4727),
        (2, 0.172264, 1.4328),
        (3, 0.172009, 1.0868),
        (4, 0.175722, 0.1634),
        (5, 0.179492, 0.1510),
    ]
    transitions = []
    for epoch, gate, ratio in rows:
        metrics = {
            "internal_probe_gate_mean": gate,
            "internal_probe_shared_ll_hf_grad_ratio": ratio,
        }
        transitions.append(
            state.update(
                epoch,
                metrics,
                min_epoch=3,
                gate_delta_threshold=0.0,
                shared_ratio_drop_threshold=0.65,
            )
        )
    assert transitions == [False, False, False, True, False]
    assert state.transition_epoch == 4


def test_internal_transition_rejects_ratio_crossing_while_gate_is_contracting():
    state = InternalDynamicsState(previous_gate_mean=0.18, previous_shared_ll_hf_ratio=1.2)
    metrics = {
        "internal_probe_gate_mean": 0.17,
        "internal_probe_shared_ll_hf_grad_ratio": 0.8,
    }
    assert not state.update(
        4,
        metrics,
        min_epoch=3,
        gate_delta_threshold=0.0,
        shared_ratio_drop_threshold=0.65,
    )


def test_relative_drop_is_invariant_to_absolute_gradient_scale():
    state = InternalDynamicsState()
    rows = [
        (1, 0.1731, 0.87),
        (2, 0.1723, 0.67),
        (3, 0.1720, 0.42),
        (4, 0.1757, 0.24),
    ]
    transitions = []
    for epoch, gate, ratio in rows:
        metrics = {
            "internal_probe_gate_mean": gate,
            "internal_probe_shared_ll_hf_grad_ratio": ratio,
        }
        transitions.append(state.update(
            epoch,
            metrics,
            min_epoch=3,
            gate_delta_threshold=0.0,
            shared_ratio_drop_threshold=0.65,
        ))
    assert transitions == [False, False, False, True]


def test_validate_internal_early_stop_model_raises_without_subband_head():
    with pytest.raises(InternalEarlyStopConfigError, match="早停已启用但门控不存在"):
        validate_internal_early_stop_model(_ModelWithoutHead(with_head=False))
    validate_internal_early_stop_model(_ModelWithoutHead(with_head=True))


def test_constant_gate_readings_raise():
    with pytest.raises(ConstantProbeReadingsError, match="gate_mean 恒为 0"):
        check_probe_constancy([
            {"gate_mean": 0.0, "rho": 1.0},
            {"gate_mean": 0.0, "rho": 1.1},
        ])


def test_constant_rho_readings_raise():
    with pytest.raises(ConstantProbeReadingsError, match="rho 方差为 0"):
        check_probe_constancy([
            {"gate_mean": 0.01, "rho": 1.0},
            {"gate_mean": 0.02, "rho": 1.0},
        ])


def test_min_steps_blocks_early_stop_until_reached():
    metrics = {
        "internal_probe_gate_mean": 0.1757,
        "internal_probe_shared_ll_hf_grad_ratio": 0.16,
    }
    blocked = InternalDynamicsState(previous_gate_mean=0.1720, previous_shared_ll_hf_ratio=1.08)
    assert not blocked.update(
        4,
        dict(metrics),
        min_epoch=3,
        gate_delta_threshold=0.0,
        shared_ratio_drop_threshold=0.65,
        min_steps=2000,
        global_step=500,
    )
    fired = InternalDynamicsState(previous_gate_mean=0.1720, previous_shared_ll_hf_ratio=1.08)
    assert fired.update(
        4,
        dict(metrics),
        min_epoch=3,
        gate_delta_threshold=0.0,
        shared_ratio_drop_threshold=0.65,
        min_steps=2000,
        global_step=2500,
    )
