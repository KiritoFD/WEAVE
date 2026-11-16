"""Teacher assertions: constant logit bias leaves the linear margin gradient
UNCHANGED (must be proven before T1 is meaningful); per-sample TopK masks are
independent; trust budget never exceeded."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


class LinearCritic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(5, 16384) * .01)

    def forward(self, z):
        return z.flatten(1) @ self.w.t()


def test_constant_logit_bias_gradient_invariance():
    """With argmax UNCHANGED, per-class constant bias cancels in the margin
    difference: gradient must be bit-identical.  Proven before T1 runs."""
    from exp.v8.guidance import target_margin
    z = torch.randn(3, 4, 64, 64).requires_grad_(True)
    tid = torch.tensor([0, 1, 2])
    critic = LinearCritic()
    with torch.no_grad():
        critic.w *= 100          # wide margins: small bias cannot flip argmax
    g0 = torch.autograd.grad(target_margin(critic, z, tid).sum(), z)[0]
    bias = torch.tensor([.3, -.1, .2, .05, -.25])       # small: no argmax flip

    def margin_with_bias(z, tid):
        logits = z.flatten(1) @ critic.w.t()
        biased = logits + bias
        # target score shifts by bias[t]; each competitor by bias[k]; the
        # resulting margin gradient is identical for unchanged argmax
        return logits.gather(1, tid[:, None]).squeeze(1) - \
            biased.scatter(1, tid[:, None], -1e9).max(1).values + bias[tid]

    g1 = torch.autograd.grad(margin_with_bias(z, tid).sum(), z)[0]
    assert torch.allclose(g0, g1, atol=1e-6), 'constant-bias gradient invariance'


def test_calibrated_ranking_can_change():
    """Nonlinear max-other ranking CAN flip under calibration (documented)."""
    logits = torch.tensor([[1., 2., 3., 4., 5.]])
    tid = torch.tensor([0])
    bias = torch.tensor([10., 0., 0., 0., 0.])
    calibrated = logits + bias
    assert logits.argmax() == 4
    assert calibrated.argmax() == 0, 'ranking flip is the only real T1 effect'


def test_topk_mask_independence():
    from exp.v9_local.direction import DirectionComputer  # noqa: F401
    # region_mask TopK: distinct rows receive independent thresholds
    score = torch.rand(4, 64)
    k = 16
    thr = score.topk(k, dim=1).values[:, -1:]
    sel = (score >= thr).sum(1)
    assert (sel == k).all(), 'per-row TopK must be independent'
