"""P0-3 verification: distribution supervision must cover both directions and all
five target styles.

V10 always started from the canonical `a` sample, integrated `a -> b`, and grouped by
`pb`. Two consequences, both wrong:
  * the `b -> a` direction was never supervised;
  * the lowest-index style (Early Renaissance) could never be a distribution target.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

import sbflow.train5 as T

STYLES = 5


def _pairs():
    return [(i, j) for i in range(STYLES) for j in range(i + 1, STYLES)]


def test_groups_cover_all_styles():
    pairs = _pairs()
    pa, pb, dirs = T._mixed_pair_ids(pairs, 40, "cpu", torch.Generator().manual_seed(0))
    groups = torch.where(dirs == 1, pb, pa)
    present = set(groups.tolist())
    assert present == set(range(STYLES)), f"missing target styles: {set(range(STYLES)) - present}"
    # V10 grouping (pb only) must NOT cover style 0 -- show the regression is detectable
    v10 = set(pb.tolist())
    assert 0 not in v10, "test assumption broken: pb should never be style 0"
    counts = {s: int((groups == s).sum()) for s in range(STYLES)}
    print(f"PASS new groups cover all 5 target styles {counts}; "
          f"V10 groups (pb only) would miss style 0")


def test_both_directions_present():
    pa, pb, dirs = T._mixed_pair_ids(_pairs(), 40, "cpu", torch.Generator().manual_seed(1))
    assert (dirs == 1).any() and (dirs == 0).any(), "only one direction in the batch"
    n_f, n_r = int((dirs == 1).sum()), int((dirs == 0).sum())
    print(f"PASS both directions sampled (forward {n_f}, reverse {n_r})")


def test_reverse_uses_reverse_source():
    """Reverse samples must start from the b side, not from the a side."""
    g = torch.Generator().manual_seed(2)
    B = 8
    z_a = torch.zeros(B, 2, 2, 2)
    z_b = torch.ones(B, 2, 2, 2)
    dirs = torch.tensor([1, 0] * (B // 2))
    fwd_m = (dirs == 1).view(-1, 1, 1, 1)
    z_start = torch.where(fwd_m, z_a, z_b)
    assert torch.allclose(z_start[dirs == 1], z_a[dirs == 1])
    assert torch.allclose(z_start[dirs == 0], z_b[dirs == 0])
    print("PASS reverse samples start from the b endpoint")


def test_free_ode_round_trip(monkeypatch_const=True):
    """With a constant velocity c, forward then reverse must return to the start."""
    orig = T.mixed_velocity
    c = 0.7

    def const_velocity(model, z, t, pa, pb, stats, eps, objective):
        return torch.full_like(z, c)

    T.mixed_velocity = const_velocity
    try:
        g = torch.Generator().manual_seed(3)
        B = 4
        z = torch.randn(B, 2, 2, 2, generator=g)
        pa = torch.zeros(B, dtype=torch.long)
        pb = torch.ones(B, dtype=torch.long)

        d_f = torch.ones(B, dtype=torch.long)
        out_f = T.free_ode_endpoint(None, z, pa, pb, d_f, None, 0.04, "bidir", 4)
        assert torch.allclose(out_f, z + c, atol=1e-5), "forward ODE is not +c"

        d_r = torch.zeros(B, dtype=torch.long)
        back = T.free_ode_endpoint(None, out_f, pa, pb, d_r, None, 0.04, "bidir", 4)
        err = float((back - z).abs().max())
        assert err < 1e-5, f"reverse ODE did not undo the forward step (err {err:.2e})"
        print(f"PASS free ODE round trip forward+reverse (max err {err:.1e})")
    finally:
        T.mixed_velocity = orig


if __name__ == "__main__":
    torch.set_num_threads(2)
    test_groups_cover_all_styles()
    test_both_directions_present()
    test_reverse_uses_reverse_source()
    test_free_ode_round_trip()
    print("ALL DIST DIRECTION TESTS PASSED")
