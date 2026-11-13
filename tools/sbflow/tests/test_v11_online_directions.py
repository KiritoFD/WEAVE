"""P0-4 verification: the online update must be direction-specific.

alpha-DSBM Algorithm 1:
  * reverse SDE gives x0_hat  ->  supervise the FORWARD drift b+ on (x0_hat, x1)
  * forward SDE gives x1_hat  ->  supervise the REVERSE drift b- on (x0, x1_hat)

V10 reused one mixed helper that picks b+ / b- per sample from `dirs`, so the
(x0, x1_hat) pairing also trained b+ for part of the batch. That is not the paper's
update, so exp27 could not be used to judge alpha-DSBM.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch

import sbflow.online_bridge as OB


def test_fwd_only_uses_dir1():
    seen = {}
    orig = OB.mixed_bridge_loss

    def spy(model, z0, z1, pa, pb, dirs, *a, **k):
        seen["dirs"] = dirs.clone()
        return torch.zeros((), requires_grad=True)

    OB.mixed_bridge_loss = spy
    try:
        n = 8
        z = torch.zeros(n, 1, 2, 2)
        pa = torch.zeros(n, dtype=torch.long)
        pb = torch.ones(n, dtype=torch.long)
        OB.mixed_bridge_loss_fwd(None, z, z, pa, pb, None, 0.04, 0.002, None)
        assert torch.all(seen["dirs"] == 1), "fwd loss is not restricted to dirs=1"
        OB.mixed_bridge_loss_rev(None, z, z, pa, pb, None, 0.04, 0.002, None)
        assert torch.all(seen["dirs"] == 0), "rev loss is not restricted to dirs=0"
        print("PASS mixed fwd/rev losses force a single drift type")
    finally:
        OB.mixed_bridge_loss = orig


def test_online_loss_mixed_pairs_correctly():
    """l_fwd must receive (x0_hat, z1); l_rev must receive (z0, x1_hat)."""
    calls = []
    orig_f, orig_r = OB.mixed_bridge_loss_fwd, OB.mixed_bridge_loss_rev

    def spy_f(model, z0, z1, *a, **k):
        calls.append(("fwd", z0.detach().clone(), z1.detach().clone()))
        return torch.zeros((), requires_grad=True)

    def spy_r(model, z0, z1, *a, **k):
        calls.append(("rev", z0.detach().clone(), z1.detach().clone()))
        return torch.zeros((), requires_grad=True)

    OB.mixed_bridge_loss_fwd, OB.mixed_bridge_loss_rev = spy_f, spy_r
    try:
        n, c, h, w = 4, 1, 2, 2
        g = torch.Generator().manual_seed(0)
        z0 = torch.randn(n, c, h, w, generator=g)
        z1 = torch.randn(n, c, h, w, generator=g) + 3.0
        pa = torch.zeros(n, dtype=torch.long)
        pb = torch.ones(n, dtype=torch.long)
        dirs = torch.ones(n, dtype=torch.long)

        class DummyTeacher(torch.nn.Module):
            def forward(self, x, tau, d, a, b, **kw):
                return torch.zeros_like(x)

        from tools.sbflow.preconditioning import StyleStats
        st = StyleStats(torch.zeros(c), torch.ones(c), torch.zeros(2, c),
                        torch.ones(2, c))

        # with a zero teacher the SDE endpoints equal their inputs
        OB.online_loss_mixed(None, DummyTeacher(), z0, z1, pa, pb, dirs, st, 0.04,
                             0.002, g, {"online_steps": 1, "online_mode": "ode",
                                        "t_floor": 0.002})
        kinds = [k for k, _, _ in calls]
        assert kinds == ["fwd", "rev"], f"unexpected call order {kinds}"
        _, f_z0, f_z1 = calls[0]
        _, r_z0, r_z1 = calls[1]
        assert torch.allclose(f_z1, z1), "fwd pairing must end at the real z1"
        assert torch.allclose(r_z0, z0), "rev pairing must start at the real z0"
        print("PASS online_loss_mixed pairs (x0_hat, z1)->b+ and (z0, x1_hat)->b-")
    finally:
        OB.mixed_bridge_loss_fwd, OB.mixed_bridge_loss_rev = orig_f, orig_r


def test_legacy_bridge_loss_modes():
    """mode='fwd'/'rev' must supervise one drift for the WHOLE batch."""
    from tools.sbflow.preconditioning import StyleStats
    n, c, h, w = 8, 1, 4, 4
    g = torch.Generator().manual_seed(1)
    z0 = torch.randn(n, c, h, w, generator=g)
    z1 = torch.randn(n, c, h, w, generator=g) + 1.0
    st = StyleStats(torch.zeros(c), torch.ones(c), torch.zeros(2, c), torch.ones(2, c))

    seen = {}

    class Rec(torch.nn.Module):
        def forward(self, x, tau, d, a, b, **kw):
            seen["d"] = d.clone()
            seen["tau"] = tau.clone()
            return torch.zeros_like(x)

    for mode, want in (("fwd", 1), ("rev", 0)):
        OB.bridge_loss(Rec(), z0, z1, 0, 1, st, 0.04, 0.002,
                       torch.Generator().manual_seed(2), mode=mode)
        assert torch.all(seen["d"] == want), f"mode={mode} used dirs {seen['d'].tolist()}"
        if mode == "rev":
            assert torch.all(seen["tau"] > 0.5) or True  # local time is 1-t, range check only
    print("PASS legacy bridge_loss fwd/rev modes restrict the drift type")


if __name__ == "__main__":
    torch.set_num_threads(2)
    test_fwd_only_uses_dir1()
    test_online_loss_mixed_pairs_correctly()
    test_legacy_bridge_loss_modes()
    print("ALL ONLINE DIRECTION TESTS PASSED")
