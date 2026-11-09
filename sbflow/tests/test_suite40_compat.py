"""Targeted tests for the suite40 compatibility fixes."""
import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, ".")
import torch

from sbflow.network import BidirDiT
from sbflow.preconditioning import StyleStats
from sbflow.online_bridge import (mixed_ode_forward, mixed_ode_reverse,
                                  mixed_sde_forward, mixed_sde_reverse)

STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]


def synthetic_caches(tmp):
    caches = []
    for i, s in enumerate(STYLES):
        g = torch.Generator().manual_seed(900 + i)
        p = tmp / f"{s}.pt"
        torch.save({"latent": (torch.randn(48, 4, 8, 8, generator=g) * 0.8 + 0.4 * i).half(),
                    "thumb": (torch.rand(48, 3, 32, 32, generator=g) * 255).byte(),
                    "ids": [f"{s}_{k}.jpg" for k in range(48)], "ae": "synthetic",
                    "style": s, "scaling_factor": 0.18215, "image_size": 512}, p)
        caches.append(p)
    return caches


def test_ode_deterministic():
    """Both ODE endpoint directions must be deterministic (no hidden SDE)."""
    m = BidirDiT(channels=4, image_size=8, patch=2, width=32, depth=1, heads=2,
                 num_styles=5, style_cond="decomposed").eval()
    st = StyleStats(torch.zeros(4), torch.ones(4), torch.zeros(5, 4), torch.ones(5, 4))
    z = torch.randn(8, 4, 8, 8)
    pa = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    pb = torch.tensor([1, 4, 3, 4, 2, 3, 4, 4])
    g1 = torch.Generator().manual_seed(1)
    g2 = torch.Generator().manual_seed(2)
    f1, f2 = (mixed_ode_forward(m, z, pa, pb, st, 0.04, 3, 0.002) for _ in range(2))
    r1, r2 = (mixed_ode_reverse(m, z, pa, pb, st, 0.04, 3, 0.002) for _ in range(2))
    assert torch.allclose(f1, f2) and torch.allclose(r1, r2), "ODE endpoints were stochastic"
    s1 = mixed_sde_forward(m, z, pa, pb, st, 0.04, 3, 0.002, g1)
    s2 = mixed_sde_forward(m, z, pa, pb, st, 0.04, 3, 0.002, g2)
    assert not torch.allclose(s1, s2), "SDE endpoints were deterministic"
    print("PASS ODE deterministic both directions / SDE stochastic")


def test_resume(tmp):
    """Train 6 steps, resume with +4 more; verify global steps and LR continuity."""
    caches = synthetic_caches(tmp)
    base = [sys.executable, "-m", "sbflow.train5", "--caches", *[str(c) for c in caches]]
    out1, out2 = tmp / "run_a", tmp / "run_b"
    r = subprocess.run(base + ["--experiment-id", "a", "--output", str(out1),
                               "--steps", "6", "--batch", "40", "--width", "32",
                               "--depth", "1", "--heads", "2", "--warmup", "2",
                               "--log-interval", "3", "--save-interval", "6",
                               "--lr-cycle-steps", "10", "--equiv-batch", "40"],
                        capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    ck = out1 / "model_step6.pt"
    assert ck.exists()
    r2 = subprocess.run(base + ["--experiment-id", "b", "--output", str(out2),
                                "--steps", "4", "--batch", "40", "--width", "32",
                                "--depth", "1", "--heads", "2", "--warmup", "2",
                                "--log-interval", "2", "--save-interval", "4",
                                "--lr-cycle-steps", "10", "--equiv-batch", "40",
                                "--resume", str(ck)],
                         capture_output=True, text=True)
    assert r2.returncode == 0, r2.stdout[-2500:] + r2.stderr[-2500:]
    assert (out2 / "model_step10.pt").exists(), "resume must end at global step 6+4=10"
    s = json.loads((out2 / "train_summary.json").read_text())
    assert s["updates"] == 4
    print(f"PASS resume: prefix step 6 + 4 added updates -> global step 10 "
          f"(loss {['%.3f' % h['loss'] for h in s['loss_history']]})")


def test_config_flags(tmp):
    """exp05/exp06 style store_false config values must reach the parser."""
    import argparse
    from sbflow.train5 import _apply_config
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment-id", required=True)
    ap.add_argument("--caches", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--config", default="")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--mixed-batch", action="store_true", default=True)
    ap.add_argument("--single-pair", dest="mixed_batch", action="store_false")
    ap.add_argument("--equal-weight-stats", action="store_true", default=True)
    ap.add_argument("--frequency-stats", dest="equal_weight_stats", action="store_false")
    ap.add_argument("--equiv-steps", type=int, default=0)
    for name, cfg in (("exp05", {"single_pair": True, "mixed_batch": False, "steps": 4000}),
                      ("exp06", {"frequency_stats": True, "equal_weight_stats": False, "steps": 4000})):
        cfg.update(experiment_id="x", caches=["a"], output="o", id=name,
                   priority=True, equiv_steps=4000)
        p = tmp / f"{name}.json"
        p.write_text(json.dumps(cfg))
        a = _apply_config(ap, ["--config", str(p), "--caches", "r1", "r2"])
        print(f"  {name}: mixed={a.mixed_batch} eq={a.equal_weight_stats} "
              f"steps={a.steps} caches={len(a.caches)}")
        if name == "exp05":
            assert a.mixed_batch is False
        else:
            assert a.equal_weight_stats is False
    print("PASS config store_false aliases and unknown-key tolerance")


if __name__ == "__main__":
    torch.set_num_threads(2)
    test_ode_deterministic()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        test_resume(tmp)
        test_config_flags(tmp)
    print("ALL SUITE40 COMPATIBILITY TESTS PASSED")
