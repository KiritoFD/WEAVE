"""CPU-only unit tests for all v7r_next modules.

Run on 3060 WSL (or local): python3 -m tools.v7r_next.tests
No GPU, no 750 eval; verifies shape / gradient / detachment / clip / bootstrap
/ frontier / step0==B4 / seed isolation / patch alignment.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.v7r_next import common as C  # noqa: E402
from tools.v7r_next import patch_transport as PT  # noqa: E402
from tools.v7r_next import corrector as CO  # noqa: E402
from tools.v7r_next import distill as DI  # noqa: E402
from tools.v7r_next import evaluate as EV  # noqa: E402

OK = []


def check(cond, label):
    if not cond:
        raise AssertionError(label)
    OK.append(label)
    print("  [PASS]", label, flush=True)


def main():
    C.set_seed()
    print("=== common ===")
    check(specificity_from_scores_ok(), "specificity formula")
    check(C.tasks_sha() == C.sha256(C.TASKS), "tasks sha")

    print("=== patch_transport ===")
    z = torch.randn(2, 4, 64, 64)
    q, s = PT.phi_geom(z), PT.phi_style(z)
    check(q.shape == (2, PT.N_PATCH, PT.GEOM_DIM), "geom shape")
    check(s.shape == (2, PT.N_PATCH, PT.STYLE_DIM), "style shape")
    check(q.shape[1] == s.shape[1] == PT.N_PATCH, "patch grid ALIGNED")
    fake = {"centroids": torch.randn(PT.N_GROUPS, PT.GEOM_DIM),
            "styles": {st: [{"a": torch.randn(8, PT.STYLE_DIM),
                             "idx": torch.arange(8)}
                            for _ in range(PT.N_GROUPS)]
                       for st in C.STYLES}}
    for mode in ("geom_soft", "geom_ot", "uniform", "geom_shuf", "wrong_tgt"):
        zz = z.clone().requires_grad_(True)
        l, info = PT.cpt_loss(zz, q, fake, 0, mode)
        l.backward()
        check(torch.isfinite(l) and zz.grad.abs().sum() > 0,
              f"cpt loss {mode}")
    q2 = q.clone().requires_grad_(True)
    l2, _ = PT.cpt_loss(z, q2, fake, 0)
    leaked = False
    try:
        gq = torch.autograd.grad(l2, q2, allow_unused=True)[0]
        leaked = gq is not None
    except RuntimeError:
        leaked = False          # loss graph contains no q2 -> correct
    check(not leaked, "geometry DETACHED: no gradient reaches q_src")
    check(torch.isfinite(PT.cpt_global_loss(z, fake, 0)), "global loss")

    print("=== corrector ===")
    class Canon:
        def __call__(self, x, mode=None, augment=False): return x
    for mode in ("geom_ot", "uniform", "wrong_tgt"):
        ed = CO.ConstrainedEditor(z, z, 0, Canon(), fake, q, mode=mode,
                                  steps=4, rho=0.1)
        zz, audit = ed.run()
        check(zz.shape == z.shape and len(audit) == 4, f"editor {mode}")
        rms = (zz - z).flatten(1).pow(2).mean(1).sqrt().max()
        check(float(rms) <= 0.1 + 1e-4, f"hard RMS constraint {mode}")
        check(all("grad_cos_cpt_r1" in a for a in audit), "per-step audit fields")

    print("=== distill ===")
    m = DI.StudentCorrector(0.1)
    z0 = torch.randn(2, 4, 64, 64)
    c, sid, tid = torch.randn_like(z0), torch.zeros(2, dtype=torch.long), torch.ones(2, dtype=torch.long)
    check(float((m(c, z0, sid, tid) - z0).abs().max()) < 1e-6, "step0 == B4")
    with torch.no_grad():
        m.net.out_conv.weight.fill_(1.0)
    rms = (m(c, z0, sid, tid) - z0).flatten(1).pow(2).mean(1).sqrt().max()
    check(float(rms) <= 0.1 + 1e-4, "ClipRMS in FORWARD path")
    check(m.n_params() < 200_000, f"student params {m.n_params()} <200K")

    print("=== evaluate ===")
    rng = np.random.default_rng(0)
    r = EV.cluster_boot(rng.normal(.01, .05, 600), np.zeros(600),
                        [f"s{i//4}" for i in range(600)], n_boot=200)
    check(r["n"] == 600 and "excludes_zero" in r, "paired cluster bootstrap")
    fr = EV.frontier([{"name": "x", "dino_c": .8, "dino_s": .43},
                      {"name": "y", "dino_c": .7, "dino_s": .44},
                      {"name": "z", "dino_c": .6, "dino_s": .42}])
    check(len(fr) == 2, "2-D frontier")

    print("=== seed isolation ===")
    a = torch.randn(4)
    C.set_seed()
    b = torch.randn(4)
    C.set_seed()
    d = torch.randn(4)
    check(torch.equal(b, d) and not torch.equal(a, b), "seed reproducibility")

    print(f"\nALL {len(OK)} TESTS PASSED")
    C.write_json(C.RUNS / "selftest.json", {"passed": len(OK), "checks": OK})


def specificity_from_scores_ok():
    v = [0.5, 0.2, 0.2, 0.2, 0.2]
    return abs(C.specificity_from_scores(v, 0) - 0.3) < 1e-9


if __name__ == "__main__":
    main()
