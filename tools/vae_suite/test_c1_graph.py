"""CPU-only verification of the C1 isolated response branch graph topology.

The running trainer does, per update:
    (1) reconstruction microbatches: y = student(z); loss.backward()   [graph freed]
    (2) AFTER the accum loop: response_loss(student, teacher, zr, delta)
        builds a FRESH graph, then off_manifold_loss builds a THIRD, and
        l_extra = resp*l_r + off*l_o; l_extra.backward()
    (3) opt.step()

Risk flagged earlier but never GPU-tested (single GPU busy): step (2)'s
backward must NOT collide with step (1)'s already-consumed graphs. This test
reproduces that exact sequence on CPU with tiny mock decoders, so it needs NO
GPU and cannot disturb the running chain.

Also verifies: teacher params get NO grad, student input grad is finite/nonzero,
delta=0 gives ~0 response, and gradients from (1)+(2) ACCUMULATE (not overwrite).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2] if "Workspace" not in str(Path(__file__)) else Path("/root/Workspace/WEAVE_VAE")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "vae_suite"))

from tools.vae_suite.sd0_losses import (make_perturbation, off_manifold_loss,
                                        response_loss)


def _ok(c, label):
    print(f"  [{'PASS' if c else 'FAIL'}] {label}", flush=True)
    if not c:
        raise AssertionError(label)


class MockDec(torch.nn.Module):
    """Tiny stand-in for a 512px decoder; same callable contract."""

    def __init__(self, tag):
        super().__init__()
        self.lin = torch.nn.Linear(64, 64)   # 4 channels x 4 x 4 (real channel count)
        self.tag = tag

    def forward(self, z):
        return self.lin(z.flatten(1)).view(z.shape[0], 4, 4, 4)


def main():
    print("[C1-graph] CPU-only isolated-response-branch test", flush=True)
    torch.manual_seed(0)
    student = MockDec("S")
    teacher = MockDec("T")
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()
    opt = torch.optim.AdamW(student.parameters(), lr=1e-3)

    # mock uses the REAL latent channel count (4); `posterior` perturbation
    # builds a (B,4,1,1) channel field, so a 1-channel mock broadcasts to
    # (B,4,4,4) and corrupts the test. Keep 4 channels everywhere.
    z = torch.randn(4, 4, 4, 4)
    x = torch.randn(4, 4, 4, 4)
    zr = torch.randn(2, 4, 4, 4)
    # `posterior` kind has no hardcoded 64x64 interpolation, so it works on a
    # tiny CPU mock. corr_gauss itself is shape-tested separately below on a
    # real-shaped latent.
    delta = make_perturbation(zr, "posterior", 0.05,
                             generator=torch.Generator().manual_seed(0))

    # ---- step (1): reconstruction microbatches, backward frees the graph ----
    opt.zero_grad(set_to_none=True)
    l_rec = F.l1_loss(student(z), x)
    l_rec.backward()
    g_after_recon = student.lin.weight.grad.detach().clone()
    _ok(g_after_recon.abs().sum() > 0, "reconstruction backward produced grad")

    # ---- step (2): isolated response branch, AFTER recon graphs are freed ----
    try:
        l_r, rp = response_loss(student, teacher, zr, delta)
        l_o, _ = off_manifold_loss(student, teacher, zr + delta)
        l_extra = 0.17 * l_r + 0.075 * l_o
        l_extra.backward()
        _ok(True, "response+off backward did NOT raise 'backward a second time'")
    except RuntimeError as e:
        _ok(False, f"response backward raised: {e}")
        raise

    g_after_resp = student.lin.weight.grad.detach()
    # gradients must ACCUMULATE (step 1 + step 2), not overwrite
    _ok(bool((g_after_resp - g_after_recon).abs().sum() > 0),
        "response grad ADDED to recon grad (accumulate, not overwrite)")
    _ok(all(p.grad is None for p in teacher.parameters()),
        "teacher parameters still have NO grad after response backward")
    _ok(all(torch.isfinite(p.grad).all() for p in student.parameters()
            if p.grad is not None), "student grads finite")

    # ---- delta = 0 -> response ~ 0 (fresh graph) ----
    opt.zero_grad(set_to_none=True)
    l0, _ = response_loss(student, teacher, zr, torch.zeros_like(zr))
    _ok(float(l0) < 1e-6, f"delta=0 response ~0 ({float(l0):.2e})")

    # ---- a second full update must also be clean (no stale graph) ----
    opt.zero_grad(set_to_none=True)
    F.l1_loss(student(z), x).backward()
    l_r2, _ = response_loss(student, teacher, zr, delta)
    l_o2, _ = off_manifold_loss(student, teacher, zr + delta)
    (0.17 * l_r2 + 0.075 * l_o2).backward()
    opt.step()
    _ok(True, "a SECOND full (recon + isolated response) update also clean")

    # ---- amplitude must not be gameable by global shrink: ratio is measured ----
    opt.zero_grad(set_to_none=True)
    _, rp2 = response_loss(student, teacher, zr, delta)
    _ok(0.0 < rp2["resp_amp_ratio"] < 10.0,
        f"resp_amp_ratio finite & bounded ({rp2['resp_amp_ratio']:.3f})")

    # real-shaped latent: corr_gauss multiscale field must match 4x64x64
    zreal = torch.randn(2, 4, 64, 64)
    dreal = make_perturbation(zreal, "corr_gauss", 0.05,
                             generator=torch.Generator().manual_seed(1))
    _ok(tuple(dreal.shape) == tuple(zreal.shape),
        f"corr_gauss on real latent shape {tuple(dreal.shape)}")
    rms = float(dreal.pow(2).mean().sqrt() / zreal.pow(2).mean().sqrt())
    _ok(abs(rms - 0.05) < 0.01, f"corr_gauss RMS ratio ~0.05 ({rms:.4f})")

    print("\n[C1-graph] ALL PASSED — isolated C1 branch graph topology is sound", flush=True)


if __name__ == "__main__":
    main()
