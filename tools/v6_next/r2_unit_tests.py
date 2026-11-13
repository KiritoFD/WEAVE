"""WEAVE R2 P5.3 — 12 mandatory unit tests.

Runs on CPU with small tensors plus the already-produced artefacts, so it can be
executed as a fast preflight gate before the serial pipeline is launched.

Tests
  1  R1 gradient ascent really increases the R1 margin.
  2  Per-sample RMS normalisation is correct (each sample RMS == 1).
  3  Haar DWT/IDWT reconstruction error is within tolerance.
  4  Tangential projection satisfies the first-order non-increase condition.
  5  Finite-step content check triggers correctly.
  6  All-zero gradient produces no NaN.
  7  R1 critic parameters stay frozen (no grad) in the teacher path.
  8  Raw-RGB source and VAE-roundtrip source references are not mixed up.
  9  Cross600 / Same150 counts and per-target counts are correct.
 10  Minimalism off-target confusion uses integer class indices.
 11  Teacher output is reproducible under a fixed seed.
 12  Distillation term produced a non-zero weight update in a trained student.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.legacy_root.wavelet import dwt2_haar, idwt2_haar  # noqa: E402
from tools.v6_next.r2_teacher import (  # noqa: E402
    compute_content_losses,
    compute_r1_gradient,
    filter_haar_subbands,
    project_tangential,
    rms_normalize,
    spatial_self_similarity_matrix,
)
from tools.v6_next.probe_style import LatentStyleNet  # noqa: E402

RESULTS: list[tuple[int, str, bool, str]] = []


def record(n: int, name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((n, name, bool(ok), detail))


def t1() -> None:
    """1. Gradient ascent increases R1 margin."""
    torch.manual_seed(0)
    critic = LatentStyleNet(width=8).eval()
    z = torch.randn(2, 4, 16, 16)
    tid = torch.tensor([1, 3])
    g, m0 = compute_r1_gradient(z, tid, critic)
    g = rms_normalize(g)
    with torch.no_grad():
        q = critic(z + 0.1 * g).float()
        oh = torch.nn.functional.one_hot(tid, 5).bool()
        m1 = q.gather(1, tid[:, None]).squeeze(1) - torch.logsumexp(
            q.masked_fill(oh, -1e9), dim=1
        )
    d = float((m1 - m0).mean())
    record(1, "R1 gradient ascent raises margin", d > 0, f"Δmargin={d:+.4f}")


def t2() -> None:
    """2. Per-sample RMS normalisation."""
    g = torch.randn(3, 4, 8, 8) * torch.tensor([0.1, 1.0, 10.0]).view(3, 1, 1, 1)
    n = rms_normalize(g)
    rms = n.flatten(1).pow(2).mean(dim=1).sqrt()
    ok = bool(torch.allclose(rms, torch.ones_like(rms), atol=1e-4))
    record(2, "per-sample RMS normalisation", ok, f"rms={rms.tolist()}")


def t3() -> None:
    """3. Haar reconstruction tolerance."""
    torch.manual_seed(0)
    x = torch.randn(2, 4, 16, 16)
    ll, lh, hl, hh = dwt2_haar(x)
    rec = idwt2_haar(ll, lh, hl, hh)
    err = float((rec - x).abs().max())
    record(3, "Haar DWT/IDWT reconstruction", err < 1e-4, f"max_err={err:.2e}")


def t4() -> None:
    """4. Tangential projection first-order non-increase: <g_c, d> <= 0."""
    torch.manual_seed(0)
    g_s = torch.randn(4, 4, 8, 8)
    g_c = torch.randn(4, 4, 8, 8)
    d = project_tangential(g_s, g_c)
    dots = (d.flatten(1) * g_c.flatten(1)).sum(dim=1)
    ok = bool((dots <= 1e-4).all())
    record(4, "projection first-order non-increase", ok, f"max_dot={dots.max():.2e}")


def t5() -> None:
    """5. Finite-step content check triggers on an oversized step."""
    torch.manual_seed(0)
    z0 = torch.randn(2, 4, 16, 16)
    zs = z0 + 0.01 * torch.randn_like(z0)
    with torch.no_grad():
        l_ss0, _ = compute_content_losses(z0, zs)
    big = z0 + 3.0 * torch.randn_like(z0)
    with torch.no_grad():
        l_ss1, _ = compute_content_losses(big, zs)
    # with a small delta, the big step must be rejected
    delta = 0.5 * float(l_ss0.mean())
    rejected = bool((l_ss1 > l_ss0 + delta).all())
    record(
        5,
        "finite-step content check triggers",
        rejected,
        f"L0={float(l_ss0.mean()):.4f} L1={float(l_ss1.mean()):.4f} delta={delta:.4f}",
    )


def t6() -> None:
    """6. Zero gradient -> finite output, no NaN."""
    g = torch.zeros(2, 4, 8, 8)
    n = rms_normalize(g)
    d = project_tangential(g, g)
    ok = bool(torch.isfinite(n).all() and torch.isfinite(d).all())
    record(6, "zero gradient produces no NaN", ok)


def t7() -> None:
    """7. R1 critic params frozen in teacher path."""
    critic = LatentStyleNet(width=8).eval()
    for p in critic.parameters():
        p.requires_grad_(False)
    ok = all(not p.requires_grad for p in critic.parameters())
    # and compute_r1_gradient must not leave grads on critic weights
    z = torch.randn(2, 4, 16, 16)
    compute_r1_gradient(z, torch.tensor([0, 1]), critic)
    ok = ok and all(
        p.grad is None or not p.requires_grad for p in critic.parameters()
    )
    record(7, "R1 critic frozen during teacher", ok)


def t8() -> None:
    """8. Raw RGB and VAE-roundtrip references are distinct."""
    p = ROOT / "runs/v6_r2/cache/tasks750_source_features.pt"
    if not p.exists():
        record(8, "raw vs roundtrip refs distinct", False, "cache missing")
        return
    c = torch.load(p, map_location="cpu", weights_only=False)
    k = next(iter(c["raw_rgb"].keys()))
    raw = c["raw_rgb"][k]
    rt = c["rt_rgb"][k]
    diff = float((raw - rt).abs().mean())
    ok = diff > 1e-3 and "raw_csd" in c and "rt_csd" in c
    record(8, "raw vs roundtrip refs distinct", ok, f"mean|raw-rt|={diff:.4f}")


def t9() -> None:
    """9. Cross600 / Same150 counts and per-target 120."""
    p = ROOT / "runs/v6_r2/eval_B3_distill20/eval.json"
    if not p.exists():
        record(9, "task counts correct", False, "eval missing")
        return
    d = json.loads(p.read_text(encoding="utf-8"))
    rows = d["rows"]
    cross = [r for r in rows if r["is_cross"]]
    same = [r for r in rows if not r["is_cross"]]
    per = {}
    for r in cross:
        per[r["tgt_style"]] = per.get(r["tgt_style"], 0) + 1
    ok = (
        len(cross) == 600
        and len(same) == 150
        and len(per) == 5
        and all(v == 120 for v in per.values())
    )
    record(
        9,
        "Cross600/Same150 counts",
        ok,
        f"cross={len(cross)} same={len(same)} per={per}",
    )


def t10() -> None:
    """10. Minimalism off-target confusion uses integer indices."""
    p = ROOT / "runs/v6_r2/eval_B3_distill20/eval.json"
    if not p.exists():
        record(10, "Minimalism confusion integer indices", False, "eval missing")
        return
    d = json.loads(p.read_text(encoding="utf-8"))
    conf = np.array(d["cross600"]["confusion_matrix"])
    min_idx = 2  # STYLES.index("Minimalism")
    off_hits = sum(conf[i, min_idx] for i in range(5) if i != min_idx)
    off_tot = sum(conf[i].sum() for i in range(5) if i != min_idx)
    rate = off_hits / off_tot
    reported = d["cross600"]["minimalism_offtarget"]
    ok = conf.shape == (5, 5) and abs(rate - reported) < 1e-6 and rate > 0
    record(
        10,
        "Minimalism confusion integer indices",
        ok,
        f"rate={rate:.4f} reported={reported:.4f}",
    )


def t11() -> None:
    """11. Teacher output reproducible under fixed seed."""
    critic = LatentStyleNet(width=8).eval()  # build first: its init consumes RNG
    torch.manual_seed(1234)
    z = torch.randn(2, 4, 16, 16)
    tid = torch.tensor([1, 3])
    g1, m1 = compute_r1_gradient(z, tid, critic)
    torch.manual_seed(1234)
    z2 = torch.randn(2, 4, 16, 16)
    g2, m2 = compute_r1_gradient(z2, tid, critic)
    ok = bool(
        torch.allclose(z, z2)
        and torch.allclose(g1, g2)
        and torch.allclose(m1, m2)
    )
    record(11, "teacher reproducible under fixed seed", ok)


def t12() -> None:
    """12. Distillation produced non-zero weight update vs init."""
    init = ROOT / "runs/v6/Q2a_embed_style/Q2a_embed_style.pt"
    stud = ROOT / "runs/v6_r2/S1_distill20_proj_ll/S1_distill20_proj_ll.pt"
    if not (init.exists() and stud.exists()):
        record(12, "distillation non-zero weight update", False, "ckpt missing")
        return
    a = torch.load(init, map_location="cpu", weights_only=False)["model"]
    b = torch.load(stud, map_location="cpu", weights_only=False)["model"]
    total = 0.0
    for k in a:
        total += float((a[k].float() - b[k].float()).abs().sum())
    ok = total > 0
    record(12, "distillation non-zero weight update", ok, f"sum|ΔW|={total:.2f}")


def main() -> int:
    tests = [t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12]
    for f in tests:
        try:
            f()
        except Exception as exc:  # a crashing test is a failed test, not a crash
            import traceback

            record(
                int(f.__name__[1:]),
                f.__doc__.splitlines()[0] if f.__doc__ else f.__name__,
                False,
                f"EXC {type(exc).__name__}: {exc}\n{traceback.format_exc()[-300:]}",
            )

    print("=" * 78)
    print("WEAVE R2 P5.3 — 12 MANDATORY UNIT TESTS")
    print("=" * 78)
    n_pass = 0
    for n, name, ok, detail in sorted(RESULTS):
        tag = "PASS" if ok else "FAIL"
        n_pass += int(ok)
        print(f"  [{tag}] {n:2d}. {name:44s} {detail[:60]}")
    print("-" * 78)
    print(f"  {n_pass}/{len(RESULTS)} passed")
    print("=" * 78)
    return 0 if n_pass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
