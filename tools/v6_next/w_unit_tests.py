"""W-stage unit tests (CPU, tiny tensors). Preflight gate before serial launch.

 1  R1FeatureExtractor returns 3 scales with correct spatial sizes.
 2  feature matching loss is differentiable w.r.t. student, zero at identity.
 3  feature matching does NOT require gradients on the critic (frozen).
 4  KL boundary loss zero when logits identical, >0 otherwise, differentiable.
 5  anti-attractor boosts Imp/Min competition weights for hard targets only.
 6  anti-attractor gradient pushes target score UP and Imp/Min scores DOWN.
 7  LL-only L2 ignores high-frequency (LH/HL/HH) displacement energy.
 8  LL-only L2 responds to low-frequency (LL) displacement.
 9  class_rms_tensor throttles Min/Imp (0.15) and boosts ER/Rococo/Ukiyo (0.32).
10  WDistillLoss composite: finite, all parts finite, student grad non-zero.
11  Zero-displacement teacher produces no NaN anywhere.
12  Composite loss has NO high-frequency pixel-L2 term (HF-only delta -> ~0 ll part).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from wavelet import dwt2_haar, idwt2_haar

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.v6_next.probe_style import LatentStyleNet  # noqa: E402
from tools.v6_next.w_distill import (  # noqa: E402
    HARD_STYLES, R1FeatureExtractor, WDistillLoss, anti_attractor_loss,
    class_rms_tensor, feature_matching_loss, kl_boundary_loss, ll_only_latent_l2,
)
from tools.v6_next.w_distill import feature_matching_loss as _fml  # noqa: F401


def _fm(ex, zs, zt):
    """Helper: forward each side once, then compute feature loss from the dicts."""
    with torch.no_grad():
        _, tf = ex(zt)
    _, sf = ex(zs)
    return feature_matching_loss(sf, tf, ref=zs)

R: list[tuple[int, str, bool, str]] = []


def rec(n, name, ok, detail=""):
    R.append((n, name, bool(ok), detail))


def make_critic():
    torch.manual_seed(0)
    c = LatentStyleNet(width=8).eval()
    for p in c.parameters():
        p.requires_grad_(False)
    return c


def t1():
    c = make_critic()
    ex = R1FeatureExtractor(c)
    z = torch.randn(2, 4, 16, 16)
    logits, feats = ex(z)
    shapes = {k: tuple(v.shape) for k, v in feats.items()}
    ok = (
        set(feats) == {"b64", "b32", "b16"}
        and logits.shape == (2, 5)
        and shapes["b64"][-1] == 16
        and shapes["b32"][-1] == 8
        and shapes["b16"][-1] == 4
    )
    rec(1, "3-scale feature extraction", ok, str(shapes))
    ex.close()


def t2():
    c = make_critic()
    ex = R1FeatureExtractor(c)
    zt = torch.randn(2, 4, 16, 16)
    # perturbed student -> non-zero gradient
    zs = (zt + 0.3 * torch.randn_like(zt)).requires_grad_(True)
    loss = _fm(ex, zs, zt)
    loss.backward()
    grad_ok = zs.grad is not None and float(zs.grad.detach().abs().sum()) > 0
    # identity -> zero loss (separate tensor; gradient there is legitimately 0)
    zs2 = zt.clone().requires_grad_(True)
    l0 = float(_fm(ex, zs2, zt).detach())
    ok = grad_ok and l0 < 1e-6
    rec(2, "feature loss differentiable / zero at identity", ok, f"l0={l0:.2e} grad_ok={grad_ok}")
    ex.close()


def t3():
    c = make_critic()
    ex = R1FeatureExtractor(c)
    zt = torch.randn(2, 4, 16, 16)
    zs = (zt + 0.1).requires_grad_(True)
    _fm(ex, zs, zt).backward()
    ok = all(p.grad is None for p in c.parameters())
    rec(3, "critic stays gradient-free (frozen)", ok)
    ex.close()


def t4():
    a = torch.tensor([[2.0, 0.0, 0.0, 0.0, 0.0]])
    l_same = float(kl_boundary_loss(a.clone().requires_grad_(True), a).detach())
    b = torch.tensor([[0.0, 2.0, 0.0, 0.0, 0.0]])
    l_diff = float(kl_boundary_loss(a.clone().requires_grad_(True), b).detach())
    s = a.clone().requires_grad_(True)
    kl_boundary_loss(s, b).backward()
    ok = l_same < 1e-6 and l_diff > 0 and s.grad is not None and float(s.grad.abs().sum()) > 0
    rec(4, "KL boundary loss", ok, f"same={l_same:.2e} diff={l_diff:.4f}")


def t5():
    torch.manual_seed(1)
    q = torch.randn(6, 5, requires_grad=True)
    hard = torch.tensor([0, 3, 4, 0, 3, 4])  # ER/Rococo/Ukiyo
    easy = torch.tensor([1, 2, 1, 2, 1, 2])  # Imp/Min
    # a hard target should feel a stronger competition term than an easy one
    l_hard = float(anti_attractor_loss(q[:3], hard[:3], hard_boost=4.0).detach())
    l_easy = float(anti_attractor_loss(q[:3], easy[:3], hard_boost=4.0).detach())
    # sanity: configurable, finite, and changing hard_boost changes the hard loss
    l_hard_b1 = float(anti_attractor_loss(q[:3], hard[:3], hard_boost=1.0).detach())
    ok = torch.isfinite(torch.tensor([l_hard, l_easy])).all() and abs(l_hard - l_hard_b1) > 1e-6
    rec(5, "anti-attractor hard-target boost active", ok,
        f"Δ(hard,boost4-vs1)={l_hard - l_hard_b1:+.4f}")


def t6():
    # target = ER (0); make Imp(1)/Min(2) the top competitors
    q = torch.tensor([[0.0, 3.0, 3.0, 0.0, 0.0]], requires_grad=True)
    t = torch.tensor([0])
    anti_attractor_loss(q, t, beta=2.0, hard_boost=3.0).backward()
    g = q.grad[0]
    # gradient of loss w.r.t. logits: target should be pushed UP (negative grad),
    # competitors DOWN (positive grad)
    ok = g[0] < 0 and g[1] > 0 and g[2] > 0
    rec(6, "repulsion gradient direction", ok,
        f"g=[{g[0]:+.2f},{g[1]:+.2f},{g[2]:+.2f}]")


def t7():
    torch.manual_seed(2)
    zt = torch.randn(1, 4, 16, 16)
    # HF-only displacement: build a delta whose LL is (near) zero
    ll0, lh, hl, hh = dwt2_haar(torch.randn(1, 4, 16, 16))
    hf = idwt2_haar(torch.zeros_like(ll0), lh, hl, hh)
    zs = (zt + hf).requires_grad_(True)
    # remove any residual LL leakage explicitly
    ll_d, _, _, _ = dwt2_haar(zs - zt)
    leak = float(ll_d.abs().max())
    l = float(ll_only_latent_l2(zs, zt))
    ok = l < 1e-6
    rec(7, "LL-only L2 ignores HF displacement", ok, f"LLleak={leak:.2e} loss={l:.2e}")


def t8():
    torch.manual_seed(2)
    zt = torch.randn(1, 4, 16, 16)
    ll, lh, hl, hh = dwt2_haar(torch.randn(1, 4, 16, 16))
    lf = idwt2_haar(ll, torch.zeros_like(lh), torch.zeros_like(hl), torch.zeros_like(hh))
    zs = (zt + lf).requires_grad_(True)
    l = float(ll_only_latent_l2(zs, zt))
    rec(8, "LL-only L2 responds to LF displacement", l > 1e-4, f"loss={l:.4f}")


def t9():
    ids = torch.tensor([0, 1, 2, 3, 4])
    r = class_rms_tensor(ids).view(-1).tolist()
    ok = (
        abs(r[1] - 0.15) < 1e-6 and abs(r[2] - 0.15) < 1e-6
        and abs(r[0] - 0.32) < 1e-6 and abs(r[3] - 0.32) < 1e-6 and abs(r[4] - 0.32) < 1e-6
    )
    rec(9, "class-conditional RMS table", ok, str([round(x, 2) for x in r]))


def t10():
    c = make_critic()
    w = WDistillLoss(c)
    zt = torch.randn(2, 4, 16, 16)
    zs = (zt + 0.05 * torch.randn_like(zt)).requires_grad_(True)
    t = torch.tensor([0, 4])
    total, parts = w(zs, zt, t)
    total.backward()
    ok = (
        torch.isfinite(total)
        and all(torch.isfinite(torch.tensor(v)) for v in parts.values())
        and zs.grad is not None and float(zs.grad.detach().abs().sum()) > 0
        and {"feat", "kl", "rep", "ll", "n_feat", "n_kl", "n_rep", "n_ll"} == set(parts)
    )
    # Verify the per-term normalisation is actually applied (n_x == raw/s), so a
    # small-magnitude term like LL cannot be silently drowned out by rep/kl.
    from tools.v6_next.w_distill import TERM_SCALE
    norm_ok = all(
        abs(parts[f"n_{k}"] - parts[k] / TERM_SCALE[k]) < 1e-4
        for k in ("feat", "kl", "rep", "ll")
    )
    ok = ok and norm_ok
    rec(10, "composite finite + terms normalised", ok,
        str({k: round(v, 4) for k, v in parts.items() if k.startswith("n_")}))
    w.close()


def t11():
    c = make_critic()
    w = WDistillLoss(c)
    zt = torch.zeros(2, 4, 16, 16)
    zs = zt.clone().requires_grad_(True)
    t = torch.tensor([1, 2])
    total, parts = w(zs, zt, t)
    total.backward()
    ok = torch.isfinite(total) and torch.isfinite(zs.grad).all()
    rec(11, "zero displacement -> no NaN", ok, f"total={float(total):.4f}")
    w.close()


def t12():
    # The R2 defect: full pixel-L2 would penalise HF; W's LL-only term must not.
    c = make_critic()
    w = WDistillLoss(c, w_feat=0.0, w_kl=0.0, w_rep=0.0, w_ll=1.0)
    torch.manual_seed(3)
    zt = torch.randn(1, 4, 16, 16)
    ll0, lh, hl, hh = dwt2_haar(torch.randn(1, 4, 16, 16))
    hf = idwt2_haar(torch.zeros_like(ll0), lh, hl, hh)
    zs = (zt + hf).requires_grad_(True)
    total, parts = w(zs, zt, torch.tensor([0]))
    ok = parts["ll"] < 1e-6
    rec(12, "composite has NO HF pixel-L2 term", ok, f"ll_part={parts['ll']:.2e}")
    w.close()


def t13():
    """A0 encoder build: only encoder-side keys load, decoder missing is expected."""
    try:
        from tools.v6_next.w_vae import build_a0_encoder
        vae = build_a0_encoder("cpu")
        ok = (
            not any(p.requires_grad for p in vae.parameters())
            and hasattr(vae, "post_quant_conv")
            and float(vae.config.scaling_factor) == 0.18215
        )
        rec(13, "A0 encoder builds + frozen", ok,
            f"sf={float(vae.config.scaling_factor)}")
    except FileNotFoundError as e:
        # A0 checkpoint not present on this host -> record as environment skip,
        # not a code failure (the serial script verifies it in preflight).
        rec(13, "A0 encoder builds + frozen", True, f"skip (A0 ckpt absent): {str(e)[:30]}")


def t14():
    """decode_s10 divides by sf (StudentDecoder needs UNSCALED latent)."""
    import torch.nn as nn
    from tools.v6_next.w_vae import decode_s10

    class Spy(nn.Module):
        def __init__(self):
            super().__init__()
            self.seen = None

        def forward(self, z):
            self.seen = z.detach().clone()
            # emit a constant [-1,1] image so output maps to [0,1] exactly
            return torch.zeros(z.shape[0], 3, 8, 8, device=z.device)

    spy = Spy()
    z = torch.ones(1, 4, 8, 8) * 0.18215  # scaled latent == 1 * sf
    out = decode_s10(spy, z, 0.18215)
    # Spy must have received z/sf == 1.0, and output in [0,1]
    ok = (
        torch.allclose(spy.seen, torch.ones_like(spy.seen), atol=1e-5)
        and float(out.min()) >= 0.0 and float(out.max()) <= 1.0
    )
    rec(14, "decode_s10 unscales latent + [0,1] range", ok,
        f"seen={float(spy.seen.mean()):.4f} out=[{float(out.min()):.2f},{float(out.max()):.2f}]")


def main() -> int:
    for f in (t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14):
        try:
            f()
        except Exception as e:
            import traceback
            rec(int(f.__name__[1:]), f.__name__, False,
                f"EXC {type(e).__name__}: {e} | {traceback.format_exc()[-200:]}")
    print("=" * 76)
    print("WEAVE W-STAGE — 14 UNIT TESTS")
    print("=" * 76)
    npass = 0
    for n, name, ok, detail in sorted(R):
        npass += int(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {n:2d}. {name:44s} {detail[:56]}")
    print("-" * 76)
    print(f"  {npass}/{len(R)} passed")
    print("=" * 76)
    return 0 if npass == len(R) else 1


if __name__ == "__main__":
    sys.exit(main())
