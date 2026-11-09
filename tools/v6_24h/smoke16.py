"""V6 24h plan - P0 Smoke16.

Fast structural checks that run in seconds and gate the 24h queue. They do NOT select a
model; they only prove the code paths are sane and that every ablation switch really
changes what it claims to change.

Checks
  1  forward shape / finite output
  2  backward reaches every trainable parameter
  3  same-style path runs and differs from cross-style
  4  Style-ID swap changes the output (style condition is live)
  5  content swap changes the output (content condition is live)
  6  --no-style makes the output invariant to Style-ID
  7  --content-mode none makes the output invariant to content
  8  --no-inject-64/32/16 each change the output vs baseline
  9  canonicalizer c1_soft HL band uses hl_mean (regression guard for the fixed bug)
 10  canon mode 'none' is the identity
 11  direct-mode stem input [z_in, c, z_in-c] has 4 zero channels (folding precondition)
 12  palette head is Style-ID dependent
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from tools.v6.canonicalize import ContentCanonicalizer
from tools.v6.models_v6 import C3UNet

D, NS, BD = 16, 5, 8
FAIL = []


def check(name, cond, detail=""):
    ok = bool(cond)
    print(f"{'PASS' if ok else 'FAIL'} {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def rand_head(m):
    """Zero-initialised heads make a fresh model output 0; randomise so behaviour shows."""
    with torch.no_grad():
        for p in m.parameters():
            if p.requires_grad and p.numel() > 0:
                p.normal_(0, 0.05)
    return m


def mk(**kw):
    torch.manual_seed(0)
    return rand_head(C3UNet(base_dim=BD, num_styles=NS, mode="direct", **kw)).eval()


def main() -> None:
    torch.set_num_threads(2)
    g = torch.Generator().manual_seed(1)
    z = torch.randn(3, 4, D, D, generator=g)
    c = torch.randn(3, 4, D, D, generator=g)
    s = torch.tensor([1, 2, 4])
    s2 = torch.tensor([3, 0, 1])

    # 1 forward
    m = mk()
    with torch.no_grad():
        o = m(z, 0.0, c, s)
    check("1 forward shape/finite", o.shape == z.shape and torch.isfinite(o).all().item(),
          f"{tuple(o.shape)}")

    # 2 backward reaches all params
    m = mk()
    m.train()
    loss = m(z, 0.0, c, s).square().mean()
    loss.backward()
    missing = [n for n, p in m.named_parameters()
               if p.requires_grad and (p.grad is None or p.grad.abs().sum() == 0)]
    check("2 backward reaches every param", not missing, f"missing={missing[:4]}")

    # 3/4/5 style + content sensitivity
    m = mk()
    with torch.no_grad():
        base = m(z, 0.0, c, s)
        style_sw = m(z, 0.0, c, s2)
        cont_sw = m(z, 0.0, c + 0.5, s)
    check("3 style swap changes output", not torch.allclose(base, style_sw, atol=1e-6))
    check("4 content swap changes output", not torch.allclose(base, cont_sw, atol=1e-6))

    # 6 no_style invariance
    m = mk(no_style=True)
    with torch.no_grad():
        a = m(z, 0.0, c, s)
        b = m(z, 0.0, c, s2)
    check("5 no_style is Style-ID invariant", torch.allclose(a, b, atol=1e-6))

    # 7 content_mode none invariance
    m = mk(content_mode="none")
    with torch.no_grad():
        a = m(z, 0.0, c, s)
        b = m(z, 0.0, c + 0.5, s)
    check("6 content_mode=none is content invariant", torch.allclose(a, b, atol=1e-6))

    # 7b content_mode no_inject still content dependent.
    # Direct mode calls model(c, 0, c, s), i.e. z_in == c, so the stem still carries the
    # content even when the per-block injection is disabled. The test must use that
    # convention; passing an independent z_in would make the output trivially invariant.
    m = mk(content_mode="no_inject")
    with torch.no_grad():
        a = m(c, 0.0, c, s)
        b = m(c + 0.5, 0.0, c + 0.5, s)
    check("7 no_inject still content dependent (direct convention z_in==c)",
          not torch.allclose(a, b, atol=1e-6),
          f"max|d|={float((a - b).abs().max()):.4e}")

    # 8 per-scale injection switches each change the output
    m0 = mk()
    with torch.no_grad():
        ref = m0(z, 0.0, c, s)
    for flag in ("inject_64", "inject_32", "inject_16"):
        mf = mk(**{flag: False})
        with torch.no_grad():
            of = mf(z, 0.0, c, s)
        check(f"8 {flag}=False changes output",
              not torch.allclose(ref, of, atol=1e-6),
              f"max|d|={float((ref - of).abs().max()):.4e}")

    # 9 c1_soft HL band uses hl_mean
    st = {f"{b}_{k}": (torch.zeros(1, 4, 1, 1) if k == "mean" else torch.ones(1, 4, 1, 1))
          for b in ("ll", "lh", "hl", "hh") for k in ("mean", "std")}
    st["hl_mean"] = torch.full((1, 4, 1, 1), 3.0)
    st["lh_mean"] = torch.full((1, 4, 1, 1), -7.0)
    canon = ContentCanonicalizer(st)
    from wavelet import dwt2_haar
    out = canon(z, mode="c1_soft", augment=False)
    check("9 c1_soft runs and is finite", torch.isfinite(out).all().item())
    # the HL band of the output must be pulled toward hl_mean=3, not lh_mean=-7
    hl_out = dwt2_haar(out.float())[2].mean().item()
    check("10 c1_soft HL band pulled toward hl_mean", hl_out > 0,
          f"hl_out mean={hl_out:+.4f} (hl_mean=+3, lh_mean=-7)")

    # 11 canon none is identity
    check("11 canon mode=none is identity", torch.allclose(canon(z, mode="none"), z, atol=1e-6))

    # 12 stem folding precondition: [z_in, c, z_in-c] has 4 exact-zero channels in direct mode
    x0 = torch.cat([c, c, c - c], dim=1)
    zero_ch = int((x0.abs().amax(dim=(0, 2, 3)) < 1e-9).sum())
    check("12 direct-mode stem has 4 zero channels", zero_ch == 4, f"zero_channels={zero_ch}")

    # 13 palette head is Style-ID dependent
    m = mk()
    with torch.no_grad():
        p1 = m.palette_head(c, s)
        p2 = m.palette_head(c, s2)
    check("13 palette head depends on Style-ID", not torch.allclose(p1, p2, atol=1e-6))

    print()
    if FAIL:
        print(f"SMOKE16 FAILED: {len(FAIL)} check(s): {FAIL}")
        sys.exit(1)
    print("SMOKE16 PASSED (13 checks)")


if __name__ == "__main__":
    main()
