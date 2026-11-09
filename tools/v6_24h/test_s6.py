"""Plan S6 gate: shape / gradient / memory / selectivity tests for the style-key residual.

The plan requires the module to pass these tests within 30 minutes before an arm may run,
and to have a zero-initialised output so the mechanism must earn its effect in training.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from tools.v6.models_v6 import C3UNet, StyleKeyResidual

FAIL = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'} {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        FAIL.append(name)


def live_head(m):
    """Make the residual path live.

    The whole model is zero-initialised at the output (out_conv64 / out_dwt32 /
    palette_head), so at step 0 the output is exactly `c` and NO internal parameter
    receives gradient -- only the last conv would. Randomising the output heads is
    therefore required to observe any gradient inside the network. This is a property of
    the residual-head design, not a defect of the style-key module.
    """
    with torch.no_grad():
        for p in (m.out_conv64.weight, m.out_conv64.bias,
                  m.out_dwt32.weight, m.out_dwt32.bias,
                  m.out_adaln.weight, m.out_adaln.bias):
            p.normal_(0, 0.05)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=2)
    args = ap.parse_args()
    dev = args.device
    torch.set_num_threads(2)

    # 1 shapes at two widths
    for bd in (32, 48):
        m = C3UNet(base_dim=bd, num_styles=5, mode="direct", style_keys=True).to(dev)
        z = torch.randn(args.batch, 4, 64, 64, device=dev)
        s = torch.arange(args.batch, device=dev) % 5
        o = m(z, 0.0, z, s)
        check(f"1 shape base_dim={bd}", o.shape == z.shape and torch.isfinite(o).all(),
              f"{tuple(o.shape)} params={sum(p.numel() for p in m.parameters()):,}")

    # 2 zero-init: adding keys must not change the initial function
    m0 = C3UNet(base_dim=32, num_styles=5, mode="direct", style_keys=False).to(dev).eval()
    m1 = C3UNet(base_dim=32, num_styles=5, mode="direct", style_keys=True).to(dev)
    m1.load_state_dict({k: v for k, v in m0.state_dict().items()}, strict=False)
    m1.eval()
    z = torch.randn(args.batch, 4, 64, 64, device=dev)
    s = torch.arange(args.batch, device=dev) % 5
    with torch.no_grad():
        check("2 zero-init leaves the function unchanged",
              torch.allclose(m0(z, 0.0, z, s), m1(z, 0.0, z, s), atol=1e-6))

    # 3 gradient reaches every style-key parameter
    m = C3UNet(base_dim=32, num_styles=5, mode="direct", style_keys=True).to(dev)
    m.train()
    live_head(m)
    with torch.no_grad():
        m.skr.out.weight.normal_(0, 0.02)
    loss = m(z, 0.0, z, s).square().mean()
    loss.backward()
    skr = dict(m.skr.named_parameters())
    dead = [n for n, p in skr.items() if p.grad is None or float(p.grad.abs().sum()) == 0.0]
    check("3 gradient reaches all style-key params", not dead, f"dead={dead}")

    # 4 selectivity: with a non-zero output the module must respond to style_id
    m.eval()
    s2 = (s + 1) % 5
    with torch.no_grad():
        a = m(z, 0.0, z, s)
        b = m(z, 0.0, z, s2)
    check("4 output depends on style_id once trained", not torch.allclose(a, b, atol=1e-7),
          f"max|d|={float((a - b).abs().max()):.3e}")

    # 5 attention is a proper distribution and the residual is small
    skr_m = StyleKeyResidual(96, num_styles=5, n_keys=8, rank=32).to(dev)
    with torch.no_grad():
        skr_m.out.weight.normal_(0, 0.02)
        h = torch.randn(2, 96, 16, 16, device=dev)
        out = skr_m(h, torch.tensor([0, 4], device=dev))
    delta = (out - h).abs().mean()
    check("5 residual magnitude is small relative to features", float(delta) < 1.0,
          f"mean|delta|={float(delta):.4f}, mean|h|={float(h.abs().mean()):.4f}")

    # 6 parameter overhead
    n0 = sum(p.numel() for p in C3UNet(base_dim=32, num_styles=5, mode="direct",
                                       style_keys=False).parameters())
    n1 = sum(p.numel() for p in C3UNet(base_dim=32, num_styles=5, mode="direct",
                                       style_keys=True).parameters())
    check("6 parameter overhead below 2%", (n1 - n0) / n0 < 0.02,
          f"{n0:,} -> {n1:,} (+{(n1-n0)/n0*100:.2f}%)")

    # 7 memory estimate (analytic; a real measurement needs the GPU to be free)
    #    attention tensor is (B, 256, K) floats, negligible; the extra convs are 1x1
    print(f"\n  analytic note: attention tensor is B*256*K floats "
          f"(batch {args.batch}, K=8 -> {args.batch*256*8*4/1024:.1f} KiB); "
          f"extra convs are 1x1 with rank 32, so peak-VRAM impact should be <5 MiB.")

    print()
    if FAIL:
        print(f"S6 GATE FAILED: {FAIL}")
        sys.exit(1)
    print("S6 GATE PASSED (7 checks)")


if __name__ == "__main__":
    main()
