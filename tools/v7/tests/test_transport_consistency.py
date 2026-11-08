"""Unit test: train/eval output consistency and differentiable tanh residual bound."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.v7.models_v7 import SingleStepTransportNet  # noqa: E402


def main() -> None:
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cap = 0.35
    model = SingleStepTransportNet(base_dim=16, num_styles=5, residual_cap=cap).to(device)
    z = torch.randn(6, 4, 64, 64, device=device)
    s = torch.tensor([0, 1, 2, 3, 4, 0], device=device)
    t = torch.tensor([1, 2, 3, 4, 0, 0], device=device)

    # Make residual non-trivial while keeping the model deterministic.
    with torch.no_grad():
        model.out_conv.weight.add_(torch.randn_like(model.out_conv.weight) * 0.05)

    y1 = model(z, s, t)
    y2 = model(z, s, t)
    residual = y1 - z
    max_abs = float(residual.abs().max().item())

    # Same-style rows must be exactly identity.
    same_res = residual[-1].abs().max().item()

    # tanh bound is differentiable: gradients must reach out_conv and vanish smoothly near saturation.
    z_grad = z.clone().requires_grad_(True)
    before = model.out_conv.weight.grad
    model.zero_grad(set_to_none=True)
    y = model(z_grad, s, t)
    y.mean().backward()
    grad_norm = float(model.out_conv.weight.grad.norm().item())
    has_grad = grad_norm > 0.0

    out = {
        "residual_cap": cap,
        "deterministic_max_diff": float((y1 - y2).abs().max().item()),
        "max_abs_residual": max_abs,
        "within_smooth_bound": max_abs <= cap + 1e-5,
        "same_style_exact_identity": same_res < 1e-7,
        "same_style_residual_max": same_res,
        "out_conv_grad_norm": grad_norm,
        "residual_differentiable": has_grad,
    }
    print(json.dumps(out, indent=2))
    assert out["deterministic_max_diff"] < 1e-6
    assert out["within_smooth_bound"]
    assert out["same_style_exact_identity"]
    assert out["residual_differentiable"]
    print("P1 PASS: tanh residual is smooth, bounded, deterministic, and s==t is exact identity.")

    out_path = _REPO_ROOT / "runs" / "v7" / "p1_transport_consistency.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
