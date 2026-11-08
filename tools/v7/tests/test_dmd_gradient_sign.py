"""Analytic unit test for the V7 DMD gradient sign.

Toy problem:
    p = N(0, 1), q = N(m, 1), m > 0
For noise y_sigma = x + sigma*eps, the analytic epsilon predictions are
    eps_real = sigma * y_sigma / (1 + sigma^2)
    eps_fake = sigma * (y_sigma - m) / (1 + sigma^2)

Thus
    direction = (eps_real - eps_fake) / sigma = m / (1 + sigma^2) > 0
and the KL-decreasing update must be
    y <- y - alpha * direction
which moves q's mean toward 0.

The old V7 implementation used target = y + alpha*(eps_real - eps_fake),
so its MSE gradient pushed y in the opposite direction.

Run:
    python tools/v7/tests/test_dmd_gradient_sign.py
Exit code 0 means the corrected sign is verified and the old sign is detected.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

M = 2.0
SIGMA = 0.8
ALPHA = 0.5
N = 4096


def analytic_eps(y_sigma: torch.Tensor, mean: float, sigma: float) -> torch.Tensor:
    """E[eps | y_sigma] for x~N(mean,1), y_sigma=x+sigma*eps."""
    return sigma * (y_sigma - mean) / (1.0 + sigma**2)


def main() -> None:
    torch.manual_seed(0)
    # Samples from q=N(m,1), then add noise. We keep y_sigma fixed for both scores.
    x_q = M + torch.randn(N)
    eps = torch.randn(N)
    y_sigma = x_q + SIGMA * eps

    eps_real = analytic_eps(y_sigma, mean=0.0, sigma=SIGMA)
    eps_fake = analytic_eps(y_sigma, mean=M, sigma=SIGMA)
    direction = (eps_real - eps_fake) / SIGMA

    # A scalar m parameter. Generator output is y = m + fixed_sample_noise;
    # the MSE-to-target loss has gradient dL/dm = mean(y - target).
    # Gradient descent uses m <- m - lr * dL/dm, so positive dL/dm decreases m.
    noise_g = torch.randn(N) * 0.0  # keep y=m analytically clean
    m_correct = torch.nn.Parameter(torch.tensor(M))
    m_old = torch.nn.Parameter(torch.tensor(M))
    y_correct = m_correct.expand_as(y_sigma) + noise_g
    y_old = m_old.expand_as(y_sigma) + noise_g

    # Corrected: target = y.detach() - alpha * direction.
    with torch.no_grad():
        target_correct = y_sigma.detach() - ALPHA * direction
    loss_correct = 0.5 * torch.nn.functional.mse_loss(y_correct, target_correct)
    grad_correct = torch.autograd.grad(loss_correct, [m_correct])[0].item()

    # Old: target = y.detach() + alpha * (eps_real - eps_fake).
    with torch.no_grad():
        target_old = y_sigma.detach() + ALPHA * (eps_real - eps_fake)
    loss_old = 0.5 * torch.nn.functional.mse_loss(y_old, target_old)
    grad_old = torch.autograd.grad(loss_old, [m_old])[0].item()

    lr = 0.1
    m_after_correct = M - lr * grad_correct
    m_after_old = M - lr * grad_old

    out = {
        "toy": {"p_mean": 0.0, "q_mean": M, "sigma": SIGMA, "alpha": ALPHA, "n": N},
        "analytic_direction_mean": float(direction.mean().item()),
        "corrected_loss_gradient_dm": grad_correct,
        "old_loss_gradient_dm": grad_old,
        "m_after_one_gd_step_corrected": m_after_correct,
        "m_after_one_gd_step_old": m_after_old,
        "corrected_moves_toward_p": m_after_correct < M,
        "old_moves_away_from_p": m_after_old > M,
    }
    print(json.dumps(out, indent=2))

    assert direction.mean().item() > 0.0
    assert grad_correct > 0.0, "Corrected DMD loss gradient must be positive so GD decreases m"
    assert grad_old < 0.0, "Old implementation should be detected as sign-flipped"
    assert m_after_correct < M
    assert m_after_old > M
    print("P0 PASS: corrected DMD sign moves q=N(m,1) toward p=N(0,1); old sign moves away.")

    out_path = Path(__file__).resolve().parents[3] / "runs" / "v7" / "p0_dmd_gradient_sign.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
