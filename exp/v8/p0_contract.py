"""P0: minimal implementation contract for V8.

These are the checks that must pass before any 24h arm runs, because each has
already caused a wasted experimental round somewhere in this repository:

  1. the correct Oracle direction is +grad(margin); a probe that minimises
     softplus(-margin) is NOT the same update and must not be labelled "fwd";
  2. u=0 through the controlled generator must equal the original checkpoint
     forward bit for bit (any control must start as a no-op);
  3. controls and targets must be PER SAMPLE (an old bug applied tid[0] to a
     whole batch);
  4. frozen assets must be loadable from the paths the queue will use.

Run on 3060:  python3 -m exp.v8.p0_contract
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
B4 = REPO / "runs/v6/P1_B4_distill50/P1_B4_distill50.pt"
Q2A = REPO / "runs/v6/Q2a_embed_style/Q2a_embed_style.pt"
R1 = REPO / "runs/v6_next/probes/r1_r1.pt"


def check_assets():
    missing = [str(p) for p in (B4, Q2A, R1) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing frozen assets: {missing}")
    return {"B4": str(B4), "Q2A": str(Q2A), "R1": str(R1),
            "B4_exists": B4.exists(), "Q2A_exists": Q2A.exists(), "R1_exists": R1.exists()}


def check_oracle_sign(device="cuda"):
    """+grad(margin) must increase the target margin; the minimising direction
    must decrease it. This pins the sign the whole B family depends on."""
    from src.infer import load_critic
    critic = load_critic(device)
    torch.manual_seed(0)
    z = torch.randn(4, 4, 64, 64, device=device)
    t = torch.tensor([0, 1, 2, 3], device=device)
    zr = z.detach().clone().requires_grad_(True)
    margin = _margin(critic, zr, t)
    g = torch.autograd.grad(margin.sum(), zr)[0]
    d = g / g.flatten(1).pow(2).mean(1).sqrt().clamp_min(1e-12).view(-1, 1, 1, 1)
    m0 = float(margin.mean())
    m_plus = float(_margin(critic, (z + 0.1 * d).detach(), t).mean())
    m_minus = float(_margin(critic, (z - 0.1 * d).detach(), t).mean())
    if not (m_plus > m0 > m_minus):
        raise AssertionError(
            f"+grad(margin) must raise the margin: m0={m0:.4f} "
            f"plus={m_plus:.4f} minus={m_minus:.4f}")
    return {"margin_base": m0, "margin_plus": m_plus, "margin_minus": m_minus,
            "sign_correct": True}


def _margin(critic, z, tid):
    logits = critic(z.float()).float()
    mask = torch.nn.functional.one_hot(tid.long(), 5).bool()
    others = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits.gather(1, tid[:, None]).squeeze(1) - torch.logsumexp(others, 1)


def check_zero_control(device="cuda"):
    """u=0 must reproduce the checkpoint forward exactly, for several specs."""
    from src.infer import canonicalizer, load_c3unet
    from exp.v8.controls import ControlSpec, Controls, assert_zero_control_equivalence, wrap
    base, _ = load_c3unet(Q2A, device)
    canon = canonicalizer(device)
    torch.manual_seed(0)
    z = torch.randn(3, 4, 64, 64, device=device)
    tid = torch.tensor([0, 2, 4], device=device)
    with torch.no_grad():
        c = canon(z, mode="c1_gentle", augment=False)
    out = {}
    for name, spec in (("A0", ControlSpec()), ("A1", ControlSpec(palette=False)),
                       ("A2", ControlSpec()), ("A4", ControlSpec(local_rank=2))):
        gen = wrap(base, spec, device)
        delta = assert_zero_control_equivalence(gen, c, 0.0, c, tid)
        out[name] = {"dim": spec.dim, "max_abs_diff": delta}
    # and a non-zero control must actually change the output
    gen = wrap(base, ControlSpec(), device)
    ctrl = Controls.zeros(3, device)
    with torch.no_grad():
        ctrl.raw.add_(0.05)
        ref = gen.base(c, 0.0, c, tid)
        moved = gen(c, 0.0, c, tid, controls=ctrl)
    if float((ref - moved).abs().max()) == 0.0:
        raise AssertionError("a non-zero control produced no change at all")
    out["nonzero_control_effect"] = float((ref - moved).abs().max())
    return out


def check_per_sample(device="cuda"):
    """Controls and targets must act per sample, not batch-wide."""
    from src.infer import canonicalizer, load_c3unet
    from exp.v8.controls import ControlSpec, Controls, wrap
    base, _ = load_c3unet(Q2A, device)
    canon = canonicalizer(device)
    torch.manual_seed(0)
    z = torch.randn(4, 4, 64, 64, device=device)
    tid = torch.tensor([0, 1, 2, 3], device=device)
    with torch.no_grad():
        c = canon(z, mode="c1_gentle", augment=False)
    # different targets must give different outputs
    gen = wrap(base, ControlSpec(), device)
    with torch.no_grad():
        o = gen.base(c, 0.0, c, tid)
    same_target = gen.base(c, 0.0, c, torch.zeros_like(tid))
    if torch.allclose(o, same_target):
        raise AssertionError("target style had no per-sample effect")
    # a per-sample control must not propagate across the batch
    ctrl = Controls.zeros(4, device)
    with torch.no_grad():
        ctrl.raw[0].add_(0.3)
        a = gen(c, 0.0, c, tid, controls=ctrl)
        ctrl2 = Controls.zeros(4, device)
        b = gen(c, 0.0, c, tid, controls=ctrl2)
    changed = (a - b).abs().flatten(1).max(1).values
    if not (changed[0] > 0 and float(changed[1:].max()) == 0.0):
        raise AssertionError(f"control leaked across samples: {changed.tolist()}")
    return {"target_effect": float((o - same_target).abs().max()),
            "per_sample_change": changed.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    results = {}
    results["assets"] = check_assets()
    results["oracle_sign"] = check_oracle_sign(a.device)
    results["zero_control"] = check_zero_control(a.device)
    results["per_sample"] = check_per_sample(a.device)
    print("P0 CONTRACT PASSED")
    for k, v in results.items():
        print(f"  {k}: {v}")
    return results


if __name__ == "__main__":
    main()
