"""Real training-step probe: measure the EXACT optimisation step the trainer runs.

The earlier bench measured a single forward/backward with the policy output
detached, which understated memory: the actual A3 step backpropagates the style
loss through the FROZEN generator activations as well. This probe calls the same
code path train_arm uses, so the number it reports is the one that decides
whether the arm can run at all.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]


def one_a3_step(batch: int, device: str = "cuda", local: bool = False):
    """Exactly the A3/A4 update, including the frozen-generator activation graph."""
    from src.infer import canonicalizer, load_c3unet, load_critic
    from exp.v8.controls import ControlSpec, Controls, wrap
    from exp.v8.energy import RealSupport, SupportedEnergy

    base, _ = load_c3unet(REPO / "runs/v6/Q2a_embed_style/Q2a_embed_style.pt", device)
    canon = canonicalizer(device)
    critic = load_critic(device)
    spec = ControlSpec(local_rank=2 if local else 0)
    gen = wrap(base, spec, device)
    policy = nn.Sequential(nn.Linear(spec.dim + 5, 256), nn.SiLU(),
                           nn.Linear(256, spec.dim)).to(device)
    nn.init.zeros_(policy[-1].weight); nn.init.zeros_(policy[-1].bias)
    opt = torch.optim.AdamW(policy.parameters(), lr=2e-4, weight_decay=1e-4)
    support_path = REPO / "runs/v8/real_support.pt"
    support = None
    if support_path.exists():
        st = torch.load(support_path, map_location="cpu", weights_only=False)
        support = RealSupport(st["prototypes"].to(device), st["sigma"].to(device))
    scorer = SupportedEnergy(critic, support, 0.1)

    z_s = torch.randn(batch, 4, 64, 64, device=device)
    tid = torch.randint(0, 5, (batch,), device=device)
    with torch.no_grad():
        c = canon(z_s, mode="c1_gentle", augment=False)
        z0 = base(c, 0.0, c, tid).detach()
    oh = nn.functional.one_hot(tid, 5).float()
    u = policy(torch.cat([torch.randn(batch, spec.dim, device=device), oh], 1))
    out = gen(c, 0.0, c, tid, controls=Controls.from_tensor(u, spec))
    margin = scorer.value(out, tid)
    loss = (nn.functional.softplus(-margin).mean()
            + 1.0 * (out - z0).square().flatten(1).mean()
            + 1e-3 * u.square().mean())
    opt.zero_grad(); loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    opt.step()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="A3")
    ap.add_argument("--batches", type=int, nargs="+", default=[96, 192])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()
    local = a.arm == "A4"
    fn = lambda b, dev: one_a3_step(b, dev, local=local)

    for b in a.batches:
        try:
            torch.cuda.empty_cache()
            fn(b, a.device)                     # warm / compile
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            for _ in range(a.repeats):
                fn(b, a.device)
            torch.cuda.synchronize()
            dt = (time.time() - t0) / a.repeats
            peak = torch.cuda.max_memory_allocated() / 2**20
            print(json.dumps({"arm": a.arm, "microbatch": b, "ok": True,
                              "sec_per_update": round(dt, 4),
                              "peak_vram_mb": round(peak, 1),
                              "samples_per_sec": round(b / dt, 1)}), flush=True)
        except torch.OutOfMemoryError as e:
            print(json.dumps({"arm": a.arm, "microbatch": b, "ok": False,
                              "error": "OOM", "detail": str(e)[:100]}), flush=True)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
