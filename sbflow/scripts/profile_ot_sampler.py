"""Isolate the two fixed-latency costs inside the coupling step and test cheaper replacements.

Findings this script is designed to confirm:
  1. torch.multinomial on a (B,B) row-conditional matrix costs ~50 ms regardless of B.
  2. Sinkhorn's 200 Python-level iterations cost ~50 ms and converge in ~10.
Then it times three alternative samplers to give an infra implementation target.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from sbflow.ot import image_cost, sinkhorn


def timeit(fn, reps=20, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - t0) / reps


def multinomial_sampler(cond, gen):
    return torch.multinomial(cond, 1, replacement=True, generator=gen).squeeze(1)


def inverse_cdf_sampler(cond, gen):
    """Row-conditional inverse-CDF sampling; O(B^2) but no sort/scan kernel."""
    cdf = torch.cumsum(cond, dim=1)
    u = torch.rand((cond.shape[0], 1), device=cond.device, dtype=cond.dtype, generator=gen)
    return torch.searchsorted(cdf, u.contiguous()).squeeze(1).clamp_max(cond.shape[1] - 1)


def gumbel_sampler(cond, gen):
    """Gumbel-max trick: argmax(log p + gumbel noise)."""
    logp = cond.clamp_min(1e-30).log()
    g = -torch.log(-torch.log(torch.rand_like(cond, generator=gen).clamp_min(1e-20)).clamp_min(1e-20))
    return (logp + g).argmax(dim=1)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = torch.Generator(device=device).manual_seed(0)
    out = {"device": device, "rows": []}

    # Real thumbnails from the cache, to measure realistic convergence.
    c0 = torch.load("runs/blog_sbflow/cache/Impressionism.pt", map_location="cpu", weights_only=False)
    c1 = torch.load("runs/blog_sbflow/cache/Ukiyo_e.pt", map_location="cpu", weights_only=False)
    th0 = (c0["thumb"].float() / 255.0).to(device)
    th1 = (c1["thumb"].float() / 255.0).to(device)

    for b in (16, 64, 256):
        i0 = torch.randint(len(th0), (b,)).to(device)
        i1 = torch.randint(len(th1), (b,)).to(device)
        cost = image_cost(th0[i0], th1[i1], "rgb")
        pi, _ = sinkhorn(cost, 0.05, 200)
        cond = pi / pi.sum(1, keepdim=True).clamp_min(1e-30)

        rec = {"batch": b}
        rec["multinomial_ms"] = round(timeit(lambda: multinomial_sampler(cond, gen)), 3)
        rec["inverse_cdf_ms"] = round(timeit(lambda: inverse_cdf_sampler(cond, gen)), 3)
        rec["gumbel_ms"] = round(timeit(lambda: gumbel_sampler(cond, gen)), 3)

        # real-data convergence
        for tol in (1e-4, 1e-6, 1e-8):
            _, info = sinkhorn(cost, 0.05, 200, tol=tol, check_every=5)
            rec[f"iters_tol_{tol:g}"] = info["iters_used"]
            rec[f"rowerr_tol_{tol:g}"] = info["row_err"]
        _, info200 = sinkhorn(cost, 0.05, 200)
        rec["rowerr_200_fixed"] = info200["row_err"]

        # sampler agreement with multinomial (distributional sanity, single draw)
        a = multinomial_sampler(cond, gen)
        c = inverse_cdf_sampler(cond, gen)
        rec["same_draw_cdf_vs_multinomial"] = int((a == c).float().mean() * 100)

        out["rows"].append(rec)
        print(json.dumps(rec, indent=2), flush=True)

    Path("runs/blog_sbflow/bench/ot_sampler_profile.json").write_text(json.dumps(out, indent=2))
    print("saved runs/blog_sbflow/bench/ot_sampler_profile.json")


if __name__ == "__main__":
    main()
