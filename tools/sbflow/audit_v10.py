"""Four audits the reviewer asked for, before any retraining.

1. sampling_histogram   actual (source, target) distribution that enters the loss
2. condition_response   does changing only the target style change the predicted velocity?
3. direction_sanity     canonical order, reverse local time, EDM stats, integration direction
4. normalisation        equal-weight-per-style vs frequency-weighted global statistics
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path("/mnt/i/Latent_Style/WEAVE")
sys.path.insert(0, str(REPO))

from tools.sbflow.data5 import STYLE_NAMES, load_caches, make_sampler, pair_schedule
from tools.sbflow.infer5 import load_model5, solve_transfer
from tools.sbflow.preconditioning import (moment_coefficients, normalized_target,
                                    precondition_input, reconstruct_drift)

CACHE_ROOT = "/mnt/i/Latent_Style/pseudo_gt_24h/orig_mean_fp16"
FILES = ["00_Early_Renaissance.pt", "01_Impressionism.pt", "02_Minimalism.pt",
         "03_Rococo.pt", "04_Ukiyo_e.pt"]


# --------------------------------------------------------------------------- #
def audit_sampling(data, steps: int = 400, batch: int = 1024, seed: int = 42):
    """Replay the exact schedule + sampler path used by train5 and count what the loss sees."""
    g = torch.Generator().manual_seed(seed * 17 + 1)
    gs = torch.Generator().manual_seed(seed * 101 + 3)
    pairs = data.pairs()
    sched = pair_schedule(pairs, steps, gs)
    samplers = [make_sampler("perm", n, g) for n in data.counts]

    dirs = Counter()
    tgt = Counter()
    src = Counter()
    pair_count = Counter()
    for step in range(steps):
        a, b = sched[step]
        pair_count[(a, b)] += 1
        half = batch // 2
        # exactly what objective_bidir consumes: half forward (a->b), half reverse (b->a)
        ia = samplers[a].draw(half)
        ib = samplers[b].draw(half)
        dirs[f"{data.styles[a]}->{data.styles[b]}"] += half
        dirs[f"{data.styles[b]}->{data.styles[a]}"] += batch - half
        tgt[data.styles[b]] += half
        tgt[data.styles[a]] += batch - half
        src[data.styles[a]] += half
        src[data.styles[b]] += batch - half

    total = steps * batch
    print(f"\n=== 1. sampling histogram over {steps} updates (batch {batch}) ===")
    print(f"  pairs used: {len(pair_count)} of 10, each {min(pair_count.values())}-"
          f"{max(pair_count.values())} updates")
    print("  by direction (expected 5% each of 20):")
    for k in sorted(dirs):
        print(f"    {k:44s} {dirs[k]:8d}  {100*dirs[k]/total:5.2f}%")
    print("  by target style (expected 20% each):")
    for k in sorted(tgt):
        print(f"    {k:22s} {tgt[k]:8d}  {100*tgt[k]/total:5.2f}%")
    print("  by source style:")
    for k in sorted(src):
        print(f"    {k:22s} {src[k]:8d}  {100*src[k]/total:5.2f}%")
    lo = min(tgt.values()) / total
    hi = max(tgt.values()) / total
    print(f"  => target imbalance ratio max/min = {hi/lo:.3f}")
    return {"by_direction": dict(dirs), "by_target": dict(tgt), "by_source": dict(src),
            "pairs_used": len(pair_count), "target_ratio": hi / lo}


# --------------------------------------------------------------------------- #
def audit_condition(ckpt: str, data, device: str, n: int = 32, t_val: float = 0.5):
    """Fixed z and t; vary only the target style. If the velocity barely changes the
    model is ignoring the target condition and degenerates to an average field."""
    from tools.sbflow.infer5 import probability_flow_velocity

    ck, model, stats = load_model5(ckpt, device, "raw")
    eps = float(ck["config"]["eps"])
    z = data.z[0][:n].clone()                      # Early_Renaissance inputs
    t = torch.full((n,), t_val, device=device)

    vel = {}
    for tgt in range(5):
        a, b = (0, tgt) if 0 < tgt else (0, 1)
        if tgt == 0:
            continue
        vel[tgt] = probability_flow_velocity(model, z, t, a, b, stats, eps, 0.002,
                                             ck.get("objective", "bidir")).float()

    keys = sorted(vel)
    print(f"\n=== 2. target-style condition response (fixed z from {data.styles[0]}, t={t_val}) ===")
    base = vel[keys[0]]
    for k in keys:
        d = vel[k] - base
        print(f"  target={data.styles[k]:22s} vel_rms={vel[k].pow(2).mean().sqrt():.4f} "
              f"delta_vs_first={d.pow(2).mean().sqrt():.4f}")
    cos = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            c = float(F.cosine_similarity(vel[keys[i]].flatten(1), vel[keys[j]].flatten(1),
                                          dim=1).mean())
            cos.append(c)
    print(f"  pairwise cosine between different targets: mean={sum(cos)/len(cos):.4f} "
          f"min={min(cos):.4f} max={max(cos):.4f}")
    rel = float((vel[keys[-1]] - base).pow(2).mean().sqrt()
                / base.pow(2).mean().sqrt())
    verdict = "IGNORES target condition" if (sum(cos)/len(cos)) > 0.95 else "uses target condition"
    print(f"  relative change {rel:.4f} -> {verdict}")
    return {"pairwise_cos_mean": sum(cos)/len(cos), "min": min(cos), "max": max(cos),
            "relative_change": rel, "verdict": verdict}


# --------------------------------------------------------------------------- #
def audit_direction(device: str):
    """Verify the forward/reverse bookkeeping and the EDM reconstruction identities."""
    from tools.sbflow.preconditioning import PairMoments
    print("\n=== 3. canonical order / reverse time / EDM statistics ===")
    torch.manual_seed(0)
    eps = 0.04
    ch = 4
    pm = PairMoments(torch.zeros(ch), torch.ones(ch), torch.ones(ch),
                     torch.full((ch,), 2.0), torch.zeros(ch))
    t = torch.tensor([0.2, 0.6])
    x = torch.randn(2, ch, 4, 4)
    bplus = torch.randn(2, ch, 4, 4)
    bminus = torch.randn(2, ch, 4, 4)

    # forward: local time t, moments pair(a,b)
    xf, cf = precondition_input(x, t, eps, pm)
    yf = normalized_target(x, bplus, cf)
    assert torch.allclose(reconstruct_drift(x, yf, cf), bplus, atol=1e-4)
    # reverse: local time 1-t, moments reversed
    xb, cb = precondition_input(x, 1 - t, eps, pm.reversed())
    yb = normalized_target(x, bminus, cb)
    assert torch.allclose(reconstruct_drift(x, yb, cb), bminus, atol=1e-4)
    print("  forward/reverse EDM reconstruction identities: OK")

    # reversing the moments must swap mu0/mu1 and var0/var1 but keep cov
    r = pm.reversed()
    assert torch.allclose(r.mu0, pm.mu1) and torch.allclose(r.mu1, pm.mu0)
    assert torch.allclose(r.var0, pm.var1) and torch.allclose(r.cov01, pm.cov01)
    print("  PairMoments.reversed swaps endpoints and keeps cross covariance: OK")

    # probability flow must be antisymmetric under swapping (a,b) and reversing time
    from tools.sbflow.network import BidirDiT
    m = BidirDiT(channels=ch, image_size=4, patch=2, width=32, depth=1, heads=2,
                 num_styles=5, style_cond="decomposed").eval()
    from tools.sbflow.preconditioning import StyleStats
    st = StyleStats(torch.zeros(ch), torch.ones(ch), torch.zeros(5, ch), torch.ones(5, ch))
    from tools.sbflow.infer5 import probability_flow_velocity
    z = torch.randn(4, ch, 4, 4)
    tt = torch.full((4,), 0.5)
    v_ab = probability_flow_velocity(m, z, tt, 1, 3, st, eps, 0.002, "bidir")
    print(f"  v(a->b) finite={bool(torch.isfinite(v_ab).all())} "
          f"rms={float(v_ab.pow(2).mean().sqrt()):.4f}")

    # identity transfer must be exact
    out = solve_transfer(m, z, 2, 2, st, eps, steps=2)
    assert torch.allclose(out, z)
    print("  identity transfer (i -> i) exact: OK")

    # a->b then b->a must return closer to the start than a random point
    zz = torch.randn(8, ch, 4, 4)
    fwd = solve_transfer(m, zz, 0, 4, st, eps, steps=2)
    back = solve_transfer(m, fwd, 4, 0, st, eps, steps=2)
    e_cycle = float((back - zz).flatten(1).norm(dim=1).mean())
    e_rand = float((torch.randn_like(zz) - zz).flatten(1).norm(dim=1).mean())
    print(f"  cycle error={e_cycle:.3f} vs random-baseline={e_rand:.3f} "
          f"({'sane' if e_cycle < e_rand else 'SUSPICIOUS'})")
    return {"cycle_error": e_cycle, "random_baseline": e_rand}


# --------------------------------------------------------------------------- #
def audit_normalisation(device: str):
    """Equal-weight-per-style statistics vs frequency-weighted ones."""
    paths = [f"{CACHE_ROOT}/{f}" for f in FILES]
    print("\n=== 4. global latent normalisation ===")
    caches = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    lat = [c["latents"].float() for c in caches]
    names = [c["subdir"] for c in caches]
    counts = [t.shape[0] for t in lat]

    allz = torch.cat(lat, 0)
    freq_mean = allz.mean((0, 2, 3))
    freq_std = allz.var((0, 2, 3), unbiased=False).sqrt()
    eq_mean = torch.stack([t.mean((0, 2, 3)) for t in lat]).mean(0)
    eq_std = torch.stack([t.var((0, 2, 3), unbiased=False).sqrt() for t in lat]).mean(0)

    print("  counts:", dict(zip(names, counts)))
    print(f"  frequency-weighted mean: {[round(float(v), 4) for v in freq_mean]}")
    print(f"  equal-weight mean      : {[round(float(v), 4) for v in eq_mean]}")
    print(f"  mean L1 difference     : {float((freq_mean - eq_mean).abs().mean()):.5f}")
    print(f"  std  L1 difference     : {float((freq_std - eq_std).abs().mean()):.5f}")
    # how far does the majority class pull the centre?
    imp = lat[1]
    print(f"  Impressionism mean: {[round(float(v), 4) for v in imp.mean((0, 2, 3))]}")
    print(f"  Impressionism share of rows: {counts[1]/sum(counts):.4f}")
    return {"freq_mean": freq_mean.tolist(), "eq_mean": eq_mean.tolist(),
            "mean_l1": float((freq_mean - eq_mean).abs().mean())}


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/blog_sbflow/v10/exp_04L/model_step6000.pt")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--out", default="runs/blog_sbflow/v10/audit_04L.json")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_caches([Path(f"{CACHE_ROOT}/{f}") for f in FILES], device, STYLE_NAMES)
    print(f"loaded 5 styles: {dict(zip(data.styles, data.counts))}")

    out = {}
    out["sampling"] = audit_sampling(data, args.steps, args.batch)
    out["direction"] = audit_direction(device)
    out["normalisation"] = audit_normalisation(device)
    ck = REPO / args.ckpt
    if ck.exists():
        out["condition"] = audit_condition(str(ck), data, device)
    else:
        print(f"\n(2) skipped: {ck} missing")

    p = REPO / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved {p}")


if __name__ == "__main__":
    main()
