"""V10.1 Gate G1: do retrieved local targets give better recovery directions?

Holdout paintings × 7 pre-registered degradations; direction cosine against
normalize(z_real - z^-). Methods: M0/M1/M2/M3 (retrieval targets), the V10
KDE direction (E4's), and the R1 margin gradient (reference floor).

Gate: best content-conditioned M > M0 AND > KDE AND > R1.
Run: python -m exp.v10.lm_exp1
Appends runs/v10_local_4070/p2_results.json (key "exp_lm_g1").
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .baselines import r1_direction
from .destyl import LADDER, degrade
from .featfn import holdout_latents, load_bank as load_kde, make_feat_fn, \
    pool_channel_stats
from .retrieval import MODES, LocalBank, lm_direction
from .splits import STYLES

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "runs/v10_local_4070"
METHODS = ["M0", "M1", "M2", "M3", "M1v", "M2v", "kde", "r1"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    device = a.device

    lb = LocalBank(OUT / "bank/lm_bank.pt", device)
    kde, hpsi = load_kde(device)
    stats = {k: v.to(device) for k, v in kde.stats.items()}
    fn_b = make_feat_fn(stats, hpsi, "B")
    from src.infer import load_critic
    critic = load_critic(device)

    res = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "n_per_style": a.n,
           "ops": LADDER, "per_style": {}}
    total = {m: [] for m in METHODS}
    t0 = time.time()
    for si, s in enumerate(STYLES):
        lat, _, _ = holdout_latents(si, a.n, seed=0)
        pc = pool_channel_stats(si)
        cos = {m: [] for m in METHODS}
        for op in LADDER:
            for i0 in range(0, lat.shape[0], a.batch):
                zr = lat[i0:i0 + a.batch].float().to(device)
                with torch.no_grad():
                    if op == "color03":
                        pm, ps = pc
                        zm = degrade(zr, op, pm, ps)
                    else:
                        zm = degrade(zr, op)
                    delta = (zr - zm).flatten(1)
                    delta = (delta / delta.norm(dim=1, keepdim=True)
                             .clamp_min(1e-12)).view_as(zr)
                tid = torch.full((zm.shape[0],), si, dtype=torch.long,
                                 device=device)
                for m in METHODS:
                    if m in MODES:
                        g = lm_direction(lb, zm, tid, m)
                    elif m == "kde":
                        g = kde.direction(zm, si, "B", fn_b)
                    else:
                        g = r1_direction(critic, zm, tid)
                    c = (g.flatten(1) * delta.flatten(1)).sum(1)
                    cos[m].extend(c.tolist())
        res["per_style"][s] = {m: round(sum(v) / len(v), 4) for m, v in cos.items()}
        for m in METHODS:
            total[m].extend(cos[m])
        print(f"[G1] {s} " + " ".join(f"{m}={res['per_style'][s][m]:.3f}"
              for m in METHODS) + f" {time.time()-t0:.0f}s", flush=True)

    res["mean"] = {m: round(sum(v) / len(v), 4) for m, v in total.items()}
    best_m = max(("M1", "M2", "M3"), key=lambda m: res["mean"][m])
    res["gate"] = {"best_m": best_m, "mean": res["mean"][best_m],
                   "M0": res["mean"]["M0"], "kde": res["mean"]["kde"],
                   "r1": res["mean"]["r1"],
                   "pass": bool(res["mean"][best_m] > res["mean"]["M0"]
                                and res["mean"][best_m] > res["mean"]["kde"]
                                and res["mean"][best_m] > res["mean"]["r1"])}
    print("[G1] mean", res["mean"])
    print("[G1] GATE", json.dumps(res["gate"]))
    p = OUT / "p2_results.json"
    cur = json.loads(p.read_text()) if p.exists() else {}
    cur["exp_lm_g1"] = res
    p.write_text(json.dumps(cur, indent=1))
    print("saved ->", p)


if __name__ == "__main__":
    main()
