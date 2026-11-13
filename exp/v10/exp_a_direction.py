"""Experiment A: auto-destylize -> recover direction consistency (plan §5-A).

For holdout paintings and the 7 pre-registered degradations:
    A_grad(method) = mean cos( g_method(z^-), normalize(z_real - z^-) )

Methods: V10-A / V10-B / V10-C (KDE variants) and baselines r1 / ld3 / v6global.
Gate (pre-registered): best V10 variant mean > r1 AND > ld3, winning >=4/5 styles.

Run: python -m exp.v10.exp_a_direction [--n 40] [--batch 8]
Writes runs/v10_local_4070/p2_results.json (key "exp_a").
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .baselines import Ld3Direction, V6GlobalDirection, normalize_dir, r1_direction
from .destyl import LADDER, degrade
from .featfn import holdout_latents, load_bank, make_feat_fn, pool_channel_stats
from .splits import STYLES

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "runs/v10_local_4070"
METHODS = ["V10-A", "V10-A_f", "V10-B", "V10-C", "r1", "ld3", "v6global"]
N_PER_STYLE = 40
BATCH = 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N_PER_STYLE)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    device = a.device

    bank, hpsi = load_bank(device)
    stats = {k: v.to(device) for k, v in bank.stats.items()}
    fns = {v: make_feat_fn(stats, hpsi, v)
           for v in ("A", "A_f", "B", "C", "B_color", "B_texture")}
    critic = None
    from src.infer import load_critic
    critic = load_critic(device)
    ld3 = Ld3Direction(device)
    v6g = V6GlobalDirection(device)

    res = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "n_per_style": a.n,
           "ops": LADDER, "per_style": {}, "per_op": {}}
    t0 = time.time()
    total_cos = {m: [] for m in METHODS}
    for si, s in enumerate(STYLES):
        lat, ids, files = holdout_latents(si, a.n, seed=0)
        pc = pool_channel_stats(si)
        cos = {m: [] for m in METHODS}
        for op in LADDER:
            for i0 in range(0, lat.shape[0], a.batch):
                zr = lat[i0:i0 + a.batch].float().to(device)
                with torch.no_grad():
                    zm = None
                    if op == "color03":
                        pm, ps = pc
                        zm = degrade(zr, op, pm, ps)
                    else:
                        zm = degrade(zr, op)
                    delta = normalize_dir(zr - zm)
                tid = torch.full((zm.shape[0],), si, dtype=torch.long, device=device)
                for m in METHODS:
                    if m.startswith("V10"):
                        g = bank.direction(zm, si, m[4:], fns[m[4:]])
                    elif m == "r1":
                        g = r1_direction(critic, zm, tid)
                    elif m == "ld3":
                        g = ld3(zm, tid)
                    else:
                        g = v6g(zm, tid)
                    c = (g.flatten(1) * delta.flatten(1)).sum(1)  # both unit-norm
                    cos[m].extend(c.tolist())
        res["per_style"][s] = {m: round(sum(v) / len(v), 4) for m, v in cos.items()}
        for m in METHODS:
            total_cos[m].extend(cos[m])
        print(f"[expA] {s} " + " ".join(f"{m}={res['per_style'][s][m]:.3f}"
                                        for m in METHODS) + f" {time.time()-t0:.0f}s",
              flush=True)

    res["mean"] = {m: round(sum(v) / len(v), 4) for m, v in total_cos.items()}
    # gate: best V10 variant vs r1 / ld3, per-style wins
    best_v10 = max((m for m in METHODS if m.startswith("V10")), key=lambda m: res["mean"][m])
    wins = sum(1 for s in STYLES
               if res["per_style"][s][best_v10] > res["per_style"][s]["r1"]
               and res["per_style"][s][best_v10] > res["per_style"][s]["ld3"])
    res["gate"] = {"best_variant": best_v10,
                   "mean": res["mean"][best_v10],
                   "r1": res["mean"]["r1"], "ld3": res["mean"]["ld3"],
                   "style_wins_vs_both": f"{wins}/5",
                   "pass": bool(res["mean"][best_v10] > res["mean"]["r1"]
                                and res["mean"][best_v10] > res["mean"]["ld3"]
                                and wins >= 4)}
    print("[expA] mean " + " ".join(f"{m}={res['mean'][m]:.3f}" for m in METHODS))
    print("[expA] GATE", json.dumps(res["gate"]))
    (OUT / "p2_results.json").parent.mkdir(exist_ok=True)
    p = OUT / "p2_results.json"
    cur = json.loads(p.read_text()) if p.exists() else {}
    cur["exp_a"] = res
    p.write_text(json.dumps(cur, indent=1))
    print("saved ->", p)


if __name__ == "__main__":
    main()
