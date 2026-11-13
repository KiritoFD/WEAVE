"""Experiment D: DSM network vs closed-form KDE on the recovery protocol.

Same holdout paintings / 7 degradations / cosine metric as exp_a_direction.
Pre-registered decision: keep the closed-form KDE unless DSM is strictly better
(mean A_grad) on this identical evaluation.

Run: python -m exp.v10.exp_d_dsm
Appends runs/v10_local_4070/p2_results.json (key "exp_d").
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .destyl import LADDER, degrade
from .dsm import dsm_direction, load_dsm
from .featfn import holdout_latents, load_bank, make_feat_fn, pool_channel_stats
from .splits import STYLES

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "runs/v10_local_4070"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    device = a.device

    bank, hpsi = load_bank(device)
    stats = {k: v.to(device) for k, v in bank.stats.items()}
    fn_b = make_feat_fn(stats, hpsi, "B")
    model = load_dsm(device)

    res = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "n_per_style": a.n,
           "ops": LADDER, "sigma_eval": "0.1", "per_style": {}}
    cos_dsm, cos_kde = [], []
    t0 = time.time()
    for si, s in enumerate(STYLES):
        lat, _, _ = holdout_latents(si, a.n, seed=0)
        pc = pool_channel_stats(si)
        cs_d, cs_k = [], []
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
                    delta = (delta / delta.norm(dim=1, keepdim=True).clamp_min(1e-12)).view_as(zr)
                for j in range(zm.shape[0]):
                    z1 = zm[j:j + 1]
                    gd = dsm_direction(model, stats, hpsi, z1, si)
                    gk = bank.direction(z1, si, "B", fn_b, samples_per_call=1)
                    cs_d.append((gd.flatten(1) * delta[j:j + 1].flatten(1)).sum().item())
                    cs_k.append((gk.flatten(1) * delta[j:j + 1].flatten(1)).sum().item())
        res["per_style"][s] = {"dsm": round(sum(cs_d) / len(cs_d), 4),
                               "kde": round(sum(cs_k) / len(cs_k), 4)}
        cos_dsm.extend(cs_d); cos_kde.extend(cs_k)
        print(f"[expD] {s} dsm={res['per_style'][s]['dsm']} "
              f"kde={res['per_style'][s]['kde']} {time.time()-t0:.0f}s", flush=True)
    res["mean"] = {"dsm": round(sum(cos_dsm) / len(cos_dsm), 4),
                   "kde": round(sum(cos_kde) / len(cos_kde), 4)}
    res["decision"] = "keep_dsm" if res["mean"]["dsm"] > res["mean"]["kde"] else "keep_kde"
    print("[expD] mean", res["mean"], "->", res["decision"])
    p = OUT / "p2_results.json"
    cur = json.loads(p.read_text()) if p.exists() else {}
    cur["exp_d"] = res
    p.write_text(json.dumps(cur, indent=1))
    print("saved ->", p)


if __name__ == "__main__":
    main()
