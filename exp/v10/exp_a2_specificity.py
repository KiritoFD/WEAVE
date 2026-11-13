"""V10.2 Experiment A: style-score target specificity (6+1 controls).

Same protocol as v10 exp A (holdout 40/style x 7 degradations, recovery cosine).
Run: python -m exp.v10.exp_a2_specificity
Appends runs/v10_local_4070/p2_results.json (key "exp_a2").
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .destyl import LADDER, degrade
from .featfn import holdout_latents, load_bank, make_feat_fn, pool_channel_stats
from .retrieval import patchify_scale
from .scorevar import VariantScores
from .features import patchify, f_feats, q_feats
from .featv2 import v_feats
from .splits import STYLES

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "runs/v10_local_4070"
VARIANTS = ["A1", "A2", "A3", "A4", "A5", "A6", "A7"]


def s16_feats(bank16_stats, z, style_idx):
    st = bank16_stats
    p = patchify_scale(z, 16)
    k = (q_feats(p) - st["q_mean"]) / st["q_std"]
    v = (v_feats(p) - st["v_mean"]) / st["v_std"]
    return {"k": k, "v": v, "style": STYLES[style_idx]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    device = a.device

    kde, hpsi = load_bank(device)
    stats = {k: v.to(device) for k, v in kde.stats.items()}
    fn_b = make_feat_fn(stats, hpsi, "B")
    vs = VariantScores(OUT, device)

    res = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "n_per_style": a.n,
           "ops": LADDER, "per_style": {}}
    total = {v: [] for v in VARIANTS}
    t0 = time.time()
    torch.manual_seed(0)
    for si, s in enumerate(STYLES):
        lat, _, _ = holdout_latents(si, a.n, seed=0)
        pc = pool_channel_stats(si)
        cos = {v: [] for v in VARIANTS}
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
                for vname in VARIANTS:
                    gs = []
                    for j in range(zm.shape[0]):
                        zi = zm[j:j + 1].detach().clone().requires_grad_(True)
                        with torch.enable_grad():
                            p = patchify(zi)
                            feats = {"q": (q_feats(p) - stats["q_mean"]) / stats["q_std"],
                                     "f": (f_feats(p) - stats["f_mean"]) / stats["f_std"]}
                            if vname == "A5":
                                shift = vs.mean_shift_q_only(feats, si)
                                L = (shift * feats["f"]).sum()
                            elif vname == "A6":
                                L = 0.5 * vs.logp_sum(feats, si, "A1") \
                                    + 0.5 * vs.logp_s16_sum(s16_feats(vs.st16, zi, si))
                            elif vname == "A7":
                                L = vs.logp_s16_sum(s16_feats(vs.st16, zi, si))
                            else:
                                L = vs.logp_sum(feats, si, vname)
                            g = torch.autograd.grad(L, zi)[0]
                        flat = g.flatten(1)
                        gnorm = flat / flat.norm(dim=1, keepdim=True).clamp_min(1e-12)
                        gs.append(gnorm)
                    g = torch.cat(gs)
                    c = (g.flatten(1) * delta.flatten(1)).sum(1)
                    cos[vname].extend(c.tolist())
        res["per_style"][s] = {v: round(sum(x) / len(x), 4) for v, x in cos.items()}
        for v in VARIANTS:
            total[v].extend(cos[v])
        print(f"[A2exp] {s} " + " ".join(f"{v}={res['per_style'][s][v]:.3f}"
              for v in VARIANTS) + f" {time.time()-t0:.0f}s", flush=True)

    res["mean"] = {v: round(sum(x) / len(x), 4) for v, x in total.items()}
    m = res["mean"]
    res["decomposition"] = {
        "style_identity_component(A1-A2)": round(m["A1"] - m["A2"], 4),
        "generic_denoising_component(A3)": m["A3"],
        "content_conditioning(A1-A4)": round(m["A1"] - m["A4"], 4),
        "state_selection(A1-A5)": round(m["A1"] - m["A5"], 4),
        "scale_gain(A6-A1)": round(m["A6"] - m["A1"], 4),
        "relations_gain(A7-A1)": round(m["A7"] - m["A1"], 4),
        "specificity_ratio(A1-A3)/A1": round((m["A1"] - m["A3"]) / max(m["A1"], 1e-6), 3),
    }
    res["verdict"] = {
        "generic_dominant": bool(m["A3"] >= 0.9 * m["A1"]),
        "style_specific": bool(m["A2"] < 0.5 * m["A1"]),
    }
    print("[A2exp] mean", res["mean"])
    print("[A2exp] decomposition", json.dumps(res["decomposition"]))
    print("[A2exp] verdict", json.dumps(res["verdict"]))
    p = OUT / "p2_results.json"
    cur = json.loads(p.read_text()) if p.exists() else {}
    cur["exp_a2"] = res
    p.write_text(json.dumps(cur, indent=1))
    print("saved ->", p)


if __name__ == "__main__":
    main()
