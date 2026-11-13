"""Experiment B: color-shortcut audit (plan §5-B).

Four bank variants on the same recovery protocol as experiment A:
  B_color (14-dim color block) / B_texture (20-dim texture block) /
  B (joint 34) / C (joint + h_psi residual).

Additionally decomposes each KDE update  (v_bar_w - v)  into the color/texture
feature blocks (energy fractions), to see WHERE the distribution pull acts.

Verdict (pre-registered): texture-only <= color-only means "colour accounting",
unless joint > color-only AND texture-only is clearly positive.

Run: python -m exp.v10.exp_b_shortcut
Appends runs/v10_local_4070/p2_results.json (key "exp_b").
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .destyl import LADDER, degrade
from .featfn import holdout_latents, load_bank, make_feat_fn, pool_channel_stats
from .features import F_COLOR, F_TEXTURE
from .kde import QUERY_CHUNK, SIGMA, TOPK, _d2
from .splits import STYLES

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "runs/v10_local_4070"
VARIANTS_B = ["B_color", "B_texture", "B", "C"]
N_PER_STYLE = 40


@torch.no_grad()
def update_decomposition(bank, stats, hpsi, z, si, n_patch_sample=2048):
    """Mean energy fraction of the KDE pull in the colour vs texture blocks."""
    s = STYLES[si]
    q_mean, q_std = stats["q_mean"], stats["q_std"]
    f_mean, f_std = stats["f_mean"], stats["f_std"]
    from .features import patchify, q_feats, f_feats
    p = patchify(z)
    q = ((q_feats(p) - q_mean) / q_std).float()
    f = ((f_feats(p) - f_mean) / f_std).float()
    n = f.shape[0]
    sel = torch.randperm(n)[:min(n, n_patch_sample)]
    q, f = q[sel], f[sel]
    Vb = bank.V["f"][s].float()
    Qb = bank.Q[s].float()
    ec, et = [], []
    for i in range(0, f.shape[0], QUERY_CHUNK):
        fc = f[i:i + QUERY_CHUNK]
        dv = _d2(fc, Vb)
        dq = _d2(q[i:i + QUERY_CHUNK], Qb)
        logits = -dv / (2 * SIGMA ** 2) - dq / (2 * 2.0 ** 2)
        k = min(TOPK, Vb.shape[0])
        w = torch.softmax(logits.topk(k, dim=1).values, dim=1)
        idx = logits.topk(k, dim=1).indices
        vbar = (w.unsqueeze(-1) * Vb[idx]).sum(1)
        shift = vbar - fc                       # (c,34) in joint f space
        ec.append(shift[:, F_COLOR[0]:F_COLOR[1]].square().sum(1))
        et.append(shift[:, F_TEXTURE[0]:F_TEXTURE[1]].square().sum(1))
    ec = torch.cat(ec); et = torch.cat(et)
    tot = (ec + et).clamp_min(1e-12)
    return {"color_frac": round((ec / tot).mean().item(), 4),
            "texture_frac": round((et / tot).mean().item(), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N_PER_STYLE)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    device = a.device

    bank, hpsi = load_bank(device)
    stats = {k: v.to(device) for k, v in bank.stats.items()}
    fns = {v: make_feat_fn(stats, hpsi, v) for v in VARIANTS_B}

    res = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "n_per_style": a.n,
           "ops": LADDER, "per_style": {}, "decomp": {}}
    total = {v: [] for v in VARIANTS_B}
    t0 = time.time()
    for si, s in enumerate(STYLES):
        lat, _, _ = holdout_latents(si, a.n, seed=0)
        pc = pool_channel_stats(si)
        cos = {v: [] for v in VARIANTS_B}
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
                    delta = delta / delta.norm(dim=1, keepdim=True).clamp_min(1e-12)
                    delta = delta.view_as(zr)
                for v in VARIANTS_B:
                    g = bank.direction(zm, si, v, fns[v])
                    c = (g.flatten(1) * delta.flatten(1)).sum(1)
                    cos[v].extend(c.tolist())
        res["per_style"][s] = {v: round(sum(x) / len(x), 4) for v, x in cos.items()}
        for v in VARIANTS_B:
            total[v].extend(cos[v])
        res["decomp"][s] = update_decomposition(bank, stats, hpsi,
                                                lat[:32].float().to(device), si)
        print(f"[expB] {s} " + " ".join(f"{v}={res['per_style'][s][v]:.3f}"
              for v in VARIANTS_B) + f" decomp={res['decomp'][s]} {time.time()-t0:.0f}s",
              flush=True)

    res["mean"] = {v: round(sum(x) / len(x), 4) for v, x in total.items()}
    m = res["mean"]
    res["verdict"] = {
        "texture_only_vs_color_only": "texture<=color" if m["B_texture"] <= m["B_color"] else "texture>color",
        "joint_beats_color_only": bool(m["B"] > m["B_color"]),
        "texture_positive": bool(m["B_texture"] > 0.05),
        "note": "pre-registered: colour-accounting verdict only if texture<=color AND not(joint>color and texture positive)",
    }
    texture_account = (m["B_texture"] <= m["B_color"]
                       and not (m["B"] > m["B_color"] and m["B_texture"] > 0.05))
    res["verdict"]["colour_accounting"] = bool(texture_account)
    print("[expB] mean", res["mean"])
    print("[expB] VERDICT", json.dumps(res["verdict"]))
    p = OUT / "p2_results.json"
    cur = json.loads(p.read_text()) if p.exists() else {}
    cur["exp_b"] = res
    p.write_text(json.dumps(cur, indent=1))
    print("saved ->", p)


if __name__ == "__main__":
    main()
