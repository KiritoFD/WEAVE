"""P1 diagnostics: paired comparisons, and the teacher fit error that decides the bottleneck.

The decision table in the plan hinges on whether the student can FIT the teacher at all:

  fits + generalises        -> supervision/target was the bottleneck
  does not fit              -> capacity, or dynamic feedback is necessary
  fits but does not improve -> generalisation or a teacher shortcut

So this measures, on TRAINING-pool latents the student never optimised for directly:

    rel_fit = E||G_theta(c,t) - z_T||^2 / E||z_T - c||^2

where z_T is the frozen teacher (Q2a Direct + one R1 step at RMS 0.25). rel_fit near 0 means the
student reproduced the teacher's edit; near or above 1 means it did not move at all relative to what
the teacher asked for.

Usage
    python3 -m tools.v6_next.fit_error
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from tools.v6.canonicalize import ContentCanonicalizer
from tools.v6.models_v6 import C3UNet
from tools.v6.train_eval_v6 import STYLES, V5_ROOT, V6_ROOT
from tools.v6_next.probe_style import LatentStyleNet

ASSET = Path("/mnt/i/Latent_Style/pseudo_gt_24h/orig_mean_fp16")
FILES = ["00_Early_Renaissance.pt", "01_Impressionism.pt", "02_Minimalism.pt",
         "03_Rococo.pt", "04_Ukiyo_e.pt"]


def load_gen(path, dev):
    ck = torch.load(ROOT / path, map_location="cpu", weights_only=False)
    a = ck.get("args", {})
    g = C3UNet(base_dim=a.get("base_dim", 16), num_styles=5, mode=a.get("mode", "direct"),
               no_style=bool(a.get("no_style", False)),
               content_mode=a.get("content_mode", "full"),
               inject_64=not a.get("no_inject_64", False),
               inject_32=not a.get("no_inject_32", False),
               inject_16=not a.get("no_inject_16", False),
               style_keys=bool(a.get("style_keys", False))).to(dev)
    g.load_state_dict(ck["model"])
    g.eval()
    return g, a.get("canon_mode", "c1_gentle")


def rank_step(critic, z, tgt, rms, steps):
    per = rms / max(1, steps)
    out = z
    for _ in range(steps):
        with torch.enable_grad():
            zr = out.detach().clone().requires_grad_(True)
            q = critic(zr.float()).float()
            oh = F.one_hot(tgt, len(STYLES)).bool()
            m = (q.gather(1, tgt[:, None]).squeeze(1)
                 - torch.logsumexp(q.masked_fill(oh, torch.finfo(q.dtype).min), dim=1))
            g = torch.autograd.grad(m.sum(), zr)[0]
        D = g[0].numel()
        gn = g.flatten(1).norm(dim=1).view(-1, 1, 1, 1)
        out = (out + per * (g / (gn / D ** 0.5 + 1e-12))).detach()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="runs/v6/Q2a_embed_style/Q2a_embed_style.pt")
    ap.add_argument("--students", default="P1_B0_ctrl218,P1_B1_distill218,"
                                          "P1_B2c_ctrl845,P1_B2_distill845")
    ap.add_argument("--critic", default="runs/v6_next/probes/r1_r1.pt")
    ap.add_argument("--per-style", type=int, default=200)
    ap.add_argument("--rms", type=float, default=0.25)
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--out", default="runs/v6_next/amortize/p1_fit_error.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = args.device

    teacher, cmode = load_gen(args.teacher, dev)
    canon = ContentCanonicalizer(torch.load(V6_ROOT / "cache" / "canonicalizer_stats.pt",
                                           map_location="cpu", weights_only=False)).to(dev)
    ckC = torch.load(ROOT / args.critic, map_location="cpu", weights_only=False)
    critic = LatentStyleNet(width=ckC.get("width", 64)).to(dev).eval()
    critic.load_state_dict(ckC["model"])
    for p in critic.parameters():
        p.requires_grad_(False)

    # TRAINING-pool latents, tail slice, so this is not the evaluation set
    zs, ts = [], []
    for sid, f in enumerate(FILES):
        d = torch.load(ASSET / f, map_location="cpu", weights_only=False)
        lat = d["latents"].float()[-args.per_style:]
        for s2 in range(5):
            if s2 == sid:
                continue          # cross-style only
            zs.append(lat)
            ts.append(torch.full((len(lat),), s2, dtype=torch.long))
    z_src = torch.cat(zs, 0).to(dev)
    tgt = torch.cat(ts, 0).to(dev)
    print(f"[fit] {len(z_src)} training-pool (source, target) pairs, cross-style only", flush=True)

    # Chunked: 4000 pairs at once OOMs the 3060 when the critic runs with grad enabled.
    CH = 256
    c_all, zT_all, zd_all = [], [], []
    for s in range(0, len(z_src), CH):
        zb, tb = z_src[s:s + CH], tgt[s:s + CH]
        with torch.inference_mode():
            cb = canon(zb, mode=cmode, augment=False)
            zd = teacher(cb, 0.0, cb, tb)
        zt = rank_step(critic, zd, tb, args.rms, args.steps)
        c_all.append(cb.float().cpu())
        zT_all.append(zt.float().cpu())
        zd_all.append(zd.float().cpu())
        del zd, zt
        torch.cuda.empty_cache()
        if (s // CH) % 4 == 0:
            print(f"    teacher {min(s+CH, len(z_src))}/{len(z_src)}", flush=True)
    c_f = torch.cat(c_all, 0)
    z_T = torch.cat(zT_all, 0)
    z_direct = torch.cat(zd_all, 0)
    denom = float(((z_T - c_f) ** 2).sum(1).mean())
    print(f"[fit] teacher displacement ||z_T - c||^2 = {denom:.4f}", flush=True)

    res = {"teacher": args.teacher, "rms": args.rms, "steps": args.steps,
           "n_pairs": int(len(z_src)), "teacher_disp_sq": denom, "students": {}}
    for name in args.students.split(","):
        name = name.strip()
        if not name:
            continue
        p = ROOT / f"runs/v6/{name}/{name}.pt"
        if not p.exists():
            print(f"[fit] {name}: missing"); continue
        stu, cm2 = load_gen(f"runs/v6/{name}/{name}.pt", dev)
        errs, moves = [], []
        for s in range(0, len(z_src), CH):
            zb, tb = z_src[s:s + CH], tgt[s:s + CH]
            with torch.inference_mode():
                cc = canon(zb, mode=cm2, augment=False)
                z_s = stu(cc, 0.0, cc, tb).float().cpu()
            errs.append(float(((z_s - z_T[s:s + CH]) ** 2).sum(1).mean()))
            moves.append(float(((z_s - z_direct[s:s + CH]) ** 2).sum(1).mean()))
            del z_s
            torch.cuda.empty_cache()
        err = float(np.mean(errs))
        move = float(np.mean(moves))
        res["students"][name] = {"abs_fit_sq": err, "rel_fit": err / max(denom, 1e-12),
                                "move_sq": move,
                                "rel_move": move / max(denom, 1e-12)}
        print(f"  {name:20s} rel_fit={err/max(denom,1e-12):.4f}  "
              f"rel_move={move/max(denom,1e-12):.4f}  "
              f"||s-T||^2={err:.3f}", flush=True)
        del stu
        torch.cuda.empty_cache()

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n  interpretation:")
    for n, v in res["students"].items():
        rf = v["rel_fit"]
        verdict = ("FITS the teacher" if rf < 0.20 else
                   "partially fits" if rf < 0.50 else
                   "DOES NOT fit the teacher")
        print(f"    {n:20s} rel_fit={rf:.3f} -> {verdict}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
