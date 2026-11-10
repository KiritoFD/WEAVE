"""P0: does per-image R1 optimisation actually produce effective edits that Direct cannot learn?

Five inference variants on the SAME 750-task protocol (600 cross + 150 same), all scored through one
path so nothing but the latent differs:

  A0  Q2a Direct output                                  baseline
  A1  A0 + one R1 gradient step, total RMS 0.25           is there an effective edit direction?
  A2  A0 + 4 R1 gradient steps, 0.0625 each, recomputed   does iterating help?
  A3  A0 + 4 steps of 0.0625 with the gradient frozen at z0
  A4  ScoreDiff N=24 from the source latent               strong-style reference

IMPORTANT NOTE ON A1 vs A3. With a fixed TOTAL RMS budget, A3's endpoint is
z0 + 4 * 0.0625 * g_hat(z0) = z0 + 0.25 * g_hat(z0), which is exactly A1's endpoint. So A1 and A3 are
the same edit by construction; A3 is an implementation check (it must match A1 bit-for-bit), not an
independent arm. The informative comparison is therefore A2 versus A1: dynamic recomputation against a
single direction at equal total displacement. This is reported explicitly rather than presenting
A1/A3 as a result.

The R1 guidance never touches CSD: it uses only the frozen critic's margin, and no per-image
hyper-parameter is selected from any evaluation metric.

Usage
    python3 -m tools.v6_next.amortize_p0 --ckpt runs/v6/Q2a_embed_style/Q2a_embed_style.pt
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
from PIL import Image

from tools.v6.canonicalize import ContentCanonicalizer
from tools.v6.models_v6 import C3UNet
from tools.v6.train_eval_v6 import (STYLES, V5_ROOT, V6_ROOT, csd_model, csd_prototypes,
                                    decode_trt, encode_dino, load_dino_model, load_trt,
                                    load_vae, _load_eval_image_tensor)
from tools.v6_24h.stats import cluster_ci
from tools.v6_next.probe_style import LatentStyleNet

OUT = ROOT / "runs/v6_next/amortize"


def load_critic(dev):
    ck = torch.load(ROOT / "runs/v6_next/probes/r1_r1.pt", map_location="cpu",
                    weights_only=False)
    m = LatentStyleNet(width=ck.get("width", 64)).to(dev).eval()
    m.load_state_dict(ck["model"])
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def rank_direction(critic, z: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """Unit-RMS direction that increases the target margin. No CSD involved."""
    zr = z.detach().clone().requires_grad_(True)
    q = critic(zr.float()).float()
    oh = F.one_hot(tgt, len(STYLES)).bool()
    m = (q.gather(1, tgt[:, None]).squeeze(1)
         - torch.logsumexp(q.masked_fill(oh, torch.finfo(q.dtype).min), dim=1))
    m.sum().backward()
    g = zr.grad.detach()
    D = g[0].numel()
    gn = g.flatten(1).norm(dim=1).view(-1, 1, 1, 1)
    return g / (gn / D ** 0.5 + 1e-12)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/v6/Q2a_embed_style/Q2a_embed_style.pt")
    ap.add_argument("--tasks", default="eval_protocol/tasks_750.json")
    ap.add_argument("--rms", type=float, default=0.25, help="total edit RMS budget")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--sd-steps", type=int, default=24)
    ap.add_argument("--sd-lr", type=float, default=0.100)
    ap.add_argument("--sd-anchor", type=float, default=0.08)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = args.device
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- model ----
    ck = torch.load(ROOT / args.ckpt, map_location="cpu", weights_only=False)
    a = ck.get("args", {})
    gen = C3UNet(base_dim=a.get("base_dim", 16), num_styles=5, mode=a.get("mode", "direct"),
                 no_style=bool(a.get("no_style", False)),
                 content_mode=a.get("content_mode", "full"),
                 inject_64=not a.get("no_inject_64", False),
                 inject_32=not a.get("no_inject_32", False),
                 inject_16=not a.get("no_inject_16", False),
                 style_keys=bool(a.get("style_keys", False))).to(dev)
    gen.load_state_dict(ck["model"])
    gen.eval()
    cmode = a.get("canon_mode", "c1_gentle")
    canon = ContentCanonicalizer(torch.load(V6_ROOT / "cache" / "canonicalizer_stats.pt",
                                           map_location="cpu", weights_only=False)).to(dev)
    critic = load_critic(dev)

    from tools.v6.models_v6 import ConditionalScoreNet
    from tools.v6.score_diff import edit_with_score_difference
    sd_ck = torch.load(ROOT / "runs/v6/score_diff/score_diff_net.pt", map_location="cpu",
                       weights_only=False)
    sd = ConditionalScoreNet(base_dim=64, num_styles=5).to(dev).eval()
    sd.load_state_dict(sd_ck["model"])

    proto = np.asarray(csd_prototypes(), dtype=np.float32)
    csd, _, _ = csd_model.load_csd(dev)
    dino = load_dino_model(device=dev)
    vae = load_vae(dev)
    trt = load_trt(vae, dev)
    import lpips
    lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()

    # ---- tasks ----
    spec = json.loads((ROOT / args.tasks).read_text(encoding="utf-8"))
    tasks = spec["tasks"]
    if args.limit:
        tasks = tasks[: args.limit]
    print(f"[A] {len(tasks)} tasks from {args.tasks} "
          f"(cross={sum(1 for t in tasks if t.get('is_cross'))})", flush=True)

    src_lat = torch.load(V5_ROOT / "cache" / "eval_src_latents.pt", map_location="cpu")
    # eval_src_latents.pt only covers the 50 Val200 sources. The 750-task protocol uses 150, so
    # encode whatever is missing from the image files and cache it -- otherwise 100 sources are
    # silently skipped and the "600 cross" protocol quietly becomes 200.
    need = {t["source_stem"] for t in tasks} - set(src_lat)
    if need:
        print(f"[A] encoding {len(need)} source latents missing from the cache", flush=True)
        cache_path = V6_ROOT / "cache" / "tasks750_src_latents.pt"
        extra = torch.load(cache_path, map_location="cpu") if cache_path.exists() else {}
        todo = [s for s in need if s not in extra]
        if todo:
            by_stem = {}
            for t in tasks:
                if t["source_stem"] in todo:
                    by_stem[t["source_stem"]] = t["source_rel_path"]
            sf = 0.18215
            with torch.inference_mode():
                for i in range(0, len(todo), 8):
                    chunk = todo[i:i + 8]
                    x = torch.stack([_load_eval_image_tensor(ROOT / by_stem[s], size=512)
                                     for s in chunk], 0).to(dev)
                    z = (vae.encode(x.to(vae.dtype)).latent_dist.mean * sf).float().cpu()
                    for j, s in enumerate(chunk):
                        extra[s] = z[j]
                    if (i // 8) % 5 == 0:
                        print(f"    {min(i+8, len(todo))}/{len(todo)}", flush=True)
            torch.save(extra, cache_path)
        for s in need:
            if s in extra:
                src_lat[s] = extra[s]
        print(f"[A] latent cache now covers "
              f"{len({t['source_stem'] for t in tasks} & set(src_lat))} of "
              f"{len({t['source_stem'] for t in tasks})} sources", flush=True)
    VAR = ["A0", "A1", "A2", "A3", "A4"]
    acc = {v: [] for v in VAR}

    for s in range(0, len(tasks), args.batch):
        grp = tasks[s: s + args.batch]
        z_src, tg, sg = [], [], []
        keep = []
        for t in grp:
            k = t["source_stem"]
            if k not in src_lat:
                continue
            keep.append(t)
            z_src.append(src_lat[k])
            tg.append(t["target_style_id"])
            sg.append(t["source_style_id"])
        if not keep:
            continue
        z_src = torch.stack(z_src).to(dev).float()
        tg = torch.tensor(tg, device=dev, dtype=torch.long)
        sg = torch.tensor(sg, device=dev, dtype=torch.long)

        with torch.inference_mode():
            c = canon(z_src, mode=cmode, augment=False)
            z0 = gen(c, 0.0, c, tg)

        # A0
        lat = {"A0": z0}
        # A1: one step, full budget
        g0 = rank_direction(critic, z0, tg)
        lat["A1"] = z0 + args.rms * g0
        # A3: same total budget in `steps` increments along the frozen direction
        per = args.rms / args.steps
        lat["A3"] = z0 + per * g0 * args.steps
        # A2: `steps` increments of the same size, gradient recomputed each time
        z = z0
        for _ in range(args.steps):
            z = z + per * rank_direction(critic, z, tg)
        lat["A2"] = z
        # A4: ScoreDiff from the raw source latent
        with torch.inference_mode():
            lat["A4"] = edit_with_score_difference(
                sd, z_src, sg, tg, num_steps=args.sd_steps, step_size=args.sd_lr,
                struct_anchor_weight=args.sd_anchor, seed=42)

        # decode + score every variant through the identical path
        decoded = {}
        for v in VAR:
            with torch.inference_mode():
                img = decode_trt(trt, vae, lat[v], batch=args.batch)
                u8 = img.mul(255.0).round().clamp(0, 255).to(torch.uint8).cpu()
                decoded[v] = (u8, u8.float().div(255.0))

        pils = {v: [Image.fromarray(decoded[v][0][i].permute(1, 2, 0).numpy())
                    for i in range(len(keep))] for v in VAR}
        # source reference (round-trip) for dS / DINO-C / LPIPS
        with torch.inference_mode():
            img_s = decode_trt(trt, vae, z_src, batch=args.batch)
            u8s = img_s.mul(255.0).round().clamp(0, 255).to(torch.uint8).cpu()
        pil_s = [Image.fromarray(u8s[i].permute(1, 2, 0).numpy()) for i in range(len(keep))]

        Fs = csd_model.feats_pils(csd, pil_s, batch=16, device=dev)
        Fs = Fs / np.linalg.norm(Fs, axis=1, keepdims=True)
        Ds, _ = encode_dino(dino, pil_s, batch=16, device=dev)
        Ds = Ds.float().cpu().numpy()
        rgb_s = u8s.float().div(255.0).to(dev)

        for v in VAR:
            Fv = csd_model.feats_pils(csd, pils[v], batch=16, device=dev)
            Fv = Fv / np.linalg.norm(Fv, axis=1, keepdims=True)
            Dv, _ = encode_dino(dino, pils[v], batch=16, device=dev)
            Dv = Dv.float().cpu().numpy()
            with torch.inference_mode():
                lp = lpips_fn(decoded[v][1].to(dev), rgb_s).flatten().cpu().numpy()
            for i, t in enumerate(keep):
                ti, si = t["target_style_id"], t["source_style_id"]
                sc = Fv[i] @ proto.T
                acc[v].append({
                    "task_id": t["task_id"], "src_stem": t["source_stem"],
                    "src_style": t["source_style"], "tgt_style": t["target_style"],
                    "is_cross": bool(t.get("is_cross")),
                    "dT": float(sc[ti] - Fs[i] @ proto[ti]),
                    "dS": float(sc[si] - Fs[i] @ proto[si]),
                    "m_target": float(sc[ti] - np.delete(sc, ti).max()),
                    "max_cos": float(sc.max()),
                    "argmax": STYLES[int(sc.argmax())],
                    "top1": int(sc.argmax() == ti),
                    "dino_c": float(Dv[i] @ Ds[i]),
                    "lpips": float(lp[i]),
                })
        print(f"[A] {min(s+args.batch, len(tasks))}/{len(tasks)}", flush=True)

    # ---- aggregate ----
    def summarise(rows):
        cross = [r for r in rows if r["is_cross"]]
        same = [r for r in rows if not r["is_cross"]]
        out = {"n_cross": len(cross), "n_same": len(same)}
        if cross:
            out["top1"] = float(np.mean([r["top1"] for r in cross]))
            out["dT"] = float(np.mean([r["dT"] for r in cross]))
            out["m_target"] = float(np.mean([r["m_target"] for r in cross]))
            out["dino_c"] = float(np.mean([r["dino_c"] for r in cross]))
            out["lpips"] = float(np.mean([r["lpips"] for r in cross]))
            cl = [r["src_stem"] for r in cross]
            out["dT_ci"] = cluster_ci([r["dT"] for r in cross], cl)
            out["top1_ci"] = cluster_ci([float(r["top1"]) for r in cross], cl)
            out["per_target"] = {}
            for st in STYLES:
                sub = [r for r in cross if r["tgt_style"] == st]
                if sub:
                    out["per_target"][st] = {
                        "n": len(sub),
                        "top1": float(np.mean([r["top1"] for r in sub])),
                        "dT": float(np.mean([r["dT"] for r in sub])),
                        "m_target": float(np.mean([r["m_target"] for r in sub])),
                    }
            conf = np.zeros((5, 5), dtype=int)
            for r in cross:
                conf[STYLES.index(r["tgt_style"])][STYLES.index(r["argmax"])] += 1
            out["confusion"] = conf.tolist()
            # non-target Minimalism rate
            nm = [r for r in cross if r["tgt_style"] != "Minimalism"]
            out["min_offdiag"] = (sum(1 for r in nm if r["argmax"] == "Minimalism"),
                                  len(nm))
        if same:
            out["same_top1"] = float(np.mean([r["top1"] for r in same]))
            out["same_dS"] = float(np.mean([r["dS"] for r in same]))
            out["same_dino_c"] = float(np.mean([r["dino_c"] for r in same]))
        return out

    res = {"ckpt": args.ckpt, "tasks": args.tasks, "rms": args.rms, "steps": args.steps,
           "variants": {v: summarise(acc[v]) for v in VAR}}

    # A1 vs A3 must be identical by construction; verify instead of assuming
    if acc["A1"] and acc["A3"]:
        d = max(abs(x["dT"] - y["dT"]) for x, y in zip(acc["A1"], acc["A3"]))
        res["A1_A3_max_abs_dT_diff"] = float(d)

    print("\n" + "=" * 96)
    print("P0  per-image R1 optimisation vs Direct (750-task protocol)")
    print("=" * 96)
    print(f"  {'variant':8s}{'Top-1':>8s}{'dT':>9s}{'m_target':>10s}"
          f"{'DINO-C':>8s}{'LPIPS':>8s}{'same Top1':>10s}{'same dS':>9s}")
    for v in VAR:
        r = res["variants"][v]
        print(f"  {v:8s}{r.get('top1',0):8.3f}{r.get('dT',0):+9.4f}"
              f"{r.get('m_target',0):+10.4f}{r.get('dino_c',0):8.4f}"
              f"{r.get('lpips',0):8.4f}{r.get('same_top1',0):10.3f}"
              f"{r.get('same_dS',0):+9.4f}")
    print(f"\n  A1 vs A3 max |d dT| = {res.get('A1_A3_max_abs_dT_diff', float('nan')):.3e}"
          "  (expected 0: same total RMS along the same direction)")

    print("\n  per-target Top-1")
    print(f"  {'variant':8s}" + "".join(f"{s[:9]:>11s}" for s in STYLES))
    for v in VAR:
        pt = res["variants"][v].get("per_target") or {}
        print(f"  {v:8s}" + "".join(f"{pt.get(s,{}).get('top1',0):11.3f}" for s in STYLES))

    print("\n  non-target Minimalism rate")
    for v in VAR:
        k, n = res["variants"][v].get("min_offdiag", (0, 0))
        print(f"  {v:8s} {k:3d}/{n:3d} = {k/max(1,n):.3f}")

    p = ROOT / (args.out or "runs/v6_next/amortize/p0_variants.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({**res, "rows": acc}, ensure_ascii=False, indent=2),
                 encoding="utf-8")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
