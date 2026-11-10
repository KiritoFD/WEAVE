"""P1 gate: does the R1 critic's gradient actually improve the REAL CSD margin?

This is the decisive test before P2. A critic that classifies real images well is necessary but not
sufficient: the plan is explicit that a critic whose gradient only attacks its own score must not be
given a generator loss. So this measures, end to end:

  1. g = grad_z margin_target at the Direct output z_out, with the critic FROZEN
     -> gradient norm, and the angle against two reference directions:
          V6 direction        z_out - z_src          (what the operator already does)
          target direction    mu_t - mu_s in CSD     (what we want)
  2. take a small, content-limited step z' = z_out + eps * g_hat, decode it
  3. measure the REAL CSD margin with the UNTRAINED-FOR critic-free evaluator
     -> did m_target rise? and what happened to DINO-C / LPIPS?

A positive result needs m_target to rise on the real CSD metric, not just on the critic's logits.
The critic's own margin is reported alongside so the two can be compared directly.

Usage
    python3 -m tools.v6_next.gate --critic runs/v6_next/probes/r1_r1.pt \
        --ckpt runs/v6/L11_k1024_dim16/L11_k1024_dim16.pt
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
from tools.v6_next.probe_style import LatentStyleNet

OUT = ROOT / "runs/v6_next"


def csd_margin(feats: np.ndarray, proto: np.ndarray, tgt: np.ndarray):
    """m_target = <f, mu_t> - max_{k != t} <f, mu_k> for each row."""
    s = feats @ proto.T
    m = np.empty(len(feats))
    for i, t in enumerate(tgt):
        m[i] = s[i, t] - np.delete(s[i], t).max()
    return m, s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--critic", default="runs/v6_next/probes/r1_r1.pt")
    ap.add_argument("--ckpt", default="runs/v6/L11_k1024_dim16/L11_k1024_dim16.pt")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--eps", default="0.02,0.05,0.10")
    ap.add_argument("--gen", choices=["direct", "scorediff"], default="direct",
                    help="scorediff measures the historical editor on the SAME tasks and metric "
                         "path, which is what the Pareto comparison requires")
    ap.add_argument("--sd-ckpt", default="runs/v6/score_diff/score_diff_net.pt")
    ap.add_argument("--sd-steps", type=int, default=24)
    ap.add_argument("--sd-lr", type=float, default=0.100)
    ap.add_argument("--sd-anchor", type=float, default=0.08)
    ap.add_argument("--tag", default="r1")
    ap.add_argument("--out", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = args.device
    eps_list = [float(x) for x in args.eps.split(",")]

    ck = torch.load(ROOT / args.ckpt, map_location="cpu", weights_only=False)
    a = ck.get("args", {})
    gen = C3UNet(base_dim=a.get("base_dim", 48), num_styles=5, mode=a.get("mode", "direct"),
                 no_style=bool(a.get("no_style", False)),
                 content_mode=a.get("content_mode", "full"),
                 inject_64=not a.get("no_inject_64", False),
                 inject_32=not a.get("no_inject_32", False),
                 inject_16=not a.get("no_inject_16", False),
                 style_keys=bool(a.get("style_keys", False))).to(dev)
    gen.load_state_dict(ck["model"]); gen.eval()
    cmode = a.get("canon_mode", "c1_gentle")
    canon = ContentCanonicalizer(torch.load(V6_ROOT / "cache" / "canonicalizer_stats.pt",
                                           map_location="cpu", weights_only=False)).to(dev)

    ckC = torch.load(ROOT / args.critic, map_location="cpu", weights_only=False)
    critic = LatentStyleNet(width=ckC.get("width", 64)).to(dev).eval()
    critic.load_state_dict(ckC["model"])
    for p in critic.parameters():
        p.requires_grad_(False)
    print(f"[gate] critic {args.critic} (width {ckC.get('width', 64)}) frozen", flush=True)

    tasks = json.loads((V5_ROOT / "v5_eval200_tasks.json").read_text(encoding="utf-8"))[: args.n]
    lat = torch.load(V5_ROOT / "cache" / "eval_src_latents.pt", map_location="cpu")
    stems = sorted({t["src_stem"] for t in tasks})
    idx = {s: i for i, s in enumerate(stems)}
    z_src = torch.stack([lat[t["src_stem"]] for t in tasks]).to(dev).float()
    sid = torch.tensor([t["src_style_id"] for t in tasks], device=dev)
    tid = torch.tensor([t["tgt_style_id"] for t in tasks], device=dev)

    proto = csd_prototypes()
    csd, _, _ = csd_model.load_csd(dev)
    dino = load_dino_model(device=dev)
    vae = load_vae(dev)
    trt = load_trt(vae, dev)
    import lpips
    lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()

    def embed_z(z):
        with torch.inference_mode():
            img = decode_trt(trt, vae, z, batch=8)
            u8 = img.mul(255.0).round().clamp(0, 255).to(torch.uint8).cpu()
            arr = u8.permute(0, 2, 3, 1).numpy()
            pils = [Image.fromarray(arr[i]) for i in range(len(arr))]
            F_ = csd_model.feats_pils(csd, pils, batch=16, device=dev)
            D_, _ = encode_dino(dino, pils, batch=16, device=dev)
            return F_, D_.float().cpu().numpy(), u8.float().div(255.0), arr

    with torch.inference_mode():
        if args.gen == "scorediff":
            from tools.v6.models_v6 import ConditionalScoreNet
            from tools.v6.score_diff import edit_with_score_difference
            sd_ck = torch.load(ROOT / args.sd_ckpt, map_location="cpu", weights_only=False)
            sd = ConditionalScoreNet(base_dim=64, num_styles=5).to(dev).eval()
            sd.load_state_dict(sd_ck["model"])
            z_out = edit_with_score_difference(
                sd, z_src, sid, tid, num_steps=args.sd_steps, step_size=args.sd_lr,
                struct_anchor_weight=args.sd_anchor, seed=42)
            print(f"[gate] generator=scorediff N={args.sd_steps} lr={args.sd_lr} "
                  f"anchor={args.sd_anchor}", flush=True)
        else:
            c = canon(z_src, mode=cmode, augment=False)
            z_out = gen(c, 0.0, c, tid)

    # reference: source-side CSD features for DINO-C / LPIPS
    _, dino_src, rgb_src, _ = embed_z(z_src)
    F_src, _, _, _ = embed_z(z_src)
    m_src, _ = csd_margin(F_src, proto, tid.cpu().numpy())

    F_out, dino_out, rgb_out, _ = embed_z(z_out)
    m_out, s_out = csd_margin(F_out, proto, tid.cpu().numpy())
    with torch.inference_mode():
        lg0 = critic(z_out.float())
        m_critic0 = lg0.gather(1, tid[:, None]).squeeze(1) - torch.logsumexp(
            lg0.masked_fill(F.one_hot(tid, 5).bool(), -1e9), dim=1)
    print(f"\n[gate] baseline Direct output: real CSD m_target = {m_out.mean():+.4f}")
    print(f"       critic margin = {float(m_critic0.mean()):+.4f}")
    dino_c0 = float(np.mean([float(dino_out[i] @ dino_src[i]) for i in range(len(tasks))]))
    print(f"       real CSD Top-1 = {float((m_out > 0).mean()):.3f}   DINO-C = {dino_c0:.4f}")

    if args.gen == "scorediff":
        # No gradient probe for ScoreDiff: it is a fixed multi-step editor, not a step along a
        # critic gradient. The point of this mode is to put it on the SAME tasks and metric path as
        # the R1 curve so the Pareto comparison is like-for-like.
        res = {"gen": "scorediff", "ckpt": args.sd_ckpt, "n": len(tasks),
               "sd_steps": args.sd_steps, "sd_lr": args.sd_lr, "sd_anchor": args.sd_anchor,
               "real_m_target": float(m_out.mean()),
               "real_m_source": float(m_src.mean()),
               "top1": float((m_out > 0).mean()), "dino_c": dino_c0,
               "per_target_top1": {
                   STYLES[t]: float((m_out[np.array([i for i, x in enumerate(tid.tolist())
                                                    if x == t])] > 0).mean())
                   for t in range(5)}}
        p = ROOT / (args.out or f"runs/v6_next/probes/gate_scorediff_n{len(tasks)}.json")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        print("\n  per-target Top-1: " + "  ".join(
            f"{k[:9]}={v:.3f}" for k, v in res["per_target_top1"].items()))
        print(f"\nwrote {p}")
        return

    # ---- 1. gradient of the critic margin w.r.t. z_out ----
    z_req = z_out.detach().clone().requires_grad_(True)
    logits = critic(z_req.float())
    mask = F.one_hot(tid, 5).bool()
    m_critic = logits.gather(1, tid[:, None]).squeeze(1) - \
        torch.logsumexp(logits.masked_fill(mask, -1e9), dim=1)
    m_critic.mean().backward()
    g = z_req.grad.detach()

    v6_dir = (z_out - z_src)
    # CSD target direction, expressed as the difference of prototype-derived gradients is not
    # available in latent space; we report the angle against the V6 direction instead, plus the
    # cosine between the critic gradient and the V6 step, which is the actionable comparison.
    def cos_flat(x, y):
        return float((x.flatten() @ y.flatten()) /
                     (x.flatten().norm() * y.flatten().norm() + 1e-12))
    print(f"\n[gate] gradient probe (n={len(tasks)})")
    print(f"  |g| = {float(g.flatten().norm()):.4e}   "
          f"mean|g| per dim = {float(g.abs().mean()):.4e}")
    print(f"  cos(g, V6 direction) = {cos_flat(g, v6_dir):+.4f}")
    print(f"  |V6 direction| = {float(v6_dir.flatten().norm()):.4f}")

    # ---- 2. content-limited steps along g_hat, then measure the REAL CSD margin ----
    rows = []
    # Per-sample normalisation. The previous version divided by the WHOLE-BATCH norm, so with B
    # samples of similar magnitude the per-image RMS was about eps/sqrt(B) and the knob changed
    # meaning with batch size. Now eps means "per-image RMS step" and is batch-invariant.
    D = g[0].numel()
    gn = g.flatten(1).norm(dim=1).view(-1, 1, 1, 1)
    g_hat = g / (gn / D ** 0.5 + 1e-12)
    print(f"  per-sample |g| (RMS per dim) mean={float(gn.mean()) / D**0.5:.4e} "
          f"min={float(gn.min()) / D**0.5:.4e} max={float(gn.max()) / D**0.5:.4e}")

    for eps in eps_list:
        z_new = z_out + eps * g_hat
        F_new, dino_new, rgb_new, _ = embed_z(z_new)
        m_new, _ = csd_margin(F_new, proto, tid.cpu().numpy())
        # LPIPS needs decoded RGB, not latents: compare the decoded images on device
        with torch.inference_mode():
            a_new = rgb_new.to(dev)
            a_out = rgb_out.to(dev)
            a_src = rgb_src.to(dev)
            lp = lpips_fn(to_lpips(a_new), to_lpips(a_out)).flatten().cpu().numpy()
            lp_src = lpips_fn(to_lpips(a_new), to_lpips(a_src)).flatten().cpu().numpy()
        dc = np.array([float(dino_new[i] @ dino_src[i]) for i in range(len(tasks))])
        rows.append({
            "eps": eps,
            "real_m_target": float(m_new.mean()),
            "d_m_target": float((m_new - m_out).mean()),
            "d_m_target_vs_src": float((m_new - m_src).mean()),
            "top1": float((m_new > 0).mean()),
            "top1_direct": float((m_out > 0).mean()),
            "dino_c": float(dc.mean()),
            "lpips_vs_direct": float(lp.mean()),
            "lpips_vs_src": float(lp_src.mean()),
        })
        print(f"  eps={eps:<5} real m_target {m_new.mean():+.4f} "
              f"(d {rows[-1]['d_m_target']:+.4f})  Top-1 {rows[-1]['top1']:.3f}  "
              f"DINO-C {dc.mean():.4f}  LPIPS(vs Direct) {lp.mean():.4f}")

    # reference points
    print(f"\n  reference: Direct DINO-C = "
          f"{np.mean([float(dino_out[i] @ dino_src[i]) for i in range(len(tasks))]):.4f}")

    best = max(rows, key=lambda r: r["d_m_target"])
    verdict = ("critic gradient IMPROVES the real CSD margin"
               if best["d_m_target"] > 0.01 else
               "critic gradient does NOT improve the real CSD margin")
    print(f"\n  VERDICT: {verdict} (best d_m_target {best['d_m_target']:+.4f} at eps={best['eps']})")

    res = {"critic": args.critic, "ckpt": args.ckpt, "n": len(tasks),
           "baseline_real_m_target": float(m_out.mean()),
           "baseline_real_m_source": float(m_src.mean()),
           "grad_norm": float(g.flatten().norm()),
           "cos_grad_v6": cos_flat(g, v6_dir),
           "v6_dir_norm": float(v6_dir.flatten().norm()),
           "steps": rows, "verdict": verdict, "best": best}
    p = ROOT / (args.out or f"runs/v6_next/probes/gate_{args.tag}.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {p}")


def to_lpips(x):
    from utils.run_evaluation import to_lpips_input
    return to_lpips_input(x)


if __name__ == "__main__":
    main()
