"""P0: gradient-conflict diagnosis (NO training, no optimizer step).

On fixed real training microbatches, compute the ACTUAL WEIGHTED gradients on
the shared Encoder parameters:

    g_base   = grad(L_rec + beta*L_KL)
    g_rank   = lambda_r * grad(L_rank)
    g_ce     = lambda_c * grad(L_ce)
    g_anchor = lambda_a * grad(L_anchor)

Report (per microbatch, then median over batches):
    cos(g_rank, g_anchor)                       <-- primary question
    ||g_anchor|| / ||g_rank||
    ||g_rank + g_ce|| / ||g_base + g_anchor||
    cos(rank, base), cos(rank, ce), cos(base, anchor), cos(anchor, ce)
    effective variance / rank of P-projected features & normalization saturation

Also evaluates three anchor settings (current, x0.1, zero) since P1 will use them.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, "/root/Workspace/WEAVE_VAE")

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL

from tools.vae_suite.losses import compute_mixed_reconstruction_loss, compute_kl_divergence
from tools.vae_suite.style_rank_data import (RamStyleDataset, EpochCoverageSampler,
                                            NegQueue, pick_negatives)
import lpips

OUT = Path("/root/Workspace/WEAVE_VAE/runs/vae_style_rank20")
MANIFEST = Path("/root/Workspace/data/wikiart_512_raw/wikiart_512_raw_manifest.json")
A0_DIR = "/root/Workspace/WEAVE_VAE/runs_vae/A0/final_diffusers_vae"
BETA_KL = 1.126e-4
MARGIN = 0.05
TAU_R = 0.10
TAU_C = 0.15


def rank_loss(s_ap, s_an, valid=None):
    if s_an.dim() == 1:
        s_an = s_an.unsqueeze(1)
    if valid is None:
        valid = torch.ones_like(s_an, dtype=torch.bool)
    per = F.softplus((s_an - s_ap.unsqueeze(1) + MARGIN) / TAU_R)
    v = valid.float()
    n = v.sum()
    if n == 0:
        return s_ap.sum() * 0.0
    return (per * v).sum() / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(OUT / "checkpoints" / "ckpt_u01500.pt"))
    ap.add_argument("--n-batches", type=int, default=12)
    ap.add_argument("--warm-batches", type=int, default=32)
    ap.add_argument("--out", default=str(OUT / "gradient_conflict.json"))
    args = ap.parse_args()

    device = "cuda"
    torch.manual_seed(0); np.random.seed(0)

    proj = torch.load(OUT / "projection.pt", weights_only=False)
    P = proj["P"].to(device).float()
    mu_z = proj["mu_z"].to(device).float()
    proto = torch.load(OUT / "prototypes.pt", weights_only=False).to(device).float()
    calib = json.loads((OUT / "calibration.json").read_text())
    lam_r, lam_c, lam_a = calib["lambda_rank"], calib["lambda_ce"], calib["lambda_anchor"]
    print(f"[Weights] lambda_rank={lam_r} lambda_ce={lam_c} lambda_anchor={lam_a}", flush=True)

    idx = torch.load(OUT / "index.pt", weights_only=False)
    pos_list = idx["pos_list"]; yn = idx["y"].numpy(); an = idx["a"].numpy()
    DINO = F.normalize(torch.load(OUT / "train_dino.pt", weights_only=False)
                       .to(device).float(), dim=-1)

    style_to_items = collections.defaultdict(list)
    for i in range(len(yn)):
        style_to_items[int(yn[i])].append((i, int(an[i])))
    ds = RamStyleDataset(MANIFEST, pos_list, preload=False, style_ids=yn)
    sampler = EpochCoverageSampler(style_to_items, seed=42, n_styles=1)
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=2, pin_memory=True)
    it = iter(loader)

    # model (eager, no compile -> clean autograd.grad)
    vae = AutoencoderKL.from_pretrained(A0_DIR, torch_dtype=torch.float32).to(device)
    ck = None
    if args.model and Path(args.model).exists():
        ck = torch.load(args.model, map_location="cpu", weights_only=False)
        vae.load_state_dict(ck["model"])
        m_c = ck.get("m_center", F.linear(mu_z, P).squeeze(0)).to(device).float()
        print(f"[Model] loaded {Path(args.model).name} (upd={ck.get('upd')}) "
              f"m_center_norm={m_c.norm().item():.2f}", flush=True)
    else:
        m_c = F.linear(mu_z, P).squeeze(0).to(device).float()
        print("[Model] A0 baseline, default center", flush=True)
    vae.train()

    vae_a0 = AutoencoderKL.from_pretrained(A0_DIR, torch_dtype=torch.float32)
    del vae_a0.decoder
    del vae_a0.post_quant_conv
    vae_a0.to(device).eval()
    for p in vae_a0.parameters():
        p.requires_grad = False

    lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    for p in lpips_fn.parameters():
        p.requires_grad = False

    enc_params = [p for p in vae.encoder.parameters()] + [p for p in vae.quant_conv.parameters()]

    negq = NegQueue(4096, 768, device)

    def next_batch():
        nonlocal it
        try:
            return next(it)
        except StopIteration:
            it = iter(loader)
            return next(it)

    # warm the queue with REAL microbatches (same as training)
    with torch.no_grad():
        for _ in range(args.warm_batches):
            b = next_batch()
            x = b["image"].to(device, non_blocking=True)
            pos = b["pos"].to(device, non_blocking=True)
            y = b["style_id"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                post = vae.encode(x).latent_dist
            mu = post.mean.float().flatten(1)
            u = F.normalize(F.linear(mu, P) - m_c.unsqueeze(0), dim=-1, eps=1e-6)
            negq.push(u, DINO[pos], y, pos)
    print(f"[Queue] warmed with {negq.n} items", flush=True)

    def single_grad(b, which, cache=None):
        """One forward per loss, graph released immediately -> low VRAM."""
        x = b["image"].to(device, non_blocking=True)
        pos = b["pos"].to(device, non_blocking=True)
        y = b["style_id"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            l_rec, _, _, post, _ = compute_mixed_reconstruction_loss(vae, x, lpips_fn)
            kl_e, _, _, _ = compute_kl_divergence(post)
        mu = post.mean.float().flatten(1)
        h = F.linear(mu, P)
        hc = h - m_c.unsqueeze(0)
        u = F.normalize(hc, dim=-1, eps=1e-6)

        if which == "base":
            loss = l_rec + BETA_KL * kl_e
        elif which == "rank":
            s_ap, s_an, neg_valid, _ = negq.pick(u, DINO[pos], y, K=4)
            loss = lam_r * rank_loss(s_ap, s_an, neg_valid)
        elif which == "ce":
            loss = lam_c * F.cross_entropy(u @ proto.t() / TAU_C, y)
        else:  # anchor
            if cache is None or "mu_a0" not in cache:
                with torch.no_grad():
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        ha = vae_a0.quant_conv(vae_a0.encoder(x))
                    cache["mu_a0"] = ha[:, :4].float().flatten(1)
            loss = lam_a * F.mse_loss(mu, cache["mu_a0"])

        gs = torch.autograd.grad(loss, enc_params, retain_graph=False, allow_unused=True)
        g = torch.cat([gg.reshape(-1) if gg is not None
                       else torch.zeros_like(p).reshape(-1)
                       for gg, p in zip(gs, enc_params)])
        diag = {
            "hc_norm_mean": float(hc.norm(dim=-1).mean()),
            "hc_norm_std": float(hc.norm(dim=-1).std()),
            "h_var_mean": float(h.var(dim=0).mean()),
            "u_pair_cos": float((u @ u.t()).mean()),
            "loss_val": float(loss.detach()),
        }
        return g, diag, u, pos, y

    records = []
    for bi in range(args.n_batches):
        b = next_batch()
        cache = {}
        g_base, d0, u, pos, y = single_grad(b, "base", cache)
        torch.cuda.empty_cache()
        g_rank, _, _, _, _ = single_grad(b, "rank", cache)
        torch.cuda.empty_cache()
        g_ce, _, _, _, _ = single_grad(b, "ce", cache)
        torch.cuda.empty_cache()
        g_anchor, d3, _, _, _ = single_grad(b, "anchor", cache)
        diag = {**d0, "anchor_loss_val": d3["loss_val"]}

        def cos(a, b_):
            return float(F.cosine_similarity(a, b_, dim=0))

        def nrm(a):
            return float(a.norm())

        # unweighted magnitudes too (to see what the weights are doing)
        with torch.no_grad():
            r = {
                "batch": bi,
                "norm_base": nrm(g_base), "norm_rank_w": nrm(g_rank),
                "norm_ce_w": nrm(g_ce), "norm_anchor_w": nrm(g_anchor),
                "cos_rank_anchor": cos(g_rank, g_anchor),
                "cos_rank_base": cos(g_rank, g_base),
                "cos_rank_ce": cos(g_rank, g_ce),
                "cos_base_anchor": cos(g_base, g_anchor),
                "cos_anchor_ce": cos(g_anchor, g_ce),
                "ratio_anchor_rank": nrm(g_anchor) / max(nrm(g_rank), 1e-12),
                "ratio_style_vs_base_anchor": nrm(g_rank + g_ce) / max(nrm(g_base + g_anchor), 1e-12),
                "ratio_style_vs_base_only": nrm(g_rank + g_ce) / max(nrm(g_base), 1e-12),
                # hypothetical: anchor scaled by 0.1 or removed
                "ratio_style_vs_base_anchor_x0.1": nrm(g_rank + g_ce) / max(nrm(g_base + 0.1 * g_anchor), 1e-12),
                "cos_rank_anchor_x0.1": cos(g_rank, 0.1 * g_anchor),
                "ratio_style_vs_base_noanchor": nrm(g_rank + g_ce) / max(nrm(g_base), 1e-12),
                "total_grad_current": nrm(g_base + g_rank + g_ce + g_anchor),
                "total_grad_noanchor": nrm(g_base + g_rank + g_ce),
                **diag,
            }
        records.append(r)
        if bi % 3 == 0:
            print(f"  b{bi}: cos(rank,anchor)={r['cos_rank_anchor']:+.3f} "
                  f"| |ga|/|gr|={r['ratio_anchor_rank']:.2f} "
                  f"| style/base+anchor={r['ratio_style_vs_base_anchor']:.3f} "
                  f"| hc_norm={r['hc_norm_mean']:.1f}", flush=True)

        # keep queue advancing (diagnostic only, no optimizer step)
        with torch.no_grad():
            negq.push(u.detach(), DINO[pos], y, pos)
        del g_base, g_rank, g_ce, g_anchor, u
        torch.cuda.empty_cache()

    keys = [k for k in records[0] if k != "batch"]
    med = {k: round(float(np.median([r[k] for r in records])), 5) for k in keys}
    mean = {k: round(float(np.mean([r[k] for r in records])), 5) for k in keys}
    res = {
        "model": args.model,
        "n_batches": args.n_batches,
        "weights": {"lambda_rank": lam_r, "lambda_ce": lam_c, "lambda_anchor": lam_a},
        "median": med, "mean": mean, "per_batch": records,
    }
    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print("P0 GRADIENT CONFLICT DIAGNOSIS (median over %d microbatches)" % args.n_batches)
    print("=" * 78)
    print(f"  cos(g_rank, g_anchor)          = {med['cos_rank_anchor']:+.4f}   <-- primary")
    print(f"  cos(g_rank, g_base)            = {med['cos_rank_base']:+.4f}")
    print(f"  cos(g_base, g_anchor)          = {med['cos_base_anchor']:+.4f}")
    print(f"  ||g_anchor|| / ||g_rank||      = {med['ratio_anchor_rank']:.3f}")
    print(f"  ||g_rank+g_ce|| / ||g_base+g_anchor|| = {med['ratio_style_vs_base_anchor']:.4f}")
    print(f"  same, anchor x0.1              = {med['ratio_style_vs_base_anchor_x0.1']:.4f}")
    print(f"  same, no anchor                = {med['ratio_style_vs_base_noanchor']:.4f}")
    print(f"  |total grad| current/noanchor = {med['total_grad_current']:.4f} / "
          f"{med['total_grad_noanchor']:.4f}")
    print(f"  norms: base={med['norm_base']:.4f} rank={med['norm_rank_w']:.4f} "
          f"ce={med['norm_ce_w']:.4f} anchor={med['norm_anchor_w']:.4f}")
    print(f"  P-feature: ||hc||={med['hc_norm_mean']:.1f} "
          f"h_var={med['h_var_mean']:.3f} u_pair_cos={med['u_pair_cos']:+.4f}")
    print("=" * 78)
    print(f"[Saved] {args.out}")


if __name__ == "__main__":
    main()
