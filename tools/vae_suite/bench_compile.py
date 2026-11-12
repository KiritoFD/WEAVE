"""A/B bench: eager vs torch.compile, with and without gradient checkpointing.

Replicates the Style-Rank training graph (rec + KL + rank + CE) so the timing and
peak VRAM are comparable to the real run.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, "/root/Workspace/WEAVE_VAE")

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL

from tools.vae_suite.losses import compute_mixed_reconstruction_loss, compute_kl_divergence
import lpips

A0_DIR = "/root/Workspace/WEAVE_VAE/runs_vae/A0/final_diffusers_vae"
OUT = Path("/root/Workspace/WEAVE_VAE/runs/vae_style_rank20")
DEV = "cuda"


def build():
    vae = AutoencoderKL.from_pretrained(A0_DIR).to(DEV)
    vae.train()
    lp = lpips.LPIPS(net="alex", verbose=False).to(DEV).eval()
    for p in lp.parameters():
        p.requires_grad = False
    return vae, lp


def step(vae, lp, x, P, mu_z, proto, lam_r, lam_c, accum=1):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        l_rec, _, _, post, _ = compute_mixed_reconstruction_loss(vae, x, lp)
        kl_e, kl_v, _, _ = compute_kl_divergence(post)
    mu = post.mean.float().flatten(1)
    u = F.normalize(F.linear(mu - mu_z, P), dim=-1, eps=1e-6)
    B = u.shape[0]
    S = u @ u.t()
    ar = torch.arange(B, device=u.device)
    partner = ar ^ 1
    valid = torch.ones(B, B, dtype=torch.bool, device=u.device)
    valid[ar, ar] = False; valid[ar, partner] = False
    c_ap = S[ar, partner]
    gaps = (S - c_ap.unsqueeze(1)).abs().masked_fill(~valid, float("inf"))
    neg = gaps.argmin(1)
    l_rank = F.softplus((S[ar, neg] - c_ap + 0.05) / 0.10).mean()
    y = torch.randint(0, proto.shape[0], (B,), device=u.device)
    l_ce = F.cross_entropy(u @ proto.t() / 0.15, y)
    loss = (l_rec + 1.126e-4 * kl_e + lam_r * l_rank + lam_c * l_ce) / accum
    loss.backward()
    return float(l_rec.detach()), float(kl_v)


def bench(tag, batch, ckpt, compile_mode, steps=8, lam_r=0.0046, lam_c=0.0025):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        vae, lp = build()
        if ckpt:
            vae.enable_gradient_checkpointing()
        P = torch.randn(768, 16384, device=DEV) * 0.01
        mu_z = torch.zeros(1, 16384, device=DEV)
        proto = F.normalize(torch.randn(20, 768, device=DEV), dim=-1)

        if compile_mode:
            vae.encoder = torch.compile(vae.encoder, mode=compile_mode, dynamic=False)
            vae.decoder = torch.compile(vae.decoder, mode=compile_mode, dynamic=False)

        opt = torch.optim.AdamW(vae.parameters(), lr=1e-6)
        x = torch.randn(batch, 3, 512, 512, device=DEV)
        # warmup (triggers compile)
        t0 = time.perf_counter()
        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            step(vae, lp, x, P, mu_z, proto, lam_r, lam_c)
            opt.step()
        warm = time.perf_counter() - t0
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            step(vae, lp, x, P, mu_z, proto, lam_r, lam_c)
            opt.step()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t1) / steps
        peak = torch.cuda.max_memory_allocated(0) / 1024**2
        print(f"[{tag}] batch={batch} ckpt={ckpt} compile={compile_mode} -> "
              f"{dt:.2f} s/step | peak={peak:.0f} MB | warmup(incl compile)={warm:.1f}s",
              flush=True)
        del vae, lp, opt
        torch.cuda.empty_cache()
        return {"tag": tag, "batch": batch, "ckpt": ckpt, "compile": compile_mode,
                "sec_per_step": round(dt, 3), "peak_mb": round(peak), "warmup_sec": round(warm, 1)}
    except torch.cuda.OutOfMemoryError as e:
        print(f"[{tag}] batch={batch} ckpt={ckpt} compile={compile_mode} -> OOM", flush=True)
        torch.cuda.empty_cache()
        return {"tag": tag, "batch": batch, "ckpt": ckpt, "compile": compile_mode, "oom": True}
    except Exception as e:
        print(f"[{tag}] batch={batch} ckpt={ckpt} compile={compile_mode} -> ERROR {type(e).__name__}: {e}",
              flush=True)
        torch.cuda.empty_cache()
        return {"tag": tag, "batch": batch, "ckpt": ckpt, "compile": compile_mode,
                "error": f"{type(e).__name__}: {e}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", default="all")
    args = ap.parse_args()

    res = []
    if args.what in ("all", "eager"):
        res.append(bench("eager", 6, False, None))
        res.append(bench("eager", 4, False, None))
        res.append(bench("eager", 2, False, None))
    if args.what in ("all", "compile"):
        res.append(bench("compile", 6, False, "default"))
        res.append(bench("compile", 4, False, "default"))
        res.append(bench("compile", 2, False, "default"))
    if args.what == "ref":
        res.append(bench("ref_eager_ckpt", 6, True, None))

    import json
    p = OUT / "compile_bench.json"
    old = json.loads(p.read_text()) if p.exists() else []
    old += res
    p.write_text(json.dumps(old, indent=2), encoding="utf-8")
    print(f"\n[Saved] {p}")


if __name__ == "__main__":
    main()
