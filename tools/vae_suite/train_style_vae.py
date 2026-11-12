"""WEAVE Style-VAE: 12h two-stage joint training (fixed orthogonal style subspace P).

S1 (0-4h)  : L_rec + beta*L_KL + 0.03*L_CSD + 0.05*L_SupCon        (E+D unfrozen)
S2 (4-11h) : S1 + 0.05*L_swap + 0.10*L_cycle                       (swap trains Decoder)

Fixed: native SD-VAE, P (768x16384, orthonormal, frozen), A0 reconstruction & KL.
No Flow, no SWD, no extra style encoder, no GAN.
"""
from __future__ import annotations

import argparse
import collections
import copy
import json
import math
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, "/root/Workspace/WEAVE_VAE")
sys.path.insert(0, "/root/Workspace/WEAVE_VAE/tools/12h")

os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/root/Workspace/.inductor_cache"
os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader, Sampler

from tools.vae_suite.dataset import RawMemmapWikiArtDataset
from tools.vae_suite.losses import compute_mixed_reconstruction_loss, compute_kl_divergence
import csd_model
import lpips

OUT = Path("/root/Workspace/WEAVE_VAE/runs_vae/style_vae")
OUT.mkdir(parents=True, exist_ok=True)
MANIFEST = Path("/root/Workspace/data/wikiart_512_raw/wikiart_512_raw_manifest.json")
A0_DIR = "/root/Workspace/WEAVE_VAE/runs_vae/A0/final_diffusers_vae"
SUBSPACE = OUT / "style_subspace.pt"

NORM_MEAN = [0.48145466, 0.4578275, 0.40821073]
NORM_STD = [0.26862954, 0.26130258, 0.27577711]
BETA_KL = 1.126e-4
W_CSD = 0.03
W_SUP = 0.05
W_SWAP = 0.05
W_CYCLE = 0.10
TAU = 0.10


class StyleArtistBatchSampler(Sampler):
    """3 styles x 2 different artists per microbatch."""

    def __init__(self, dataset, n_styles=3, n_per_style=2, num_batches=10**9, seed=42):
        self.n_styles = n_styles
        self.n_per_style = n_per_style
        self.num_batches = num_batches
        self.seed = seed
        self.batch_size = n_styles * n_per_style
        self.sa = collections.defaultdict(lambda: collections.defaultdict(list))
        for pos, raw in enumerate(dataset.indices):
            rec = dataset.records_by_idx[raw]
            self.sa[rec["style_id"]][rec.get("artist_id", -1)].append(pos)
        self.usable = sorted([s for s, am in self.sa.items()
                              if len([a for a, v in am.items() if a >= 0 and v]) >= 2])
        print(f"[Sampler] usable styles={len(self.usable)}", flush=True)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        rng = random.Random(self.seed)
        i = 0
        while i < self.num_batches:
            sts = rng.sample(self.usable, self.n_styles)
            b = []
            for s in sts:
                am = self.sa[s]
                arts = [a for a, v in am.items() if a >= 0 and v]
                rng.shuffle(arts)
                for a in arts[:self.n_per_style]:
                    b.append(rng.choice(am[a]))
            if len(b) == self.batch_size:
                i += 1
                yield b


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        for p in self.shadow.parameters():
            p.requires_grad = False

    def update(self, model):
        with torch.no_grad():
            for s, m in zip(self.shadow.parameters(), model.parameters()):
                s.data.mul_(self.decay).add_(m.data, alpha=1 - self.decay)

    def state_dict(self):
        return self.shadow.state_dict()


def supcon(s, labels, tau=TAU):
    sim = s @ s.t() / tau
    B = s.shape[0]
    eye = torch.eye(B, dtype=torch.bool, device=s.device)
    pos = (labels[:, None] == labels[None, :]) & ~eye
    logits = sim - sim.max(1, keepdim=True).values.detach()
    exp = torch.exp(logits) * (~eye)
    logp = logits - torch.log(exp.sum(1, keepdim=True).clamp(min=1e-12))
    npos = pos.sum(1)
    valid = npos > 0
    if int(valid.sum()) == 0:
        return torch.tensor(0.0, device=s.device)
    per = -(logp * pos).sum(1) / npos.clamp(min=1)
    return per[valid].mean()


def gauss_blur(x, sigma=0.6, ksize=5):
    coords = torch.arange(ksize, device=x.device, dtype=x.dtype) - (ksize - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kx = g.view(1, 1, 1, ksize).expand(3, 1, 1, ksize)
    ky = g.view(1, 1, ksize, 1).expand(3, 1, ksize, 1)
    x = F.conv2d(x, kx, padding=(0, ksize // 2), groups=3)
    x = F.conv2d(x, ky, padding=(ksize // 2, 0), groups=3)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=0, help="0 = run by wall clock")
    ap.add_argument("--s1-hours", type=float, default=4.0)
    ap.add_argument("--total-hours", type=float, default=11.0)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr-enc", type=float, default=3e-5)
    ap.add_argument("--lr-dec", type=float, default=2e-5)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--speed-test", type=int, default=0, help="run N updates then exit")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stage", default="both", choices=["both", "s1", "s2"])
    args = ap.parse_args()

    device = "cuda"
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    sub = torch.load(SUBSPACE, weights_only=False)
    P = sub["P"].to(device).float()                 # (768, 16384)
    mu_z = sub["mu_z"].to(device).float()           # (1, 16384)
    t_bar = sub["t_bar"].to(device).float()         # (1, 768)
    proto = sub["prototypes"].to(device).float()    # (136, 768)
    n_styles = sub["n_styles"]
    print(f"[Subspace] P={tuple(P.shape)} orth_err="
          f"{(P @ P.t() - torch.eye(P.shape[0], device=device)).abs().max().item():.2e} "
          f"train_cos={sub['train_cos_mean']:.4f}", flush=True)

    ds = RawMemmapWikiArtDataset(MANIFEST, is_training=True)
    sampler = StyleArtistBatchSampler(ds, 3, 2, num_batches=10**9, seed=args.seed)
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=4, pin_memory=True)

    vae = AutoencoderKL.from_pretrained(A0_DIR).to(device)
    vae.train()
    vae.enable_gradient_checkpointing()

    csd, _, _ = csd_model.load_csd(device)
    csd = csd.float().eval()
    for p in csd.parameters():
        p.requires_grad = False
    nm = torch.tensor(NORM_MEAN, device=device).view(1, 3, 1, 1)
    ns = torch.tensor(NORM_STD, device=device).view(1, 3, 1, 1)

    lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    for p in lpips_fn.parameters():
        p.requires_grad = False

    @torch.no_grad()
    def teacher(x):
        x01 = (x.clamp(-1, 1) + 1.0) / 2.0
        x224 = F.interpolate(x01, (224, 224), mode="bicubic", align_corners=False)
        _, _, sty = csd(((x224 - nm) / ns).half())
        return F.normalize(sty.float() - t_bar, dim=-1)

    def csd_grad(x):
        x01 = (x.clamp(-1, 1) + 1.0) / 2.0
        x224 = F.interpolate(x01, (224, 224), mode="bicubic", align_corners=False)
        _, _, sty = csd((x224 - nm) / ns)
        return F.normalize(sty.float() - t_bar, dim=-1)

    opt = torch.optim.AdamW([
        {"params": list(vae.encoder.parameters()), "lr": args.lr_enc},
        {"params": list(vae.decoder.parameters()), "lr": args.lr_dec},
        {"params": list(vae.quant_conv.parameters()) + list(vae.post_quant_conv.parameters()),
         "lr": args.lr_enc},
    ], betas=(0.9, 0.99), weight_decay=args.wd)

    def lr_lambda(s):
        if s < args.warmup:
            return float(s) / max(1, args.warmup)
        return 0.2 + 0.8 * 0.5 * (1 + math.cos(math.pi * min(1.0, (s - args.warmup) / 20000)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    ema = EMA(vae, 0.999)

    it = iter(loader)
    t_start = time.time()
    s1_sec = args.s1_hours * 3600
    total_sec = args.total_hours * 3600
    upd = 0
    hist = []
    stage_log = []

    def next_batch():
        nonlocal it
        try:
            return next(it)
        except StopIteration:
            it = iter(loader)
            return next(it)

    def stage_of(elapsed):
        if args.stage == "s1":
            return 1
        if args.stage == "s2":
            return 2
        return 1 if elapsed < s1_sec else 2

    while True:
        elapsed = time.time() - t_start
        stg = stage_of(elapsed)
        if args.steps > 0 and upd >= args.steps:
            break
        if args.steps == 0 and elapsed >= total_sec:
            break
        if args.speed_test and upd >= args.speed_test:
            break

        if not stage_log or stage_log[-1]["stage"] != stg:
            stage_log.append({"stage": stg, "start_sec": round(elapsed, 1), "start_upd": upd})
            print(f"\n=== [STAGE {stg}] start at {elapsed/3600:.2f}h, update {upd} ===", flush=True)

        t_upd = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        acc = collections.defaultdict(float)

        for _ in range(args.accum):
            b = next_batch()
            x = b["image"].to(device, non_blocking=True)
            st_ids = b["style_id"].to(device, non_blocking=True)

            # ---------- part 1: reconstruction + CSD + SupCon ----------
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                l_rec, _, _, post, _ = compute_mixed_reconstruction_loss(vae, x, lpips_fn)
                kl_e, kl_v, _, _ = compute_kl_divergence(post)
                mu = post.mean.float().flatten(1)                    # (B,16384)
                u = (mu - mu_z) @ P.t()                              # (B,768)
                s = F.normalize(u, dim=-1)
                t = teacher(x)
                l_csd = (1.0 - (s * t).sum(1)).mean()
                l_sup = supcon(s, st_ids)
                loss1 = (l_rec + BETA_KL * kl_e + W_CSD * l_csd + W_SUP * l_sup) / args.accum
            loss1.backward()

            acc["rec"] += float(l_rec.detach()) / args.accum
            acc["kl"] += float(kl_v) / args.accum
            acc["csd"] += float(l_csd.detach()) / args.accum
            acc["sup"] += float(l_sup.detach()) / args.accum

            # ---------- part 2: swap (S2 only, microbatch=1) ----------
            if stg == 2:
                # 3 styles x 2 artists: index 0 and index 3 are different styles
                ia, ib = 0, 3
                xa, xb = x[ia:ia + 1], x[ib:ib + 1]
                tgt_style = int(st_ids[ib].item())
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    za = vae.encode(xa).latent_dist.mean.float().flatten(1)
                    zb = vae.encode(xb).latent_dist.mean.float().flatten(1)
                    ua = (za - mu_z) @ P.t()
                    ub = (zb - mu_z) @ P.t()
                    z_swap = (za + (ub - ua) @ P).view(1, 4, 64, 64)
                    xp = vae.decode(z_swap).sample
                    # differentiable mild blur (view augmentation)
                    xp_aug = gauss_blur(xp, sigma=0.6, ksize=5)
                    f = csd_grad(xp_aug)
                    l_swap = (1.0 - (f * proto[tgt_style].unsqueeze(0)).sum(1)).mean()
                    # cycle consistency
                    z2 = vae.encode(xp).latent_dist.mean.float().flatten(1)
                    u2 = (z2 - mu_z) @ P.t()
                    perp2 = z2 - u2 @ P
                    perp_a = za - ua @ P
                    l_cycle = F.mse_loss(u2, ub.detach()) + F.mse_loss(perp2, perp_a.detach())
                    loss2 = (W_SWAP * l_swap + W_CYCLE * l_cycle) / args.accum
                loss2.backward()
                acc["swap"] += float(l_swap.detach()) / args.accum
                acc["cycle"] += float(l_cycle.detach()) / args.accum

        gn = float(torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0))
        opt.step(); sched.step(); ema.update(vae)
        upd += 1
        dt = time.perf_counter() - t_upd

        if upd % 10 == 0 or upd == 1:
            vram = torch.cuda.max_memory_allocated(0) / 1024**2
            el = time.time() - t_start
            extra = f" swap={acc['swap']:.4f} cyc={acc['cycle']:.5f}" if stg == 2 else ""
            print(f"[S{stg}] upd {upd:5d} | rec={acc['rec']:.4f} kl={acc['kl']:.2f} "
                  f"csd={acc['csd']:.4f} sup={acc['sup']:.3f}{extra} | g={gn:.2f} "
                  f"| {dt:.2f}s | VRAM={vram:.0f}M | {el/3600:.2f}h", flush=True)
            hist.append({"upd": upd, "stage": stg, **{k: round(v, 5) for k, v in acc.items()},
                         "g": round(gn, 3), "sec": round(dt, 2), "elapsed_h": round(el / 3600, 3)})

        if upd % args.save_every == 0:
            ck = OUT / f"ckpt_{'s1' if stg == 1 else 's2'}_u{upd:05d}.pt"
            torch.save({"model": vae.state_dict(), "ema": ema.state_dict(),
                        "optimizer": opt.state_dict(), "upd": upd, "stage": stg,
                        "args": vars(args)}, ck)
            print(f"[Saved] {ck}", flush=True)

    # final export
    final = OUT / "final_diffusers_vae"
    vae.load_state_dict(ema.state_dict())
    vae.save_pretrained(str(final))
    (OUT / "train_log.json").write_text(json.dumps({
        "updates": upd, "hours": round((time.time() - t_start) / 3600, 3),
        "stage_log": stage_log, "history": hist,
    }, indent=2), encoding="utf-8")
    print(f"\n[Done] {upd} updates in {(time.time()-t_start)/3600:.2f}h -> {final}", flush=True)


if __name__ == "__main__":
    main()
