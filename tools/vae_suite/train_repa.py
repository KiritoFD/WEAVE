"""WEAVE-VAE Style-REPA: Relation-Alignment Distillation from Frozen CSD.

Design (per approved plan):
  Teacher : frozen CSD ViT-L  ->  t_i = Normalize(CSD_style(x_i))
  Student : z = E(x)          ->  s_i = Normalize(Flatten(Pool8x8(Whiten(z_i))))
  Loss    : L_align = mean_{i != j} (S_ij - T_ij)^2 ,  S = s s^T ,  T = t t^T
  Total   : L = L_rec + beta * L_KL + lambda * L_align
  Batch   : 3 styles x 2 artists = 6 per microbatch, accumulate 8 -> effective 48
  Init    : A0 checkpoint (recon+KL pretrained)
  Run     : 500 steps short validation (P1)

No SWD, no reference bank, no MoCo queue, no hard-negative mining, no learnable head.
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

REPO_ROOT = Path("/root/Workspace/WEAVE_VAE")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools" / "12h"))

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

NORM_MEAN = [0.48145466, 0.4578275, 0.40821073]
NORM_STD = [0.26862954, 0.26130258, 0.27577711]


class StyleArtistBatchSampler(Sampler):
    """Yields batches of `n_styles * n_per_style` indices.

    Each batch: n_styles distinct styles, each contributing n_per_style images
    from DIFFERENT artists (when the style has >= n_per_style artists).
    """

    def __init__(self, dataset, n_styles=3, n_per_style=2, num_batches=500,
                 seed=42, min_artists=2):
        self.dataset = dataset
        self.n_styles = n_styles
        self.n_per_style = n_per_style
        self.num_batches = num_batches
        self.seed = seed
        self.batch_size = n_styles * n_per_style

        # build style -> artist -> [dataset positions]
        self.style_artist = collections.defaultdict(lambda: collections.defaultdict(list))
        for pos, raw_idx in enumerate(dataset.indices):
            rec = dataset.records_by_idx[raw_idx]
            self.style_artist[rec["style_id"]][rec.get("artist_id", -1)].append(pos)

        # only styles with >= min_artists distinct artists
        self.usable_styles = [
            s for s, amap in self.style_artist.items()
            if len([a for a, v in amap.items() if a >= 0 and len(v) > 0]) >= min_artists
        ]
        self.usable_styles.sort()
        print(f"[Sampler] usable styles (>= {min_artists} artists): {len(self.usable_styles)}", flush=True)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        rng = random.Random(self.seed)
        for _ in range(self.num_batches):
            styles = rng.sample(self.usable_styles, min(self.n_styles, len(self.usable_styles)))
            batch = []
            for s in styles:
                amap = self.style_artist[s]
                artists = [a for a, v in amap.items() if a >= 0 and len(v) > 0]
                rng.shuffle(artists)
                picked = 0
                for a in artists:
                    if picked >= self.n_per_style:
                        break
                    batch.append(rng.choice(amap[a]))
                    picked += 1
                # fallback: not enough artists -> sample any
                while picked < self.n_per_style:
                    a = rng.choice(list(amap.keys()))
                    batch.append(rng.choice(amap[a]))
                    picked += 1
            yield batch[:self.batch_size]


class LatentWhitener:
    """Fixed channel-wise + per-dimension whitening stats computed from A0 on train set."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean
        self.std = std

    @classmethod
    def compute(cls, vae, dataset, device, n_batches=60, batch_size=16):
        print(f"[Whiten] Computing pooled-latent stats over {n_batches*batch_size} training images...", flush=True)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        feats = []
        with torch.no_grad():
            for i, b in enumerate(loader):
                if i >= n_batches:
                    break
                post = vae.encode(b["image"].to(device)).latent_dist
                pooled = F.adaptive_avg_pool2d(post.mean.float(), (8, 8)).flatten(1)
                feats.append(pooled.cpu())
        f = torch.cat(feats)
        mean = f.mean(0)
        std = f.std(0).clamp(min=1e-5)
        print(f"[Whiten] done. mean range [{mean.min():.3f}, {mean.max():.3f}], "
              f"std range [{std.min():.3f}, {std.max():.3f}]", flush=True)
        return cls(mean, std)

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        pooled = F.adaptive_avg_pool2d(z.float(), (8, 8)).flatten(1)
        w = (pooled - self.mean.to(pooled.device)) / self.std.to(pooled.device)
        return F.normalize(w, dim=-1)


def relation_align_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """mean_{i != j} (S_ij - T_ij)^2, S and T are cosine similarity matrices."""
    S = student @ student.t()
    T = teacher @ teacher.t()
    B = S.shape[0]
    mask = ~torch.eye(B, dtype=torch.bool, device=S.device)
    diff = (S - T)[mask]
    return diff.pow(2).mean()


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae-init", default="/root/Workspace/WEAVE_VAE/runs_vae/A0/final_diffusers_vae")
    ap.add_argument("--raw-manifest", default="/root/Workspace/data/wikiart_512_raw/wikiart_512_raw_manifest.json")
    ap.add_argument("--output-dir", default="/root/Workspace/WEAVE_VAE/runs_vae/REPA")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--microbatch", type=int, default=6)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lambda-align", type=float, default=0.05)
    ap.add_argument("--beta-kl", type=float, default=2.252e-5)
    ap.add_argument("--lr-encoder", type=float, default=1e-5)
    ap.add_argument("--lr-decoder", type=float, default=5e-6)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = "cuda"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Style-REPA Training === steps={args.steps} mb={args.microbatch} accum={args.accum} "
          f"lambda_align={args.lambda_align}", flush=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # ---------- data ----------
    ds = RawMemmapWikiArtDataset(Path(args.raw_manifest), is_training=True)
    print(f"[Data] train pool: {len(ds)}", flush=True)

    # ---------- model (init from A0) ----------
    vae = AutoencoderKL.from_pretrained(args.vae_init).to(device)
    vae.train()
    vae.enable_gradient_checkpointing()

    # ---------- whitening stats ----------
    whiten = LatentWhitener.compute(vae, ds, device)
    torch.save({"mean": whiten.mean, "std": whiten.std}, out_dir / "whiten_stats.pt")

    # ---------- frozen CSD teacher ----------
    print("[Teacher] Loading frozen CSD ViT-L...", flush=True)
    csd, _, _ = csd_model.load_csd(device)
    csd.eval()
    for p in csd.parameters():
        p.requires_grad = False
    nm = torch.tensor(NORM_MEAN, device=device).view(1, 3, 1, 1)
    ns = torch.tensor(NORM_STD, device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def teacher_feat(x):
        x01 = (x.clamp(-1, 1) + 1.0) / 2.0
        x224 = F.interpolate(x01, (224, 224), mode="bicubic", align_corners=False)
        xin = ((x224 - nm) / ns).half()
        _, _, sty = csd(xin)
        return F.normalize(sty.float(), dim=-1)

    # ---------- sampler ----------
    sampler = StyleArtistBatchSampler(
        ds, n_styles=3, n_per_style=2,
        num_batches=args.steps * args.accum + 16, seed=args.seed)
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=4, pin_memory=True)

    # ---------- optimizer ----------
    opt = torch.optim.AdamW([
        {"params": list(vae.encoder.parameters()), "lr": args.lr_encoder},
        {"params": list(vae.decoder.parameters()), "lr": args.lr_decoder},
        {"params": list(vae.quant_conv.parameters()) + list(vae.post_quant_conv.parameters()),
         "lr": args.lr_encoder},
    ], betas=(0.9, 0.99), weight_decay=1e-4)

    warmup = max(10, int(0.02 * args.steps))
    def lr_lambda(s):
        if s < warmup:
            return float(s) / warmup
        prog = (s - warmup) / max(1, args.steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * prog))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    import lpips
    lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    ema = EMA(vae, 0.999)

    it = iter(loader)
    hist = []
    t0 = time.time()

    for step in range(1, args.steps + 1):
        ts = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        acc = {"rec": 0.0, "kl": 0.0, "align": 0.0}

        for _ in range(args.accum):
            try:
                b = next(it)
            except StopIteration:
                it = iter(loader)
                b = next(it)

            x = b["image"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                l_rec, _, _, post, _ = compute_mixed_reconstruction_loss(vae, x, lpips_fn)
                kl_e, kl_v, _, _ = compute_kl_divergence(post)

                t_feat = teacher_feat(x)                 # (B, D) normalized, no grad
                s_feat = whiten(post.mean)               # (B, 256) normalized, grad
                l_align = relation_align_loss(s_feat, t_feat)

                loss = (l_rec + args.beta_kl * kl_e + args.lambda_align * l_align) / args.accum

            loss.backward()
            acc["rec"] += float(l_rec.detach()) / args.accum
            acc["kl"] += float(kl_v) / args.accum
            acc["align"] += float(l_align.detach()) / args.accum

        gnorm = float(torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0))
        opt.step()
        sched.step()
        ema.update(vae)

        dt = time.perf_counter() - ts
        if step % 10 == 0 or step == 1:
            vram = torch.cuda.max_memory_allocated(0) / 1024**2
            eta = (args.steps - step) * dt / 3600
            print(f"[REPA] {step:4d}/{args.steps} | L_rec={acc['rec']:.4f} | KL={acc['kl']:.2f} | "
                  f"L_align={acc['align']:.5f} | g={gnorm:.2f} | lr={sched.get_last_lr()[0]:.2e} | "
                  f"VRAM={vram:.0f}M | {dt:.2f}s | ETA={eta:.2f}h", flush=True)
            hist.append({"step": step, **{k: round(v, 5) for k, v in acc.items()},
                         "grad_norm": round(gnorm, 3), "sec": round(dt, 2)})

        if step % args.save_every == 0 or step == args.steps:
            ck = out_dir / f"checkpoint_step_{step:06d}.pt"
            torch.save({"step": step, "model": vae.state_dict(), "ema": ema.state_dict(),
                        "optimizer": opt.state_dict(), "args": vars(args)}, ck)
            print(f"[Saved] {ck}", flush=True)

    final = out_dir / "final_diffusers_vae"
    vae.load_state_dict(ema.state_dict())
    vae.save_pretrained(str(final))
    print(f"[Export] {final}", flush=True)

    (out_dir / "train_log.json").write_text(json.dumps({
        "steps": args.steps, "hours": round((time.time() - t0) / 3600, 3), "history": hist,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
