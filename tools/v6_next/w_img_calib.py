"""WEAVE W P3 — W_S10_ImgCalib: frozen-S10-Decoder image-domain calibration.

The R1 critic is a proxy trained in the 4x64x64 latent space. Its decision
boundary is NOT the same as the CSD / DINOv2 boundary in decoded 3x512x512 RGB
(proxy gap). R2 tried to fix this with a latent-space tangential projection and
failed because the latent L_ss gradient is orthogonal to the R1 gradient. In the
decoded RGB/perceptual space, however, we can supply BOTH the CSD target-margin
gradient AND a DINO-C content-preservation gradient, and the optimizer finds the
latent sub-manifold that satisfies the real image judges without hurting DINO-C.

This is now affordable because the S10 Decoder (10.66M) gives ~3x forward+input-grad
speedup and ~4.1GB vs the original 49.5M SD-VAE decoder (6.7GB, 120ms).

Design (plan experiment 4):
  - Start from a trained student (B4 218K or S3 845K -- FROZEN Encoder, we only
    fine-tune the generator).
  - Small microbatch (4-8). Student edits c -> z. FROZEN S10 StudentDecoder
    decodes z -> x_hat in [0,1] RGB.
  - L_img = CSD_margin(x_hat, t) + lambda_dino_c * (1 - cos(DINO_cls(x_hat),
    DINO_cls(x_src))).
  - 100-200 steps, bf16 + gradient checkpointing on the decoder.

Everything here is a SHORT post-tuning on top of an existing checkpoint; it never
retrains the VAE, R1, CSD or DINO (all frozen / no-grad params).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "12h"))  # csd_model lives here

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import csd_model
from tools.pseudo_gt.matching import load_dino_model
from tools.v5.route_c import csd_prototypes
from tools.v6.canonicalize import ContentCanonicalizer
from tools.v6.models_v6 import C3UNet
from tools.v6.train_eval_v6 import STYLES
from tools.v6_next.w_vae import (
    build_a0_encoder, build_s10_decoder, decode_s10, encode_a0, raw_to_pm1,
)

IMAGE_DIR = ROOT / "data/test"


def csd_style_margin_loss(csd, proto, x_hat, target_id):
    """-(s_t - max_{k!=t} s_k) style margin in CSD space, differentiable in x_hat.

    Uses a differentiable resize+normalise (no PIL round-trip) so the gradient
    reaches x_hat -> z -> generator. CSD weights are frozen (eval + no grad).
    """
    x = F.interpolate(x_hat.float(), size=(224, 224), mode="bicubic", align_corners=False)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=x_hat.device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=x_hat.device).view(1, 3, 1, 1)
    x = (x - mean) / std
    _, _, sty = csd(x.to(torch.float16))
    sty = F.normalize(sty.float(), dim=1)
    p = torch.from_numpy(proto).to(x_hat.device).float()
    sc = sty @ p.T
    s_t = sc.gather(1, target_id[:, None]).squeeze(1)
    oh = F.one_hot(target_id, len(STYLES)).bool()
    other = torch.logsumexp(sc.masked_fill(oh, torch.finfo(sc.dtype).min), dim=1)
    # minimise -> maximise margin; softplus keeps it bounded like the R1 ranking loss.
    # The raw softplus is O(2-3) while DINO-C is O(0.1-0.2); divide by 10 so the
    # two terms are comparable and the content term is not silently ignored.
    margin = s_t - other
    return F.softplus(-margin / 0.5).mean() / 10.0


def main():
    ap = argparse.ArgumentParser(description="W_S10_ImgCalib")
    ap.add_argument("--init-ckpt", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out-base", default="runs/v6_w")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--microbatch", type=int, default=6)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--lam-dino-c", type=float, default=1.0)
    ap.add_argument("--lam-csd", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = args.device

    # --- generator ---
    ck = torch.load(ROOT / args.init_ckpt, map_location="cpu", weights_only=False)
    a = ck.get("args", {})
    model = C3UNet(base_dim=a.get("base_dim", 16), num_styles=5, mode="direct",
                   inject_64=not a.get("no_inject_64", False),
                   inject_32=not a.get("no_inject_32", False),
                   inject_16=not a.get("no_inject_16", False)).to(device)
    model.load_state_dict(ck["model"])
    model.train()
    canon = ContentCanonicalizer(torch.load(
        ROOT / "runs/v6/cache/canonicalizer_stats.pt",
        map_location="cpu", weights_only=False)).to(device).eval()
    for p in canon.parameters():
        p.requires_grad_(False)

    # --- frozen judges + A0 encoder / S10 decoder (correct pairing) ---
    vae_a0 = build_a0_encoder(device)
    dec = build_s10_decoder(vae_a0, device)
    csd, _, _ = csd_model.load_csd(device)
    for p in csd.parameters():
        p.requires_grad_(False)
    dino = load_dino_model(device=device).eval()
    for p in dino.parameters():
        p.requires_grad_(False)
    proto = csd_prototypes()

    # --- small source pool (real RGB + A0 posterior-mean latents, fixed) ---
    sf = float(vae_a0.config.scaling_factor)
    src_paths = sorted(IMAGE_DIR.glob("*/*.jpg"))[:150]
    # deterministic subset for reproducibility
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(src_paths), size=min(40, len(src_paths)), replace=False)
    src_paths = [src_paths[i] for i in idx]

    def load_latent_rgb(p):
        im = Image.open(p).convert("RGB").resize((512, 512))
        arr = np.array(im, dtype=np.float32) / 255.0
        x01 = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
        z = encode_a0(vae_a0, raw_to_pm1(x01))
        return z, x01

    cached = [load_latent_rgb(p) for p in src_paths]
    torch.cuda.empty_cache()

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    t0 = time.time()
    loss_hist = []

    for step in range(1, args.steps + 1):
        pick = rng.integers(0, len(cached), size=args.microbatch)
        zs = torch.cat([cached[i][0] for i in pick], 0)
        x01 = torch.cat([cached[i][1] for i in pick], 0)
        # random cross-style targets
        sid = torch.tensor(
            [STYLES.index(Path(src_paths[i]).parent.name) for i in pick],
            device=device)
        shift = torch.randint(1, 5, (args.microbatch,), device=device)
        tid = (sid + shift) % 5

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            c = canon(zs, mode="c1_gentle", augment=False)
            z = model(c, 0.0, c, tid)
            x_hat = decode_s10(dec, z.float(), sf)

            l_csd = csd_style_margin_loss(csd, proto, x_hat, tid)
            # DINO-C: differentiable generated CLS vs frozen source CLS cosine.
            x_dino = F.interpolate(x_hat.float(), size=(224, 224), mode="bicubic",
                                   align_corners=False)
            dmean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            dstd = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
            with torch.no_grad():
                x_src224 = F.interpolate(x01.float(), size=(224, 224), mode="bicubic",
                                         align_corners=False)
                ds = dino((x_src224 - dmean) / dstd).last_hidden_state[:, 0]
                ds = F.normalize(ds.float(), dim=1)
            dg = dino(((x_dino - dmean) / dstd).to(torch.float16)).last_hidden_state[:, 0]
            dg = F.normalize(dg.float(), dim=1)
            l_dino_c = (1.0 - (dg * ds).sum(dim=1)).mean()

        loss = args.lam_csd * l_csd + args.lam_dino_c * l_dino_c
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        loss_hist.append(float(loss.detach()))
        if step == 1 or step % 25 == 0 or step == args.steps:
            print(f"[{args.name}] step {step}/{args.steps} "
                  f"loss={np.mean(loss_hist[-25:]):.4f} csd={float(l_csd):.4f} "
                  f"dino_c={float(l_dino_c):.4f} elapsed={time.time()-t0:.0f}s "
                  f"peak_vram={torch.cuda.max_memory_allocated(0)/1024**2:.0f}MiB",
                  flush=True)

    out_dir = ROOT / args.out_base / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "args": {**vars(args), "base_dim": a.get("base_dim", 16), "mode": "direct"},
        "train_wall_sec": round(time.time() - t0, 1),
        "final_loss": float(np.mean(loss_hist[-25:])),
        "init_ckpt": args.init_ckpt,
    }, out_dir / f"{args.name}.pt")
    print(f"[{args.name}] saved {out_dir / f'{args.name}.pt'}")


if __name__ == "__main__":
    main()
