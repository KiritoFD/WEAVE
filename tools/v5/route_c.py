"""WEAVE V5 Route C: Style-Conditional Rectified Flow on real D5 art latents.

Implements the decision-tree fallback after P1 Oracle and B1 Style-Encoder both
failed to raise independent CSD dT:

    p(z | style),  z_tau = (1-tau)*eps + tau*z_real,  v* = z_real - eps

Then SDEdit-style editing from a partially noised source:

    z_tau0 = tau0*z_source + (1-tau0)*eps ; integrate tau0 -> 1 conditioned on
    the requested target Style-ID (never on the current sample's GT).

Stages (run serially, one GPU process at a time):
  train : train the ~2M-parameter flow prior (optionally low/mid-noise only)
  check : pure-noise generation prior credibility (5x5 CSD prototype matrix)
  quick : 200-task sweep over tau0 x coarse-guidance x seeds + Style-ID swap
  final : 600 cross + 150 same full render (two seeds) + v4_eval750 protocol
          + final A/B/C verdict

All evaluation uses VAE posterior mean, cached real-art latents, and the same
CSD / DINO / LPIPS protocol frozen in P0.  No RGB training loss, no GAN, no
CSD backprop into the generator, no extra teacher.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools" / "12h"))
sys.path.insert(0, str(REPO_ROOT / "tools" / "trt"))

import paths  # noqa: E402
import csd_model  # noqa: E402
from tools.eval_fast import (  # noqa: E402
    METRIC_HEADER, DINO_HEADER, MetricEngine, Timer, build_protocol, write_csv,
)
from tools.pseudo_gt.matching import encode_dino, load_dino_model  # noqa: E402
from tools.legacy_utils.run_evaluation import _load_eval_image_tensor, to_lpips_input  # noqa: E402
from tools.legacy_root.wavelet import dwt2_haar, idwt2_haar  # noqa: E402

STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
V5 = REPO_ROOT / "runs" / "v5"
RC = V5 / "route_c"
RENDER_ROOT = paths.asset_root() / "eval_assets" / "v5_route_c"

# ============================================================================
# model
# ============================================================================

class FiLMResBlock(nn.Module):
    def __init__(self, channels: int, cond_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, channels), channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, channels), channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.film = nn.Linear(cond_dim, channels * 2)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.film(cond).chunk(2, dim=-1)
        s = 1.0 + scale[:, :, None, None]
        b = shift[:, :, None, None]
        x = F.silu(self.norm1(h))
        x = self.conv1(x)
        x = F.silu(self.norm2(x) * s + b)
        x = self.conv2(x)
        return h + x


class StyleRectifiedFlow(nn.Module):
    """~2M params, 4x64x64 latent velocity field conditioned on (tau, Style-ID)."""

    def __init__(self, base: int = 128, n_blocks: int = 6, num_styles: int = 5):
        super().__init__()
        self.base = base
        cond_dim = base * 2
        self.time_mlp = nn.Sequential(
            nn.Linear(base, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim)
        )
        self.style_emb = nn.Embedding(num_styles, cond_dim)
        self.stem = nn.Conv2d(4, base, 3, padding=1)
        self.blocks = nn.ModuleList([FiLMResBlock(base, cond_dim) for _ in range(n_blocks)])
        self.out_norm = nn.GroupNorm(min(8, base), base)
        self.out = nn.Conv2d(base, 4, 3, padding=1)
        # Route C is a generative prior, not an identity-initialised operator:
        # a small non-zero head lets the residual trunk receive gradients
        # from step 1 instead of waiting for the last layer to break symmetry.
        nn.init.normal_(self.out.weight, std=0.01)
        nn.init.zeros_(self.out.bias)

    @staticmethod
    def _time_embedding(tau: torch.Tensor, dim: int) -> torch.Tensor:
        half = dim // 2
        freq = torch.exp(torch.arange(half, device=tau.device, dtype=torch.float32)
                         * (-np.log(10000.0) / max(half - 1, 1)))
        args = tau.float().view(-1, 1) * freq.view(1, -1) * 2.0 * np.pi
        return torch.cat([args.sin(), args.cos()], dim=-1)[:, :dim]

    def forward(self, z: torch.Tensor, tau: torch.Tensor, style_id: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(tau):
            tau = torch.full((z.shape[0],), float(tau), device=z.device, dtype=z.dtype)
        t_emb = self.time_mlp(self._time_embedding(tau, self.base))
        cond = t_emb + self.style_emb(style_id.long())
        h = self.stem(z)
        for blk in self.blocks:
            h = blk(h, cond)
        return self.out(F.silu(self.out_norm(h)))


def load_flow(ckpt: Path, device: str) -> StyleRectifiedFlow:
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    args = payload.get("model_args", {})
    model = StyleRectifiedFlow(**args).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model


@torch.inference_mode()
def integrate(
    model: StyleRectifiedFlow,
    z_start: torch.Tensor,
    tau0: float,
    style_id: torch.Tensor,
    steps: int = 8,
) -> torch.Tensor:
    """Euler integration of dz/dtau = v_theta(z, tau, style_id) from tau0 to 1."""
    z = z_start
    dtau = (1.0 - float(tau0)) / int(steps)
    for i in range(int(steps)):
        tau = float(tau0) + dtau * i
        t = torch.full((z.shape[0],), tau, device=z.device, dtype=z.dtype)
        v = model(z, t, style_id)
        z = z + dtau * v
    return z


def coarse_guidance(z_gen: torch.Tensor, z_src: torch.Tensor) -> torch.Tensor:
    """Keep source normalised coarse layout, allow generated LL appearance stats."""
    ll_g, lh, hl, hh = dwt2_haar(z_gen)
    ll_s, _, _, _ = dwt2_haar(z_src)

    def stats(x: torch.Tensor):
        m = x.mean(dim=(1, 2, 3), keepdim=True)
        s = x.std(dim=(1, 2, 3), keepdim=True)
        return m, s

    mu_g, sd_g = stats(ll_g)
    mu_s, sd_s = stats(ll_s)
    ll_new = (ll_s - mu_s) / (sd_s + 1e-6) * sd_g + mu_g
    return idwt2_haar(ll_new, lh, hl, hh)


# ============================================================================
# shared assets (VAE / TRT / CSD / DINO / prototypes)
# ============================================================================

def find_vae_dir() -> Path:
    candidates = [
        paths.asset_root() / "eval_cache" / "hf" / "modelscope" / "stabilityai_sd-vae-ft-ema" / "models" / "stabilityai--sd-vae-ft-ema" / "snapshots",
        REPO_ROOT / "eval_cache" / "hf" / "modelscope" / "stabilityai_sd-vae-ft-ema" / "models" / "stabilityai--sd-vae-ft-ema" / "snapshots",
    ]
    for c in candidates:
        if c.exists():
            for snap in sorted(c.glob("*")):
                if (snap / "config.json").exists():
                    return snap
    raise FileNotFoundError("SD1.5 VAE local snapshot not found")


def load_vae(device: str):
    from diffusers import AutoencoderKL
    return AutoencoderKL.from_pretrained(
        str(find_vae_dir()), torch_dtype=torch.float16, local_files_only=True
    ).to(device).eval()


def load_trt(vae, device: str):
    import trt_vae_decoder
    return trt_vae_decoder.TrtVaeDecoder(
        REPO_ROOT / "runs" / "trt" / "engines", vae, precision="fp16", device=device
    )


def decode_trt(trt, vae, z: torch.Tensor, batch: int = 8) -> torch.Tensor:
    """Decode VAE-scaled latents to [0,1] float tensors via fixed-shape TRT b8."""
    out = []
    with torch.inference_mode():
        for s in range(0, z.shape[0], batch):
            chunk = z[s:s + batch].half()
            n = chunk.shape[0]
            if n < batch:
                pad = chunk[-1:].expand(batch - n, -1, -1, -1)
                chunk = torch.cat([chunk, pad], 0)
            out.append(trt.decode(chunk)[:n].float().clamp(0.0, 1.0))
    return torch.cat(out, 0)


def csd_prototypes() -> np.ndarray:
    p = paths.asset_root() / "pseudo_gt_24h" / "csd_proto.pt"
    d = torch.load(p, map_location="cpu", weights_only=False)
    return np.stack([d[s] for s in STYLES])


def source_latents(device: str, max_src: int) -> Tuple[torch.Tensor, List[Path], list, dict]:
    """Deterministic VAE posterior-mean latents for the frozen eval protocol."""
    test_images, requests, style_meta = build_protocol(
        REPO_ROOT / "data" / "test", max_src_samples=max_src, ref_pool="first"
    )
    # build_protocol returns {sid: (style_name, [Path, ...])}
    src_paths = [p for _, (_, paths) in sorted(test_images.items()) for p in paths]
    bad = [p for p in src_paths if not isinstance(p, Path) or not p.exists()]
    if bad:
        raise RuntimeError(f"source_latents: invalid source paths: {bad[:3]}")
    vae = load_vae(device)
    chunks = []
    with torch.inference_mode():
        for s in range(0, len(src_paths), 8):
            imgs = torch.stack([_load_eval_image_tensor(p, size=512) for p in src_paths[s:s + 8]]).to(device)
            imgs = imgs * 2.0 - 1.0
            dist = vae.encode(imgs.half()).latent_dist
            chunks.append((dist.mean * vae.config.scaling_factor).float().cpu())
    z = torch.cat(chunks, 0).to(device)
    del vae
    torch.cuda.empty_cache()
    return z, src_paths, requests, style_meta


# ============================================================================
# stage: train
# ============================================================================

def stage_train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    RC.mkdir(parents=True, exist_ok=True)
    out = Path(args.out)

    orig_dir = paths.asset_root() / "pseudo_gt_24h" / "orig_mean_fp16"
    lats, labels = [], []
    for sid, style in enumerate(STYLES):
        d = torch.load(orig_dir / f"{sid:02d}_{style}.pt", map_location="cpu", weights_only=False)
        x = d["latents"].float()
        lats.append(x)
        labels.append(torch.full((x.shape[0],), sid, dtype=torch.long))
        print(f"[train] {style}: {x.shape[0]} real art latents", flush=True)
    X = torch.cat(lats, 0).to(device)          # all 15.5k latents resident on GPU
    Y = torch.cat(labels, 0).to(device)
    n = X.shape[0]
    print(f"[train] dataset on GPU: {X.shape}, {X.numel()*X.element_size()/2**20:.0f} MiB", flush=True)

    torch.manual_seed(args.seed)
    model = StyleRectifiedFlow(base=args.base, n_blocks=args.n_blocks).to(device)
    if args.init:
        init = torch.load(args.init, map_location="cpu", weights_only=False)
        model.load_state_dict(init["model"])
        print(f"[train] initialised from {args.init}", flush=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] model params={n_params/1e6:.2f}M noise_mode={args.noise_mode}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    mb, accum = int(args.microbatch), int(args.accum)
    loss_hist = []
    t0 = time.time()
    model.train()

    step = 1
    while step <= args.steps:
        try:
            optimizer.zero_grad(set_to_none=True)
            for _ in range(accum):
                idx = torch.randint(0, n, (mb,), device=device)
                z_real = X[idx]
                sid = Y[idx]
                eps = torch.randn_like(z_real)
                if args.noise_mode == "lowmid":
                    tau = torch.rand(mb, device=device) * 0.6 + 0.4   # tau in [0.4, 1.0]
                else:
                    tau = torch.rand(mb, device=device)
                tv = tau.view(-1, 1, 1, 1)
                z_tau = (1.0 - tv) * eps + tv * z_real
                v_star = z_real - eps
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    v_pred = model(z_tau, tau, sid)
                    loss = F.mse_loss(v_pred.float(), v_star) / accum
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        except torch.cuda.OutOfMemoryError:
            if mb <= 32:
                raise
            mb = max(32, mb // 2)
            accum *= 2
            torch.cuda.empty_cache()
            optimizer.zero_grad(set_to_none=True)
            print(f"[train] OOM -> fallback microbatch={mb} accum={accum}", flush=True)
            continue

        loss_hist.append(float(loss.detach()) * accum)
        if step % 200 == 0 or step == 1:
            peak = torch.cuda.max_memory_allocated(0) / 2**20
            print(f"[train] step {step}/{args.steps} loss={np.mean(loss_hist[-200:]):.5f} "
                  f"lr={scheduler.get_last_lr()[0]:.2e} mb={mb}x{accum} "
                  f"peak_vram={peak:.0f}MiB elapsed={time.time()-t0:.0f}s", flush=True)
        if step in (1000, 2000, 3000, 4000) and step <= args.steps:
            pass
        step += 1

    payload = {
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "model_args": {"base": args.base, "n_blocks": args.n_blocks},
        "steps": args.steps, "noise_mode": args.noise_mode,
        "effective_batch": mb * accum, "seed": args.seed,
        "final_loss": float(np.mean(loss_hist[-100:])),
        "peak_vram_mib": round(torch.cuda.max_memory_allocated(0) / 2**20, 1),
        "wall_sec": round(time.time() - t0, 1),
        "n_params": int(n_params),
    }
    torch.save(payload, out)
    print(f"[train] saved {out} | final_loss={payload['final_loss']:.5f} "
          f"peak_vram={payload['peak_vram_mib']}MiB wall={payload['wall_sec']}s", flush=True)


# ============================================================================
# stage: check (pure-noise generation credibility)
# ============================================================================

def stage_check(args):
    device = "cuda"
    out = Path(args.out)
    model = load_flow(Path(args.ckpt), device)
    csd, _, _ = csd_model.load_csd(device)
    proto = csd_prototypes()
    vae = load_vae(device)
    trt = load_trt(vae, device)

    torch.manual_seed(args.seed)
    B = args.n_per_style * len(STYLES)
    eps = torch.randn(B, 4, 64, 64, device=device)
    tgt = torch.tensor([i // args.n_per_style for i in range(B)], device=device, dtype=torch.long)
    if args.tau0 > 0.0:
        # Low/mid-noise denoising check: start from a partially noised *real*
        # artwork latent of the same style, not from pure eps.
        orig_dir = paths.asset_root() / "pseudo_gt_24h" / "orig_mean_fp16"
        real = []
        for sid, style in enumerate(STYLES):
            d = torch.load(orig_dir / f"{sid:02d}_{style}.pt", map_location="cpu", weights_only=False)
            pool = d["latents"].float()
            idx = torch.randint(0, pool.shape[0], (args.n_per_style,))
            real.append(pool[idx])
        z_real = torch.cat(real, 0).to(device)
        z_start = args.tau0 * z_real + (1.0 - args.tau0) * eps
        start_mode = f"partially_noised_real_art_tau{args.tau0:.2f}"
    else:
        z_start = eps
        start_mode = "pure_noise"
    with torch.inference_mode():
        z = integrate(model, z_start, args.tau0, tgt, steps=args.steps)
        img = decode_trt(trt, vae, z, batch=8)
    u8 = img.mul(255.0).round().clamp(0, 255).to(torch.uint8).cpu()
    arr = u8.permute(0, 2, 3, 1).numpy()
    pils = [Image.fromarray(arr[i]) for i in range(B)]
    feats = csd_model.feats_pils(csd, pils, batch=16, device=device)
    sim = feats @ proto.T                          # (B, 5)
    matrix = np.zeros((5, 5), dtype=float)
    for s in range(5):
        matrix[s] = sim[s * args.n_per_style:(s + 1) * args.n_per_style].mean(0)
    diag = float(np.mean(np.diag(matrix)))
    offdiag = float((matrix.sum() - np.trace(matrix)) / 20)
    credible = bool(diag - offdiag >= 0.05)
    report = {
        "ckpt": args.ckpt, "tau0": args.tau0, "start_mode": start_mode,
        "noise_mode": torch.load(args.ckpt, map_location="cpu", weights_only=False).get("noise_mode"),
        "n_per_style": args.n_per_style, "csd_matrix": matrix.tolist(),
        "mean_diag": diag, "mean_offdiag": offdiag, "margin": diag - offdiag,
        "generation_prior_credible": credible,
    }
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


# ============================================================================
# stage: quick (200-task sweep)
# ============================================================================

def stage_quick(args):
    device = "cuda"
    RC.mkdir(parents=True, exist_ok=True)
    model = load_flow(Path(args.ckpt), device)

    tasks = json.loads((V5 / "v5_eval200_tasks.json").read_text(encoding="utf-8"))
    lat_dict = torch.load(V5 / "cache" / "eval_src_latents.pt", map_location="cpu")
    stems = sorted({t["src_stem"] for t in tasks})
    z_all = torch.stack([lat_dict[s] for s in stems]).to(device)
    stem_idx = {s: i for i, s in enumerate(stems)}
    src_idx = torch.tensor([stem_idx[t["src_stem"]] for t in tasks], device=device)
    tgt_ids = torch.tensor([t["tgt_style_id"] for t in tasks], device=device, dtype=torch.long)

    proto = csd_prototypes()
    csd, _, _ = csd_model.load_csd(device)
    dino = load_dino_model(device=device)
    vae = load_vae(device)
    trt = load_trt(vae, device)
    import lpips
    lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    # original source images (LPIPS reference, not VAE roundtrip)
    src_tensors = {}
    for t in tasks:
        if t["src_stem"] not in src_tensors:
            src_tensors[t["src_stem"]] = _load_eval_image_tensor(
                Path(t["src_path"]), size=512).unsqueeze(0).to(device)

    # source features come from the decoded posterior-mean latents (frozen P0 protocol)
    csd_src50 = torch.load(V5 / "cache" / "eval_src_csd_feats.pt", map_location="cpu", weights_only=False)
    csd_src_cache = {s: np.asarray(csd_src50[s]) for s in stems}
    dino_src_cache = {}
    with torch.inference_mode():
        z_src50 = z_all
        img_src = decode_trt(trt, vae, z_src50, batch=8)
        u8_src = img_src.mul(255.0).round().clamp(0, 255).to(torch.uint8).cpu()
        arr_src = u8_src.permute(0, 2, 3, 1).numpy()
        pils_src = [Image.fromarray(arr_src[i]) for i in range(len(stems))]
        dcls, _ = encode_dino(dino, pils_src, batch=16, device=device)
        for i, s in enumerate(stems):
            dino_src_cache[s] = dcls[i].float().cpu().numpy()

    configs = []
    for tau0 in (0.35, 0.55, 0.75):
        for guide in (0, 1):
            configs.append({"name": f"t{tau0:.2f}_g{guide}", "tau0": tau0,
                            "guide": bool(guide), "seed": 42})

    def generate_and_score(cfg, ids_override=None, tag=""):
        seed = cfg["seed"]
        torch.manual_seed(seed)
        eps = torch.randn(len(tasks), 4, 64, 64, device=device)
        z_src = z_all[src_idx]
        t0 = float(cfg["tau0"])
        z_start = t0 * z_src + (1.0 - t0) * eps
        ids = tgt_ids if ids_override is None else ids_override
        with torch.inference_mode():
            z_gen = integrate(model, z_start, t0, ids, steps=args.steps)
            if cfg["guide"]:
                z_gen = coarse_guidance(z_gen, z_src)
            img = decode_trt(trt, vae, z_gen, batch=8)
        # Match the frozen V4/C4 protocol exactly: quantise to PNG uint8 and
        # dequantise before LPIPS/CSD/DINO, so C3 and C4 scores are comparable.
        u8 = img.mul(255.0).round().clamp(0, 255).to(torch.uint8).cpu()
        gen01 = u8.float().div_(255.0)
        arr_gen = u8.permute(0, 2, 3, 1).numpy()
        pils = [Image.fromarray(arr_gen[i]) for i in range(len(tasks))]
        with torch.inference_mode():
            fg = csd_model.feats_pils(csd, pils, batch=16, device=device)
            dg, _ = encode_dino(dino, pils, batch=16, device=device)
            dg = dg.float().cpu().numpy()
            lps = []
            for s in range(0, len(tasks), 16):
                chunk = gen01[s:s + 16].to(device)
                src = torch.cat([src_tensors[t["src_stem"]] for t in tasks[s:s + 16]], 0)
                lps.append(lpips_fn(to_lpips_input(chunk), to_lpips_input(src)).flatten().cpu())
            lps = torch.cat(lps, 0).numpy()

        rows = []
        for i, t in enumerate(tasks):
            fs = csd_src_cache[t["src_stem"]]
            tgt, sid = t["tgt_style_id"], t["src_style_id"]
            dT = float(fg[i] @ proto[tgt] - fs @ proto[tgt])
            dS = float(fg[i] @ proto[sid] - fs @ proto[sid])
            rows.append({
                "task_id": t["task_id"], "src_style": t["src_style"], "tgt_style": t["tgt_style"],
                "dT": dT, "dS": dS, "dM": dT - dS,
                "dino_c": float(dg[i] @ dino_src_cache[t["src_stem"]]),
                "lpips": float(lps[i]),
                "cos_tgt": float(fg[i] @ proto[tgt]),
            })
        dT = np.array([r["dT"] for r in rows])
        per_style = {}
        for s in STYLES:
            v = np.array([r["dT"] for r in rows if r["tgt_style"] == s])
            per_style[s] = float(v.mean())
        rng = np.random.default_rng(0)
        boot = rng.choice(dT, (2000, len(dT))).mean(1)
        summary = {
            "name": cfg["name"] + tag, "tau0": cfg["tau0"], "guide": cfg["guide"], "seed": seed,
            "n": len(rows), "mean_dT": float(dT.mean()),
            "dT_ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "mean_dS": float(np.mean([r["dS"] for r in rows])),
            "mean_dM": float(np.mean([r["dM"] for r in rows])),
            "mean_dino_c": float(np.mean([r["dino_c"] for r in rows])),
            "mean_lpips": float(np.mean([r["lpips"] for r in rows])),
            "frac_dT_positive": float(np.mean(dT > 0)),
            "per_target_dT": per_style,
            "styles_improved": int(np.sum([v > 0 for v in per_style.values()])),
            "ukiyo_dT": per_style["Ukiyo_e"],
            "peak_vram_mib": round(torch.cuda.max_memory_allocated(0) / 2**20, 1),
        }
        return summary, rows

    report = {"ckpt": args.ckpt, "sweep": {}, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    for cfg in configs:
        t0 = time.time()
        s, _ = generate_and_score(cfg)
        s["wall_sec"] = round(time.time() - t0, 1)
        report["sweep"][s["name"]] = s
        print(f"[quick] {s['name']}: dT={s['mean_dT']:+.4f} dM={s['mean_dM']:+.4f} "
              f"DINO-C={s['mean_dino_c']:.4f} LPIPS={s['mean_lpips']:.4f} "
              f"styles+={s['styles_improved']}/5 ukiyo={s['ukiyo_dT']:+.4f} "
              f"vram={s['peak_vram_mib']}MiB {s['wall_sec']}s", flush=True)

    # gate: independent CSD dT + content preservation + style coverage
    def gate_ok(s):
        return (s["mean_dT"] >= 0.03 and s["mean_dino_c"] >= 0.78 and
                s["mean_lpips"] <= 0.30 and s["styles_improved"] >= 4 and
                s["ukiyo_dT"] > 0)

    eligible = [s for s in report["sweep"].values() if gate_ok(s)]
    # score by dT with a content penalty for degraded LPIPS
    def score(s):
        return s["mean_dT"] - max(0.0, s["mean_lpips"] - 0.27) * 0.1
    best = max(eligible, key=score) if eligible else max(report["sweep"].values(), key=score)

    # Style-ID selectivity on the best config (correct vs wrong Style-ID)
    wrong_ids = torch.tensor([(t["tgt_style_id"] + 1) % 5 if (t["tgt_style_id"] + 1) % 5 != t["src_style_id"]
                              else (t["tgt_style_id"] + 2) % 5 for t in tasks],
                             device=device, dtype=torch.long)
    best_name = best["name"]
    best_cfg = next(c for c in configs if c["name"] == best_name)
    _, rows_correct = generate_and_score(best_cfg, tag="_regen")
    cos_correct = float(np.mean([r["cos_tgt"] for r in rows_correct]))
    # wrong-ID pass: same noise/guidance, only the conditioning label changes
    torch.manual_seed(best_cfg["seed"])
    eps = torch.randn(len(tasks), 4, 64, 64, device=device)
    z_src = z_all[src_idx]
    t0 = float(best_cfg["tau0"])
    with torch.inference_mode():
        z_wrong = integrate(model, t0 * z_src + (1 - t0) * eps, t0, wrong_ids, steps=args.steps)
        if best_cfg["guide"]:
            z_wrong = coarse_guidance(z_wrong, z_src)
        img_w = decode_trt(trt, vae, z_wrong, batch=8)
    u8_w = img_w.mul(255.0).round().clamp(0, 255).to(torch.uint8).cpu()
    arr_w = u8_w.permute(0, 2, 3, 1).numpy()
    pils_w = [Image.fromarray(arr_w[i]) for i in range(len(tasks))]
    fw = csd_model.feats_pils(csd, pils_w, batch=16, device=device)
    cos_wrong = float(np.mean([fw[i] @ proto[t["tgt_style_id"]] for i, t in enumerate(tasks)]))

    report["decision"] = {
        "ckpt": args.ckpt, "gate_passed": bool(eligible), "best": best["name"],
        "best_config": {"ckpt": args.ckpt, "tau0": best["tau0"], "guide": best["guide"], "seed": 42},
        "best_dT": best["mean_dT"], "best_dino_c": best["mean_dino_c"], "best_lpips": best["mean_lpips"],
        "style_id_selectivity": {"cos_correct_id": cos_correct, "cos_wrong_id": cos_wrong,
                                 "margin": cos_correct - cos_wrong,
                                 "selective": bool(cos_correct - cos_wrong > 0.005)},
        "next_stage": "final_750_full_eval",
    }
    print(json.dumps(report["decision"], indent=2), flush=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    # machine-readable best config for the shell driver
    (RC / "best_config.json").write_text(json.dumps(report["decision"]["best_config"]), encoding="utf-8")
    print(f"[quick] saved {args.out}", flush=True)


# ============================================================================
# stage: final (600 cross + 150 same, two seeds, full protocol)
# ============================================================================

def render_cell(args, model, cfg, cell: str, seed: int, device: str) -> dict:
    out_dir = RENDER_ROOT / cell
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "v4_eval750.json").exists() and not args.force:
        print(f"[final] {cell} already evaluated, skipping", flush=True)
        return json.loads((out_dir / "v4_eval750.json").read_text(encoding="utf-8"))

    z_src, src_paths, requests, style_meta = source_latents(device, max_src=30)
    num_styles = len(style_meta)
    req_map = {(str(r["src_path"]), int(r["tgt_style_id"])): r for r in requests}

    vae = load_vae(device)
    trt = load_trt(vae, device)
    ns = argparse.Namespace(
        disable_clip=False, disable_lpips=False, disable_dino=False,
        test_dir=str(REPO_ROOT / "data" / "test"),
        clip_model="openai/clip-vit-base-patch32", clip_local_only=True,
        clip_cache_dir=str(paths.eval_cache_root("hf") / "hub"), clip_skip_processor=False,
        amp=False, lpips_net="alex", lpips_chunk=32,
        dino_model="facebook/dinov2-small",
        dino_cache_dir=str(paths.eval_cache_root("hf")), allow_network=False,
        ref_cache="", max_ref_cache=16, dino_max_refs=30, image_size=512,
        metric_batch=32, ref_batch=32, only_clip_style=True,
        clip_style_idt_baseline=0.0, clip_vectors=False,
    )
    engine = MetricEngine(ns, {}, style_meta, device)
    engine.load(Timer())
    engine.prepare(Timer(), src_paths)

    torch.manual_seed(seed)
    eps_all = torch.randn(len(src_paths), 4, 64, 64, device=device)
    t0 = float(cfg["tau0"])
    rows_all = []
    save_pool = ThreadPoolExecutor(max_workers=8)
    futs = []
    t_start = time.perf_counter()
    for b in range(0, len(src_paths), 8):
        chunk_paths = src_paths[b:b + 8]
        z_s = z_src[b:b + len(chunk_paths)]
        eps = eps_all[b:b + len(chunk_paths)]
        n = z_s.shape[0]
        for tgt in range(num_styles):
            rows = [req_map[(str(p), tgt)] for p in chunk_paths]
            ids = torch.full((n,), tgt, dtype=torch.long, device=device)
            with torch.inference_mode():
                z_start = t0 * z_s + (1.0 - t0) * eps
                z_gen = integrate(model, z_start, t0, ids, steps=args.steps)
                if cfg["guide"]:
                    z_gen = coarse_guidance(z_gen, z_s)
                img = decode_trt(trt, vae, z_gen, batch=8)
            u8 = img.mul(255.0).round().clamp(0, 255).to(torch.uint8)
            gen01 = u8.float().div_(255.0)
            arr = u8.cpu().numpy()
            for i, r in enumerate(rows):
                futs.append(save_pool.submit(
                    lambda a, p: Image.fromarray(a).save(p, format="PNG"),
                    arr[i].transpose(1, 2, 0).copy(), img_dir / r["gen_name"]))
            out_rows, _ = engine.batch_metrics(gen01, rows, Timer())
            rows_all.extend(out_rows)
        if (b // 8) % 5 == 0:
            print(f"[final:{cell}] {min(b+8, len(src_paths))}/{len(src_paths)} sources "
                  f"({time.perf_counter()-t_start:.0f}s)", flush=True)
    for f in futs:
        f.result()
    save_pool.shutdown(wait=True)

    write_csv(out_dir / "metrics.csv", rows_all, METRIC_HEADER)
    write_csv(out_dir / "dino_metrics.csv", rows_all, DINO_HEADER)
    manifest = {
        "cell": cell, "route": "C", "ckpt": str(cfg.get("ckpt", args.ckpt)), "tau0": cfg["tau0"],
        "coarse_guidance": cfg["guide"], "seed": seed, "num_images": len(rows_all),
        "wall_sec_render": round(time.perf_counter() - t_start, 1),
        "trt": trt.summary(),
    }
    (out_dir / "render_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    del engine, vae, trt
    torch.cuda.empty_cache()

    # v4_eval750 needs the exact frozen V4 render-root layout; provide a
    # symlink (or copy) so the historical evaluator can consume Route C PNGs
    # without duplicating 750 images.
    legacy_root = paths.asset_root() / "pseudo_gt_24h" / "eval_render"
    legacy_root.mkdir(parents=True, exist_ok=True)
    legacy_cell = legacy_root / cell
    if not legacy_cell.exists():
        try:
            legacy_cell.symlink_to(out_dir, target_is_directory=True)
        except OSError:
            # Windows/WSL drvfs without symlink privilege: hard-link the PNGs.
            legacy_cell.mkdir(parents=True, exist_ok=True)
            (legacy_cell / "images").mkdir(exist_ok=True)
            for src_file in (img_dir.glob("*.png")):
                dst_file = legacy_cell / "images" / src_file.name
                if not dst_file.exists():
                    try:
                        os.link(src_file, dst_file)
                    except OSError:
                        shutil.copy2(src_file, dst_file)
            for fn in ("metrics.csv", "dino_metrics.csv"):
                shutil.copy2(out_dir / fn, legacy_cell / fn)

    import subprocess
    cmd = [sys.executable, str(REPO_ROOT / "tools" / "pseudo_gt" / "v4_eval750.py"),
           "--cell", cell, "--render-root", str(RENDER_ROOT), "--max-src", "30"]
    print(f"[final] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
    return json.loads((out_dir / "v4_eval750.json").read_text(encoding="utf-8"))


def stage_final(args):
    device = "cuda"
    RC.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((RC / "best_config.json").read_text(encoding="utf-8")) if args.config == "best" else json.loads(args.config)
    ckpt_path = Path(cfg.get("ckpt", args.ckpt))
    model = load_flow(ckpt_path, device)
    results = {}
    for seed, tag in ((42, "s42"), (43, "s43")):
        cell = f"routeC_t{cfg['tau0']:.2f}_g{int(cfg['guide'])}_{tag}"
        results[tag] = render_cell(args, model, cfg, cell, seed, device)
        print(f"[final] {tag} cross600 CSD_dT={results[tag]['cross600']['CSD_dT']:+.5f} "
              f"DINO_C={results[tag]['cross600']['DINO_C']:.5f} "
              f"LPIPS={results[tag]['cross600']['LPIPS']:.5f}", flush=True)

    # two-seed mean (stochastic route must average, not cherry-pick)
    def mean2(key, group):
        return float(np.mean([results[t][group][key] for t in results]))
    summary = {
        "route": "C", "ckpt": str(ckpt_path), "best_config": cfg,
        "per_seed": {t: {g: {k: results[t][g][k] for k in
                            ("n", "CSD_dT", "CSD_dS", "CSD_dM", "DINO_S", "DINO_C", "CLIP_S", "LPIPS")}
                         for g in ("cross600", "same150", "all750")} for t in results},
        "two_seed_mean_cross600": {
            k: mean2(k, "cross600") for k in ("CSD_dT", "CSD_dS", "CSD_dM", "DINO_S", "DINO_C", "CLIP_S", "LPIPS")
        },
        "two_seed_mean_same150": {
            k: mean2(k, "same150") for k in ("CSD_dT", "CSD_dS", "CSD_dM", "DINO_C", "LPIPS")
        },
        "seed_gap_cross600_dT": abs(results["s42"]["cross600"]["CSD_dT"] - results["s43"]["cross600"]["CSD_dT"]),
    }

    # success criteria (plan §10), with the frozen 512px A0 anchor from V4
    anchor = {"CSD_dT": 0.020284, "DINO_C": 0.878826, "LPIPS": 0.265337}  # base_ep10 600cross
    c = summary["two_seed_mean_cross600"]
    passed = {
        "csd_dT_challenge_0.04": bool(c["CSD_dT"] >= 0.04),
        "csd_dT_beats_old_WEAVE": bool(c["CSD_dT"] > anchor["CSD_dT"]),
        "dino_c_within_0.01": bool(c["DINO_C"] >= anchor["DINO_C"] - 0.01),
        "lpips_within_0.01": bool(c["LPIPS"] <= anchor["LPIPS"] + 0.01),
        "same_style_no_extra_damage": bool(summary["two_seed_mean_same150"]["CSD_dM"] >= -0.01),
    }
    verdict = "B_generative_prior" if (passed["csd_dT_beats_old_WEAVE"] and
                                       passed["dino_c_within_0.01"] and passed["lpips_within_0.01"]) else "C_all_models_failed"
    summary["anchor_600cross_base_ep10"] = anchor
    summary["success_criteria"] = passed
    summary["final_verdict"] = verdict
    summary["answers"] = {
        "VAE_latent_can_express_strong_style_transfer": "No: P1 O1/O2/O3 + B1 encoder Oracle all dT<+0.01",
        "original_supervision_wrong": "Yes: paired pseudo-GT endpoints replaced by real distribution flow; see p1/b1 reports",
        "direct_operator_vs_ODE": "Not entered: Oracle failed, so M1/M2 branch was skipped by the decision tree",
        "amortisation_bottleneck": "Not the primary cause: per-image Oracle itself could not reach target style",
        "need_generative_prior_randomness": f"Route C tested; two-seed mean cross600 CSD_dT={c['CSD_dT']:+.5f}",
        "model_to_continue": verdict,
    }
    out = RC / "final_report.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"two_seed_mean_cross600": summary["two_seed_mean_cross600"],
                      "success_criteria": passed, "final_verdict": verdict}, indent=2), flush=True)
    print(f"[final] saved {out}", flush=True)


# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="stage", required=True)

    p = sub.add_parser("train")
    p.add_argument("--out", default=str(RC / "flow_full.pt"))
    p.add_argument("--init", default="")
    p.add_argument("--steps", type=int, default=2500)
    p.add_argument("--microbatch", type=int, default=128)
    p.add_argument("--accum", type=int, default=2)
    p.add_argument("--base", type=int, default=128)
    p.add_argument("--n-blocks", type=int, default=6)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--noise-mode", choices=["all", "lowmid"], default="all")
    p.set_defaults(func=stage_train)

    p = sub.add_parser("check")
    p.add_argument("--ckpt", default=str(RC / "flow_full.pt"))
    p.add_argument("--out", default=str(RC / "noise_check.json"))
    p.add_argument("--n-per-style", type=int, default=4)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--tau0", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=123)
    p.set_defaults(func=stage_check)

    p = sub.add_parser("quick")
    p.add_argument("--ckpt", default=str(RC / "flow_full.pt"))
    p.add_argument("--out", default=str(RC / "quick_report.json"))
    p.add_argument("--steps", type=int, default=8)
    p.set_defaults(func=stage_quick)

    p = sub.add_parser("final")
    p.add_argument("--ckpt", default=str(RC / "flow_full.pt"))
    p.add_argument("--config", default="best")
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=stage_final)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
