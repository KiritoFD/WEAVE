"""Head-to-Head Training Duel on Frozen pool_V22_clean.pt (Zero New Teacher Data):

Route A (Bridge4_V22_paired):
    4-Step Paired Latent Bridge using the 11.5M U-Net + S10-BF16 ConvNeXt/VGG Patch-SWD
    + TTO2 Gray-VGG relu4_2 self-sim & P_Y LCN structural shield (0 CFG, exact lambda=0 identity).

Route B (Adv1NFE_V22_projgan):
    Warm-start 1-NFE from Final_5k_seed0_final.pt + progressive high-lambda L1 relaxation
    + Referee-free Multi-Scale Projected GAN Discriminator (frozen VGG-16 + ConvNeXt-Tiny)
    + ConvNeXt/VGG Patch-SWD + TTO2 structural shield.
"""
from __future__ import annotations

import copy
import gc
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw

from exp.v18 import tasks_and_sources
from exp.v19.aniso_teacher import latent_lcn_cost
from exp.v20.submode_patchswd import MultiScalePatchSWDExtractor, sliced_wasserstein_1d
from exp.v27_mainline.rollout_refine import FrozenVGG16Rollout, gray_self_sim_loss
from exp.v27_single.bare_q32 import load_cache
from exp.v27_single.unified.train import conditioning_stats
from exp.v28_unified import (
    OUT,
    ROOT,
    SEED_TRAIN,
    RawWriterV28,
    assert_vram_budget,
    atomic_save,
    run_measure_v28,
)
from exp.v28_unified.student_models import PixelBridgeV28, build_student_model
from exp.v28_unified.train_student import (
    MODE_PATTERN_2TIER,
    _compute_targets,
    _sample_strengths_and_targets,
    run_final_eval,
)
from src.eval.metrics import paired_cluster_ci
from src.utils import decode_s10, load_s10_decoder
from src.utils.protocol import EVAL_SEED
from src.utils.runtime import bf16_autocast, read_json, sanitize_cuda_env, save_json, seed_all, utc
from src.utils.structure import dwt2_haar


def relaxed_l1_weight(levels: torch.Tensor, min_weight: float = 0.12) -> torch.Tensor:
    """Keep L1=1.0 for lambda<=0.40 and smoothly relax to min_weight for lambda>=0.75."""
    frac = ((0.75 - levels) / 0.35).clamp(0.0, 1.0)
    return min_weight + (1.0 - min_weight) * frac


class _SpectralProjHead(nn.Module):
    """Lightweight spectral-normed spatial+class-projection discriminator on a 64-ch projected map."""

    def __init__(self, in_ch: int = 64, width: int = 96, n_styles: int = 5):
        super().__init__()
        sn = torch.nn.utils.spectral_norm
        self.net = nn.Sequential(
            sn(nn.Conv2d(in_ch, width, 3, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            sn(nn.Conv2d(width, width, 3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.rf_conv = sn(nn.Conv2d(width, 1, 3, padding=1))
        self.embed = sn(nn.Embedding(n_styles, width))

    def forward(self, x: torch.Tensor, tid: torch.Tensor) -> torch.Tensor:
        h = self.net(x)
        rf = self.rf_conv(h).mean(dim=(1, 2, 3))
        proj = (self.embed(tid) * h.mean(dim=(2, 3))).sum(dim=1)
        return rf + proj


class ProjectedMultiScaleD(nn.Module):
    """Multi-scale Projected Discriminator over frozen VGG-16 + ConvNeXt-Tiny features (referee-free)."""

    _SPECS = (
        ("vgg_s2", 128, 32),
        ("vgg_s3", 256, 16),
        ("vgg_s4", 512, 16),
        ("cnx_s1", 96, 32),
        ("cnx_s2", 192, 16),
        ("cnx_s3", 384, 16),
    )

    def __init__(self, seed: int = SEED_TRAIN + 303):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.heads = nn.ModuleDict()
        for name, c_dim, _ in self._SPECS:
            q, _ = torch.linalg.qr(torch.randn(c_dim, 64, generator=g), mode="reduced")
            self.register_buffer(f"proj_{name}", q)
            self.heads[name] = _SpectralProjHead(64, 96, n_styles=5)

    def forward(self, feats: dict[str, torch.Tensor], tid: torch.Tensor) -> torch.Tensor:
        scores = []
        for name, _, size in self._SPECS:
            f = F.adaptive_avg_pool2d(feats[name].float(), size)
            proj = getattr(self, f"proj_{name}")
            p = torch.einsum("bchw,cd->bdhw", f, proj)
            scores.append(self.heads[name](p, tid))
        return torch.stack(scores, dim=0).mean(dim=0)


class _DuelPerceptualCritic:
    """Shared referee-free S10-BF16 pixel loss: ConvNeXt+VGG Patch-SWD + Oklab moments + TTO2 struct shield."""

    def __init__(self, v18_luma: dict, device: str = "cuda"):
        self.device = device
        self.pixel = PixelBridgeV28(device=device)
        self.vgg_ro = FrozenVGG16Rollout().to(device).eval().requires_grad_(False)
        self.swd_ext = MultiScalePatchSWDExtractor(n_slices=64, seed=SEED_TRAIN + 404).to(device).eval().requires_grad_(False)
        self.w_luma = v18_luma["w"].to(device).float()
        self.b_luma = v18_luma["b"].to(device).float()

    def set_moment_scale(self, zs_samples: torch.Tensor) -> None:
        with torch.no_grad():
            xx = torch.cat([
                self.pixel.encode(zs_samples[i:i + 4].to(self.device).float(), False)[0]
                for i in range(0, len(zs_samples), 4)
            ])
            self.pixel.scale = xx.std(0, unbiased=False).clamp_min(0.03)

    def extract_d_feats(self, rgb01: torch.Tensor) -> dict[str, torch.Tensor]:
        with bf16_autocast(self.device):
            _, f2, f3, f4 = self.vgg_ro.forward_style(rgb01)
            x256 = F.interpolate(rgb01.float(), size=(256, 256), mode="bilinear", align_corners=False)
            xn = (x256 - self.swd_ext.im_mean) / self.swd_ext.im_std
            c1 = self.swd_ext.s1(xn)
            c2 = self.swd_ext.s2(c1)
            c3 = self.swd_ext.s3(c2)
        return {"vgg_s2": f2, "vgg_s3": f3, "vgg_s4": f4, "cnx_s1": c1, "cnx_s2": c2, "cnx_s3": c3}

    def compute_pixel_and_struct(
        self,
        z_pred: torch.Tensor,
        z_tgt: torch.Tensor,
        z_src: torch.Tensor,
        same_mask: torch.Tensor,
        levels: torch.Tensor,
        proj: tuple[torch.Tensor, ...],
        *,
        sample_weights: torch.Tensor | None = None,
        w_mom: float = 0.25,
        w_vgg_swd: float = 0.22,
        w_cnx_swd: float = 0.28,
        w_gss: float = 1.4,
        w_lcn: float = 1.6,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (total_pix_loss, rgb_pred, rgb_tgt_detached, struct_diag)."""
        B = len(z_pred)
        rgb_pred = decode_s10(
            self.pixel.decoder, z_pred, dtype=torch.bfloat16, batch=B, enable_grad=True, output_cpu=False
        ).clamp(0.0, 1.0)
        self.pixel.layers.clear()
        with bf16_autocast(self.device):
            cur_f = self.pixel.vgg(rgb_pred.float(), multi_scale=False).float()
        cur_t = tuple(F.adaptive_avg_pool2d(x, 16).float().flatten(2).transpose(1, 2) for x in self.pixel.layers)
        self.pixel.layers.clear()

        with bf16_autocast(self.device):
            cnx_cur = self.swd_ext.extract_tokens(rgb_pred)
            r42_cur = self.vgg_ro.forward_gray_r42(rgb_pred).float()

        with torch.no_grad():
            rgb_tgt = decode_s10(
                self.pixel.decoder, z_tgt, dtype=torch.bfloat16, batch=B, enable_grad=False, output_cpu=False
            ).clamp(0.0, 1.0)
            rgb_src = decode_s10(
                self.pixel.decoder, z_src, dtype=torch.bfloat16, batch=B, enable_grad=False, output_cpu=False
            ).clamp(0.0, 1.0)
            self.pixel.layers.clear()
            with bf16_autocast(self.device):
                tgt_f = self.pixel.vgg(rgb_tgt.float(), multi_scale=False).float()
            tgt_t = tuple(F.adaptive_avg_pool2d(x, 16).float().flatten(2).transpose(1, 2) for x in self.pixel.layers)
            self.pixel.layers.clear()
            with bf16_autocast(self.device):
                cnx_tgt = self.swd_ext.extract_tokens(rgb_tgt)
                r42_src = self.vgg_ro.forward_gray_r42(rgb_src).float()

        mom_ps = F.smooth_l1_loss((cur_f - tgt_f) / self.pixel.scale, torch.zeros_like(cur_f), beta=0.20, reduction="none").mean(dim=1)
        vgg_swd_ps = sum(
            ((a @ p).sort(dim=1).values - (b @ p).sort(dim=1).values).abs().mean(dim=(1, 2))
            for a, b, p in zip(cur_t, tgt_t, proj)
        ) / len(proj)
        cnx_swd_ps = (
            1.1 * sliced_wasserstein_1d(cnx_cur["rgb"].float(), cnx_tgt["rgb"].float(), self.swd_ext.p_rgb)
            + 0.35 * sliced_wasserstein_1d(cnx_cur["s1"].float(), cnx_tgt["s1"].float(), self.swd_ext.p_s1)
            + 0.45 * sliced_wasserstein_1d(cnx_cur["s2"].float(), cnx_tgt["s2"].float(), self.swd_ext.p_s2)
            + 0.55 * sliced_wasserstein_1d(cnx_cur["s3"].float(), cnx_tgt["s3"].float(), self.swd_ext.p_s3)
        )

        # TTO2 structural lock strictly on cross-style tasks (weighted by active strength)
        cross_w = (~same_mask).float() * (levels > 0.15).float()
        a16 = F.normalize(F.adaptive_avg_pool2d(r42_cur, 16).flatten(2), dim=1)
        b16 = F.normalize(F.adaptive_avg_pool2d(r42_src, 16).flatten(2), dim=1)
        gss_ps = F.smooth_l1_loss(
            torch.bmm(a16.transpose(1, 2), a16), torch.bmm(b16.transpose(1, 2), b16), beta=0.05, reduction="none"
        ).mean(dim=(1, 2))
        lcn_ps = latent_lcn_cost(z_pred, z_src, self.w_luma, self.b_luma)

        style_ps = w_mom * mom_ps + w_vgg_swd * vgg_swd_ps + w_cnx_swd * cnx_swd_ps
        if sample_weights is not None:
            style_ps = style_ps * sample_weights.to(style_ps.device, dtype=style_ps.dtype)
        struct_ps = cross_w * (w_gss * gss_ps + w_lcn * lcn_ps)
        return (style_ps + struct_ps).mean(), rgb_pred, rgb_tgt, (gss_ps * cross_w).mean().detach()


def _sample_duel_batch(
    pairs: dict,
    same_ids: torch.Tensor,
    cross_ids: torch.Tensor,
    step: int,
    eff_batch: int,
    pixel_micro: int,
    gen: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """20% identity + 80% continuous 2tier cross sampling (zero l060 dead zone)."""
    prev_same, next_same = ((step - 1) * eff_batch) // 5, (step * eff_batch) // 5
    n_same = next_same - prev_same
    n_cross = eff_batch - n_same

    si = same_ids[torch.randint(len(same_ids), (n_same,), generator=gen)]
    ci = cross_ids[torch.randint(len(cross_ids), (n_cross,), generator=gen)]
    prev_cross = (step - 1) * eff_batch - prev_same
    modes_b = MODE_PATTERN_2TIER[torch.arange(prev_cross, prev_cross + n_cross) % len(MODE_PATTERN_2TIER)]
    modes_b = modes_b[torch.randperm(n_cross, generator=gen)]
    cross_l, cross_tc = _sample_strengths_and_targets(modes_b, n_cross, gen, "2tier")

    same_l = torch.rand(n_same, generator=gen)
    same_tc = torch.zeros(n_same, dtype=torch.long)

    all_idx = torch.cat((ci, si))
    all_l = torch.cat((cross_l, same_l))
    all_tc = torch.cat((cross_tc, same_tc))
    is_cross = torch.cat((torch.ones(n_cross, dtype=torch.bool), torch.zeros(n_same, dtype=torch.bool)))

    perm = torch.randperm(eff_batch, generator=gen)
    pix_cands = perm[is_cross[perm]][:pixel_micro]
    rest = perm[~torch.isin(perm, pix_cands)]
    order = torch.cat((pix_cands, rest))
    return all_idx[order], all_l[order], all_tc[order]


def train_bridge4_route(
    arm_name: str = "Bridge4_V22_paired",
    *,
    updates: int = 3200,
    eff_batch: int = 96,
    micro: int = 32,
    pixel_micro: int = 4,
    lr: float = 9.0e-4,
    smoke_steps: int = 0,
    device: str = "cuda",
) -> dict:
    """Route A: 4-Step Paired Latent Bridge trained cleanly on pool_V22_clean.pt."""
    sanitize_cuda_env()
    seed_all(SEED_TRAIN)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.cuda.reset_peak_memory_stats(device)

    run_dir = OUT / "students" / arm_name
    run_dir.mkdir(parents=True, exist_ok=True)
    final_ckpt = run_dir / f"{arm_name}_final.pt"
    if final_ckpt.exists() and smoke_steps == 0 and (run_dir / "results.json").exists():
        print(f"[{arm_name}] Reusing completed results.", flush=True)
        return read_json(run_dir / "results.json")

    v18 = load_cache("runs/v18/cache/ccf_pool_cache.pt")
    tr = v18["train"]
    ts = load_cache("runs/v18/cache/test750_struct_cache.pt")
    c21 = load_cache("runs/v21/cache/v21_stiefel_ref_cache.pt")
    foundation = load_cache("runs/v19/cache/v19_foundation.pt")
    priors = {k: foundation["priors"][k].to(device).float() for k in ("P_Y", "P_C", "mu_c", "std_c")}

    critic = _DuelPerceptualCritic(v18["luma_proj"], device=device)
    raw_stats = conditioning_stats(tr, foundation, critic.pixel)
    del foundation

    pairs = torch.load(OUT / "pools/pool_V22_clean.pt", map_location="cpu", weights_only=False)
    same_ids = pairs["same"].nonzero().flatten()
    cross_ids = (~pairs["same"]).nonzero().flatten()
    critic.set_moment_scale(pairs["hiS"][cross_ids[::125][:48]])

    model = build_student_model("bridge4", raw_stats.shape[-1], device=device)
    warm_p = OUT / "students/Final_5k_seed0/Final_5k_seed0_final.pt"
    if warm_p.exists():
        warm_ck = torch.load(warm_p, map_location="cpu", weights_only=False)
        model.load_state_dict(warm_ck["ema"], strict=False)
        print(f"[{arm_name}] Warm-initialized backbone from {warm_p.name}", flush=True)
    num_params = sum(p.numel() for p in model.parameters())
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=2.0e-4)
    gen = torch.Generator().manual_seed(SEED_TRAIN + 11)

    target_steps = smoke_steps if smoke_steps > 0 else updates
    euler_knots = torch.tensor([0.0, 0.25, 0.50, 0.75], device=device)
    t0 = time.monotonic()

    for step in range(1, target_steps + 1):
        frac = max(0, step - 120) / max(1, updates - 120)
        cur_lr = lr * step / 120.0 if step <= 120 else 6e-5 + 0.5 * (lr - 6e-5) * (1.0 + math.cos(math.pi * frac))
        for pg in opt.param_groups:
            pg["lr"] = cur_lr
        opt.zero_grad(set_to_none=True)

        all_idx, all_l, all_tc = _sample_duel_batch(pairs, same_ids, cross_ids, step, eff_batch, pixel_micro, gen)
        pg_s = torch.Generator(device=device).manual_seed(SEED_TRAIN + 5000 + step)
        proj = tuple(torch.linalg.qr(torch.randn(c, 64, device=device, generator=pg_s), mode="reduced").Q for c in (64, 128, 256))

        def _bridge_step_slice(sl: slice):
            bi = all_idx[sl]
            b_l = all_l[sl].to(device)
            b_tc = all_tc[sl].to(device)
            si = pairs["src"][bi]
            zs = tr["z"][si].to(device).float()
            c64b = tr["c_64"][si].to(device).float()
            q32b = c21["train"]["q32"][si].to(device).float()
            tidb = pairs["tid"][bi].to(device)
            tgt = _compute_targets(zs, pairs["hiC"][bi].to(device).float(), None, pairs["hiS"][bi].to(device).float(), b_l, b_tc, "2tier")
            same_m = pairs["same"][bi].to(device)
            tgt = torch.where(same_m[:, None, None, None], zs, tgt)

            eps = torch.randn_like(zs)
            z_start, _ = model.bridge_start(zs, tidb, b_l, priors, noise=eps)
            v_tgt = tgt - z_start

            # 50% exact Euler grid knots {0, .25, .50, .75}, 50% continuous U(0, 0.90)
            n_b = len(zs)
            use_knot = torch.rand(n_b, device=device) < 0.5
            t_knot = euler_knots[torch.randint(0, 4, (n_b,), device=device)]
            t_cont = 0.90 * torch.rand(n_b, device=device)
            t_b = torch.where(use_knot, t_knot, t_cont)

            zt = (1.0 - t_b[:, None, None, None]) * z_start + t_b[:, None, None, None] * tgt
            v_pred = model.predict_velocity(zt, zs, c64b, q32b, tidb, b_l, t_b, raw_stats, priors)
            z1_pred = zt + (1.0 - t_b[:, None, None, None]) * v_pred

            # Relax early-t velocity contraction on high lambda so multi-step denoising keeps crisp variance
            v_err = F.smooth_l1_loss(v_pred, v_tgt, beta=0.06, reduction="none").mean(dim=(1, 2, 3))
            v_damp = torch.where((t_b < 0.20) & (b_l > 0.60), torch.full_like(b_l, 0.55), torch.ones_like(b_l))
            fp, ft = dwt2_haar(z1_pred), dwt2_haar(tgt)
            bands = sum(w * F.l1_loss(a, b) for w, a, b in zip((1.0, 1.0, 0.5), fp[1:], ft[1:]))
            l_lat = (v_err * v_damp).mean() + 0.35 * bands
            return l_lat, z1_pred, tgt, zs, same_m, b_l, t_b

        tot_lat, tot_pix, tot_gss = 0.0, 0.0, 0.0
        for st in range(0, eff_batch, micro):
            stop = min(eff_batch, st + micro)
            l_lat, z1_p, tgt, zs, same_m, lv, t_b = _bridge_step_slice(slice(st, stop))
            w_slice = (stop - st) / float(eff_batch)
            if st == 0:
                sl_pix = slice(0, pixel_micro)
                pix_sw = (0.35 + 0.65 * t_b[sl_pix]) * torch.where(lv[sl_pix] > 0.6, torch.full_like(lv[sl_pix], 1.6), torch.ones_like(lv[sl_pix]))
                l_pix, _, _, gss_d = critic.compute_pixel_and_struct(
                    z1_p[sl_pix], tgt[sl_pix], zs[sl_pix], same_m[sl_pix], lv[sl_pix], proj,
                    sample_weights=pix_sw, w_gss=1.2, w_lcn=1.4,
                )
                l_pix = 0.52 * l_pix
                (w_slice * l_lat + l_pix).backward()
                tot_pix += float(l_pix.item())
                tot_gss += float(gss_d.item())
            else:
                (w_slice * l_lat).backward()
            tot_lat += float((w_slice * l_lat).item())

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5, error_if_nonfinite=True)
        opt.step()
        with torch.no_grad():
            torch._foreach_lerp_(list(ema.parameters()), [p.detach() for p in model.parameters()], 0.006)

        if step == 1 or step % 100 == 0 or step == target_steps:
            dt = (time.monotonic() - t0) / step
            print(f"[{arm_name}] {step:4d}/{target_steps} lat={tot_lat:.4f} pix={tot_pix:.4f} gss={tot_gss:.4f} grad={float(grad_norm):.3f} ({dt:.3f}s/st)", flush=True)

        if smoke_steps == 0 and (step % 500 == 0 or step == updates):
            atomic_save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "step": step, "updates": updates}, run_dir / "resume.pt")

    vram_gb = assert_vram_budget(device=device, tag=arm_name)
    elapsed = time.monotonic() - t0
    del critic, model, opt
    gc.collect()
    torch.cuda.empty_cache()

    if smoke_steps > 0:
        # Also verify bit-exact lambda=0 4-step rollout identity!
        zs_t = tr["z"][:2].to(device).float()
        id_out = ema(zs_t, tr["c_64"][:2].to(device).float(), c21["train"]["q32"][:2].to(device).float(),
                     torch.tensor([0, 4], device=device), torch.zeros(2, device=device), raw_stats, priors, noise=torch.randn_like(zs_t))
        id_err = float((id_out - zs_t).abs().max().item())
        del ema
        torch.cuda.empty_cache()
        return {"arm": arm_name, "params": num_params, "sec_per_step": round(elapsed / smoke_steps, 4),
                "est_min": round((elapsed / smoke_steps) * updates / 60.0, 1), "peak_vram_gb": round(vram_gb, 3), "lambda_zero_max_abs": id_err}

    atomic_save({"ema": ema.state_dict(), "raw_stats": raw_stats, "priors": priors, "step": updates, "params": num_params, "arch": "bridge4"}, final_ckpt)
    rep = run_final_eval(ema, "bridge4", raw_stats, priors, ts, c21["test750"]["q32"], updates,
                         run_dir=run_dir, tag=arm_name, checkpoint_name=final_ckpt.name, device=device)
    rep["peak_vram_gb"] = round(vram_gb, 3)
    rep["train_elapsed_min"] = round(elapsed / 60.0, 2)
    save_json(run_dir / "results.json", rep)
    del ema
    gc.collect()
    torch.cuda.empty_cache()
    return rep


def train_adv1nfe_route(
    arm_name: str = "Adv1NFE_V22_projgan",
    *,
    updates: int = 2600,
    eff_batch: int = 96,
    micro: int = 32,
    pixel_micro: int = 4,
    lr_g: float = 1.4e-4,
    lr_d: float = 2.2e-4,
    smoke_steps: int = 0,
    device: str = "cuda",
) -> dict:
    """Route B: Warm-start 1-NFE + High-Lambda L1 Relaxation + Multi-Scale Projected GAN + TTO2 Struct Lock."""
    sanitize_cuda_env()
    seed_all(SEED_TRAIN + 22)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.cuda.reset_peak_memory_stats(device)

    run_dir = OUT / "students" / arm_name
    run_dir.mkdir(parents=True, exist_ok=True)
    final_ckpt = run_dir / f"{arm_name}_final.pt"
    if final_ckpt.exists() and smoke_steps == 0 and (run_dir / "results.json").exists():
        print(f"[{arm_name}] Reusing completed results.", flush=True)
        return read_json(run_dir / "results.json")

    v18 = load_cache("runs/v18/cache/ccf_pool_cache.pt")
    tr = v18["train"]
    ts = load_cache("runs/v18/cache/test750_struct_cache.pt")
    c21 = load_cache("runs/v21/cache/v21_stiefel_ref_cache.pt")
    warm_ck = torch.load(OUT / "students/Final_5k_seed0/Final_5k_seed0_final.pt", map_location="cpu", weights_only=False)
    raw_stats = warm_ck["raw_stats"].to(device)
    priors = {k: warm_ck["priors"][k].to(device).float() for k in warm_ck["priors"]}

    critic = _DuelPerceptualCritic(v18["luma_proj"], device=device)
    pairs = torch.load(OUT / "pools/pool_V22_clean.pt", map_location="cpu", weights_only=False)
    same_ids = pairs["same"].nonzero().flatten()
    cross_ids = (~pairs["same"]).nonzero().flatten()
    critic.set_moment_scale(pairs["hiS"][cross_ids[::125][:48]])

    model = build_student_model("unet", raw_stats.shape[-1], device=device)
    model.load_state_dict(warm_ck["ema"])
    num_params = sum(p.numel() for p in model.parameters())
    ema = copy.deepcopy(model).eval().requires_grad_(False)

    disc = ProjectedMultiScaleD(seed=SEED_TRAIN + 303).to(device)
    opt_g = torch.optim.AdamW(model.parameters(), lr=lr_g, betas=(0.5, 0.99), weight_decay=1.0e-4)
    opt_d = torch.optim.AdamW(disc.parameters(), lr=lr_d, betas=(0.5, 0.99), weight_decay=1.0e-4)
    gen = torch.Generator().manual_seed(SEED_TRAIN + 33)

    target_steps = smoke_steps if smoke_steps > 0 else updates
    t0 = time.monotonic()

    for step in range(1, target_steps + 1):
        frac = float(step) / float(max(1, updates))
        lr_scale = 0.25 + 0.75 * 0.5 * (1.0 + math.cos(math.pi * frac))
        for pg in opt_g.param_groups:
            pg["lr"] = lr_g * lr_scale
        for pg in opt_d.param_groups:
            pg["lr"] = lr_d * lr_scale

        all_idx, all_l, all_tc = _sample_duel_batch(pairs, same_ids, cross_ids, step, eff_batch, pixel_micro, gen)
        pg_s = torch.Generator(device=device).manual_seed(SEED_TRAIN + 7000 + step)
        proj = tuple(torch.linalg.qr(torch.randn(c, 64, device=device, generator=pg_s), mode="reduced").Q for c in (64, 128, 256))

        # 1) Generator update: relaxed latent L1 + pixel Patch-SWD + TTO2 struct shield + Hinge GAN adv
        opt_g.zero_grad(set_to_none=True)
        tot_lat, tot_pix, tot_adv, tot_gss = 0.0, 0.0, 0.0, 0.0
        cached_real_rgb, cached_fake_rgb, cached_tid, cached_ramp = None, None, None, None

        for st in range(0, eff_batch, micro):
            stop = min(eff_batch, st + micro)
            bi = all_idx[st:stop]
            b_l = all_l[st:stop].to(device)
            b_tc = all_tc[st:stop].to(device)
            si = pairs["src"][bi]
            zs = tr["z"][si].to(device).float()
            c64b = tr["c_64"][si].to(device).float()
            q32b = c21["train"]["q32"][si].to(device).float()
            tidb = pairs["tid"][bi].to(device)
            same_m = pairs["same"][bi].to(device)
            tgt = _compute_targets(zs, pairs["hiC"][bi].to(device).float(), None, pairs["hiS"][bi].to(device).float(), b_l, b_tc, "2tier")
            tgt = torch.where(same_m[:, None, None, None], zs, tgt)

            pred = model(zs, c64b, q32b, tidb, b_l, raw_stats, priors, noise=torch.randn_like(zs))
            l1_ps = F.smooth_l1_loss(pred, tgt, beta=0.06, reduction="none").mean(dim=(1, 2, 3))
            l1_w = torch.where(same_m, torch.ones_like(b_l), relaxed_l1_weight(b_l, min_weight=0.12))
            fp, ft = dwt2_haar(pred), dwt2_haar(tgt)
            bands = sum(w * F.l1_loss(a, b) for w, a, b in zip((1.0, 1.0, 0.5), fp[1:], ft[1:]))
            l_lat = (l1_ps * l1_w).mean() + 0.20 * bands
            w_slice = (stop - st) / float(eff_batch)

            if st == 0:
                sl_p = slice(0, pixel_micro)
                sw = torch.where(b_l[sl_p] > 0.6, torch.full_like(b_l[sl_p], 1.8), torch.ones_like(b_l[sl_p]))
                l_pix, rgb_p, rgb_r, gss_d = critic.compute_pixel_and_struct(
                    pred[sl_p], tgt[sl_p], zs[sl_p], same_m[sl_p], b_l[sl_p], proj,
                    sample_weights=sw, w_gss=1.6, w_lcn=1.8,
                )
                ramp = ((b_l[sl_p] - 0.35) / 0.40).clamp(0.0, 1.0) * (~same_m[sl_p]).float()
                for p in disc.parameters():
                    p.requires_grad_(False)
                d_fake = disc(critic.extract_d_feats(rgb_p), tidb[sl_p])
                for p in disc.parameters():
                    p.requires_grad_(True)
                l_adv = -(ramp * d_fake).mean()
                adv_w = 0.06 * min(1.0, step / 250.0)
                (w_slice * l_lat + 0.38 * l_pix + adv_w * l_adv).backward()
                tot_pix = float((0.32 * l_pix).item())
                tot_adv = float(l_adv.item())
                tot_gss = float(gss_d.item())
                cached_real_rgb = rgb_r.detach()
                cached_fake_rgb = rgb_p.detach()
                cached_tid = tidb[sl_p].detach()
                cached_ramp = ramp.detach()
            else:
                (w_slice * l_lat).backward()
            tot_lat += float((w_slice * l_lat).item())

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5, error_if_nonfinite=True)
        opt_g.step()
        with torch.no_grad():
            torch._foreach_lerp_(list(ema.parameters()), [p.detach() for p in model.parameters()], 0.006)

        # 2) Discriminator hinge update on cached real vs detached fake
        opt_d.zero_grad(set_to_none=True)
        assert cached_real_rgb is not None and cached_fake_rgb is not None
        if float(cached_ramp.sum().item()) > 0.0:
            d_real = disc(critic.extract_d_feats(cached_real_rgb), cached_tid)
            d_fk = disc(critic.extract_d_feats(cached_fake_rgb), cached_tid)
            l_d = (cached_ramp * (F.relu(1.0 - d_real) + F.relu(1.0 + d_fk))).mean()
            l_d.backward()
            torch.nn.utils.clip_grad_norm_(disc.parameters(), 2.0)
            opt_d.step()
            d_val = float(l_d.item())
        else:
            d_val = 0.0

        if step == 1 or step % 100 == 0 or step == target_steps:
            dt = (time.monotonic() - t0) / step
            print(f"[{arm_name}] {step:4d}/{target_steps} lat={tot_lat:.4f} pix={tot_pix:.4f} adv={tot_adv:.3f} D={d_val:.3f} gss={tot_gss:.4f} grad={float(grad_norm):.3f} ({dt:.3f}s/st)", flush=True)

        if smoke_steps == 0 and (step % 500 == 0 or step == updates):
            atomic_save({"model": model.state_dict(), "ema": ema.state_dict(), "disc": disc.state_dict(), "step": step, "updates": updates}, run_dir / "resume.pt")

    vram_gb = assert_vram_budget(device=device, tag=arm_name)
    elapsed = time.monotonic() - t0
    del critic, disc, model, opt_g, opt_d
    gc.collect()
    torch.cuda.empty_cache()

    if smoke_steps > 0:
        zs_t = tr["z"][:2].to(device).float()
        id_out = ema(zs_t, tr["c_64"][:2].to(device).float(), c21["train"]["q32"][:2].to(device).float(),
                     torch.tensor([0, 4], device=device), torch.zeros(2, device=device), raw_stats, priors, noise=torch.randn_like(zs_t))
        id_err = float((id_out - zs_t).abs().max().item())
        del ema
        torch.cuda.empty_cache()
        return {"arm": arm_name, "params": num_params, "sec_per_step": round(elapsed / smoke_steps, 4),
                "est_min": round((elapsed / smoke_steps) * updates / 60.0, 1), "peak_vram_gb": round(vram_gb, 3), "lambda_zero_max_abs": id_err}

    atomic_save({"ema": ema.state_dict(), "raw_stats": raw_stats, "priors": priors, "step": updates, "params": num_params, "arch": "unet"}, final_ckpt)
    rep = run_final_eval(ema, "unet", raw_stats, priors, ts, c21["test750"]["q32"], updates,
                         run_dir=run_dir, tag=arm_name, checkpoint_name=final_ckpt.name, device=device)
    rep["peak_vram_gb"] = round(vram_gb, 3)
    rep["train_elapsed_min"] = round(elapsed / 60.0, 2)
    save_json(run_dir / "results.json", rep)
    del ema
    gc.collect()
    torch.cuda.empty_cache()
    return rep


def _measure_duel_latency(ckpt_path: Path, arch: str, device: str = "cuda") -> dict:
    saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw_stats = saved["raw_stats"].to(device)
    priors = {k: v.to(device) for k, v in saved["priors"].items()}
    model = build_student_model(arch, raw_stats.shape[-1], device=device).eval()
    model.load_state_dict(saved["ema"])
    dec = load_s10_decoder(device)
    out = {}
    with torch.no_grad():
        for bs in (1, 16):
            zs = torch.randn(bs, 4, 64, 64, device=device)
            c64 = torch.randn(bs, 4, 64, 64, device=device)
            q32 = torch.randn(bs, 32, 16, 16, device=device)
            tid = torch.zeros(bs, dtype=torch.long, device=device)
            st = torch.ones(bs, device=device)
            eps = torch.randn_like(zs)
            for _ in range(6):
                _ = model(zs, c64, q32, tid, st, raw_stats, priors, noise=eps)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(25):
                z_out = model(zs, c64, q32, tid, st, raw_stats, priors, noise=eps)
            torch.cuda.synchronize()
            m_ms = (time.perf_counter() - t0) * 1000.0 / (25 * bs)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            for _ in range(25):
                z_out = model(zs, c64, q32, tid, st, raw_stats, priors, noise=eps)
                _ = decode_s10(dec, z_out, batch=bs, output_cpu=False)
            torch.cuda.synchronize()
            e_ms = (time.perf_counter() - t1) * 1000.0 / (25 * bs)
            out[f"b{bs}"] = {"model_ms_per_img": round(m_ms, 2), "e2e_s10_ms_per_img": round(e_ms, 2), "e2e_img_per_s": round(1000.0 / e_ms, 1)}
    del model, dec
    torch.cuda.empty_cache()
    return out


def _plot_duel_pareto(seed0_rep: dict, bridge_rep: dict, adv_rep: dict, out_dir: Path) -> None:
    from exp.v28_unified.phase3_4_5 import _frontier_indices, _literature_points

    lit = _literature_points()
    lit_f = _frontier_indices([(p["dino_c"], p["dino_s"]) for p in lit])
    lit_ft = _frontier_indices([(p["dino_c"], p["csd_top1"] * 100.0) for p in lit])
    hist = [("V20 D2", 0.605, 0.567, 60.5), ("V22 Peak1500", 0.578, 0.582, 76.8), ("StyleGallery", 0.358, 0.574, 92.67)]

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4), dpi=180)
    lvls = ("l025", "l040", "l060", "l075", "l100")
    curves = [
        ("Final_5k_seed0 (1-NFE)", seed0_rep["arms"], "#7f8c8d", "--o"),
        ("Bridge4 (4-Step Paired)", bridge_rep["arms"], "#1b6b50", "-D"),
        ("Adv1NFE (1-NFE ProjGAN)", adv_rep["arms"], "#d95f02", "-s"),
    ]
    for ax_i, (ax, yk, scale, yl) in enumerate([(axes[0], "dino_s", 1.0, "DINO-S ↑"), (axes[1], "csd_top1", 100.0, "CSD Top-1 (%) ↑")]):
        for p in lit:
            ax.scatter(p["dino_c"], p[yk] * scale, s=26, color="#666", alpha=0.35, edgecolors="none")
        fl = lit_f if ax_i == 0 else lit_ft
        arr = [(p["dino_c"], p[yk] * scale) for p in lit]
        ax.plot([arr[i][0] for i in fl], [arr[i][1] for i in fl], "--o", color="#555", lw=1.1, ms=3.5, alpha=0.6, label="External lit Pareto")
        for nm, dc, ds, tp in hist:
            yv = ds if ax_i == 0 else tp
            ax.scatter(dc, yv, marker="X", s=62, color="#b8860b", zorder=4)
            ax.annotate(nm, (dc, yv), xytext=(4, 3), textcoords="offset points", fontsize=7.8, color="#8a6500")
        for cname, arms, col, fmt in curves:
            xs = [arms[lv]["dino_c"] for lv in lvls]
            ys = [arms[lv][yk] * scale for lv in lvls]
            ax.plot(xs, ys, fmt, color=col, lw=2.0, ms=6.2, label=cname, zorder=6)
            for lv, xv, yv in zip(lvls, xs, ys):
                if lv in ("l040", "l075", "l100"):
                    ax.annotate(lv[1:], (xv, yv), xytext=(4, 3), textcoords="offset points", fontsize=7.5, color=col, fontweight="bold")
        if ax_i == 0:
            ax.axhline(0.565, ls=":", lw=1.0, color="#b2182b", alpha=0.7)
            ax.axvline(0.58, ls=":", lw=1.0, color="#b2182b", alpha=0.7)
        ax.set_xlabel("DINO-C ↑", fontsize=10.5)
        ax.set_ylabel(yl, fontsize=10.5)
        ax.grid(alpha=0.22)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylim(0.32, 0.62)
    axes[1].set_ylim(0, 102)
    axes[0].legend(frameon=False, fontsize=7.8, loc="lower left")
    fig.suptitle("V28 Training-Side Duel: 4-Step Paired Bridge vs 1-NFE Projected GAN (Frozen pool_V22_clean) · Cross600", fontsize=11.3, fontweight="bold")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(out_dir / f"pareto_v28_duel.{ext}", bbox_inches="tight")
    plt.close(fig)


def _export_duel_visual_strip(out_png: Path) -> None:
    tasks, _, _ = tasks_and_sources()
    wanted = [
        ("Impressionism", "Early_Renaissance"),
        ("Early_Renaissance", "Ukiyo_e"),
        ("Early_Renaissance", "Minimalism"),
    ]
    picks = [next(t for t in tasks if t["source_style"] == ss and t["target_style"] == ts) for ss, ts in wanted]
    arms = [
        ("Final_5k_seed0_l000", "Source (l=0)"),
        ("Final_5k_seed0_l100", "Seed0 1-NFE l100"),
        ("Bridge4_V22_paired_l040", "Bridge4 l040"),
        ("Bridge4_V22_paired_l100", "Bridge4 l100"),
        ("Adv1NFE_V22_projgan_l100", "Adv1NFE l100"),
    ]
    cell = 190
    W = len(arms) * (cell + 6) + 10
    H = len(picks) * (cell + 26) + 28
    canvas = Image.new("RGB", (W, H), (24, 24, 24))
    dr = ImageDraw.Draw(canvas)
    for ci, (_, lbl) in enumerate(arms):
        dr.text((8 + ci * (cell + 6), 6), lbl, fill=(200, 245, 220))
    for ri, t in enumerate(picks):
        y0 = 26 + ri * (cell + 26)
        dr.text((8, y0 + cell + 2), f"{t['source_style'][:4]}->{t['target_style'][:4]}: {t['source_stem'][:22]}", fill=(255, 220, 120))
        for ci, (arm, _) in enumerate(arms):
            raw_dir = OUT / "raw" / arm
            for bf in sorted(raw_dir.glob("batch_*.pt")):
                pl = torch.load(bf, map_location="cpu", weights_only=False)
                if t["task_id"] in pl["ids"]:
                    j = list(pl["ids"]).index(t["task_id"])
                    im = Image.fromarray(pl["rgb_u8"][j].permute(1, 2, 0).numpy()).resize((cell, cell))
                    canvas.paste(im, (6 + ci * (cell + 6), y0))
                    break
    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png)


def run_duel_smoke(device: str = "cuda") -> dict:
    """16-step VRAM, speed, loss stability, and exact lambda=0 smoke check for both routes."""
    r_a = train_bridge4_route("SMOKE_bridge4", updates=3000, smoke_steps=16, device=device)
    r_b = train_adv1nfe_route("SMOKE_adv1nfe", updates=2400, smoke_steps=16, device=device)
    rep = {"updated_utc": utc(), "bridge4": r_a, "adv1nfe": r_b}
    save_json(OUT / "duel_smoke_report.json", rep)
    print(f"[DUEL SMOKE] {rep}", flush=True)
    return rep


def run_training_duel(device: str = "cuda") -> dict:
    """Execute full head-to-head training duel (Bridge4 vs Adv1NFE) and audit on test750."""
    t0 = time.monotonic()
    bridge_rep = train_bridge4_route("Bridge4_V22_paired", updates=3200, device=device)
    adv_rep = train_adv1nfe_route("Adv1NFE_V22_projgan", updates=2600, device=device)

    seed0_rep = read_json(OUT / "students/Final_5k_seed0/results.json")
    lat_bridge = _measure_duel_latency(OUT / "students/Bridge4_V22_paired/Bridge4_V22_paired_final.pt", "bridge4", device=device)
    lat_adv = _measure_duel_latency(OUT / "students/Adv1NFE_V22_projgan/Adv1NFE_V22_projgan_final.pt", "unet", device=device)

    r_s0 = read_json(OUT / "measured/Final_5k_seed0_l100/eval.json")["rows"]
    r_br = read_json(OUT / "measured/Bridge4_V22_paired_l100/eval.json")["rows"]
    r_ad = read_json(OUT / "measured/Adv1NFE_V22_projgan_l100/eval.json")["rows"]
    ci = {
        "Bridge4_minus_seed0_l100": paired_cluster_ci(r_br, r_s0),
        "Adv1NFE_minus_seed0_l100": paired_cluster_ci(r_ad, r_s0),
        "Bridge4_minus_Adv1NFE_l100": paired_cluster_ci(r_br, r_ad),
    }

    _plot_duel_pareto(seed0_rep, bridge_rep, adv_rep, OUT)
    _export_duel_visual_strip(OUT / "visual_audit_strip_duel.png")

    audit = {
        "updated_utc": utc(),
        "total_elapsed_min": round((time.monotonic() - t0) / 60.0, 2),
        "bridge4": bridge_rep,
        "adv1nfe": adv_rep,
        "latency_4090": {"bridge4": lat_bridge, "adv1nfe": lat_adv},
        "bootstrap_ci": ci,
    }
    save_json(OUT / "phase6_training_duel.json", audit)
    print(f"[DUEL COMPLETE] total={audit['total_elapsed_min']} min -> {OUT / 'phase6_training_duel.json'}", flush=True)
    return audit
