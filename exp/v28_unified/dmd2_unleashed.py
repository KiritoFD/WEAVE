"""Two-Arm 1-NFE Distribution Matching Verification on 4090 (Single Ckpt, Single Knob, Zero Anchors):

Arm 1 (RealGAN_Unleashed):
    ~19.5M UnifiedUnleashedUNet (2x16 global Self-Attn, K=8 scene-matched u_vivid, lambda=1 zero skip gate)
    + High-lambda L1 & Haar relaxation (min_w=0.03) + Coarse y16 LCN + Gray VGG self-sim
    + 5-6x stronger 7-Scale Projected GAN (VGG-16 s2/s3/s4 + ConvNeXt s1..s4 @ 384px) trained on
      100% REAL WikiArt train_4400 paintings (S10-matched domain, zero V22 teacher synthesis bias).

Arm 2 (FlowDMD2_RealGAN):
    Arm 1 + Flow-DMD2 distribution matching distillation (v_real^CFG - v_fake) using the frozen
    20.8M PEAK1500 CFG flow teacher + online trainable fake_flow score critic (both discarded after training).
"""
from __future__ import annotations

import argparse
import copy
import gc
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from exp.v19.aniso_teacher import latent_lcn_cost
from exp.v20.slf_flow import SLFUNet, project_latent_y16_lcn
from exp.v20.submode_patchswd import (
    MultiScalePatchSWDExtractor,
    load_convnext_tiny_offline,
    match_scene_submode,
    sliced_wasserstein_1d,
)
from exp.v22.core import CFGConfig, D5_SOLVER, predict_guided_velocity
from exp.v27_mainline.rollout_refine import FrozenVGG16Rollout
from exp.v27_single.bare_q32 import load_cache
from exp.v27_single.unified.train import conditioning_stats
from exp.v28_unified import (
    OUT,
    PEAK1500_CKPT,
    ROOT,
    SEED_TRAIN,
    assert_vram_budget,
    atomic_save,
)
from exp.v28_unified.student_models import PixelBridgeV28, build_student_model
from exp.v28_unified.train_duel import (
    _SpectralProjHead,
    _sample_duel_batch,
    relaxed_l1_weight,
)
from exp.v28_unified.train_student import _compute_targets, run_final_eval
from src.utils import decode_s10, load_s10_decoder
from src.utils.runtime import bf16_autocast, read_json, sanitize_cuda_env, save_json, seed_all
from src.utils.structure import dwt2_haar


def dmd2_surrogate_loss(
    z_hat: torch.Tensor,
    v_real: torch.Tensor,
    v_fake: torch.Tensor,
    tau: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute Flow-DMD2 stop-gradient surrogate loss whose gradient pushes z_hat toward (v_real - v_fake).

    Under rectified flow z_tau = (1 - tau) * eps + tau * z_1:
        z1_real - z1_fake = (1 - tau) * (v_real - v_fake).
    Minimizing 0.5 * ||z_hat - sg(z_hat + normalized(z1_real - z1_fake))||^2 produces exact
    descent direction +(v_real - v_fake).
    """
    one_m_t = (1.0 - tau.float()).view(-1, 1, 1, 1)
    delta_z1 = one_m_t * (v_real.float() - v_fake.float())
    scale = delta_z1.abs().mean(dim=(1, 2, 3), keepdim=True).clamp_min(0.15)
    step_dir = (delta_z1 / scale).clamp(-4.0, 4.0)
    target = (z_hat.float() + step_dir).detach()
    per_sample = 0.5 * (z_hat.float() - target).pow(2).mean(dim=(1, 2, 3))
    if sample_weights is not None:
        per_sample = per_sample * sample_weights.to(per_sample.device, dtype=per_sample.dtype)
    return per_sample.mean()


class ProjectedUnleashedD(nn.Module):
    """7-Scale Projected GAN Discriminator over frozen VGG-16 + ConvNeXt Stages 1-4 (384px)."""

    _SPECS = (
        ("vgg_s2", 128, 32),
        ("vgg_s3", 256, 16),
        ("vgg_s4", 512, 16),
        ("cnx_s1", 96, 32),
        ("cnx_s2", 192, 24),
        ("cnx_s3", 384, 16),
        ("cnx_s4", 768, 12),
    )

    def __init__(self, width: int = 128, seed: int = SEED_TRAIN + 808):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.heads = nn.ModuleDict()
        for name, c_dim, _ in self._SPECS:
            q, _ = torch.linalg.qr(torch.randn(c_dim, 64, generator=g), mode="reduced")
            self.register_buffer(f"proj_{name}", q)
            self.heads[name] = _SpectralProjHead(64, width, n_styles=5)

    def forward(self, feats: dict[str, torch.Tensor], tid: torch.Tensor) -> torch.Tensor:
        scores = []
        for name, _, size in self._SPECS:
            f = F.adaptive_avg_pool2d(feats[name].float(), size)
            proj = getattr(self, f"proj_{name}")
            p = torch.einsum("bchw,cd->bdhw", f, proj)
            scores.append(self.heads[name](p, tid))
        return torch.stack(scores, dim=0).mean(dim=0)


class _UnleashedCritic:
    """Referee-free S10-BF16 critic: 4-stage ConvNeXt+VGG features + Patch-SWD + Y16/VGG-SS struct shield."""

    def __init__(self, v18_luma: dict, device: str = "cuda"):
        self.device = device
        self.pixel = PixelBridgeV28(device=device)
        self.vgg_ro = FrozenVGG16Rollout().to(device).eval().requires_grad_(False)
        self.swd_ext = MultiScalePatchSWDExtractor(n_slices=64, seed=SEED_TRAIN + 505).to(device).eval().requires_grad_(False)
        cnx_full = load_convnext_tiny_offline(device)
        self.cnx_s4 = cnx_full[6:8].to(device).eval().requires_grad_(False)
        del cnx_full
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
            x384 = F.interpolate(rgb01.float(), size=(384, 384), mode="bilinear", align_corners=False)
            xn = (x384 - self.swd_ext.im_mean) / self.swd_ext.im_std
            c1 = self.swd_ext.s1(xn)
            c2 = self.swd_ext.s2(c1)
            c3 = self.swd_ext.s3(c2)
            c4 = self.cnx_s4(c3)
        return {"vgg_s2": f2, "vgg_s3": f3, "vgg_s4": f4, "cnx_s1": c1, "cnx_s2": c2, "cnx_s3": c3, "cnx_s4": c4}

    def compute_pixel_and_struct(
        self,
        z_pred: torch.Tensor,
        z_style_ref: torch.Tensor,
        z_src: torch.Tensor,
        y16_src: torch.Tensor,
        same_mask: torch.Tensor,
        levels: torch.Tensor,
        proj: tuple[torch.Tensor, ...],
        *,
        w_mom: float = 0.25,
        w_vgg_swd: float = 0.25,
        w_cnx_swd: float = 0.32,
        w_gss: float = 1.35,
        w_y16: float = 0.45,
        w_lcn: float = 0.45,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (total_pix_loss, rgb_pred, gss_diag)."""
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
            rgb_ref = decode_s10(
                self.pixel.decoder, z_style_ref, dtype=torch.bfloat16, batch=B, enable_grad=False, output_cpu=False
            ).clamp(0.0, 1.0)
            rgb_src = decode_s10(
                self.pixel.decoder, z_src, dtype=torch.bfloat16, batch=B, enable_grad=False, output_cpu=False
            ).clamp(0.0, 1.0)
            self.pixel.layers.clear()
            with bf16_autocast(self.device):
                ref_f = self.pixel.vgg(rgb_ref.float(), multi_scale=False).float()
            ref_t = tuple(F.adaptive_avg_pool2d(x, 16).float().flatten(2).transpose(1, 2) for x in self.pixel.layers)
            self.pixel.layers.clear()
            with bf16_autocast(self.device):
                cnx_ref = self.swd_ext.extract_tokens(rgb_ref)
                r42_src = self.vgg_ro.forward_gray_r42(rgb_src).float()

        mom_ps = F.smooth_l1_loss((cur_f - ref_f) / self.pixel.scale, torch.zeros_like(cur_f), beta=0.20, reduction="none").mean(dim=1)
        vgg_swd_ps = sum(
            ((a @ p).sort(dim=1).values - (b @ p).sort(dim=1).values).abs().mean(dim=(1, 2))
            for a, b, p in zip(cur_t, ref_t, proj)
        ) / len(proj)
        cnx_swd_ps = (
            1.1 * sliced_wasserstein_1d(cnx_cur["rgb"].float(), cnx_ref["rgb"].float(), self.swd_ext.p_rgb)
            + 0.35 * sliced_wasserstein_1d(cnx_cur["s1"].float(), cnx_ref["s1"].float(), self.swd_ext.p_s1)
            + 0.45 * sliced_wasserstein_1d(cnx_cur["s2"].float(), cnx_ref["s2"].float(), self.swd_ext.p_s2)
            + 0.55 * sliced_wasserstein_1d(cnx_cur["s3"].float(), cnx_ref["s3"].float(), self.swd_ext.p_s3)
        )

        # Coarse composition shields on cross-style samples (release fine 64x64 LCN gently at high lambda)
        cross_w = (~same_mask).float() * (levels > 0.15).float()
        a16 = F.normalize(F.adaptive_avg_pool2d(r42_cur, 16).flatten(2), dim=1)
        b16 = F.normalize(F.adaptive_avg_pool2d(r42_src, 16).flatten(2), dim=1)
        gss_ps = F.smooth_l1_loss(
            torch.bmm(a16.transpose(1, 2), a16), torch.bmm(b16.transpose(1, 2), b16), beta=0.05, reduction="none"
        ).mean(dim=(1, 2))
        y16_cur = project_latent_y16_lcn(z_pred, self.w_luma, self.b_luma)
        y16_ps = F.smooth_l1_loss(y16_cur, y16_src.float(), beta=0.08, reduction="none").mean(dim=(1, 2, 3))
        lcn_ps = latent_lcn_cost(z_pred, z_src, self.w_luma, self.b_luma)
        fine_gate = (1.0 - 0.50 * ((levels - 0.50) / 0.50).clamp(0.0, 1.0))

        sw = torch.where(levels > 0.60, torch.full_like(levels, 1.6), torch.ones_like(levels))
        style_ps = sw * (w_mom * mom_ps + w_vgg_swd * vgg_swd_ps + w_cnx_swd * cnx_swd_ps)
        struct_ps = cross_w * (w_gss * gss_ps + w_y16 * y16_ps + fine_gate * w_lcn * lcn_ps)
        return (style_ps + struct_ps).mean(), rgb_pred, (gss_ps * cross_w).mean().detach()


class _FlowDMD2Guide:
    """Frozen PEAK1500 CFG flow teacher + online trainable fake_flow score critic for DMD2."""

    CFG_SOLVER = dict(D5_SOLVER, tau_min=0.28, tau_hi=0.55, wc_fig=1.05, wc_abs=1.00, ws_fig=3.90, wsp_fig_late=2.40)

    def __init__(self, device: str = "cuda", lr_fake: float = 1.5e-4):
        self.device = device
        ckpt = torch.load(PEAK1500_CKPT, map_location="cpu", weights_only=False)
        self.real_flow = SLFUNet(base_dim=144, u_dim=128).to(device).eval().requires_grad_(False)
        self.real_flow.load_state_dict(ckpt["ema"], strict=True)

        self.fake_flow = SLFUNet(base_dim=144, u_dim=128).to(device).train()
        self.fake_flow.load_state_dict(ckpt["ema"], strict=True)
        self.opt_fake = torch.optim.AdamW(self.fake_flow.parameters(), lr=lr_fake, betas=(0.9, 0.99), weight_decay=1e-4)
        self.cfg = CFGConfig(branches="full", rescale_phi=0.48)

    @torch.no_grad()
    def real_and_fake_velocities(
        self,
        z_tau: torch.Tensor,
        tau: torch.Tensor,
        y16: torch.Tensor,
        d16: torch.Tensor,
        tid: torch.Tensor,
        u_vivid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = len(z_tau)
        dev = self.device
        g_guid = torch.full((B,), 0.42, device=dev)
        lscale = torch.full((B,), 0.58, device=dev)
        content = {"y_in": y16, "d_in": d16}
        prog = float(tau.mean().item())
        v_real = predict_guided_velocity(
            self.real_flow, z_tau, tau, content, tid, u_vivid, lscale, g_guid, self.cfg, self.CFG_SOLVER, prog
        )
        self.fake_flow.eval()
        f0 = torch.zeros(B, dtype=torch.bool, device=dev)
        with bf16_autocast(dev):
            v_fake = self.fake_flow(
                z_tau, y16, d16, tid, u_vivid, tau, drop_style=f0, drop_content=f0, layout_scale=lscale
            ).float()
        self.fake_flow.train()
        return v_real, v_fake

    def update_fake_flow(
        self,
        z_hat_det: torch.Tensor,
        y16: torch.Tensor,
        d16: torch.Tensor,
        tid: torch.Tensor,
        u_vivid: torch.Tensor,
        n_steps: int = 2,
    ) -> float:
        B = len(z_hat_det)
        dev = self.device
        f0 = torch.zeros(B, dtype=torch.bool, device=dev)
        lscale = torch.full((B,), 0.58, device=dev)
        tot = 0.0
        self.fake_flow.train()
        for _ in range(n_steps):
            tau = 0.05 + 0.87 * torch.rand(B, device=dev)
            eps = torch.randn_like(z_hat_det)
            zt = (1.0 - tau.view(B, 1, 1, 1)) * eps + tau.view(B, 1, 1, 1) * z_hat_det
            target_v = z_hat_det - eps
            self.opt_fake.zero_grad(set_to_none=True)
            with bf16_autocast(dev):
                pred_v = self.fake_flow(zt, y16, d16, tid, u_vivid, tau, drop_style=f0, drop_content=f0, layout_scale=lscale)
                loss = F.mse_loss(pred_v.float(), target_v)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.fake_flow.parameters(), 1.0)
            self.opt_fake.step()
            tot += float(loss.item()) / float(n_steps)
        return tot


def _sample_real_wikiart_indices(
    tid_batch: torch.Tensor,
    core_cls: list[torch.Tensor],
    all_cls: list[torch.Tensor],
    gen: torch.Generator,
) -> torch.Tensor:
    out = []
    for t in tid_batch.detach().cpu().tolist():
        use_core = float(torch.rand(1, generator=gen).item()) < 0.72
        pool = core_cls[int(t)] if (use_core and len(core_cls[int(t)]) > 0) else all_cls[int(t)]
        pick = int(torch.randint(len(pool), (1,), generator=gen).item())
        out.append(int(pool[pick].item()))
    return torch.tensor(out, dtype=torch.long)


def train_unleashed_warm(
    arm_name: str = "Unleashed_Warm",
    *,
    updates: int = 2400,
    eff_batch: int = 96,
    micro: int = 24,
    pixel_micro: int = 6,
    lr: float = 9.5e-4,
    smoke_steps: int = 0,
    device: str = "cuda",
) -> Path:
    """Fast Stage-1 pretraining of 18.7M UnifiedUnleashedUNet on pool_V22_clean (no GAN/DMD2)."""
    sanitize_cuda_env()
    seed_all(SEED_TRAIN + 17)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.cuda.reset_peak_memory_stats(device)

    run_dir = OUT / "students" / arm_name
    run_dir.mkdir(parents=True, exist_ok=True)
    final_ckpt = run_dir / f"{arm_name}_final.pt"
    if final_ckpt.exists() and smoke_steps == 0:
        print(f"[{arm_name}] Reusing completed warm checkpoint.", flush=True)
        return final_ckpt

    v18 = load_cache("runs/v18/cache/ccf_pool_cache.pt")
    tr = v18["train"]
    ts = load_cache("runs/v18/cache/test750_struct_cache.pt")
    c21 = load_cache("runs/v21/cache/v21_stiefel_ref_cache.pt")
    sub_bank = load_cache("runs/v20/cache/core_knn_bank.pt")
    seed0_ck = torch.load(OUT / "students/Final_5k_seed0/Final_5k_seed0_final.pt", map_location="cpu", weights_only=False)
    raw_stats = seed0_ck["raw_stats"].to(device)
    priors = {k: seed0_ck["priors"][k].to(device).float() for k in seed0_ck["priors"]}
    del seed0_ck

    critic = _UnleashedCritic(v18["luma_proj"], device=device)
    pairs = torch.load(OUT / "pools/pool_V22_clean.pt", map_location="cpu", weights_only=False)
    same_ids = pairs["same"].nonzero().flatten()
    cross_ids = (~pairs["same"]).nonzero().flatten()
    critic.set_moment_scale(pairs["hiS"][cross_ids[::125][:48]])

    model = build_student_model("unleashed", raw_stats.shape[-1], device=device)
    num_params = sum(p.numel() for p in model.parameters())
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=2.0e-4)
    gen = torch.Generator().manual_seed(SEED_TRAIN + 19)

    target_steps = smoke_steps if smoke_steps > 0 else updates
    t0 = time.monotonic()

    for step in range(1, target_steps + 1):
        warm = min(1.0, step / 120.0)
        frac = max(0, step - 120) / float(max(1, updates - 120))
        cos_s = 0.08 + 0.92 * 0.5 * (1.0 + math.cos(math.pi * frac))
        for pg in opt.param_groups:
            pg["lr"] = lr * warm * cos_s

        all_idx, all_l, all_tc = _sample_duel_batch(pairs, same_ids, cross_ids, step, eff_batch, pixel_micro, gen)
        pg_s = torch.Generator(device=device).manual_seed(SEED_TRAIN + 5000 + step)
        proj = tuple(torch.linalg.qr(torch.randn(c, 64, device=device, generator=pg_s), mode="reduced").Q for c in (64, 128, 256))

        opt.zero_grad(set_to_none=True)
        tot_lat, tot_pix = 0.0, 0.0
        for st in range(0, eff_batch, micro):
            stop = min(eff_batch, st + micro)
            bi = all_idx[st:stop]
            b_l = all_l[st:stop].to(device)
            b_tc = all_tc[st:stop].to(device)
            si = pairs["src"][bi]
            zs = tr["z"][si].to(device).float()
            c64b = tr["c_64"][si].to(device).float()
            q32b = c21["train"]["q32"][si].to(device).float()
            d16b = tr["dino_16"][si].to(device).float()
            y16b = c21["train"]["y16"][si].to(device).float()
            tidb = pairs["tid"][bi].to(device)
            same_m = pairs["same"][bi].to(device)
            _, u_viv = match_scene_submode(d16b, tidb, sub_bank)

            tgt = _compute_targets(zs, pairs["hiC"][bi].to(device).float(), None, pairs["hiS"][bi].to(device).float(), b_l, b_tc, "2tier")
            tgt = torch.where(same_m[:, None, None, None], zs, tgt)

            with bf16_autocast(device):
                pred = model(zs, c64b, q32b, tidb, b_l, raw_stats, priors, noise=torch.randn_like(zs), u_vivid=u_viv).float()
            l1_ps = F.smooth_l1_loss(pred, tgt, beta=0.06, reduction="none").mean(dim=(1, 2, 3))
            fp, ft = dwt2_haar(pred), dwt2_haar(tgt)
            bands = sum(w * F.l1_loss(a, b) for w, a, b in zip((1.0, 1.0, 0.5), fp[1:], ft[1:]))
            l_lat = l1_ps.mean() + 0.25 * bands
            w_slice = (stop - st) / float(eff_batch)

            if st == 0:
                sl_p = slice(0, pixel_micro)
                l_pix, _, _ = critic.compute_pixel_and_struct(
                    pred[sl_p], tgt[sl_p], zs[sl_p], y16b[sl_p], same_m[sl_p], b_l[sl_p], proj,
                )
                (w_slice * l_lat + 0.24 * l_pix).backward()
                tot_pix = float((0.24 * l_pix).item())
            else:
                (w_slice * l_lat).backward()
            tot_lat += float((w_slice * l_lat).item())

        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5, error_if_nonfinite=True)
        opt.step()
        with torch.no_grad():
            torch._foreach_lerp_(list(ema.parameters()), [p.detach() for p in model.parameters()], 0.008)

        if step == 1 or step % 200 == 0 or step == target_steps:
            dt = (time.monotonic() - t0) / step
            print(f"[{arm_name}] {step:4d}/{target_steps} lat={tot_lat:.4f} pix={tot_pix:.4f} gn={float(gn):.2f} ({dt:.3f}s/st)", flush=True)

    elapsed = time.monotonic() - t0
    del critic, model, opt
    gc.collect()
    torch.cuda.empty_cache()
    if smoke_steps == 0:
        atomic_save({"ema": ema.state_dict(), "raw_stats": raw_stats, "priors": priors, "step": updates, "params": num_params, "arch": "unleashed"}, final_ckpt)
        from exp.v28_unified.train_student import evaluate_student_v28
        quick = evaluate_student_v28(ema, "unleashed", raw_stats, priors, ts, c21["test750"]["q32"],
                                     tag=arm_name, levels=(0.40, 0.75, 1.00), sub_bank=sub_bank, device=device)
        save_json(run_dir / "warm_quick_eval.json", {"train_min": round(elapsed / 60.0, 2), "levels": quick})
    del ema
    gc.collect()
    torch.cuda.empty_cache()
    return final_ckpt


def train_unleashed_arm(
    arm_name: str,
    *,
    use_dmd2: bool,
    warm_ckpt_path: Path | None = None,
    updates: int = 2000,
    eff_batch: int = 96,
    micro: int = 16,
    pixel_micro: int = 6,
    dmd_micro: int = 12,
    lr_g: float = 1.8e-4,
    lr_d: float = 1.2e-4,
    adv_w_max: float = 0.25,
    dmd_w_max: float = 0.28,
    smoke_steps: int = 0,
    device: str = "cuda",
) -> dict:
    """Train either Arm 1 (RealGAN_Unleashed_v2) or Arm 2 (FlowDMD2_RealGAN_v2) on 4090."""
    sanitize_cuda_env()
    seed_all(SEED_TRAIN + (99 if use_dmd2 else 55))
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
    sub_bank = load_cache("runs/v20/cache/core_knn_bank.pt")

    w_ckpt = warm_ckpt_path if warm_ckpt_path is not None else (OUT / "students/Unleashed_Warm/Unleashed_Warm_final.pt")
    if not w_ckpt.exists():
        w_ckpt = OUT / "students/Final_5k_seed0/Final_5k_seed0_final.pt"
    warm_ck = torch.load(w_ckpt, map_location="cpu", weights_only=False)
    raw_stats = warm_ck["raw_stats"].to(device)
    priors = {k: warm_ck["priors"][k].to(device).float() for k in warm_ck["priors"]}

    critic = _UnleashedCritic(v18["luma_proj"], device=device)
    pairs = torch.load(OUT / "pools/pool_V22_clean.pt", map_location="cpu", weights_only=False)
    same_ids = pairs["same"].nonzero().flatten()
    cross_ids = (~pairs["same"]).nonzero().flatten()

    tr_sid = tr["sid"].long()
    all_cls = [(tr_sid == s).nonzero().flatten() for s in range(5)]
    core_cls = [ci.cpu().long() for ci in sub_bank["core_idx_per_style"]]
    real_samp_idx = torch.cat([c[:10] for c in core_cls])
    critic.set_moment_scale(tr["z"][real_samp_idx])

    model = build_student_model("unleashed", raw_stats.shape[-1], device=device)
    if warm_ck.get("arch") == "unleashed":
        model.load_state_dict(warm_ck["ema"], strict=True)
        print(f"[{arm_name}] Warm-started 18.7M Unleashed UNet from {w_ckpt.name}", flush=True)
    del warm_ck
    num_params = sum(p.numel() for p in model.parameters())
    ema = copy.deepcopy(model).eval().requires_grad_(False)

    disc = ProjectedUnleashedD(width=128, seed=SEED_TRAIN + 808).to(device)
    opt_g = torch.optim.AdamW(model.parameters(), lr=lr_g, betas=(0.5, 0.99), weight_decay=1.0e-4)
    opt_d = torch.optim.AdamW(disc.parameters(), lr=lr_d, betas=(0.5, 0.99), weight_decay=1.0e-4)
    dmd_guide = _FlowDMD2Guide(device=device) if use_dmd2 else None
    gen = torch.Generator().manual_seed(SEED_TRAIN + (202 if use_dmd2 else 101))

    target_steps = smoke_steps if smoke_steps > 0 else updates
    t0 = time.monotonic()

    for step in range(1, target_steps + 1):
        warm = min(1.0, step / 150.0)
        frac = max(0, step - 150) / float(max(1, updates - 150))
        cos_s = 0.15 + 0.85 * 0.5 * (1.0 + math.cos(math.pi * frac))
        for pg in opt_g.param_groups:
            pg["lr"] = lr_g * warm * cos_s
        for pg in opt_d.param_groups:
            pg["lr"] = lr_d * warm * cos_s

        # Put high-weight pixel & dmd candidates at front of batch
        all_idx, all_l, all_tc = _sample_duel_batch(pairs, same_ids, cross_ids, step, eff_batch, dmd_micro, gen)
        pg_s = torch.Generator(device=device).manual_seed(SEED_TRAIN + 9000 + step)
        proj = tuple(torch.linalg.qr(torch.randn(c, 64, device=device, generator=pg_s), mode="reduced").Q for c in (64, 128, 256))

        opt_g.zero_grad(set_to_none=True)
        tot_lat, tot_pix, tot_adv, tot_dmd, tot_gss = 0.0, 0.0, 0.0, 0.0, 0.0
        cached_real_rgb, cached_fake_rgb, cached_tid, cached_ramp = None, None, None, None
        dmd_cache = None

        for st in range(0, eff_batch, micro):
            stop = min(eff_batch, st + micro)
            bi = all_idx[st:stop]
            b_l = all_l[st:stop].to(device)
            b_tc = all_tc[st:stop].to(device)
            si = pairs["src"][bi]
            zs = tr["z"][si].to(device).float()
            c64b = tr["c_64"][si].to(device).float()
            q32b = c21["train"]["q32"][si].to(device).float()
            d16b = tr["dino_16"][si].to(device).float()
            y16b = c21["train"]["y16"][si].to(device).float()
            tidb = pairs["tid"][bi].to(device)
            same_m = pairs["same"][bi].to(device)
            flat_sub, u_viv = match_scene_submode(d16b, tidb, sub_bank)

            tgt = _compute_targets(zs, pairs["hiC"][bi].to(device).float(), None, pairs["hiS"][bi].to(device).float(), b_l, b_tc, "2tier")
            tgt = torch.where(same_m[:, None, None, None], zs, tgt)

            with bf16_autocast(device):
                pred = model(zs, c64b, q32b, tidb, b_l, raw_stats, priors, noise=torch.randn_like(zs), u_vivid=u_viv).float()
            l1_ps = F.smooth_l1_loss(pred, tgt, beta=0.06, reduction="none").mean(dim=(1, 2, 3))
            fp, ft = dwt2_haar(pred), dwt2_haar(tgt)
            bands_ps = sum(w * (a - b).abs().mean(dim=(1, 2, 3)) for w, a, b in zip((1.0, 1.0, 0.5), fp[1:], ft[1:]))
            l1_w = torch.where(same_m, torch.ones_like(b_l), relaxed_l1_weight(b_l, min_weight=0.03))
            l_lat = ((l1_ps + 0.20 * bands_ps) * l1_w).mean()
            w_slice = (stop - st) / float(eff_batch)

            if st == 0:
                sl_p = slice(0, pixel_micro)
                # For high-lambda cross tasks, Patch-SWD targets the K=8 vivid REAL WikiArt exemplar!
                ex_z = sub_bank["sub_exemplar_z"].to(device).float()[tidb[sl_p], flat_sub[sl_p] % int(sub_bank["k_sub"])]
                use_real_ex = ((~same_m[sl_p]) & (b_l[sl_p] > 0.60))[:, None, None, None]
                z_style_ref = torch.where(use_real_ex, ex_z, tgt[sl_p])

                l_pix, rgb_p, gss_d = critic.compute_pixel_and_struct(
                    pred[sl_p], z_style_ref, zs[sl_p], y16b[sl_p], same_m[sl_p], b_l[sl_p], proj,
                )

                # RealGAN against 100% real train_4400 WikiArt paintings in same S10 domain
                r_idx = _sample_real_wikiart_indices(tidb[sl_p], core_cls, all_cls, gen)
                with torch.no_grad():
                    rgb_real = decode_s10(
                        critic.pixel.decoder, tr["z"][r_idx].to(device).float(), dtype=torch.bfloat16, batch=pixel_micro, enable_grad=False, output_cpu=False
                    ).clamp(0.0, 1.0)
                ramp_adv = ((b_l[sl_p] - 0.35) / 0.40).clamp(0.0, 1.0) * (~same_m[sl_p]).float()
                for p in disc.parameters():
                    p.requires_grad_(False)
                d_fake = disc(critic.extract_d_feats(rgb_p), tidb[sl_p])
                for p in disc.parameters():
                    p.requires_grad_(True)
                l_adv = -(ramp_adv * d_fake).mean()
                adv_w = adv_w_max * min(1.0, step / 250.0)

                loss_total = w_slice * l_lat + 0.36 * l_pix + adv_w * l_adv

                if dmd_guide is not None:
                    sl_d = slice(0, min(dmd_micro, stop - st))
                    dmd_m = (~same_m[sl_d]) & (b_l[sl_d] > 0.50)
                    if bool(dmd_m.any()):
                        z_hat_d = pred[sl_d][dmd_m]
                        y16_d = y16b[sl_d][dmd_m]
                        d16_d = d16b[sl_d][dmd_m]
                        tid_d = tidb[sl_d][dmd_m]
                        u_d = u_viv[sl_d][dmd_m]
                        lvl_d = b_l[sl_d][dmd_m]
                        tau_d = 0.08 + 0.78 * torch.rand(len(z_hat_d), device=device)
                        zt_d = (1.0 - tau_d.view(-1, 1, 1, 1)) * torch.randn_like(z_hat_d) + tau_d.view(-1, 1, 1, 1) * z_hat_d.detach()
                        v_r, v_f = dmd_guide.real_and_fake_velocities(zt_d, tau_d, y16_d, d16_d, tid_d, u_d)
                        ramp_d = ((lvl_d - 0.50) / 0.35).clamp(0.0, 1.0)
                        l_dmd = dmd2_surrogate_loss(z_hat_d, v_r, v_f, tau_d, sample_weights=ramp_d)
                        dmd_w = dmd_w_max * min(1.0, max(0, step - 80) / 160.0)
                        loss_total = loss_total + dmd_w * l_dmd
                        tot_dmd = float(l_dmd.item())
                        dmd_cache = (z_hat_d.detach(), y16_d, d16_d, tid_d, u_d)

                loss_total.backward()
                tot_pix = float((0.36 * l_pix).item())
                tot_adv = float(l_adv.item())
                tot_gss = float(gss_d.item())
                cached_real_rgb = rgb_real.detach()
                cached_fake_rgb = rgb_p.detach()
                cached_tid = tidb[sl_p].detach()
                cached_ramp = ramp_adv.detach()
            else:
                (w_slice * l_lat).backward()
            tot_lat += float((w_slice * l_lat).item())

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5, error_if_nonfinite=True)
        opt_g.step()
        with torch.no_grad():
            torch._foreach_lerp_(list(ema.parameters()), [p.detach() for p in model.parameters()], 0.005)

        # Discriminator Hinge Update on 100% Real WikiArt vs Fake S10 (anti-saturation margin + skip <0.35)
        opt_d.zero_grad(set_to_none=True)
        d_val = 0.0
        assert cached_real_rgb is not None and cached_fake_rgb is not None
        if float(cached_ramp.sum().item()) > 0.0:
            d_real = disc(critic.extract_d_feats(cached_real_rgb), cached_tid)
            d_fk = disc(critic.extract_d_feats(cached_fake_rgb), cached_tid)
            l_d = (cached_ramp * (F.relu(0.95 - d_real) + F.relu(0.95 + d_fk))).mean()
            d_val = float(l_d.item())
            if d_val >= 0.35:
                l_d.backward()
                torch.nn.utils.clip_grad_norm_(disc.parameters(), 2.0)
                opt_d.step()

        # Update fake score Flow critic for DMD2
        f_loss = 0.0
        if dmd_guide is not None and dmd_cache is not None:
            f_loss = dmd_guide.update_fake_flow(*dmd_cache, n_steps=2)

        if step == 1 or step % 100 == 0 or step == target_steps:
            dt = (time.monotonic() - t0) / step
            print(
                f"[{arm_name}] {step:4d}/{target_steps} lat={tot_lat:.4f} pix={tot_pix:.4f} "
                f"adv={tot_adv:.3f} D={d_val:.3f} dmd={tot_dmd:.3f} fk={f_loss:.3f} gss={tot_gss:.4f} "
                f"gn={float(grad_norm):.2f} ({dt:.3f}s/st)",
                flush=True,
            )

        if smoke_steps == 0 and (step % 500 == 0 or step == updates):
            atomic_save(
                {"model": model.state_dict(), "ema": ema.state_dict(), "disc": disc.state_dict(), "step": step, "updates": updates},
                run_dir / "resume.pt",
            )

    vram_gb = assert_vram_budget(device=device, tag=arm_name)
    elapsed = time.monotonic() - t0
    del critic, disc, model, opt_g, opt_d, dmd_guide
    gc.collect()
    torch.cuda.empty_cache()

    if smoke_steps > 0:
        zs_t = tr["z"][:2].to(device).float()
        _, uv_t = match_scene_submode(tr["dino_16"][:2].to(device).float(), torch.tensor([0, 4], device=device), sub_bank)
        id_out = ema(zs_t, tr["c_64"][:2].to(device).float(), c21["train"]["q32"][:2].to(device).float(),
                     torch.tensor([0, 4], device=device), torch.zeros(2, device=device), raw_stats, priors,
                     noise=torch.randn_like(zs_t), u_vivid=uv_t)
        id_err = float((id_out - zs_t).abs().max().item())
        del ema
        torch.cuda.empty_cache()
        return {
            "arm": arm_name,
            "params": num_params,
            "sec_per_step": round(elapsed / smoke_steps, 4),
            "est_min": round((elapsed / smoke_steps) * updates / 60.0, 1),
            "peak_vram_gb": round(vram_gb, 3),
            "lambda_zero_max_abs": id_err,
        }

    atomic_save({"ema": ema.state_dict(), "raw_stats": raw_stats, "priors": priors, "step": updates, "params": num_params, "arch": "unleashed"}, final_ckpt)
    rep = run_final_eval(
        ema, "unleashed", raw_stats, priors, ts, c21["test750"]["q32"], updates,
        run_dir=run_dir, tag=arm_name, checkpoint_name=final_ckpt.name, sub_bank=sub_bank, device=device,
    )
    rep["peak_vram_gb"] = round(vram_gb, 3)
    rep["train_elapsed_min"] = round(elapsed / 60.0, 2)
    del ema
    gc.collect()
    torch.cuda.empty_cache()
    rep["latency_4090"] = _measure_unleashed_latency(final_ckpt, sub_bank, device=device)
    save_json(run_dir / "results.json", rep)
    return rep


def _measure_unleashed_latency(ckpt_path: Path, sub_bank: dict, device: str = "cuda") -> dict:
    saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw_stats = saved["raw_stats"].to(device)
    priors = {k: v.to(device) for k, v in saved["priors"].items()}
    model = build_student_model("unleashed", raw_stats.shape[-1], device=device).eval()
    model.load_state_dict(saved["ema"])
    dec = load_s10_decoder(device)
    sub_sc = sub_bank["sub_scene"].to(device).float()
    sub_u = sub_bank["sub_u"].to(device).float()
    local_bank = {"k_sub": sub_bank["k_sub"], "sub_scene": sub_sc, "sub_u": sub_u}
    out = {}
    with torch.no_grad():
        for bs in (1, 16):
            zs = torch.randn(bs, 4, 64, 64, device=device)
            c64 = torch.randn(bs, 4, 64, 64, device=device)
            q32 = torch.randn(bs, 32, 16, 16, device=device)
            d16 = torch.randn(bs, 384, 16, 16, device=device)
            tid = torch.zeros(bs, dtype=torch.long, device=device)
            st = torch.ones(bs, device=device)
            eps = torch.randn_like(zs)
            for _ in range(6):
                _, u_viv = match_scene_submode(d16, tid, local_bank)
                _ = model(zs, c64, q32, tid, st, raw_stats, priors, noise=eps, u_vivid=u_viv)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(25):
                _, u_viv = match_scene_submode(d16, tid, local_bank)
                z_out = model(zs, c64, q32, tid, st, raw_stats, priors, noise=eps, u_vivid=u_viv)
            torch.cuda.synchronize()
            m_ms = (time.perf_counter() - t0) * 1000.0 / (25 * bs)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            for _ in range(25):
                _, u_viv = match_scene_submode(d16, tid, local_bank)
                z_out = model(zs, c64, q32, tid, st, raw_stats, priors, noise=eps, u_vivid=u_viv)
                _ = decode_s10(dec, z_out, batch=bs, output_cpu=False)
            torch.cuda.synchronize()
            e_ms = (time.perf_counter() - t1) * 1000.0 / (25 * bs)
            out[f"b{bs}"] = {
                "model_ms_per_img": round(m_ms, 2),
                "e2e_s10_ms_per_img": round(e_ms, 2),
                "e2e_img_per_s": round(1000.0 / e_ms, 1),
            }
    del model, dec
    torch.cuda.empty_cache()
    return out


def run_two_arm_smoke(device: str = "cuda") -> dict:
    w_ck = train_unleashed_warm("Unleashed_Warm_smoke", smoke_steps=10, device=device)
    r1 = train_unleashed_arm("RealGAN_Unleashed_v2_smoke", use_dmd2=False, smoke_steps=10, device=device)
    r2 = train_unleashed_arm("FlowDMD2_RealGAN_v2_smoke", use_dmd2=True, smoke_steps=10, device=device)
    rep = {"RealGAN_Unleashed_v2": r1, "FlowDMD2_RealGAN_v2": r2}
    save_json(OUT / "dmd2_smoke_report.json", rep)
    print(f"[SMOKE OK] {rep}", flush=True)
    return rep


def run_two_arm_full(updates: int = 2000, warm_updates: int = 2400, device: str = "cuda") -> dict:
    w_ck = train_unleashed_warm("Unleashed_Warm", updates=warm_updates, device=device)
    r1 = train_unleashed_arm("RealGAN_Unleashed_v2", use_dmd2=False, warm_ckpt_path=w_ck, updates=updates, device=device)
    r2 = train_unleashed_arm("FlowDMD2_RealGAN_v2", use_dmd2=True, warm_ckpt_path=w_ck, updates=updates, device=device)
    summary = {"RealGAN_Unleashed_v2": r1, "FlowDMD2_RealGAN_v2": r2}
    save_json(OUT / "phase7_dmd2_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="V28 1-NFE RealGAN & Flow-DMD2 2-Arm Verification")
    parser.add_argument("--phase", choices=("smoke", "warm", "arm1", "arm2", "full"), required=True)
    parser.add_argument("--updates", type=int, default=2000)
    parser.add_argument("--warm-updates", type=int, default=2400)
    args = parser.parse_args()
    if args.phase == "smoke":
        run_two_arm_smoke()
    elif args.phase == "warm":
        train_unleashed_warm("Unleashed_Warm", updates=args.warm_updates)
    elif args.phase == "arm1":
        train_unleashed_arm("RealGAN_Unleashed_v2", use_dmd2=False, updates=args.updates)
    elif args.phase == "arm2":
        train_unleashed_arm("FlowDMD2_RealGAN_v2", use_dmd2=True, updates=args.updates)
    else:
        run_two_arm_full(updates=args.updates, warm_updates=args.warm_updates)


if __name__ == "__main__":
    main()
