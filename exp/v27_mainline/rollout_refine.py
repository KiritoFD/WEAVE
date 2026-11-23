"""阶段 2a / 2b / 2c & 阶段 3: 4 步无截断可微 Rollout + Part-Bank Spatial AdaGN + Part-Bary-SWD 闭环训练与评测.

严格控制实验阶梯:
  - P2_Rollout_Glob (2a): 无部件银行, 4 步 Rollout + 全局 VGG-SWD + Haar 能量比 + 灰度自相似 + P1-Reg
  - P2_Part_Glob    (2b): 接入 Part-Bank Spatial AdaGN, 仅用同 2a 全局损失 (单变量隔离架构增量)
  - P2_Part_Closed  (2c): 接入 Part-Bank Spatial AdaGN + 全局 SWD + Part-Bary-SWD + Haar + 灰度自相似 (完整闭环)
  - P2_1Shot_Bonus  (3):  零训练复用 2c 权重, 以 lookup_exemplar(kappa=8.0) 测 SG-5 / Holdout 单图 1-Shot

用法 (4090):
  python -m exp.v27_mainline.rollout_refine --arm P2_Rollout_Glob --updates 2500
  python -m exp.v27_mainline.rollout_refine --arm P2_Part_Glob    --updates 2500
  python -m exp.v27_mainline.rollout_refine --arm P2_Part_Closed  --updates 2500
"""
from __future__ import annotations

import argparse
import copy
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from exp.v18 import tasks_and_sources
from exp.v19.fiber_duality import apply_latent_chroma_shift
from exp.v20.run_v20 import get_test750_y16_and_d16
from exp.v20.slf_flow import coarse_y16_lcn, match_scene_submode
from exp.v20.submode_patchswd import build_or_load_core_knn_bank
from exp.v21.stiefel_ref_cache import SG_5_STEMS, build_or_load_v21_cache
from exp.v27_mainline import OUT, ROOT, RawWriterV27M, build_wrong_task_index, run_measure_v27m
from exp.v27_mainline.dir1_reflow import (
    ALPHA_BRIDGE,
    SLFUNetGamma,
    build_or_load_dir1_teacher_19800,
    prepare_q32_cond_inputs,
)
from exp.v27_mainline.part_bank import (
    PART_CACHE,
    PartStyleBank,
    SLFUNetPart,
    build_or_load_part_bank,
    extract_patch_102d,
    part_barycentric_swd,
    part_transport_plan,
)
from src.utils import decode_s10, load_s10_decoder
from src.utils._local_eval_refs import load_dino_model
from src.utils.protocol import EVAL_SEED, load_vgg16_features_offline
from src.utils.runtime import bf16_autocast as autocast, sanitize_cuda_env, save_json, seed_all, utc
from src.utils.structure import extract_color_fiber_27d, local_contrast_norm, rgb_to_gray


class FrozenVGG16Rollout(nn.Module):
    """Frozen ImageNet VGG-16 extracting relu1_2, relu2_2, relu3_3, relu4_3 (style) & relu4_2 (gray self-sim)."""

    def __init__(self):
        super().__init__()
        vgg = load_vgg16_features_offline("cpu")
        self.s1 = vgg[:4]     # relu1_2 (64ch, 256x256)
        self.s2 = vgg[4:9]    # relu2_2 (128ch, 128x128)
        self.s3 = vgg[9:16]   # relu3_3 (256ch, 64x64)
        self.s4_2 = vgg[16:21]  # relu4_2 (512ch, 32x32)
        self.s4_3 = vgg[21:23]  # relu4_3 (512ch, 32x32)
        self.register_buffer("im_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("im_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        for p in self.parameters():
            p.requires_grad_(False)

    def forward_style(self, rgb01: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = F.interpolate(rgb01.float(), size=(256, 256), mode="bilinear", align_corners=False)
        xn = (x - self.im_mean) / self.im_std
        f1 = self.s1(xn)
        f2 = self.s2(f1)
        f3 = self.s3(f2)
        f4 = self.s4_3(self.s4_2(f3))
        return f1, f2, f3, f4

    def forward_gray_r42(self, rgb01: torch.Tensor) -> torch.Tensor:
        y = rgb_to_gray(rgb01.float()).repeat(1, 3, 1, 1)
        x = F.interpolate(y, size=(256, 256), mode="bilinear", align_corners=False)
        xn = (x - self.im_mean) / self.im_std
        return self.s4_2(self.s3(self.s2(self.s1(xn))))


def dynamic_swd(f_out: torch.Tensor, f_ref: torch.Tensor, n_proj: int = 64) -> torch.Tensor:
    B, C, _, _ = f_out.shape
    a = f_out.flatten(2).transpose(1, 2)
    b = f_ref.detach().flatten(2).transpose(1, 2)
    if a.shape[1] > 1024:
        perm = torch.randperm(a.shape[1], device=f_out.device)[:1024]
        a = a[:, perm, :]
        b = b[:, perm, :]
    proj = F.normalize(torch.randn(C, n_proj, device=f_out.device, dtype=f_out.dtype), dim=0)
    pa = torch.sort(a @ proj, dim=1).values
    pb = torch.sort(b @ proj, dim=1).values
    return (pa - pb).abs().mean()


def vgg_moment_and_gram_loss(f_out: torch.Tensor, f_ref: torch.Tensor) -> torch.Tensor:
    """Channel-mean, std, and spatial Gram distance on deep VGG layers (captures Rococo/ER/Uki texture & contrast)."""
    a = f_out.flatten(2)
    b = f_ref.detach().flatten(2)
    mu_a, mu_b = a.mean(-1), b.mean(-1)
    sd_a = (a.var(-1, unbiased=False) + 1e-6).sqrt()
    sd_b = (b.var(-1, unbiased=False) + 1e-6).sqrt()
    mom_l = F.smooth_l1_loss(mu_a, mu_b, beta=0.1) + F.smooth_l1_loss(sd_a, sd_b, beta=0.1)
    # Normalized Gram matching (channel correlations)
    an = a / math.sqrt(a.shape[-1])
    bn = b / math.sqrt(b.shape[-1])
    ga = torch.bmm(an, an.transpose(1, 2))
    gb = torch.bmm(bn, bn.transpose(1, 2))
    return mom_l + 0.05 * F.smooth_l1_loss(ga, gb, beta=0.5)


def _haar_level_ratio(y: torch.Tensor, eps: float = 1e-4) -> tuple[torch.Tensor, torch.Tensor]:
    y00, y01 = y[:, :, 0::2, 0::2], y[:, :, 0::2, 1::2]
    y10, y11 = y[:, :, 1::2, 0::2], y[:, :, 1::2, 1::2]
    ll = 0.5 * (y00 + y01 + y10 + y11)
    lh = 0.5 * (y00 - y01 + y10 - y11)
    hl = 0.5 * (y00 + y01 - y10 - y11)
    hh = 0.5 * (y00 - y01 - y10 + y11)
    e_hf = (lh.square() + hl.square() + hh.square()).mean(dim=(1, 2, 3))
    e_lf = (ll - ll.mean(dim=(2, 3), keepdim=True)).square().mean(dim=(1, 2, 3))
    return torch.log((e_hf + eps) / (e_lf + eps)), ll


def haar_hf_lf_ratio_2level(rgb01: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    y = rgb_to_gray(rgb01.float())
    r1, ll1 = _haar_level_ratio(y, eps=eps)
    r2, _ = _haar_level_ratio(ll1, eps=eps)
    return 0.5 * (r1 + r2)


def gray_self_sim_loss(vgg_r42_out: torch.Tensor, vgg_r42_src: torch.Tensor) -> torch.Tensor:
    a = F.normalize(F.adaptive_avg_pool2d(vgg_r42_out, 16).flatten(2), dim=1)
    b = F.normalize(F.adaptive_avg_pool2d(vgg_r42_src.detach(), 16).flatten(2), dim=1)
    sim_a = torch.bmm(a.transpose(1, 2), a)
    sim_b = torch.bmm(b.transpose(1, 2), b)
    return F.smooth_l1_loss(sim_a, sim_b, beta=0.05)


def latent_y32_lcn_loss(z_hat: torch.Tensor, z_src: torch.Tensor, w_luma: torch.Tensor, b_luma: torch.Tensor) -> torch.Tensor:
    w = w_luma.to(z_hat.device).float().view(1, 4, 1, 1)
    b = b_luma.to(z_hat.device).float()
    yh = ((z_hat.float() * w).sum(1, keepdim=True) + b).clamp(0.0, 1.0)
    ys = ((z_src.float() * w).sum(1, keepdim=True) + b).clamp(0.0, 1.0)
    lcn_h = local_contrast_norm(F.avg_pool2d(yh, 2), kernel=5, eps=0.03)
    lcn_s = local_contrast_norm(F.avg_pool2d(ys, 2), kernel=5, eps=0.03)
    return F.smooth_l1_loss(lcn_h, lcn_s.detach(), beta=0.05)


def differentiable_euler_rollout(
    model: nn.Module,
    z_init: torch.Tensor,
    y_in: torch.Tensor,
    q_in: torch.Tensor,
    tid: torch.Tensor,
    u_core: torch.Tensor,
    lscale: torch.Tensor,
    gamma: torch.Tensor,
    *,
    s16_part: torch.Tensor | None = None,
    steps: int = 4,
    device: str = "cuda",
) -> torch.Tensor:
    z = z_init
    dt = 1.0 / float(steps)
    B = z.shape[0]
    for i in range(steps):
        tau = torch.full((B,), i * dt, device=z.device, dtype=torch.float32)

        def _fn(z_in, t_in):
            with autocast(device):
                if s16_part is not None:
                    return model(z_in, y_in, q_in, tid, u_core, t_in,
                                 s16_part=s16_part, layout_scale=lscale, gamma=gamma).float()
                return model(z_in, y_in, q_in, tid, u_core, t_in,
                             layout_scale=lscale, gamma=gamma).float()

        v = checkpoint(_fn, z, tau, use_reentrant=False)
        z = z + dt * v
    return z


@torch.no_grad()
def eval_model_750(
    model: nn.Module,
    part_bank: PartStyleBank | None,
    arm_tag: str,
    bundle: dict,
    c21: dict,
    bank: dict,
    priors: dict,
    dec,
    *,
    is_wrong: bool = False,
    same_bypass: bool = True,
    device: str = "cuda",
) -> dict:
    m_dir = OUT / "measured" / arm_tag
    if (m_dir / "summary_v27m.json").exists():
        return run_measure_v27m(arm_tag, device=device)

    tasks, z_src_750, _ = tasks_and_sources()
    y16_750, d16_750 = get_test750_y16_and_d16(bundle)
    q32_750 = c21["test750"]["q32"]
    eps_750 = torch.randn(z_src_750.shape, generator=torch.Generator().manual_seed(EVAL_SEED))
    sid_750 = torch.tensor([t["source_style_id"] for t in tasks])
    tid_750 = torch.tensor([t["target_style_id"] for t in tasks])
    same_750 = sid_750 == tid_750
    tid_wrong_750, _ = build_wrong_task_index(tasks, sid_750, tid_750)

    model.eval()
    wr = RawWriterV27M(arm_tag, dec=dec, device=device)
    for st in range(0, 750, 25):
        sl = slice(st, min(st + 25, 750))
        zs = z_src_750[sl].to(device).float()
        ep = eps_750[sl].to(device).float()
        y16b = y16_750[sl].to(device).float()
        d16b = d16_750[sl].to(device).float()
        q32b = q32_750[sl].to(device).float()
        tid_u = (tid_wrong_750[sl] if is_wrong else tid_750[sl]).to(device)
        sm = (torch.zeros_like(same_750[sl]) if is_wrong else same_750[sl]).to(device)

        y_in, q_in, u_core, lscale = prepare_q32_cond_inputs(y16b, d16b, q32b, tid_u, bank, lscale_abs=0.15)
        z0_perp = apply_latent_chroma_shift(zs, tid_u, priors, alpha=1.0)
        z0_perp = torch.where(sm.view(-1, 1, 1, 1), zs, z0_perp)
        z = (1.0 - ALPHA_BRIDGE) * ep + ALPHA_BRIDGE * z0_perp
        s16_part = None
        if part_bank is not None:
            s16_part, _ = part_bank.lookup_class(q32b, tid_u)

        gamma = torch.ones(len(zs), device=device)
        dt = 0.25
        for i in range(4):
            tau = torch.full((len(zs),), i * dt, device=device)
            with autocast(device):
                if s16_part is not None:
                    v = model(z, y_in, q_in, tid_u, u_core, tau, s16_part=s16_part, layout_scale=lscale, gamma=gamma).float()
                else:
                    v = model(z, y_in, q_in, tid_u, u_core, tau, layout_scale=lscale, gamma=gamma).float()
            z = z + dt * v

        if same_bypass:
            z = torch.where(same_750[sl].view(-1, 1, 1, 1).to(device), zs, z)
        wr.write_batch(z, tasks[sl])
    wr.finalize(arm=arm_tag, is_wrong=is_wrong, same_bypass=same_bypass)
    return run_measure_v27m(arm_tag, device=device)


@torch.no_grad()
def measure_part_texture_differentiation(
    arm_tag: str, q32_750: torch.Tensor, part_bank: PartStyleBank, device: str = "cuda"
) -> dict:
    """Measure inter-part texture differentiation (ANOVA R^2 on VGG relu2_2 across M=32 Q32 parts)."""
    raw_dir = OUT / "raw" / arm_tag
    if not raw_dir.exists():
        return {}
    tasks, _, _ = tasks_and_sources()
    sid_750 = torch.tensor([t["source_style_id"] for t in tasks])
    tid_750 = torch.tensor([t["target_style_id"] for t in tasks])
    cross_mask = (sid_750 != tid_750).numpy()

    vgg = FrozenVGG16Rollout().to(device).eval()
    import glob
    all_r2 = []
    idx_off = 0
    for bf in sorted(glob.glob(str(raw_dir / "batch_*.pt"))):
        d = torch.load(bf, map_location="cpu", weights_only=False)
        rgb = d["rgb_u8"].to(device).float() / 255.0
        B = len(rgb)
        q32b = q32_750[idx_off:idx_off + B].to(device).float()
        _, f2, _, _ = vgg.forward_style(rgb)                                      # (B, 128, 128, 128)
        f2_16 = F.adaptive_avg_pool2d(f2, 16).flatten(2).transpose(1, 2)       # (B, 256, 128)
        pi = part_bank.assign(q32b).argmax(dim=-1)                             # (B, 256)
        for bi in range(B):
            if not cross_mask[idx_off + bi]:
                continue
            tok = f2_16[bi]
            lbl = pi[bi]
            tot_var = (tok - tok.mean(0, keepdim=True)).square().sum()
            between = torch.tensor(0.0, device=device)
            mu_glob = tok.mean(0)
            for m in lbl.unique():
                mm = lbl == m
                if mm.sum() >= 2:
                    between += mm.sum().float() * (tok[mm].mean(0) - mu_glob).square().sum()
            all_r2.append(float((between / tot_var.clamp_min(1e-6)).item()))
        idx_off += B
    del vgg
    torch.cuda.empty_cache()
    return {"part_anova_r2_mean": round(sum(all_r2) / max(len(all_r2), 1), 4), "n_cross": len(all_r2)}


def train_stage2_arm(
    arm: str,
    base_ckpt: str,
    bundle: dict,
    c21: dict,
    bank: dict,
    tcache: dict,
    pbank_dict: dict,
    updates: int = 2500,
    eval_every: int = 500,
    warmup_steps: int = 500,
    seed_offset: int = 0,
    tag_prefix: str = "v2",
    device: str = "cuda",
) -> dict:
    use_part_arch = arm in ("P2_Part_Glob", "P2_Part_Closed")
    use_part_loss = arm == "P2_Part_Closed"
    seed_all(20268200 + (0 if arm == "P2_Rollout_Glob" else 10 if arm == "P2_Part_Glob" else 20) + seed_offset)

    is_24gb = torch.cuda.get_device_properties(0).total_memory > 18e9
    micro = 3 if is_24gb else 2
    accum = 6 if is_24gb else 9

    tr = bundle["v18_cache"]["train"]
    priors = {k: v.to(device) if torch.is_tensor(v) else v for k, v in bundle["fnd"]["priors"].items()}
    w_luma = bundle["v18_cache"]["luma_proj"]["w"].to(device).float()
    b_luma = bundle["v18_cache"]["luma_proj"]["b"].to(device).float()
    z_tr = tr["z"]
    d16_tr = tr["dino_16"]
    q32_tr = c21["train"]["q32"]
    y16_tr = coarse_y16_lcn(tr["y_64"].float()).half()

    src_idx_all = tcache["src_idx"].long()
    tid_all = tcache["tid"].long()
    is_same_all = tcache["is_same"].bool()
    z_hiS_all = tcache["z_hiS"]
    eps_all = tcache["eps"]
    cross_pool = (~is_same_all).nonzero(as_tuple=True)[0]
    same_pool = is_same_all.nonzero(as_tuple=True)[0]

    dec_fp16 = load_s10_decoder(device)
    dec_bf16 = copy.deepcopy(dec_fp16).to(device=device, dtype=torch.bfloat16).eval()
    for p in dec_bf16.parameters():
        p.requires_grad_(False)
    vgg = FrozenVGG16Rollout().to(device).eval()

    part_bank = PartStyleBank(pbank_dict["centers"], pbank_dict["bank"]).to(device).eval()

    # Pre-extract 5x8=40 submode exemplar features, Oklab 27D, Haar ratios & Q32 part assignments once
    with torch.no_grad():
        ex_z = bank["sub_exemplar_z"].reshape(40, 4, 64, 64).to(device).float()
        ex_rgb = decode_s10(dec_fp16, ex_z, batch=8, output_cpu=False).clamp(0.0, 1.0)
        ex_f1, ex_f2, ex_f3, ex_f4 = vgg.forward_style(ex_rgb)
        ex_ok = extract_color_fiber_27d(ex_rgb)
        ex_haar = haar_hf_lf_ratio_2level(ex_rgb)
        ex_f1 = ex_f1.view(5, bank["k_sub"], *ex_f1.shape[1:])
        ex_f2 = ex_f2.view(5, bank["k_sub"], *ex_f2.shape[1:])
        ex_f3 = ex_f3.view(5, bank["k_sub"], *ex_f3.shape[1:])
        ex_f4 = ex_f4.view(5, bank["k_sub"], *ex_f4.shape[1:])
        ex_ok = ex_ok.view(5, bank["k_sub"], 27)
        ex_haar = ex_haar.view(5, bank["k_sub"])

        # Match each of the 40 exemplars back to its source painting in train_4400 to get its exact Q32!
        # Find nearest z in z_tr for exact Q32 of sub_exemplar_z
        ex_flat_v = ex_z.view(40, -1).cpu()
        z_flat_v = z_tr.float().view(len(z_tr), -1)
        ex_src_idx = (torch.cdist(ex_flat_v, z_flat_v)).argmin(dim=1)  # (40,)
        ex_q32 = q32_tr[ex_src_idx].to(device).float()
        ex_pi = part_bank.assign(ex_q32).view(5, bank["k_sub"], 256, 32)

    base_m = SLFUNetGamma(base_dim=144, u_dim=128).to(device)
    base_m.load_state_dict(torch.load(base_ckpt, map_location="cpu", weights_only=False)["ema"], strict=True)
    if use_part_arch:
        model: nn.Module = SLFUNetPart(base_m, part_in_dim=102, part_dim=64).to(device)
    else:
        model = base_m

    # Freeze dino_proj + enc64 + enc32 (+ down32/in_conv) to protect quotient & encoder path
    for mod in (model.dino_proj, model.in_conv, model.enc64, model.down32, model.enc32):
        for p in mod.parameters():
            p.requires_grad_(False)

    part_params = model.part_parameters() if use_part_arch else []
    part_ids = {id(p) for p in part_params}
    backbone_params = [p for p in model.parameters() if p.requires_grad and id(p) not in part_ids]
    all_trainable = part_params + backbone_params

    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)

    core_cls = [ci.long() for ci in bank["core_idx_per_style"]]

    def _compute_losses(step_idx: int, is_calib: bool = False):
        n_same = int((torch.rand(micro) < 0.15).sum().item())
        if step_idx == 0 and n_same == 0:
            n_same = 1
        n_cross = micro - n_same
        ic = cross_pool[torch.randint(0, len(cross_pool), (n_cross,))]
        idx = torch.cat([ic, same_pool[torch.randint(0, len(same_pool), (n_same,))]], dim=0) if n_same > 0 else ic

        si = src_idx_all[idx]
        ti = tid_all[idx].to(device, non_blocking=True)
        same_b = is_same_all[idx].to(device, non_blocking=True)
        B = len(idx)

        zs = z_tr[si].to(device, non_blocking=True).float()
        z_his = z_hiS_all[idx].to(device, non_blocking=True).float()
        ep = eps_all[idx].to(device, non_blocking=True).float()
        if not is_calib:
            fresh = torch.rand(B, 1, 1, 1, device=device) < 0.25
            ep = torch.where(fresh, torch.randn_like(ep), ep)

        z0_perp = apply_latent_chroma_shift(zs, ti, priors, alpha=1.0)
        z0_perp = torch.where(same_b.view(B, 1, 1, 1), zs, z0_perp)
        z_init = (1.0 - ALPHA_BRIDGE) * ep + ALPHA_BRIDGE * z0_perp

        y16b = y16_tr[si].to(device, non_blocking=True).float()
        d16b = d16_tr[si].to(device, non_blocking=True).float()
        q32b = q32_tr[si].to(device, non_blocking=True).float()
        y_in, q_in, u_core, lscale = prepare_q32_cond_inputs(y16b, d16b, q32b, ti, bank, lscale_abs=0.15)
        gamma = torch.ones(B, device=device)

        s16_part = None
        if use_part_arch:
            s16_part, pi_src = part_bank.lookup_class(q32b, ti)
        else:
            pi_src = part_bank.assign(q32b)

        with torch.no_grad():
            flat_c, _ = match_scene_submode(d16b, ti, bank)
            best_c = flat_c % int(bank["k_sub"])
            x_s = decode_s10(dec_fp16, zs, batch=B, output_cpu=False).clamp(0.0, 1.0)
            s_f1, s_f2, s_f3, s_f4 = vgg.forward_style(x_s)
            s_ok = extract_color_fiber_27d(x_s)
            s_r42 = vgg.forward_gray_r42(x_s)
            s_haar = haar_hf_lf_ratio_2level(x_s)

            # 50% scene-matched submode exemplar + 50% diverse Core-kNN painting (180 per style)
            if step_idx % 2 == 0:
                r_f1 = torch.where(same_b.view(B, 1, 1, 1), s_f1, ex_f1[ti, best_c])
                r_f2 = torch.where(same_b.view(B, 1, 1, 1), s_f2, ex_f2[ti, best_c])
                r_f3 = torch.where(same_b.view(B, 1, 1, 1), s_f3, ex_f3[ti, best_c])
                r_f4 = torch.where(same_b.view(B, 1, 1, 1), s_f4, ex_f4[ti, best_c])
                r_ok = torch.where(same_b.view(B, 1), s_ok, ex_ok[ti, best_c])
                r_haar = torch.where(same_b, s_haar, ex_haar[ti, best_c])
                pi_ref = torch.where(same_b.view(B, 1, 1), pi_src, ex_pi[ti, best_c])
            else:
                r_idx = torch.tensor(
                    [int(core_cls[int(t_i)][torch.randint(0, len(core_cls[int(t_i)]), (1,)).item()].item()) for t_i in ti.cpu()],
                    dtype=torch.long,
                )
                zr = z_tr[r_idx].to(device, non_blocking=True).float()
                xr = decode_s10(dec_fp16, zr, batch=B, output_cpu=False).clamp(0.0, 1.0)
                rf1, rf2, rf3, rf4 = vgg.forward_style(xr)
                rok = extract_color_fiber_27d(xr)
                rh = haar_hf_lf_ratio_2level(xr)
                rpi = part_bank.assign(q32_tr[r_idx].to(device, non_blocking=True).float())
                r_f1 = torch.where(same_b.view(B, 1, 1, 1), s_f1, rf1)
                r_f2 = torch.where(same_b.view(B, 1, 1, 1), s_f2, rf2)
                r_f3 = torch.where(same_b.view(B, 1, 1, 1), s_f3, rf3)
                r_f4 = torch.where(same_b.view(B, 1, 1, 1), s_f4, rf4)
                r_ok = torch.where(same_b.view(B, 1), s_ok, rok)
                r_haar = torch.where(same_b, s_haar, rh)
                pi_ref = torch.where(same_b.view(B, 1, 1), pi_src, rpi)

        z_hat = differentiable_euler_rollout(
            model, z_init, y_in, q_in, ti, u_core, lscale, gamma,
            s16_part=s16_part, steps=4, device=device,
        )
        x_hat = checkpoint(
            lambda zz: decode_s10(dec_bf16, zz, batch=B, dtype=torch.bfloat16, enable_grad=True, output_cpu=False),
            z_hat,
            use_reentrant=False,
        ).clamp(0.0, 1.0)

        h_f1, h_f2, h_f3, h_f4 = vgg.forward_style(x_hat)
        h_r42 = vgg.forward_gray_r42(x_hat)
        h_ok = extract_color_fiber_27d(x_hat)

        l_swd = (0.7 * dynamic_swd(h_f1, r_f1, 64)
                 + 1.0 * dynamic_swd(h_f2, r_f2, 64)
                 + 1.1 * dynamic_swd(h_f3, r_f3, 64)
                 + 1.2 * dynamic_swd(h_f4, r_f4, 64))
        l_deep = vgg_moment_and_gram_loss(h_f3, r_f3) + 1.2 * vgg_moment_and_gram_loss(h_f4, r_f4)
        l_ok = F.smooth_l1_loss(h_ok, r_ok.detach(), beta=0.05)
        l_glob = l_swd + 0.35 * l_deep + 4.0 * l_ok

        l_haar = (haar_hf_lf_ratio_2level(x_hat) - r_haar).abs().mean()
        l_struct = 0.7 * gray_self_sim_loss(h_r42, s_r42) + 0.3 * latent_y32_lcn_loss(z_hat, zs, w_luma, b_luma)

        # Anchor reg: Dir 1 pseudo-huber at random tau
        tau_r = torch.sigmoid(torch.randn(B, device=device) * 0.85).clamp(0.0, 0.995)
        t4_r = tau_r.view(B, 1, 1, 1)
        z_star_r = torch.where(same_b.view(B, 1, 1, 1), zs, z_his)
        z_tau_r = (1.0 - t4_r) * z_init + t4_r * z_star_r
        v_tgt_r = z_star_r - z_init
        with autocast(device):
            if s16_part is not None:
                v_pred_r = model(z_tau_r, y_in, q_in, ti, u_core, tau_r, s16_part=s16_part, layout_scale=lscale, gamma=gamma)
            else:
                v_pred_r = model(z_tau_r, y_in, q_in, ti, u_core, tau_r, layout_scale=lscale, gamma=gamma)
        l_reg = (torch.sqrt((v_pred_r.float() - v_tgt_r).square().mean(dim=(1, 2, 3)) + 0.01) - 0.1).mean()

        if use_part_loss:
            gamma_part = part_transport_plan(pi_src, pi_ref, kappa=4.0)
            l_part = 0.6 * part_barycentric_swd(h_f2, r_f2, gamma_part, grid=16, n_proj=64) \
                   + 1.0 * part_barycentric_swd(h_f3, r_f3, gamma_part, grid=16, n_proj=64)
            return l_glob, l_part, l_haar, l_struct, l_reg
        return l_glob, l_haar, l_struct, l_reg

    # ---- 50-step Auto-Gradient Norm Calibration ----
    print(f"[{arm}] Running 50-step gradient norm calibration (micro={micro}, accum={accum})...", flush=True)
    n_losses = 5 if use_part_loss else 4
    norms = torch.zeros(n_losses, device=device)
    for k_cal in range(50):
        ls = _compute_losses(k_cal, is_calib=True)
        for j, lv in enumerate(ls):
            grads = torch.autograd.grad(lv, all_trainable, retain_graph=(j < n_losses - 1), allow_unused=True)
            gn = torch.sqrt(sum((g.float().square().sum() for g in grads if g is not None), torch.tensor(0.0, device=device)))
            norms[j] += gn.detach() / 50.0
    norms = norms.clamp_min(1e-6)
    target_ratios = (
        torch.tensor([0.85, 0.55, 0.20, 0.30, 0.20], device=device)
        if use_part_loss else
        torch.tensor([1.00, 0.20, 0.30, 0.20], device=device)
    )
    w_loss = (target_ratios / norms) * norms[0]
    peak_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    print(f"[{arm}] Calib grad norms={[round(float(x), 4) for x in norms]} → weights={[round(float(x), 4) for x in w_loss]} (peak_vram={peak_gb:.2f} GB)", flush=True)

    param_groups = []
    if use_part_arch:
        param_groups.append({"params": part_params, "lr": 2.0e-4, "name": "part"})
    param_groups.append({"params": backbone_params, "lr": 0.0 if (use_part_arch and warmup_steps > 0) else 2.5e-5, "name": "backbone"})
    opt = torch.optim.AdamW(param_groups, betas=(0.9, 0.99), weight_decay=1e-4)

    out_log = {"arm": arm, "base_ckpt": base_ckpt, "seed_offset": seed_offset,
               "calib_norms": [round(float(x), 4) for x in norms],
               "calib_weights": [round(float(x), 4) for x in w_loss], "checkpoints": {}}
    t0 = time.time()
    q32_750 = c21["test750"]["q32"]

    for step in range(1, updates + 1):
        # Staged unfreezing schedule
        if use_part_arch and step <= warmup_steps:
            lr_part = 2.0e-4
            lr_back = 0.0
        else:
            eff_step = step - (warmup_steps if use_part_arch else 0)
            eff_tot = max(1, updates - (warmup_steps if use_part_arch else 0))
            prog = eff_step / float(eff_tot)
            lr_back = 3.0e-6 + 0.5 * (2.5e-5 - 3.0e-6) * (1.0 + math.cos(math.pi * prog))
            lr_part = 1.0e-5 + 0.5 * (1.0e-4 - 1.0e-5) * (1.0 + math.cos(math.pi * prog))
        for pg in opt.param_groups:
            pg["lr"] = lr_part if pg["name"] == "part" else lr_back

        opt.zero_grad(set_to_none=True)
        l_acc = [0.0] * n_losses
        for _ in range(accum):
            ls = _compute_losses(step, is_calib=False)
            total = sum(w_loss[j] * ls[j] for j in range(n_losses)) / accum
            total.backward()
            for j in range(n_losses):
                l_acc[j] += float(ls[j].item()) / accum

        nn.utils.clip_grad_norm_(all_trainable, 1.0)
        opt.step()
        with torch.no_grad():
            for pe, pw in zip(ema.parameters(), model.parameters()):
                pe.mul_(0.995).add_(pw, alpha=0.005)

        if step == 1 or step % 100 == 0:
            l_str = " ".join(f"L{j}={l_acc[j]:.4f}" for j in range(n_losses))
            print(f"[{arm}] {step:4d}/{updates} {l_str} ({(time.time()-t0)/60:.1f}m)", flush=True)

        if step % eval_every == 0 or step == updates:
            suf = f"_seed{seed_offset}" if seed_offset else ""
            px = f"{tag_prefix}_" if tag_prefix else ""
            ck_tag = f"{px}{arm}{suf}_s{step}"
            torch.save({"ema": ema.state_dict(), "step": step, "arm": arm}, OUT / f"{ck_tag}.pt")
            res_s = eval_model_750(
                ema, part_bank if use_part_arch else None, ck_tag, bundle, c21, bank, priors, dec_fp16, device=device,
            )
            res_s["part_diff"] = measure_part_texture_differentiation(ck_tag, q32_750, part_bank, device=device)
            out_log["checkpoints"][str(step)] = res_s
            save_json(OUT / f"stage2_{px}{arm}{suf}.json", out_log)

    suf = f"_seed{seed_offset}" if seed_offset else ""
    px = f"{tag_prefix}_" if tag_prefix else ""
    out_log["wrong_id"] = eval_model_750(
        ema, part_bank if use_part_arch else None, f"{px}{arm}{suf}_s{updates}_wrong",
        bundle, c21, bank, priors, dec_fp16, is_wrong=True, device=device,
    )
    save_json(OUT / f"stage2_{px}{arm}{suf}.json", out_log)
    return out_log


@torch.no_grad()
def eval_1shot_part_bonus(
    ckpt_2c: str,
    bundle: dict,
    c21: dict,
    bank: dict,
    pbank_dict: dict,
    device: str = "cuda",
) -> dict:
    """Stage 3: Zero-training 1-shot extrapolation using 2c checkpoint + lookup_exemplar(kappa=8.0)."""
    tasks, z_src_750, _ = tasks_and_sources()
    y16_750, d16_750 = get_test750_y16_and_d16(bundle)
    q32_750 = c21["test750"]["q32"]
    eps_750 = torch.randn(z_src_750.shape, generator=torch.Generator().manual_seed(EVAL_SEED))
    priors = {k: v.to(device) if torch.is_tensor(v) else v for k, v in bundle["fnd"]["priors"].items()}
    dec = load_s10_decoder(device)
    dino = load_dino_model(device=device)

    base_m = SLFUNetGamma(base_dim=144, u_dim=128).to(device)
    model = SLFUNetPart(base_m, part_in_dim=102, part_dim=64).to(device).eval()
    model.load_state_dict(torch.load(ckpt_2c, map_location="cpu", weights_only=False)["ema"], strict=True)
    part_bank = PartStyleBank(pbank_dict["centers"], pbank_dict["bank"]).to(device).eval()

    fnd = bundle["fnd"]
    from exp.v19.fiber_duality import StyleFiberEncoder
    fiber_enc = StyleFiberEncoder(in_dim=fnd["tr_fib"].shape[1], dim=128).to(device).eval()
    fiber_enc.load_state_dict(fnd["fiber_state"])

    # Build 5-style 1-shot exemplar banks for BOTH SG-5 and Holdout-single using pre-cached c21["sg5"] & c21["holdout"]
    ho = bundle["v18_cache"]["holdout"]
    ho_idx = c21["holdout"]["single_idx"].long()
    ho_z = ho["z"][ho_idx].to(device).float()
    ho_q32 = c21["holdout"]["q32"][ho_idx].to(device).float()
    ho_u = fiber_enc.encode(fnd["ho_fib"][ho_idx].to(device).float())

    sg5_z = c21["sg5"]["z"].to(device).float()
    sg5_q32 = c21["sg5"]["q32"].to(device).float()
    sg5_u = c21["sg5"]["u"].to(device).float()

    def _ref_vtok(z_ref: torch.Tensor) -> torch.Tensor:
        rgb_ref = decode_s10(dec, z_ref, batch=5, output_cpu=False).clamp(0.0, 1.0)
        raw102 = extract_patch_102d(rgb_ref, z_ref, dino, pbank_dict["pca_basis"], pbank_dict["pca_mean"], device=device)
        return (raw102 - pbank_dict["v_mean"].to(device)) / pbank_dict["v_std"].to(device)

    sg5_vtok = _ref_vtok(sg5_z)
    ho_vtok = _ref_vtok(ho_z)

    sid_750 = torch.tensor([t["source_style_id"] for t in tasks])
    tid_750 = torch.tensor([t["target_style_id"] for t in tasks])
    same_750 = sid_750 == tid_750

    out_1s = {"ckpt": ckpt_2c, "arms": {}}
    for arm_tag, rq32, rvtok, ru in (
        ("P3_1Shot_SG5_PartBank", sg5_q32, sg5_vtok, sg5_u),
        ("P3_1Shot_Holdout_PartBank", ho_q32, ho_vtok, ho_u),
    ):
        if not (OUT / "measured" / arm_tag / "summary_v27m.json").exists():
            wr = RawWriterV27M(arm_tag, dec=dec, device=device)
            for st in range(0, 750, 25):
                sl = slice(st, min(st + 25, 750))
                zs = z_src_750[sl].to(device).float()
                ep = eps_750[sl].to(device).float()
                y16b = y16_750[sl].to(device).float()
                d16b = d16_750[sl].to(device).float()
                q32b = q32_750[sl].to(device).float()
                ti = tid_750[sl].to(device)
                sm = same_750[sl].to(device)

                y_in, q_in, _, lscale = prepare_q32_cond_inputs(y16b, d16b, q32b, ti, bank, lscale_abs=0.15)
                u_ref = ru[ti]
                s16_part, _ = part_bank.lookup_exemplar(q32b, rq32[ti], rvtok[ti], kappa=8.0)

                z0_perp = apply_latent_chroma_shift(zs, ti, priors, alpha=1.0)
                z0_perp = torch.where(sm.view(-1, 1, 1, 1), zs, z0_perp)
                z = (1.0 - ALPHA_BRIDGE) * ep + ALPHA_BRIDGE * z0_perp
                gamma = torch.ones(len(zs), device=device)
                for i in range(4):
                    tau = torch.full((len(zs),), i * 0.25, device=device)
                    with autocast(device):
                        v = model(z, y_in, q_in, ti, u_ref, tau, s16_part=s16_part, layout_scale=lscale, gamma=gamma).float()
                    z = z + 0.25 * v
                wr.write_batch(torch.where(sm.view(-1, 1, 1, 1), zs, z), tasks[sl])
            wr.finalize(arm=arm_tag)
        out_1s["arms"][arm_tag] = run_measure_v27m(arm_tag, device=device)
    save_json(OUT / "stage3_oneshot.json", out_1s)
    return out_1s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--base_ckpt", default=str(OUT / "d1_reflow_v2.pt"))
    ap.add_argument("--teacher_anchor", default="p1", choices=["p1", "b4", "chroma"])
    ap.add_argument("--arm", default="P2_Rollout_Glob", choices=["P2_Rollout_Glob", "P2_Part_Glob", "P2_Part_Closed", "P3_1Shot"])
    ap.add_argument("--updates", type=int, default=2500)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--warmup_steps", type=int, default=500)
    ap.add_argument("--seed_offset", type=int, default=0)
    ap.add_argument("--ckpt_2c", default=str(OUT / "P2_Part_Closed_s2500.pt"))
    ap.add_argument("--tag_prefix", default="v2")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    sanitize_cuda_env()
    OUT.mkdir(parents=True, exist_ok=True)

    pack = build_or_load_v21_cache(args.pool, device=args.device)
    bundle, c21 = pack["bundle"], pack["c21"]
    bank = build_or_load_core_knn_bank(bundle, k_sub=8, core_per_style=180, device=args.device)
    pbank_dict = build_or_load_part_bank(args.pool, device=args.device)

    if args.arm == "P3_1Shot":
        eval_1shot_part_bonus(args.ckpt_2c, bundle, c21, bank, pbank_dict, device=args.device)
        return

    tcache = build_or_load_dir1_teacher_19800(bundle, c21, bank, teacher_anchor=args.teacher_anchor, device=args.device)
    train_stage2_arm(
        args.arm, args.base_ckpt, bundle, c21, bank, tcache, pbank_dict,
        updates=args.updates, eval_every=args.eval_every, warmup_steps=args.warmup_steps,
        seed_offset=args.seed_offset, tag_prefix=args.tag_prefix, device=args.device,
    )
    print(f"[{args.arm}] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
