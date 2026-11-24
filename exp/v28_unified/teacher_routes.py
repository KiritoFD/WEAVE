"""Phase 1 & 2 Teacher Routes: T1 (Multi-Expert 3-Tier), T2 (Zero-Legacy Opt), T3 (In-Domain Reversed), T4 (Partial-FGW Gap-Filler)."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from exp.v18.structure import local_contrast_norm
from exp.v19.aniso_teacher import (
    AnisotropicS10Teacher,
    _smooth_grad_3x3,
    aniso_rms_direction,
    latent_lcn_cost,
    project_anisotropic_trust,
)
from exp.v19.fiber_duality import (
    BalancedVGGCriticHead,
    UnnormalizedVGGFeatureExtractor,
    apply_latent_chroma_shift,
)
from exp.v20.slf_flow import SLFUNet, coarse_y16_lcn
from exp.v20.submode_patchswd import MultiScalePatchSWDExtractor, match_scene_submode, sliced_wasserstein_1d
from exp.v21.stiefel_ref_cache import stiefel_polar_quotient
from exp.v22.core import D2_SOLVER, D5_SOLVER, CFGConfig, GateConfig, PrismCore
from exp.v27.icme_step2_closedset_accel import D5_CFG
from exp.v27.icme_step3_q32_vs_dino16 import make_bottleneck_net
from exp.v27_mainline.dir1_reflow import _integrate_q32_seg
from exp.v27_mainline.followup.partial_fgw import partial_fgw
from exp.v27_mainline.part_bank import PART_CACHE, PartStyleBank, part_barycentric_swd
from exp.v27_mainline.rollout_refine import (
    FrozenVGG16Rollout,
    gray_self_sim_loss,
    haar_hf_lf_ratio_2level,
)
from exp.v28_unified import (
    PEAK1500_CKPT,
    Q32_FLAGSHIP_CKPT,
    V20_D2_CKPT,
    SEED_PLAN,
)
from exp.v28_unified.referee_free import RefereeFreeSelector, extract_color_fiber_27d
from src.utils import decode_s10
from src.utils.runtime import bf16_autocast


class _P1AnchorMixin:
    """Shared P1 + B4 + Canonicalizer anchor helper without loading unused experts."""

    def _init_p1_anchor(self, bundle: dict, device: str) -> None:
        from exp.v19.fiber_duality import StyleFiberEncoder
        from exp.v19.routed_student import SpatiallyRoutedChordNet
        from exp.v28_unified import B4_CKPT, P1_CKPT
        from src.infer import canonicalizer, load_c3unet

        self.device = device
        self.bundle = bundle
        fnd = bundle["fnd"]
        self.priors = {k: v.to(device) if torch.is_tensor(v) else v for k, v in fnd["priors"].items()}
        self.b4, _ = load_c3unet(B4_CKPT, device)
        self.canon = canonicalizer(device)
        self.p1 = SpatiallyRoutedChordNet(width=96, chord_dim=128, use_spatial_router=True).to(device).eval()
        self.p1.load_state_dict(torch.load(P1_CKPT, map_location="cpu", weights_only=False)["model"])
        for p in list(self.b4.parameters()) + list(self.p1.parameters()):
            p.requires_grad_(False)

        fiber_enc = StyleFiberEncoder(in_dim=fnd["tr_fib"].shape[1], dim=128).to(device).eval()
        fiber_enc.load_state_dict(fnd["fiber_state"])
        with torch.no_grad():
            self.u_all = fiber_enc.encode(fnd["tr_fib"].to(device).float()).cpu()
            self.bary = fiber_enc.class_barycenters().detach()
        del fiber_enc

    @torch.no_grad()
    def compute_p1_anchor(
        self, z_src: torch.Tensor, d16: torch.Tensor, c64: torch.Tensor, fib: torch.Tensor | None, tid: torch.Tensor,
        src_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dev = self.device
        zs = z_src.to(dev).float()
        ti = tid.to(dev)
        c_can = self.canon(zs, mode="c1_gentle", augment=False)
        z0_b = self.b4(c_can, 0.0, c_can, ti)
        z_anc = apply_latent_chroma_shift(z0_b, ti, self.priors, alpha=0.20)
        if src_idx is not None:
            u_s = self.u_all[src_idx].to(dev)
        else:
            from exp.v19.fiber_duality import StyleFiberEncoder
            fnd = self.bundle["fnd"]
            fe = StyleFiberEncoder(in_dim=fnd["tr_fib"].shape[1], dim=128).to(dev).eval()
            fe.load_state_dict(fnd["fiber_state"])
            u_s = fe.encode(fib.to(dev).float())
            del fe
        chord = self.bary[ti] - u_s
        z_luma = torch.einsum("ij,bjhw->bihw", self.priors["P_Y"], zs)
        delta = self.p1(
            z_anc, z_luma, c64.to(dev).float(), d16.to(dev).float(),
            chord, same_mask=torch.zeros(len(ti), dtype=torch.bool, device=dev),
        )
        return (z_anc + delta).detach()


class T1MultiExpertTeacher(_P1AnchorMixin):
    """5-expert multi-tier teacher: Peak1500(P1), V20 D2, Q32 e2full_d5, Q32 csj_d6h, V19 S10-VGG."""

    CANDIDATE_NAMES = (
        "q32_csj_d6h",
        "q32_e2full_d5",
        "peak1500_hiC",
        "peak1500_mid",
        "peak1500_hiS",
        "v20d2_hiC",
        "v20d2_hiS",
        "v19_s10vgg_hiC",
        "v19_s10vgg_sweet",
        "v19_s10vgg_strong",
    )

    def __init__(self, bundle: dict, c21: dict, bank: dict, dec_bf16: nn.Module, device: str = "cuda"):
        self._init_p1_anchor(bundle, device)
        self.c21 = c21
        self.bank = bank
        self.dec_bf16 = dec_bf16

        self.aniso = AnisotropicS10Teacher(bundle, device=device)
        # Share bf16 decoder to save VRAM
        del self.aniso.dec_bf16
        self.aniso.dec_bf16 = dec_bf16

        self.q32_flow = make_bottleneck_net(32).to(device).eval().requires_grad_(False)
        self.q32_flow.load_state_dict(torch.load(Q32_FLAGSHIP_CKPT, map_location="cpu", weights_only=False)["ema"])

        self.core_peak = PrismCore(bundle, bank, PEAK1500_CKPT, device=device)
        self.core_d2 = PrismCore(bundle, bank, V20_D2_CKPT, device=device)

    @torch.no_grad()
    def generate_candidates(
        self,
        z_src: torch.Tensor,
        y16: torch.Tensor,
        d16: torch.Tensor,
        q32: torch.Tensor,
        c64: torch.Tensor,
        fib: torch.Tensor,
        tid: torch.Tensor,
        eps: torch.Tensor,
        *,
        src_idx: torch.Tensor | None = None,
        batch: int = 16,
    ) -> list[torch.Tensor]:
        """Return 9 candidate endpoint tensors (each (N, 4, 64, 64) fp16 CPU)."""
        dev = self.device
        N = len(z_src)
        cands = [[] for _ in self.CANDIDATE_NAMES]

        hi_c = dict(D5_CFG, tau_min=0.48, tau_hi=0.78, ws_fig=5.2)
        hi_s = dict(D5_CFG, tau_min=0.30, tau_hi=0.68, ws_fig=5.5, wc_fig=1.65,
                    wc_abs=0.70, wsp_fig_late=3.6, wsp_abs=5.2)

        peak_hiC = dict(D5_SOLVER, tau_min=0.35, tau_hi=0.70, steps=6, wc_fig=1.40, wc_abs=1.40)
        peak_mid = dict(D5_SOLVER, tau_min=0.28, tau_hi=0.66, steps=6, wc_fig=1.15, wc_abs=1.15)
        peak_hiS = dict(D5_SOLVER, tau_min=0.28, tau_hi=0.55, steps=6, wc_fig=1.00, wc_abs=1.00)

        d2_hiC = dict(D2_SOLVER, tau_min=0.10, tau_hi=0.50, steps=6, wc_fig=1.60, wc_abs=0.80)
        d2_hiS = dict(D5_SOLVER, tau_min=0.25, tau_hi=0.62, steps=6, wc_fig=1.10, wc_abs=0.70)

        g_p45 = GateConfig(mode="fixed", fixed_g=0.45)
        g_p42 = GateConfig(mode="fixed", fixed_g=0.42)
        cfg_full = CFGConfig(branches="full", rescale_phi=0.48)

        for st in range(0, N, batch):
            sl = slice(st, min(st + batch, N))
            zs = z_src[sl].to(dev).float()
            yb = y16[sl].to(dev).float()
            db = d16[sl].to(dev).float()
            qb = q32[sl].to(dev).float()
            cb = c64[sl].to(dev).float()
            fb = fib[sl].to(dev).float()
            ti = tid[sl].to(dev)
            ep = eps[sl].to(dev).float()
            s_idx = src_idx[sl] if src_idx is not None else None

            anc = self.compute_p1_anchor(zs, db, cb, fb, ti, src_idx=s_idx)

            # 0, 1: Q32 csj_d6h & e2full_d5
            cands[0].append(_integrate_q32_seg(self.q32_flow, anc, ep, yb, db, qb, ti, self.bank,
                                               steps=6, full_steps=set(), cfg_kw=hi_c).cpu().half())
            cands[1].append(_integrate_q32_seg(self.q32_flow, anc, ep, yb, db, qb, ti, self.bank,
                                               steps=6, full_steps={0, 1}, cfg_kw=hi_s).cpu().half())

            # 2, 3, 4: Peak1500 (P1 anchor)
            z_p_hic, _ = self.core_peak.integrate_flow(ep, anc, yb, db, ti, gate=g_p45, cfg=cfg_full, solver=peak_hiC)
            z_p_mid, _ = self.core_peak.integrate_flow(ep, anc, yb, db, ti, gate=g_p45, cfg=cfg_full, solver=peak_mid)
            z_p_his, _ = self.core_peak.integrate_flow(ep, anc, yb, db, ti, gate=g_p42, cfg=cfg_full, solver=peak_hiS)
            cands[2].append(z_p_hic.cpu().half())
            cands[3].append(z_p_mid.cpu().half())
            cands[4].append(z_p_his.cpu().half())

            # 5, 6: V20 D2 (P1 anchor)
            z_d2_hic, _ = self.core_d2.integrate_flow(ep, anc, yb, db, ti, gate=GateConfig(mode="original"), cfg=cfg_full, solver=d2_hiC)
            z_d2_his, _ = self.core_d2.integrate_flow(ep, anc, yb, db, ti, gate=g_p45, cfg=cfg_full, solver=d2_hiS)
            cands[5].append(z_d2_hic.cpu().half())
            cands[6].append(z_d2_his.cpu().half())

            # 7, 8, 9: V19 S10-VGG solver (high-content, sweet, strong)
            z_v19_hc = self.aniso.solve_batch(
                zs, zs, zs, fb, ti,
                mode="s10_vgg", alpha_chroma=0.18, m_steps=1,
                rho_y_edge=0.050, rho_y_low=0.090, rho_c=0.20, lcn_budget=0.020, etas=(0.05, 0.09),
            )
            z_v19_sw = self.aniso.solve_batch(
                anc, anc, zs, fb, ti,
                mode="s10_vgg", alpha_chroma=0.18, m_steps=2,
                rho_y_edge=0.075, rho_y_low=0.14, rho_c=0.26, lcn_budget=0.030, etas=(0.06, 0.12),
            )
            z_v19_st = self.aniso.solve_batch(
                anc, anc, zs, fb, ti,
                mode="s10_vgg", alpha_chroma=0.40, m_steps=2,
                rho_y_edge=0.15, rho_y_low=0.26, rho_c=0.44, lcn_budget=0.062, etas=(0.12, 0.22),
            )
            cands[7].append(z_v19_hc.cpu().half())
            cands[8].append(z_v19_sw.cpu().half())
            cands[9].append(z_v19_st.cpu().half())

        return [torch.cat(chunks, dim=0) for chunks in cands]


class V22CleanTeacher(_P1AnchorMixin):
    """Single function-consistent neural teacher: Peak1500 + P1 anchor, two coherent tiers."""

    PEAK_HIC = dict(D5_SOLVER, tau_min=0.38, tau_hi=0.74, steps=6, ws_fig=3.8, wc_fig=1.80, wc_abs=1.50)
    PEAK_HIS = dict(D5_SOLVER, tau_min=0.28, tau_hi=0.55, steps=6, wc_fig=1.00, wc_abs=1.00)

    def __init__(self, bundle: dict, bank: dict, device: str = "cuda"):
        self._init_p1_anchor(bundle, device)
        self.core_peak = PrismCore(bundle, bank, PEAK1500_CKPT, device=device)

    @torch.no_grad()
    def generate_two_tiers(
        self,
        z_src: torch.Tensor,
        y16: torch.Tensor,
        d16: torch.Tensor,
        c64: torch.Tensor,
        fib: torch.Tensor,
        tid: torch.Tensor,
        eps: torch.Tensor,
        *,
        src_idx: torch.Tensor | None = None,
        batch: int = 32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (hiC_fp16_cpu, hiS_fp16_cpu) with matched P1 anchor and deterministic noise."""
        dev = self.device
        N = len(z_src)
        hi_c_out, hi_s_out = [], []
        g_c = GateConfig(mode="fixed", fixed_g=0.50)
        g_s = GateConfig(mode="fixed", fixed_g=0.42)
        cfg_full = CFGConfig(branches="full", rescale_phi=0.48)

        for st in range(0, N, batch):
            sl = slice(st, min(st + batch, N))
            zs = z_src[sl].to(dev).float()
            yb = y16[sl].to(dev).float()
            db = d16[sl].to(dev).float()
            cb = c64[sl].to(dev).float()
            fb = fib[sl].to(dev).float()
            ti = tid[sl].to(dev)
            ep = eps[sl].to(dev).float()
            s_idx = src_idx[sl] if src_idx is not None else None

            anc = self.compute_p1_anchor(zs, db, cb, fb, ti, src_idx=s_idx)
            z_hic, _ = self.core_peak.integrate_flow(ep, anc, yb, db, ti, gate=g_c, cfg=cfg_full, solver=self.PEAK_HIC)
            z_his, _ = self.core_peak.integrate_flow(ep, anc, yb, db, ti, gate=g_s, cfg=cfg_full, solver=self.PEAK_HIS)
            hi_c_out.append(z_hic.cpu().half())
            hi_s_out.append(z_his.cpu().half())

        return torch.cat(hi_c_out, dim=0), torch.cat(hi_s_out, dim=0)


class T4RegionalGapFiller:
    """Offline Partial-FGW + 15-step S10-BF16 Part-Bary-SWD gap-filler for failed ER/Uki tasks."""

    def __init__(self, bundle: dict, bank: dict, dec_bf16: nn.Module, pbank_dict: dict, device: str = "cuda"):
        self.device = device
        self.dec_bf16 = dec_bf16
        self.priors = {k: v.to(device) if torch.is_tensor(v) else v for k, v in bundle["fnd"]["priors"].items()}
        self.w_luma = bundle["v18_cache"]["luma_proj"]["w"].to(device).float()
        self.b_luma = bundle["v18_cache"]["luma_proj"]["b"].to(device).float()
        self.vgg = FrozenVGG16Rollout().to(device).eval()
        self.part_bank = PartStyleBank(pbank_dict["centers"], pbank_dict["bank"]).to(device).eval()

        dist_grid = np.mgrid[:16, :16].reshape(2, -1).T / 15.0
        self.dist_mat = np.linalg.norm(dist_grid[:, None] - dist_grid[None], axis=-1) / np.sqrt(2.0)

        # Precompute target exemplar VGG f2, f3, Q32, Oklab, Haar
        with torch.no_grad():
            ex_z = bank["sub_exemplar_z"].reshape(40, 4, 64, 64).to(device).float()
            ex_rgb = decode_s10(dec_bf16, ex_z, batch=8, dtype=torch.bfloat16, output_cpu=False).clamp(0.0, 1.0)
            with bf16_autocast(device):
                _, ex_f2, ex_f3, _ = self.vgg.forward_style(ex_rgb)
            self.ex_f2 = ex_f2.float().view(5, 8, *ex_f2.shape[1:])
            self.ex_f3 = ex_f3.float().view(5, 8, *ex_f3.shape[1:])
            self.ex_ok = extract_color_fiber_27d(ex_rgb).view(5, 8, 27)
            self.ex_haar = haar_hf_lf_ratio_2level(ex_rgb).view(5, 8)
            # Recover exact Q32 from train_4400 nearest z
            z_tr = bundle["v18_cache"]["train"]["z"]
            from exp.v21.stiefel_ref_cache import build_or_load_v21_cache
            # Match ex_z to train_4400
            ex_src = torch.cdist(ex_z.cpu().view(40, -1), z_tr.float().view(len(z_tr), -1)).argmin(dim=1)
            self.ex_src_idx = ex_src

    def _compute_partial_fgw_plan(self, pi_s: torch.Tensor, pi_r: torch.Tensor, keep: int = 192) -> torch.Tensor:
        scores = (pi_s @ pi_r.transpose(1, 2)).cpu().numpy()  # (B, 256, 256)
        plans = []
        for aff in scores:
            cost = 1.0 - aff / max(float(aff.max()), 1e-8)
            p = partial_fgw(cost, self.dist_mat, keep=keep, weight=0.6, iterations=8)
            row_sum = p.sum(axis=1, keepdims=True)
            # For unmatched rows, fall back to soft part affinity
            soft = aff / np.maximum(aff.sum(axis=1, keepdims=True), 1e-8)
            p_norm = np.where(row_sum > 1e-6, p / np.maximum(row_sum, 1e-8), soft)
            plans.append(p_norm.astype(np.float32))
        return torch.from_numpy(np.stack(plans, axis=0)).to(self.device)

    def refine_batch(
        self,
        z_init: torch.Tensor,
        z_src: torch.Tensor,
        rgb_src: torch.Tensor,
        q32_src: torch.Tensor,
        q32_tr: torch.Tensor,
        tid: torch.Tensor,
        best_c: torch.Tensor,
        *,
        steps: int = 15,
        step_size: float = 0.055,
    ) -> torch.Tensor:
        """15-step Part-Bary-SWD optimization under anisotropic trust region."""
        dev = self.device
        B = len(z_init)
        if B == 0:
            return z_init
        P_Y, P_C = self.priors["P_Y"], self.priors["P_C"]

        with torch.no_grad():
            ex_idx_flat = tid * 8 + best_c
            q32_ref = q32_tr[self.ex_src_idx[ex_idx_flat.cpu()]].to(dev).float()
            pi_s = self.part_bank.assign(q32_src.to(dev).float())
            pi_r = self.part_bank.assign(q32_ref)
            gamma_fgw = self._compute_partial_fgw_plan(pi_s, pi_r, keep=192)
            with bf16_autocast(dev):
                src_r42 = self.vgg.forward_gray_r42(rgb_src.to(dev))
            r_f2 = self.ex_f2[tid, best_c]
            r_f3 = self.ex_f3[tid, best_c]
            r_ok = self.ex_ok[tid, best_c]
            r_haar = self.ex_haar[tid, best_c]

        z = z_init.to(dev).float().clone()
        z_anc = apply_latent_chroma_shift(z_src.to(dev).float(), tid.to(dev), self.priors, alpha=0.80)

        for _ in range(steps):
            grads = []
            for st in range(0, B, 4):
                sl = slice(st, min(st + 4, B))
                with torch.enable_grad():
                    zr = z[sl].detach().requires_grad_(True)
                    rgb = decode_s10(self.dec_bf16, zr, batch=len(zr), dtype=torch.bfloat16,
                                     enable_grad=True, output_cpu=False).clamp(0.0, 1.0)
                    with bf16_autocast(dev):
                        _, h_f2, h_f3, _ = self.vgg.forward_style(rgb)
                        h_r42 = self.vgg.forward_gray_r42(rgb)
                    l_bary = (
                        0.6 * part_barycentric_swd(h_f2.float(), r_f2[sl], gamma_fgw[sl], grid=16, n_proj=48)
                        + 1.0 * part_barycentric_swd(h_f3.float(), r_f3[sl], gamma_fgw[sl], grid=16, n_proj=48)
                    )
                    l_ok = F.smooth_l1_loss(extract_color_fiber_27d(rgb), r_ok[sl], beta=0.05)
                    l_haar = (haar_hf_lf_ratio_2level(rgb) - r_haar[sl]).abs().mean()
                    l_sim = gray_self_sim_loss(h_r42.float(), src_r42[sl].float())
                    loss = l_bary + 2.5 * l_ok + 0.25 * l_haar + 2.0 * l_sim
                    g_sub = torch.autograd.grad(loss, zr)[0].detach()
                grads.append(g_sub)
            g = _smooth_grad_3x3(torch.cat(grads, dim=0), blend=0.25)
            d = aniso_rms_direction(-g, P_Y, P_C, w_y=0.45, w_c=0.90)
            z = project_anisotropic_trust(
                z + step_size * d, z_anc, P_Y, P_C,
                rho_y_edge=0.14, rho_y_low=0.25, rho_c=0.46,
            )
        return z.detach().cpu().half()


class T2ZeroLegacyOptimizer(nn.Module):
    """T2: Zero-legacy single differentiable optimizer (NO B4/n2c/SLFUNet/FiberEncoder).

    Starts from analytical chroma alignment z0^perp and optimizes through S10-BF16 over 3/6/10 steps.
    """

    def __init__(self, bundle: dict, selector: RefereeFreeSelector, dec_bf16: nn.Module, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.selector = selector
        self.dec_bf16 = dec_bf16
        fnd = bundle["fnd"]
        self.priors = {k: v.to(device) if torch.is_tensor(v) else v for k, v in fnd["priors"].items()}
        self.w_luma = bundle["v18_cache"]["luma_proj"]["w"].to(device).float()
        self.b_luma = bundle["v18_cache"]["luma_proj"]["b"].to(device).float()

        self.vgg_ext = UnnormalizedVGGFeatureExtractor(gram_rank=48).to(device).eval().requires_grad_(False)
        self.vgg_head = BalancedVGGCriticHead(in_dim=fnd["vgg_in_dim"], num_styles=5).to(device).eval()
        self.vgg_head.load_state_dict(fnd["vgg_head_state"])
        for p in self.vgg_head.parameters():
            p.requires_grad_(False)
        self.vgg_protos = fnd["vgg_cls_protos"].to(device).float()

    def optimize_batch(
        self,
        z_src: torch.Tensor,
        rgb_src: torch.Tensor,
        tid: torch.Tensor,
        *,
        total_steps: int = 10,
        snap_steps: tuple[int, ...] = (3, 6, 10),
        step_size: float = 0.088,
    ) -> dict[int, torch.Tensor]:
        """Run 10-step S10-BF16 optimization from analytical z0^perp, returning snapshots at 3/6/10."""
        dev = self.device
        B = len(z_src)
        P_Y, P_C = self.priors["P_Y"], self.priors["P_C"]
        zs = z_src.to(dev).float()
        ti = tid.to(dev)
        z_anc = apply_latent_chroma_shift(zs, ti, self.priors, alpha=0.85)
        z = apply_latent_chroma_shift(zs, ti, self.priors, alpha=0.70)

        with torch.no_grad():
            best_c = self.selector.match_gray_exemplar(rgb_src.to(dev), ti)
            with bf16_autocast(dev):
                src_r42 = self.selector.vgg.forward_gray_r42(rgb_src.to(dev))
                src_vgg_f = self.vgg_ext(rgb_src.to(dev), multi_scale=True)
            src_q = self.vgg_head(src_vgg_f.float(), calibrated=True)
            src_probs = F.softmax(src_q / 1.5, dim=-1)
            # Pool all 8 core exemplars per style for low-variance distribution matching
            r_vgg = [ev.reshape(5, -1, ev.shape[-1])[ti] for ev in self.selector.ex_vgg]
            r_cnx = {k: v.reshape(5, -1, v.shape[-1])[ti] for k, v in self.selector.ex_cnx.items()}
            r_ok = self.selector.ex_oklab[ti, best_c]
            r_haar = self.selector.ex_haar[ti].mean(dim=1)
            oh = F.one_hot(ti, 5).bool()

        # Trust-region schedule widens smoothly from tight (step 3) -> mid (step 6) -> strong (step 10)
        snaps = {}
        vel = torch.zeros_like(z)
        for st_idx in range(1, total_steps + 1):
            frac = st_idx / float(total_steps)
            ry_e = 0.055 + 0.125 * frac
            ry_l = 0.10 + 0.22 * frac
            rc = 0.22 + 0.30 * frac
            w_struct = 1.8 * (1.0 - 0.78 * frac)

            grads = []
            for sub in range(0, B, 8):
                sl = slice(sub, min(sub + 8, B))
                with torch.enable_grad():
                    zr = z[sl].detach().requires_grad_(True)
                    rgb = decode_s10(self.dec_bf16, zr, batch=len(zr), dtype=torch.bfloat16,
                                     enable_grad=True, output_cpu=False).clamp(0.0, 1.0)
                    with bf16_autocast(dev):
                        f1, f2, f3, _ = self.selector.vgg.forward_style(rgb)
                        r42_c = self.selector.vgg.forward_gray_r42(rgb)
                        cnx_c = self.selector.swd_ext.extract_tokens(rgb)
                        vgg_feat = self.vgg_ext(rgb, multi_scale=True)

                    # Orthogonal VGG + ConvNeXt Patch-SWD
                    l_vgg_swd = 0.0
                    for i_l, feat, rt in zip((1, 2, 3), (f1, f2, f3), r_vgg):
                        proj = getattr(self.selector, f"vgg_proj_{i_l}")
                        tok = F.adaptive_avg_pool2d(feat.float(), 16).flatten(2).transpose(1, 2)
                        l_vgg_swd = l_vgg_swd + sliced_wasserstein_1d(tok, rt[sl], proj).mean()
                    l_vgg_swd = l_vgg_swd / 3.0

                    l_cnx_swd = (
                        1.0 * sliced_wasserstein_1d(cnx_c["rgb"].float(), r_cnx["rgb"][sl], self.selector.swd_ext.p_rgb).mean()
                        + 0.4 * sliced_wasserstein_1d(cnx_c["s1"].float(), r_cnx["s1"][sl], self.selector.swd_ext.p_s1).mean()
                        + 0.5 * sliced_wasserstein_1d(cnx_c["s2"].float(), r_cnx["s2"][sl], self.selector.swd_ext.p_s2).mean()
                        + 0.6 * sliced_wasserstein_1d(cnx_c["s3"].float(), r_cnx["s3"][sl], self.selector.swd_ext.p_s3).mean()
                    )

                    l_ok = F.smooth_l1_loss(extract_color_fiber_27d(rgb), r_ok[sl], beta=0.05)
                    l_haar = (haar_hf_lf_ratio_2level(rgb) - r_haar[sl]).abs().mean()
                    l_sim = gray_self_sim_loss(r42_c.float(), src_r42[sl].float())

                    # Non-referee 5-class balanced VGGHead margin + inferred source repulsion
                    q_logits = self.vgg_head(vgg_feat.float(), calibrated=True)
                    others = q_logits.masked_fill(oh[sl], -1e9).max(dim=1).values
                    m_tgt = (q_logits.gather(1, ti[sl, None]).squeeze(1) - others).mean()
                    p_other = src_probs[sl].masked_fill(oh[sl], 0.0)
                    repel = (p_other * q_logits).sum(dim=1).mean()
                    feat_std = ((vgg_feat.float() - self.vgg_head.x_mean) / self.vgg_head.x_std).clamp(-8.0, 8.0)
                    proto_d = (feat_std - self.vgg_protos[ti[sl]]).square().mean(dim=1).sqrt().mean()

                    loss = (
                        6.0 * l_vgg_swd + 5.0 * l_cnx_swd + 1.5 * l_ok + 1.0 * l_haar
                        - 0.25 * m_tgt + 0.08 * repel + 0.80 * proto_d + w_struct * l_sim
                    )
                    g_sub = torch.autograd.grad(loss, zr)[0].detach()
                grads.append(g_sub)
            g = _smooth_grad_3x3(torch.cat(grads, dim=0), blend=0.25)
            d = aniso_rms_direction(-g, P_Y, P_C, w_y=0.55, w_c=0.90)
            vel = 0.45 * vel + d
            z = project_anisotropic_trust(
                z + step_size * vel, z_anc, P_Y, P_C,
                rho_y_edge=ry_e, rho_y_low=ry_l, rho_c=rc,
            )
            if st_idx in snap_steps:
                snaps[st_idx] = z.detach().cpu().half()
        return snaps
