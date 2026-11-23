"""Strictly referee-free (0 CSD/DINO/CLIP/LPIPS) structural gate & style ranking for teachers & bridges."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from exp.v18.structure import local_contrast_norm
from exp.v19.fiber_duality import UnnormalizedVGGFeatureExtractor, extract_color_fiber_27d
from exp.v20.submode_patchswd import MultiScalePatchSWDExtractor, sliced_wasserstein_1d
from exp.v27_mainline.rollout_refine import (
    FrozenVGG16Rollout,
    dynamic_swd,
    gray_self_sim_loss,
    haar_hf_lf_ratio_2level,
)
from exp.v28_unified import OUT
from src.utils import decode_s10
from src.utils.runtime import bf16_autocast


def latent_luma_lcn_dist(z: torch.Tensor, z_src: torch.Tensor, w_luma: torch.Tensor, b_luma: torch.Tensor) -> torch.Tensor:
    """Chrominance-invariant luminance local contrast distance (B,)."""
    w = w_luma.to(z.device).float().view(1, 4, 1, 1)
    b = b_luma.to(z.device).float()
    y_cur = ((z.float() * w).sum(dim=1, keepdim=True) + b).clamp(0.0, 1.0)
    y_src = ((z_src.float() * w).sum(dim=1, keepdim=True) + b).clamp(0.0, 1.0)
    lcn_c = local_contrast_norm(y_cur, kernel=5, eps=0.03)
    lcn_s = local_contrast_norm(y_src, kernel=5, eps=0.03)
    return (lcn_c - lcn_s).abs().mean(dim=(1, 2, 3))


class RefereeFreeSelector(nn.Module):
    """Referee-free structural gate (gray VGG relu4_2 + luma LCN) & orthogonal Patch-SWD + Oklab/Haar ranker."""

    def __init__(self, bundle: dict, bank: dict, dec_bf16: nn.Module, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.dec_bf16 = dec_bf16
        self.vgg = FrozenVGG16Rollout().to(device).eval()
        self.swd_ext = MultiScalePatchSWDExtractor(n_slices=64, seed=20261128).to(device).eval()
        self.w_luma = bundle["v18_cache"]["luma_proj"]["w"].to(device).float()
        self.b_luma = bundle["v18_cache"]["luma_proj"]["b"].to(device).float()

        # Precompute fixed orthogonal projections for VGG s1/s2/s3 (64, 128, 256) -> 48 slices
        g = torch.Generator().manual_seed(20261129)
        for idx, ch in ((1, 64), (2, 128), (3, 256)):
            q, _ = torch.linalg.qr(torch.randn(ch, 48, generator=g), mode="reduced")
            self.register_buffer(f"vgg_proj_{idx}", q.to(device))

        # Precompute target style core distributions (5, K=8, ...) strictly from train_4400 core exemplars!
        with torch.no_grad():
            ex_z = bank["sub_exemplar_z"].reshape(40, 4, 64, 64).to(device).float()
            ex_rgb = decode_s10(dec_bf16, ex_z, batch=8, dtype=torch.bfloat16, output_cpu=False).clamp(0.0, 1.0)
            with bf16_autocast(device):
                f1, f2, f3, _ = self.vgg.forward_style(ex_rgb)
                cnx_tok = self.swd_ext.extract_tokens(ex_rgb)
            self.ex_vgg = [
                F.adaptive_avg_pool2d(f, 16).float().flatten(2).transpose(1, 2).view(5, 8, 256, -1)
                for f in (f1, f2, f3)
            ]
            self.ex_cnx = {k: v.float().view(5, 8, v.shape[1], v.shape[2]) for k, v in cnx_tok.items()}
            self.ex_oklab = extract_color_fiber_27d(ex_rgb).view(5, 8, 27)
            self.ex_haar = haar_hf_lf_ratio_2level(ex_rgb).view(5, 8)

            # Gray VGG relu4_2 scene descriptors of exemplars for 100% DINO-free scene matching!
            r42_ex = self.vgg.forward_gray_r42(ex_rgb)
            sc_ex = F.normalize(F.adaptive_avg_pool2d(r42_ex.float(), 4).flatten(1), dim=-1)
            self.ex_gray_scene = sc_ex.view(5, 8, -1)

    @torch.no_grad()
    def match_gray_exemplar(self, rgb_src: torch.Tensor, tid: torch.Tensor) -> torch.Tensor:
        with bf16_autocast(self.device):
            r42 = self.vgg.forward_gray_r42(rgb_src)
        sc = F.normalize(F.adaptive_avg_pool2d(r42.float(), 4).flatten(1), dim=-1)
        sim = torch.einsum("bd,bkd->bk", sc, self.ex_gray_scene[tid])
        return sim.argmax(dim=1)  # (B,) in [0..7]

    @torch.no_grad()
    def score_candidates(
        self,
        z_src: torch.Tensor,
        rgb_src: torch.Tensor,
        tid: torch.Tensor,
        cand_latents: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Score M candidates (each (B, 4, 64, 64)) for structural fidelity & referee-free style match."""
        B = len(z_src)
        best_c = self.match_gray_exemplar(rgb_src, tid)
        with bf16_autocast(self.device):
            src_r42 = self.vgg.forward_gray_r42(rgb_src)
        src_r42_16 = F.normalize(F.adaptive_avg_pool2d(src_r42.float(), 16).flatten(2), dim=1)
        src_sim = torch.bmm(src_r42_16.transpose(1, 2), src_r42_16)

        ref_ok = self.ex_oklab[tid, best_c]
        ref_haar = self.ex_haar[tid, best_c]
        ref_vgg = [ev[tid, best_c] for ev in self.ex_vgg]
        ref_cnx = {k: v[tid, best_c] for k, v in self.ex_cnx.items()}

        gray_sim_errs, lcn_errs, style_dists = [], [], []
        for zc in cand_latents:
            zc_f = zc.to(self.device).float()
            lcn_e = latent_luma_lcn_dist(zc_f, z_src.to(self.device).float(), self.w_luma, self.b_luma)
            rgb_c = decode_s10(self.dec_bf16, zc_f, batch=min(B, 8), dtype=torch.bfloat16, output_cpu=False).clamp(0.0, 1.0)
            with bf16_autocast(self.device):
                r42_c = self.vgg.forward_gray_r42(rgb_c)
                f1, f2, f3, _ = self.vgg.forward_style(rgb_c)
                cnx_c = self.swd_ext.extract_tokens(rgb_c)
            c_16 = F.normalize(F.adaptive_avg_pool2d(r42_c.float(), 16).flatten(2), dim=1)
            c_sim = torch.bmm(c_16.transpose(1, 2), c_16)
            gs_err = F.smooth_l1_loss(c_sim, src_sim, beta=0.05, reduction="none").mean(dim=(1, 2))

            # Orthogonal VGG Patch-SWD + ConvNeXt Patch-SWD + Oklab + Haar
            vgg_swd = torch.zeros(B, device=self.device)
            for idx, feat, r_tok in zip((1, 2, 3), (f1, f2, f3), ref_vgg):
                proj = getattr(self, f"vgg_proj_{idx}")
                tok = F.adaptive_avg_pool2d(feat.float(), 16).flatten(2).transpose(1, 2)
                vgg_swd = vgg_swd + sliced_wasserstein_1d(tok, r_tok, proj)
            vgg_swd = vgg_swd / 3.0

            cnx_swd = (
                1.0 * sliced_wasserstein_1d(cnx_c["rgb"].float(), ref_cnx["rgb"], self.swd_ext.p_rgb)
                + 0.4 * sliced_wasserstein_1d(cnx_c["s1"].float(), ref_cnx["s1"], self.swd_ext.p_s1)
                + 0.5 * sliced_wasserstein_1d(cnx_c["s2"].float(), ref_cnx["s2"], self.swd_ext.p_s2)
                + 0.6 * sliced_wasserstein_1d(cnx_c["s3"].float(), ref_cnx["s3"], self.swd_ext.p_s3)
            )
            ok_d = (extract_color_fiber_27d(rgb_c) - ref_ok).abs().mean(dim=1)
            haar_d = (haar_hf_lf_ratio_2level(rgb_c) - ref_haar).abs()
            s_dist = vgg_swd + 0.6 * cnx_swd + 3.0 * ok_d + 0.25 * haar_d

            gray_sim_errs.append(gs_err)
            lcn_errs.append(lcn_e)
            style_dists.append(s_dist)

        return {
            "gray_sim_err": torch.stack(gray_sim_errs, dim=1),  # (B, M)
            "lcn_err": torch.stack(lcn_errs, dim=1),            # (B, M)
            "style_dist": torch.stack(style_dists, dim=1),      # (B, M)
        }

    @torch.no_grad()
    def select_tiers(
        self,
        z_src: torch.Tensor,
        rgb_src: torch.Tensor,
        tid: torch.Tensor,
        cand_latents: list[torch.Tensor],
    ) -> dict:
        """Select ONE winning endpoint per tier (hiC, mid, hiS) without blending."""
        metrics = self.score_candidates(z_src, rgb_src, tid, cand_latents)
        g_err = metrics["gray_sim_err"]  # (B, M)
        l_err = metrics["lcn_err"]       # (B, M)
        s_dst = metrics["style_dist"]    # (B, M)
        B, M = g_err.shape
        stack_z = torch.stack([z.cpu().half() for z in cand_latents], dim=1)  # (B, M, 4, 64, 64)

        # Tier structural gates calibrated against real gray_sim_err & lcn_err distributions
        tier_gates = {
            "hiC": (0.068, 0.145),
            "mid": (0.090, 0.205),
            "hiS": (0.100, 0.230),
        }
        out = {}
        for tier, (max_g, max_l) in tier_gates.items():
            ok = (g_err <= max_g) & (l_err <= max_l)
            has_ok = ok.any(dim=1)
            # Rank qualified subset by lowest style_dist; fallback to lowest (g_err + 0.1*l_err)
            masked_style = torch.where(ok, s_dst, torch.full_like(s_dst, 1e9))
            best_ok_idx = masked_style.argmin(dim=1)
            fallback_idx = (g_err + 0.1 * l_err).argmin(dim=1)
            pick = torch.where(has_ok, best_ok_idx, fallback_idx).cpu()
            out[tier] = stack_z[torch.arange(B), pick]
            out[f"{tier}_pick"] = pick
            out[f"{tier}_gate_pass"] = has_ok.cpu()
        return out
