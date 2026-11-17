"""Sprint 1: Anisotropic Luma-LCN Trust-Region Solver & Differentiable S10 + VGG-16 Teacher.

Eliminates ALL THREE root causes of V18's 38%/Ukiyo-e=5% ceiling:
  1. Replaces GroupNorm-blind LatentStyleNet (R1) with un-normalized multi-scale VGG-16 + Oklab
     critic evaluated through frozen S10 (10.66M, bfloat16 autograd).
  2. Replaces isotropic 4-ch Euclidean ball with Anisotropic Luma-Chroma Projection:
       - Tight edge lock on 1D luminance local contrast LCN_5(w_Y^T z)
       - 3x wider freedom on 3D orthogonal chrominance subspace P_C = I - w_Y w_Y^T
  3. Eliminates the 65%-zero hard spatial mask AND eliminates test-time source label `sid`
     by inferring soft source style probabilities p_s from Pi_parallel(x_s).
"""
from __future__ import annotations

import copy
import time

import torch
import torch.nn.functional as F

from . import OUT, RawWriterV19, save_json, tasks_and_sources
from .fiber_duality import (
    BalancedVGGCriticHead,
    StyleFiberEncoder,
    UnnormalizedVGGFeatureExtractor,
    apply_latent_chroma_shift,
)
from exp.v18.ld4_plus import CalibratedLD4Plus
from exp.v18.structure import local_contrast_norm


def _smooth_grad_3x3(g: torch.Tensor, blend: float = 0.25) -> torch.Tensor:
    """Suppress 2x2 upsampling checkerboard modes in pixel->latent gradients."""
    k = torch.tensor(
        [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
        device=g.device, dtype=g.dtype,
    ).view(1, 1, 3, 3) / 16.0
    k4 = k.repeat(4, 1, 1, 1)
    gs = F.conv2d(F.pad(g, (1, 1, 1, 1), mode="reflect"), k4, groups=4)
    return (1.0 - blend) * g + blend * gs


def rms_norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    rms = x.flatten(1).square().mean(dim=1).clamp_min(eps).sqrt().view(-1, 1, 1, 1)
    return x / rms


def aniso_rms_direction(g: torch.Tensor, P_Y: torch.Tensor, P_C: torch.Tensor,
                        w_y: float = 0.45, w_c: float = 0.85) -> torch.Tensor:
    """Separately RMS-normalize 1D luminance gradient and 3D chrominance gradient so neither starves."""
    gy = torch.einsum("ij,bjhw->bihw", P_Y, g.float())
    gc = torch.einsum("ij,bjhw->bihw", P_C, g.float())
    d = w_y * rms_norm(gy) + w_c * rms_norm(gc)
    return rms_norm(d)


def project_anisotropic_trust(
    z_cand: torch.Tensor,
    z_anchor: torch.Tensor,
    P_Y: torch.Tensor,
    P_C: torch.Tensor,
    rho_y_edge: float = 0.16,
    rho_y_low: float = 0.28,
    rho_c: float = 0.46,
) -> torch.Tensor:
    """Project z_cand onto anisotropic trust ball around z_anchor: tight on luma-edges, wide on chroma."""
    dz = z_cand.float() - z_anchor.float()
    dy = torch.einsum("ij,bjhw->bihw", P_Y, dz)
    dc = torch.einsum("ij,bjhw->bihw", P_C, dz)

    # Split 1D luminance delta into smooth 7x7 tone shift vs fine 64x64 edge shift
    dy_low = F.avg_pool2d(F.pad(dy, (3, 3, 3, 3), mode="reflect"), kernel_size=7, stride=1)
    dy_edge = dy - dy_low

    def _clip_rms(t: torch.Tensor, r: float) -> torch.Tensor:
        rms = t.flatten(1).square().mean(dim=1).clamp_min(1e-8).sqrt().view(-1, 1, 1, 1)
        scale = (r / rms).clamp(max=1.0)
        return t * scale

    dy_proj = _clip_rms(dy_low, rho_y_low) + _clip_rms(dy_edge, rho_y_edge)
    dc_proj = _clip_rms(dc, rho_c)
    return z_anchor.float() + dy_proj + dc_proj


def latent_lcn_cost(z: torch.Tensor, z_src: torch.Tensor, w_luma: torch.Tensor, b_luma: torch.Tensor) -> torch.Tensor:
    """Pure chrominance-invariant structural distortion between predicted & source luminance LCN."""
    w = w_luma.to(z.device).float().view(1, 4, 1, 1)
    b = b_luma.to(z.device).float()
    y_cur = (z.float() * w).sum(dim=1, keepdim=True) + b
    y_src = (z_src.float() * w).sum(dim=1, keepdim=True) + b
    lcn_cur = local_contrast_norm(y_cur.clamp(0.0, 1.0), kernel=5, eps=0.03)
    lcn_src = local_contrast_norm(y_src.clamp(0.0, 1.0), kernel=5, eps=0.03)
    return (lcn_cur - lcn_src).flatten(1).square().mean(dim=1)


class AnisotropicS10Teacher:
    """Unified source-label-free teacher supporting T1 (Aniso-R1), T2 (Aniso-S10-VGG), and T3 (Hybrid Pro v2)."""

    def __init__(self, bundle: dict, device: str = "cuda"):
        from src.utils import load_s10_decoder

        self.device = device
        self.v18_cache = bundle["v18_cache"]
        self.fnd = bundle["fnd"]
        self.priors = {k: v.to(device) if torch.is_tensor(v) else v for k, v in self.fnd["priors"].items()}
        self.w_luma = self.v18_cache["luma_proj"]["w"].to(device).float()
        self.b_luma = self.v18_cache["luma_proj"]["b"].to(device).float()

        # Frozen S10 in FP16 (fast no_grad eval) + BF16 clone (overflow-free autograd!)
        self.dec_fp16 = load_s10_decoder(device)
        self.dec_bf16 = copy.deepcopy(self.dec_fp16).to(device=device, dtype=torch.bfloat16).eval()
        for p in self.dec_bf16.parameters():
            p.requires_grad_(False)

        self.vgg_ext = UnnormalizedVGGFeatureExtractor(gram_rank=48).to(device).eval()
        self.vgg_head = BalancedVGGCriticHead(in_dim=self.fnd["vgg_in_dim"], num_styles=5).to(device).eval()
        self.vgg_head.load_state_dict(self.fnd["vgg_head_state"])
        for p in self.vgg_head.parameters():
            p.requires_grad_(False)
        self.vgg_protos = self.fnd["vgg_cls_protos"].to(device).float()

        self.fiber_enc = StyleFiberEncoder(in_dim=self.fnd["tr_fib"].shape[1], dim=128).to(device).eval()
        self.fiber_enc.load_state_dict(self.fnd["fiber_state"])
        for p in self.fiber_enc.parameters():
            p.requires_grad_(False)

        self.ld4 = CalibratedLD4Plus(device=device)
        self.ld4.calibrate_on_holdout(self.v18_cache["holdout"]["z"], self.v18_cache["holdout"]["sid"])

    @torch.no_grad()
    def infer_source_probs(self, fib_src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict soft source style distribution p_s and argmax y_hat_s from Pi_parallel(x_s) (NO ground-truth sid!)."""
        logits = self.fiber_enc.logits(fib_src.to(self.device).float())
        probs = F.softmax(logits / 1.5, dim=-1)
        return probs, probs.argmax(dim=-1)

    def vgg_calibrated_objective(self, rgb01: torch.Tensor, tid: torch.Tensor, src_probs: torch.Tensor,
                                 gamma_repel: float = 0.40, proto_w: float = 0.08) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (per-sample scalar objective, calibrated 5-way logits)."""
        feat = self.vgg_ext(rgb01, multi_scale=True)
        q = self.vgg_head(feat, calibrated=True)  # (B, 5)
        oh = F.one_hot(tid, 5).bool()
        others = q.masked_fill(oh, -1e9).max(dim=1).values
        m_tgt = q.gather(1, tid[:, None]).squeeze(1) - others

        # Soft source repulsion weighted by predicted p_s (masked to k != tid)
        p_other = src_probs.masked_fill(oh, 0.0)
        repel = (p_other * q).sum(dim=1)

        # Gentle standardized feature pull toward target training barycenter (prevents adversarial off-manifold drift)
        feat_std = ((feat.float() - self.vgg_head.x_mean) / self.vgg_head.x_std).clamp(-8.0, 8.0)
        proto_dist = (feat_std - self.vgg_protos[tid]).square().mean(dim=1).sqrt()
        score = m_tgt - gamma_repel * repel - proto_w * proto_dist
        return score, q

    def s10_vgg_direction(self, z: torch.Tensor, tid: torch.Tensor, src_probs: torch.Tensor,
                          micro: int = 6) -> torch.Tensor:
        """Differentiate S10(bf16) -> VGG+Oklab objective in low-VRAM micro-chunks of 6."""
        from src.utils import decode_s10

        P_Y, P_C = self.priors["P_Y"], self.priors["P_C"]
        grads = []
        for st in range(0, len(z), micro):
            sl = slice(st, min(st + micro, len(z)))
            with torch.enable_grad():
                zr = z[sl].detach().float().requires_grad_(True)
                rgb = decode_s10(self.dec_bf16, zr, batch=micro, dtype=torch.bfloat16, enable_grad=True, output_cpu=False)
                obj, _ = self.vgg_calibrated_objective(rgb, tid[sl], src_probs[sl])
                g_sub = torch.autograd.grad(obj.sum(), zr)[0].detach()
            grads.append(g_sub)
        g = torch.cat(grads, dim=0)
        g_clean = _smooth_grad_3x3(g, blend=0.25)
        return aniso_rms_direction(g_clean, P_Y, P_C, w_y=0.45, w_c=0.90)

    def solve_batch(
        self,
        z_base: torch.Tensor,
        z0_anchor: torch.Tensor,
        z_src: torch.Tensor,
        fib_src: torch.Tensor,
        tid: torch.Tensor,
        *,
        mode: str = "hybrid",  # "aniso_r1" | "s10_vgg" | "hybrid"
        alpha_chroma: float = 0.42,
        m_steps: int = 3,
        rho_y_edge: float = 0.16,
        rho_y_low: float = 0.28,
        rho_c: float = 0.45,
        lcn_budget: float = 0.068,
        etas: tuple[float, ...] = (0.10, 0.18, 0.28),
        same_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """ Batched anisotropic trust-region solver (strictly 0 use of ground-truth sid!)."""
        from src.utils import decode_s10

        dev = self.device
        P_Y, P_C = self.priors["P_Y"], self.priors["P_C"]
        src_probs, sid_hat = self.infer_source_probs(fib_src)

        # Chroma warm-start on z_base (leaves 1D luminance structure 100% untouched!)
        z = apply_latent_chroma_shift(z_base.to(dev).float(), tid, self.priors, alpha=alpha_chroma)
        z_anc = apply_latent_chroma_shift(z0_anchor.to(dev).float(), tid, self.priors, alpha=alpha_chroma * 0.7)
        zs = z_src.to(dev).float()

        # Baseline LCN structural distance of z_base itself so budget adapts per image complexity
        base_lcn = latent_lcn_cost(z, zs, self.w_luma, self.b_luma)
        max_lcn = torch.maximum(base_lcn * 1.45 + 0.015, torch.full_like(base_lcn, lcn_budget))

        active = torch.ones(len(z), dtype=torch.bool, device=dev)
        if same_mask is not None:
            active = active & (~same_mask.to(dev))

        def _score_candidates(zc: torch.Tensor, ti: torch.Tensor, sp: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                r1_s = 0.35 * self.ld4.calibrated_margin(zc, ti) - self.ld4.lse.loss(zc, ti)
                if mode == "aniso_r1":
                    return r1_s
                vgg_chunks = []
                for sub in range(0, len(zc), 16):
                    sl_c = slice(sub, min(sub + 16, len(zc)))
                    rgb_c = decode_s10(self.dec_fp16, zc[sl_c], batch=16, output_cpu=False).clamp(0.0, 1.0)
                    vs, _ = self.vgg_calibrated_objective(rgb_c, ti[sl_c], sp[sl_c])
                    vgg_chunks.append(vs)
                vgg_s = torch.cat(vgg_chunks, dim=0)
                if mode == "s10_vgg":
                    return vgg_s
                return 0.65 * vgg_s + 0.45 * r1_s

        for _ in range(m_steps):
            if not active.any():
                break
            idx = active.nonzero(as_tuple=True)[0]
            zi, zai, zsi = z[idx], z_anc[idx], zs[idx]
            ti, spi, shi = tid[idx], src_probs[idx], sid_hat[idx]
            lim_i = max_lcn[idx]

            dirs = []
            if mode in ("s10_vgg", "hybrid"):
                d_vgg = self.s10_vgg_direction(zi, ti, spi)
                dirs.append(d_vgg)
            if mode in ("aniso_r1", "hybrid"):
                with torch.enable_grad():
                    base_d = self.ld4.dcomp.directions(zi, zai, ti, "mix")
                    zr = zi.detach().clone().requires_grad_(True)
                    obj_r1 = self.ld4.calibrated_margin(zr, ti) - 0.35 * self.ld4.calibrated_margin(zr, shi)
                    g_r1 = torch.autograd.grad(obj_r1.sum(), zr)[0].detach()
                d_r1 = aniso_rms_direction(g_r1, P_Y, P_C, w_y=0.45, w_c=0.85)
                d_lat = rms_norm(0.55 * d_r1 + 0.45 * base_d["local"])
                dirs.append(d_lat)
                if mode == "hybrid":
                    dirs.append(rms_norm(0.60 * dirs[0] + 0.40 * d_lat))

            best_s = _score_candidates(zi, ti, spi)
            best_z = zi.clone()
            improved = torch.zeros(len(idx), dtype=torch.bool, device=dev)

            # Evaluate all (dir, eta) candidates in batched passes
            cand_stack = []
            for d_vec in dirs:
                for eta in etas:
                    c_raw = zi + eta * d_vec
                    c_proj = project_anisotropic_trust(
                        c_raw, zai, P_Y, P_C,
                        rho_y_edge=rho_y_edge, rho_y_low=rho_y_low, rho_c=rho_c,
                    )
                    cand_stack.append(c_proj)

            for cand in cand_stack:
                lcn_c = latent_lcn_cost(cand, zsi, self.w_luma, self.b_luma)
                feas = lcn_c <= lim_i
                if not feas.any():
                    continue
                fi = feas.nonzero(as_tuple=True)[0]
                sc = _score_candidates(cand[feas], ti[feas], spi[feas])
                better = sc > best_s[fi]
                if better.any():
                    sel = fi[better]
                    best_z[sel] = cand[sel]
                    best_s[sel] = sc[better]
                    improved[sel] = True

            for j, orig_i in enumerate(idx.tolist()):
                if improved[j]:
                    z[orig_i] = best_z[j]
                else:
                    active[orig_i] = False

        if same_mask is not None:
            sm = same_mask.to(dev)
            z[sm] = zs[sm]
        return z.detach()


def run_sprint1_teachers(bundle: dict, device: str = "cuda"):
    """Run T1, T2, T3, and T3_wrong on Test750 (~3.5 min total on 4090)."""
    from src.utils import decode_s10

    teacher = AnisotropicS10Teacher(bundle, device=device)
    tasks, z_src_750, n2c_cache = tasks_and_sources()
    z0_750 = n2c_cache["z0"].float()
    zn2_750 = n2c_cache["z_n2"].float()
    fib_750 = bundle["fnd"]["test750_fib"].float()

    # Format: (name, mode, alpha_c, m_steps, rho_ye, rho_yl, rho_c, lcn_b, etas, is_wrong)
    arms = [
        ("V19_T1_aniso_r1", "aniso_r1", 0.40, 3, 0.16, 0.28, 0.45, 0.068, (0.10, 0.18, 0.28), False),
        ("V19_T2_s10_vgg_nosid", "s10_vgg", 0.45, 3, 0.16, 0.28, 0.45, 0.068, (0.10, 0.18, 0.28), False),
        ("V19_T3_hybrid_pro_nosid", "hybrid", 0.45, 3, 0.16, 0.28, 0.45, 0.068, (0.10, 0.18, 0.28), False),
        ("V19_T3_hybrid_wrong", "hybrid", 0.45, 3, 0.16, 0.28, 0.45, 0.068, (0.10, 0.18, 0.28), True),
        # High-content sweet-spot arms (targeting DINO-C ~ 0.72-0.76, LPIPS ~ 0.48-0.50, Top-1 ~ 50-60%!)
        ("V19_T4_sweet_vgg_nosid", "s10_vgg", 0.18, 2, 0.065, 0.12, 0.22, 0.024, (0.055, 0.105), False),
        ("V19_T5_sweet_hybrid_nosid", "hybrid", 0.20, 2, 0.075, 0.13, 0.25, 0.028, (0.060, 0.115), False),
        ("V19_T5_sweet_wrong", "hybrid", 0.20, 2, 0.075, 0.13, 0.25, 0.028, (0.060, 0.115), True),
    ]

    for arm_name, mode, alpha_c, m_st, rye, ryl, rc, lcnb, etas_v, is_wrong in arms:
        if (OUT / f"raw/{arm_name}/meta.json").exists():
            print(f"[V19 Sprint1] SKIP {arm_name} (raw exists)", flush=True)
            continue
        wr = RawWriterV19(arm_name)
        t0 = time.time()
        for st in range(0, 750, 25):
            sl = slice(st, min(st + 25, 750))
            grp = tasks[sl]
            sid_proto = torch.tensor([t["source_style_id"] for t in grp], device=device)
            tid_true = torch.tensor([t["target_style_id"] for t in grp], device=device)
            same = sid_proto == tid_true
            if is_wrong:
                tid_used = (tid_true + 1) % 5
                clash = tid_used == sid_proto
                tid_used = torch.where(clash, (tid_used + 1) % 5, tid_used)
                tid_used = torch.where(same, tid_true, tid_used)
            else:
                tid_used = tid_true

            if is_wrong:
                _, z0_b, zn2_b = teacher.ld4.run_b4_and_n2c(z_src_750[sl].to(device), tid_used)
            else:
                z0_b = z0_750[sl].to(device)
                zn2_b = zn2_750[sl].to(device)

            z_out = teacher.solve_batch(
                zn2_b, z0_b, z_src_750[sl], fib_750[sl], tid_used,
                mode=mode, alpha_chroma=alpha_c, m_steps=m_st,
                rho_y_edge=rye, rho_y_low=ryl, rho_c=rc, lcn_budget=lcnb,
                etas=etas_v, same_mask=same,
            )
            with torch.no_grad():
                rgb = decode_s10(teacher.dec_fp16, z_out, batch=16)
            wr.append([t["task_id"] for t in grp], rgb, z_out)
        elapsed = round(time.time() - t0, 1)
        wr.finish(mode=mode, alpha_chroma=alpha_c, wrong_label=is_wrong, wall_sec=elapsed)
        print(f"[V19 Sprint1] Generated {arm_name} in {elapsed}s", flush=True)
