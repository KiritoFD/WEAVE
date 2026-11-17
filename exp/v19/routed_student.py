"""Sprint 2: Spatially-Routed Universal-Chord 1-Step Student (PRISM-1 v2).

Key innovations over V18 FastDINOStudent:
  1. Source-Label-Free Universal Chord: e(x_s -> tau) = u_tau - phi(Pi_parallel(x_s)) in R^128.
     Supports BOTH registered class barycenters AND 1-shot single-painting exemplars (zero class table!),
     with exact zero chord e(x_s -> x_s) == 0 on self-exemplar.
  2. Latent Chrominance Pre-Alignment driven continuously by predicted chroma moments from u_tau,
     preserving 100% of 1D luminance structure P_Y(z_s).
  3. Semantic-Chord Spatial Router: 16x16 whitened DINO tokens P_16 interact multiplicatively with
     chord e to produce spatially heterogeneous modulation fields Gamma_64, Beta_64 across a
     dilated [1, 1, 2, 2, 4, 1] backbone (53x53 latent receptive field = 424x424 pixels!).
  4. Hybrid Latent + Differentiable S10 VGG-Moment Distillation Loss (cures L2 color averaging).
"""
from __future__ import annotations

import copy
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import OUT, RawWriterV19, autocast, save_json, seed_all, tasks_and_sources
from .aniso_teacher import AnisotropicS10Teacher
from .fiber_duality import StyleFiberEncoder, apply_latent_chroma_shift
from tools.legacy_root.wavelet import dwt2_haar


class SemanticChordRouter(nn.Module):
    """Generates spatially heterogeneous (Gamma_64, Beta_64) from whitened DINO P_16 and chord e."""

    def __init__(self, dino_dim: int = 384, chord_dim: int = 128, width: int = 96):
        super().__init__()
        self.q_proj = nn.Sequential(
            nn.Conv2d(dino_dim, width, 1),
            nn.GroupNorm(16, width),
            nn.SiLU(inplace=True),
        )
        self.k_proj = nn.Linear(chord_dim, width * 2)
        self.smooth = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1, groups=width),
            nn.Conv2d(width, width * 2, 1),
        )
        nn.init.zeros_(self.smooth[-1].weight)
        nn.init.zeros_(self.smooth[-1].bias)

    def forward(self, dino_16: torch.Tensor, chord: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.q_proj(dino_16)  # (B, W, 16, 16)
        kg, kb = self.k_proj(chord).chunk(2, dim=-1)
        inter = F.silu(q * (1.0 + kg[:, :, None, None]) + kb[:, :, None, None])
        gb_16 = self.smooth(inter)
        gb_64 = F.interpolate(gb_16, size=(64, 64), mode="bilinear", align_corners=False)
        return gb_64.chunk(2, dim=1)


class DilatedSpatialResBlock(nn.Module):
    def __init__(self, width: int = 96, dilation: int = 1, chord_dim: int = 128):
        super().__init__()
        self.norm1 = nn.GroupNorm(16, width)
        self.fc_global = nn.Linear(chord_dim, width * 2)
        self.conv1 = nn.Conv2d(width, width, 3, padding=dilation, dilation=dilation)
        self.norm2 = nn.GroupNorm(16, width)
        self.conv2 = nn.Conv2d(width, width, 3, padding=1)

    def forward(self, h: torch.Tensor, chord: torch.Tensor, sp_g: torch.Tensor, sp_b: torch.Tensor) -> torch.Tensor:
        gg, gb = self.fc_global(chord).chunk(2, dim=-1)
        x = self.norm1(h)
        x = x * (1.0 + gg[:, :, None, None] + sp_g) + (gb[:, :, None, None] + sp_b)
        x = self.conv1(F.silu(x))
        x = self.conv2(F.silu(self.norm2(x)))
        return h + x


class SpatiallyRoutedChordNet(nn.Module):
    """~1.22M Single-Step Source-Label-Free Chord Residual Network (PRISM-1 v2)."""

    def __init__(self, width: int = 96, chord_dim: int = 128, use_spatial_router: bool = True):
        super().__init__()
        self.use_spatial_router = use_spatial_router
        self.stem = nn.Sequential(
            nn.Conv2d(12, width, 3, padding=1),
            nn.GroupNorm(16, width),
            nn.SiLU(inplace=True),
        )
        self.dino_add = nn.Sequential(
            nn.Conv2d(384, width, 1),
            nn.GroupNorm(16, width),
            nn.SiLU(inplace=True),
        )
        if use_spatial_router:
            self.router = SemanticChordRouter(384, chord_dim, width)
        self.blocks = nn.ModuleList([
            DilatedSpatialResBlock(width=width, dilation=d, chord_dim=chord_dim)
            for d in (1, 1, 2, 2, 4, 1)
        ])
        self.out_norm = nn.GroupNorm(16, width)
        self.out_conv = nn.Conv2d(width, 4, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        z_anchor: torch.Tensor,
        z_luma: torch.Tensor,
        c_64: torch.Tensor,
        dino_16: torch.Tensor,
        chord: torch.Tensor,
        *,
        same_mask: torch.Tensor | None = None,
        soft_id_sigma: float = 0.25,
    ) -> torch.Tensor:
        d_up = F.interpolate(self.dino_add(dino_16), size=(64, 64), mode="bilinear", align_corners=False)
        h = self.stem(torch.cat([z_anchor, z_luma, c_64], dim=1)) + d_up
        if self.use_spatial_router:
            sp_g, sp_b = self.router(dino_16, chord)
        else:
            sp_g = torch.zeros_like(h)
            sp_b = torch.zeros_like(h)

        for blk in self.blocks:
            h = blk(h, chord, sp_g, sp_b)
        delta = self.out_conv(F.silu(self.out_norm(h)))

        # Continuous chord-magnitude identity gate: exact 0 when chord == 0 (e.g., x_ref == x_s!)
        cnorm = chord.norm(dim=-1).view(-1, 1, 1, 1)
        soft_gate = torch.tanh(cnorm / soft_id_sigma).square()
        delta = delta * soft_gate
        if same_mask is not None:
            delta = torch.where(same_mask.view(-1, 1, 1, 1), torch.zeros_like(delta), delta)
        return delta


def build_or_load_v19_distill_pairs(bundle: dict, teacher_mode: str = "hybrid",
                                    n_per_style: int = 100, device: str = "cuda") -> dict:
    """Generate 2,500 clean balanced sweet-spot cross-domain trajectories (500 sources x 5 targets) in ~4.5 min!"""
    pairs_path = OUT / f"cache/v19_sweet_pairs_{teacher_mode}_{n_per_style*5}.pt"
    if pairs_path.exists():
        print(f"[V19 Sprint2] Reusing {pairs_path.name}", flush=True)
        return torch.load(pairs_path, map_location="cpu", weights_only=False)

    t0 = time.time()
    teacher = AnisotropicS10Teacher(bundle, device=device)
    tr = bundle["v18_cache"]["train"]
    tr_fib = bundle["fnd"]["tr_fib"].float()

    g = torch.Generator().manual_seed(20261119)
    sel_idx = []
    for s in range(5):
        ms = (tr["sid"] == s).nonzero(as_tuple=True)[0]
        pick = ms[torch.randperm(len(ms), generator=g)[:n_per_style]]
        sel_idx.append(pick)
    src_idx = torch.cat(sel_idx, dim=0)  # (500,)

    N_src = len(src_idx)
    z_src_all, z_anc_all, z_star_all = [], [], []
    c64_all, d16_all, fib_all, sid_all, tid_all = [], [], [], [], []

    src_rep = src_idx.repeat_interleave(5)
    tid_rep = torch.arange(5).repeat(N_src)

    for st in range(0, len(src_rep), 50):
        bi = src_rep[st:st + 50]
        ti = tid_rep[st:st + 50].to(device)
        si = tr["sid"][bi].to(device)
        zs = tr["z"][bi].to(device).float()
        fb = tr_fib[bi].to(device).float()
        same = si == ti

        with torch.no_grad():
            _, z0_b, zn2_b = teacher.ld4.run_b4_and_n2c(zs, ti)
            z0_aniso = apply_latent_chroma_shift(z0_b, ti, teacher.priors, alpha=0.20)
            z0_aniso[same] = zs[same]

        # Fast 2-step sweet-spot trajectory preserving high DINO-C (~0.72+) while transferring full 5-style palette!
        z_star = teacher.solve_batch(
            zn2_b, z0_b, zs, fb, ti,
            mode=teacher_mode, alpha_chroma=0.20, m_steps=2,
            rho_y_edge=0.075, rho_y_low=0.13, rho_c=0.25, lcn_budget=0.028,
            etas=(0.060, 0.115), same_mask=same,
        )

        z_src_all.append(zs.cpu().half())
        z_anc_all.append(z0_aniso.cpu().half())
        z_star_all.append(z_star.cpu().half())
        c64_all.append(tr["c_64"][bi].half())
        d16_all.append(tr["dino_16"][bi].half())
        fib_all.append(tr_fib[bi].half())
        sid_all.append(si.cpu())
        tid_all.append(ti.cpu())
        if (st // 50) % 10 == 0:
            print(f"[V19 Pairs] {st+len(bi)}/{len(src_rep)} ({time.time()-t0:.1f}s)", flush=True)

    payload = {
        "z_src": torch.cat(z_src_all, 0),
        "z_anc": torch.cat(z_anc_all, 0),
        "z_star": torch.cat(z_star_all, 0),
        "c_64": torch.cat(c64_all, 0),
        "dino_16": torch.cat(d16_all, 0),
        "fib_src": torch.cat(fib_all, 0),
        "sid": torch.cat(sid_all, 0),
        "tid": torch.cat(tid_all, 0),
        "teacher_mode": teacher_mode,
        "wall_sec": round(time.time() - t0, 1),
    }
    torch.save(payload, pairs_path)
    print(f"[V19 Pairs] Saved {pairs_path.name} in {payload['wall_sec']}s", flush=True)
    return payload


def continuous_chroma_shift(z_b4: torch.Tensor, u_tgt: torch.Tensor, fiber_enc: StyleFiberEncoder,
                            priors: dict, alpha: float = 0.42) -> torch.Tensor:
    """Chroma shift driven purely by continuous target vector u_tgt (works for BOTH class & 1-shot exemplar!)."""
    dev = z_b4.device
    P_Y = priors["P_Y"].to(dev)
    P_C = priors["P_C"].to(dev)
    pred8 = fiber_enc.chroma_head(u_tgt)
    mu_t = pred8[:, :4].view(-1, 4, 1, 1)
    sd_t = pred8[:, 4:].clamp(-4.0, 2.0).exp().view(-1, 4, 1, 1)

    zy = torch.einsum("ij,bjhw->bihw", P_Y, z_b4.float())
    zc = torch.einsum("ij,bjhw->bihw", P_C, z_b4.float())
    mu_s = zc.mean(dim=(2, 3), keepdim=True)
    sd_s = zc.std(dim=(2, 3), keepdim=True).clamp_min(1e-4)
    zc_al = torch.einsum("ij,bjhw->bihw", P_C, (zc - mu_s) / sd_s * sd_t + mu_t)
    return zy + (1.0 - alpha) * zc + alpha * zc_al


def train_and_eval_v19_students(bundle: dict, teacher_mode: str = "hybrid",
                                steps: int = 3600, device: str = "cuda"):
    """Train S1 (global chord) and S2 (SpatiallyRouted PRISM-1 v2) & evaluate single-step + continuous strength arms."""
    from src.utils import load_s10_decoder, decode_s10
    from src.infer import load_c3unet, canonicalizer

    pairs = build_or_load_v19_distill_pairs(bundle, teacher_mode=teacher_mode, n_per_style=100, device=device)
    fnd = bundle["fnd"]
    priors = {k: v.to(device) if torch.is_tensor(v) else v for k, v in fnd["priors"].items()}
    P_Y = priors["P_Y"]

    fiber_enc = StyleFiberEncoder(in_dim=fnd["tr_fib"].shape[1], dim=128).to(device).eval()
    fiber_enc.load_state_dict(fnd["fiber_state"])
    for p in fiber_enc.parameters():
        p.requires_grad_(False)

    with torch.no_grad():
        u_src_all = fiber_enc.encode(pairs["fib_src"].to(device).float()).cpu()
        bary = fiber_enc.class_barycenters().detach().cpu()  # (5, 128)
        # Also encode all 4,400 train fibers by class so 50% of steps sample random exemplar chords!
        tr_u_all = fiber_enc.encode(fnd["tr_fib"].to(device).float()).cpu()
        tr_sid_all = bundle["v18_cache"]["train"]["sid"]
        cls_u_pools = [tr_u_all[tr_sid_all == s] for s in range(5)]

    dec_fp16 = load_s10_decoder(device)
    dec_bf16 = copy.deepcopy(dec_fp16).to(device=device, dtype=torch.bfloat16).eval()
    for p in dec_bf16.parameters():
        p.requires_grad_(False)
    from .fiber_duality import UnnormalizedVGGFeatureExtractor
    vgg_ext = UnnormalizedVGGFeatureExtractor(gram_rank=48).to(device).eval()
    vgg_mean = fnd["vgg_head_state"]["x_mean"].to(device).float()
    vgg_std = fnd["vgg_head_state"]["x_std"].to(device).float()

    stu_configs = [
        ("s1_global", False, False, 2400),
        ("s2_prism1_v2", True, True, steps),
    ]

    trained_models = {}
    for tag, use_router, use_rgb_mom, n_steps in stu_configs:
        ckpt_p = OUT / f"students/{tag}.pt"
        ckpt_p.parent.mkdir(parents=True, exist_ok=True)
        net = SpatiallyRoutedChordNet(width=96, chord_dim=128, use_spatial_router=use_router).to(device)
        if ckpt_p.exists():
            print(f"[V19 Student] Loading {ckpt_p.name}", flush=True)
            net.load_state_dict(torch.load(ckpt_p, map_location=device, weights_only=False)["model"])
            trained_models[tag] = net.eval()
            continue

        seed_all(20261120)
        opt = torch.optim.AdamW(net.parameters(), lr=1.2e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps, eta_min=8e-5)
        ema = copy.deepcopy(net).eval()
        for p in ema.parameters():
            p.requires_grad_(False)

        N = len(pairs["z_src"])
        t0 = time.time()
        for step in range(1, n_steps + 1):
            idx = torch.randint(0, N, (64,))
            zs = pairs["z_src"][idx].to(device).float()
            z_anc = pairs["z_anc"][idx].to(device).float()
            z_star = pairs["z_star"][idx].to(device).float()
            c64 = pairs["c_64"][idx].to(device).float()
            d16 = pairs["dino_16"][idx].to(device).float()
            us = u_src_all[idx].to(device).float()
            ti = pairs["tid"][idx]
            si = pairs["sid"][idx]
            same = (si == ti).to(device)

            # 50% class barycenter chord, 50% random single-exemplar chord -> trains BOTH modes jointly!
            ut_cls = bary[ti].to(device)
            if step % 2 == 0:
                ut_ex = torch.stack([
                    cls_u_pools[int(t)][torch.randint(0, len(cls_u_pools[int(t)]), (1,)).item()]
                    for t in ti
                ]).to(device)
                ut = torch.where(same.unsqueeze(-1), us, ut_ex)
            else:
                ut = torch.where(same.unsqueeze(-1), us, ut_cls)

            chord = ut - us  # identically 0 on same-style / self-exemplar!
            z_luma = torch.einsum("ij,bjhw->bihw", P_Y, zs)

            delta = net(z_anc, z_luma, c64, d16, chord, same_mask=None)
            z_hat = z_anc + delta

            loss_lat = F.smooth_l1_loss(z_hat, z_star, beta=0.06)
            _, lh_p, hl_p, hh_p = dwt2_haar(z_hat)
            _, lh_t, hl_t, hh_t = dwt2_haar(z_star)
            loss_hf = (
                F.mse_loss(lh_p, lh_t) + F.mse_loss(hl_p, hl_t) + 0.5 * F.mse_loss(hh_p, hh_t)
            )
            loss = loss_lat + 0.35 * loss_hf

            # Every 2nd step on a 6-sample cross slice: direct S10 VGG+Oklab moment alignment (<6GB VRAM)!
            if use_rgb_mom and (step % 2 == 0):
                cross_i = (~same).nonzero(as_tuple=True)[0][:6]
                if len(cross_i) >= 2:
                    rgb_p = decode_s10(dec_bf16, z_hat[cross_i], batch=6, dtype=torch.bfloat16, enable_grad=True, output_cpu=False)
                    fp = vgg_ext(rgb_p, multi_scale=True)
                    with torch.no_grad():
                        rgb_t = decode_s10(dec_fp16, z_star[cross_i], batch=6, output_cpu=False).clamp(0.0, 1.0)
                        ft = vgg_ext(rgb_t, multi_scale=True)
                    loss_rgb = F.smooth_l1_loss((fp - vgg_mean) / vgg_std, (ft - vgg_mean) / vgg_std, beta=0.2)
                    loss = loss + 0.12 * loss_rgb

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
            opt.step()
            sched.step()
            with torch.no_grad():
                for pe_w, pw in zip(ema.parameters(), net.parameters()):
                    pe_w.mul_(0.995).add_(pw, alpha=0.005)

            if step == 1 or step % 600 == 0 or step == n_steps:
                print(f"[V19 Stu:{tag}] {step:4d}/{n_steps} loss={loss.item():.4f} lat={loss_lat.item():.4f} ({time.time()-t0:.1f}s)", flush=True)

        torch.save({"model": ema.state_dict(), "params": ema.n_params()}, ckpt_p)
        trained_models[tag] = ema.eval()

    # ---- Fast 1-Step Inference on Test750 across 5 Evaluation Arms ----
    tasks, z_src_750, n2c_cache = tasks_and_sources()
    z0_750 = n2c_cache["z0"].float()
    c64_750 = bundle["test_struct"]["c_64"].float()
    d16_750 = bundle["test_struct"]["dino_16"].float()
    fib_750 = fnd["test750_fib"].float()
    oneshot_fib = fnd["oneshot_fib"].to(device).float()

    b4, _ = load_c3unet(OUT.parents[1] / "runs/v6/P1_B4_distill50/P1_B4_distill50.pt", device)
    canon = canonicalizer(device)

    with torch.no_grad():
        u_oneshot = fiber_enc.encode(oneshot_fib)  # (5, 128) single holdout painting per style!
        bary_gpu = fiber_enc.class_barycenters()   # (5, 128)

    # Format: (arm_name, model_tag, cond_mode, proto_same_gate, is_wrong, strength_lambda)
    eval_arms = [
        ("V19_S1_global_nosid", "s1_global", "class_proto", True, False, 1.0),
        ("V19_S2_prism1_l075", "s2_prism1_v2", "class_proto", True, False, 0.75),
        ("V19_S2_prism1_v2", "s2_prism1_v2", "class_proto", True, False, 1.0),
        ("V19_S2_prism1_l130", "s2_prism1_v2", "class_proto", True, False, 1.30),
        ("V19_S2b_prism1_blind_same", "s2_prism1_v2", "class_proto", False, False, 1.0),
        ("V19_S2_prism1_wrong", "s2_prism1_v2", "class_proto", True, True, 1.0),
        ("V19_S3_oneshot_exemplar", "s2_prism1_v2", "oneshot_ref", True, False, 1.0),
    ]

    for arm_name, m_tag, cond_mode, proto_same_gate, is_wrong, lam in eval_arms:
        if (OUT / f"raw/{arm_name}/meta.json").exists():
            continue
        stu = trained_models[m_tag]
        wr = RawWriterV19(arm_name)
        t0 = time.time()
        torch.cuda.synchronize()
        for st in range(0, 750, 50):
            sl = slice(st, min(st + 50, 750))
            grp = tasks[sl]
            zs = z_src_750[sl].to(device)
            c64 = c64_750[sl].to(device)
            d16 = d16_750[sl].to(device)
            fb = fib_750[sl].to(device)

            sid_proto = torch.tensor([t["source_style_id"] for t in grp], device=device)
            tid_true = torch.tensor([t["target_style_id"] for t in grp], device=device)
            same = sid_proto == tid_true

            if is_wrong:
                tid_u = (tid_true + 1) % 5
                tid_u = torch.where(tid_u == sid_proto, (tid_u + 1) % 5, tid_u)
                tid_u = torch.where(same, tid_true, tid_u)
            else:
                tid_u = tid_true

            with torch.no_grad():
                u_src = fiber_enc.encode(fb)  # 100% inferred from x_s! ZERO use of sid_proto in chord!
                if is_wrong:
                    c_can = canon(zs, mode="c1_gentle", augment=False)
                    z0_b = b4(c_can, 0.0, c_can, tid_u)
                else:
                    z0_b = z0_750[sl].to(device)

                alpha_eff = min(0.45, 0.20 * lam)
                if cond_mode == "oneshot_ref":
                    u_tgt = u_oneshot[tid_u]
                    if proto_same_gate:
                        u_tgt = torch.where(same.unsqueeze(-1), u_src, u_tgt)
                    z_anc = continuous_chroma_shift(z0_b, u_tgt, fiber_enc, priors, alpha=alpha_eff)
                else:
                    u_tgt = bary_gpu[tid_u]
                    z_anc = apply_latent_chroma_shift(z0_b, tid_u, priors, alpha=alpha_eff)

                chord = u_tgt - u_src
                z_luma = torch.einsum("ij,bjhw->bihw", P_Y, zs)
                sm = same if proto_same_gate else None
                delta = stu(z_anc, z_luma, c64, d16, chord, same_mask=sm)
                z_out = z_anc + lam * delta
                if proto_same_gate:
                    z_out[same] = zs[same]
                rgb = decode_s10(dec_fp16, z_out, batch=25)
            wr.append([t["task_id"] for t in grp], rgb, z_out)
        torch.cuda.synchronize()
        wall = round(time.time() - t0, 2)
        ms_per_img = round(wall * 1000.0 / 750.0, 2)
        wr.finish(model=m_tag, cond_mode=cond_mode, strength_lambda=lam, wall_sec=wall, ms_per_img=ms_per_img)
        print(f"[V19 Sprint2] {arm_name:28s} | 750 imgs in {wall:.2f}s ({ms_per_img:.2f} ms/img)", flush=True)
