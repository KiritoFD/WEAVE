"""Test-Time Manifold Optimization (TTO) 4-Arm Ablation on Frozen Final_5k_seed0 (Zero Retraining).

Compares 4 strictly isolated 10-step Adam configurations from identical base (Final_5k_seed0 @ lambda=1.0):
  1. TTO1_10step_z_bare    : 10-step Adam directly on final z + pooled VGG-SWD (vs 3-step s=2.00 baseline).
  2. TTO2_z_struct_lock    : TTO1 + Gray VGG relu4_2 self-similarity & P_Y luminance LCN hard locks.
  3. TTO3_z_subschool_cnx  : TTO2 + single gray-matched sub-school + ConvNeXt-Tiny 7x7 Patch-SWD + Q32 soft local SWD.
  4. TTO4_unet_manifold    : Same objective as TTO3, optimizing (dz_in, deps, dq32, dcond) through frozen 11M U-Net.
"""
from __future__ import annotations

import gc
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.nn.functional as F

from exp.v18 import tasks_and_sources
from exp.v19.aniso_teacher import (
    _smooth_grad_3x3,
    latent_lcn_cost,
    project_anisotropic_trust,
)
from exp.v19.fiber_duality import UnnormalizedVGGFeatureExtractor, extract_color_fiber_27d
from exp.v20.submode_patchswd import MultiScalePatchSWDExtractor, _kmeans_unit, sliced_wasserstein_1d
from exp.v27_mainline.rollout_refine import (
    FrozenVGG16Rollout,
    gray_self_sim_loss,
    haar_hf_lf_ratio_2level,
)
from exp.v27_single.bare_q32 import load_cache
from exp.v28_unified import OUT, ROOT, SEED_TRAIN, RawWriterV28, atomic_save, peak_vram_gb, recipe_hash, run_measure_v28
from exp.v28_unified.student_models import PixelBridgeV28, build_student_model
from src.eval.metrics import paired_cluster_ci
from src.utils import decode_s10, load_s10_decoder
from src.utils.protocol import EVAL_SEED, pick_wrong_cross_tid
from src.utils.runtime import bf16_autocast, read_json, save_json, utc


def _cosine_lr_mult(step: int, total_steps: int, min_ratio: float = 0.15) -> float:
    if total_steps <= 1:
        return 1.0
    progress = float(step) / float(max(total_steps - 1, 1))
    return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * progress))


def _set_adam_lr_scale(opt: torch.optim.Optimizer, base_lrs: list[float], scale: float) -> None:
    for pg, base_lr in zip(opt.param_groups, base_lrs):
        pg["lr"] = base_lr * scale


@torch.no_grad()
def build_or_load_pure_train_subschool_bank(
    tr: dict,
    c21_tr_q32: torch.Tensor,
    dec_bf16: nn.Module,
    vgg_ro: FrozenVGG16Rollout,
    swd_ext: MultiScalePatchSWDExtractor,
    vgg_ext: UnnormalizedVGGFeatureExtractor,
    device: str = "cuda",
    k_sub: int = 12,
) -> dict:
    """Cluster clean train_4400 paintings into K=12 vivid sub-schools strictly referee-free."""
    bank_p = OUT / f"cache/pure_train_subschool_bank_k{k_sub}.pt"
    r_hash = recipe_hash({"kind": "pure_train_subschool", "k_sub": k_sub, "seed": SEED_TRAIN})
    if bank_p.exists():
        ck = torch.load(bank_p, map_location="cpu", weights_only=False)
        if ck.get("recipe_hash") == r_hash:
            print(f"[TTO Bank] Reusing {bank_p.name}", flush=True)
            return ck

    t0 = time.monotonic()
    tr_z = tr["z"]
    tr_sid = tr["sid"].long()
    tr_y64 = tr["y_64"].float()
    # Simple luminance HF proxy strictly from y_64 to filter blank scans before decoding
    hf_energy = (tr_y64 - F.avg_pool2d(tr_y64, 5, stride=1, padding=2)).flatten(1).square().mean(dim=1)

    g_b = torch.Generator().manual_seed(SEED_TRAIN + 777)
    cand_indices: list[torch.Tensor] = []
    feats_list, sid_list = [], []
    per_style = 180

    for s in range(5):
        ms = (tr_sid == s).nonzero(as_tuple=True)[0]
        non_blank = ms[hf_energy[ms] > hf_energy[ms].quantile(0.20)]
        perm = torch.randperm(len(non_blank), generator=g_b)[:per_style]
        sel = non_blank[perm]
        cand_indices.append(sel)
        for sl in sel.split(16):
            rgb = decode_s10(dec_bf16, tr_z[sl].to(device).float(), batch=16, dtype=torch.bfloat16, output_cpu=False).clamp(0.0, 1.0)
            with bf16_autocast(device):
                vf = vgg_ext(rgb, multi_scale=True).float()
            feats_list.append(vf.cpu())
            sid_list.append(tr_sid[sl])

    X = torch.cat(feats_list, dim=0)  # (N, D)
    Y_sid = torch.cat(sid_list, dim=0)
    mu = X.mean(dim=0, keepdim=True)
    sd = X.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-4)
    Xn = ((X - mu) / sd).clamp(-6.0, 6.0)

    # Closed-form 5-way ridge classifier on pure VGG+Oklab descriptors -> 0 referee leakage
    Y_oh = F.one_hot(Y_sid, 5).float() * 2.0 - 1.0
    gram = Xn.T @ Xn + 50.0 * torch.eye(Xn.shape[1])
    W_cls = torch.linalg.solve(gram, Xn.T @ Y_oh)
    logits = Xn @ W_cls  # (N, 5)

    pick_ids = torch.zeros(5, k_sub, dtype=torch.long)
    offset = 0
    for s in range(5):
        sel = cand_indices[s]
        n_s = len(sel)
        Xs = F.normalize(Xn[offset:offset + n_s], dim=-1)
        ls = logits[offset:offset + n_s]
        offset += n_s

        oh = F.one_hot(torch.full((n_s,), s), 5).bool()
        purity = ls[:, s] - ls.masked_fill(oh, -1e9).max(dim=1).values
        centers, labels = _kmeans_unit(Xs, k=k_sub, iters=35, seed=SEED_TRAIN + 800 + s)
        for c in range(k_sub):
            cm = (labels == c).nonzero(as_tuple=True)[0]
            if len(cm) == 0:
                cm = torch.arange(n_s)
            cent_sim = (Xs[cm] @ centers[c]).clamp(-1.0, 1.0)
            score = purity[cm] + 0.5 * cent_sim
            best_local = cm[score.argmax()]
            pick_ids[s, c] = sel[int(best_local.item())]

    # Pre-extract frozen exemplar tensors for 5*K=60 vivid train paintings
    flat_ids = pick_ids.reshape(-1)
    ex_z = tr_z[flat_ids].to(device).float()
    ex_q32 = F.normalize(c21_tr_q32[flat_ids].float().flatten(2).transpose(1, 2), dim=-1).half()

    g_p = torch.Generator().manual_seed(SEED_TRAIN + 909)
    vgg_projs = tuple(
        torch.linalg.qr(torch.randn(ch, 64, generator=g_p), mode="reduced").Q
        for ch in (64, 128, 256, 512)
    )

    ex_gray_scene, ex_r42_16, ex_oklab, ex_haar = [], [], [], []
    ex_vgg_toks = [[] for _ in range(4)]
    ex_cnx_toks = {k: [] for k in ("rgb", "s1", "s2", "s3")}

    for sl in torch.arange(len(flat_ids)).split(10):
        rgb = decode_s10(dec_bf16, ex_z[sl], batch=10, dtype=torch.bfloat16, output_cpu=False).clamp(0.0, 1.0)
        with bf16_autocast(device):
            r42 = vgg_ro.forward_gray_r42(rgb).float()
            f1, f2, f3, f4 = vgg_ro.forward_style(rgb)
            ct = swd_ext.extract_tokens(rgb)
        ex_gray_scene.append(F.normalize(F.adaptive_avg_pool2d(r42, 4).flatten(1), dim=-1).cpu().half())
        ex_r42_16.append(F.normalize(F.adaptive_avg_pool2d(r42, 16).flatten(2).transpose(1, 2), dim=-1).cpu().half())
        for j, f in enumerate((f1, f2, f3, f4)):
            tk = F.adaptive_avg_pool2d(f.float(), 16).flatten(2).transpose(1, 2)
            ex_vgg_toks[j].append(tk.cpu().half())
        for k, v in ct.items():
            ex_cnx_toks[k].append(v.float().cpu().half())
        ex_oklab.append(extract_color_fiber_27d(rgb.float()).cpu().half())
        ex_haar.append(haar_hf_lf_ratio_2level(rgb.float()).cpu().half())

    bank = {
        "recipe_hash": r_hash,
        "k_sub": k_sub,
        "pick_ids": pick_ids,
        "ex_gray_scene": torch.cat(ex_gray_scene, 0).view(5, k_sub, -1),
        "ex_r42_16": torch.cat(ex_r42_16, 0).view(5, k_sub, 256, -1),
        "ex_q32_16": ex_q32.view(5, k_sub, 256, 32).cpu(),
        "ex_vgg_toks": tuple(torch.cat(x, 0).view(5, k_sub, 256, -1) for x in ex_vgg_toks),
        "ex_cnx_toks": {k: torch.cat(v, 0).view(5, k_sub, v[0].shape[1], v[0].shape[2]) for k, v in ex_cnx_toks.items()},
        "ex_oklab": torch.cat(ex_oklab, 0).view(5, k_sub, 27),
        "ex_haar": torch.cat(ex_haar, 0).view(5, k_sub),
        "vgg_projs": vgg_projs,
        "wall_sec": round(time.monotonic() - t0, 1),
    }
    atomic_save(bank, bank_p)
    print(f"[TTO Bank] Built pure-train K={k_sub} subschool bank (0 DINO/CSD) in {bank['wall_sec']}s", flush=True)
    return bank


@torch.no_grad()
def _match_gray_subschool(src_r42: torch.Tensor, ex_gray_scene: torch.Tensor, tid: torch.Tensor) -> torch.Tensor:
    sc = F.normalize(F.adaptive_avg_pool2d(src_r42.float(), 4).flatten(1), dim=-1)
    sim = torch.einsum("bd,bkd->bk", sc, ex_gray_scene[tid])
    return sim.argmax(dim=1)


def _soft_q32_local_s3_loss(
    cur_s3: torch.Tensor,
    ref_s3: torch.Tensor,
    src_q32_n: torch.Tensor,
    ref_q32_n: torch.Tensor,
    p_s3: torch.Tensor,
) -> torch.Tensor:
    """Soft semantic-region correspondence Patch-SWD on ConvNeXt 16x16 s3 tokens (fallback=global where unaligned)."""
    sim = torch.bmm(src_q32_n, ref_q32_n.transpose(1, 2))  # (B, 256, 256)
    conf = ((sim.max(dim=-1).values - 0.35) / 0.35).clamp(0.0, 1.0).unsqueeze(-1)  # (B, 256, 1)
    w_tr = F.softmax(sim / 0.12, dim=-1)
    aligned_ref = torch.bmm(w_tr, ref_s3)  # (B, 256, 384)
    return (conf * (cur_s3 @ p_s3 - aligned_ref @ p_s3).abs()).mean()


TTO_ARM_CONFIGS = {
    "TTO1_10step_z_bare": {
        "n_steps": 10,
        "mode": "z",
        "struct_lock": False,
        "subschool_cnx": False,
        "lr_z": 0.075,
        "rho_y_edge": 0.17,
        "rho_y_low": 0.28,
        "rho_c": 0.56,
        "mom_w": 0.32,
        "swd_w": 0.55,
        "pullback_w": 0.015,
    },
    "TTO2_z_struct_lock": {
        "n_steps": 10,
        "mode": "z",
        "struct_lock": True,
        "subschool_cnx": False,
        "lr_z": 0.075,
        "rho_y_edge": 0.065,
        "rho_y_low": 0.14,
        "rho_c": 0.58,
        "mom_w": 0.35,
        "swd_w": 0.62,
        "pullback_w": 0.015,
        "w_gss": 5.5,
        "w_lcn": 5.5,
    },
    "TTO3_z_subschool_cnx": {
        "n_steps": 10,
        "mode": "z",
        "struct_lock": True,
        "subschool_cnx": True,
        "lr_z": 0.080,
        "rho_y_edge": 0.070,
        "rho_y_low": 0.16,
        "rho_c": 0.62,
        "w_cnx": 0.72,
        "w_vgg_swd": 0.45,
        "w_oklab": 2.6,
        "w_haar": 0.10,
        "w_local_s3": 0.40,
        "pullback_w": 0.010,
        "w_gss": 5.5,
        "w_lcn": 5.5,
    },
    "TTO4_unet_manifold": {
        "n_steps": 10,
        "mode": "unet",
        "struct_lock": True,
        "subschool_cnx": True,
        "lr_dz": 0.050,
        "lr_dq32": 0.065,
        "lr_deps": 0.130,
        "lr_dcond": 0.040,
        "reg_w": 0.005,
        "w_cnx": 0.72,
        "w_vgg_swd": 0.45,
        "w_oklab": 2.6,
        "w_haar": 0.10,
        "w_local_s3": 0.40,
        "w_gss": 5.5,
        "w_lcn": 5.5,
    },
}


class _TTORunner:
    """Holds shared frozen models/banks across TTO probe and 4-arm ablation."""

    def __init__(self, ckpt_path: Path, device: str = "cuda"):
        self.device = device
        saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.raw_stats = saved["raw_stats"].to(device)
        self.priors = {k: v.to(device) for k, v in saved["priors"].items()}
        self.model = build_student_model("unet", self.raw_stats.shape[-1], device=device).eval()
        self.model.load_state_dict(saved["ema"])
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.pixel = PixelBridgeV28(device=device)
        self.dec_bf16 = self.pixel.decoder
        self.vgg_ro = FrozenVGG16Rollout().to(device).eval()
        self.swd_ext = MultiScalePatchSWDExtractor(n_slices=96, seed=SEED_TRAIN + 606).to(device).eval()

        v18 = load_cache("runs/v18/cache/ccf_pool_cache.pt")
        tr = v18["train"]
        self.w_luma = v18["luma_proj"]["w"].to(device).float()
        self.b_luma = v18["luma_proj"]["b"].to(device).float()

        from exp.v28_unified.phase3_4_5 import _build_train_vgg_swd_refs
        self.pool_moms, self.pool_toks, self.pool_proj = _build_train_vgg_swd_refs(self.pixel, tr, device)

        c21_tr_q = load_cache("runs/v21/cache/v21_stiefel_ref_cache.pt")["train"]["q32"]
        self.bank = build_or_load_pure_train_subschool_bank(
            tr, c21_tr_q, self.dec_bf16, self.vgg_ro, self.swd_ext, self.pixel.vgg, device=device, k_sub=12,
        )
        self.ex_gray_scene = self.bank["ex_gray_scene"].to(device).float()
        self.ex_q32_16 = self.bank["ex_q32_16"].to(device).float()
        self.ex_vgg_toks = tuple(t.to(device).float() for t in self.bank["ex_vgg_toks"])
        self.ex_cnx_toks = {k: v.to(device).float() for k, v in self.bank["ex_cnx_toks"].items()}
        self.ex_oklab = self.bank["ex_oklab"].to(device).float()
        self.ex_haar = self.bank["ex_haar"].to(device).float()
        self.vgg_projs = tuple(p.to(device).float() for p in self.bank["vgg_projs"])

    def optimize_batch(
        self,
        zs: torch.Tensor,
        c64: torch.Tensor,
        q32: torch.Tensor,
        eps: torch.Tensor,
        tgt: torch.Tensor,
        cfg: dict,
        *,
        return_diag: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        dev = self.device
        B = len(zs)
        st_1 = torch.ones(B, device=dev, dtype=torch.float32)
        with torch.no_grad():
            z0 = self.model(zs, c64, q32, tgt, st_1, self.raw_stats, self.priors, noise=eps)
            src_rgb = decode_s10(self.dec_bf16, zs, batch=B, dtype=torch.bfloat16, enable_grad=False, output_cpu=False).clamp(0.0, 1.0)
            with bf16_autocast(dev):
                src_r42 = self.vgg_ro.forward_gray_r42(src_rgb).float().detach()
            best_c = _match_gray_subschool(src_r42, self.ex_gray_scene, tgt) if cfg["subschool_cnx"] else None

        if cfg["subschool_cnx"]:
            assert best_c is not None
            ref_vgg = [ev[tgt, best_c] for ev in self.ex_vgg_toks]
            ref_cnx = {k: v[tgt, best_c] for k, v in self.ex_cnx_toks.items()}
            ref_ok = self.ex_oklab[tgt, best_c]
            ref_haar = self.ex_haar[tgt, best_c]
            ref_q32 = self.ex_q32_16[tgt, best_c]
            src_q32_n = F.normalize(q32.flatten(2).transpose(1, 2), dim=-1)

        P_Y, P_C = self.priors["P_Y"], self.priors["P_C"]
        n_steps = int(cfg["n_steps"])

        if cfg["mode"] == "z":
            z_param = z0.detach().clone().requires_grad_(True)
            opt = torch.optim.Adam([z_param], lr=cfg["lr_z"], betas=(0.8, 0.99))
            base_lrs = [cfg["lr_z"]]
        else:
            dz_in = torch.zeros_like(z0, requires_grad=True)
            deps = torch.zeros_like(eps, requires_grad=True)
            dq32 = torch.zeros_like(q32, requires_grad=True)
            dcond = torch.zeros((B, 192), device=dev, dtype=torch.float32, requires_grad=True)
            params_groups = [
                {"params": [dz_in], "lr": cfg["lr_dz"]},
                {"params": [deps], "lr": cfg["lr_deps"]},
                {"params": [dq32], "lr": cfg["lr_dq32"]},
                {"params": [dcond], "lr": cfg["lr_dcond"]},
            ]
            opt = torch.optim.Adam(params_groups, betas=(0.8, 0.99))
            base_lrs = [pg["lr"] for pg in opt.param_groups]

        diag: dict[str, list[float]] = {"loss": [], "gss": [], "lcn": [], "style": []}

        for step in range(n_steps):
            _set_adam_lr_scale(opt, base_lrs, _cosine_lr_mult(step, n_steps, min_ratio=0.15))
            opt.zero_grad(set_to_none=True)

            with torch.enable_grad():
                if cfg["mode"] == "z":
                    z_cur = z_param
                else:
                    z_cur = self.model.forward_latent_opt(
                        zs, c64, q32, tgt, st_1, self.raw_stats, self.priors,
                        noise=eps + deps, dz_in=dz_in, dq32=dq32, dcond=dcond,
                    )

                rgb_cur = decode_s10(
                    self.dec_bf16, z_cur, dtype=torch.bfloat16, batch=B, enable_grad=True, output_cpu=False
                ).clamp(0.0, 1.0)

                if cfg["subschool_cnx"]:
                    with bf16_autocast(dev):
                        f1, f2, f3, f4 = self.vgg_ro.forward_style(rgb_cur)
                        cnx_cur = self.swd_ext.extract_tokens(rgb_cur)
                    l_vgg_swd = sum(
                        sliced_wasserstein_1d(F.adaptive_avg_pool2d(f.float(), 16).flatten(2).transpose(1, 2), rt, proj).mean()
                        for f, rt, proj in zip((f1, f2, f3, f4), ref_vgg, self.vgg_projs)
                    ) / 4.0
                    l_cnx = (
                        1.1 * sliced_wasserstein_1d(cnx_cur["rgb"].float(), ref_cnx["rgb"], self.swd_ext.p_rgb).mean()
                        + 0.35 * sliced_wasserstein_1d(cnx_cur["s1"].float(), ref_cnx["s1"], self.swd_ext.p_s1).mean()
                        + 0.45 * sliced_wasserstein_1d(cnx_cur["s2"].float(), ref_cnx["s2"], self.swd_ext.p_s2).mean()
                        + 0.55 * sliced_wasserstein_1d(cnx_cur["s3"].float(), ref_cnx["s3"], self.swd_ext.p_s3).mean()
                    )
                    l_local = _soft_q32_local_s3_loss(cnx_cur["s3"].float(), ref_cnx["s3"], src_q32_n, ref_q32, self.swd_ext.p_s3)
                    l_ok = (extract_color_fiber_27d(rgb_cur.float()) - ref_ok).abs().mean()
                    l_haar = (haar_hf_lf_ratio_2level(rgb_cur.float()) - ref_haar).abs().mean()
                    l_style = (
                        cfg["w_cnx"] * l_cnx
                        + cfg["w_vgg_swd"] * l_vgg_swd
                        + cfg["w_oklab"] * l_ok
                        + cfg["w_haar"] * l_haar
                        + cfg["w_local_s3"] * l_local
                    )
                else:
                    self.pixel.layers.clear()
                    cur_f = self.pixel.vgg(rgb_cur.float(), multi_scale=False).float()
                    cur_t = tuple(F.adaptive_avg_pool2d(x, 16).float().flatten(2).transpose(1, 2) for x in self.pixel.layers)
                    self.pixel.layers.clear()
                    l_mom = F.smooth_l1_loss((cur_f - self.pool_moms[tgt]) / self.pixel.scale, torch.zeros_like(cur_f), beta=0.20)
                    l_swd = sum(
                        ((a @ p).sort(dim=1).values - (b[tgt] @ p).sort(dim=1).values).abs().mean()
                        for a, b, p in zip(cur_t, self.pool_toks, self.pool_proj)
                    ) / 3.0
                    l_style = cfg["mom_w"] * l_mom + cfg["swd_w"] * l_swd

                l_gss = z_cur.new_zeros(())
                l_lcn = z_cur.new_zeros(())
                loss = l_style
                if cfg["struct_lock"]:
                    with bf16_autocast(dev):
                        r42_cur = self.vgg_ro.forward_gray_r42(rgb_cur).float()
                    l_gss = gray_self_sim_loss(r42_cur, src_r42)
                    l_lcn = latent_lcn_cost(z_cur, zs, self.w_luma, self.b_luma).mean()
                    loss = loss + cfg["w_gss"] * l_gss + cfg["w_lcn"] * l_lcn

                if cfg["mode"] == "z":
                    loss = loss + cfg["pullback_w"] * (z_cur - z0).square().mean()
                else:
                    loss = loss + cfg["reg_w"] * (
                        dz_in.square().mean() + 0.5 * deps.square().mean() + 0.5 * dq32.square().mean() + 0.25 * dcond.square().mean()
                    )

                loss.backward()

            if cfg["mode"] == "z":
                if z_param.grad is not None:
                    z_param.grad.data = _smooth_grad_3x3(z_param.grad.data, blend=0.20)
                opt.step()
                with torch.no_grad():
                    z_param.data = project_anisotropic_trust(
                        z_param.data, z0, P_Y, P_C,
                        rho_y_edge=cfg["rho_y_edge"], rho_y_low=cfg["rho_y_low"], rho_c=cfg["rho_c"],
                    )
            else:
                opt.step()

            if return_diag and step in (0, n_steps // 2, n_steps - 1):
                diag["loss"].append(round(float(loss.item()), 4))
                diag["style"].append(round(float(l_style.item()), 4))
                diag["gss"].append(round(float(l_gss.item()), 4))
                diag["lcn"].append(round(float(l_lcn.item()), 4))

        with torch.no_grad():
            if cfg["mode"] == "z":
                z_out = z_param.detach()
            else:
                z_out = self.model.forward_latent_opt(
                    zs, c64, q32, tgt, st_1, self.raw_stats, self.priors,
                    noise=eps + deps.detach(), dz_in=dz_in.detach(), dq32=dq32.detach(), dcond=dcond.detach(),
                ).detach()
            if return_diag:
                dz = z_out - z0
                dy = torch.einsum("ij,bjhw->bihw", P_Y, dz)
                dc = torch.einsum("ij,bjhw->bihw", P_C, dz)
                diag["dz_y_rms"] = [round(float(dy.square().mean().sqrt().item()), 4)]
                diag["dz_c_rms"] = [round(float(dc.square().mean().sqrt().item()), 4)]
                diag["nan_count"] = [int(torch.isnan(z_out).sum().item())]
        return z_out, diag


def run_tto_probe(device: str = "cuda") -> dict:
    """16-sample train-only stability, VRAM, speed, and loss-scale probe (0 test/referee peek)."""
    ckpt_p = OUT / "students/Final_5k_seed0/Final_5k_seed0_final.pt"
    runner = _TTORunner(ckpt_p, device=device)
    v18 = load_cache("runs/v18/cache/ccf_pool_cache.pt")
    tr = v18["train"]
    c21_tr = load_cache("runs/v21/cache/v21_stiefel_ref_cache.pt")["train"]

    idx = torch.arange(0, 16)
    zs = tr["z"][idx].to(device).float()
    c64 = tr["c_64"][idx].to(device).float()
    q32 = c21_tr["q32"][idx].to(device).float()
    tgt = (tr["sid"][idx].to(device).long() + 1) % 5
    eps = torch.randn(zs.shape, generator=torch.Generator().manual_seed(EVAL_SEED)).to(device)

    report = {}
    probe_bs = 6
    for arm_name, cfg in TTO_ARM_CONFIGS.items():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        diags = []
        for st in range(0, len(zs), probe_bs):
            sl = slice(st, min(st + probe_bs, len(zs)))
            _, d = runner.optimize_batch(zs[sl], c64[sl], q32[sl], eps[sl], tgt[sl], cfg, return_diag=True)
            diags.append(d)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        vram = round(peak_vram_gb(device), 3)
        ms_img = round(dt * 1000.0 / 16.0, 1)
        report[arm_name] = {"ms_per_img": ms_img, "est_750_min": round(dt * (750.0 / 16.0) / 60.0 + 0.4, 2), "peak_vram_gb": vram, "diag_b0": diags[0]}
        print(f"[TTO PROBE] {arm_name:22s} | {ms_img:6.1f} ms/img (~{report[arm_name]['est_750_min']}m/750) | VRAM={vram:.2f} GB | {diags[0]}", flush=True)

    save_json(OUT / "tto_probe_report.json", report)
    del runner
    gc.collect()
    torch.cuda.empty_cache()
    return report


def _plot_tto_pareto(arms: dict, out_dir: Path) -> None:
    from exp.v28_unified.phase3_4_5 import _frontier_indices, _literature_points

    lit = _literature_points()
    lit_xy_c = [(p["dino_c"], p["dino_s"]) for p in lit]
    lit_f = _frontier_indices(lit_xy_c)
    lit_xy_t = [(p["dino_c"], p["csd_top1"] * 100.0) for p in lit]
    lit_ft = _frontier_indices(lit_xy_t)

    hist = [
        ("PRISM-1", 0.744, 0.444, 44.33),
        ("V23 HiC", 0.631, 0.533, 59.0),
        ("V20 D2", 0.605, 0.567, 60.5),
        ("V22 Peak1500", 0.578, 0.582, 76.8),
        ("StyleGallery", 0.358, 0.574, 92.67),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4), dpi=180)
    seq_keys = ["Final_5k_l100", "s200_3step", "TTO1_10step_z_bare", "TTO2_z_struct_lock", "TTO3_z_subschool_cnx", "TTO4_unet_manifold"]
    labels = ["1-NFE (s=1)", "3st Bare (s=2)", "TTO1 10st z", "TTO2 +Struct", "TTO3 +Sub/CNX", "TTO4 U-Net"]
    colors = ["#1b6b50", "#b2182b", "#7f8c8d", "#2b5c8f", "#8e44ad", "#d95f02"]

    for ax_i, (ax, yk, scale, yl) in enumerate([
        (axes[0], "dino_s", 1.0, "DINO-S ↑"),
        (axes[1], "csd_top1", 100.0, "CSD Top-1 (%) ↑"),
    ]):
        for p in lit:
            ax.scatter(p["dino_c"], p[yk] * scale, s=26, color="#666", alpha=0.35, edgecolors="none", zorder=2)
        fl = lit_f if ax_i == 0 else lit_ft
        arr = lit_xy_c if ax_i == 0 else lit_xy_t
        ax.plot([arr[i][0] for i in fl], [arr[i][1] for i in fl], "--o", color="#555", lw=1.1, ms=3.5, alpha=0.6, label="External lit Pareto")
        for nm, dc, ds, tp in hist:
            yv = ds if ax_i == 0 else tp
            ax.scatter(dc, yv, marker="X", s=62, color="#b8860b", zorder=4)
            ax.annotate(nm, (dc, yv), xytext=(4, 3), textcoords="offset points", fontsize=7.8, color="#8a6500")

        for k, lb, col in zip(seq_keys, labels, colors):
            if k not in arms:
                continue
            r = arms[k]
            xv, yv = r["dino_c"], r[yk] * scale
            ax.scatter(xv, yv, s=85, color=col, edgecolor="white", linewidth=0.9, zorder=7, label=lb if ax_i == 0 else None)
            ax.annotate(lb, (xv, yv), xytext=(5, 4), textcoords="offset points", fontsize=8.0, color=col, fontweight="bold")

        ax.set_xlabel("DINO-C ↑", fontsize=10.5)
        ax.set_ylabel(yl, fontsize=10.5)
        ax.grid(alpha=0.22)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylim(0.28, 0.62)
    axes[1].set_ylim(30, 102)
    axes[0].legend(frameon=False, fontsize=7.6, loc="lower left")
    fig.suptitle("V28 Test-Time Manifold Optimization 4-Arm Ablation (Frozen Final_5k_seed0) · Cross600", fontsize=11.5, fontweight="bold")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(out_dir / f"pareto_v28_tto.{ext}", bbox_inches="tight")
    plt.close(fig)


def _export_tto_visual_strip(out_png: Path) -> None:
    tasks, _, _ = tasks_and_sources()
    wanted = [
        ("Impressionism", "Early_Renaissance"),
        ("Early_Renaissance", "Ukiyo_e"),
        ("Early_Renaissance", "Minimalism"),
    ]
    picks = [next(t for t in tasks if t["source_style"] == ss and t["target_style"] == ts) for ss, ts in wanted]
    arms = [
        ("Final_5k_seed0_l100", "1-NFE (s=1.0)"),
        ("Final_5k_s200", "3st Bare (s=2.0)"),
        ("TTO1_10step_z_bare", "TTO1 10st z"),
        ("TTO2_z_struct_lock", "TTO2 +Struct"),
        ("TTO3_z_subschool_cnx", "TTO3 +Sub/CNX"),
        ("TTO4_unet_manifold", "TTO4 U-Net"),
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


def run_tto_ablation(device: str = "cuda", batch_size: int = 6) -> dict:
    """Run all 4 10-step Adam TTO arms (+ TTO4_wrong counterfactual) on full test750."""
    ckpt_p = OUT / "students/Final_5k_seed0/Final_5k_seed0_final.pt"
    runner = _TTORunner(ckpt_p, device=device)

    ts = load_cache("runs/v18/cache/test750_struct_cache.pt")
    q = load_cache("runs/v21/cache/v21_stiefel_ref_cache.pt")["test750"]["q32"]
    tasks, src, _ = tasks_and_sources()
    sid_all = torch.tensor([t["source_style_id"] for t in tasks])
    tid_all = torch.tensor([t["target_style_id"] for t in tasks])
    eps_fixed = torch.randn(src.shape, generator=torch.Generator().manual_seed(EVAL_SEED))
    dec_fp16 = load_s10_decoder(device)

    run_specs = [(k, cfg, False) for k, cfg in TTO_ARM_CONFIGS.items()]
    run_specs.append(("TTO4_unet_manifold_wrong", TTO_ARM_CONFIGS["TTO4_unet_manifold"], True))

    latencies = {}
    results = {}

    for arm_name, cfg, is_wrong in run_specs:
        m_dir = OUT / "measured" / arm_name
        if (m_dir / "summary_v28.json").exists():
            results[arm_name] = run_measure_v28(arm_name, device=device)
            continue

        labels_all = pick_wrong_cross_tid(sid_all, tid_all) if is_wrong else tid_all
        wr = RawWriterV28(arm_name, dec=dec_fp16, device=device)
        t0 = time.monotonic()
        for st in range(0, len(tasks), batch_size):
            sl = slice(st, min(st + batch_size, len(tasks)))
            z_out, _ = runner.optimize_batch(
                src[sl].to(device).float(),
                ts["c_64"][sl].to(device).float(),
                q[sl].to(device).float(),
                eps_fixed[sl].to(device).float(),
                labels_all[sl].to(device),
                cfg,
                return_diag=False,
            )
            wr.write_batch(z_out, tasks[sl])
        elapsed = round(time.monotonic() - t0, 2)
        wr.finalize(arm=arm_name, n_steps=cfg["n_steps"], mode=cfg["mode"], wrong=is_wrong, elapsed_s=elapsed, same_bypass=False)
        latencies[arm_name] = {"elapsed_s": elapsed, "ms_per_img_batch": round(elapsed * 1000.0 / len(tasks), 2)}
        print(f"[TTO GEN] {arm_name} finished in {elapsed}s ({latencies[arm_name]['ms_per_img_batch']} ms/img)", flush=True)
        results[arm_name] = run_measure_v28(arm_name, device=device)

    # b=1 latency measurement for TTO3 and TTO4
    b1_latency = {}
    zs1 = src[:1].to(device).float()
    c1 = ts["c_64"][:1].to(device).float()
    q1 = q[:1].to(device).float()
    e1 = eps_fixed[:1].to(device).float()
    t1 = tid_all[:1].to(device)
    for k in ("TTO3_z_subschool_cnx", "TTO4_unet_manifold"):
        runner.optimize_batch(zs1, c1, q1, e1, t1, TTO_ARM_CONFIGS[k])
        torch.cuda.synchronize()
        tb = time.perf_counter()
        n_rep = 6
        for _ in range(n_rep):
            z_b1, _ = runner.optimize_batch(zs1, c1, q1, e1, t1, TTO_ARM_CONFIGS[k])
            _ = decode_s10(dec_fp16, z_b1, batch=1, output_cpu=False)
        torch.cuda.synchronize()
        b1_latency[k] = round((time.perf_counter() - tb) * 1000.0 / n_rep, 1)

    del runner, dec_fp16
    gc.collect()
    torch.cuda.empty_cache()

    seed0_rep = read_json(OUT / "students/Final_5k_seed0/results.json")
    s200_rep = read_json(OUT / "phase5_knob2_audit.json")["knob2_arms"]["s200"]
    results["Final_5k_l100"] = seed0_rep["arms"]["l100"]
    results["s200_3step"] = s200_rep

    # Paired bootstrap CIs: read rows cleanly for paired_cluster_ci
    r_s200 = read_json(OUT / "measured/Final_5k_s200/eval.json")["rows"]
    r_t2 = read_json(OUT / "measured/TTO2_z_struct_lock/eval.json")["rows"]
    r_t3 = read_json(OUT / "measured/TTO3_z_subschool_cnx/eval.json")["rows"]
    r_t4 = read_json(OUT / "measured/TTO4_unet_manifold/eval.json")["rows"]
    ci = {
        "TTO2_minus_s200": paired_cluster_ci(r_t2, r_s200),
        "TTO3_minus_s200": paired_cluster_ci(r_t3, r_s200),
        "TTO4_minus_s200": paired_cluster_ci(r_t4, r_s200),
        "TTO4_minus_TTO3": paired_cluster_ci(r_t4, r_t3),
    }

    _plot_tto_pareto(results, OUT)
    _export_tto_visual_strip(OUT / "visual_audit_strip_tto.png")

    best_tto = max(
        ("TTO1_10step_z_bare", "TTO2_z_struct_lock", "TTO3_z_subschool_cnx", "TTO4_unet_manifold"),
        key=lambda k: results[k]["dino_s"] + 0.5 * results[k]["dino_c"],
    )
    audit = {
        "updated_utc": utc(),
        "arms": results,
        "batch_latencies": latencies,
        "b1_e2e_ms_per_img": b1_latency,
        "bootstrap_ci": ci,
        "stop_loss_check": {
            "best_arm": best_tto,
            "best_dino_c": results[best_tto]["dino_c"],
            "best_dino_s": results[best_tto]["dino_s"],
            "beats_c058_s0565": bool(results[best_tto]["dino_c"] >= 0.58 and results[best_tto]["dino_s"] >= 0.565),
        },
    }
    save_json(OUT / "phase5_tto_audit.json", audit)
    print("[Phase 5 TTO] COMPLETE ->", OUT / "phase5_tto_audit.json", flush=True)
    return audit
