r"""P0 第 1 步: AnisotropicS10RefTeacher — 免标签离线任意参考教师 + 熔断验收.

用户 2026-11-22 批准 (升级版 C 第 1 步, ~40 min): 保留 AnisotropicS10Teacher 已验证的
核心几何 (P_Y/P_C 各向异性信任域, latent_lcn_cost 结构预算, S10-BF16 平滑梯度),
把三个 5 类 tid 依赖模块替换为 100% 免标签的逐参考图目标:

  - 色度热启动: apply_latent_chroma_shift(tid) → 对齐单张参考图 latent 的 3D 色度矩
    (_chroma_align_to_ref, 无 tid);
  - 梯度目标: 删 vgg_head(5)/vgg_protos/ld4(5), 直接对 S10(z) 与 x_ref 计算
    vgg_ext 全局描述子距离 (含 Oklab/moments/低秩 Gram) 梯度;
  - 信任域/结构预算/eta 配置沿用 T4 (rho_y_edge=0.075, rho_y_low=0.13, rho_c=0.25,
    lcn_budget=0.028, etas=(0.060, 0.115), m_steps=2, alpha_chroma=0.20).

熔断验收 (红线 1): 150 Cross600 对 (每目标画派 30, 参考 = 该画派 SG-5 exemplar,
1-shot 语义), 直测 Top-1 ≥ 45% / 五画派最低 ≥ 20% / DINO-C ≥ 0.62 → RuntimeError.

用法:  python -m exp.v26.gpu4090.p0_ref_teacher --pool ../pseudo_gt_24h/orig_mean_fp16
"""
from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from exp.v26 import OUT, ROOT, sanitize_cuda_env, save_json, utc
from exp.v26.gpu4090.p0_common import STYLES, _pil

N_FUSE_PER_STYLE = 30


def _chroma_align_to_ref(z: torch.Tensor, z_ref: torch.Tensor,
                         priors: dict, alpha: float) -> torch.Tensor:
    """apply_latent_chroma_shift 的免标签版: 目标色度矩来自单张参考 latent (非 tid 先验)."""
    if alpha <= 0.0:
        return z
    dev = z.device
    P_Y = priors["P_Y"].to(dev)
    P_C = priors["P_C"].to(dev)
    zr = z_ref.to(dev).float()
    zc_r = torch.einsum("ij,bjhw->bihw", P_C, zr)
    mu_r = zc_r.mean(dim=(2, 3), keepdim=True)
    sd_r = zc_r.std(dim=(2, 3), keepdim=True).clamp_min(1e-4)
    zy = torch.einsum("ij,bjhw->bihw", P_Y, z.float())
    zc = torch.einsum("ij,bjhw->bihw", P_C, z.float())
    mu_s = zc.mean(dim=(2, 3), keepdim=True)
    sd_s = zc.std(dim=(2, 3), keepdim=True).clamp_min(1e-4)
    zc_aligned = torch.einsum("ij,bjhw->bihw", P_C, (zc - mu_s) / sd_s * sd_r + mu_r)
    return zy + (1.0 - alpha) * zc + alpha * zc_aligned


class AnisotropicS10RefTeacher:
    """免标签任意参考教师: 信任域几何复用, 5 类模块 (vgg_head/protos/ld4) 全部移除."""

    T4_CFG = dict(alpha_chroma=0.20, m_steps=2, rho_y_edge=0.075, rho_y_low=0.13,
                  rho_c=0.25, lcn_budget=0.028, etas=(0.060, 0.115))

    def __init__(self, bundle: dict, device: str = "cuda"):
        import copy

        from exp.v19.aniso_teacher import (AnisotropicS10Teacher, aniso_rms_direction,
                                           latent_lcn_cost, project_anisotropic_trust,
                                           rms_norm, _smooth_grad_3x3)
        from exp.v19.fiber_duality import UnnormalizedVGGFeatureExtractor
        from src.utils import load_s10_decoder

        base = AnisotropicS10Teacher(bundle, device=device)
        self.device = device
        self.priors = base.priors
        self.w_luma, self.b_luma = base.w_luma, base.b_luma
        self.dec_fp16, self.dec_bf16 = base.dec_fp16, base.dec_bf16
        self.vgg_ext = base.vgg_ext
        self._aniso_rms_direction = aniso_rms_direction
        self._project = project_anisotropic_trust
        self._lcn_cost = latent_lcn_cost
        self._rms_norm = rms_norm
        self._smooth = _smooth_grad_3x3
        self.d_vgg_ref_cache: dict[int, torch.Tensor] = {}
        del base

    def _desc(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.vgg_ext(rgb.to(self.device), multi_scale=True).float()

    def ref_distance(self, rgb: torch.Tensor, d_ref: torch.Tensor) -> torch.Tensor:
        return F.pairwise_distance(self._desc(rgb), d_ref)

    @torch.no_grad()
    def prepare_ref(self, rgb_ref: torch.Tensor) -> torch.Tensor:
        return self._desc(rgb_ref)

    def ref_direction(self, z: torch.Tensor, d_ref: torch.Tensor,
                      micro: int = 6) -> torch.Tensor:
        """S10-BF16 反传的参考距离梯度 → 各向异性 RMS 方向 (同 s10_vgg_direction 结构)."""
        grads = []
        for st in range(0, len(z), micro):
            sl = slice(st, min(st + micro, len(z)))
            with torch.enable_grad():
                zr = z[sl].detach().float().requires_grad_(True)
                rgb = self._decode(zr)
                obj = -self.ref_distance(rgb, d_ref[sl])
                g_sub = torch.autograd.grad(obj.sum(), zr)[0].detach()
            grads.append(g_sub)
        g = torch.cat(grads, dim=0)
        g = self._smooth(g, blend=0.25)
        return self._aniso_rms_direction(g, self.priors["P_Y"], self.priors["P_C"],
                                         w_y=0.45, w_c=0.90)

    @torch.no_grad()
    def solve_ref_batch(self, z_base: torch.Tensor, z0_anchor: torch.Tensor,
                        z_src: torch.Tensor, d_ref: torch.Tensor, z_ref: torch.Tensor,
                        same_mask: torch.Tensor | None = None) -> torch.Tensor:
        """免标签信任域求解 (T4 配置): 色度对齐参考 + 参考距离梯度 + 结构预算."""
        cfg = self.T4_CFG
        dev = self.device
        z = _chroma_align_to_ref(z_base.to(dev).float(), z_ref, self.priors, cfg["alpha_chroma"])
        z_anc = _chroma_align_to_ref(z0_anchor.to(dev).float(), z_ref,
                                     self.priors, cfg["alpha_chroma"] * 0.7)
        zs = z_src.to(dev).float()
        base_lcn = self._lcn_cost(z, zs, self.w_luma, self.b_luma)
        max_lcn = torch.maximum(base_lcn * 1.45 + 0.015,
                                torch.full_like(base_lcn, cfg["lcn_budget"]))
        active = torch.ones(len(z), dtype=torch.bool, device=dev)
        if same_mask is not None:
            active = active & (~same_mask.to(dev))
            z[same_mask.to(dev)] = zs[same_mask.to(dev)]

        for _ in range(cfg["m_steps"]):
            if not bool(active.any()):
                break
            idx = active.nonzero(as_tuple=True)[0]
            zi, zai, zsi = z[idx], z_anc[idx], zs[idx]
            lim_i = max_lcn[idx]
            d_vec = self.ref_direction(zi, d_ref)
            with torch.no_grad():
                best_d = self.ref_distance(self._decode(zi), d_ref).detach()
            best_z = zi.clone()
            improved = torch.zeros(len(idx), dtype=torch.bool, device=dev)
            for eta in cfg["etas"]:
                cand = self._project(zi + eta * d_vec, zai, self.priors["P_Y"],
                                     self.priors["P_C"], rho_y_edge=cfg["rho_y_edge"],
                                     rho_y_low=cfg["rho_y_low"], rho_c=cfg["rho_c"])
                lcn_c = self._lcn_cost(cand, zsi, self.w_luma, self.b_luma)
                feas = lcn_c <= lim_i
                if not bool(feas.any()):
                    continue
                fi = feas.nonzero(as_tuple=True)[0]
                with torch.no_grad():
                    d_c = self.ref_distance(self._decode(cand[fi]), d_ref).detach()
                better = d_c < best_d[fi]
                if bool(better.any()):
                    sel = fi[better]
                    best_z[sel] = cand[sel]
                    best_d[sel] = d_c[better]
                    improved[sel] = True
            z[idx[improved]] = best_z[improved]
            active[idx[improved]] = False  # 收敛即停 (保守预算)
        return z

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec_bf16((z / 0.18215).to(torch.bfloat16)).float()


# ------------------------------------------------- 熔断验收 ----

@torch.no_grad()
def run_fuse(pool: str, device: str = "cuda") -> dict:
    from exp.v18 import tasks_and_sources
    from exp.v19.fiber_duality import build_or_load_v19_foundation
    from src.utils import FeatureStack, decode_s10, load_s10_decoder

    bundle = build_or_load_v19_foundation(pool, device=device)
    tasks, z_src_750, cache = tasks_and_sources()
    cross = [i for i, t in enumerate(tasks) if t["source_style_id"] != t["target_style_id"]]
    g = torch.Generator().manual_seed(20266500)
    by_style = {t: [] for t in range(5)}
    for i in cross:
        by_style[tasks[i]["target_style_id"]].append(i)
    sel = []
    for t in range(5):
        sel += [by_style[t][j] for j in torch.randperm(len(by_style[t]), generator=g)[:N_FUSE_PER_STYLE]]
    sel_t = torch.tensor(sel)
    tid = torch.tensor([tasks[i]["target_style_id"] for i in sel])

    c21 = torch.load(ROOT / "runs/v21/cache/v21_stiefel_ref_cache.pt",
                     map_location="cpu", weights_only=False)
    z_ref5 = c21["sg5"]["z"].float().to(device)  # (5,4,64,64) SG-5 exemplar latents
    teacher = AnisotropicS10RefTeacher(bundle, device=device)

    z_base = cache["z_n2"][sel_t].to(device).float()
    z0 = cache["z0"][sel_t].to(device).float()
    z_src = z_src_750[sel_t].to(device).float()
    rgb_ref5 = decode_s10(load_s10_decoder(device), z_ref5, batch=5).clamp(0.0, 1.0)
    d_ref_all = teacher.prepare_ref(rgb_ref5)  # (5, D) — vgg_ext 支持批
    outs = []
    for st in range(0, len(sel), 24):
        sl = slice(st, min(st + 24, len(sel)))
        same = torch.zeros(sl.stop - sl.start, dtype=torch.bool, device=device)
        z_out = teacher.solve_ref_batch(z_base[sl], z0[sl], z_src[sl],
                                        d_ref_all[tid[sl]], z_ref5[tid[sl]], same)
        outs.append(z_out.cpu())
    z_out = torch.cat(outs, 0)

    dec = load_s10_decoder(device)
    stack = FeatureStack(device)
    from PIL import Image as PILImage
    pred_all, dc_all = [], []
    src_dino = torch.load(ROOT / "runs/v6_r2/cache/tasks750_source_features.pt",
                          map_location="cpu", weights_only=False)
    for st in range(0, len(sel), 24):
        rgb = decode_s10(dec, z_out[st:st + 24].to(device).float(), batch=24).clamp(0.0, 1.0)
        feats = stack.extract([_pil(x) for x in rgb], batch=24)
        cs = feats["csd"] @ stack.csd_prototypes.T
        pred_all += cs.argmax(1).tolist()
        want_stems = [tasks[i]["source_stem"] for i in sel][st:st + 24]
        f_src = torch.stack([torch.as_tensor(src_dino["raw_dino"][s]) for s in want_stems])
        dc_all += torch.diagonal(feats["dino"] @ f_src.to(feats["dino"].device).T).tolist()
    per_style = {STYLES[t]: round(sum(1 for p, w in zip(pred_all, tid.tolist())
                                      if w == t and p == t) / N_FUSE_PER_STYLE, 4) for t in range(5)}
    top1 = round(sum(int(p == w) for p, w in zip(pred_all, tid.tolist())) / len(pred_all), 4)
    dino_c = round(sum(dc_all) / len(dc_all), 4)
    res = {"arm": "RefTeacher_fuse", "n_pairs": len(pred_all), "top1": top1,
           "per_style": per_style, "dino_c": dino_c,
           "gate": {"top1": 0.45, "min_style": 0.20, "dino_c": 0.62}, "updated_utc": utc()}
    ok = top1 >= 0.45 and min(per_style.values()) >= 0.20 and dino_c >= 0.62
    res["gate_pass"] = bool(ok)
    save_json(OUT / "gpu4090" / "p0_ref_teacher_fuse.json", res)
    print(f"[RefTeacher] Top-1={top1} per_style={per_style} DINO-C={dino_c} "
          f"gate={'PASS' if ok else 'FAIL'}", flush=True)
    if not ok:
        raise RuntimeError(
            f"[红线1] RefTeacher 熔断: Top-1={top1}<0.45 或 单类最低={min(per_style.values())}<0.20 "
            f"或 DINO-C={dino_c}<0.62 — 按两步止损跳到流 B (78k 开放域)")
    return res


def _pil_rgb(rgb: torch.Tensor):
    from PIL import Image as PILImage
    return PILImage.fromarray((rgb.permute(1, 2, 0).float().cpu().numpy() * 255)
                              .round().clip(0, 255).astype("uint8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    sanitize_cuda_env()
    run_fuse(pool=args.pool, device=args.device)


if __name__ == "__main__":
    main()
