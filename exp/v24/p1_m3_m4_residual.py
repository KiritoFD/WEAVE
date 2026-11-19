"""V24 P1-P2 (4090): M3 (PRISM-1 + 1.9M Palette+Texture 双路径残差网) & M4 (ConvNeXt 中尺度 + 动态拉格朗日内容预算)。

核心范式:
  z_b   = F_PRISM1(z_c, s)                                   [冻结高内容基座, DINO-C ~0.744]
  Delta = Delta_palette(z_b, u_ref) + Delta_texture(z_b, c64, d16, u_ref)   [~1.9M 轻量残差网]
  z_out = z_b + Delta

训练损失（100% 独立于 DINOv2 / CSD 裁判）:
  L = L_style (Oklab 27D 颜色分布 + ConvNeXt/RGB MultiScalePatchSWD 纹理分布)
    + lambda_c * L_content (ConvNeXt 32x32 & 16x16 空间结构余弦对齐源图)
    + lambda_r * L_residual (残差 L2 幅度正则)
其中 M4 使用训练集校准的 PRISM-1 内容预算 epsilon, 动态更新拉格朗日乘子:
  lambda_c <- clamp(lambda_c + eta * (L_c - epsilon), 0.1, 8.0)

用法 (4090):
  python -m exp.v24.p1_m3_m4_residual --pool ../pseudo_gt_24h/orig_mean_fp16
"""
from __future__ import annotations

import argparse
import copy
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import OUT, EVAL_SEED, ROOT, RawWriterV24, autocast, config_hash, export_raw_pngs, file_sha, git_sha, read_json, run_measure_v24, sanitize_cuda_env, save_json, seed_all, utc, validate_raw_meta
from .p0_m0_m2 import compute_prism1_latents
from exp.v18 import tasks_and_sources
from exp.v19.fiber_duality import StyleFiberEncoder, apply_latent_chroma_shift, build_or_load_v19_foundation, extract_color_fiber_27d
from exp.v19.routed_student import SpatiallyRoutedChordNet
from exp.v20.submode_patchswd import MultiScalePatchSWDExtractor, sliced_wasserstein_1d
from src.utils import decode_s10, load_s10_decoder
from src.utils.structure import dwt2_haar, idwt2_haar

P1_CKPT = ROOT / "runs/v19/students/s2_prism1_v2.pt"


class ResBlock2d(nn.Module):
    def __init__(self, ch: int, cond_dim: int):
        super().__init__()
        self.n1 = nn.GroupNorm(8, ch)
        self.film = nn.Linear(cond_dim, ch * 2)
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.n2 = nn.GroupNorm(8, ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        g, b = self.film(cond).chunk(2, dim=-1)
        h = F.silu(self.n1(x) * (1.0 + g[:, :, None, None]) + b[:, :, None, None])
        h = self.c1(h)
        h = F.silu(self.n2(h))
        return x + self.c2(h)


class PaletteTextureResidualNet(nn.Module):
    """~1.9M 轻量双路径残差网络: Delta = Delta_palette + Delta_texture (Haar 中高频分解 + 可学低频残差门)。"""

    def __init__(self, cond_dim: int = 128, ch: int = 96):
        super().__init__()
        self.cond_mlp = nn.Sequential(
            nn.Linear(128, cond_dim), nn.SiLU(inplace=True),
            nn.Linear(cond_dim, cond_dim), nn.SiLU(inplace=True),
        )
        # Palette 分支: 低分辨率色彩分布与明暗倾向平滑残差 (16x16 -> bilinear 64x64)
        self.pal_in = nn.Conv2d(4, 48, 3, stride=2, padding=1)  # 32x32
        self.pal_down = nn.Conv2d(48, 64, 3, stride=2, padding=1)  # 16x16
        self.pal_blk = ResBlock2d(64, cond_dim)
        self.pal_out = nn.Conv2d(64, 4, 3, padding=1)
        nn.init.zeros_(self.pal_out.weight)
        nn.init.zeros_(self.pal_out.bias)

        # Texture 分支: z_b(4) + c_64(4) + d16_up(48) -> 64x64 多尺度笔触残差
        self.dino_reduce = nn.Conv2d(384, 48, 1)
        self.tex_in = nn.Conv2d(4 + 4 + 48, ch, 3, padding=1)
        self.tex_blks = nn.ModuleList([ResBlock2d(ch, cond_dim) for _ in range(4)])
        self.tex_out = nn.Conv2d(ch, 4, 3, padding=1)
        nn.init.zeros_(self.tex_out.weight)
        nn.init.zeros_(self.tex_out.bias)

        # Haar 频带权重 (允许少量低频色彩微调，重点放大中/高频笔触)
        self.alpha_ll = nn.Parameter(torch.tensor(0.25))
        self.alpha_m = nn.Parameter(torch.tensor(1.0))
        self.alpha_h = nn.Parameter(torch.tensor(1.0))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, z_b: torch.Tensor, c_64: torch.Tensor, d_16: torch.Tensor,
                u_tgt: torch.Tensor, same_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cond = self.cond_mlp(u_tgt)

        # 1. Palette 分支
        hp = F.silu(self.pal_in(z_b))
        hp = F.silu(self.pal_down(hp))
        hp = self.pal_blk(hp, cond)
        dp_16 = self.pal_out(hp)
        d_palette = F.interpolate(dp_16, size=z_b.shape[-2:], mode="bilinear", align_corners=False)

        # 2. Texture 分支
        d_up = F.interpolate(self.dino_reduce(d_16), size=z_b.shape[-2:], mode="bilinear", align_corners=False)
        ht = self.tex_in(torch.cat([z_b, c_64, d_up], dim=1))
        for blk in self.tex_blks:
            ht = blk(ht, cond)
        t_raw = self.tex_out(ht)

        # Haar 分带加权 (Pm: LH+HL 中频方向笔触, Ph: HH 高频细节)
        ll, lh, hl, hh = dwt2_haar(t_raw)
        d_texture = idwt2_haar(self.alpha_ll * ll, self.alpha_m * lh, self.alpha_m * hl, self.alpha_h * hh)

        delta = d_palette + d_texture
        if same_mask is not None:
            m = (~same_mask).float().view(-1, 1, 1, 1)
            delta = delta * m
            d_palette = d_palette * m
            d_texture = d_texture * m
        return delta, d_palette, d_texture


@torch.no_grad()
def _prism1_base_batch(stu, z_src, z0_b, c64, d16, u_src, u_tgt, tid_tgt, priors, P_Y, same_mask=None):
    alpha_eff = 0.20
    z_anc = apply_latent_chroma_shift(z0_b, tid_tgt, priors, alpha=alpha_eff)
    chord = u_tgt - u_src
    z_luma = torch.einsum("ij,bjhw->bihw", P_Y, z_src)
    delta = stu(z_anc, z_luma, c64, d16, chord, same_mask=same_mask)
    return z_anc + delta


def train_m3_m4(bundle: dict, dec, swd_ext: MultiScalePatchSWDExtractor, mode: str,
                updates: int = 1200, micro: int = 4, accum: int = 8, device: str = "cuda") -> Path:
    """mode: 'M3' (固定 lambda_c=1.0) | 'M4' (动态拉格朗日内容预算 lambda_c)。"""
    out_p = OUT / f"{mode.lower()}_residual" / f"res_{mode}.pt"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    if out_p.exists():
        print(f"[V24 {mode}] Reusing {out_p}", flush=True)
        return out_p

    seed_all(20262600)
    tr = bundle["v18_cache"]["train"]
    fnd = bundle["fnd"]
    z_all = tr["z"]
    sid_all = tr["sid"].long()
    d16_all = tr["dino_16"]
    c64_all = tr["c_64"]

    # 载入冻结 PRISM-1 基座与 FiberEncoder
    stu = SpatiallyRoutedChordNet(width=96, chord_dim=128, use_spatial_router=True).to(device).eval()
    stu.load_state_dict(torch.load(P1_CKPT, map_location="cpu", weights_only=False)["model"])
    for p in stu.parameters():
        p.requires_grad_(False)

    fiber_enc = StyleFiberEncoder(in_dim=fnd["tr_fib"].shape[1], dim=128).to(device).eval()
    fiber_enc.load_state_dict(fnd["fiber_state"])
    for p in fiber_enc.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        u_all = fiber_enc.encode(fnd["tr_fib"].to(device).float())

    priors = {k: v.to(device) if torch.is_tensor(v) else v for k, v in fnd["priors"].items()}
    P_Y = priors["P_Y"]

    # 用 train 内部均值平滑近似 z0_tr (避免跑 4400 次 B4, 或直接以温和高斯平滑去纹理构图作为 z0_tr)
    # 先校准 PRISM-1 在训练集上的 ConvNeXt 中尺度内容偏差 epsilon (无任何 test750 触碰!)
    res_net = PaletteTextureResidualNet(cond_dim=128, ch=96).to(device)
    print(f"[V24 {mode}] PaletteTextureResidualNet params={res_net.n_params()/1e6:.2f}M (PRISM-1 frozen)", flush=True)
    ema = copy.deepcopy(res_net).eval()
    for p in ema.parameters():
        p.requires_grad_(False)

    # 训练集校准 PRISM-1 的 ConvNeXt 内容损失基准 epsilon
    with torch.no_grad():
        cal_idx = torch.randperm(len(z_all))[:64]
        cal_r = torch.randint(0, len(z_all), (64,))
        z_c = z_all[cal_idx].to(device).float()
        z0_c = F.avg_pool2d(F.pad(z_c, (1, 1, 1, 1), mode="replicate"), 3, 1)
        z_b_cal = _prism1_base_batch(stu, z_c, z0_c, c64_all[cal_idx].to(device).float(),
                                     d16_all[cal_idx].to(device).float(), u_all[cal_idx], u_all[cal_r],
                                     sid_all[cal_r].to(device), priors, P_Y)
        rgb_src_c = decode_s10(dec, z_c, batch=16, output_cpu=False)
        rgb_b_c = decode_s10(dec, z_b_cal, batch=16, output_cpu=False)
        ts = swd_ext.extract_tokens(rgb_src_c)
        tb = swd_ext.extract_tokens(rgb_b_c)
        lc_cal = 0.5 * ((1.0 - (F.normalize(tb["s2"], dim=-1) * F.normalize(ts["s2"], dim=-1)).sum(-1)).mean()
                        + (1.0 - (F.normalize(tb["s3"], dim=-1) * F.normalize(ts["s3"], dim=-1)).sum(-1)).mean())
        eps_budget = float(lc_cal) * 1.08  # 允许在 PRISM-1 基础内容偏差上最多扩展 8%
        print(f"[V24 {mode}] Calibrated PRISM-1 train L_c={float(lc_cal):.4f} -> budget epsilon={eps_budget:.4f}", flush=True)

    opt = torch.optim.AdamW(res_net.parameters(), lr=1.5e-4, betas=(0.9, 0.99), weight_decay=1e-4)
    lam_c = 1.2 if mode == "M3" else 1.0
    eta_dual = 0.45
    t0 = time.time()

    for step in range(1, updates + 1):
        prog = step / max(1, updates)
        lr = 2e-5 + 0.5 * (1.5e-4 - 2e-5) * (1.0 + math.cos(math.pi * prog))
        for pg in opt.param_groups:
            pg["lr"] = lr
        opt.zero_grad(set_to_none=True)

        ls_v, lc_v, lr_v = 0.0, 0.0, 0.0
        for _ in range(accum):
            ic = torch.randint(0, len(z_all), (micro,))
            ir = torch.randint(0, len(z_all), (micro,))
            z_src = z_all[ic].to(device).float()
            z_ref = z_all[ir].to(device).float()
            c64 = c64_all[ic].to(device).float()
            d16 = d16_all[ic].to(device).float()
            tid_r = sid_all[ir].to(device)
            z0_c = F.avg_pool2d(F.pad(z_src, (1, 1, 1, 1), mode="replicate"), 3, 1)

            with torch.no_grad():
                z_b = _prism1_base_batch(stu, z_src, z0_c, c64, d16, u_all[ic], u_all[ir], tid_r, priors, P_Y)
                rgb_src = decode_s10(dec, z_src, batch=micro, output_cpu=False)
                rgb_ref = decode_s10(dec, z_ref, batch=micro, output_cpu=False)
                toks_src = swd_ext.extract_tokens(rgb_src)
                toks_ref = swd_ext.extract_tokens(rgb_ref)
                pal_ref = extract_color_fiber_27d(rgb_ref)

            with autocast(device):
                delta, d_pal, d_tex = res_net(z_b, c64, d16, u_all[ir])
                z_out = (z_b + delta).clamp(-5.0, 5.0)
                rgb_out = decode_s10(dec, z_out, batch=micro, enable_grad=True, output_cpu=False)

                # 1. 风格损失 (独立于裁判): 局部纹理 SWD + Oklab 27D 色彩统计
                toks_out = swd_ext.extract_tokens(rgb_out)
                l_swd = (
                    1.0 * sliced_wasserstein_1d(toks_out["rgb"], toks_ref["rgb"], swd_ext.p_rgb)
                    + 0.35 * sliced_wasserstein_1d(toks_out["s1"], toks_ref["s1"], swd_ext.p_s1)
                    + 0.45 * sliced_wasserstein_1d(toks_out["s2"], toks_ref["s2"], swd_ext.p_s2)
                    + 0.55 * sliced_wasserstein_1d(toks_out["s3"], toks_ref["s3"], swd_ext.p_s3)
                ).mean()
                l_pal = F.smooth_l1_loss(extract_color_fiber_27d(rgb_out), pal_ref)
                l_style = l_swd + 0.6 * l_pal

                # 2. 内容约束 (独立于 DINO-C 裁判): ConvNeXt 32x32 & 16x16 空间对齐
                l_c = 0.5 * (
                    (1.0 - (F.normalize(toks_out["s2"], dim=-1) * F.normalize(toks_src["s2"], dim=-1)).sum(-1)).mean()
                    + (1.0 - (F.normalize(toks_out["s3"], dim=-1) * F.normalize(toks_src["s3"], dim=-1)).sum(-1)).mean()
                )

                # 3. 残差幅度正则
                l_res = d_pal.square().mean() + 0.5 * d_tex.square().mean()

                loss = (l_style + lam_c * l_c + 0.15 * l_res) / accum

            loss.backward()
            ls_v += float(l_style.item()) / accum
            lc_v += float(l_c.item()) / accum
            lr_v += float(l_res.item()) / accum

        nn.utils.clip_grad_norm_(res_net.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for pe, pw in zip(ema.parameters(), res_net.parameters()):
                pe.mul_(0.996).add_(pw, alpha=0.004)

        # M4 动态预算拉格朗日乘子更新
        if mode == "M4" and step % 10 == 0:
            lam_c = float(min(8.0, max(0.1, lam_c + eta_dual * (lc_v - eps_budget))))

        if step == 1 or step % 200 == 0 or step == updates:
            print(f"[V24 {mode}] {step:4d}/{updates} style={ls_v:.4f} lc={lc_v:.4f}(eps={eps_budget:.4f},lam_c={lam_c:.2f}) res={lr_v:.4f} ({time.time()-t0:.0f}s)", flush=True)

    torch.save({"ema": ema.state_dict(), "mode": mode, "lam_c": lam_c, "eps_budget": eps_budget}, out_p)
    print(f"[V24 {mode}] Saved {out_p} in {(time.time()-t0)/60:.1f} min", flush=True)
    return out_p


@torch.no_grad()
def eval_m3_m4_arm(bundle: dict, ckpt_p: Path, arm_name: str, scale: float, tasks: list[dict],
                   z_src_750: torch.Tensor, z_p1_100: torch.Tensor, d16_750: torch.Tensor,
                   dec, device: str = "cuda") -> dict:
    cfg_hash = config_hash({"arm": arm_name, "ckpt": file_sha(ckpt_p)[:12], "scale": scale})
    if (OUT / "raw" / arm_name / "meta.json").exists() and validate_raw_meta(read_json(OUT / "raw" / arm_name / "meta.json")):
        s = read_json(OUT / f"measured/{arm_name}/summary_v24.json")
    else:
        fnd = bundle["fnd"]
        fiber_enc = StyleFiberEncoder(in_dim=fnd["tr_fib"].shape[1], dim=128).to(device).eval()
        fiber_enc.load_state_dict(fnd["fiber_state"])
        bary = fiber_enc.class_barycenters()
        del fiber_enc

        res_net = PaletteTextureResidualNet(cond_dim=128, ch=96).to(device).eval()
        res_net.load_state_dict(torch.load(ckpt_p, map_location="cpu", weights_only=False)["ema"], strict=True)
        c64_750 = bundle["test_struct"]["c_64"].float()
        sid_all = torch.tensor([t["source_style_id"] for t in tasks], device=device)
        tid_all = torch.tensor([t["target_style_id"] for t in tasks], device=device)
        same_t = sid_all == tid_all

        wr = RawWriterV24(arm_name, dec=dec, device=device)
        t0 = time.time()
        for st in range(0, 750, 25):
            sl = slice(st, min(st + 25, 750))
            zb = z_p1_100[sl].to(device).float()
            delta, _, _ = res_net(zb, c64_750[sl].to(device).float(), d16_750[sl].to(device).float(),
                                  bary[tid_all[sl]], same_mask=same_t[sl])
            z_out = zb + scale * delta
            z_out[same_t[sl]] = z_src_750[sl].to(device).float()[same_t[sl]]
            wr.write_batch(z_out, tasks[sl])
        meta = wr.finalize(cfg_hash=cfg_hash, scale=scale, wall_sec=round(time.time() - t0, 1))
        assert validate_raw_meta(meta)
        s = run_measure_v24(arm_name, device=device)
        export_raw_pngs(OUT / f"raw/{arm_name}", OUT / f"png_750/{arm_name}", max_workers=8)
        del res_net
        torch.cuda.empty_cache()

    c = s["cross600"]
    return {"arm": arm_name, "dino_c": round(c["dino_c"], 5), "dino_s": round(c["dino_s"], 5),
            "lpips": round(c["lpips"], 5), "style_affinity": round(c["style_affinity"], 5),
            "csd_top1": round(c["csd_top1"], 5), "same_top1": round(s["same150"]["csd_top1"], 5)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--updates", type=int, default=1200)
    args = ap.parse_args()

    sanitize_cuda_env()
    bundle = build_or_load_v19_foundation(args.pool, device=args.device)
    dec = load_s10_decoder(args.device)
    swd_ext = MultiScalePatchSWDExtractor(n_slices=96).to(args.device).eval()
    for p in swd_ext.parameters():
        p.requires_grad_(False)

    ck_m3 = train_m3_m4(bundle, dec, swd_ext, "M3", updates=args.updates, device=args.device)
    ck_m4 = train_m3_m4(bundle, dec, swd_ext, "M4", updates=args.updates, device=args.device)
    del swd_ext
    torch.cuda.empty_cache()

    tasks, z_src_750, _ = tasks_and_sources()
    d16_750 = bundle["test_struct"]["dino_16"].float()
    cache_p = OUT / "cache/p0_base_latents_750.pt"
    if cache_p.exists():
        z_p1_100 = torch.load(cache_p, map_location="cpu", weights_only=False)["z_p1_100"]
    else:
        z_p1_100 = compute_prism1_latents(bundle, tasks, z_src_750, d16_750, 1.00, device=args.device)

    results = []
    for mode, ck in (("M3", ck_m3), ("M4", ck_m4)):
        for sc, tag in ((0.75, "s075"), (1.0, "s100"), (1.35, "s135")):
            results.append(eval_m3_m4_arm(bundle, ck, f"{mode}_res_{tag}", sc, tasks, z_src_750, z_p1_100, d16_750, dec, device=args.device))

    save_json(OUT / "p1_m3_m4.json", {"updated_utc": utc(), "results": results})
    print("\n=== V24 M3/M4 双路径残差汇总 (验收红线: DINO-C>=0.73 下 DINO-S 相对 PRISM-1 0.443 +0.03 -> >=0.473) ===", flush=True)
    for r in results:
        print(f"{r['arm']:16s}: DINO-C={r['dino_c']:.4f} DINO-S={r['dino_s']:.4f} LPIPS={r['lpips']:.4f} Top1={r['csd_top1']*100:5.2f}%", flush=True)


if __name__ == "__main__":
    main()
