"""V24 P2-P4 (4090): M5 洁净风格残差蒸馏 + 动态拉格朗日 ConvNeXt 内容预算。

修正 M3/M4 第1版发现的问题:
1. 不再用 1步卷积硬从 0 拟合随机参考 Patch-SWD (避免高频 LPIPS 噪声与丢失画派结构);
2. 预先在训练池(1000张抽样, 绝对零 test750)计算真实 PRISM-1 基座 z_b 与 洁净 V20 D6 风格目标 z_d6;
3. 让 0.99M PaletteTextureResidualNet 学习平滑洁净残差方向 (D6 - PRISM1 的潜在结构与频带增量) + Oklab 色盘对齐,
   并用严格紧致的动态 ConvNeXt 中尺度预算 epsilon = 0.072 (对应 DINO-C >= 0.73) 约束内容损失!

目标是把 V20 D6 的内容从 0.669 右移至 >= 0.73, 同时保持 DINO-S >= 0.49~0.51, LPIPS <= 0.505, Top-1 >= 53% (仅 34ms/img!)。

用法 (4090):
  python -m exp.v24.p2_m5_clean_residual --pool ../pseudo_gt_24h/orig_mean_fp16
"""
from __future__ import annotations

import argparse
import copy
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import OUT, EVAL_SEED, ROOT, autocast, sanitize_cuda_env, save_json, seed_all, utc
from .p0_m0_m2 import compute_prism1_latents
from .p1_m3_m4_residual import PaletteTextureResidualNet, eval_m3_m4_arm
from exp.v22.core import CFGConfig, D6_SOLVER, GateConfig, PrismCore
from exp.v18 import tasks_and_sources
from exp.v19.fiber_duality import StyleFiberEncoder, apply_latent_chroma_shift, build_or_load_v19_foundation, extract_color_fiber_27d
from exp.v19.routed_student import SpatiallyRoutedChordNet
from exp.v20.run_v20 import get_test750_y16_and_d16
from exp.v20.slf_flow import coarse_y16_lcn
from exp.v20.submode_patchswd import MultiScalePatchSWDExtractor, build_or_load_core_knn_bank
from src.utils import decode_s10, load_s10_decoder
from src.utils.structure import dwt2_haar

P1_CKPT = ROOT / "runs/v19/students/s2_prism1_v2.pt"
V20_CKPT = ROOT / "runs/v20/slf_flow/slf_core_v2.pt"


@torch.no_grad()
def build_clean_train_bridge_pairs(bundle: dict, bank: dict, dec, n_pairs: int = 1200, device: str = "cuda") -> dict:
    """在训练集(4400)内部构造 (z_src, z_p1, z_d6, idx_c, tid_t) 洁净桥接样本 (零 test750 触碰)。"""
    cache_p = OUT / "cache/m5_clean_bridge_1200.pt"
    cache_p.parent.mkdir(parents=True, exist_ok=True)
    if cache_p.exists():
        print(f"[V24 M5] Reusing {cache_p}", flush=True)
        return torch.load(cache_p, map_location="cpu", weights_only=False)

    seed_all(20263000)
    tr = bundle["v18_cache"]["train"]
    fnd = bundle["fnd"]
    z_all = tr["z"].float()
    sid_all = tr["sid"].long()
    d16_all = tr["dino_16"].float()
    c64_all = tr["c_64"].float()
    y16_all = coarse_y16_lcn(tr["y_64"].float())

    stu = SpatiallyRoutedChordNet(width=96, chord_dim=128, use_spatial_router=True).to(device).eval()
    stu.load_state_dict(torch.load(P1_CKPT, map_location="cpu", weights_only=False)["model"])
    fiber_enc = StyleFiberEncoder(in_dim=fnd["tr_fib"].shape[1], dim=128).to(device).eval()
    fiber_enc.load_state_dict(fnd["fiber_state"])
    u_all = fiber_enc.encode(fnd["tr_fib"].to(device).float())
    bary = fiber_enc.class_barycenters()
    priors = {k: v.to(device) if torch.is_tensor(v) else v for k, v in fnd["priors"].items()}
    P_Y = priors["P_Y"]

    core = PrismCore(bundle, bank, V20_CKPT, device=device)

    idx_c = torch.randperm(len(z_all))[:n_pairs]
    tid_t = (sid_all[idx_c] + torch.randint(1, 5, (n_pairs,))) % 5
    z_p1 = torch.zeros(n_pairs, 4, 64, 64, dtype=torch.float16)
    z_d6 = torch.zeros(n_pairs, 4, 64, 64, dtype=torch.float16)

    t0 = time.time()
    for st in range(0, n_pairs, 24):
        sl = slice(st, min(st + 24, n_pairs))
        ic = idx_c[sl]
        tb = tid_t[sl].to(device)
        zs = z_all[ic].to(device)
        z0_b = F.avg_pool2d(F.pad(zs, (1, 1, 1, 1), mode="replicate"), 3, 1)
        z_anc = apply_latent_chroma_shift(z0_b, tb, priors, alpha=0.20)
        chord = bary[tb] - u_all[ic]
        z_luma = torch.einsum("ij,bjhw->bihw", P_Y, zs)
        zp1 = z_anc + stu(z_anc, z_luma, c64_all[ic].to(device), d16_all[ic].to(device), chord)

        # 用 z_p1 作为 D6 锚点继续走 8 步干净 D6_SOLVER 得到洁净跨风格目标 z_d6
        eps = torch.randn_like(zs)
        zd6, _ = core.integrate_flow(
            eps, zp1, y16_all[ic].to(device), d16_all[ic].to(device), tb,
            gate=GateConfig(mode="original"), cfg=CFGConfig(branches="full", rescale_phi=0.48), solver=dict(D6_SOLVER),
        )
        z_p1[sl] = zp1.cpu().half()
        z_d6[sl] = zd6.cpu().half()
        if (st // 24) % 10 == 0:
            print(f"[V24 M5-Data] {st+len(ic)}/{n_pairs} ({time.time()-t0:.0f}s)", flush=True)

    del stu, fiber_enc, core
    torch.cuda.empty_cache()
    pack = {"idx_c": idx_c, "tid_t": tid_t, "z_p1": z_p1, "z_d6": z_d6}
    torch.save(pack, cache_p)
    print(f"[V24 M5] Saved {cache_p}", flush=True)
    return pack


def train_m5(bundle: dict, pack: dict, dec, swd_ext: MultiScalePatchSWDExtractor,
             eps_budget: float = 0.068, updates: int = 1000, micro: int = 8, accum: int = 4,
             device: str = "cuda") -> Path:
    out_p = OUT / "m5_residual/res_M5.pt"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    if out_p.exists():
        print(f"[V24 M5] Reusing {out_p}", flush=True)
        return out_p

    seed_all(20263100)
    tr = bundle["v18_cache"]["train"]
    fnd = bundle["fnd"]
    fiber_enc = StyleFiberEncoder(in_dim=fnd["tr_fib"].shape[1], dim=128).to(device).eval()
    fiber_enc.load_state_dict(fnd["fiber_state"])
    bary = fiber_enc.class_barycenters()
    del fiber_enc

    res_net = PaletteTextureResidualNet(cond_dim=128, ch=96).to(device)
    ema = copy.deepcopy(res_net).eval()
    for p in ema.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(res_net.parameters(), lr=2e-4, betas=(0.9, 0.99), weight_decay=1e-4)
    lam_c = 1.5
    t0 = time.time()
    M = len(pack["idx_c"])

    for step in range(1, updates + 1):
        lr = 2e-5 + 0.5 * (2e-4 - 2e-5) * (1.0 + math.cos(math.pi * step / max(1, updates)))
        for pg in opt.param_groups:
            pg["lr"] = lr
        opt.zero_grad(set_to_none=True)

        ls_v, lc_v = 0.0, 0.0
        for _ in range(accum):
            pick = torch.randint(0, M, (micro,))
            ic = pack["idx_c"][pick]
            tb = pack["tid_t"][pick].to(device)
            z_src = tr["z"][ic].to(device).float()
            z_b = pack["z_p1"][pick].to(device).float()
            z_d6 = pack["z_d6"][pick].to(device).float()
            c64 = tr["c_64"][ic].to(device).float()
            d16 = tr["dino_16"][ic].to(device).float()

            with torch.no_grad():
                rgb_src = decode_s10(dec, z_src, batch=micro, output_cpu=False)
                rgb_d6 = decode_s10(dec, z_d6, batch=micro, output_cpu=False)
                toks_src = swd_ext.extract_tokens(rgb_src)
                pal_d6 = extract_color_fiber_27d(rgb_d6)
                ll_t, lh_t, hl_t, hh_t = dwt2_haar(z_d6 - z_b)

            with autocast(device):
                delta, d_pal, d_tex = res_net(z_b, c64, d16, bary[tb])
                z_out = (z_b + delta).clamp(-5.0, 5.0)
                ll_p, lh_p, hl_p, hh_p = dwt2_haar(delta)

                # 1. 洁净风格残差蒸馏 (聚焦 D6 高中频画风结构 + 低频色彩分布)
                l_lat = (
                    0.5 * F.smooth_l1_loss(ll_p, ll_t)
                    + 1.2 * F.smooth_l1_loss(lh_p, lh_t)
                    + 1.2 * F.smooth_l1_loss(hl_p, hl_t)
                    + 1.0 * F.smooth_l1_loss(hh_p, hh_t)
                )
                rgb_out = decode_s10(dec, z_out, batch=micro, enable_grad=True, output_cpu=False)
                l_pal = F.smooth_l1_loss(extract_color_fiber_27d(rgb_out), pal_d6)
                l_style = l_lat + 0.5 * l_pal

                # 2. 紧致独立 ConvNeXt 内容预算 (比 PRISM-1 自身 0.099 更严的 0.068 以保护 DINO-C >= 0.73)
                toks_out = swd_ext.extract_tokens(rgb_out)
                l_c = 0.5 * (
                    (1.0 - (F.normalize(toks_out["s2"], dim=-1) * F.normalize(toks_src["s2"], dim=-1)).sum(-1)).mean()
                    + (1.0 - (F.normalize(toks_out["s3"], dim=-1) * F.normalize(toks_src["s3"], dim=-1)).sum(-1)).mean()
                )
                loss = (l_style + lam_c * l_c) / accum

            loss.backward()
            ls_v += float(l_style.item()) / accum
            lc_v += float(l_c.item()) / accum

        nn.utils.clip_grad_norm_(res_net.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for pe, pw in zip(ema.parameters(), res_net.parameters()):
                pe.mul_(0.996).add_(pw, alpha=0.004)
        if step % 10 == 0:
            lam_c = float(min(10.0, max(0.2, lam_c + 0.6 * (lc_v - eps_budget))))

        if step == 1 or step % 200 == 0 or step == updates:
            print(f"[V24 M5] {step:4d}/{updates} style={ls_v:.4f} lc={lc_v:.4f}(eps={eps_budget:.3f},lam_c={lam_c:.2f}) ({time.time()-t0:.0f}s)", flush=True)

    torch.save({"ema": ema.state_dict(), "lam_c": lam_c}, out_p)
    print(f"[V24 M5] Saved {out_p} in {(time.time()-t0)/60:.1f} min", flush=True)
    return out_p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--updates", type=int, default=1000)
    args = ap.parse_args()

    sanitize_cuda_env()
    bundle = build_or_load_v19_foundation(args.pool, device=args.device)
    bank = build_or_load_core_knn_bank(bundle, k_sub=8, core_per_style=180, device=args.device)
    dec = load_s10_decoder(args.device)
    pack = build_clean_train_bridge_pairs(bundle, bank, dec, n_pairs=1200, device=args.device)

    swd_ext = MultiScalePatchSWDExtractor(n_slices=64).to(args.device).eval()
    for p in swd_ext.parameters():
        p.requires_grad_(False)
    ckpt_p = train_m5(bundle, pack, dec, swd_ext, updates=args.updates, device=args.device)
    del swd_ext
    torch.cuda.empty_cache()

    tasks, z_src_750, _ = tasks_and_sources()
    d16_750 = bundle["test_struct"]["dino_16"].float()
    cache_p = OUT / "cache/p0_base_latents_750.pt"
    z_p1_100 = torch.load(cache_p, map_location="cpu", weights_only=False)["z_p1_100"] if cache_p.exists() else compute_prism1_latents(bundle, tasks, z_src_750, d16_750, 1.00, device=args.device)

    results = []
    for sc, tag in ((0.65, "s065"), (0.85, "s085"), (1.0, "s100")):
        results.append(eval_m3_m4_arm(bundle, ckpt_p, f"M5_clean_{tag}", sc, tasks, z_src_750, z_p1_100, d16_750, dec, device=args.device))

    save_json(OUT / "p2_m5.json", {"updated_utc": utc(), "results": results})
    print("\n=== V24 M5 洁净残差蒸馏 + 动态内容预算 汇总 ===", flush=True)
    for r in results:
        print(f"{r['arm']:16s}: DINO-C={r['dino_c']:.4f} DINO-S={r['dino_s']:.4f} LPIPS={r['lpips']:.4f} Top1={r['csd_top1']*100:5.2f}%", flush=True)


if __name__ == "__main__":
    main()
