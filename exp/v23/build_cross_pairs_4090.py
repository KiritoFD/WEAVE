"""V23 P2 §2.3 (4090 极速生产): 生成并筛选训练内部跨风格伪配对 (x_c, x_r, z*_{c->r})。

严格零泄露规则:
1. 内容图 idx_c 与跨画派参考图 idx_r (sid[c] != sid[r]) 均严格来自训练集 train (4400张), 绝不读取 test750 或 SG-5;
2. 使用训练内部 PRISM-1 学生锚点 + Peak1500 (g=0.50, wc=1.5) 生成跨风格伪目标 z_cr;
3. 使用独立的 ConvNeXt-Tiny 中尺度空间余弦相似度筛选结构保留最好的前 1000 对 (从 1500 候选中剔除内容损坏样本);
4. 输出轻量 fp16 包 runs/v23/cache/train_cross_pairs_1000.pt (~32MB)，便于直接传给 3060 跑 M0/M1/M2。

用法 (4090):
  python -m exp.v23.build_cross_pairs_4090 --pool ../pseudo_gt_24h/orig_mean_fp16
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

from . import OUT, ROOT, sanitize_cuda_env, save_json, seed_all, utc
from .p1_midfreq_loss import extract_convnext_mid_spatial
from exp.v22.core import CFGConfig, D5_SOLVER, GateConfig, PrismCore
from exp.v19.fiber_duality import StyleFiberEncoder, apply_latent_chroma_shift, build_or_load_v19_foundation
from exp.v20.slf_flow import coarse_y16_lcn
from exp.v20.submode_patchswd import build_or_load_core_knn_bank, load_convnext_tiny_offline
from src.utils import decode_s10, load_s10_decoder

PEAK_CKPT = ROOT / "runs/v22/tmatrix/slf_T4Ssnap1500.pt"


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-cand", type=int, default=1500)
    ap.add_argument("--n-keep", type=int, default=1000)
    args = ap.parse_args()

    sanitize_cuda_env()
    out_p = OUT / "cache/train_cross_pairs_1000.pt"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    if out_p.exists():
        print(f"[V23 CrossPairs] Reusing {out_p}", flush=True)
        return

    seed_all(20261600)
    bundle = build_or_load_v19_foundation(args.pool, device=args.device)
    bank = build_or_load_core_knn_bank(bundle, k_sub=8, core_per_style=180, device=args.device)
    tr = bundle["v18_cache"]["train"]
    fnd = bundle["fnd"]
    z_all = tr["z"].float()
    sid_all = tr["sid"].long()
    d16_all = tr["dino_16"].float()
    y16_all = coarse_y16_lcn(tr["y_64"].float())
    c64_all = tr["c_64"].float() if "c_64" in tr else None
    N = len(z_all)

    # 抽样 1500 组跨画派 (idx_c, idx_r) 均衡覆盖 20 个有向对
    cls_idx = [(sid_all == s).nonzero(as_tuple=True)[0] for s in range(5)]
    per_pair = max(1, args.n_cand // 20)
    idx_c_list, idx_r_list = [], []
    for sc in range(5):
        for sr in range(5):
            if sc == sr:
                continue
            ic = cls_idx[sc][torch.randint(0, len(cls_idx[sc]), (per_pair,))]
            ir = cls_idx[sr][torch.randint(0, len(cls_idx[sr]), (per_pair,))]
            idx_c_list.append(ic)
            idx_r_list.append(ir)
    idx_c = torch.cat(idx_c_list, dim=0)
    idx_r = torch.cat(idx_r_list, dim=0)
    M = len(idx_c)
    print(f"[V23 CrossPairs] Generating {M} train cross-style pairs...", flush=True)

    # 加载 PRISM-1 & FiberEncoder 构造锚点
    from exp.v19.routed_student import SpatiallyRoutedChordNet
    stu = SpatiallyRoutedChordNet(width=96, chord_dim=128, use_spatial_router=True).to(args.device).eval()
    stu.load_state_dict(torch.load(ROOT / "runs/v19/students/s2_prism1_v2.pt", map_location="cpu", weights_only=False)["model"])
    fiber_enc = StyleFiberEncoder(in_dim=fnd["tr_fib"].shape[1], dim=128).to(args.device).eval()
    fiber_enc.load_state_dict(fnd["fiber_state"])
    u_tr = fiber_enc.encode(fnd["tr_fib"].to(args.device).float())
    bary = fiber_enc.class_barycenters()
    priors = {k: v.to(args.device) if torch.is_tensor(v) else v for k, v in fnd["priors"].items()}
    P_Y = priors["P_Y"]

    core = PrismCore(bundle, bank, PEAK_CKPT, device=args.device)
    dec = load_s10_decoder(args.device)
    cnx = load_convnext_tiny_offline(args.device).eval()

    gate = GateConfig(mode="fixed", fixed_g=0.50)
    cfg = CFGConfig(branches="full", rescale_phi=0.48)
    solver = dict(D5_SOLVER, wc_fig=1.5, wc_abs=1.5)

    z_cr = torch.zeros(M, 4, 64, 64, dtype=torch.float16)
    scores = torch.zeros(M, dtype=torch.float32)
    t0 = time.time()

    for st in range(0, M, 25):
        sl = slice(st, min(st + 25, M))
        ic, ir = idx_c[sl], idx_r[sl]
        B = len(ic)
        zs = z_all[ic].to(args.device)
        tid_r = sid_all[ir].to(args.device)

        z_anc = apply_latent_chroma_shift(zs, tid_r, priors, alpha=0.20)
        chord = bary[tid_r] - u_tr[ic]
        z_luma = torch.einsum("ij,bjhw->bihw", P_Y, zs)
        c64_b = c64_all[ic].to(args.device) if c64_all is not None else torch.cat([tr["y_64"][ic].to(args.device).float(), torch.zeros_like(tr["y_64"][ic].to(args.device).float())], dim=1)
        z_p1 = z_anc + stu(z_anc, z_luma, c64_b, d16_all[ic].to(args.device), chord)

        eps = torch.randn_like(zs)
        u_style = u_tr[ir]  # 直接传入跨画派参考图的真实 fiber u_r
        z_gen, _ = core.integrate_flow(
            eps, z_p1, y16_all[ic].to(args.device), d16_all[ic].to(args.device),
            tid_r, u_style=u_style, dsd_ref=torch.full((B,), float(bank["mid_dsd"]), device=args.device),
            gate=gate, cfg=cfg, solver=solver,
        )
        z_cr[sl] = z_gen.detach().cpu().half()

        # 独立内容质量检查（ConvNeXt 中尺度空间余弦相似度）
        rgb_src = decode_s10(dec, zs, batch=16, output_cpu=False)
        rgb_gen = decode_s10(dec, z_gen, batch=16, output_cpu=False)
        s32, s16 = extract_convnext_mid_spatial(cnx, rgb_src)
        g32, g16 = extract_convnext_mid_spatial(cnx, rgb_gen)
        cos_b = 0.5 * ((s32 * g32).sum(1).mean(dim=(1, 2)) + (s16 * g16).sum(1).mean(dim=(1, 2)))
        scores[sl] = cos_b.cpu()
        if (st // 25) % 10 == 0:
            print(f"[V23 CrossPairs] {st+B}/{M} mean_mid_cos={float(scores[:st+B].mean()):.4f} ({time.time()-t0:.0f}s)", flush=True)

    keep_idx = scores.topk(min(args.n_keep, M)).indices.sort().values
    torch.save({
        "idx_c": idx_c[keep_idx].cpu(),
        "idx_r": idx_r[keep_idx].cpu(),
        "z_cr": z_cr[keep_idx].cpu(),
        "mid_cos": scores[keep_idx].cpu(),
        "mean_all_cos": float(scores.mean()),
        "mean_kept_cos": float(scores[keep_idx].mean()),
    }, out_p)
    print(f"[V23 CrossPairs] Saved {len(keep_idx)} kept pairs -> {out_p} | "
          f"all_cos={float(scores.mean()):.4f} -> kept_cos={float(scores[keep_idx].mean()):.4f}", flush=True)


if __name__ == "__main__":
    main()
