"""V24 §O2 (4090 极速生产): 真参考条件多尺度外观迁移教师 T_ref(x_c, x_r) -> 2000 对无类别跨风格伪配对。

彻底修正 V23 伪配对依赖目标画派标签与类重心的缺陷:
- 全程绝对不读取 sid_r / tid_r / bary[tid_r] 等任何画派标签或类重心!
- 仅对任意抽样的 (x_c, x_r) 对源 latent z_c 经可微 S10 执行 24 步受内容约束的梯度优化:
    L_teacher(z) = L_style(S10(z), x_r)  [参考图 MultiScalePatchSWD + Oklab 27D 色彩统计 + 局部 DINO-match 外观一致性]
                 + w_c * L_content(S10(z), x_c) [源图 ConvNeXt 32x32 & 16x16 空间结构保持]
                 + w_r * ||z - z_c||^2
- 优化完成后按 ConvNeXt 内容保持余弦得分筛选高内容保真的前 2000 对 (从 2400 候选中筛选)，
  用于 V24 One-Shot O2/O3/O4-D 彻底脱离五分类先验的真参考条件训练！

用法 (4090):
  python -m exp.v24.o2_ref_teacher_pairs --pool ../pseudo_gt_24h/orig_mean_fp16
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

from . import OUT, ROOT, sanitize_cuda_env, seed_all
from exp.v19.fiber_duality import build_or_load_v19_foundation, extract_color_fiber_27d
from exp.v20.submode_patchswd import MultiScalePatchSWDExtractor, sliced_wasserstein_1d
from src.utils import decode_s10, load_s10_decoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-cand", type=int, default=2400)
    ap.add_argument("--n-keep", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=22)
    args = ap.parse_args()

    sanitize_cuda_env()
    out_p = OUT / "cache/ref_teacher_pairs_2000.pt"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    if out_p.exists():
        print(f"[V24 O2-Teacher] Reusing {out_p}", flush=True)
        return

    seed_all(20262800)
    bundle = build_or_load_v19_foundation(args.pool, device=args.device)
    tr = bundle["v18_cache"]["train"]
    z_all = tr["z"].float()
    d16_all = tr["dino_16"].float()
    N = len(z_all)

    dec = load_s10_decoder(args.device)
    swd_ext = MultiScalePatchSWDExtractor(n_slices=96).to(args.device).eval()
    for p in swd_ext.parameters():
        p.requires_grad_(False)

    # 随机抽取 n_cand 对互异训练集图像 (完全不读取任何类别标签)
    idx_c = torch.randint(0, N, (args.n_cand,))
    idx_r = (idx_c + torch.randint(1, N, (args.n_cand,))) % N

    z_cr = torch.zeros(args.n_cand, 4, 64, 64, dtype=torch.float16)
    mid_scores = torch.zeros(args.n_cand, dtype=torch.float32)

    bs = 12
    t0 = time.time()
    for st in range(0, args.n_cand, bs):
        sl = slice(st, min(st + bs, args.n_cand))
        ic, ir = idx_c[sl], idx_r[sl]
        z_src = z_all[ic].to(args.device)
        z_ref = z_all[ir].to(args.device)

        with torch.no_grad():
            rgb_src = decode_s10(dec, z_src, batch=bs, output_cpu=False)
            rgb_ref = decode_s10(dec, z_ref, batch=bs, output_cpu=False)
            toks_src = swd_ext.extract_tokens(rgb_src)
            toks_ref = swd_ext.extract_tokens(rgb_ref)
            pal_ref = extract_color_fiber_27d(rgb_ref)

            # Q/K 灰度 DINO 语义匹配矩阵用于局部外观一致性辅助
            dc = F.normalize(d16_all[ic].to(args.device).flatten(2).transpose(1, 2), dim=-1)  # (B, 256, 384)
            dr = F.normalize(d16_all[ir].to(args.device).flatten(2).transpose(1, 2), dim=-1)
            sim = torch.bmm(dc, dr.transpose(1, 2))
            best_sim, best_j = sim.max(dim=-1)  # (B, 256)
            gate_loc = (best_sim > 0.25).float().unsqueeze(-1)  # 低置信不强迫贴局部
            ref_s3 = toks_ref["s3"]  # (B, 256, 384)
            matched_s3 = torch.gather(ref_s3, 1, best_j.unsqueeze(-1).expand_as(ref_s3))
            matched_s3_n = F.normalize(matched_s3, dim=-1)

        z_opt = z_src.clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([z_opt], lr=0.055)
        for _ in range(args.steps):
            opt.zero_grad()
            rgb_hat = decode_s10(dec, z_opt.clamp(-5.0, 5.0), batch=bs, enable_grad=True, output_cpu=False)
            toks_hat = swd_ext.extract_tokens(rgb_hat)
            l_swd = (
                1.0 * sliced_wasserstein_1d(toks_hat["rgb"], toks_ref["rgb"], swd_ext.p_rgb)
                + 0.35 * sliced_wasserstein_1d(toks_hat["s1"], toks_ref["s1"], swd_ext.p_s1)
                + 0.45 * sliced_wasserstein_1d(toks_hat["s2"], toks_ref["s2"], swd_ext.p_s2)
                + 0.55 * sliced_wasserstein_1d(toks_hat["s3"], toks_ref["s3"], swd_ext.p_s3)
            ).mean()
            l_pal = F.smooth_l1_loss(extract_color_fiber_27d(rgb_hat), pal_ref)
            l_loc = ((1.0 - (F.normalize(toks_hat["s3"], dim=-1) * matched_s3_n).sum(-1, keepdim=True)) * gate_loc).mean()
            l_content = 0.5 * (
                (1.0 - (F.normalize(toks_hat["s2"], dim=-1) * F.normalize(toks_src["s2"], dim=-1)).sum(-1)).mean()
                + (1.0 - (F.normalize(toks_hat["s3"], dim=-1) * F.normalize(toks_src["s3"], dim=-1)).sum(-1)).mean()
            )
            l_reg = (z_opt - z_src).square().mean()
            loss = (l_swd + 0.7 * l_pal + 0.25 * l_loc) + 2.2 * l_content + 0.15 * l_reg
            loss.backward()
            opt.step()

        with torch.no_grad():
            z_fin = z_opt.detach().clamp(-5.0, 5.0)
            rgb_fin = decode_s10(dec, z_fin, batch=bs, output_cpu=False)
            tf = swd_ext.extract_tokens(rgb_fin)
            cos_mid = 0.5 * (
                (F.normalize(tf["s2"], dim=-1) * F.normalize(toks_src["s2"], dim=-1)).sum(-1).mean(dim=1)
                + (F.normalize(tf["s3"], dim=-1) * F.normalize(toks_src["s3"], dim=-1)).sum(-1).mean(dim=1)
            )
            z_cr[sl] = z_fin.cpu().half()
            mid_scores[sl] = cos_mid.cpu()

        if (st // bs) % 20 == 0 or st + bs >= args.n_cand:
            print(f"[V24 O2-Teacher] {min(st+bs, args.n_cand)}/{args.n_cand} mean_mid_cos={float(mid_scores[:min(st+bs, args.n_cand)].mean()):.4f} ({time.time()-t0:.0f}s)", flush=True)

    keep = mid_scores.topk(min(args.n_keep, args.n_cand)).indices.sort().values
    torch.save({
        "idx_c": idx_c[keep].cpu(), "idx_r": idx_r[keep].cpu(),
        "z_cr": z_cr[keep].cpu(), "mid_cos": mid_scores[keep].cpu(),
        "mean_all_cos": float(mid_scores.mean()), "mean_kept_cos": float(mid_scores[keep].mean()),
    }, out_p)
    print(f"[V24 O2-Teacher] Saved {len(keep)} label-free cross-pairs -> {out_p} | "
          f"all_cos={float(mid_scores.mean()):.4f} -> kept_cos={float(mid_scores[keep].mean()):.4f}", flush=True)


if __name__ == "__main__":
    main()
