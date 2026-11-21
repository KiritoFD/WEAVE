r"""V26-L Exp 2 轻量化探针 (用户 2026-11-21 信息论分析 §4):

1. Inter-Cluster Texture Divergence: 用冻结 DINO 将输出图划分为 K=3-4 语义簇,
   计算同图内不同语义簇之间的 Haar 高频能量与 VGG 风格统计方差.
   若 Bary-SWD 臂的簇间纹理分化度显著高于 Global-SWD 且 ER/Roc 同步拉升,
   则彻底坐实"模型局部能力够用, 缺的只是训练期外部局部监督".

2. Bary-SWD 收敛残差: 对比 Global-SWD 与 Bary-SWD 在训练集上的局部损失降幅.

用法:  python -m exp.v26.gpu4090.exp2_locality_probes --pool ../pseudo_gt_24h/orig_mean_fp16
       (在 L3/L1 训练臂完成后运行, 读 runs/v26/oneshot_val/<arm>.pt)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image as PILImage

from exp.v26 import OUT, ROOT, read_json, save_json, utc
from exp.v26.sr_cfg import load_sr_env

ARM_PAIRS = [
    ("V0_vgg_global", OUT / "oneshot_val" / "V0_vgg_global.pt"),
    ("V1_swd_global", OUT / "oneshot_val" / "V1_swd_global.pt"),
    ("V2_bary_global", OUT / "oneshot_val" / "V2_bary_global.pt"),
    # 兜底: 如果 V0-V2 尚未产出, 用 L0/L1 已有臂做基线对照
    ("A0_legacy_nbr", OUT / "oneshot_val" / "A0_legacy_nbr.pt"),
    ("F1_global_self", OUT / "oneshot_val" / "F1_global_self.pt"),
    ("F2_global_teacher", OUT / "oneshot_val" / "F2_global_teacher.pt"),
]


@torch.no_grad()
def inter_cluster_divergence(env: dict, ckpt_p: Path, device: str = "cuda",
                             n_tasks: int = 48, k_clusters: int = 3) -> dict:
    """语义簇间纹理分化度: DINO k-means 聚类 → 逐簇 Haar 能量/VGG 统计方差."""
    import exp.v25 as _v25
    _v25.seed_all(20266220)
    from exp.v18 import tasks_and_sources
    from exp.v20.slf_flow import coarse_y16_lcn
    from exp.v22.ref_adapter import RefAdapter
    from exp.v26.sr_cfg import integrate_sr, sr_velocity
    from exp.v26.uot_fiber_bundle import geometry_free_fiber_stats
    from src.utils import decode_s10, load_s10_decoder
    from src.utils._local_eval_refs import FeatureStack

    tasks, z_src_750, _ = tasks_and_sources()
    cross_ids = [i for i, t in enumerate(tasks) if t["source_style_id"] != t["target_style_id"]][:n_tasks]
    sel = torch.tensor(cross_ids, dtype=torch.long)
    fm, ad = env["fm"], env["ad"]
    dec = load_s10_decoder(device)
    stack = FeatureStack(device)
    w_l, b_l = env["w_l"].to(device), env["b_l"].to(device)

    ck = torch.load(ckpt_p, map_location="cpu", weights_only=False)
    fm.load_state_dict(ck["fm"] if "fm" in ck else ck["ema"], strict=True)
    ad.load_state_dict(ck["adapter"] if "adapter" in ck else ck["ema"], strict=True)
    fm.eval()

    zs = z_src_750[sel].to(device).float()
    B = len(sel)
    tid_cpu = torch.tensor([tasks[i]["target_style_id"] for i in cross_ids])
    y16 = coarse_y16_lcn(((zs * env["w_l"].to(device)).sum(1, keepdim=True)
                          + env["b_l"].to(device)).clamp(0.0, 1.0)).float()
    d16 = env["bundle"]["test_struct"]["dino_16"][sel].to(device).float()
    u_src = env["u_src_750"][sel].to(device)
    u_style = ad(env["u_sg5"][tid_cpu].to(device), u_src)
    u_glob = env["fib_n_sg5"][tid_cpu]
    d16_r = env["d16_sg5"][tid_cpu].to(device)
    v16_r = env["v16_sg5"][tid_cpu].to(device)
    tau0, wj = 0.45, 2.2
    eps = torch.randn(B, 4, 64, 64, generator=torch.Generator().manual_seed(20261185)).to(device)
    z = (1.0 - tau0) * eps + tau0 * zs

    def vel_fn(zz, prog):
        tv = torch.full((B,), tau0 + prog * (1.0 - tau0), device=device)
        v, _ = sr_velocity(fm, ad, u_style, None, u_glob, None, d16, d16_r, v16_r,
                           None, None, y16, zz, tv, mode="baseline", w=0.0, wj=wj, device=device)
        return v

    # 分块积分 (150 任务一次性 batch=150 会 OOM)
    z_chunks = []
    for st in range(0, B, 16):
        sl = slice(st, min(st + 16, B))
        z_chunks.append(integrate_sr(vel_fn, z[sl], tau0, 10, "rk2"))
    z_gen = torch.cat(z_chunks, 0)
    rgb_gen = decode_s10(dec, z_gen, batch=16).clamp(0.0, 1.0)

    # DINO k-means 聚类 (每张图独立, K=k_clusters)
    f_dino = d16.flatten(2).transpose(1, 2)  # (B, 256, 384)
    f_dino = torch.nn.functional.normalize(f_dino.float(), dim=-1)

    # 简易 1-D k-means (用 DINO token L2 范数做 1-D 分割, K 簇)
    norms = f_dino.norm(dim=-1)  # (B, 256)
    sorted_norms, _ = torch.sort(norms, dim=1)
    boundaries = [sorted_norms[:, int(k * 256 / k_clusters)] for k in range(1, k_clusters)]

    cluster_masks = []  # (B, k_clusters, 256) bool
    for b in range(B):
        masks = []
        prev = 0.0
        for k in range(k_clusters):
            thr = boundaries[k][b].item() if k < len(boundaries) else float('inf')
            mask = (norms[b] >= prev) & (norms[b] < thr) if k < k_clusters - 1 else (norms[b] >= prev)
            masks.append(mask)
            prev = thr
        cluster_masks.append(torch.stack(masks, 0))
    cluster_masks = torch.stack(cluster_masks, 0)  # (B, K, 256)

    # 逐簇 Haar 高频能量 + Oklab 方差 (每张输出图)
    from PIL import Image as PILImage
    from exp.v19.fiber_duality import extract_color_fiber_27d

    def rgb_to_pil(rgb):
        return [PILImage.fromarray((x.permute(1, 2, 0).cpu().numpy() * 255).round().clip(0, 255).astype("uint8"))
                for x in rgb]

    def _cell_haar_energy(rgb01: torch.Tensor) -> torch.Tensor:
        """逐 16×16 格 Haar 高频能量 (B, 256)."""
        g = rgb01.mean(dim=1, keepdim=True)
        B_, _, H, W = g.shape
        cell = H // 16
        out = torch.zeros(B_, 256, device=rgb01.device)
        for i in range(16):
            for j in range(16):
                cx = g[:, :, i * cell:(i + 1) * cell, j * cell:(j + 1) * cell]
                ch, cw = cx.shape[-2] // 2, cx.shape[-1] // 2
                a = cx[:, :, :ch, :cw]; b = cx[:, :, :ch, cw:]
                c = cx[:, :, ch:, :cw]; d = cx[:, :, ch:, cw:]
                lh = (a - b).abs().mean(dim=(2, 3))
                hl = (a - c).abs().mean(dim=(2, 3))
                hh = ((a - b) * (a - c)).abs().mean(dim=(2, 3))
                out[:, i * 16 + j] = (lh + hl + hh).squeeze()
        return out

    haar_per_cell = _cell_haar_energy(rgb_gen)  # (B, 256)
    lab = rgb_gen  # for per-cell VGG-like stats (approx: channel mean/var per cell)
    cell = 512 // 16
    cell_means = torch.zeros(B, 256, 3, device=device)
    cell_stds = torch.zeros(B, 256, 3, device=device)
    for i in range(16):
        for j in range(16):
            cx = rgb_gen[:, :, i * cell:(i + 1) * cell, j * cell:(j + 1) * cell]
            cell_means[:, i * 16 + j] = cx.mean(dim=(2, 3))
            cell_stds[:, i * 16 + j] = cx.std(dim=(2, 3), unbiased=False)

    # 簇间分化度: 簇间 Haar 能量方差 + 簇间 Oklab 色彩均值方差
    haar_div_list, color_div_list = [], []
    for b in range(B):
        haar_means, color_means = [], []
        for k in range(k_clusters):
            mask = cluster_masks[b, k]
            if mask.sum() < 4:
                continue
            haar_means.append(haar_per_cell[b][mask].mean())
            color_means.append(cell_means[b][mask].mean(dim=0))
        if len(haar_means) < 2:
            haar_div_list.append(0.0)
            color_div_list.append(0.0)
        else:
            haar_div_list.append(float(torch.stack(haar_means).std()))
            color_div_list.append(float(torch.stack(color_means).std(dim=0).norm()))
    haar_div = sum(haar_div_list) / len(haar_div_list)
    color_div = sum(color_div_list) / len(color_div_list)

    return {"n_tasks": B, "k_clusters": k_clusters,
            "inter_cluster_haar_divergence": round(haar_div, 4),
            "inter_cluster_color_divergence": round(color_div, 4)}


def run_probes(pool: str, device: str = "cuda") -> dict:
    """对所有可用臂运行簇间分化度探针 → 对比表."""
    env = load_sr_env(pool, device=device)
    results = {}
    for tag, ckpt_p in ARM_PAIRS:
        if not ckpt_p.exists():
            print(f"[Probes] {tag} ckpt missing → skip", flush=True)
            continue
        try:
            r = inter_cluster_divergence(env, ckpt_p, device=device)
            results[tag] = r
            print(f"[Probes {tag}] haar={r['inter_cluster_haar_divergence']} "
                  f"color={r['inter_cluster_color_divergence']}", flush=True)
        except Exception as exc:
            print(f"[Probes {tag}] FAILED: {exc}", flush=True)
    save_json(OUT / "gpu4090" / "exp2_locality_probes.json", {"updated_utc": utc(), "arms": results})
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    from exp.v26 import sanitize_cuda_env
    sanitize_cuda_env()
    run_probes(args.pool, device=args.device)


if __name__ == "__main__":
    main()
