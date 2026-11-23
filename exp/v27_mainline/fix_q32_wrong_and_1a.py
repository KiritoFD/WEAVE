"""前置收尾② + 方向 1 阶段 1A: 修复 Q32 错标签锚点泄漏 & 零训练廉价锚点替换摸底 (4090).

1) 修复 ICME_Q32_s6_csj_wrong / e2full_wrong: 同步替换 z_t4[wrong_idx] 与 tid_wrong.
2) 阶段 1A: 在冻结 6k Q32 旗舰 (s6_e2full_d5 & s6_csj_d5 & s6_csj_d6h) 上替换 z_t4 为三种零泄漏廉价起点:
   - A_p1:     PRISM-1 1-Step 输出 (B4 + s2_prism1_v2, 无 n2c_z 泄漏)
   - A_b4:     B4 轻量锚点 (218K, canonicalizer c1_gentle)
   - A_chroma: 零参数解析色度对齐 z_0^perp (锁死 P_Y z_s, 矩对齐 P_C z_s)

用法 (4090):
  python -m exp.v27_mainline.fix_q32_wrong_and_1a --pool ../pseudo_gt_24h/orig_mean_fp16
"""
from __future__ import annotations

import argparse
import time

import torch

from exp.v18 import tasks_and_sources
from exp.v19.fiber_duality import StyleFiberEncoder, apply_latent_chroma_shift, build_or_load_v19_foundation
from exp.v19.routed_student import SpatiallyRoutedChordNet
from exp.v20.run_v20 import get_test750_y16_and_d16
from exp.v20.submode_patchswd import build_or_load_core_knn_bank
from exp.v21.stiefel_ref_cache import build_or_load_v21_cache
from exp.v22.t4 import build_or_load_t4_latents_750
from exp.v27.icme_step2_closedset_accel import D5_CFG
from exp.v27.icme_step3_q32_vs_dino16 import make_bottleneck_net
from exp.v27.icme_step3b_q32_flagship import _integrate_q32_seg
from exp.v27_mainline import (
    B4_CKPT,
    OUT,
    P1_CKPT,
    Q32_FLAGSHIP_CKPT,
    ROOT,
    RawWriterV27M,
    build_wrong_task_index,
    run_measure_v27m,
)
from src.infer import canonicalizer, load_c3unet
from src.utils import load_s10_decoder
from src.utils.protocol import EVAL_SEED
from src.utils.runtime import sanitize_cuda_env, save_json, utc


@torch.no_grad()
def build_clean_cheap_anchors_750(
    bundle: dict,
    tasks: list[dict],
    z_src_750: torch.Tensor,
    d16_750: torch.Tensor,
    device: str = "cuda",
) -> dict[str, torch.Tensor]:
    """构建无 tid_true 泄漏的三种廉价起点 (750 任务): A_b4, A_p1, A_chroma."""
    cache_p = OUT / "cache/cheap_anchors_750.pt"
    if cache_p.exists():
        print(f"[1A] Reusing {cache_p}", flush=True)
        return torch.load(cache_p, map_location="cpu", weights_only=False)

    cache_p.parent.mkdir(parents=True, exist_ok=True)
    fnd = bundle["fnd"]
    priors = {k: v.to(device) if torch.is_tensor(v) else v for k, v in fnd["priors"].items()}
    P_Y = priors["P_Y"]
    c64_750 = bundle["test_struct"]["c_64"].float()

    sid_all = torch.tensor([t["source_style_id"] for t in tasks], device=device)
    tid_all = torch.tensor([t["target_style_id"] for t in tasks], device=device)
    same_t = sid_all == tid_all

    # 1) A_b4: B4 (218K)
    b4, _ = load_c3unet(B4_CKPT, device)
    canon = canonicalizer(device)
    a_b4 = torch.zeros_like(z_src_750, dtype=torch.float32)
    t0 = time.time()
    for st in range(0, 750, 50):
        sl = slice(st, min(st + 50, 750))
        zb = z_src_750[sl].to(device).float()
        c_can = canon(zb, mode="c1_gentle", augment=False)
        a_b4[sl] = b4(c_can, 0.0, c_can, tid_all[sl]).cpu()
    del b4, canon
    print(f"[1A] Built A_b4 in {time.time()-t0:.1f}s", flush=True)

    # 2) A_chroma: 零参数解析色度对齐 z_0^perp (alpha=1.0, 锁死 P_Y z_s, 仅对齐 P_C 均值方差)
    a_chroma = torch.zeros_like(z_src_750, dtype=torch.float32)
    t0 = time.time()
    for st in range(0, 750, 50):
        sl = slice(st, min(st + 50, 750))
        zs = z_src_750[sl].to(device).float()
        zc = apply_latent_chroma_shift(zs, tid_all[sl], priors, alpha=1.0)
        zc[same_t[sl]] = zs[same_t[sl]]
        a_chroma[sl] = zc.cpu()
    print(f"[1A] Built A_chroma in {time.time()-t0:.2f}s", flush=True)

    # 3) A_p1: PRISM-1 (s2_prism1_v2) 基于 A_b4 (绝不用带泄漏的 n2c_z.pt["z0"]!)
    stu = SpatiallyRoutedChordNet(width=96, chord_dim=128, use_spatial_router=True).to(device).eval()
    stu.load_state_dict(torch.load(P1_CKPT, map_location="cpu", weights_only=False)["model"])
    fiber_enc = StyleFiberEncoder(in_dim=fnd["tr_fib"].shape[1], dim=128).to(device).eval()
    fiber_enc.load_state_dict(fnd["fiber_state"])
    u_src_all = fiber_enc.encode(fnd["test750_fib"].to(device).float())
    bary = fiber_enc.class_barycenters()
    a_p1 = torch.zeros_like(z_src_750, dtype=torch.float32)
    t0 = time.time()
    for st in range(0, 750, 50):
        sl = slice(st, min(st + 50, 750))
        zs = z_src_750[sl].to(device).float()
        z0_b = a_b4[sl].to(device)
        z_anc = apply_latent_chroma_shift(z0_b, tid_all[sl], priors, alpha=0.20)
        chord = bary[tid_all[sl]] - u_src_all[sl]
        z_luma = torch.einsum("ij,bjhw->bihw", P_Y, zs)
        delta = stu(z_anc, z_luma, c64_750[sl].to(device), d16_750[sl].to(device), chord, same_mask=same_t[sl])
        z_out = z_anc + delta
        z_out[same_t[sl]] = zs[same_t[sl]]
        a_p1[sl] = z_out.cpu()
    del stu, fiber_enc
    torch.cuda.empty_cache()
    print(f"[1A] Built A_p1 in {time.time()-t0:.1f}s", flush=True)

    payload = {"A_b4": a_b4.half(), "A_chroma": a_chroma.half(), "A_p1": a_p1.half()}
    torch.save(payload, cache_p)
    return payload


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    sanitize_cuda_env()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = args.device

    pack = build_or_load_v21_cache(args.pool, device=dev)
    bundle, c21 = pack["bundle"], pack["c21"]
    bank = build_or_load_core_knn_bank(bundle, k_sub=8, core_per_style=180, device=dev)

    tasks, z_src_750, _ = tasks_and_sources()
    z_t4, _ = build_or_load_t4_latents_750(bundle, device=dev)
    y16_750, d16_750 = get_test750_y16_and_d16(bundle)
    q32_750 = c21["test750"]["q32"]
    eps_750 = torch.randn(z_src_750.shape, generator=torch.Generator().manual_seed(EVAL_SEED))
    dec = load_s10_decoder(dev)

    sid_all = torch.tensor([t["source_style_id"] for t in tasks])
    tid_all = torch.tensor([t["target_style_id"] for t in tasks])
    same_all = sid_all == tid_all
    tid_wrong_all, wrong_idx_all = build_wrong_task_index(tasks, sid_all, tid_all)

    anchors = build_clean_cheap_anchors_750(bundle, tasks, z_src_750, d16_750, device=dev)

    model = make_bottleneck_net(32).to(dev).eval()
    model.load_state_dict(torch.load(Q32_FLAGSHIP_CKPT, map_location="cpu", weights_only=False)["ema"], strict=True)
    for p in model.parameters():
        p.requires_grad_(False)

    e2full_cfg = dict(D5_CFG, tau_min=0.30, tau_hi=0.68, ws_fig=5.5,
                      wc_fig=1.65, wc_abs=0.70, wsp_fig_late=3.6, wsp_abs=5.2)
    d6h_cfg = dict(D5_CFG, tau_min=0.48, tau_hi=0.78, ws_fig=5.2)

    specs = [
        # 前置收尾②: 彻底替换锚点与 tid 的 6k Q32 错标签对照
        ("Q32_s6_csj_wrong_fixed",   z_t4,                6, set(),   D5_CFG,     True),
        ("Q32_s6_e2full_wrong_fixed",z_t4,                6, {0, 1},  e2full_cfg, True),
        # 阶段 1A: 三种廉价起点替换 (主测 s6_e2full_d5 + s6_csj_d5, 加测 A_p1 d6h)
        ("S1A_Ap1_e2full_d5",        anchors["A_p1"],     6, {0, 1},  e2full_cfg, False),
        ("S1A_Ab4_e2full_d5",        anchors["A_b4"],     6, {0, 1},  e2full_cfg, False),
        ("S1A_Achroma_e2full_d5",    anchors["A_chroma"], 6, {0, 1},  e2full_cfg, False),
        ("S1A_Ap1_csj_d5",           anchors["A_p1"],     6, set(),   D5_CFG,     False),
        ("S1A_Ab4_csj_d5",           anchors["A_b4"],     6, set(),   D5_CFG,     False),
        ("S1A_Achroma_csj_d5",       anchors["A_chroma"], 6, set(),   D5_CFG,     False),
        ("S1A_Ap1_csj_d6h",          anchors["A_p1"],     6, set(),   d6h_cfg,    False),
    ]

    results = {"task": "Fix Q32 wrong-ID + Stage 1A cheap anchor replacement on 6k Q32 flagship",
               "updated_utc": utc(), "arms": {}}
    for arm_name, anc_tensor, steps, fset, kw, is_wrong in specs:
        m_dir = OUT / "measured" / arm_name
        if not (m_dir / "summary_v27m.json").exists():
            wr = RawWriterV27M(arm_name, dec=dec, device=dev)
            for st in range(0, 750, 25):
                sl = slice(st, min(st + 25, 750))
                if is_wrong:
                    tid_u = tid_wrong_all[sl]
                    anc_b = anc_tensor[wrong_idx_all[sl]].to(dev).float()
                else:
                    tid_u = tid_all[sl]
                    anc_b = anc_tensor[sl].to(dev).float()
                zg = _integrate_q32_seg(
                    model, anc_b, eps_750[sl].to(dev).float(),
                    y16_750[sl].to(dev).float(), d16_750[sl].to(dev).float(),
                    q32_750[sl].to(dev).float(), tid_u.to(dev), bank,
                    steps=steps, full_steps=fset, cfg_kw=kw,
                )
                wr.write_batch(
                    torch.where(same_all[sl].view(-1, 1, 1, 1).to(dev), z_src_750[sl].to(dev).float(), zg),
                    tasks[sl],
                )
            wr.finalize(arm=arm_name, is_wrong=is_wrong)
        results["arms"][arm_name] = run_measure_v27m(arm_name, device=dev)
        save_json(OUT / "step1a_anchors.json", results)

    print("[Step1A] ALL DONE →", OUT / "step1a_anchors.json", flush=True)


if __name__ == "__main__":
    main()
