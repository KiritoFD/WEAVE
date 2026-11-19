"""V25 共享前置 (4090, ~10 min): 训练侧 4400 张教师目标缓存 —— 机制 A/B 的唯一教师定义。

产出 runs/v25/cache/train_latents_4400.pt: {z0_tr, z_anc_tr, z_p1_tr, z_d6_tr, sid_tr, tid_tr, teacher_spec}
- z0_tr    = tr["z"] (真实训练集干净 latent, 非 avg_pool 伪造 —— V24 M3/M5 教训)
- z_anc_tr = B4(canonicalizer(z0_tr, c1_gentle)) —— T4 锚的训练侧等价物 (exp/v19/routed_student.py:407 同配方)
- z_p1_tr  = PRISM-1(s2_prism1_v2, λ=1.0) 作用于 z0_tr (chroma-shift 锚, 与测试侧语义一致)
- z_d6_tr  = slf_core_v2 cs_j + D6H_SOLVER + **B4 锚 + original Gate** 的流输出 (教师三元组与评测侧对齐)

教师三元组铁律 (agent2.md §二 / DATA-01): 训练与评测必须同 (ckpt, anchor, solver, branches);
机制 A/B 的所有训练与对照评测统一引用本缓存里的 teacher_spec, 不得混用 T4 锚评测读数归因。

用法 (4090):
  python -m exp.v25.precompute_train_latents --pool ../pseudo_gt_24h/orig_mean_fp16
"""
from __future__ import annotations

import argparse

import torch

from . import OUT, ROOT, sanitize_cuda_env, save_json, seed_all, utc
from exp.v19.fiber_duality import StyleFiberEncoder, apply_latent_chroma_shift, build_or_load_v19_foundation
from exp.v19.routed_student import SpatiallyRoutedChordNet
from exp.v20.slf_flow import coarse_y16_lcn
from exp.v20.submode_patchswd import build_or_load_core_knn_bank
from exp.v22.core import PrismCore
from exp.v25.r1_inference_ablation import D6H_SOLVER, run_core_flow
from src.infer import canonicalizer, load_c3unet
from src.utils import load_s10_decoder

P1_CKPT = ROOT / "runs/v19/students/s2_prism1_v2.pt"
V20_CKPT = ROOT / "runs/v20/slf_flow/slf_core_v2.pt"
B4_CKPT = ROOT / "runs/v6/P1_B4_distill50/P1_B4_distill50.pt"

TEACHER_SPEC = {
    "d6_ckpt": str(V20_CKPT.relative_to(ROOT)),
    "anchor": "B4_distill50 + canonicalizer(c1_gentle) [T4-equivalent]",
    "solver": {k: v for k, v in D6H_SOLVER.items()},
    "branches": "cs_j",
    "gate": "original",
    "p1_ckpt": str(P1_CKPT.relative_to(ROOT)),
    "p1_lambda": 1.0,
}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    sanitize_cuda_env()
    seed_all(20263500)
    device = args.device
    cache_p = OUT / "cache/train_latents_4400.pt"
    cache_p.parent.mkdir(parents=True, exist_ok=True)
    if cache_p.exists():
        print(f"[V25 pre] Reusing {cache_p}", flush=True)
        return

    bundle = build_or_load_v19_foundation(args.pool, device=device)
    tr = bundle["v18_cache"]["train"]
    z0_tr = tr["z"].float()
    sid_tr = tr["sid"].long()
    d16_tr = tr["dino_16"].float()
    c64_tr = tr["c_64"].float()
    N = len(z0_tr)
    print(f"[V25 pre] train N={N}", flush=True)

    # 跨风格目标分配 (确定性, tid != sid)
    g = torch.Generator().manual_seed(20263501)
    tid_tr = (sid_tr + 1 + torch.randint(0, 4, (N,), generator=g)) % 5

    # ---- 1) B4 + canonicalizer 训练侧 T4 等价锚 ----
    b4, _ = load_c3unet(B4_CKPT, device)
    canon = canonicalizer(device)
    z_anc_tr = torch.zeros_like(z0_tr)
    for st in range(0, N, 50):
        sl = slice(st, min(st + 50, N))
        zb = z0_tr[sl].to(device)
        c_can = canon(zb, mode="c1_gentle", augment=False)
        z_anc_tr[sl] = b4(c_can, 0.0, c_can, tid_tr[sl].to(device)).cpu().float()
    print(f"[V25 pre] B4 anchors done", flush=True)

    # ---- 2) PRISM-1 (λ=1.0) 训练侧输出 ----
    fnd = bundle["fnd"]
    stu = SpatiallyRoutedChordNet(width=96, chord_dim=128, use_spatial_router=True).to(device).eval()
    stu.load_state_dict(torch.load(P1_CKPT, map_location="cpu", weights_only=False)["model"])
    fiber_enc = StyleFiberEncoder(in_dim=fnd["tr_fib"].shape[1], dim=128).to(device).eval()
    fiber_enc.load_state_dict(fnd["fiber_state"])
    u_all = fiber_enc.encode(fnd["tr_fib"].to(device).float())
    bary = fiber_enc.class_barycenters()
    priors = {k: v.to(device) if torch.is_tensor(v) else v for k, v in fnd["priors"].items()}
    P_Y = priors["P_Y"]
    del fiber_enc

    z_p1_tr = torch.zeros_like(z0_tr)
    for st in range(0, N, 50):
        sl = slice(st, min(st + 50, N))
        zs = z0_tr[sl].to(device)
        tid = tid_tr[sl].to(device)
        z_anc = apply_latent_chroma_shift(z0_tr[sl].to(device), tid, priors, alpha=0.20)
        chord = bary[tid] - u_all[sl]
        z_luma = torch.einsum("ij,bjhw->bihw", P_Y, zs)
        delta = stu(z_anc, z_luma, c64_tr[sl].to(device), d16_tr[sl].to(device), chord, same_mask=torch.zeros(len(zs), dtype=torch.bool, device=device))
        z_p1_tr[sl] = (z_anc + delta).cpu().float()
    del stu
    torch.cuda.empty_cache()
    print(f"[V25 pre] PRISM-1 targets done", flush=True)

    # ---- 3) D6 教师 (cs_j + D6H + B4 锚, 与评测侧三元组一致) ----
    bank = build_or_load_core_knn_bank(bundle, k_sub=8, core_per_style=180, device=device)
    tasks_dummy = [{"source_style_id": int(s), "target_style_id": int(t)} for s, t in zip(sid_tr.tolist(), tid_tr.tolist())]
    eps = torch.randn(z0_tr.shape, generator=torch.Generator().manual_seed(20263502))
    dec = load_s10_decoder(device)  # 保持解码器在场以防下游需要; 本脚本不解码
    z_d6_tr, _ = run_core_flow(bundle, bank, V20_CKPT, tasks_dummy, z0_tr, z_anc_tr,
                               y16_750=coarse_y16_lcn(tr["y_64"].float()), d16_750=d16_tr,
                               eps_750=eps, branches="cs_j", solver=dict(D6H_SOLVER), device=device)
    del bank
    torch.cuda.empty_cache()
    print(f"[V25 pre] D6 teacher done", flush=True)

    torch.save({
        "z0_tr": z0_tr.half(), "z_anc_tr": z_anc_tr.half(), "z_p1_tr": z_p1_tr.half(),
        "z_d6_tr": z_d6_tr.half(), "sid_tr": sid_tr, "tid_tr": tid_tr,
        "teacher_spec": TEACHER_SPEC, "updated_utc": utc(),
    }, cache_p)
    save_json(OUT / "precompute_train_latents.json", {"teacher_spec": TEACHER_SPEC, "n": N, "updated_utc": utc()})
    print(f"[V25 pre] Saved {cache_p}", flush=True)


if __name__ == "__main__":
    main()
