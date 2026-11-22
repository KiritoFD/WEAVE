r"""P0-0: 宿主接口审计 + 免标签锚点解耦 + 强教师熔断验收 (零训练, 用户 2026-11-22 立项).

三陷阱对策:
  陷阱1 (锚点泄漏/拔锚暴跌): 96 cross 任务四锚对比 (宿主 = V23_HiC_JPk) —
    A_t4 (泄漏上界对照, tid_true 生成) / B_p1_1shot (合法: oneshot chord + 预测tid色度平移) /
    C_z0perp (合法: 色度矩对齐) / D_src (合法: 源锚原生 SDEdit, 量化锚点悬崖).
    条件侧全部免标签: tid_pred = argmax cos(u_ref_emb, class_barycenters),
    u_vivid = encode(fib_ref) (单参考直连), dsd_ref = bank[tid_pred] 均值.
  陷阱2 (单宿主盲押): 接口审计三宿主; 生成解耦跑 V23_HiC_JPk (内容最优) —
    V25_Tr_T1 与 V20_D2 (4 分支复杂配方) 的解耦放 P0-1 (同脚本换 cfg_host 即可).
  陷阱3 (5 类隐式分类器): 教师熔断硬门控 (红线1) + tid_pred 从参考预测 (红线2).

红线: assert_label_free (2) / teacher_fuse_gate (1) / dual_gate_record (3) 全部启用.

用法:  python -m exp.v26.gpu4090.p0_host_audit --pool ../pseudo_gt_24h/orig_mean_fp16
       --parts audit,anchor,teacher
"""
from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from exp.v26 import OUT, ROOT, sanitize_cuda_env, save_json, utc
from exp.v26.gpu4090.p0_common import (STYLES, assert_label_free, chroma_align_anchor,
                                       dual_gate_record, teacher_fuse_gate, _pil)

N_ANCHOR_TASKS = 96
HOSTS = {
    "V23_HiC_JPk": dict(
        ckpt=ROOT / "runs/v23/joint_c1ab/slf_Joint_Pk_C1AB.pt",
        solver=dict(tau_min=0.36, tau_hi=0.72, steps=10, ws_fig=6.5, wsp_fig_late=2.4,
                    wsp_abs=5.2, wc_fig=1.85, wc_abs=0.80, lscale_abs=0.34, rescale_phi=0.48)),
    "V25_Tr_T1": dict(
        ckpt=ROOT / "runs/v25/r2_tmatrix/slf_Tr_T1.pt",
        solver=dict(tau_min=0.28, tau_hi=0.66, steps=10, ws_fig=4.2, wsp_fig_late=2.4,
                    wsp_abs=5.2, wc_fig=1.85, wc_abs=0.80, lscale_abs=0.34, rescale_phi=0.48)),
}


def _csd_top1(stack, pil_imgs, want: list[int]) -> list[int]:
    feats = stack.extract(pil_imgs, batch=24)
    cs = feats["csd"] @ stack.csd_prototypes.T
    return cs.argmax(1).tolist()


def _decode(dec, z):
    from src.utils import decode_s10
    return decode_s10(dec, z.to(device="cuda"), batch=min(len(z), 24)).clamp(0.0, 1.0)


@torch.no_grad()
def part2_anchor_decouple(pool: str, device: str = "cuda") -> dict:
    """96 cross 任务 × 4 锚 (宿主 V23): 免标签条件 + 锚点解耦 Top-1/DINO-C."""
    from exp.v18 import tasks_and_sources
    from exp.v19.fiber_duality import (StyleFiberEncoder, apply_latent_chroma_shift)
    from exp.v19.routed_student import SpatiallyRoutedChordNet
    from exp.v20.slf_flow import coarse_y16_lcn
    from exp.v22.core import CFGConfig, GateConfig, PrismCore
    from exp.v20.submode_patchswd import build_or_load_core_knn_bank
    from exp.v19.fiber_duality import build_or_load_v19_foundation
    from src.utils import FeatureStack, load_s10_decoder

    bundle = build_or_load_v19_foundation(pool, device=device)
    bank = build_or_load_core_knn_bank(bundle, device=device)
    fnd = bundle["fnd"]
    enc = StyleFiberEncoder(in_dim=fnd["tr_fib"].shape[1], dim=128).to(device).eval()
    enc.load_state_dict(fnd["fiber_state"])
    bary = enc.class_barycenters()  # (5,128)
    priors = {k: v.to(device) if torch.is_tensor(v) else v for k, v in fnd["priors"].items()}
    P_Y = priors["P_Y"]

    tasks, z_src_750, _ = tasks_and_sources()
    cross = [i for i, t in enumerate(tasks) if t["source_style_id"] != t["target_style_id"]]
    g0 = torch.Generator().manual_seed(20266300)
    cross = [cross[j] for j in torch.randperm(len(cross), generator=g0)[:N_ANCHOR_TASKS]]
    sel = torch.tensor(cross)
    tid_true = torch.tensor([tasks[i]["target_style_id"] for i in cross])
    z_src = z_src_750[sel].to(device).float()
    d16 = bundle["test_struct"]["dino_16"][sel].to(device).float()
    w_l = bundle["v18_cache"]["luma_proj"]["w"].float().view(1, 4, 1, 1).to(device)
    b_l = bundle["v18_cache"]["luma_proj"]["b"].float().to(device)
    y16 = coarse_y16_lcn(((z_src * w_l).sum(1, keepdim=True) + b_l).clamp(0.0, 1.0)).float()

    # 免标签条件: 参考图 = 目标画派 SG-5 exemplar (任务定义的 1-shot 参考, 与官方协议一致);
    # tid_pred 由参考纤维对 class barycenter 的余弦预测 (贯穿色度平移与 dsd 选择).
    c21 = torch.load(ROOT / "runs/v21/cache/v21_stiefel_ref_cache.pt",
                     map_location="cpu", weights_only=False)
    from src.utils import decode_s10 as _dec
    dec = load_s10_decoder(device)
    z_ref5 = c21["sg5"]["z"].float().to(device)
    rgb_ref5 = _dec(dec, z_ref5, batch=5).clamp(0.0, 1.0)
    stack = FeatureStack(device)
    from exp.v19.fiber_duality import extract_color_fiber_27d, extract_dino_rgb_moments_768d
    from src.utils._local_eval_refs import load_dino_model
    dino_m = load_dino_model(device=device)
    fib_ref5 = torch.cat([extract_color_fiber_27d(x[None]) for x in rgb_ref5], 0).to(device)
    fib_ref5 = torch.cat([fib_ref5, extract_dino_rgb_moments_768d(
        dino_m, rgb_ref5.to(device), batch=5, device=device).to(device)], dim=-1)
    u_ref_emb = enc.encode(fib_ref5.to(device))  # (5,128) — 每画派 exemplar 的单参考向量
    tid_pred_ex = (F.normalize(u_ref_emb) @ F.normalize(bary).T).argmax(1)  # (5,) 由参考预测
    tid_pred = tid_pred_ex[tid_true]  # 任务的参考 (目标画派 exemplar) 所预测的画派 id
    assert_label_free({"anchor_variants": "A_t4/B_p1_1shot/C_z0perp/D_src",
                       "cond": "u_vivid=encode(fib_ref); tid_pred=argmax cos(u_ref,bary)"})

    t4 = torch.load(ROOT / "runs/v22/cache/t4_latents_750.pt", map_location="cpu", weights_only=False)
    z_t4 = t4["z_t4"][sel].to(device).float()  # 泄漏上界对照 (tid_true 生成)

    # PRISM-1 1-shot 锚 (合法): chord = u_ref_emb − u_src_emb, 色度平移 tid=tid_pred
    stu = SpatiallyRoutedChordNet(width=96, chord_dim=128, use_spatial_router=True).to(device).eval()
    ck19 = torch.load(ROOT / "runs/v19/students/s2_prism1_v2.pt", map_location="cpu", weights_only=False)
    stu.load_state_dict(ck19["model"])
    u_src_emb = enc.encode(fnd["test750_fib"][sel].to(device).float())
    z_anc = apply_latent_chroma_shift(z_src.clone(), tid_pred, priors, alpha=0.20)
    z_luma = torch.einsum("ij,bjhw->bihw", P_Y, z_src)
    c64 = bundle["test_struct"]["c_64"][sel].to(device).float()
    chord = u_ref_emb[tid_true].to(device) - u_src_emb
    delta = stu(z_anc, z_luma, c64, d16, chord,
                same_mask=torch.zeros(len(sel), dtype=torch.bool, device=device))
    anchor_p1 = (z_anc + delta).detach()
    del stu
    torch.cuda.empty_cache()

    results = {}
    host = "V23_HiC_JPk"
    cfg_host = HOSTS[host]
    core = PrismCore(bundle=bundle, core_bank=bank, ckpt_path=cfg_host["ckpt"], device=device)
    for atag, anchor in (
            ("A_t4_native", z_t4),
            ("B_p1_1shot", anchor_p1),
            ("C_z0perp", chroma_align_anchor(z_src, z_ref5[tid_true])),
            ("D_src", z_src)):
        eps = torch.randn(z_src.shape, generator=torch.Generator().manual_seed(20261185)).to(device)
        u_style = u_ref_emb[tid_true].to(device)
        dsd_ref = bank["sub_dsd"].to(device).float()[tid_pred].mean(dim=1)
        z_gen, _ = core.integrate_flow(eps, anchor, y16, d16, tid_pred,
                                       u_style=u_style, dsd_ref=dsd_ref,
                                       gate=GateConfig(mode="original"),
                                       cfg=CFGConfig(branches="cs_j"),
                                       solver=cfg_host["solver"])
        pred_all, dc_all = [], []
        for st in range(0, len(sel), 24):
            rgb = _decode(dec, z_gen[st:st + 24].float())
            feats = stack.extract([_pil(x) for x in rgb], batch=24)
            cs = feats["csd"] @ stack.csd_prototypes.T
            pred_all += cs.argmax(1).tolist()
            f_src = stack.extract([_pil(x) for x in _decode(dec, z_src[st:st + 24])], batch=24)["dino"]
            dc_all += torch.diagonal(feats["dino"] @ f_src.T).tolist()
        want = tid_true.tolist()
        res = {"top1": round(sum(int(p == w) for p, w in zip(pred_all, want)) / len(pred_all), 4),
               "dino_c_vs_src": round(sum(dc_all) / len(dc_all), 4)}
        results[f"{host}_{atag}"] = res
        dual_gate_record(f"P0AUD_{host}_{atag}",
                         {"csd_top1": res["top1"], "dino_c": res["dino_c_vs_src"]},
                         None, out_name=f"p0_dualgate_{host}_{atag}.json")
        print(f"[P0-0.2] {host}_{atag}: {res}", flush=True)
    del core
    torch.cuda.empty_cache()
    return results


@torch.no_grad()
def part3_teacher_rebuild(pool: str, device: str = "cuda") -> dict:
    """强教师重造: V23 宿主 + 原生条件 (true tid, core8 bank) + z_p1_tr 锚 → 熔断验收."""
    from exp.v19.fiber_duality import build_or_load_v19_foundation
    from exp.v20.slf_flow import coarse_y16_lcn
    from exp.v22.core import CFGConfig, GateConfig, PrismCore
    from exp.v20.submode_patchswd import build_or_load_core_knn_bank
    from src.utils import FeatureStack, load_s10_decoder

    bundle = build_or_load_v19_foundation(pool, device=device)
    bank = build_or_load_core_knn_bank(bundle, device=device)
    tr = bundle["v18_cache"]["train"]
    sid_tr = tr["sid"].long()
    d16_tr = tr["dino_16"]
    w_l = bundle["v18_cache"]["luma_proj"]["w"].float().view(1, 4, 1, 1).to(device)
    b_l = bundle["v18_cache"]["luma_proj"]["b"].float().to(device)
    z_tr = tr["z"].float().to(device)
    y16_tr = coarse_y16_lcn(((z_tr * w_l).sum(1, keepdim=True) + b_l).clamp(0.0, 1.0)).half()
    tl = torch.load(ROOT / "runs/v25/cache/train_latents_4400.pt", map_location="cpu", weights_only=False)
    z_p1_tr = tl["z_p1_tr"].float()  # PRISM-1 锚 (教师监督允许 true tid 生成 — 红线2只管推理)
    cfg_host = HOSTS["V23_HiC_JPk"]
    core = PrismCore(bundle=bundle, core_bank=bank, ckpt_path=cfg_host["ckpt"], device=device)
    g = torch.Generator().manual_seed(20266400)
    lat_by_style: dict[int, list[torch.Tensor]] = {}
    for t in range(5):
        pool_i = (sid_tr != t).nonzero(as_tuple=True)[0]
        pick = pool_i[torch.randperm(len(pool_i), generator=g)[:120]]
        tid_true = torch.full((len(pick),), t, dtype=torch.long, device=device)
        lat_by_style[t] = []
        for st in range(0, len(pick), 24):
            chunk = pick[st:st + 24].tolist()
            B = len(chunk)
            eps = torch.randn(B, 4, 64, 64, generator=g).to(device)
            y16 = y16_tr[chunk].to(device).float()
            d16 = d16_tr[chunk].to(device).float()
            anchor = z_p1_tr[chunk].to(device)
            u_style, dsd_ref = core.build_style_condition(d16, tid_true, mode="core8")
            z_gen, _ = core.integrate_flow(eps, anchor, y16, d16, tid_true,
                                           u_style=u_style, dsd_ref=dsd_ref,
                                           gate=GateConfig(mode="original"),
                                           cfg=CFGConfig(branches="cs_j"),
                                           solver=cfg_host["solver"])
            lat_by_style[t] += [z_gen[j].cpu() for j in range(B)]
    stack = FeatureStack(device)
    dec = load_s10_decoder(device)
    fuse = teacher_fuse_gate(dec, stack, lat_by_style, n_per_style=30, device=device,
                             gate_top1=0.50, gate_min_style=0.20, tag="P0_V23_zp1")
    return fuse


def part1_interface_audit(pool: str, device: str = "cuda") -> dict:
    from exp.v22.core import PrismCore
    from exp.v20.submode_patchswd import build_or_load_core_knn_bank
    from exp.v19.fiber_duality import build_or_load_v19_foundation
    bundle = build_or_load_v19_foundation(pool, device=device)
    bank = build_or_load_core_knn_bank(bundle, device=device)
    audit = {}
    for host, cfg_host in HOSTS.items():
        core = PrismCore(bundle=bundle, core_bank=bank, ckpt_path=cfg_host["ckpt"], device=device)
        n_par = sum(p.numel() for p in core.model.parameters())
        audit[host] = {"ckpt": str(cfg_host["ckpt"]), "params_m": round(n_par / 1e6, 2),
                       "interface": "SLFUNet(z,y16,d16,style_id,u_vivid,tau) cond=199d "
                                    "[time64|onehot5|u_vivid128|drops2] — 连续向量+离散onehot并存",
                       "solver": cfg_host["solver"]}
        del core
        torch.cuda.empty_cache()
    audit["V20_D2"] = {"ckpt": "runs/v20/slf_flow/slf_core_v2.pt (4090 独占)",
                       "interface": "同 SLFUNet; 4 分支 CFG 曲线 (run_v20.run_stage2c_core_slf) — 生成解耦放 P0-1",
                       "note": "T4 锚 = V19_T4_sweet_vgg_nosid, tid_true 泄漏 (t4.py L68), 仅作上界对照"}
    audit["label_leak_audit"] = {
        "t4_anchor": "LEAKED (tid_true) — 仅上界对照, 禁止用于 1-Shot 评测",
        "compute_prism1_anchor_750": "LEAKED (chord=bary[tid_true]) — 闭集口径",
        "prism1_oneshot_mode": "LEGAL (u_tgt = encode(fib_ref), routed_student.py:413)",
        "condition_bank": "sub_u (5,8,128) 连续向量, 选择依赖 tid + 源场景 → 1-shot 用 tid_pred 替代",
    }
    return audit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--parts", default="audit,anchor,teacher")
    args = ap.parse_args()
    sanitize_cuda_env()
    out = {"task": "P0-0 host audit + anchor decouple + teacher fuse", "updated_utc": utc()}
    if "audit" in args.parts:
        out["part1_interface_audit"] = part1_interface_audit(args.pool, device=args.device)
        print(f"[P0-0.1] {json.dumps(out['part1_interface_audit'], default=str)[:600]}", flush=True)
    if "anchor" in args.parts:
        out["part2_anchor_decouple"] = part2_anchor_decouple(args.pool, device=args.device)
    if "teacher" in args.parts:
        out["part3_teacher_fuse"] = part3_teacher_rebuild(args.pool, device=args.device)
    save_json(OUT / "gpu4090" / "p0_host_audit.json", out)
    print(f"[P0-0] saved → {OUT / 'gpu4090' / 'p0_host_audit.json'}", flush=True)


if __name__ == "__main__":
    main()
