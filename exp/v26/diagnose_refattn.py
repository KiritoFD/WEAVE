r"""V26 K1 (4090): RefAttnV24 因果干预审计 + Source-Lock 速度场几何诊断 (零训练).

docs/1120/06_4090_diagnostic_queue.md §K1 (调整版: C2nt 降级后本实验的目标改为
"查明任意参考瓶颈, 指导把 PRISM-Flow 强迁移转化为任意参考能力").

三部分
------
K1-A Hook 遥测: 对 C2nt checkpoint 的 ref_attn16 做插桩前向, 在 test750 前 N 条
     cross 任务上记录 (按 src→tgt 画派分组):
       attention entropy / log(256), max(QK), gate g 均值/分位数,
       ||g·V_local|| vs ||(1-g)·u_global|| 范数比, ||out_proj(R)|| / ||h16||.

K1-B 因果干预 (R0–R6, 同 ckpt 固定噪声/内容/strength, 分布外压力测试):
       R0 原模型 | R1 g=0 纯全局 | R2 g=1 纯局部 | R3 V=0 | R4 跨样本打乱 V |
       R5 只换 K | R6 只换 V.
     150 条 cross 任务 in-process DINO 评分 (dino_c vs 源内容, dino_s vs 目标参考,
     dino_s vs 源图) — 不走裁判子进程 (因果审计用代理评分, 干预后退化不等于重训表现).

K1-C 速度场几何: 训练集 64 组 (内容, 跨风格参考) 对, 在 tau 网格上比较
     v_pos / v_src / v_diff 的范数、夹角、亮度/色度能量占比 — 判断 SR-CFG
     差分方向是否有实质风格驱动分量 (非严格正交, 仅几何诊断).

产出: runs/v26/gpu4090/k1_refattn_causal.json + docs/v26/4090/K1_REFATTN_CAUSAL_AUDIT.md

用法 (4090):  python -m exp.v26.diagnose_refattn --pool ../pseudo_gt_24h/orig_mean_fp16
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import MethodType

import torch
import torch.nn.functional as F
from PIL import Image as PILImage

from . import C2NT_CKPT, OUT, ROOT, V21_CACHE, sanitize_cuda_env, save_json, seed_all, utc
from exp.v18 import tasks_and_sources
from exp.v19.fiber_duality import StyleFiberEncoder, build_or_load_v19_foundation
from exp.v20.slf_flow import coarse_y16_lcn
from exp.v20.submode_patchswd import load_convnext_tiny_offline
from exp.v22.ref_adapter import RefAdapter
from exp.v24.p0_o1_o3_value import SLFUNetV24OneShot, _sg5_cnx_v16
from exp.v23.p2_attn_cross import _sg5_ref_d16
from exp.v26.sr_cfg import chroma_projector, integrate_sr, sr_velocity, load_sr_env
from src.utils import decode_s10, load_s10_decoder

REPORT_MD = ROOT / "docs/v26/4090/K1_REFATTN_CAUSAL_AUDIT.md"
N_TELEMETRY_TASKS = 150
N_INTERVENTION_TASKS = 150
INTERVENTIONS = ("R0_original", "R1_gate0", "R2_gate1", "R3_vzero", "R4_vshuffle",
                 "R5_key_only", "R6_value_only")
N_VELOCITY_PAIRS = 64
TAU_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


# ------------------------------------------------- 插桩前向 ----

def _instrumented_forward(self, h16, d16_c, d16_r, v16_r, u_glob, smask2, cmask4,
                          *, stats: list | None = None, mode: str = "R0_original"):
    """RefAttnV24.forward 的插桩/干预版 (严格复刻原实现, 仅注入记录与干预)."""
    B, _, H, W = h16.shape
    dev = h16.device
    Q = F.normalize(self.q_proj(d16_c.flatten(2).transpose(1, 2)), dim=-1)
    K = F.normalize(self.k_proj(d16_r.flatten(2).transpose(1, 2)), dim=-1)
    V = self.v_proj(v16_r.flatten(2).transpose(1, 2))

    if mode == "R5_key_only":      # 只换 K: 语义匹配错位, Value 保持正确
        K = K.roll(1, dims=0)
    if mode in ("R4_vshuffle", "R6_value_only"):  # 打乱/替换 Value: 纹理载荷错位
        V = V.roll(1, dims=0)
    if mode == "R3_vzero":         # 局部 Value 置零
        V = torch.zeros_like(V)

    sim = torch.bmm(Q, K.transpose(1, 2))
    R = torch.bmm(torch.softmax(sim * 12.0, dim=-1), V)

    if self.use_conf_gate:
        max_sim = sim.max(dim=-1, keepdim=True).values
        g_i = torch.sigmoid(self.conf_temp * (max_sim + self.conf_bias))
        u_g = self.glob_proj(u_glob).unsqueeze(1).expand_as(R)
        if mode == "R1_gate0":
            g_i = torch.zeros_like(g_i)
        elif mode == "R2_gate1":
            g_i = torch.ones_like(g_i)
        R = g_i * R + (1.0 - g_i) * u_g

    if stats is not None:
        with torch.no_grad():
            p = torch.softmax(sim * 12.0, dim=-1)
            ent_per = (-(p.clamp_min(1e-9) * p.log()).sum(-1).mean(-1) / math.log(256))  # (Btot,)
            max_tok = sim.max(dim=-1).values                                             # (Btot, 256) 逐 token (与门控同口径)
            max_per = max_tok.mean(-1)                                                   # (Btot,) 逐样本
            if self.use_conf_gate:
                g_tok = torch.sigmoid(self.conf_temp * (max_tok.unsqueeze(-1) + self.conf_bias))  # (Btot,256,1)
                g_per = g_tok.squeeze(-1).mean(-1)                                       # (Btot,)
                loc_per = (g_tok * R).norm(dim=-1).mean(-1)
                u_g = self.glob_proj(u_glob).unsqueeze(1).expand_as(R)
                glob_per = ((1.0 - g_tok) * u_g).norm(dim=-1).mean(-1)
            else:
                g_per = torch.ones_like(max_per)
                loc_per = R.norm(dim=-1).mean(-1)
                glob_per = torch.zeros_like(max_per)
            out_map = self.out_proj(R.transpose(1, 2).view(B, -1, H, W))
            out_rel_per = (out_map.flatten(1).norm(dim=1)
                           / h16.flatten(1).norm(dim=1).clamp_min(1e-8))                 # (Btot,)
            stats.append({
                "entropy_ratio": ent_per.detach().cpu(),
                "max_qk": max_per.detach().cpu(),
                "gate": g_per.detach().cpu(),
                "local_norm": loc_per.detach().cpu(),
                "global_norm": glob_per.detach().cpu(),
                "out_rel_norm": out_rel_per.detach().cpu(),
            })

    R_map = R.transpose(1, 2).view(B, -1, H, W)
    return h16 + self.out_proj(R_map) * (smask2.view(B, 1, 1, 1) * cmask4)


def patch_refattn(fm, stats: list | None = None, mode: str = "R0_original") -> MethodType:
    import functools
    return MethodType(functools.partial(_instrumented_forward, stats=stats, mode=mode), fm.ref_attn16)


# ------------------------------------------------- K1-A 遥测 ----

@torch.no_grad()
def k1a_telemetry(env: dict, device: str = "cuda", n_tasks: int = N_TELEMETRY_TASKS) -> dict:
    tasks, z_src_750, _ = tasks_and_sources()
    cross_ids = [i for i, t in enumerate(tasks) if t["source_style_id"] != t["target_style_id"]][:n_tasks]
    sel = torch.tensor(cross_ids, dtype=torch.long)
    fm, ad = env["fm"], env["ad"]
    dec = load_s10_decoder(device)
    w_l, b_l = env["w_l"].to(device), env["b_l"].to(device)
    stats: list[dict] = []
    fm.ref_attn16.forward = patch_refattn(fm, stats=stats, mode="R0_original")

    zs = z_src_750[sel].to(device).float()
    sid_p = torch.tensor([tasks[i]["source_style_id"] for i in cross_ids], device=device)
    tid_t = torch.tensor([tasks[i]["target_style_id"] for t_i, i in enumerate(cross_ids)], device=device) \
        if False else torch.tensor([tasks[i]["target_style_id"] for i in cross_ids], device=device)
    B = len(sel)
    y16 = coarse_y16_lcn(((zs * w_l).sum(1, keepdim=True) + b_l).clamp(0.0, 1.0)).float()
    d16 = env["bundle"]["test_struct"]["dino_16"][sel].to(device).float()
    u_src = env["u_src_750"][sel].to(device)
    tid_cpu = tid_t.cpu()
    u_style = ad(env["u_sg5"][tid_cpu].to(device), u_src)
    u_glob = env["fib_n_sg5"][tid_cpu]  # SG-5 目标画派纤维 (协议口径)
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

    integrate_sr(vel_fn, z, tau0, 10, "rk2")
    del fm.ref_attn16.forward  # 还原 (移除实例属性补丁, 回落类方法)

    # stats: 每 fm call 一条 {key: (Btot,) 张量}; baseline CFG call 为 cat([zz,zz]) →
    # 前 B 行 = 条件分支 (目标参考), 与 sel 任务一一对应. RK2 每步 2 次 vel_fn → 20 calls.
    B = len(sel)
    cond_keys = ("entropy_ratio", "max_qk", "gate", "local_norm", "global_norm", "out_rel_norm")
    per_task = {k: {key: [] for key in cond_keys} for k in range(B)}
    for call in stats:
        for key in cond_keys:
            cond = call[key][:B]
            for k in range(B):
                per_task[k][key].append(float(cond[k]))

    def task_row(k: int) -> dict:
        r = {key: round(sum(v) / len(v), 4) for key, v in per_task[k].items() if key != "gate"}
        g = per_task[k]["gate"]
        r["gate_mean"] = round(sum(g) / len(g), 4)
        gs = sorted(g)
        r["gate_p10"] = round(gs[int(0.10 * (len(gs) - 1))], 4)
        r["gate_p90"] = round(gs[int(0.90 * (len(gs) - 1))], 4)
        r["gate_low_frac"] = round(sum(1 for x in g if x < 0.2) / len(g), 4)
        return r

    by_tgt, by_src = {}, {}
    for k, i in enumerate(cross_ids):
        t = tasks[i]
        row = task_row(k)
        by_tgt.setdefault(t["target_style"], []).append(row)
        by_src.setdefault(t["source_style"], []).append(row)
    agg_keys = ("entropy_ratio", "max_qk", "gate_mean", "gate_p10", "gate_p90",
                "gate_low_frac", "local_norm", "global_norm", "out_rel_norm")
    agg = lambda rows: {k: round(float(sum(r[k] for r in rows) / len(rows)), 4) for k in agg_keys}
    return {
        "n_tasks": B,
        "n_calls": len(stats),
        "overall": agg([task_row(k) for k in range(B)]),
        "by_target_style": {k: agg(v) for k, v in sorted(by_tgt.items())},
        "by_source_style": {k: agg(v) for k, v in sorted(by_src.items())},
    }


# ------------------------------------------------- K1-B 干预 ----

@torch.no_grad()
def k1b_interventions(env: dict, device: str = "cuda", n_tasks: int = N_INTERVENTION_TASKS) -> dict:
    from src.utils import FeatureStack
    tasks, z_src_750, _ = tasks_and_sources()
    cross_ids = [i for i, t in enumerate(tasks) if t["source_style_id"] != t["target_style_id"]][:n_tasks]
    sel = torch.tensor(cross_ids, dtype=torch.long)
    fm, ad = env["fm"], env["ad"]
    dec = load_s10_decoder(device)
    stack = FeatureStack(device)
    w_l, b_l = env["w_l"].to(device), env["b_l"].to(device)

    zs = z_src_750[sel].to(device).float()
    B = len(sel)
    tid_t = torch.tensor([tasks[i]["target_style_id"] for i in cross_ids], device=device)
    tid_cpu = tid_t.cpu()
    y16 = coarse_y16_lcn(((zs * w_l).sum(1, keepdim=True) + b_l).clamp(0.0, 1.0)).float()
    d16 = env["bundle"]["test_struct"]["dino_16"][sel].to(device).float()
    u_src = env["u_src_750"][sel].to(device)
    u_style = ad(env["u_sg5"][tid_cpu].to(device), u_src)
    u_glob = env["fib_n_sg5"][tid_cpu]  # SG-5 目标画派纤维 (协议口径)
    d16_r = env["d16_sg5"][tid_cpu].to(device)
    v16_r = env["v16_sg5"][tid_cpu].to(device)
    tau0, wj = 0.45, 2.2
    eps = torch.randn(B, 4, 64, 64, generator=torch.Generator().manual_seed(20261185)).to(device)
    z = (1.0 - tau0) * eps + tau0 * zs

    rgb_src = decode_s10(dec, zs, batch=16).clamp(0.0, 1.0)
    z_ref5 = env["c21"]["sg5"]["z"].float().to(device)
    rgb_ref5 = decode_s10(dec, z_ref5, batch=5).clamp(0.0, 1.0)

    def dino_of(imgs):
        pil = [PILImage.fromarray((x.permute(1, 2, 0).cpu().numpy() * 255).round().clip(0, 255).astype("uint8"))
               for x in imgs]
        return stack.extract(pil, batch=16)["dino"].cpu()

    f_content = dino_of(rgb_src)
    f_ref5 = dino_of(rgb_ref5)[tid_cpu]          # (B, D) 各任务目标参考特征
    f_src_style = dino_of(rgb_src)               # 源图自身风格代理 (源图)

    out = {"n_tasks": B, "arms": {}}
    for mode in INTERVENTIONS:
        fm.ref_attn16.forward = patch_refattn(fm, stats=None, mode=mode)

        def vel_fn(zz, prog):
            tv = torch.full((B,), tau0 + prog * (1.0 - tau0), device=device)
            v, _ = sr_velocity(fm, ad, u_style, None, u_glob, None, d16, d16_r, v16_r,
                               None, None, y16, zz, tv, mode="baseline", w=0.0, wj=wj, device=device)
            return v

        z_gen = integrate_sr(vel_fn, z, tau0, 10, "rk2")
        rgb_gen = decode_s10(dec, z_gen, batch=16).clamp(0.0, 1.0)
        f_gen = dino_of(rgb_gen)
        row = {
            "dino_c_vs_content": round(float(F.cosine_similarity(f_gen, f_content, dim=-1).mean()), 5),
            "dino_s_vs_target_ref": round(float(F.cosine_similarity(f_gen, f_ref5, dim=-1).mean()), 5),
            "dino_s_vs_source": round(float(F.cosine_similarity(f_gen, f_src_style, dim=-1).mean()), 5),
        }
        out["arms"][mode] = row
        print(f"[K1-B {mode}] {row}", flush=True)
    del fm.ref_attn16.forward  # 还原 (移除实例属性补丁)
    return out


# ------------------------------------------------- K1-C 速度场 ----

@torch.no_grad()
def k1c_velocity_field(env: dict, device: str = "cuda", n_pairs: int = N_VELOCITY_PAIRS) -> dict:
    bundle = env["bundle"]
    tr = bundle["v18_cache"]["train"]
    c21 = env["c21"]
    z_all, d16_all, sid_all = tr["z"], tr["dino_16"], tr["sid"].long()
    nbr_idx = c21["train"]["nbr_idx"]
    ad, fm = env["ad"], env["fm"]
    fiber_enc = StyleFiberEncoder(in_dim=bundle["fnd"]["tr_fib"].shape[1], dim=128).to(device).eval()
    fiber_enc.load_state_dict(bundle["fnd"]["fiber_state"])
    u_all = fiber_enc.encode(bundle["fnd"]["tr_fib"].to(device).float())
    del fiber_enc

    g = torch.Generator().manual_seed(20261185)
    ic_list, ir_list = [], []
    while len(ic_list) < n_pairs:
        c = int(torch.randint(0, len(z_all), (1,), generator=g))
        cols = torch.randint(0, nbr_idx.shape[1], (4,), generator=g).tolist()
        r = next((int(nbr_idx[c, k]) for k in cols if int(sid_all[int(nbr_idx[c, k])]) != int(sid_all[c])), None)
        if r is not None:
            ic_list.append(c); ir_list.append(r)
    ic = torch.tensor(ic_list); ir = torch.tensor(ir_list)

    w_l, b_l = env["w_l"].to(device), env["b_l"].to(device)
    wy = w_l.flatten() / w_l.flatten().norm()
    pc = chroma_projector(w_l).to(device)

    rows = []
    for k in range(0, len(ic), 16):
        sl = slice(k, min(k + 16, len(ic)))
        B = len(ic[sl])
        zs = z_all[ic[sl]].to(device).float()
        y16 = coarse_y16_lcn(((zs * w_l).sum(1, keepdim=True) + b_l).clamp(0.0, 1.0)).float()
        d16 = d16_all[ic[sl]].to(device).float()
        d16_r = d16_all[ir[sl]].to(device).float()
        v16_r = env["v16_tr"][ir[sl]].to(device).float()
        u_src = u_all[ic[sl]]
        u_style_t = ad(u_all[ir[sl]], u_src)
        u_style_s = ad(u_src, u_src)
        u_glob_t = ((bundle["fnd"]["tr_fib"].to(device).float()[ir[sl]]
                     - env["fib_mean"]) / env["fib_std"])
        u_glob_s = ((bundle["fnd"]["tr_fib"].to(device).float()[ic[sl]]
                     - env["fib_mean"]) / env["fib_std"])

        for prog in TAU_GRID:
            tv = torch.full((B,), 0.45 + prog * (1.0 - 0.45), device=device)
            zz = torch.randn(B, 4, 64, 64, generator=torch.Generator().manual_seed(20261185 + int(prog * 100))).to(device)
            v_pos, _ = sr_velocity(fm, ad, u_style_t, None, u_glob_t, None, d16, d16_r, v16_r,
                                   None, None, y16, zz, tv, mode="baseline", w=0.0, wj=2.2, device=device)
            v_src, _ = sr_velocity(fm, ad, u_style_s, None, u_glob_s, None, d16, d16, v16_r,
                                   None, None, y16, zz, tv, mode="baseline", w=0.0, wj=2.2, device=device)
            diff = v_pos - v_src
            cos_dp = float(F.cosine_similarity(diff.flatten(1), v_pos.flatten(1), dim=-1).mean())
            luma_e = float(((diff * wy.view(1, 4, 1, 1)).square().sum(dim=(1, 2, 3))).mean())
            total_e = float(diff.square().sum(dim=(1, 2, 3)).mean().clamp_min(1e-12))
            rows.append({
                "prog": prog,
                "norm_v_pos": round(float(v_pos.flatten(1).norm(dim=1).mean()), 4),
                "norm_v_src": round(float(v_src.flatten(1).norm(dim=1).mean()), 4),
                "norm_diff": round(float(diff.flatten(1).norm(dim=1).mean()), 4),
                "cos_diff_vpos": round(cos_dp, 4),
                "luma_energy_ratio": round(luma_e / max(total_e, 1e-9), 4),
            })
    agg = {}
    for prog in TAU_GRID:
        rs = [r for r in rows if r["prog"] == prog]
        agg[prog] = {k: round(sum(r[k] for r in rs) / len(rs), 4)
                     for k in ("norm_v_pos", "norm_v_src", "norm_diff", "cos_diff_vpos", "luma_energy_ratio")}
    return {"n_pairs": len(ic), "tau0": 0.45, "by_prog": agg,
            "note": "diff=v_pos−v_src 非严格正交分量; 几何诊断仅判断方向有效性"}


# ------------------------------------------------------- 报告 ----

def _write_report(telemetry: dict, interv: dict, vel: dict) -> None:
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    L = ["# K1 — RefAttnV24 因果干预与 Source-Lock 速度场审计 (V26, 零训练)",
         "", f"> 生成时间: {utc()} ｜ ckpt: `runs/v25/oneshot_val/C2nt_sw6_lc0.8.pt` (降级后的探索分支, 诊断用于任意参考瓶颈定位)",
         "",
         "## K1-A 遥测 (test750 前 150 条 cross 任务)",
         "",
         "| 分组 | entropy/log256 | max(QK) | gate 均值 | gate p10 | gate p90 | ‖g·V‖ | ‖(1-g)·G‖ | out/h16 |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    o = telemetry["overall"]
    L.append(f"| overall | {o['entropy_ratio']} | {o['max_qk']} | {o['gate_mean']} | {o['gate_p10']} | "
             f"{o['gate_p90']} | {o['local_norm']} | {o['global_norm']} | {o['out_rel_norm']} |")
    for k, v in telemetry["by_target_style"].items():
        L.append(f"| tgt:{k} | {v['entropy_ratio']} | {v['max_qk']} | {v['gate_mean']} | {v['gate_p10']} | "
                 f"{v['gate_p90']} | {v['local_norm']} | {v['global_norm']} | {v['out_rel_norm']} |")
    L += ["", "## K1-B 因果干预 (150 cross 任务, in-process DINO 代理评分)",
          "", "| 干预 | dino_c vs 内容 | dino_s vs 目标参考 | dino_s vs 源 |",
          "|---|---:|---:|---:|"]
    for mode, r in interv["arms"].items():
        L.append(f"| {mode} | {r['dino_c_vs_content']} | {r['dino_s_vs_target_ref']} | {r['dino_s_vs_source']} |")
    L += ["", "## K1-C 速度场几何 (64 训练对, tau0=0.45)",
          "", "| prog | ‖v_pos‖ | ‖v_src‖ | ‖v_pos−v_src‖ | cos(diff, v_pos) | 亮度能量占比 |",
          "|---|---:|---:|---:|---:|---:|"]
    for prog, r in vel["by_prog"].items():
        L.append(f"| {prog} | {r['norm_v_pos']} | {r['norm_v_src']} | {r['norm_diff']} | "
                 f"{r['cos_diff_vpos']} | {r['luma_energy_ratio']} |")
    L += ["", "## 判读 (docs/1120 调整版 §K1)", ""]
    r0 = interv["arms"]["R0_original"]; r1 = interv["arms"]["R1_gate0"]
    r2 = interv["arms"]["R2_gate1"]; r3 = interv["arms"]["R3_vzero"]; r6 = interv["arms"]["R6_value_only"]
    L.append(f"- 门控贡献: Δ(R1−R0) dino_s_ref = {round(r1['dino_s_vs_target_ref'] - r0['dino_s_vs_target_ref'], 4)} "
             f"(负值大 → 局部参考被门控压制);")
    L.append(f"- 局部 Value 重要性: Δ(R3−R0) dino_s_ref = {round(r3['dino_s_vs_target_ref'] - r0['dino_s_vs_target_ref'], 4)};")
    L.append(f"- 纹理载荷错位代价: Δ(R6−R0) dino_s_ref = {round(r6['dino_s_vs_target_ref'] - r0['dino_s_vs_target_ref'], 4)};")
    L.append("- 若 R3/R6 退化小 → 参考信息未被利用 (瓶颈在条件注入/训练, 指导 PRISM-Flow 任意参考设计"
             "优先改条件通路而非堆注意力); 若 R3/R6 退化大 → 局部 Value 有真实贡献 (瓶颈在频率/分辨率).")
    REPORT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[K1] Report → {REPORT_MD}", flush=True)


def run_k1(pool: str, device: str = "cuda") -> dict:
    seed_all(20261185)
    env = load_sr_env(pool, device=device)
    telemetry = k1a_telemetry(env, device=device)
    interv = k1b_interventions(env, device=device)
    vel = k1c_velocity_field(env, device=device)
    out = {"task": "V26-K1 RefAttn causal audit + source-lock velocity field",
           "updated_utc": utc(), "telemetry": telemetry, "interventions": interv,
           "velocity_field": vel}
    save_json(OUT / "gpu4090" / "k1_refattn_causal.json", out)
    _write_report(telemetry, interv, vel)
    return out


def main():
    from . import ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    sanitize_cuda_env()
    run_k1(args.pool, device=args.device)


if __name__ == "__main__":
    main()
