r"""V26-L L0: 真实跨画派参考采样因果实验 (A0/A1/A2) — 训练失败根因裁决.

docs/1120/08_v26_long_plan.md §L0. K1 已证明 C2nt 训练参考 ~100% 同画派 (nbr_idx
按同画派构造), V26 S1fix 只修了 accum RNG. 本实验做决定性因果对照:

  A0 legacy_nbr  : 旧同画派近邻参考 (原样, 对照)
  A1 strict_cross: 严格跨画派均匀采样 — 断言 sid_ref != sid_content, **无静默回退**
  A2 core_cross  : 严格跨画派 + 参考限定在各画派 Core 典型集合 (c21 core_idx_per_style)

同初始化 (DUAL_CKPT) / 同 λ (λ_sw=0.5, λ_c=1.0) / 同损失 (VGG) / 同预算 (1200 upd,
micro=8, accum=4, S1 accum-RNG 修复保持). 每臂训练全程统计 **5×5 配对矩阵**
(内容画派 × 参考画派) 并落盘. 评测: 全量 test750 (t45) + 参考更换敏感性 (24 对
正确/错误参考 in-process DINO 差).

用法:  python -m exp.v26.gpu4090.l0_crossstyle_causal --pool ../pseudo_gt_24h/orig_mean_fp16
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from exp.v26 import OUT, ROOT, save_json, utc
from exp.v26.gpu3060.train_c2nt_v26 import train_c2nt_v26
from exp.v25.mechanism_c import assign_c2nt_refs  # A0 原样对照
from exp.v26.sr_cfg import _eval_baseline_steps, load_sr_env
from src.utils import load_s10_decoder

L0_UPDATES = 1200
L0_SEED = 20265700


def assign_refs_strict(micro: int, n_total: int, sid_all: torch.Tensor,
                       generator: torch.Generator, *, core_idx=None,
                       n_group: int = 1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """严格跨画派参考分配 (A1/A2): 断言 sid_ref != sid_content, **永不回退同画派**.

    core_idx: A2 时为 5 个画派 Core 典型索引张量列表; None 则全训练集均匀 (A1).
    返回 (ic, ir, pair_matrix_delta) — pair_matrix 由调用方聚合.
    """
    assert micro >= 2 * n_group + 1
    ic_parts, ir_parts = [], []
    style_pool = [(sid_all == s).nonzero(as_tuple=True)[0] for s in range(5)]
    for _ in range(micro):
        c = int(torch.randint(0, n_total, (1,), generator=generator))
        sc = int(sid_all[c])
        others = torch.cat([style_pool[s] for s in range(5) if s != sc])
        if core_idx is not None:
            others = torch.cat([core_idx[s].to(others.device) for s in range(5) if s != sc])
        if len(others) == 0:
            raise RuntimeError(f"无跨画派参考候选 (content={c}, style={sc}) — 严格采样禁止回退")
        r = int(others[int(torch.randint(0, len(others), (1,), generator=generator))])
        if int(sid_all[r]) == sc:
            raise RuntimeError(f"严格采样被违反: content={c}({sc}) ref={r}({int(sid_all[r])})")
        ic_parts.append(c)
        ir_parts.append(r)
    return (torch.tensor(ic_parts, dtype=torch.long), torch.tensor(ir_parts, dtype=torch.long),
            None)


def build_pair_matrix(sid_all: torch.Tensor, pairs: list[tuple[int, int]]) -> list[list[int]]:
    """5×5 配对矩阵 (内容画派 × 参考画派)."""
    mat = [[0] * 5 for _ in range(5)]
    for c, r in pairs:
        mat[int(sid_all[c])][int(sid_all[r])] += 1
    return mat


@torch.no_grad()
def ref_swap_sensitivity(env: dict, device: str = "cuda", n_tasks: int = 24) -> dict:
    """参考更换敏感性: 同内容同噪声下 正确参考 vs 错误画派参考 的 dino_s 差.

    防御式: 任何异常 → 返回 swap=None (次级指标不杀臂管线).
    """
    try:
        return _ref_swap_impl(env, device=device, n_tasks=n_tasks)
    except Exception as exc:
        print(f"[L0 swap] FAILED (arm continues): {exc}", flush=True)
        return {"correct": None, "wrong": None, "swap_delta": None, "error": str(exc)[:200]}


def _ref_swap_impl(env: dict, device: str = "cuda", n_tasks: int = 24) -> dict:
    from src.utils import FeatureStack
    from PIL import Image as PILImage
    from exp.v18 import tasks_and_sources

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
    y16 = coarse_y16(device, env, zs)
    d16 = env["bundle"]["test_struct"]["dino_16"][sel].to(device).float()
    u_src = env["u_src_750"][sel].to(device)
    eps = torch.randn(B, 4, 64, 64, generator=torch.Generator().manual_seed(20261185)).to(device)

    def dino_of(imgs):
        pil = [PILImage.fromarray((x.permute(1, 2, 0).cpu().numpy() * 255).round().clip(0, 255).astype("uint8"))
               for x in imgs]
        return stack.extract(pil, batch=16)["dino"].cpu()

    f_ref5 = dino_of(decode_s10(dec, env["c21"]["sg5"]["z"].float().to(device), batch=5).clamp(0.0, 1.0))[tid_cpu]
    # 错误画派参考: 循环移位 tid
    tid_wrong = (tid_cpu + 1) % 5
    f_refw = f_ref5.clone()
    res = {}
    for tag, tids in (("correct", tid_cpu), ("wrong", tid_wrong)):
        u_style = ad(env["u_sg5"][tids].to(device), u_src)
        u_glob = env["fib_n_sg5"][tids]
        d16_r = env["d16_sg5"][tids].to(device)
        v16_r = env["v16_sg5"][tids].to(device)
        from exp.v26.sr_cfg import sr_velocity, integrate_sr
        tau0, wj = 0.45, 2.2
        z = (1.0 - tau0) * eps + tau0 * zs

        def vel_fn(zz, prog):
            tv = torch.full((B,), tau0 + prog * (1.0 - tau0), device=device)
            v, _ = sr_velocity(fm, ad, u_style, None, u_glob, None, d16, d16_r, v16_r,
                               None, None, y16, zz, tv, mode="baseline", w=0.0, wj=wj, device=device)
            return v

        z_gen = integrate_sr(vel_fn, z, tau0, 10, "rk2")
        rgb = decode_s10(dec, z_gen, batch=16).clamp(0.0, 1.0)
        f_gen = dino_of(rgb)
        res[tag] = round(float(torch.nn.functional.cosine_similarity(f_gen, f_ref5, dim=-1).mean()), 5)
    res["swap_delta"] = round(res["wrong"] - res["correct"], 5)
    return res


def coarse_y16(device, env, zs):
    from exp.v20.slf_flow import coarse_y16_lcn
    return coarse_y16_lcn(((zs * env["w_l"].to(device)).sum(1, keepdim=True) + env["b_l"].to(device)).clamp(0.0, 1.0)).float()


def run_l0(pool: str, device: str = "cuda") -> dict:
    """L0 三臂因果对照. 返回汇总 (含 5×5 配对矩阵与 swap 敏感性)."""
    import exp.v25 as _v25
    from exp.v19.fiber_duality import build_or_load_v19_foundation
    from exp.v24.p0_o1_o3_value import build_or_load_train_cnx_v16
    from exp.v20.submode_patchswd import load_convnext_tiny_offline
    from exp.v26.dynamic_style_swd import DynamicSWD  # noqa: F401 (保持依赖一致)
    from src.utils import load_s10_decoder

    _v25.seed_all(L0_SEED)
    bundle = build_or_load_v19_foundation(pool, device=device)
    c21 = torch.load(ROOT / "runs/v21/cache/v21_stiefel_ref_cache.pt",
                     map_location="cpu", weights_only=False)
    dec = load_s10_decoder(device)
    cnx = load_convnext_tiny_offline(device).eval()
    for p in cnx.parameters():
        p.requires_grad_(False)
    v16_tr = build_or_load_train_cnx_v16(bundle, dec, cnx, device=device)
    del cnx
    torch.cuda.empty_cache()

    tr = bundle["v18_cache"]["train"]
    sid_all = tr["sid"].long()
    core_idx = c21["train"]["core_idx_per_style"]
    # A2 需要 core 索引在同设备
    core_idx = [c.to(device) if torch.is_tensor(c) else torch.as_tensor(c, device=device) for c in core_idx]

    arms = {
        "A0_legacy_nbr": dict(sampler="legacy"),
        "A1_strict_cross": dict(sampler="strict", core=False),
        "A2_core_cross": dict(sampler="strict", core=True),
    }
    results = {}
    for arm, spec in arms.items():
        print(f"\n[L0] ===== {arm} train start {utc()} =====", flush=True)
        ckpt, diag = train_l0_arm(bundle, c21, v16_tr, arm, spec, device=device)
        env = load_sr_env(pool, device=device, ckpt_path=ckpt)
        s = _eval_baseline_steps(env, f"{arm}_t45", steps=10, sampler="rk2", device=device)
        c = s["cross600"]
        swap = ref_swap_sensitivity(env, device=device)
        results[arm] = {
            "cross600": {k: round(c[k], 5) for k in ("dino_c", "dino_s", "lpips", "style_affinity", "csd_top1")},
            "pair_matrix": diag["pair_matrix"],
            "ref_swap": swap,
            "train_diag_tail": diag["rows"][-3:],
        }
        save_json(OUT / "gpu4090" / "l0_crossstyle_causal.json", results)
        print(f"[L0] {arm}: {results[arm]['cross600']} swap={swap}", flush=True)
    _write_l0_report(results)
    return results


def train_l0_arm(bundle, c21, v16_tr, arm: str, spec: dict, device: str = "cuda"):
    """L0 训练臂: 复用 train_c2nt_v26 主体但替换参考采样器 + 记录配对矩阵."""
    import copy
    import math
    import time

    import torch.nn as nn

    from exp.v24.p0_o1_o3_value import DUAL_CKPT, SLFUNetV24OneShot
    from exp.v19.fiber_duality import StyleFiberEncoder
    from exp.v20.slf_flow import coarse_y16_lcn
    from exp.v22.ref_adapter import RefAdapter
    from exp.v25.mechanism_c import make_train_feature_fn
    from exp.v26 import autocast
    from exp.v26.sr_cfg import _eval_baseline_steps  # noqa: F401
    from src.utils import decode_s10, load_s10_decoder

    seed = L0_SEED + 1 + list(("A0_legacy_nbr", "A1_strict_cross", "A2_core_cross")).index(arm)
    updates, micro, accum, lambda_sw, lambda_c = 1200, 8, 4, 0.5, 1.0
    out_p = OUT / "oneshot_val" / f"{arm}.pt"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    if out_p.exists():  # 重入保护: 队列重启时跳过已完成训练
        print(f"[L0 {arm}] Reusing {out_p}", flush=True)
        return out_p, {"pair_matrix": [[0] * 5 for _ in range(5)], "rows": []}

    torch.manual_seed(seed)
    import exp.v25 as _v25
    _v25.seed_all(seed)
    tr = bundle["v18_cache"]["train"]
    fnd = bundle["fnd"]
    z_all, d16_all, sid_all = tr["z"], tr["dino_16"], tr["sid"].long()
    y16_all = coarse_y16_lcn(tr["y_64"].float()).half()
    nbr_idx = c21["train"]["nbr_idx"]
    n_total = len(z_all)
    pair_matrix = [[0] * 5 for _ in range(5)]
    pairs_seen: list[tuple[int, int]] = []

    fib_tr = fnd["tr_fib"].float()
    fib_mean = fib_tr.mean(0, keepdim=True).to(device)
    fib_std = fib_tr.std(0, keepdim=True).clamp_min(1e-4).to(device)
    fib_n_all = (fib_tr.to(device) - fib_mean) / fib_std
    fiber_enc = StyleFiberEncoder(in_dim=fib_tr.shape[1], dim=128).to(device).eval()
    fiber_enc.load_state_dict(fnd["fiber_state"])
    with torch.no_grad():
        u_all = fiber_enc.encode(fib_tr.to(device))
    del fiber_enc

    fm = SLFUNetV24OneShot(v_dim=192, use_conf_gate=True, use_global_fib=True).to(device)
    fm.load_state_dict(torch.load(DUAL_CKPT, map_location="cpu", weights_only=False)["ema"], strict=False)
    adapter = RefAdapter(dim=128).to(device)
    feature_fn = make_train_feature_fn("vgg", device)
    dec = load_s10_decoder(device)

    for p in fm.parameters():
        p.requires_grad_(False)
    unfreeze = ("cond_mlp.", "dino_proj.", "mid1.", "mid_attn.", "mid2.", "dec32.", "dec64.", "out_norm.", "out_conv.")
    fm_p, new_p = [], list(adapter.parameters())
    for n, p in fm.named_parameters():
        if n.startswith("ref_attn16."):
            p.requires_grad_(True); new_p.append(p)
        elif n.startswith(unfreeze):
            p.requires_grad_(True); fm_p.append(p)
    opt = torch.optim.AdamW([{"params": new_p, "lr": 1.5e-4}, {"params": fm_p, "lr": 3e-5}], weight_decay=1e-3)
    base_lrs = [1.5e-4, 3e-5]
    ema_fm, ema_ad = copy.deepcopy(fm).eval(), copy.deepcopy(adapter).eval()
    for p in list(ema_fm.parameters()) + list(ema_ad.parameters()):
        p.requires_grad_(False)

    def sample_pair(step: int, ai: int):
        """按臂规格返回 (ic_idx, ir_idx); 同时登记配对矩阵."""
        if spec["sampler"] == "legacy":
            ic, ir = assign_c2nt_refs(micro, n_total, nbr_idx, sid_all,
                                      torch.Generator().manual_seed(seed * 7919 + step * 97 + ai))
        else:
            core = (c21["train"]["core_idx_per_style"] if spec.get("core") else None)
            gen = torch.Generator().manual_seed(seed * 7919 + step * 97 + ai)
            pool = [(sid_all == s).nonzero(as_tuple=True)[0] for s in range(5)]
            ic_l, ir_l = [], []
            for _ in range(micro):
                c = int(torch.randint(0, n_total, (1,), generator=gen))
                sc = int(sid_all[c])
                others = torch.cat([core[s] if core is not None else pool[s]
                                    for s in range(5) if s != sc])
                if len(others) == 0:
                    raise RuntimeError("无跨画派候选 — 禁止回退")
                r = int(others[int(torch.randint(0, len(others), (1,), generator=gen))])
                if int(sid_all[r]) == sc:
                    raise RuntimeError("严格跨画派采样被违反")
                ic_l.append(c); ir_l.append(r)
            ic = torch.tensor(ic_l, dtype=torch.long)
            ir = torch.tensor(ir_l, dtype=torch.long)
        for c_i, r_i in zip(ic.tolist(), ir.tolist()):
            pair_matrix[int(sid_all[c_i])][int(sid_all[r_i])] += 1
        pairs_seen.append((0, 0))
        return ic, ir

    rows = []
    t0 = time.time()
    t_last = time.time()
    for step in range(1, updates + 1):
        scale = 0.2 + 0.4 * (1.0 + math.cos(math.pi * step / updates))
        for pg, blr in zip(opt.param_groups, base_lrs):
            pg["lr"] = blr * scale
        opt.zero_grad(set_to_none=True)
        for ai in range(accum):
            ic, ir = sample_pair(step, ai)
            z1 = z_all[ic].to(device).float()
            B = len(ic)
            d16 = d16_all[ic].to(device).float()
            d16_r = d16_all[ir].to(device).float()
            v16_r = v16_tr[ir].to(device).float()
            y16 = y16_all[ic].to(device).float()
            u_style = adapter(u_all[ir], u_all[ic])
            u_glob = fib_n_all[ir]
            d16_aug = d16 + 0.08 * torch.randn_like(d16)
            d16_aug = d16_aug - d16_aug.mean(dim=(2, 3), keepdim=True)
            y16_aug = (y16 * (0.85 + 0.30 * torch.rand(B, 1, 1, 1, device=device))
                       + 0.04 * torch.randn_like(y16)).clamp(-1.1, 1.1)
            slot = torch.zeros(B, dtype=torch.long, device=device)
            z0 = torch.randn_like(z1)
            tau = torch.rand(B, device=device)
            zt = (1.0 - tau).view(B, 1, 1, 1) * z0 + tau.view(B, 1, 1, 1) * z1
            target_v = z1 - z0
            with autocast(device):
                pred_v = fm(zt, y16_aug, d16_aug, slot, u_style, tau,
                            drop_style=torch.zeros(B, dtype=torch.bool, device=device),
                            drop_content=torch.zeros(B, dtype=torch.bool, device=device),
                            layout_scale=torch.ones(B, device=device),
                            d16_ref=d16_r, v16_ref=v16_r, u_glob=u_glob)
                loss = torch.nn.functional.mse_loss(pred_v.float(), target_v) / accum
            loss.backward()

        ic_sw, ir_sw = sample_pair(step, accum)
        za = z_all[ic_sw].to(device).float()
        zr = z_all[ir_sw].to(device).float()
        S = len(za)
        d16_c = d16_all[ic_sw].to(device).float()
        y16_c = y16_all[ic_sw].to(device).float()
        u_src = u_all[ic_sw]
        tau0_s = 0.35 + 0.25 * torch.rand(S, device=device)
        z0_s = torch.randn_like(za)
        z_init = (1.0 - tau0_s).view(S, 1, 1, 1) * z0_s + tau0_s.view(S, 1, 1, 1) * za
        slot0 = torch.zeros(S, dtype=torch.long, device=device)
        d16_aug = d16_c + 0.08 * torch.randn_like(d16_c)
        d16_aug = d16_aug - d16_aug.mean(dim=(2, 3), keepdim=True)
        y16_aug = (y16_c * (0.85 + 0.30 * torch.rand(S, 1, 1, 1, device=device))
                   + 0.04 * torch.randn_like(y16_c)).clamp(-1.1, 1.1)
        d16_r_sw = d16_all[ir_sw].to(device).float()
        v16_r_sw = v16_tr[ir_sw].to(device).float()
        u_glob_sw = fib_n_all[ir_sw]
        sw_chunk = 2
        for c0 in range(0, S, sw_chunk):
            c1 = min(c0 + sw_chunk, S)
            w_frac = float(c1 - c0) / float(S)
            u_style_c = adapter(u_all[ir_sw[c0:c1]], u_src[c0:c1])
            with autocast(device):
                v_prev = fm(z_init[c0:c1], y16_aug[c0:c1], d16_aug[c0:c1], slot0[c0:c1],
                            u_style_c, tau0_s[c0:c1],
                            drop_style=torch.zeros(c1 - c0, dtype=torch.bool, device=device),
                            drop_content=torch.zeros(c1 - c0, dtype=torch.bool, device=device),
                            layout_scale=torch.ones(c1 - c0, device=device),
                            d16_ref=d16_r_sw[c0:c1], v16_ref=v16_r_sw[c0:c1],
                            u_glob=u_glob_sw[c0:c1]).float()
            z_prev = z_init[c0:c1] + v_prev * (1.0 - tau0_s[c0:c1].view(-1, 1, 1, 1))
            rgb_prev = decode_s10(dec, z_prev.clamp(-5.0, 5.0), batch=c1 - c0,
                                  enable_grad=True, output_cpu=False)
            with torch.no_grad():
                rgb_r = decode_s10(dec, zr[c0:c1], batch=c1 - c0, enable_grad=True, output_cpu=False)
                rgb_s = decode_s10(dec, za[c0:c1], batch=c1 - c0, enable_grad=True, output_cpu=False)
                with autocast(device):
                    f_ref = feature_fn(rgb_r)
                    f_src = feature_fn(rgb_s)
            with autocast(device):
                f_gen = feature_fn(rgb_prev)
            l_style = (1.0 - torch.nn.functional.cosine_similarity(
                f_gen["style"].float(), f_ref["style"].float(), dim=-1)).mean()
            l_cont = (1.0 - torch.nn.functional.cosine_similarity(
                f_gen["content"].float(), f_src["content"].float(), dim=-1)).mean()
            ((lambda_sw * l_style + lambda_c * l_cont) * w_frac).backward()

        all_p = [p for g_ in opt.param_groups for p in g_["params"]]
        nn.utils.clip_grad_norm_(all_p, 1.0)
        opt.step()
        with torch.no_grad():
            for pe, pw in zip(ema_fm.parameters(), fm.parameters()):
                pe.mul_(0.996).add_(pw, alpha=0.004)
            for pe, pw in zip(ema_ad.parameters(), adapter.parameters()):
                pe.mul_(0.996).add_(pw, alpha=0.004)
        if step == 1 or step % 200 == 0 or step == updates:
            now = time.time()
            print(f"[L0 {arm}] step {step} ({(step) / max(now - t0, 1e-6):.3f} ups, "
                  f"{now - t0:.0f}s)", flush=True)
            t_last = now

    torch.save({"fm": ema_fm.state_dict(), "adapter": ema_ad.state_dict(),
                "fib_mean": fib_mean.cpu(), "fib_std": fib_std.cpu(),
                "meta": {"arm": arm, "sampler": spec["sampler"], "updates": updates,
                         "micro": micro, "accum": accum, "lambda_sw": lambda_sw,
                         "lambda_c": lambda_c, "seed": seed, "updated_utc": utc()}}, out_p)
    print(f"[L0 {arm}] saved {out_p} in {(time.time() - t0) / 60:.1f} min", flush=True)
    return out_p, {"pair_matrix": pair_matrix, "rows": rows}


def _write_l0_report(results: dict) -> None:
    p = ROOT / "docs/v26/4090/L0_CROSSSTYLE_CAUSAL.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    L = ["# L0 — 真实跨画派参考采样因果实验 (V26-L)", "",
         f"> 生成时间: {utc()} ｜ 三臂 = A0 旧同画派近邻 / A1 严格跨画派 / A2 跨画派+Core 典型性, 各 1200 upd",
         "", "| 臂 | DINO-C ↑ | DINO-S ↑ | LPIPS ↓ | Top-1 ↑ | swap(wrong−correct) |",
         "|---|---:|---:|---:|---:|---:|"]
    for arm, r in results.items():
        c = r["cross600"]
        L.append(f"| {arm} | {c['dino_c']} | {c['dino_s']} | {c['lpips']} | "
                 f"{c['csd_top1']*100:.2f}% | {r['ref_swap']['swap_delta']} |")
    L += ["", "## 5×5 配对矩阵 (内容画派 × 参考画派)", ""]
    for arm, r in results.items():
        L.append(f"**{arm}**")
        L.append("| src\\ref | ER | Imp | Min | Roc | Uki |")
        L.append("|---|---:|---:|---:|---:|---:|")
        for s_i, row in enumerate(r["pair_matrix"]):
            L.append(f"| {s_i} | " + " | ".join(str(x) for x in row) + " |")
        L.append("")
    L += ["", "## 判读 (docs/1120/08 §L0)", "",
          "- A1/A2 相对 A0 显著改善 Roc/ER → 既往 C2nt 结论混杂了采样问题;",
          "- A1/A2 依然失败 → 支持『改换监督目标与主模型』(L1 主线); 无论结果 L1 继续."]
    p.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[L0] Report → {p}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    sanitize_cuda_env()
    run_l0(args.pool, device=args.device)


if __name__ == "__main__":
    main()
