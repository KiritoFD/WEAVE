r"""V26-L L1: PRISM-Flow 任意参考化 (F0/F1/F2/F3) — 闭集强风格 Flow 的 One-Shot 转化.

docs/1120/08_v26_long_plan.md §L1. 主线实验:

  F0  闭集上界 (V20_D2 已测 60.50% Top-1, 不重训, 不参与严格 One-Shot 比较)
  F1  全局参考 Fiber + 原训练配方 (自配对 flow-MSE + VGG swap)  — 无跨画派目标
  F2  全局参考 Fiber + 跨画派 Flow 教师监督 (锚桥插值到 V20_D2 闭集教师 latent)
  F3  F2 + 局部 RefAttn (仅 F2 有价值后触发, 本模块预留 run_l1(arms="F3"))

共同约束:
  - 学生 = SLFUNetV24OneShot (V22 Dual 初始化) — K1 结论: 不传 d16_ref/v16_ref 即
    天然 Global-only (ref_attn16 被跳过), F3 才传入 (结构保护待实现);
  - **推理完全免目标画派 ID**: sid 槽位恒零, 条件 = (内容, 参考图全局纤维);
  - **训练参考严格跨画派** (复用 L0 采样器, 禁止静默回退), S1 accum-RNG 修复保持;
  - 教师 = V20_D2 闭集求解器 (真画派条件, 训练期用标签, 论文如实声明), 预计算缓存;
  - 训练冻结主干大部, 训练 cond_mlp/dec/out + ref_attn16 (F3) + adapter.

评测: 全量 test750 (t45) — `_eval_baseline_steps` 同 SR0 协议 (2 分支 CFG, u_style
= adapter(u_ref, u_src), u_glob = fib_n_sg5), 与全部 One-Shot 历史口径可比.

用法:  python -m exp.v26.gpu4090.l1_flow_arbitrary_ref --pool ../pseudo_gt_24h/orig_mean_fp16
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from exp.v26 import OUT, ROOT, save_json, utc
from exp.v26.sr_cfg import _eval_baseline_steps, load_sr_env

L1_SEED = 20265800
L1_UPDATES = 2000
L1_MICRO = 8
L1_ACCUM = 4
TEACHER_CACHE = ROOT / "runs/v26/cache/l1_teacher_v20d2.pt"


# ------------------------------------------------- 教师 (V20_D2 闭集) ----

@torch.no_grad()
def build_teacher(pool: str, device: str = "cuda", batch: int = 25) -> Path:
    """闭集教师: V22 Dual (SLFUNetV24OneShot) 的真画派 core-exemplar 条件模式.

    条件 = 真画派 core 典型样本的全局纤维 (无 adapter, 闭集语义), sid 槽位恒零
    (V24OneShot 结构性置零), 积分协议与 SR0/评测完全一致 (tau0=0.45, 10 步 RK2,
    CFG wj=2.2). 训练期使用真画派标签选择 core exemplar — 论文如实声明.
    """
    if TEACHER_CACHE.exists():
        print(f"[L1] teacher cache hit {TEACHER_CACHE}", flush=True)
        return TEACHER_CACHE
    from exp.v18 import tasks_and_sources  # noqa: F401 (保证数据集就绪检查)
    import exp.v25 as _v25
    from exp.v19.fiber_duality import StyleFiberEncoder, build_or_load_v19_foundation
    from exp.v24.p0_o1_o3_value import SLFUNetV24OneShot
    from exp.v26.sr_cfg import integrate_sr, sr_velocity

    _v25.seed_all(20261185)
    bundle = build_or_load_v19_foundation(pool, device=device)
    c21 = torch.load(ROOT / "runs/v21/cache/v21_stiefel_ref_cache.pt",
                     map_location="cpu", weights_only=False)

    ck = torch.load(ROOT / "runs/v22/attn_flow/prism_dual_branch.pt",
                    map_location="cpu", weights_only=False)
    fm = SLFUNetV24OneShot(v_dim=192, use_conf_gate=True, use_global_fib=True).to(device).eval()
    fm.load_state_dict(ck["ema"], strict=False)  # 双分支 ckpt 布局: {"ema": state_dict, ...}

    fib_tr = bundle["fnd"]["tr_fib"].float()
    fib_mean = fib_tr.mean(0, keepdim=True).to(device)
    fib_std = fib_tr.std(0, keepdim=True).clamp_min(1e-4).to(device)
    fiber_enc = StyleFiberEncoder(in_dim=bundle["fnd"]["tr_fib"].shape[1], dim=128).to(device).eval()
    fiber_enc.load_state_dict(bundle["fnd"]["fiber_state"])
    tr = bundle["v18_cache"]["train"]
    z_all, sid_all = tr["z"], tr["sid"].long()
    u_all = fiber_enc.encode(bundle["fnd"]["tr_fib"].to(device).float())
    fib_n_all = (bundle["fnd"]["tr_fib"].to(device).float() - fib_mean) / fib_std
    del fiber_enc

    w_l = bundle["v18_cache"]["luma_proj"]["w"].float().view(1, 4, 1, 1).to(device)
    b_l = bundle["v18_cache"]["luma_proj"]["b"].float().to(device)
    from exp.v20.slf_flow import coarse_y16_lcn
    y16_all = coarse_y16_lcn(((z_all.float().to(device) * w_l).sum(1, keepdim=True) + b_l).clamp(0.0, 1.0)).half()
    d16_all = tr["dino_16"]
    g = torch.Generator().manual_seed(20261185)
    eps_all = torch.randn(z_all.shape, generator=g)
    eps_all = torch.randn(z_all.shape, generator=g)

    # 每样本的闭集条件: 同画派随机 core 典型样本 (排除自身)
    core_idx = [c.clone() for c in c21["train"]["core_idx_per_style"]]
    u_closed = torch.empty_like(u_all)
    glob_closed = torch.empty_like(fib_n_all)
    for i in range(len(z_all)):
        s = int(sid_all[i])
        cand = core_idx[s].to(u_all.device)
        cand = cand[cand != i]
        if len(cand) == 0:
            cand = (sid_all == s).nonzero(as_tuple=True)[0]
            cand = cand[cand != i]
        j = int(cand[int(torch.randint(0, len(cand), (1,), generator=g))])
        u_closed[i] = u_all[j]
        glob_closed[i] = fib_n_all[j]

    z_t = torch.empty_like(z_all)
    tau0, wj = 0.45, 2.2
    for st in range(0, len(z_all), batch):
        sl = slice(st, min(st + batch, len(z_all)))
        zs = z_all[sl].to(device).float()
        B = len(zs)
        y16 = y16_all[sl].to(device).float()
        d16 = d16_all[sl].to(device).float()
        u_style = u_closed[sl].to(device)
        u_glob = glob_closed[sl].to(device)
        z = (1.0 - tau0) * eps_all[sl].to(device).float() + tau0 * zs
        slot = torch.zeros(B, dtype=torch.long, device=device)
        f0 = torch.zeros(B, dtype=torch.bool, device=device)
        f1 = torch.ones(B, dtype=torch.bool, device=device)
        l1 = torch.ones(B, device=device)

        def vel_fn(zz, prog):
            tv = torch.full((B,), tau0 + prog * (1.0 - tau0), device=device)
            out = fm(torch.cat([zz, zz], 0), y16.repeat(2, 1, 1, 1), d16.repeat(2, 1, 1, 1),
                     slot.repeat(2), torch.cat([u_style, u_style], 0), tv.repeat(2),
                     drop_style=torch.cat([f0, f1], 0),
                     drop_content=torch.zeros(2 * B, dtype=torch.bool, device=device),
                     layout_scale=l1.repeat(2),
                     d16_ref=None, v16_ref=None, u_glob=u_glob.repeat(2, 1)).float()
            v_cond, v_unc = out.chunk(2, 0)
            return v_unc + wj * (v_cond - v_unc)

        z_t[sl] = integrate_sr(vel_fn, z, tau0, 10, "rk2").cpu()
        print(f"[L1 teacher] {st + B}/{len(z_all)}", flush=True)
    TEACHER_CACHE.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"z_teacher": z_t.half(), "recipe": "V22Dual_closedset_core_exemplar",
                "updated_utc": utc()}, TEACHER_CACHE)
    print(f"[L1] teacher cached → {TEACHER_CACHE}", flush=True)
    return TEACHER_CACHE


_CORE_BANK: dict | None = None


def _core_bank_for(device: str) -> dict:
    """core bank = c21 的 core 索引/统计的运行时视图 (integrate_slf_core_batch 兼容)."""
    global _CORE_BANK
    if _CORE_BANK is None:
        import torch.nn.functional as NF

        c21 = torch.load(ROOT / "runs/v21/cache/v21_stiefel_ref_cache.pt",
                         map_location="cpu", weights_only=False)
        tr = torch.load(ROOT / "runs/v18/…", weights_only=False) if False else None
        # core bank 字段 (与 build_or_load_core_knn_bank 的消费字段对齐):
        _CORE_BANK = {
            "core_idx_per_style": c21["train"]["core_idx_per_style"],
            "mid_dsd": c21["mid_dsd"],
        }
    return _CORE_BANK


# ------------------------------------------------- L1 训练臂 ----

def train_l1_arm(bundle, c21, v16_tr, teacher_z: torch.Tensor, arm: str, *,
                 supervision: str, local_attn: bool = False,
                 lambda_sw: float = 0.5, lambda_c: float = 1.0,
                 updates: int = L1_UPDATES, micro: int = L1_MICRO, accum: int = L1_ACCUM,
                 device: str = "cuda", seed: int = L1_SEED) -> Path:
    """L1 训练臂 (label-free 条件 + 严格跨画派参考 + 自锚/教师两种监督)."""
    import copy

    import exp.v25 as _v25
    from exp.v19.fiber_duality import StyleFiberEncoder
    from exp.v20.slf_flow import coarse_y16_lcn
    from exp.v22.ref_adapter import RefAdapter
    from exp.v24.p0_o1_o3_value import DUAL_CKPT, SLFUNetV24OneShot
    from exp.v25.mechanism_c import make_train_feature_fn
    from exp.v26 import autocast
    from src.utils import decode_s10, load_s10_decoder

    assert supervision in ("self", "teacher")
    _v25.seed_all(seed)
    torch.manual_seed(seed)
    tr = bundle["v18_cache"]["train"]
    fnd = bundle["fnd"]
    z_all, d16_all, sid_all = tr["z"], tr["dino_16"], tr["sid"].long()
    y16_all = coarse_y16_lcn(tr["y_64"].float()).half()
    n_total = len(z_all)

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
            p.requires_grad_(local_attn)
            if local_attn:
                new_p.append(p)
        elif n.startswith(unfreeze):
            p.requires_grad_(True); fm_p.append(p)
    if local_attn:
        pass  # ref_attn16 参数已加入 new_p
    opt = torch.optim.AdamW([{"params": new_p, "lr": 1.5e-4}, {"params": fm_p, "lr": 3e-5}], weight_decay=1e-3)
    base_lrs = [1.5e-4, 3e-5]
    ema_fm, ema_ad = copy.deepcopy(fm).eval(), copy.deepcopy(adapter).eval()
    for p in list(ema_fm.parameters()) + list(ema_ad.parameters()):
        p.requires_grad_(False)

    style_pool = [(sid_all == s).nonzero(as_tuple=True)[0] for s in range(5)]

    def strict_cross(gen):
        c = int(torch.randint(0, n_total, (1,), generator=gen))
        sc = int(sid_all[c])
        others = torch.cat([style_pool[s] for s in range(5) if s != sc])
        r = int(others[int(torch.randint(0, len(others), (1,), generator=gen))])
        assert int(sid_all[r]) != sc, "严格跨画派采样被违反"
        return c, r

    sw_chunk = 2
    t0 = time.time()
    for step in range(1, updates + 1):
        scale = 0.2 + 0.4 * (1.0 + math.cos(math.pi * step / max(1, updates)))
        for pg, blr in zip(opt.param_groups, base_lrs):
            pg["lr"] = blr * scale
        opt.zero_grad(set_to_none=True)
        for ai in range(accum):
            gen = torch.Generator().manual_seed(seed * 7919 + step * 97 + ai)
            ic, ir = [], []
            for _ in range(micro):
                c, r = strict_cross(gen)
                ic.append(c); ir.append(r)
            ic = torch.tensor(ic, dtype=torch.long); ir = torch.tensor(ir, dtype=torch.long)
            z_tgt = (z_all[ic].to(device).float() if supervision == "self"
                     else teacher_z[ic].to(device).float())
            za = z_all[ic].to(device).float()
            B = len(ic)
            d16 = d16_all[ic].to(device).float()
            y16 = y16_all[ic].to(device).float()
            u_style = adapter(u_all[ir], u_all[ic])
            u_glob = fib_n_all[ir]
            d16_aug = d16 + 0.08 * torch.randn_like(d16)
            d16_aug = d16_aug - d16_aug.mean(dim=(2, 3), keepdim=True)
            y16_aug = (y16 * (0.85 + 0.30 * torch.rand(B, 1, 1, 1, device=device))
                       + 0.04 * torch.randn_like(y16)).clamp(-1.1, 1.1)
            slot = torch.zeros(B, dtype=torch.long, device=device)
            tau0 = 0.35 + 0.25 * torch.rand(B, device=device)
            z0 = torch.randn_like(za)
            z_init = (1.0 - tau0).view(B, 1, 1, 1) * z0 + tau0.view(B, 1, 1, 1) * za
            tau = tau0 + (1.0 - tau0) * torch.rand(B, device=device)
            frac = ((tau - tau0) / (1.0 - tau0).clamp_min(1e-4)).view(B, 1, 1, 1)
            zt = z_init + frac * (z_tgt - z_init)
            target_v = (z_tgt - z_init) / (1.0 - tau0).clamp_min(1e-4).view(B, 1, 1, 1)
            with autocast(device):
                pred_v = fm(zt, y16_aug, d16_aug, slot, u_style, tau,
                            drop_style=torch.zeros(B, dtype=torch.bool, device=device),
                            drop_content=torch.zeros(B, dtype=torch.bool, device=device),
                            layout_scale=torch.ones(B, device=device),
                            d16_ref=None, v16_ref=None, u_glob=u_glob)
                loss = F.mse_loss(pred_v.float(), target_v) / accum
            loss.backward()

        # swap 正则 (vgg 均值, 参考侧真参考) — 两种监督共用, 保持除教师因子外同配
        gen = torch.Generator().manual_seed(seed * 104729 + step)
        ic_sw, ir_sw = [], []
        for _ in range(micro):
            c, r = strict_cross(gen)
            ic_sw.append(c); ir_sw.append(r)
        ic_sw = torch.tensor(ic_sw, dtype=torch.long); ir_sw = torch.tensor(ir_sw, dtype=torch.long)
        za = z_all[ic_sw].to(device).float()
        zr = z_all[ir_sw].to(device).float()
        S = len(za)
        d16_c = d16_all[ic_sw].to(device).float()
        y16_c = y16_all[ic_sw].to(device).float()
        u_src = u_all[ic_sw]
        tau0_s = 0.35 + 0.25 * torch.rand(S, device=device)
        z0_s = torch.randn_like(za)
        z_init_s = (1.0 - tau0_s).view(S, 1, 1, 1) * z0_s + tau0_s.view(S, 1, 1, 1) * za
        slot0 = torch.zeros(S, dtype=torch.long, device=device)
        d16_aug = d16_c + 0.08 * torch.randn_like(d16_c)
        d16_aug = d16_aug - d16_aug.mean(dim=(2, 3), keepdim=True)
        y16_aug = (y16_c * (0.85 + 0.30 * torch.rand(S, 1, 1, 1, device=device))
                   + 0.04 * torch.randn_like(y16_c)).clamp(-1.1, 1.1)
        d16_r_sw = d16_all[ir_sw].to(device).float()
        v16_r_sw = v16_tr[ir_sw].to(device).float()
        u_glob_sw = fib_n_all[ir_sw]
        for c0 in range(0, S, sw_chunk):
            c1 = min(c0 + sw_chunk, S)
            w_frac = float(c1 - c0) / float(S)
            u_style_c = adapter(u_all[ir_sw[c0:c1]], u_src[c0:c1])
            with autocast(device):
                v_prev = fm(z_init_s[c0:c1], y16_aug[c0:c1], d16_aug[c0:c1], slot0[c0:c1],
                            u_style_c, tau0_s[c0:c1],
                            drop_style=torch.zeros(c1 - c0, dtype=torch.bool, device=device),
                            drop_content=torch.zeros(c1 - c0, dtype=torch.bool, device=device),
                            layout_scale=torch.ones(c1 - c0, device=device),
                            d16_ref=None, v16_ref=None, u_glob=u_glob_sw[c0:c1]).float()
            z_prev = z_init_s[c0:c1] + v_prev * (1.0 - tau0_s[c0:c1].view(-1, 1, 1, 1))
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
            l_style = (1.0 - F.cosine_similarity(f_gen["style"].float(), f_ref["style"].float(), dim=-1)).mean()
            l_cont = (1.0 - F.cosine_similarity(f_gen["content"].float(), f_src["content"].float(), dim=-1)).mean()
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
            print(f"[L1 {arm}] step {step} ({step / max(time.time() - t0, 1e-6):.3f} ups, "
                  f"{time.time() - t0:.0f}s)", flush=True)

    out_p = OUT / "oneshot_val" / f"{arm}.pt"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"fm": ema_fm.state_dict(), "adapter": ema_ad.state_dict(),
                "fib_mean": fib_mean.cpu(), "fib_std": fib_std.cpu(),
                "meta": {"arm": arm, "supervision": supervision, "local_attn": local_attn,
                         "updates": updates, "micro": micro, "accum": accum,
                         "lambda_sw": lambda_sw, "lambda_c": lambda_c, "seed": seed,
                         "ref_sampling": "strict_cross", "label_free_inference": True,
                         "updated_utc": utc()}}, out_p)
    print(f"[L1 {arm}] saved {out_p} in {(time.time() - t0) / 60:.1f} min", flush=True)
    return out_p


def run_l1(pool: str, device: str = "cuda") -> str:
    """L1 主入口: 教师预计算 → F1 (self) → F2 (teacher) → 各自全量 test750 评测."""
    results = {}
    tcache = build_teacher(pool, device=device)
    teacher_z = torch.load(tcache, map_location="cpu", weights_only=False)["z_teacher"].float()
    bundle = _bundle_for(pool, device)
    c21 = torch.load(ROOT / "runs/v21/cache/v21_stiefel_ref_cache.pt",
                     map_location="cpu", weights_only=False)
    import exp.v25 as _v25
    _v25.seed_all(L1_SEED)
    from exp.v24.p0_o1_o3_value import build_or_load_train_cnx_v16
    from exp.v20.submode_patchswd import load_convnext_tiny_offline
    from src.utils import load_s10_decoder
    dec = load_s10_decoder(device)
    cnx = load_convnext_tiny_offline(device).eval()
    for p in cnx.parameters():
        p.requires_grad_(False)
    v16_tr = build_or_load_train_cnx_v16(bundle, dec, cnx, device=device)
    del cnx
    torch.cuda.empty_cache()

    for arm, sup in (("F1_global_self", "self"), ("F2_global_teacher", "teacher")):
        ckpt = train_l1_arm(bundle, c21, v16_tr, teacher_z, arm, supervision=sup, device=device)
        env = load_sr_env(pool, device=device, ckpt_path=ckpt)
        s = _eval_baseline_steps(env, f"{arm}_t45", steps=10, sampler="rk2", device=device)
        c = s["cross600"]
        results[arm] = {k: round(c[k], 5) for k in ("dino_c", "dino_s", "lpips", "style_affinity", "csd_top1")}
        save_json(OUT / "gpu4090" / "l1_flow_arbitrary_ref.json", results)
        print(f"[L1] {arm}: {results[arm]}", flush=True)
    return "done"


def _bundle_for(pool: str, device: str):
    from exp.v19.fiber_duality import build_or_load_v19_foundation
    return build_or_load_v19_foundation(pool, device=device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    sanitize_cuda_env()
    run_l1(args.pool, device=args.device)


if __name__ == "__main__":
    main()
