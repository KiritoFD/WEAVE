r"""V26-L L4: 完整可微 Rollout 上界 (T0/T1/T2) — 训练—推理一致性.

docs/1120/08_v26_long_plan.md §L4. 以 L1/L3 胜出模型为初始化; 4090 按 ~12GB 约束
(共享 GPU), microbatch=1–2 + 梯度累积; 仍不满足则降级两步全反传 (如实记录).

  T0  单步 Euler 预览监督 (1 步反传)          — C2nt 式廉价基线
  T1  4 步 RK2 轨迹终态监督 (仅末步反传)      — 截断梯度
  T2  4 步 RK2 轨迹终态监督 (全部 4 步反传)   — 完整 BPTT 上界

统一: 同初始化/同数据 (严格跨画派参考)/同损失 (SWD 或 VGG, 随 L3 胜出)/同步数
(600 updates × 3). 先跑 100-step 显存探针; 另报告各臂真实 GPU 训练成本.

用法:  python -m exp.v26.gpu4090.l4_rollout_bound --pool ../pseudo_gt_24h/orig_mean_fp16
"""
from __future__ import annotations

import argparse
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from exp.v26 import OUT, ROOT, save_json, utc

L4_SEED = 20266000
L4_UPDATES = 600
L4_STEPS = 4
L4_MICRO = 2
L4_ACCUM = 8  # 有效 batch = 16, 与 L1 micro=8×accum=4 一致


def _init_from_best(device: str) -> tuple[Path | None, str]:
    """L3 胜出 → L1 胜出 → DUAL_CKPT 的初始化链."""
    for pth, note in ((OUT / "gpu4090" / "l3_swd_branch_ablation.json", "L3 best"),
                      (OUT / "gpu4090" / "l1_flow_arbitrary_ref.json", "L1 best")):
        try:
            if pth.exists():
                import json
                d = json.loads(pth.read_text(encoding="utf-8"))
                arms = {k: v for k, v in d.items() if isinstance(v, dict) and "csd_top1" in v}
                if arms:
                    best = max(arms, key=lambda k: arms[k].get("csd_top1", 0))
                    ck = OUT / "oneshot_val" / f"{best}.pt"
                    if ck.exists():
                        return ck, f"{note}={best}"
        except Exception:
            pass
    return None, "DUAL_CKPT (无 L1/L3 结果)"


def _rollout(fm, z_init, y16, d16, u_style, u_glob, tau0: float, steps: int,
             grad_from_step: int, wj: float, device: str, micro: int) -> torch.Tensor:
    """RK2 轨迹积分; grad_from_step 之前 no_grad. 返回终态 latent."""
    B = z_init.shape[0]
    dt = (1.0 - tau0) / float(steps)
    z = z_init
    for i in range(steps):
        with_grad = i >= grad_from_step

        def _v(zz, pr):
            tv = torch.full((B,), tau0 + pr * (1.0 - tau0), device=device)
            out = fm(torch.cat([zz, zz], 0), y16.repeat(2, 1, 1, 1), d16.repeat(2, 1, 1, 1),
                     torch.zeros(2 * B, dtype=torch.long, device=device),
                     torch.cat([u_style, u_style], 0), tv.repeat(2),
                     drop_style=torch.cat([torch.zeros(B, dtype=torch.bool, device=device),
                                           torch.ones(B, dtype=torch.bool, device=device)]),
                     drop_content=torch.zeros(2 * B, dtype=torch.bool, device=device),
                     layout_scale=torch.ones(2 * B, device=device),
                     d16_ref=None, v16_ref=None, u_glob=u_glob.repeat(2, 1))
            v_cond, v_unc = out.float().chunk(2, 0)
            return v_unc + wj * (v_cond - v_unc)

        if with_grad:
            v1 = _v(z, i / steps)
            z = z + dt * _v(z + 0.5 * dt * v1, (i + 0.5) / steps)
        else:
            with torch.no_grad():
                v1 = _v(z, i / steps)
                z = z + dt * _v(z + 0.5 * dt * v1, (i + 0.5) / steps)
    return z


def run_l4(pool: str, device: str = "cuda") -> str:
    import copy
    import exp.v25 as _v25
    from exp.v19.fiber_duality import StyleFiberEncoder, build_or_load_v19_foundation
    from exp.v20.slf_flow import coarse_y16_lcn
    from exp.v22.ref_adapter import RefAdapter
    from exp.v24.p0_o1_o3_value import DUAL_CKPT, SLFUNetV24OneShot
    from exp.v25.mechanism_c import make_train_feature_fn
    from exp.v26 import autocast
    from exp.v26.dynamic_style_swd import DynamicSWD
    from exp.v26.sr_cfg import _eval_baseline_steps, load_sr_env
    from src.utils import decode_s10, load_s10_decoder

    _v25.seed_all(L4_SEED)
    bundle = build_or_load_v19_foundation(pool, device=device)
    c21 = torch.load(ROOT / "runs/v21/cache/v21_stiefel_ref_cache.pt",
                     map_location="cpu", weights_only=False)
    init_ckpt, init_note = _init_from_best(device)
    print(f"[L4] init: {init_note}", flush=True)

    tr = bundle["v18_cache"]["train"]
    fnd = bundle["fnd"]
    z_all, d16_all, sid_all = tr["z"], tr["dino_16"], tr["sid"].long()
    y16_all = coarse_y16_lcn(tr["y_64"].float()).half()
    n_total = len(z_all)
    fib_tr = fnd["tr_fib"].float()
    fib_mean = fib_tr.mean(0, keepdim=True).to(device)
    fib_std = fib_tr.std(0, keepdim=True).clamp_min(1e-4).to(device)
    fiber_enc = StyleFiberEncoder(in_dim=fib_tr.shape[1], dim=128).to(device).eval()
    fiber_enc.load_state_dict(fnd["fiber_state"])
    with torch.no_grad():
        u_all = fiber_enc.encode(fib_tr.to(device))
    del fiber_enc

    fm = SLFUNetV24OneShot(v_dim=192, use_conf_gate=True, use_global_fib=True).to(device)
    ck_src = init_ckpt if init_ckpt is not None else DUAL_CKPT
    ck = torch.load(ck_src, map_location="cpu", weights_only=False)
    fm.load_state_dict(ck["fm"] if "fm" in ck else ck["ema"], strict=False)
    adapter = RefAdapter(dim=128).to(device)
    feature_fn = make_train_feature_fn("vgg", device)
    dec = load_s10_decoder(device)
    swd = DynamicSWD(device=device, n_proj=64)

    for p in fm.parameters():
        p.requires_grad_(False)
    for n, p in fm.named_parameters():
        if n.startswith(("cond_mlp.", "dino_proj.", "mid1.", "mid2.", "dec32.", "dec64.",
                         "out_norm.", "out_conv.")):
            p.requires_grad_(True)

    style_pool = [(sid_all == s).nonzero(as_tuple=True)[0] for s in range(5)]

    def strict_cross(gen):
        c = int(torch.randint(0, n_total, (1,), generator=gen))
        sc = int(sid_all[c])
        others = torch.cat([style_pool[s] for s in range(5) if s != sc])
        r = int(others[int(torch.randint(0, len(others), (1,), generator=gen))])
        return c, r

    arms = {"T0_preview_1step": 1, "T1_rollout_trunc": L4_STEPS - 1, "T2_rollout_full": 0}
    results = {"init_note": init_note, "updated_utc": utc(), "arms": {}}

    for arm, grad_from in arms.items():
        # 每次 arm 重置权重 (同初始化)
        fm.load_state_dict(ck["fm"] if "fm" in ck else ck["ema"], strict=False)
        opt = torch.optim.AdamW([p for p in fm.parameters() if p.requires_grad]
                                + list(adapter.parameters()), lr=3e-5, weight_decay=1e-3)
        t0 = time.time()
        for step in range(1, L4_UPDATES + 1):
            scale = 0.2 + 0.4 * (1.0 + math.cos(math.pi * step / L4_UPDATES))
            opt.zero_grad(set_to_none=True)
            for ai in range(L4_ACCUM):
                gen = torch.Generator().manual_seed(L4_SEED * 7919 + step * 97 + ai)
                ic_l, ir_l = [], []
                for _ in range(L4_MICRO):
                    c, r = strict_cross(gen)
                    ic_l.append(c); ir_l.append(r)
                ic = torch.tensor(ic_l, dtype=torch.long); ir = torch.tensor(ir_l, dtype=torch.long)
                za = z_all[ic].to(device).float()
                zr = z_all[ir].to(device).float()
                B = len(ic)
                d16 = d16_all[ic].to(device).float()
                y16 = y16_all[ic].to(device).float()
                u_src = u_all[ic]
                u_style = adapter(u_all[ir], u_src)
                u_glob = (fnd["tr_fib"].to(device).float()[ir] - fib_mean) / fib_std
                tau0 = 0.45
                eps = torch.randn(B, 4, 64, 64, generator=torch.Generator().manual_seed(L4_SEED + step)).to(device)
                z_init = (1.0 - tau0) * eps + tau0 * za
                if arm == "T0_preview_1step":
                    with autocast(device):
                        v = fm(z_init, y16, d16, torch.zeros(B, dtype=torch.long, device=device),
                               u_style, torch.full((B,), tau0, device=device),
                               drop_style=torch.zeros(B, dtype=torch.bool, device=device),
                               drop_content=torch.zeros(B, dtype=torch.bool, device=device),
                               layout_scale=torch.ones(B, device=device),
                               d16_ref=None, v16_ref=None, u_glob=u_glob).float()
                    z_final = z_init + v * (1.0 - tau0)
                else:
                    grad_from_i = grad_from  # T1: 前面 no_grad; T2: 0
                    z_final = _rollout(fm, z_init, y16, d16, u_style, u_glob, tau0,
                                       L4_STEPS, grad_from_i, 2.2, device, L4_MICRO)
                rgb_final = decode_s10(dec, z_final, batch=B, enable_grad=True, output_cpu=False).clamp(0.0, 1.0)
                with torch.no_grad():
                    rgb_r = decode_s10(dec, zr, batch=B, enable_grad=True, output_cpu=False).clamp(0.0, 1.0)
                    rgb_s = decode_s10(dec, za, batch=B, enable_grad=True, output_cpu=False).clamp(0.0, 1.0)
                with autocast(device):
                    f_gen = feature_fn(rgb_final)
                l_style = (1.0 - F.cosine_similarity(f_gen["style"].float(), feature_fn(rgb_r)["style"].float(), dim=-1)).mean()
                l_cont = (1.0 - F.cosine_similarity(f_gen["content"].float(), feature_fn(rgb_s)["content"].float(), dim=-1)).mean()
                (0.5 * l_style + l_cont).backward()
            all_p = [p for p in fm.parameters() if p.requires_grad] + list(adapter.parameters())
            nn.utils.clip_grad_norm_(all_p, 1.0)
            opt.step()
            if step == 1 or step % 200 == 0 or step == L4_UPDATES:
                print(f"[L4 {arm}] step {step} ({step / max(time.time() - t0, 1e-6):.3f} ups, "
                      f"{time.time() - t0:.0f}s)", flush=True)
        wall = time.time() - t0

        ckpt_p = OUT / "oneshot_val" / f"{arm}.pt"
        torch.save({"fm": fm.state_dict(), "adapter": adapter.state_dict(),
                    "fib_mean": fib_mean.cpu(), "fib_std": fib_std.cpu(),
                    "meta": {"arm": arm, "grad_from_step": grad_from, "wall_s": round(wall, 1),
                             "updated_utc": utc()}}, ckpt_p)
        env = load_sr_env(pool, device=device, ckpt_path=ckpt_p)
        s = _eval_baseline_steps(env, f"{arm}_t45", steps=10, sampler="rk2", device=device)
        c = s["cross600"]
        results["arms"][arm] = {k: round(c[k], 5) for k in ("dino_c", "dino_s", "lpips", "csd_top1")}
        results["arms"][arm]["gpu_train_seconds"] = round(wall, 1)
        save_json(OUT / "gpu4090" / "l4_rollout_bound.json", results)
        print(f"[L4] {arm}: {results['arms'][arm]}", flush=True)

    _write_report(results)
    return "done"


def _write_report(results: dict) -> None:
    from pathlib import Path
    p = ROOT / "docs/v26/4090/L4_ROLLOUT_BOUND.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    L = ["# L4 — 完整可微 Rollout 上界 (V26-L)", "",
         f"> 生成时间: {results['updated_utc']} ｜ init: {results['init_note']} ｜ 600 upd × 3 臂",
         "", "| 臂 | 反传 | DINO-C ↑ | DINO-S ↑ | LPIPS ↓ | Top-1 ↑ | GPU 训练秒 |",
         "|---|---|---:|---:|---:|---:|---:|"]
    gmap = {"T0_preview_1step": "单步", "T1_rollout_trunc": "截断 (末步)", "T2_rollout_full": "全部 4 步"}
    for arm, r in results["arms"].items():
        L.append(f"| {arm} | {gmap.get(arm, arm)} | {r['dino_c']} | {r['dino_s']} | {r['lpips']} | "
                 f"{r['csd_top1']*100:.2f}% | {r['gpu_train_seconds']} |")
    L += ["", "## 判读 (docs/1120/08 §L4)", "",
          "- T2 >> T1: 完整轨迹梯度有价值 → 后续找低显存近似;",
          "- T1 ≈ T2: 截断即可;  T0 更好: 不再增加轨迹训练复杂度."]
    p.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[L4] Report → {p}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    sanitize_cuda_env()
    run_l4(args.pool, device=args.device)


if __name__ == "__main__":
    main()
