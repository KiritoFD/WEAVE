r"""V26-L L3: 动态 SWD × 全局/局部参考分支 2×2 消融 (S0–S3).

docs/1120/08_v26_long_plan.md §L3. K2 已证 SWD 梯度方向与 mean-VGG 近正交
(cos 0.179), 但未证 SWD 训练提高生成质量 — 本阶段完成严格对照.

  S0  VGG均值 × Global-only        (L1 F1 配方基线)
  S1  动态SWD × Global-only        — 分布级风格监督净效应
  S2  VGG均值 × Global+RefAttn     — 局部注入净效应 (置信度门控 = 结构保护)
  S3  动态SWD × Global+RefAttn     — 组合效应

统一: L1 最佳初始化 (F1/F2 按 Top-1 择优; json 缺失则 DUAL_CKPT), 严格跨画派参考,
同数据/同步数 (1200)/同有效 batch. SWD 训练初期**梯度尺度校准** (K2-B: ‖∇SWD‖≈20×
‖∇meanVGG‖ → 缩放至同量级, 保证公平). 胜出臂延长交由预算填充池.

用法:  python -m exp.v26.gpu4090.l3_swd_branch_ablation --pool ../pseudo_gt_24h/orig_mean_fp16
"""
from __future__ import annotations

import argparse

import torch

from exp.v26 import OUT, ROOT, read_json, sanitize_cuda_env, save_json, utc
from exp.v26.sr_cfg import _eval_baseline_steps, load_sr_env

L3_SEED = 20265900
L3_UPDATES = 1200
L3_MICRO = 8
L3_ACCUM = 4


def _l1_best_init(device: str) -> tuple[Path | None, str]:
    """L1 胜出臂 ckpt (F1/F2 按 Top-1 择优; 缺省 None → DUAL_CKPT)."""
    p = OUT / "gpu4090" / "l1_flow_arbitrary_ref.json"
    try:
        d = read_json(p) if p.exists() else {}
        f1, f2 = d.get("F1_global_self", {}), d.get("F2_global_teacher", {})
        if f1 and f2:
            best = "F1_global_self" if f1.get("csd_top1", 0) >= f2.get("csd_top1", 0) else "F2_global_teacher"
            ckpt = OUT / "oneshot_val" / f"{best}.pt"
            if ckpt.exists():
                return ckpt, f"L1 best={best} (Top-1 {max(f1.get('csd_top1',0), f2.get('csd_top1',0))*100:.2f}%)"
    except Exception:
        pass
    return None, "L1 json 缺失 → DUAL_CKPT 初始化"


def calibrate_grad_scale(bundle, v16_tr, device: str, swd) -> dict:
    """梯度尺度校准: 同一 batch 上 ‖∇风格损失‖ vs ‖∇VGG_style‖ → 缩放比 (SWD 与 Bary 各自独立)."""
    import torch.nn.functional as F
    from exp.v24.p0_o1_o3_value import DUAL_CKPT, SLFUNetV24OneShot
    from exp.v19.fiber_duality import StyleFiberEncoder
    from exp.v22.ref_adapter import RefAdapter
    from exp.v25.mechanism_c import make_train_feature_fn
    from src.utils import decode_s10, load_s10_decoder
    import exp.v25 as _v25

    _v25.seed_all(L3_SEED + 7)
    tr = bundle["v18_cache"]["train"]
    z_all, d16_all, sid_all = tr["z"], tr["dino_16"], tr["sid"].long()
    fm = SLFUNetV24OneShot(v_dim=192, use_conf_gate=True, use_global_fib=True).to(device).eval()
    fm.load_state_dict(torch.load(DUAL_CKPT, map_location="cpu", weights_only=False)["ema"], strict=False)
    adapter = RefAdapter(dim=128).to(device).eval()
    feature_fn = make_train_feature_fn("vgg", device)
    dec = load_s10_decoder(device)
    fnd = bundle["fnd"]
    fib_tr = fnd["tr_fib"].float()
    fib_mean = fib_tr.mean(0, keepdim=True).to(device)
    fib_std = fib_tr.std(0, keepdim=True).clamp_min(1e-4).to(device)
    fiber_enc = StyleFiberEncoder(in_dim=fib_tr.shape[1], dim=128).to(device).eval()
    fiber_enc.load_state_dict(fnd["fiber_state"])
    with torch.no_grad():
        u_all = fiber_enc.encode(fib_tr.to(device))
    del fiber_enc

    B = 4
    gen = torch.Generator().manual_seed(L3_SEED)
    style_pool = [(sid_all == s).nonzero(as_tuple=True)[0] for s in range(5)]
    ic_l, ir_l = [], []
    for _ in range(B):
        c = int(torch.randint(0, len(z_all), (1,), generator=gen))
        sc = int(sid_all[c])
        others = torch.cat([style_pool[s] for s in range(5) if s != sc])
        r = int(others[int(torch.randint(0, len(others), (1,), generator=gen))])
        ic_l.append(c); ir_l.append(r)
    ic = torch.tensor(ic_l, dtype=torch.long); ir = torch.tensor(ir_l, dtype=torch.long)
    za = z_all[ic].to(device).float()
    zr = z_all[ir].to(device).float()
    z_init = za.clone()
    slot0 = torch.zeros(B, dtype=torch.long, device=device)
    d16_c = d16_all[ic].to(device).float()
    y16_c = torch.randn(B, 1, 16, 16, device=device)
    tau0_s = torch.full((B,), 0.45, device=device)
    d16_r = d16_all[ir].to(device).float()
    v16_r = v16_tr[ir].to(device).float()
    u_src = u_all[ic]
    u_style_c = adapter(u_all[ir], u_src)
    fib_n_all = (fnd["tr_fib"].to(device).float() - fib_mean) / fib_std
    u_glob_sw = fib_n_all[ir]

    def style_grad(loss_kind):
        z_in = z_init.clone().requires_grad_(True)
        rgb_prev = decode_s10(dec, z_in, batch=B, enable_grad=True, output_cpu=False).clamp(0.0, 1.0)
        with torch.no_grad():
            rgb_r = decode_s10(dec, zr, batch=B, enable_grad=True, output_cpu=False).clamp(0.0, 1.0)
            with torch.autocast("cuda", torch.bfloat16):
                f_ref = feature_fn(rgb_r)
        with torch.autocast("cuda", torch.bfloat16):
            f_gen = feature_fn(rgb_prev)
        if loss_kind == "vgg":
            loss = (1.0 - F.cosine_similarity(f_gen["style"].float(), f_ref["style"].float(), dim=-1)).mean()
        else:
            if loss_kind == "bary":
                from exp.v26.uot_fiber_bundle import uot_gamma_bar
                gbar = uot_gamma_bar(d16_c, d16_r)
                loss = swd.loss_bary(rgb_prev, rgb_r, gbar,
                                     gen=torch.Generator().manual_seed(L3_SEED))["loss"]
            else:
                loss = swd.loss(rgb_prev, rgb_r, gen=torch.Generator().manual_seed(L3_SEED))["loss"]
        loss.backward()
        g = z_in.grad.detach().clone()
        z_in.grad = None
        return float(g.norm())

    n_vgg = style_grad("vgg")
    n_swd = style_grad("swd")
    n_bary = style_grad("bary")
    print(f"[L3 calib] ‖∇VGG‖={n_vgg:.4f} ‖∇SWD‖={n_swd:.4f} ‖∇Bary‖={n_bary:.4f} → "
          f"scale(swd)={n_vgg / max(n_swd, 1e-8):.4f} scale(bary)={n_vgg / max(n_bary, 1e-8):.4f}",
          flush=True)
    del fm, adapter, feature_fn, dec
    torch.cuda.empty_cache()
    return {"swd": n_vgg / max(n_swd, 1e-8), "bary": n_vgg / max(n_bary, 1e-8)}


def run_l3(pool: str, device: str = "cuda") -> str:
    import exp.v25 as _v25
    from exp.v19.fiber_duality import build_or_load_v19_foundation
    from exp.v24.p0_o1_o3_value import build_or_load_train_cnx_v16
    from exp.v20.submode_patchswd import load_convnext_tiny_offline
    from exp.v26.dynamic_style_swd import DynamicSWD
    from exp.v26.gpu4090.l1_flow_arbitrary_ref import build_teacher, train_l1_arm
    from src.utils import load_s10_decoder

    _v25.seed_all(L3_SEED)
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

    tcache = build_teacher(pool, device=device)
    tdata = torch.load(tcache, map_location="cpu", weights_only=False)
    teacher = {"z": tdata["z"], "valid": tdata["valid"]}
    init_ckpt, init_note = _l1_best_init(device)
    print(f"[L3] init: {init_note}", flush=True)

    swd = DynamicSWD(device=device, n_proj=64, dtype=torch.float32)
    scales = calibrate_grad_scale(bundle, v16_tr, device, swd)

    arms = {
        "S0_vgg_global": dict(supervision="teacher", style_loss="vgg", use_refattn=False),
        "S1_swd_global": dict(supervision="teacher", style_loss="swd", use_refattn=False),
        "S4_bary_global": dict(supervision="teacher", style_loss="bary", use_refattn=False),
        "S2_vgg_refattn": dict(supervision="teacher", style_loss="vgg", use_refattn=True),
        "S3_swd_refattn": dict(supervision="teacher", style_loss="swd", use_refattn=True),
    }
    results = {}
    for arm, spec in arms.items():
        print(f"\n[L3] ===== {arm} train start {utc()} =====", flush=True)
        ckpt = train_l1_arm(bundle, c21, v16_tr, teacher, arm,
                            supervision=spec["supervision"],
                            local_attn=spec["use_refattn"],
                            style_loss=spec["style_loss"],
                            swd=swd if spec["style_loss"] in ("swd", "bary") else None,
                            grad_scale=scales[spec["style_loss"]] if spec["style_loss"] in ("swd", "bary") else 1.0,
                            updates=L3_UPDATES, micro=L3_MICRO, accum=L3_ACCUM,
                            device=device, seed=L3_SEED + 1 + list(arms).index(arm),
                            init_ckpt=init_ckpt)
        env = load_sr_env(pool, device=device, ckpt_path=ckpt)
        s = _eval_baseline_steps(env, f"{arm}_t45", steps=10, sampler="rk2", device=device)
        c = s["cross600"]
        results[arm] = {k: round(c[k], 5) for k in ("dino_c", "dino_s", "lpips", "style_affinity", "csd_top1")}
        results["init_note"] = init_note
        results["grad_scale"] = {k: round(v, 5) for k, v in scales.items()}
        save_json(OUT / "gpu4090" / "l3_swd_branch_ablation.json", results)
        print(f"[L3] {arm}: {results[arm]}", flush=True)
    _write_l3_report(results)
    return "done"


def _write_l3_report(results: dict) -> None:
    from pathlib import Path
    p = ROOT / "docs/v26/4090/L3_SWD_BRANCH_ABLATION.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    L = ["# L3 — 动态 SWD × 全局/局部参考分支消融 (V26-L)", "",
         f"> 生成时间: {utc()} ｜ init: {results.get('init_note')} ｜ 梯度尺度校准: {results.get('grad_scale')}",
         "", "| 臂 | 风格损失 | 参考注入 | DINO-C ↑ | DINO-S ↑ | LPIPS ↓ | Top-1 ↑ |",
         "|---|---|---|---:|---:|---:|---:|"]
    spec_map = {"S0_vgg_global": ("VGG均值", "Global"), "S1_swd_global": ("SWD", "Global"),
                "S4_bary_global": ("UOT Bary-SWD", "Global"),
                "S2_vgg_refattn": ("VGG均值", "Global+RefAttn"), "S3_swd_refattn": ("SWD", "Global+RefAttn")}
    for arm, r in results.items():
        if arm not in spec_map:
            continue
        sl, inj = spec_map[arm]
        L.append(f"| {arm} | {sl} | {inj} | {r['dino_c']} | {r['dino_s']} | {r['lpips']} | {r['csd_top1']*100:.2f}% |")
    L += ["", "## 判读 (docs/1120/08 §L3 + 局部性理论 §5.2)", "",
          "- S1>S0: 分布级风格监督有效;  S4>S0 且困难画派起飞: 摊销式局部性 (Bary-SWD) 有效;",
          "- S2>S0: 局部注入有效;  S3>S1: SWD 下局部有额外价值;",
          "- 全部无收益: 回退最简 Global-only. 不能只看 DINO-S — 必须同看 DINO-C/LPIPS/困难风格/灰化."]
    p.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[L3] Report → {p}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    sanitize_cuda_env()
    run_l3(args.pool, device=args.device)


if __name__ == "__main__":
    main()
