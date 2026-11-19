"""V23 P0-C (3060): 修复双分支在 V22 Dual (F2 vs F3) 与 RefAdapter 重评，并隔离 one-hot 归零 OOD 偏移。

实验内容:
1. F2_dual_buggy_g50 vs F3_dual_fixed_orig / F3_dual_fixed_g50 (V22 Dual ckpt, Core-8 条件 + T4 锚点)
2. P2A_1shot_cs_j_fixed (修复后的 RefAdapter 双分支 SG-5 1-shot 重评)
3. P0-C one-hot 清零偏移对照 (全部使用 V22 Dual ckpt + 修复后 cs_j 双分支 + Core-8 token + T4 锚点):
   - P0C_dual_oh_gt   : 正常 onehot(tid)
   - P0C_dual_oh_zero : onehot 全零 (测出纯清零造成的分布偏移)
   - P0C_dual_oh_pred : 训练集拟合的 Fiber→5类 softmax 连续概率 onehot_hat (无测试标签)

用法 (3060):
  python -m exp.v23.p0c_oneshot_reval --pool ../pseudo_gt_24h/orig_mean_fp16
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import OUT, EVAL_SEED, ROOT, RawWriterV23, autocast, config_hash, export_raw_pngs, file_sha, git_sha, read_json, run_measure_v23, sanitize_cuda_env, save_json, seed_all, utc, validate_raw_meta
from exp.v22.core import CFGConfig, D5_SOLVER, GateConfig, PrismCore, _guidance_curves, build_content_condition, effective_gate_components, make_initial_state, rk2_integrate
from exp.v22.t4 import build_or_load_t4_latents_750
from exp.v18 import tasks_and_sources
from exp.v18.ccf_model import sinusoidal_time_emb
from exp.v19.fiber_duality import StyleFiberEncoder, build_or_load_v19_foundation
from exp.v20.run_v20 import get_test750_y16_and_d16
from exp.v20.slf_flow import SLFUNet, coarse_y16_lcn
from exp.v20.submode_patchswd import build_or_load_core_knn_bank

DUAL_CKPT = ROOT / "runs/v22/attn_flow/prism_dual_branch.pt"


class CustomOnehotSLFUNet(SLFUNet):
    """支持传入任意连续/置零 oh_override (B, 5) 的 SLFUNet。"""

    def forward(self, z_tau, y16_lcn, dino_16, style_id, u_vivid, tau, *,
                drop_style=None, drop_content=None, layout_scale=None, oh_override=None):
        B = z_tau.shape[0]
        dev = z_tau.device
        if drop_style is None:
            drop_style = torch.zeros(B, dtype=torch.bool, device=dev)
        if drop_content is None:
            drop_content = torch.zeros(B, dtype=torch.bool, device=dev)
        cmask4 = (~drop_content).float().view(B, 1, 1, 1)
        smask2 = (~drop_style).float().view(B, 1)
        if layout_scale is None:
            lscale = (~drop_content).float()
        else:
            lscale = layout_scale.to(dev).float().view(B).clamp(0.0, 1.0) * (~drop_content).float()
        lscale4 = lscale.view(B, 1, 1, 1)

        d_feat = self.dino_proj(dino_16 * cmask4) * lscale4
        s16 = torch.cat([y16_lcn * lscale4, d_feat], dim=1)

        if oh_override is not None:
            oh = oh_override.to(dev).float() * smask2
        else:
            oh = F.one_hot(style_id.clamp(0, self.N_STYLE - 1), self.N_STYLE).float() * smask2
        uv = u_vivid * smask2
        temb = sinusoidal_time_emb(tau, 64)
        flags = torch.stack([drop_style.float(), 1.0 - lscale], dim=-1)
        cond = self.cond_mlp(torch.cat([temb, oh, uv, flags], dim=-1))

        h64 = self.in_conv(z_tau)
        for blk in self.enc64:
            h64 = blk(h64, cond)
        h32 = self.down32(h64)
        for blk in self.enc32:
            h32 = blk(h32, cond)
        h16 = self.down16(h32)
        h16 = self.mid1(h16, s16, cond)
        h16 = self.mid_attn(h16)
        h16 = self.mid2(h16, s16, cond)
        u32 = self.up32(h16.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1))
        u32 = self.skip32(torch.cat([u32, h32], dim=1))
        for blk in self.dec32:
            u32 = blk(u32, cond)
        u64 = self.up64(u32.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1))
        u64 = self.skip64(torch.cat([u64, h64], dim=1))
        for blk in self.dec64:
            u64 = blk(u64, cond)
        return self.out_conv(F.silu(self.out_norm(u64)))


def fit_train_fiber_probe(bundle: dict, device: str = "cuda") -> nn.Module:
    """仅用训练集 (train split) 拟合 u(128D) → 5 类 softmax 探针，绝对不读测试集标签。"""
    seed_all(20261300)
    fnd = bundle["fnd"]
    enc = StyleFiberEncoder(in_dim=fnd["tr_fib"].shape[1], dim=128).to(device).eval()
    enc.load_state_dict(fnd["fiber_state"])
    with torch.no_grad():
        u_tr = enc.encode(fnd["tr_fib"].to(device).float())
    y_tr = bundle["v18_cache"]["train"]["sid"].long().to(device)
    probe = nn.Linear(128, 5).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=2e-2, weight_decay=1e-4)
    for _ in range(400):
        opt.zero_grad()
        loss = F.cross_entropy(probe(u_tr), y_tr)
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = float((probe(u_tr).argmax(1) == y_tr).float().mean())
    print(f"[V23 P0-C Probe] train fiber->5way linear acc={acc*100:.2f}%", flush=True)
    probe.eval()
    return probe


@torch.no_grad()
def run_dual_arm(model: CustomOnehotSLFUNet, bank: dict, ctx: dict, arm_name: str, gate: GateConfig,
                 oh_mode: str = "gt", probe: nn.Module | None = None,
                 buggy_ls0: bool = False, device: str = "cuda") -> dict:
    cfg_hash = config_hash({"arm": arm_name, "oh_mode": oh_mode, "gate": str(gate), "buggy": buggy_ls0})
    out_dir = OUT / "raw" / arm_name
    if out_dir.joinpath("meta.json").exists():
        old = read_json(out_dir / "meta.json")
        if old.get("cfg_hash") == cfg_hash and validate_raw_meta(old):
            s = read_json(OUT / f"measured/{arm_name}/summary_v23.json")
            c = s["cross600"]
            return {"arm": arm_name, "csd_top1": round(c["csd_top1"], 5), "dino_s": round(c["dino_s"], 5),
                    "dino_c": round(c["dino_c"], 5), "style_affinity": round(c["style_affinity"], 5),
                    "lpips": round(c["lpips"], 5)}

    tasks = ctx["tasks"]
    z_src_750 = ctx["z_src_750"]
    sid_all = torch.tensor([t["source_style_id"] for t in tasks])
    tid_all = torch.tensor([t["target_style_id"] for t in tasks])
    same_all = sid_all == tid_all
    anchor_750 = ctx["z_t4"]
    eps_750 = ctx["eps_750"]
    y16_750, d16_750 = ctx["y16_750"], ctx["d16_750"]
    dec = ctx["dec"]

    core_helper = PrismCore(bundle=ctx["bundle"], core_bank=bank, model=model, device=device)
    solver = dict(D5_SOLVER)
    wr = RawWriterV23(arm_name, dec=dec, device=device)
    t0 = time.time()

    for st in range(0, 750, 25):
        sl = slice(st, min(st + 25, 750))
        grp = tasks[sl]
        B = len(grp)
        tid_b = tid_all[sl].to(device)
        d16_b = d16_750[sl].to(device).float()
        y16_b = y16_750[sl].to(device).float()
        u_style, dsd_ref = core_helper.build_style_condition(d16_b, tid_b, mode="core8")
        g_tau, g_layout, g_guid, _ = effective_gate_components(dsd_ref, gate, float(bank["mid_dsd"]))
        lscale = (solver["lscale_abs"] + (1.0 - solver["lscale_abs"]) * g_layout).clamp(0.15, 1.0)
        content = build_content_condition(y16_b, d16_b, g_layout)
        tau_v = (solver["tau_min"] + (solver["tau_hi"] - solver["tau_min"]) * g_tau).clamp(0.0, 0.78)
        z = make_initial_state(eps_750[sl].to(device).float(), anchor_750[sl].to(device).float(), tau_v)

        if oh_mode == "gt":
            oh_ovr = None
        elif oh_mode == "zero":
            oh_ovr = torch.zeros(B, 5, device=device)
        elif oh_mode == "pred":
            oh_ovr = F.softmax(probe(u_style), dim=-1)
        else:
            raise ValueError(oh_mode)

        f0 = torch.zeros(B, dtype=torch.bool, device=device)
        f1 = torch.ones(B, dtype=torch.bool, device=device)
        ls0 = torch.zeros(B, device=device)

        def vel_fn(zz, prog):
            tv = tau_v + prog * (1.0 - tau_v)
            z2 = torch.cat([zz, zz], 0)
            y2 = content["y_in"].repeat(2, 1, 1, 1)
            d2 = content["d_in"].repeat(2, 1, 1, 1)
            s2 = tid_b.repeat(2)
            u2 = u_style.repeat(2, 1)
            t2 = tv.repeat(2)
            ds2 = torch.cat([f0, f1], 0)
            dc2 = torch.cat([f0, f0], 0)
            l2 = torch.cat([lscale, ls0 if buggy_ls0 else lscale], 0)
            oh2 = oh_ovr.repeat(2, 1) if oh_ovr is not None else None
            with autocast(device):
                v_sc, v_0c = model(z2, y2, d2, s2, u2, t2, drop_style=ds2, drop_content=dc2,
                                   layout_scale=l2, oh_override=oh2).float().chunk(2, 0)
            _, ws_j, _ = _guidance_curves(g_guid, solver, prog)
            return v_0c + ws_j * (v_sc - v_0c)

        z_gen = rk2_integrate(vel_fn, z, tau_v, int(solver["steps"]))
        z_out = torch.where(same_all[sl].view(-1, 1, 1, 1).to(device),
                            z_src_750[sl].to(device).float(), z_gen)
        wr.write_batch(z_out, grp)

    meta = wr.finalize(cfg_hash=cfg_hash, git_sha=git_sha(), ckpt_sha256=file_sha(DUAL_CKPT),
                       tasks_sha256=file_sha(ROOT / "eval_protocol/tasks_750.json"),
                       seed=EVAL_SEED, oh_mode=oh_mode, buggy_ls0=buggy_ls0,
                       wall_sec=round(time.time() - t0, 1))
    assert validate_raw_meta(meta)
    s = run_measure_v23(arm_name, device=device)
    export_raw_pngs(OUT / f"raw/{arm_name}", OUT / f"png_750/{arm_name}", max_workers=8)
    c = s["cross600"]
    return {"arm": arm_name, "csd_top1": round(c["csd_top1"], 5), "dino_s": round(c["dino_s"], 5),
            "dino_c": round(c["dino_c"], 5), "style_affinity": round(c["style_affinity"], 5),
            "lpips": round(c["lpips"], 5), "same_top1": round(s["same150"]["csd_top1"], 5)}


@torch.no_grad()
def rerun_ref_adapter_fixed_csj(bundle: dict, c21: dict, device: str = "cuda") -> dict:
    """用修复后的 exp/v22/ref_adapter.py 双分支路径重测 SG-5 1-Shot (P2A_1shot_cs_j_fixed)。"""
    from exp.v22.ref_adapter import RefAdapter, RefAdapterModel
    from src.utils import decode_s10, load_s10_decoder

    arm_name = "P2A_1shot_cs_j_fixed"
    out_dir = OUT / "raw" / arm_name
    if out_dir.joinpath("meta.json").exists() and validate_raw_meta(read_json(out_dir / "meta.json")):
        s = read_json(OUT / f"measured/{arm_name}/summary_v23.json")
        c = s["cross600"]
        return {"arm": arm_name, "csd_top1": round(c["csd_top1"], 5), "dino_s": round(c["dino_s"], 5),
                "dino_c": round(c["dino_c"], 5), "style_affinity": round(c["style_affinity"], 5), "lpips": round(c["lpips"], 5)}

    tasks, z_src_750, _ = tasks_and_sources()
    dec = load_s10_decoder(device)
    adapter_p = ROOT / "runs/v22/ref_adapter/ref_adapter.pt"
    pack = RefAdapterModel(DUAL_CKPT, bundle["fnd"]["fiber_state"], bundle["fnd"]["tr_fib"].shape[1], device=device)
    ad = RefAdapter(dim=128).to(device)
    ad.load_state_dict(torch.load(adapter_p, map_location="cpu", weights_only=False)["adapter"])
    pack.adapter = ad

    u_src_750 = pack.u_from_fiber(bundle["fnd"]["test750_fib"].float()).cpu()
    if "y_64" in bundle["test_struct"]:
        y16_750 = coarse_y16_lcn(bundle["test_struct"]["y_64"].float()).half()
    else:
        w_l = bundle["v18_cache"]["luma_proj"]["w"].float().view(1, 4, 1, 1)
        b_l = bundle["v18_cache"]["luma_proj"]["b"].float()
        y16_750 = coarse_y16_lcn(((z_src_750.float() * w_l).sum(1, keepdim=True) + b_l).clamp(0.0, 1.0)).half()
    eps_750 = torch.randn(z_src_750.shape, generator=torch.Generator().manual_seed(20261185))
    u_sg5 = c21["sg5"]["u"].float()

    wr = RawWriterV23(arm_name, dec=dec, device=device)
    t0 = time.time()
    for st in range(0, 750, 25):
        sl = slice(st, min(st + 25, 750))
        grp = tasks[sl]
        B = len(grp)
        zs = z_src_750[sl].to(device).float()
        sid_p = torch.tensor([t["source_style_id"] for t in grp], device=device)
        tid_t = torch.tensor([t["target_style_id"] for t in grp], device=device)
        same = sid_p == tid_t
        u_style = pack.adapter(u_sg5[tid_t.cpu()].to(device), u_src_750[sl].to(device))
        y16 = y16_750[sl].to(device).float()
        d16 = bundle["test_struct"]["dino_16"][sl].to(device).float()
        z = eps_750[sl].to(device).float()
        f0 = torch.zeros(B, dtype=torch.bool, device=device)
        f1 = torch.ones(B, dtype=torch.bool, device=device)
        l1 = torch.ones(B, device=device)
        sid0 = torch.zeros(B, dtype=torch.long, device=device)
        steps, dt = 10, 0.1

        def vel(zz, prog):
            zz2 = torch.cat([zz, zz], 0)
            y2, d2 = y16.repeat(2, 1, 1, 1), d16.repeat(2, 1, 1, 1)
            s2, u2 = sid0.repeat(2), u_style.repeat(2, 1)
            t2 = torch.full((B * 2,), prog, device=device)
            ds2, dc2 = torch.cat([f0, f1], 0), torch.cat([f0, f0], 0)
            l2 = torch.cat([l1, l1], 0)  # 修复后双分支 l2=[1,1]
            with autocast(device):
                v_sc, v_0c = pack.fm(zz2, y2, d2, s2, u2, t2, drop_style=ds2, drop_content=dc2, layout_scale=l2).float().chunk(2, 0)
            return v_0c + 2.2 * (v_sc - v_0c)

        for i in range(steps):
            v1 = vel(z, i / steps)
            zm = z + 0.5 * dt * v1
            v2 = vel(zm, (i + 0.5) / steps)
            z = z + dt * v2
        z_out = torch.where(same.view(B, 1, 1, 1), zs, z)
        wr.write_batch(z_out, grp)

    meta = wr.finalize(arm=arm_name, fixed_csj=True, wall_sec=round(time.time() - t0, 1))
    assert validate_raw_meta(meta)
    s = run_measure_v23(arm_name, device=device)
    export_raw_pngs(OUT / f"raw/{arm_name}", OUT / f"png_750/{arm_name}", max_workers=8)
    c = s["cross600"]
    return {"arm": arm_name, "csd_top1": round(c["csd_top1"], 5), "dino_s": round(c["dino_s"], 5),
            "dino_c": round(c["dino_c"], 5), "style_affinity": round(c["style_affinity"], 5), "lpips": round(c["lpips"], 5)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    sanitize_cuda_env()
    bundle = build_or_load_v19_foundation(args.pool, device=args.device)
    bank = build_or_load_core_knn_bank(bundle, k_sub=8, core_per_style=180, device=args.device)
    c21 = torch.load(ROOT / "runs/v21/cache/v21_stiefel_ref_cache.pt", map_location="cpu", weights_only=False)
    tasks, z_src_750, _ = tasks_and_sources()
    z_t4, _ = build_or_load_t4_latents_750(bundle, device=args.device)
    y16_750, d16_750 = get_test750_y16_and_d16(bundle)
    eps_750 = torch.randn(z_src_750.shape, generator=torch.Generator().manual_seed(EVAL_SEED))
    from src.utils import load_s10_decoder
    dec = load_s10_decoder(args.device)
    ctx = {"bundle": bundle, "tasks": tasks, "z_src_750": z_src_750, "y16_750": y16_750,
           "d16_750": d16_750, "eps_750": eps_750, "z_t4": z_t4, "dec": dec}

    model = CustomOnehotSLFUNet(base_dim=144, u_dim=128).to(args.device).eval()
    model.load_state_dict(torch.load(DUAL_CKPT, map_location="cpu", weights_only=False)["ema"], strict=True)
    for p in model.parameters():
        p.requires_grad_(False)
    probe = fit_train_fiber_probe(bundle, device=args.device)

    results = []
    # 1. F2 (buggy Dual) vs F3 (fixed Dual)
    results.append(run_dual_arm(model, bank, ctx, "F2_dual_buggy_g50", GateConfig(mode="fixed", fixed_g=0.5), oh_mode="gt", buggy_ls0=True, device=args.device))
    results.append(run_dual_arm(model, bank, ctx, "F3_dual_fixed_orig", GateConfig(mode="original"), oh_mode="gt", buggy_ls0=False, device=args.device))
    results.append(run_dual_arm(model, bank, ctx, "F3_dual_fixed_g50", GateConfig(mode="fixed", fixed_g=0.5), oh_mode="gt", buggy_ls0=False, device=args.device))

    # 2. P0-C one-hot 清零偏移对照 (全部固定 g=0.5 + 修复后双分支 + Core-8 token)
    results.append(run_dual_arm(model, bank, ctx, "P0C_dual_oh_zero", GateConfig(mode="fixed", fixed_g=0.5), oh_mode="zero", buggy_ls0=False, device=args.device))
    results.append(run_dual_arm(model, bank, ctx, "P0C_dual_oh_pred", GateConfig(mode="fixed", fixed_g=0.5), oh_mode="pred", probe=probe, buggy_ls0=False, device=args.device))

    # 3. 修复后 RefAdapter 1-Shot cs_j 重评
    results.append(rerun_ref_adapter_fixed_csj(bundle, c21, device=args.device))

    save_json(OUT / "p0c_oneshot_reval.json", {"updated_utc": utc(), "results": results})
    print("\n=== V23 P0-C (3060) 汇总 ===", flush=True)
    for r in results:
        print(f"{r['arm']:24s}: Top1={r['csd_top1']*100:5.2f}% DINO-S={r['dino_s']:.4f} DINO-C={r['dino_c']:.4f} LPIPS={r['lpips']:.4f}", flush=True)


if __name__ == "__main__":
    main()
