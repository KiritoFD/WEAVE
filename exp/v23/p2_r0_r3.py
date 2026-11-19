"""V23 P2 §2.1 (3060): One-Shot R0–R3 渐进解冻联合微调（恢复有效全局参考条件基线）。

不再仅训 65K Adapter 并冻结全主干；按照 V23 §2.1 序列做公平对照（同起点 V22 Dual / 1200 updates / micro=24）:
- R0: Adapter-only + 参考图预测连续 5D softmax onehot_hat (主干冻结)
- R1: 同 R0 + 解冻 cond_mlp
- R2: 同 R0 + 解冻 cond_mlp + 16×16 注入层 (dino_proj, mid1, mid_attn, mid2)
- R3: 正式无类别 One-Shot 路线 —— 不依赖五分类探针 (onehot=0), 联合微调 (cond_mlp + 16×16 + 后两层解码块 + Adapter)

评测 (SG-5 真实 1-Shot, 无测试标签, 修复后的双分支 cs_j):
- 每个臂测两个无标签起点: 纯噪声 (tau0=0) 与 源图内容锚点 A0 (tau0=0.40, z_src)。

用法 (3060):
  python -m exp.v23.p2_r0_r3 --pool ../pseudo_gt_24h/orig_mean_fp16
"""
from __future__ import annotations

import argparse
import copy
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import OUT, EVAL_SEED, ROOT, RawWriterV23, autocast, config_hash, export_raw_pngs, file_sha, git_sha, read_json, run_measure_v23, sanitize_cuda_env, save_json, seed_all, utc, validate_raw_meta
from .p0c_oneshot_reval import CustomOnehotSLFUNet, fit_train_fiber_probe
from exp.v22.ref_adapter import RefAdapter
from exp.v18 import tasks_and_sources
from exp.v19.fiber_duality import StyleFiberEncoder, build_or_load_v19_foundation
from exp.v20.slf_flow import coarse_y16_lcn
from src.utils import load_s10_decoder

DUAL_CKPT = ROOT / "runs/v22/attn_flow/prism_dual_branch.pt"

R_MODES = ("R0", "R1", "R2", "R3")


def _configure_trainable(fm: CustomOnehotSLFUNet, adapter: RefAdapter, mode: str) -> list[dict]:
    for p in fm.parameters():
        p.requires_grad_(False)
    for p in adapter.parameters():
        p.requires_grad_(True)

    groups = [{"params": list(adapter.parameters()), "lr": 1.5e-4}]
    if mode == "R0":
        return groups

    fm_prefixes = ["cond_mlp."]
    if mode in ("R2", "R3"):
        fm_prefixes += ["dino_proj.", "mid1.", "mid_attn.", "mid2."]
    if mode == "R3":
        fm_prefixes += ["up32.", "skip32.", "dec32.", "up64.", "skip64.", "dec64.", "out_norm.", "out_conv."]

    fm_params = []
    for n, p in fm.named_parameters():
        if n.startswith(tuple(fm_prefixes)):
            p.requires_grad_(True)
            fm_params.append(p)
    groups.append({"params": fm_params, "lr": 3e-5})
    return groups


def train_r_arm(bundle: dict, c21: dict, probe: nn.Module, mode: str,
                updates: int = 1200, micro: int = 24, accum: int = 2, device: str = "cuda") -> Path:
    out_p = OUT / "oneshot_r" / f"oneshot_{mode}.pt"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    if out_p.exists():
        print(f"[V23 {mode}] Reusing {out_p}", flush=True)
        return out_p

    seed_all(20261400)
    tr = bundle["v18_cache"]["train"]
    z_all = tr["z"]
    sid_all = tr["sid"].long()
    d16_all = tr["dino_16"]
    y16_all = coarse_y16_lcn(tr["y_64"].float()).half()
    nbr_idx = c21["train"]["nbr_idx"]

    fiber_enc = StyleFiberEncoder(in_dim=bundle["fnd"]["tr_fib"].shape[1], dim=128).to(device).eval()
    fiber_enc.load_state_dict(bundle["fnd"]["fiber_state"])
    with torch.no_grad():
        u_all = fiber_enc.encode(bundle["fnd"]["tr_fib"].to(device).float())
    del fiber_enc

    fm = CustomOnehotSLFUNet(base_dim=144, u_dim=128).to(device)
    fm.load_state_dict(torch.load(DUAL_CKPT, map_location="cpu", weights_only=False)["ema"], strict=True)
    adapter = RefAdapter(dim=128).to(device)

    param_groups = _configure_trainable(fm, adapter, mode)
    n_train = sum(p.numel() for g in param_groups for p in g["params"])
    print(f"[V23 {mode}] trainable params={n_train/1e6:.2f}M", flush=True)

    ema_fm = copy.deepcopy(fm).eval()
    ema_ad = copy.deepcopy(adapter).eval()
    for p in list(ema_fm.parameters()) + list(ema_ad.parameters()):
        p.requires_grad_(False)

    opt = torch.optim.AdamW(param_groups, weight_decay=1e-3)
    base_lrs = [g["lr"] for g in param_groups]
    t0 = time.time()

    for step in range(1, updates + 1):
        prog = step / max(1, updates)
        scale = 0.2 + 0.4 * (1.0 + math.cos(math.pi * prog))
        for pg, blr in zip(opt.param_groups, base_lrs):
            pg["lr"] = blr * scale
        opt.zero_grad(set_to_none=True)

        loss_v = 0.0
        for _ in range(accum):
            idx = torch.randint(0, len(z_all), (micro,))
            perm_nbr = torch.rand(micro, 12).topk(1, dim=1).indices
            ref_idx = nbr_idx[idx[:, None], perm_nbr].squeeze(1)

            z1 = z_all[idx].to(device, non_blocking=True).float()
            sid = sid_all[idx].to(device, non_blocking=True)
            d16 = d16_all[idx].to(device, non_blocking=True).float()
            y16 = y16_all[idx].to(device, non_blocking=True).float()

            u_src = u_all[idx]
            u_ref = u_all[ref_idx]
            u_style = adapter(u_ref, u_src)
            if mode == "R3":
                oh_ovr = torch.zeros(micro, 5, device=device)  # 正式无类别：不依赖五分类探针
            else:
                with torch.no_grad():
                    oh_ovr = F.softmax(probe(u_ref), dim=-1)

            d16_aug = d16 + 0.08 * torch.randn_like(d16)
            d16_aug = d16_aug - d16_aug.mean(dim=(2, 3), keepdim=True)
            y16_aug = (y16 * (0.85 + 0.30 * torch.rand(micro, 1, 1, 1, device=device)) + 0.04 * torch.randn_like(y16)).clamp(-1.1, 1.1)

            ds = torch.rand(micro, device=device) < 0.12
            dc = torch.zeros(micro, dtype=torch.bool, device=device)
            lscale = torch.ones(micro, device=device)

            z0 = torch.randn_like(z1)
            tau = torch.rand(micro, device=device)
            zt = (1.0 - tau).view(micro, 1, 1, 1) * z0 + tau.view(micro, 1, 1, 1) * z1
            target_v = z1 - z0

            with autocast(device):
                pred_v = fm(zt, y16_aug, d16_aug, sid, u_style, tau,
                            drop_style=ds, drop_content=dc, layout_scale=lscale, oh_override=oh_ovr)
                loss = F.mse_loss(pred_v.float(), target_v) / accum

            loss.backward()
            loss_v += float(loss.item())

        all_trainable = [p for g in param_groups for p in g["params"]]
        nn.utils.clip_grad_norm_(all_trainable, 1.0)
        opt.step()
        with torch.no_grad():
            for pe, pw in zip(ema_fm.parameters(), fm.parameters()):
                pe.mul_(0.996).add_(pw, alpha=0.004)
            for pe, pw in zip(ema_ad.parameters(), adapter.parameters()):
                pe.mul_(0.996).add_(pw, alpha=0.004)

        if step == 1 or step % 300 == 0 or step == updates:
            print(f"[V23 {mode}] {step:4d}/{updates} loss={loss_v:.4f} alpha={float(adapter.alpha.detach()):.3f} ({time.time()-t0:.0f}s)", flush=True)

    torch.save({"fm": ema_fm.state_dict(), "adapter": ema_ad.state_dict(), "mode": mode, "updates": updates}, out_p)
    print(f"[V23 {mode}] Saved {out_p} in {(time.time()-t0)/60:.1f} min", flush=True)
    return out_p


@torch.no_grad()
def eval_r_arm(bundle: dict, c21: dict, probe: nn.Module, ckpt_p: Path, mode: str, device: str = "cuda") -> list[dict]:
    tasks, z_src_750, _ = tasks_and_sources()
    dec = load_s10_decoder(device)

    ck = torch.load(ckpt_p, map_location="cpu", weights_only=False)
    fm = CustomOnehotSLFUNet(base_dim=144, u_dim=128).to(device).eval()
    fm.load_state_dict(ck["fm"], strict=True)
    ad = RefAdapter(dim=128).to(device).eval()
    ad.load_state_dict(ck["adapter"], strict=True)

    fiber_enc = StyleFiberEncoder(in_dim=bundle["fnd"]["tr_fib"].shape[1], dim=128).to(device).eval()
    fiber_enc.load_state_dict(bundle["fnd"]["fiber_state"])
    u_src_750 = fiber_enc.encode(bundle["fnd"]["test750_fib"].to(device).float()).cpu()
    del fiber_enc

    if "y_64" in bundle["test_struct"]:
        y16_750 = coarse_y16_lcn(bundle["test_struct"]["y_64"].float()).half()
    else:
        w_l = bundle["v18_cache"]["luma_proj"]["w"].float().view(1, 4, 1, 1)
        b_l = bundle["v18_cache"]["luma_proj"]["b"].float()
        y16_750 = coarse_y16_lcn(((z_src_750.float() * w_l).sum(1, keepdim=True) + b_l).clamp(0.0, 1.0)).half()
    eps_750 = torch.randn(z_src_750.shape, generator=torch.Generator().manual_seed(20261185))
    u_sg5 = c21["sg5"]["u"].float()

    res = []
    for tau0, anc_tag in ((0.0, "noise"), (0.40, "srcA0")):
        arm_name = f"{mode}_1shot_{anc_tag}"
        cfg_hash = config_hash({"arm": arm_name, "ckpt": file_sha(ckpt_p)[:12], "tau0": tau0})
        if (OUT / "raw" / arm_name / "meta.json").exists() and validate_raw_meta(read_json(OUT / "raw" / arm_name / "meta.json")):
            s = read_json(OUT / f"measured/{arm_name}/summary_v23.json")
        else:
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

                u_ref_b = u_sg5[tid_t.cpu()].to(device)
                u_style = ad(u_ref_b, u_src_750[sl].to(device))
                oh_ovr = torch.zeros(B, 5, device=device) if mode == "R3" else F.softmax(probe(u_ref_b), dim=-1)

                y16 = y16_750[sl].to(device).float()
                d16 = bundle["test_struct"]["dino_16"][sl].to(device).float()
                eps_b = eps_750[sl].to(device).float()
                z = (1.0 - tau0) * eps_b + tau0 * zs if tau0 > 0 else eps_b

                f0 = torch.zeros(B, dtype=torch.bool, device=device)
                f1 = torch.ones(B, dtype=torch.bool, device=device)
                l1 = torch.ones(B, device=device)
                sid0 = torch.zeros(B, dtype=torch.long, device=device)
                steps = 10
                dt = (1.0 - tau0) / float(steps)

                def vel(zz, prog):
                    tv = torch.full((B,), tau0 + prog * (1.0 - tau0), device=device)
                    zz2 = torch.cat([zz, zz], 0)
                    y2, d2 = y16.repeat(2, 1, 1, 1), d16.repeat(2, 1, 1, 1)
                    s2, u2, t2 = sid0.repeat(2), u_style.repeat(2, 1), tv.repeat(2)
                    ds2, dc2 = torch.cat([f0, f1], 0), torch.cat([f0, f0], 0)
                    l2 = torch.cat([l1, l1], 0)  # 修复后双分支
                    oh2 = oh_ovr.repeat(2, 1)
                    with autocast(device):
                        v_sc, v_0c = fm(zz2, y2, d2, s2, u2, t2, drop_style=ds2, drop_content=dc2,
                                        layout_scale=l2, oh_override=oh2).float().chunk(2, 0)
                    return v_0c + 2.4 * (v_sc - v_0c)

                for i in range(steps):
                    v1 = vel(z, i / float(steps))
                    zm = z + 0.5 * dt * v1
                    v2 = vel(zm, (i + 0.5) / float(steps))
                    z = z + dt * v2
                z_out = torch.where(same.view(B, 1, 1, 1), zs, z)
                wr.write_batch(z_out, grp)
            meta = wr.finalize(cfg_hash=cfg_hash, arm=arm_name, mode=mode, tau0=tau0, wall_sec=round(time.time() - t0, 1))
            assert validate_raw_meta(meta)
            s = run_measure_v23(arm_name, device=device)
            export_raw_pngs(OUT / f"raw/{arm_name}", OUT / f"png_750/{arm_name}", max_workers=8)
        c = s["cross600"]
        res.append({"arm": arm_name, "csd_top1": round(c["csd_top1"], 5), "dino_s": round(c["dino_s"], 5),
                    "dino_c": round(c["dino_c"], 5), "style_affinity": round(c["style_affinity"], 5),
                    "lpips": round(c["lpips"], 5), "pred_eq_src": round(c.get("pred_eq_src", 0.0), 4)})
    del fm, ad
    torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--updates", type=int, default=1200)
    args = ap.parse_args()

    sanitize_cuda_env()
    bundle = build_or_load_v19_foundation(args.pool, device=args.device)
    c21 = torch.load(ROOT / "runs/v21/cache/v21_stiefel_ref_cache.pt", map_location="cpu", weights_only=False)
    probe = fit_train_fiber_probe(bundle, device=args.device)

    ckpts = {}
    for m in R_MODES:
        ckpts[m] = train_r_arm(bundle, c21, probe, m, updates=args.updates, device=args.device)

    results = []
    for m in R_MODES:
        results.extend(eval_r_arm(bundle, c21, probe, ckpts[m], m, device=args.device))

    save_json(OUT / "p2_r0_r3.json", {"updated_utc": utc(), "results": results})
    print("\n=== V23 P2 R0–R3 One-Shot 基线恢复汇总 ===", flush=True)
    for r in results:
        print(f"{r['arm']:18s}: Top1={r['csd_top1']*100:5.2f}% DINO-S={r['dino_s']:.4f} DINO-C={r['dino_c']:.4f} LPIPS={r['lpips']:.4f}", flush=True)


if __name__ == "__main__":
    main()
