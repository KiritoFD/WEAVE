"""V23 P1 第三组 (4090): 中尺度空间特征内容约束（C0/C1/C2，独立于 DINO-C 裁判）。

采用 ConvNeXt-Tiny（独立视觉特征器，非 DINOv2）的 32×32 & 16×16 空间对齐特征:
  L_mid = mean_{l in {2,4}, h, w} [1 - cos(Norm(F_l(x_hat)[h,w]), Norm(F_l(x_c)[h,w]))]
经由可微 S10 解码 z1_hat = z_t + (1 - tau) * v_pred，在 tau > 0.35 & ~drop_content 子批上计算。

对照三臂（全部从 Peak1500 初始化，仅解冻中后层，900 updates，同 seed 同数据顺序）:
- C0: 原有 FM + LCN16 (w_lcn=0.10, w_mid=0.00)
- C1: FM + 中尺度特征  (w_lcn=0.00, w_mid=0.15)
- C2: FM + 中尺度 + 弱 LCN (w_lcn=0.05, w_mid=0.15)

用法 (4090):
  python -m exp.v23.p1_midfreq_loss --pool ../pseudo_gt_24h/orig_mean_fp16
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
from exp.v22.core import CFGConfig, D5_SOLVER, GateConfig, PrismCore
from exp.v22.t4 import build_or_load_t4_latents_750
from exp.v18 import tasks_and_sources
from exp.v19.fiber_duality import build_or_load_v19_foundation
from exp.v20.run_v20 import get_test750_y16_and_d16
from exp.v20.slf_flow import SLFUNet, coarse_y16_lcn, project_latent_y16_lcn
from exp.v20.submode_patchswd import build_or_load_core_knn_bank, load_convnext_tiny_offline
from src.utils import decode_s10, load_s10_decoder

PEAK_CKPT = ROOT / "runs/v22/tmatrix/slf_T4Ssnap1500.pt"

C_CONFIGS = {
    "C0": dict(w_lcn=0.10, w_mid=0.00),
    "C1": dict(w_lcn=0.00, w_mid=0.15),
    "C2": dict(w_lcn=0.05, w_mid=0.15),
}


def extract_convnext_mid_spatial(cnx_feats: nn.Module, rgb01: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """输入 rgb01 (B,3,512,512) -> 下采样至 256×256 后提取 stage2 (32×32) 与 stage4 (16×16) 空间特征。"""
    x = F.interpolate(rgb01.float(), size=(256, 256), mode="bilinear", align_corners=False)
    mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    h = (x - mean) / std
    h = cnx_feats[0](h)
    h = cnx_feats[1](h)
    f32 = cnx_feats[2](h)  # (B, 192, 32, 32)
    h = cnx_feats[3](f32)
    f16 = cnx_feats[4](h)  # (B, 384, 16, 16)
    return F.normalize(f32, dim=1), F.normalize(f16, dim=1)


def _mid_spatial_cos_loss(cnx_feats: nn.Module, dec, z1_hat: torch.Tensor, z1_gt: torch.Tensor) -> torch.Tensor:
    rgb_hat = decode_s10(dec, z1_hat, batch=16, enable_grad=True, output_cpu=False)
    with torch.no_grad():
        rgb_gt = decode_s10(dec, z1_gt, batch=16, enable_grad=False, output_cpu=False)
        gt32, gt16 = extract_convnext_mid_spatial(cnx_feats, rgb_gt)
    pr32, pr16 = extract_convnext_mid_spatial(cnx_feats, rgb_hat)
    l32 = (1.0 - (pr32 * gt32).sum(dim=1)).mean()
    l16 = (1.0 - (pr16 * gt16).sum(dim=1)).mean()
    return 0.5 * (l32 + l16)


def train_c_arm(bundle: dict, core_bank: dict, dec, cnx_feats: nn.Module,
                arm: str, updates: int = 900, device: str = "cuda") -> Path:
    cfg = C_CONFIGS[arm]
    out_p = OUT / "midfreq" / f"slf_{arm}.pt"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    if out_p.exists():
        print(f"[V23 {arm}] Reusing {out_p}", flush=True)
        return out_p

    seed_all(20261310)
    tr = bundle["v18_cache"]["train"]
    w_luma = bundle["v18_cache"]["luma_proj"]["w"].to(device).float()
    b_luma = bundle["v18_cache"]["luma_proj"]["b"].to(device).float()
    z_all, sid_all = tr["z"], tr["sid"].long()
    d16_all = tr["dino_16"]
    y16_all = coarse_y16_lcn(tr["y_64"].float()).half()
    u_exact_all = core_bank["u_exact_all"]

    all_cls_idx = [(sid_all == s).nonzero(as_tuple=True)[0] for s in range(5)]
    core_cls_idx = [ci.long() for ci in core_bank["core_idx_per_style"]]

    model = SLFUNet(base_dim=144, u_dim=128).to(device)
    model.load_state_dict(torch.load(PEAK_CKPT, map_location="cpu", weights_only=False)["ema"], strict=True)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)

    # 初期只解冻中后层（16×16 bottleneck + 32/64 decoder）
    unfreeze_prefixes = ("mid1.", "mid_attn.", "mid2.", "up32.", "skip32.", "dec32.",
                         "up64.", "skip64.", "dec64.", "out_norm.", "out_conv.")
    trainable = []
    for n, p in model.named_parameters():
        req = n.startswith(unfreeze_prefixes)
        p.requires_grad_(req)
        if req:
            trainable.append(p)

    opt = torch.optim.AdamW(trainable, lr=5e-5, betas=(0.9, 0.99), weight_decay=1e-4)
    micro, accum = 10, 6  # 有效 batch = 60 (5×2 per micro)，给可微 S10+ConvNeXt 留足显存余量
    t0 = time.time()
    grad_ratio_logged = False

    for step in range(1, updates + 1):
        prog = step / max(1, updates)
        lr = 1e-5 + 0.5 * (5e-5 - 1e-5) * (1.0 + math.cos(math.pi * prog))
        for pg in opt.param_groups:
            pg["lr"] = lr
        opt.zero_grad(set_to_none=True)

        fm_v, st_v, mid_v = 0.0, 0.0, 0.0
        for acc_i in range(accum):
            per_cls = micro // 5
            cur_pools = core_cls_idx if (step % 5 != 0) else all_cls_idx
            idx = torch.cat([ci[torch.randint(0, len(ci), (per_cls,))] for ci in cur_pools], dim=0)
            B = len(idx)

            z1 = z_all[idx].to(device, non_blocking=True).float()
            sid = sid_all[idx].to(device, non_blocking=True)
            d16 = d16_all[idx].to(device, non_blocking=True).float()
            y16 = y16_all[idx].to(device, non_blocking=True).float()
            uv = u_exact_all[idx].to(device, non_blocking=True).float()

            d16_aug = d16 + 0.08 * torch.randn_like(d16)
            d16_aug = d16_aug - d16_aug.mean(dim=(2, 3), keepdim=True)
            y16_aug = (y16 * (0.85 + 0.30 * torch.rand(B, 1, 1, 1, device=device)) + 0.04 * torch.randn_like(y16)).clamp(-1.1, 1.1)

            u_r = torch.rand(B, device=device)
            ds = u_r < 0.12
            dc = (u_r >= 0.12) & (u_r < 0.42)
            both = u_r >= 0.95
            ds, dc = ds | both, dc | both
            lscale = torch.ones(B, device=device)  # T4 配方：layout=1.0 固定

            z0 = torch.randn_like(z1)
            tau = torch.rand(B, device=device)
            zt = (1.0 - tau).view(B, 1, 1, 1) * z0 + tau.view(B, 1, 1, 1) * z1
            target_v = z1 - z0

            with autocast(device):
                pred_v = model(zt, y16_aug, d16_aug, sid, uv, tau,
                               drop_style=ds, drop_content=dc, layout_scale=lscale)
                fm_loss = F.mse_loss(pred_v.float(), target_v)

                z1_hat = (zt + (1.0 - tau).view(B, 1, 1, 1) * pred_v.float()).clamp(-5.0, 5.0)
                st_loss = z1.new_zeros(())
                if cfg["w_lcn"] > 0:
                    y16_hat = project_latent_y16_lcn(z1_hat, w_luma, b_luma)
                    w_tau = ((tau > 0.25) & (~dc)).float()
                    st_loss = (((y16_hat - y16).abs().mean(dim=(1, 2, 3))) * w_tau).sum() / w_tau.sum().clamp_min(1.0)

                mid_loss = z1.new_zeros(())
                if cfg["w_mid"] > 0 and acc_i == 0:
                    valid = (tau > 0.35) & (~dc)
                    if valid.sum() >= 2:
                        sub = valid.nonzero(as_tuple=True)[0][:3]
                        mid_loss = _mid_spatial_cos_loss(cnx_feats, dec, z1_hat[sub], z1[sub])
                        if not grad_ratio_logged:
                            print(f"[V23 {arm}] initial fm={float(fm_loss):.4f}, mid_raw={float(mid_loss):.4f}, mid*w={float(cfg['w_mid']*mid_loss):.4f}", flush=True)
                            grad_ratio_logged = True

                loss = (fm_loss + cfg["w_lcn"] * st_loss + cfg["w_mid"] * mid_loss * (accum if acc_i == 0 else 0.0)) / accum

            loss.backward()
            fm_v += float(fm_loss.item()) / accum
            st_v += float(st_loss.item()) / accum
            mid_v += float(mid_loss.item()) if (cfg["w_mid"] > 0 and acc_i == 0) else 0.0

        nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        with torch.no_grad():
            for pe_w, pw in zip(ema.parameters(), model.parameters()):
                pe_w.mul_(0.996).add_(pw, alpha=0.004)

        if step == 1 or step % 300 == 0 or step == updates:
            print(f"[V23 {arm}] {step:4d}/{updates} fm={fm_v:.4f} lcn={st_v:.4f} mid={mid_v:.4f} ({time.time()-t0:.0f}s)", flush=True)

    torch.save({"ema": ema.state_dict(), "updates": updates, "arm": arm, "cfg": cfg}, out_p)
    print(f"[V23 {arm}] Saved {out_p} in {(time.time()-t0)/60:.1f} min", flush=True)
    return out_p


@torch.no_grad()
def eval_c_arm(bundle: dict, bank: dict, ckpt_path: Path, arm: str, ctx: dict, device: str = "cuda") -> list[dict]:
    core = PrismCore(bundle, bank, ckpt_path, device=device)
    tasks, z_src_750 = ctx["tasks"], ctx["z_src_750"]
    sid_all = torch.tensor([t["source_style_id"] for t in tasks])
    tid_all = torch.tensor([t["target_style_id"] for t in tasks])
    same_all = sid_all == tid_all
    res = []
    for g_val, wc_val, tag in ((0.45, 1.0, "hiS"), (0.50, 1.5, "hiC")):
        name = f"Mid_{arm}_{tag}"
        cfg_hash = config_hash({"name": name, "ckpt": file_sha(ckpt_path)[:12], "g": g_val, "wc": wc_val})
        if (OUT / "raw" / name / "meta.json").exists() and validate_raw_meta(read_json(OUT / "raw" / name / "meta.json")):
            s = read_json(OUT / f"measured/{name}/summary_v23.json")
        else:
            gate = GateConfig(mode="fixed", fixed_g=g_val)
            cfg = CFGConfig(branches="full", rescale_phi=0.48)
            solver = dict(D5_SOLVER, wc_fig=wc_val, wc_abs=wc_val)
            wr = RawWriterV23(name, dec=ctx["dec"], device=device)
            t0 = time.time()
            for st in range(0, 750, 25):
                sl = slice(st, min(st + 25, 750))
                z_gen, _ = core.integrate_flow(
                    ctx["eps_750"][sl], ctx["z_t4"][sl].to(device), ctx["y16_750"][sl], ctx["d16_750"][sl],
                    tid_all[sl].to(device), gate=gate, cfg=cfg, solver=solver,
                )
                z_out = torch.where(same_all[sl].view(-1, 1, 1, 1).to(device), z_src_750[sl].to(device).float(), z_gen)
                wr.write_batch(z_out, tasks[sl])
            meta = wr.finalize(cfg_hash=cfg_hash, git_sha=git_sha(), ckpt_sha256=file_sha(ckpt_path),
                               tasks_sha256=file_sha(ROOT / "eval_protocol/tasks_750.json"),
                               seed=EVAL_SEED, g=g_val, wc=wc_val, wall_sec=round(time.time() - t0, 1))
            assert validate_raw_meta(meta)
            s = run_measure_v23(name, device=device)
            export_raw_pngs(OUT / f"raw/{name}", OUT / f"png_750/{name}", max_workers=8)
        c = s["cross600"]
        res.append({"arm": name, "csd_top1": round(c["csd_top1"], 5), "dino_s": round(c["dino_s"], 5),
                    "dino_c": round(c["dino_c"], 5), "style_affinity": round(c["style_affinity"], 5),
                    "lpips": round(c["lpips"], 5)})
    del core
    torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--updates", type=int, default=900)
    args = ap.parse_args()

    sanitize_cuda_env()
    bundle = build_or_load_v19_foundation(args.pool, device=args.device)
    bank = build_or_load_core_knn_bank(bundle, k_sub=8, core_per_style=180, device=args.device)
    dec = load_s10_decoder(args.device)
    cnx = load_convnext_tiny_offline(args.device).eval()
    for p in cnx.parameters():
        p.requires_grad_(False)

    ckpts = {}
    for arm in ("C0", "C1", "C2"):
        ckpts[arm] = train_c_arm(bundle, bank, dec, cnx, arm, updates=args.updates, device=args.device)
    del cnx
    torch.cuda.empty_cache()

    tasks, z_src_750, _ = tasks_and_sources()
    z_t4, _ = build_or_load_t4_latents_750(bundle, device=args.device)
    y16_750, d16_750 = get_test750_y16_and_d16(bundle)
    eps_750 = torch.randn(z_src_750.shape, generator=torch.Generator().manual_seed(EVAL_SEED))
    ctx = {"tasks": tasks, "z_src_750": z_src_750, "z_t4": z_t4, "y16_750": y16_750,
           "d16_750": d16_750, "eps_750": eps_750, "dec": dec}

    results = []
    for arm in ("C0", "C1", "C2"):
        results.extend(eval_c_arm(bundle, bank, ckpts[arm], arm, ctx, device=args.device))

    save_json(OUT / "p1_midfreq_loss.json", {"updated_utc": utc(), "results": results})
    print("\n=== V23 P1 第三组（中尺度特征约束 C0/C1/C2）汇总 ===", flush=True)
    for r in results:
        print(f"{r['arm']:16s}: Top1={r['csd_top1']*100:5.2f}% DINO-S={r['dino_s']:.4f} DINO-C={r['dino_c']:.4f} LPIPS={r['lpips']:.4f}", flush=True)


if __name__ == "__main__":
    main()
