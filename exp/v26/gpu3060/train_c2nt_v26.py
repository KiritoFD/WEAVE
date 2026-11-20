r"""V26 3060 单轮受控试验: 修复采样后的 C2nt 训练 — S1fix-VGG 基线 vs S1fix-SWD.

docs/1120 调整版 §P1/P2 (C2nt 降级后的"最多 4–6 小时"预算):
  - S1 采样修复: 参考采样种子由 (seed, step, **accum_index**) 唯一确定, 消除同一
    update 内 4 个 accumulation microbatch 复用同一 (ic, ir) 对的 bug;
  - 两臂 (各 1200 updates, 同初始化/同数据流/同 λ):
      S1fix_vgg : 修复采样 + 原 VGG-mean (1−cos) 风格项  → 修复采样净效应
      S1fix_swd : 修复采样 + 动态多尺度 SWD 风格项       → 分布式纹理监督净效应
  - 每臂训完立即 t45 全量 test750 评测 (runs/v26 口径, 2 分支 CFG 同 SR0 协议).

停机条件 (docs 预注册): 两臂均未在相近内容水平改善 DINO-S / 困难画派
(Rococo/ER) → 正式冻结 C2nt 训练配方, 3060 余量转向少步 Flow / Q32 / 解码器.

用法 (3060):
  python -m exp.v26.gpu3060.train_c2nt_v26 --style-loss vgg
  python -m exp.v26.gpu3060.train_c2nt_v26 --style-loss swd
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

from exp.v26 import OUT, autocast, save_json, utc
from exp.v26.sr_cfg import _eval_baseline_steps, load_sr_env
from exp.v25.mechanism_c import assign_c2nt_refs, c2nt_swap_losses, make_train_feature_fn
from exp.v24.p0_o1_o3_value import DUAL_CKPT, SLFUNetV24OneShot
from exp.v19.fiber_duality import StyleFiberEncoder
from exp.v20.slf_flow import coarse_y16_lcn
from exp.v22.ref_adapter import RefAdapter
from exp.v26.dynamic_style_swd import DynamicSWD
from src.utils import decode_s10, load_s10_decoder

TRAIN_SEED = 20265600
UPDATES_DEFAULT = 1200
MICRO_3060 = 12
SW_CHUNK = 4


def train_c2nt_v26(bundle, c21, v16_tr, arm: str, *, style_loss: str = "vgg",
                   lambda_sw: float = 0.5, lambda_c: float = 1.0,
                   updates: int = UPDATES_DEFAULT, micro: int = MICRO_3060,
                   accum: int = 4, device: str = "cuda",
                   swd_n_proj: int = 64, diag_every: int = 200) -> Path:
    """C2nt V26 单轮受控训练 (S1 采样修复固定开启). 返回 ckpt 路径."""
    assert style_loss in ("vgg", "swd")
    out_p = OUT / "oneshot_val" / f"{arm}.pt"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    if out_p.exists():
        print(f"[V26 {arm}] Reusing {out_p}", flush=True)
        return out_p

    seed = TRAIN_SEED
    torch.manual_seed(seed)
    import exp.v25 as _v25
    _v25.seed_all(seed)
    tr = bundle["v18_cache"]["train"]
    fnd = bundle["fnd"]
    z_all, d16_all, sid_all = tr["z"], tr["dino_16"], tr["sid"].long()
    y16_all = coarse_y16_lcn(tr["y_64"].float()).half()
    nbr_idx = c21["train"]["nbr_idx"]

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
    # swd 臂: bf16 VGG (显存减半) + 更小 swap 分块 (3060 12GB 安全水位)
    sw_chunk = 2 if style_loss == "swd" else SW_CHUNK
    swd = DynamicSWD(device=device, n_proj=swd_n_proj,
                     dtype=torch.bfloat16 if style_loss == "swd" else torch.float32) \
        if style_loss == "swd" else None

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

    n_total = len(z_all)
    diag_rows = []
    t0 = time.time()
    for step in range(1, updates + 1):
        scale = 0.2 + 0.4 * (1.0 + math.cos(math.pi * step / max(1, updates)))
        for pg, blr in zip(opt.param_groups, base_lrs):
            pg["lr"] = blr * scale
        opt.zero_grad(set_to_none=True)
        loss_v = 0.0
        for ai in range(accum):  # ---- flow-MSE 自配对 (S1: seed 含 accum_index) ----
            # S1 采样修复: (seed, step, accum_index) 唯一确定随机流 — 原 V25 bug 为
            # 同一 update 的 4 个 accum 复用同一 (ic, ir) 对
            ic, ir = assign_c2nt_refs(micro, n_total, nbr_idx, sid_all,
                                      torch.Generator().manual_seed(seed * 7919 + step * 97 + ai))
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
                loss = F.mse_loss(pred_v.float(), target_v) / accum
            loss.backward()
            loss_v += float(loss.item())

        # ---- 参考正则 pass (每 update 一次) ----
        ic_sw, ir_sw = assign_c2nt_refs(micro, n_total, nbr_idx, sid_all,
                                        torch.Generator().manual_seed(seed * 104729 + step))
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

        l_sw_v = l_c_v = 0.0
        cos_st_parts, cos_ct_parts, en_parts = [], [], []
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
            if style_loss == "swd":
                l_style_c = swd.loss(rgb_prev, rgb_r,
                                     gen=torch.Generator().manual_seed(step))["loss"]
                l_style_v = float(l_style_c.detach()) * w_frac
            else:
                l_style_c = (1.0 - F.cosine_similarity(
                    f_gen["style"].float(), f_ref["style"].float(), dim=-1)).mean()
                l_style_v = float(l_style_c.detach()) * w_frac
            l_cont_c = (1.0 - F.cosine_similarity(
                f_gen["content"].float(), f_src["content"].float(), dim=-1)).mean()
            ((lambda_sw * l_style_c + lambda_c * l_cont_c) * w_frac).backward()
            l_c_v += float(l_cont_c.detach()) * w_frac
            cos_st_parts.append(F.cosine_similarity(f_gen["style"].float().detach(),
                                                    f_ref["style"].float().detach(), dim=-1))
            cos_ct_parts.append(F.cosine_similarity(f_gen["content"].float().detach(),
                                                    f_src["content"].float().detach(), dim=-1))
            en_parts.append((z_prev.detach() - za[c0:c1]).flatten(1).norm(dim=1))
            l_sw_v += l_style_v

        cos_style = torch.cat(cos_st_parts)
        cos_cont = torch.cat(cos_ct_parts)
        edit_norms = torch.cat(en_parts)
        if step == 1 or step % diag_every == 0 or step == updates:
            row = {"step": step, "style_loss": style_loss,
                   "preview_content_cos": round(float(cos_cont.mean()), 5),
                   "preview_style_cos": round(float(cos_style.mean()), 5),
                   "preview_edit_norm": round(float(edit_norms.mean()), 4),
                   "l_style": round(l_sw_v, 5), "l_cont": round(l_c_v, 5)}
            diag_rows.append(row)
            print(f"[V26 {arm}] diag {row}", flush=True)

        all_p = [p for g_ in opt.param_groups for p in g_["params"]]
        nn.utils.clip_grad_norm_(all_p, 1.0)
        opt.step()
        with torch.no_grad():
            for pe, pw in zip(ema_fm.parameters(), fm.parameters()):
                pe.mul_(0.996).add_(pw, alpha=0.004)
            for pe, pw in zip(ema_ad.parameters(), adapter.parameters()):
                pe.mul_(0.996).add_(pw, alpha=0.004)

    torch.save({"fm": ema_fm.state_dict(), "adapter": ema_ad.state_dict(),
                "fib_mean": fib_mean.cpu(), "fib_std": fib_std.cpu(),
                "meta": {"arm": arm, "style_loss": style_loss, "lambda_sw": lambda_sw,
                         "lambda_c": lambda_c, "sampler_fix": True, "micro": micro,
                         "accum": accum, "updates": updates, "seed": seed,
                         "updated_utc": utc()}}, out_p)
    save_json(OUT / "oneshot_val" / f"{arm}_train_diag.json",
              {"arm": arm, "style_loss": style_loss, "rows": diag_rows, "updated_utc": utc()})
    print(f"[V26 {arm}] Saved {out_p} in {(time.time() - t0) / 60:.1f} min", flush=True)
    return out_p


def main():
    from exp.v26 import ROOT, sanitize_cuda_env, seed_all
    ap = argparse.ArgumentParser()
    ap.add_argument("--style-loss", default="vgg", choices=("vgg", "swd"))
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--updates", type=int, default=UPDATES_DEFAULT)
    args = ap.parse_args()
    sanitize_cuda_env()
    seed_all(TRAIN_SEED)

    from exp.v19.fiber_duality import build_or_load_v19_foundation
    from exp.v24.p0_o1_o3_value import build_or_load_train_cnx_v16
    bundle = build_or_load_v19_foundation(args.pool, device=args.device)
    c21 = torch.load(ROOT / "runs/v21/cache/v21_stiefel_ref_cache.pt",
                     map_location="cpu", weights_only=False)
    dec = load_s10_decoder(args.device)
    cnx = load_convnext_tiny_offline(args.device).eval()
    for p in cnx.parameters():
        p.requires_grad_(False)
    v16_tr = build_or_load_train_cnx_v16(bundle, dec, cnx, device=args.device)

    arm = f"V26_S1fix_{args.style_loss}"
    ckpt = train_c2nt_v26(bundle, c21, v16_tr, arm, style_loss=args.style_loss,
                          updates=args.updates, device=args.device)
    env = load_sr_env(args.pool, device=args.device, ckpt_path=ckpt)
    s = _eval_baseline_steps(env, f"{arm}_t45", steps=10, sampler="rk2", device=args.device)
    c = s["cross600"]
    print(f"[V26 {arm}_t45] DINO-C={c['dino_c']:.4f} DINO-S={c['dino_s']:.4f} "
          f"LPIPS={c['lpips']:.4f} Top1={c['csd_top1']*100:.2f}%", flush=True)


if __name__ == "__main__":
    main()
