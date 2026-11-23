r"""ICME 收束第 3b 步 (4090): Q32 主干收敛 (+3000 步 → 总 6000 步) + 6 步工作点 + 解码器复用验证.

背景: Step 3 证明 Q32 在 3000 步同起点单变量对照中反超 dino_16 (+1.86pp Top-1, +0.0105 Margin,
+0.0086 DINO-S, CI 排除零), 正式满足 "若 Q32 持平或反超 → 切换主干权重为 Q32, 理论/Fig.3/代码 100% 一致".

本脚本从 icme_step3_Q32_test.pt (3000 步) 热继续训练 3000 步 (总 6000 步, 匹配 V20 5000+V25 1500),
并一次性完成:
  1. 三工作点 + 错标签对照 750 评测:
     - Q32_s6_csj_d5     (6 步 cs_j, tau=[0.36, 0.72], 24 NFE, 3060 172.8ms cached)
     - Q32_s6_csj_d6h    (6 步 cs_j, tau=[0.48, 0.78], ws=5.2, 高内容 DINO-C 档)
     - Q32_s6_e2full_d5  (6 步前2 full+后4 cs_j, tau=[0.30, 0.68], 32 NFE, 204.9ms cached)
     - Q32_s6_csj_wrong  (错目标反事实对照 → Δ_causal)
  2. 复用 Step 1 D1_full_ft.pt 解码 Q32_s6_csj_d5 输出 latent → 证明 S10 > D1_full_ft 在 Q32 旗舰上成立;
  3. 导出 Fig. 4 Imp→Min 对比图.

用法 (4090):  python -m exp.v27.icme_step3b_q32_flagship --pool ../pseudo_gt_24h/orig_mean_fp16
"""
from __future__ import annotations

import argparse
import copy
import glob
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from exp.v18 import tasks_and_sources
from exp.v20.run_v20 import get_test750_y16_and_d16
from exp.v20.slf_flow import coarse_y16_lcn, project_latent_y16_lcn
from exp.v20.submode_patchswd import build_or_load_core_knn_bank, match_scene_submode
from exp.v21.stiefel_ref_cache import build_or_load_v21_cache
from exp.v22.t4 import build_or_load_t4_latents_750
from exp.v26.gpu4090.l2_decoder_matched import _load_full_vae, wrap_decoder
from exp.v27 import OUT, ROOT, RawWriterV27, run_measure_v27
from exp.v27.icme_step1_decoder_closedset import _cluster_paired_ci
from exp.v27.icme_step2_closedset_accel import D5_CFG, _export_imp_min_strip
from exp.v27.icme_step3_q32_vs_dino16 import make_bottleneck_net
from src.eval.runner import run_measure_subprocess
from src.utils import load_s10_decoder
from src.utils.protocol import EVAL_SEED, pick_wrong_cross_tid
from src.utils.raw_io import RawWriter
from src.utils.runtime import bf16_autocast as autocast, read_json, sanitize_cuda_env, save_json, seed_all, tier_micro_accum, utc


def continue_q32(bundle: dict, c21: dict, bank: dict, updates: int = 3000,
                 device: str = "cuda", seed: int = 20267500) -> str:
    init_p = OUT / "icme_step3_Q32_test.pt"
    out_p = OUT / "icme_q32_flagship.pt"
    if out_p.exists():
        print(f"[Step3b] Reusing {out_p}", flush=True)
        return str(out_p)
    assert init_p.exists(), f"Missing {init_p}"

    seed_all(seed)
    micro, accum = tier_micro_accum(120, default_micro_24gb=60, max_micro_12gb=24, device=device)
    tr = bundle["v18_cache"]["train"]
    w_luma = bundle["v18_cache"]["luma_proj"]["w"].to(device).float()
    b_luma = bundle["v18_cache"]["luma_proj"]["b"].to(device).float()

    z_all = tr["z"]
    sid_all = tr["sid"].long()
    q32_all = c21["train"]["q32"]
    y16_all = coarse_y16_lcn(tr["y_64"].float()).half()
    u_exact = bank["u_exact_all"]

    all_cls = [(sid_all == s).nonzero(as_tuple=True)[0] for s in range(5)]
    core_cls = [ci.long() for ci in bank["core_idx_per_style"]]

    model = make_bottleneck_net(32).to(device)
    model.load_state_dict(torch.load(init_p, map_location="cpu", weights_only=False)["ema"], strict=True)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=8.0e-5, betas=(0.9, 0.99), weight_decay=1e-4)

    t0 = time.time()
    print(f"[Step3b Q32_flagship] continue +3000 steps (micro={micro}×accum={accum})", flush=True)
    for step in range(1, updates + 1):
        prog = step / max(1, updates)
        lr = 1.0e-5 + 0.5 * (8.0e-5 - 1.0e-5) * (1.0 + math.cos(math.pi * prog))
        for pg in opt.param_groups:
            pg["lr"] = lr
        opt.zero_grad(set_to_none=True)
        fm_v, st_v = 0.0, 0.0

        for _ in range(accum):
            per_cls = max(1, micro // 5)
            pools = core_cls if (step % 5 != 0) else all_cls
            idx = torch.cat([ci[torch.randint(0, len(ci), (per_cls,))] for ci in pools], dim=0)
            B = len(idx)
            z1 = z_all[idx].to(device, non_blocking=True).float()
            sid = sid_all[idx].to(device, non_blocking=True)
            f16 = q32_all[idx].to(device, non_blocking=True).float()
            y16 = y16_all[idx].to(device, non_blocking=True).float()
            uv = u_exact[idx].to(device, non_blocking=True).float()

            f16_aug = f16 + 0.06 * torch.randn_like(f16)
            f16_aug = f16_aug - f16_aug.mean(dim=(2, 3), keepdim=True)
            y16_aug = (y16 * (0.85 + 0.30 * torch.rand(B, 1, 1, 1, device=device))
                       + 0.04 * torch.randn_like(y16)).clamp(-1.1, 1.1)

            u_r = torch.rand(B, device=device)
            ds = u_r < 0.14
            dc = (u_r >= 0.14) & (u_r < 0.44)
            both = u_r >= 0.95
            ds, dc = ds | both, dc | both
            lscale = torch.ones(B, device=device)

            z0 = torch.randn_like(z1)
            tau = torch.rand(B, device=device)
            zt = (1.0 - tau).view(B, 1, 1, 1) * z0 + tau.view(B, 1, 1, 1) * z1
            target_v = z1 - z0

            with autocast(device):
                pred_v = model(zt, y16_aug, f16_aug, sid, uv, tau,
                               drop_style=ds, drop_content=dc, layout_scale=lscale)
                fm_loss = F.mse_loss(pred_v.float(), target_v)
                z1_hat = (zt + (1.0 - tau).view(B, 1, 1, 1) * pred_v.float()).clamp(-5.0, 5.0)
                y16_hat = project_latent_y16_lcn(z1_hat, w_luma, b_luma)
                w_tau = (tau > 0.25).float()
                st_loss = ((y16_hat - y16).abs().mean(dim=(1, 2, 3)) * w_tau).sum() / w_tau.sum().clamp_min(1.0)
                loss = (fm_loss + 0.10 * st_loss) / accum
            loss.backward()
            fm_v += float(fm_loss.item()) / accum
            st_v += float(st_loss.item()) / accum

        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for pe, pw in zip(ema.parameters(), model.parameters()):
                pe.mul_(0.996).add_(pw, alpha=0.004)
        if step == 1 or step % 500 == 0 or step == updates:
            print(f"[Step3b Q32] {step:4d}/{updates} fm={fm_v:.4f} st={st_v:.4f} ({time.time()-t0:.0f}s)", flush=True)

    torch.save({"ema": ema.state_dict(), "updates": 6000, "use_q32": True}, out_p)
    print(f"[Step3b Q32] Saved {out_p} in {(time.time()-t0)/60:.1f} min", flush=True)
    return str(out_p)


@torch.no_grad()
def _integrate_q32_seg(model, z_t4, eps, y16, d16_match, q32_in, tid, bank, *,
                       steps: int, full_steps: set[int], cfg_kw: dict) -> torch.Tensor:
    dev = z_t4.device
    B = len(z_t4)
    _, u_core = match_scene_submode(d16_match, tid, bank)
    sc_src = F.normalize(F.avg_pool2d(d16_match.float(), 4).flatten(1), dim=-1)
    best_c = torch.einsum("bd,bkd->bk", sc_src, bank["sub_scene"].to(dev).float()[tid]).argmax(dim=1)
    ex_dsd = bank["sub_dsd"].to(dev).float()[tid, best_c]
    g_fig = torch.sigmoid((ex_dsd - float(bank["mid_dsd"])) * 550.0)
    gf4 = g_fig.view(B, 1, 1, 1)
    lscale = (cfg_kw["lscale_abs"] + (1.0 - cfg_kw["lscale_abs"]) * g_fig).clamp(0.15, 1.0)
    c8_w = (1.0 - g_fig).clamp(0.0, 0.85).view(B, 1, 1, 1)
    f8 = F.avg_pool2d(q32_in, 2).repeat_interleave(2, -2).repeat_interleave(2, -1)
    f8 = f8 - f8.mean(dim=(2, 3), keepdim=True)
    y8 = F.avg_pool2d(y16, 2).repeat_interleave(2, -2).repeat_interleave(2, -1)
    f_in = (1.0 - c8_w) * q32_in + c8_w * f8
    y_in = (1.0 - c8_w) * y16 + c8_w * y8

    tau_v = (cfg_kw["tau_min"] + (cfg_kw["tau_hi"] - cfg_kw["tau_min"]) * g_fig).clamp(0.0, 0.78)
    z = (1.0 - tau_v.view(B, 1, 1, 1)) * eps + tau_v.view(B, 1, 1, 1) * z_t4
    f0 = torch.zeros(B, dtype=torch.bool, device=dev)
    f1 = torch.ones(B, dtype=torch.bool, device=dev)
    ls0 = torch.zeros(B, device=dev)
    rescale_phi = float(cfg_kw.get("rescale_phi", 0.48))

    def _vel(zz, prog, use_full):
        tv = tau_v + prog * (1.0 - tau_v)
        gl = min(1.0, max(0.0, (prog - 0.36) / 0.32))
        ws_j = gf4 * (cfg_kw["ws_fig"] - 1.1 * gl) + (1.0 - gf4) * 1.4
        if use_full:
            with autocast(str(dev)):
                v_sc, v_s0, v_0c, v_00 = model(
                    zz.repeat(4, 1, 1, 1), y_in.repeat(4, 1, 1, 1), f_in.repeat(4, 1, 1, 1),
                    tid.repeat(4), u_core.repeat(4, 1), tv.repeat(4),
                    drop_style=torch.cat([f0, f0, f1, f1], 0),
                    drop_content=torch.cat([f0, f1, f0, f1], 0),
                    layout_scale=torch.cat([lscale, ls0, lscale, ls0], 0),
                ).float().chunk(4, 0)
            wc = gf4 * (cfg_kw["wc_fig"] - 0.45 * gl) + (1.0 - gf4) * cfg_kw["wc_abs"]
            ws_p = gf4 * (gl * cfg_kw["wsp_fig_late"]) + (1.0 - gf4) * cfg_kw["wsp_abs"]
            vr = v_00 + wc * (v_0c - v_00) + ws_j * (v_sc - v_0c) + ws_p * (v_s0 - v_00)
            if rescale_phi > 0.0:
                s_ref = (gf4 * v_sc + (1.0 - gf4) * v_s0).flatten(1).std(1).view(B, 1, 1, 1).clamp_min(1e-4)
                s_raw = vr.flatten(1).std(1).view(B, 1, 1, 1).clamp_min(1e-4)
                vr = rescale_phi * vr * (s_ref / s_raw) + (1.0 - rescale_phi) * vr
            return vr
        with autocast(str(dev)):
            v_sc, v_0c = model(
                zz.repeat(2, 1, 1, 1), y_in.repeat(2, 1, 1, 1), f_in.repeat(2, 1, 1, 1),
                tid.repeat(2), u_core.repeat(2, 1), tv.repeat(2),
                drop_style=torch.cat([f0, f1], 0), drop_content=torch.cat([f0, f0], 0),
                layout_scale=lscale.repeat(2),
            ).float().chunk(2, 0)
        return v_0c + ws_j * (v_sc - v_0c)

    dt = (1.0 - tau_v).view(B, 1, 1, 1) / float(steps)
    for i in range(steps):
        uf = i in full_steps
        v1 = _vel(z, i / float(steps), uf)
        zm = z + 0.5 * dt * v1
        v2 = _vel(zm, (i + 0.5) / float(steps), uf)
        z = z + dt * v2
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    sanitize_cuda_env()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = args.device

    pack = build_or_load_v21_cache(args.pool, device=dev)
    bundle, c21 = pack["bundle"], pack["c21"]
    bank = build_or_load_core_knn_bank(bundle, k_sub=8, core_per_style=180, device=dev)
    ckpt_p = continue_q32(bundle, c21, bank, updates=3000, device=dev)

    tasks, z_src_750, _ = tasks_and_sources()
    z_t4, _ = build_or_load_t4_latents_750(bundle, device=dev)
    y16_750, d16_750 = get_test750_y16_and_d16(bundle)
    q32_750 = c21["test750"]["q32"]
    eps_750 = torch.randn(z_src_750.shape, generator=torch.Generator().manual_seed(EVAL_SEED))
    dec = load_s10_decoder(dev)

    sid_all = torch.tensor([t["source_style_id"] for t in tasks])
    tid_all = torch.tensor([t["target_style_id"] for t in tasks])
    same_all = sid_all == tid_all

    model = make_bottleneck_net(32).to(dev).eval()
    model.load_state_dict(torch.load(ckpt_p, map_location="cpu", weights_only=False)["ema"], strict=True)
    for p in model.parameters():
        p.requires_grad_(False)

    specs = [
        ("ICME_Q32_s6_csj_d5", 6, set(), D5_CFG, False),
        ("ICME_Q32_s6_csj_d6h", 6, set(), dict(D5_CFG, tau_min=0.48, tau_hi=0.78, ws_fig=5.2), False),
        ("ICME_Q32_s6_e2full_d5", 6, {0, 1}, dict(D5_CFG, tau_min=0.30, tau_hi=0.68, ws_fig=5.5,
                                                  wc_fig=1.65, wc_abs=0.70, wsp_fig_late=3.6, wsp_abs=5.2), False),
        ("ICME_Q32_s6_csj_wrong", 6, set(), D5_CFG, True),
    ]
    out = {"task": "ICME Step3b: Q32 Flagship (6000 steps) operating points + D1_full_ft decoder check",
           "updated_utc": utc(), "arms": {}, "decoder_check": {}}

    with torch.no_grad():
        for arm_name, steps, fset, kw, is_wrong in specs:
            m_dir = OUT / "measured" / arm_name
            if (m_dir / "summary_v27.json").exists():
                s = read_json(m_dir / "summary_v27.json")
            else:
                wr = RawWriterV27(arm_name, dec=dec, device=dev)
                for st in range(0, 750, 25):
                    sl = slice(st, min(st + 25, 750))
                    tid_u = pick_wrong_cross_tid(sid_all[sl], tid_all[sl]) if is_wrong else tid_all[sl]
                    zg = _integrate_q32_seg(
                        model, z_t4[sl].to(dev).float(), eps_750[sl].to(dev).float(),
                        y16_750[sl].to(dev).float(), d16_750[sl].to(dev).float(),
                        q32_750[sl].to(dev).float(), tid_u.to(dev), bank,
                        steps=steps, full_steps=fset, cfg_kw=kw,
                    )
                    wr.write_batch(torch.where(same_all[sl].view(-1, 1, 1, 1).to(dev),
                                               z_src_750[sl].to(dev).float(), zg), tasks[sl])
                wr.finalize(arm=arm_name, steps=steps, use_q32=True, is_wrong=is_wrong)
                s = run_measure_v27(arm_name, device=dev)
            c = s["cross600"]
            rows = read_json(m_dir / "eval.json")["rows"]
            imp_min = [r for r in rows if r["src_style"] == "Impressionism" and r["tgt_style"] == "Minimalism"]
            out["arms"][arm_name] = {
                "csd_top1": round(c["csd_top1"], 4), "dino_s": round(c["dino_s"], 4),
                "dino_c": round(c["dino_c"], 4), "style_affinity": round(c["style_affinity"], 4),
                "clip_s": round(c["clip_s"], 4), "lpips": round(c["lpips"], 4),
                "same150_top1": round(s["same150"]["csd_top1"], 4),
                "imp_to_min_top1": round(sum(int(r["target_top1"]) for r in imp_min) / max(len(imp_min), 1), 4),
                "per_target": {k: round(v["csd_top1"], 4) for k, v in s.get("per_target_cross120", {}).items()},
            }
            print(f"[Step3b] {arm_name:24s} | {out['arms'][arm_name]}", flush=True)

    # 2. 用 D1_full_ft.pt 解码 ICME_Q32_s6_csj_d5 latent → 闭环解码器因果在 Q32 旗舰上成立
    d1_p = OUT / "icme_dec" / "D1_full_ft.pt"
    if d1_p.exists():
        full = _load_full_vae(dev)
        full.load_state_dict(torch.load(d1_p, map_location=dev, weights_only=False))
        fn_d1 = wrap_decoder("D1", full, dev, kind="full")
        arm_d1 = "ICME_Q32_s6_D1_full_ft"
        m_dir_d1 = OUT / "measured" / arm_d1
        if not (m_dir_d1 / "eval.json").exists():
            wr = RawWriter(arm_d1, raw_root=OUT / "raw", kind="v27")
            for bf in sorted(glob.glob(str(OUT / "raw/ICME_Q32_s6_csj_d5/batch_*.pt"))):
                d = torch.load(bf, map_location="cpu", weights_only=False)
                rgb = fn_d1(d["z_final"].float().to(dev))
                wr.append(d["ids"], rgb.cpu(), z_final=d["z_final"].cpu())
            wr.finish()
            s_d1 = run_measure_subprocess(OUT / "raw" / arm_d1, m_dir_d1, arm_name=arm_d1, device=dev)
        else:
            s_d1 = run_measure_subprocess(OUT / "raw" / arm_d1, m_dir_d1, arm_name=arm_d1, device=dev)
        c1 = s_d1["cross600"]
        out["decoder_check"] = {
            "S10_on_Q32": out["arms"]["ICME_Q32_s6_csj_d5"],
            "D1_full_ft_on_Q32": {"csd_top1": round(c1["csd_top1"], 4), "dino_s": round(c1["dino_s"], 4),
                                  "dino_c": round(c1["dino_c"], 4), "lpips": round(c1["lpips"], 4),
                                  "same150_top1": round(s_d1["same150"]["csd_top1"], 4)},
            "S10_minus_D1_ci": _cluster_paired_ci(OUT / "measured" / "ICME_Q32_s6_csj_d5", m_dir_d1),
        }
        print(f"[Step3b Dec] {out['decoder_check']}", flush=True)

    _export_imp_min_strip([s[0] for s in specs[:3]], tasks, OUT / "icme_step3b_q32_imp_min.png")
    save_json(OUT / "icme_step3b_q32_flagship.json", out)
    print(f"[Step3b] DONE → {OUT / 'icme_step3b_q32_flagship.json'}", flush=True)


if __name__ == "__main__":
    main()
