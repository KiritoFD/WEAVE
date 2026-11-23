r"""ICME 收束第 2 步: 闭集主力 Flow 定案 + 零训练少步 (4/6/8步) × 分段 CFG 加速.

目的 (用户 2026-11-23 收束清单 §三.2):
  1. 统一对比三个闭集候选主力 (V20_D2 slf_core_v2.pt / V23_HiC_JPk / V25_Tr_T1);
  2. 在 V20_D2 上零训练扫描 4/6/8 步 × 分段 CFG (前 k 步 full + 后 s-k 步 cs_j vs
     后 k 步 full vs 全 cs_j vs 全 full), 兼顾 3060 延迟压降 (~260-320ms) 与
     Impressionism→Minimalism 路线准确率 (paper.tex 已证 full 对该路线关键);
  3. 单独导出 Fig. 4 Row 2 (Imp→Min: armand-guillaumin_valley-in-pontgibaud-1890)
     及另外 3 个 Imp→Min 难例的跨臂对比图, 供视觉审查替换.

用法 (4090):  python -m exp.v27.icme_step2_closedset_accel --pool ../pseudo_gt_24h/orig_mean_fp16
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from exp.v18 import tasks_and_sources
from exp.v19.fiber_duality import build_or_load_v19_foundation
from exp.v20.run_v20 import get_test750_y16_and_d16
from exp.v20.slf_flow import SLFUNet
from exp.v20.submode_patchswd import build_or_load_core_knn_bank, match_scene_submode
from exp.v22.t4 import build_or_load_t4_latents_750
from exp.v27 import OUT, ROOT, RawWriterV27, run_measure_v27
from src.utils import decode_s10, load_s10_decoder
from src.utils.protocol import EVAL_SEED
from src.utils.runtime import bf16_autocast as autocast, read_json, sanitize_cuda_env, save_json, utc

HOST_CKPTS = {
    "V20_D2": ROOT / "runs/v20/slf_flow/slf_core_v2.pt",
    "V23_HiC_JPk": ROOT / "runs/v23/joint_c1ab/slf_Joint_Pk_C1AB.pt",
    "V25_Tr_T1": ROOT / "runs/v25/r2_tmatrix/slf_Tr_T1.pt",
}

D2_CFG = dict(tau_min=0.0, tau_hi=0.44, ws_fig=3.8, wsp_fig_late=4.8, wsp_abs=5.4,
              wc_fig=1.60, wc_abs=0.60, lscale_abs=0.28, rescale_phi=0.48)
D5_CFG = dict(tau_min=0.36, tau_hi=0.72, ws_fig=6.5, wsp_fig_late=2.4, wsp_abs=5.2,
              wc_fig=1.85, wc_abs=0.80, lscale_abs=0.34, rescale_phi=0.48)

FIG4_STEM = "Impressionism__armand-guillaumin_valley-in-pontgibaud-1890"


@torch.no_grad()
def integrate_segmented(model: SLFUNet, z_t4: torch.Tensor, eps: torch.Tensor,
                        y16: torch.Tensor, d16: torch.Tensor, tid: torch.Tensor,
                        core_bank: dict, *, steps: int, full_steps: set[int],
                        cfg_kw: dict) -> tuple[torch.Tensor, int]:
    """V20 D2/Core-SLF 积分器 + 逐物理步切换 full(4-branch) / cs_j(2-branch).

    返回 (z_final, nfe_per_sample), 其中 NFE = 2 * sum(4 if i in full_steps else 2).
    """
    dev = z_t4.device
    B = len(z_t4)
    _, u_core = match_scene_submode(d16, tid, core_bank)
    sc_src = F.normalize(F.avg_pool2d(d16.float(), 4).flatten(1), dim=-1)
    best_c = torch.einsum("bd,bkd->bk", sc_src, core_bank["sub_scene"].to(dev).float()[tid]).argmax(dim=1)
    ex_dsd = core_bank["sub_dsd"].to(dev).float()[tid, best_c]
    g_fig = torch.sigmoid((ex_dsd - float(core_bank["mid_dsd"])) * 550.0)
    gf4 = g_fig.view(B, 1, 1, 1)

    lscale = (cfg_kw["lscale_abs"] + (1.0 - cfg_kw["lscale_abs"]) * g_fig).clamp(0.15, 1.0)
    c8_w = (1.0 - g_fig).clamp(0.0, 0.85).view(B, 1, 1, 1)
    d8 = F.avg_pool2d(d16, 2).repeat_interleave(2, -2).repeat_interleave(2, -1)
    d8 = d8 - d8.mean(dim=(2, 3), keepdim=True)
    y8 = F.avg_pool2d(y16, 2).repeat_interleave(2, -2).repeat_interleave(2, -1)
    d_in = (1.0 - c8_w) * d16 + c8_w * d8
    y_in = (1.0 - c8_w) * y16 + c8_w * y8

    tau_v = (cfg_kw["tau_min"] + (cfg_kw["tau_hi"] - cfg_kw["tau_min"]) * g_fig).clamp(0.0, 0.78)
    z = (1.0 - tau_v.view(B, 1, 1, 1)) * eps + tau_v.view(B, 1, 1, 1) * z_t4

    f0 = torch.zeros(B, dtype=torch.bool, device=dev)
    f1 = torch.ones(B, dtype=torch.bool, device=dev)
    ls0 = torch.zeros(B, device=dev)
    rescale_phi = float(cfg_kw.get("rescale_phi", 0.48))

    def _vel(z_cur: torch.Tensor, prog: float, use_full: bool) -> torch.Tensor:
        tv = tau_v + prog * (1.0 - tau_v)
        gl = min(1.0, max(0.0, (prog - 0.36) / 0.32))
        ws_j = gf4 * (cfg_kw["ws_fig"] - 1.1 * gl) + (1.0 - gf4) * 1.4
        if use_full:
            with autocast(str(dev)):
                v_sc, v_s0, v_0c, v_00 = model(
                    z_cur.repeat(4, 1, 1, 1), y_in.repeat(4, 1, 1, 1), d_in.repeat(4, 1, 1, 1),
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
                z_cur.repeat(2, 1, 1, 1), y_in.repeat(2, 1, 1, 1), d_in.repeat(2, 1, 1, 1),
                tid.repeat(2), u_core.repeat(2, 1), tv.repeat(2),
                drop_style=torch.cat([f0, f1], 0),
                drop_content=torch.cat([f0, f0], 0),
                layout_scale=torch.cat([lscale, lscale], 0),
            ).float().chunk(2, 0)
        return v_0c + ws_j * (v_sc - v_0c)

    dt = (1.0 - tau_v).view(B, 1, 1, 1) / float(steps)
    nfe = 0
    for i in range(steps):
        uf = i in full_steps
        nfe += 8 if uf else 4
        v1 = _vel(z, i / float(steps), uf)
        zm = z + 0.5 * dt * v1
        v2 = _vel(zm, (i + 0.5) / float(steps), uf)
        z = z + dt * v2
    return z, nfe


def _export_imp_min_strip(arms_evaluated: list[str], tasks: list[dict], out_png: Path) -> None:
    """导出 Fig. 4 Row 2 (Imp→Min) + 3 个额外 Imp→Min 样本对比条, 供视觉检查去灰化."""
    from PIL import Image, ImageDraw
    picks = [i for i, t in enumerate(tasks)
             if t["source_style"] == "Impressionism" and t["target_style"] == "Minimalism"]
    # 确保 FIG4_STEM 排在第一行
    fig4_idx = next((i for i in picks if t_stem(tasks[i]) == FIG4_STEM), picks[0])
    chosen = [fig4_idx] + [i for i in picks if i != fig4_idx][:3]
    col_arms = arms_evaluated[:8]
    cell = 224
    W = (1 + len(col_arms)) * (cell + 6) + 10
    H = len(chosen) * (cell + 26) + 28
    canvas = Image.new("RGB", (W, H), (24, 24, 24))
    dr = ImageDraw.Draw(canvas)
    dr.text((8, 6), "Source", fill=(255, 255, 255))
    for ci, arm in enumerate(col_arms):
        dr.text((8 + (ci + 1) * (cell + 6), 6), arm[:22], fill=(200, 240, 255))
    for ri, tidx in enumerate(chosen):
        t = tasks[tidx]
        y0 = 26 + ri * (cell + 26)
        src_p = ROOT / "data/test" / t["source_style"] / f"{t['source_stem']}.jpg"
        if src_p.exists():
            canvas.paste(Image.open(src_p).convert("RGB").resize((cell, cell)), (6, y0))
        dr.text((6, y0 + cell + 2), t["source_stem"][:32], fill=(255, 220, 120))
        for ci, arm in enumerate(col_arms):
            png_p = OUT / "measured" / arm / "images" / f"{t['task_id']}.png"
            if not png_p.exists():
                # 兜底从 raw 取
                raw_dir = OUT / "raw" / arm
                for bf in sorted(raw_dir.glob("batch_*.pt")):
                    pl = torch.load(bf, map_location="cpu", weights_only=False)
                    if t["task_id"] in pl["ids"]:
                        j = list(pl["ids"]).index(t["task_id"])
                        im = Image.fromarray(pl["rgb_u8"][j].permute(1, 2, 0).numpy()).resize((cell, cell))
                        canvas.paste(im, (6 + (ci + 1) * (cell + 6), y0))
                        break
            else:
                canvas.paste(Image.open(png_p).convert("RGB").resize((cell, cell)),
                             (6 + (ci + 1) * (cell + 6), y0))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png)
    print(f"[Step2] Imp→Min strip → {out_png}", flush=True)


def t_stem(t: dict) -> str:
    return t["source_stem"]


def run_step2(pool: str, device: str = "cuda") -> dict:
    bundle = build_or_load_v19_foundation(pool, device=device)
    bank = build_or_load_core_knn_bank(bundle, k_sub=8, core_per_style=180, device=device)
    tasks, z_src_750, _ = tasks_and_sources()
    z_t4, _ = build_or_load_t4_latents_750(bundle, device=device)
    y16_750, d16_750 = get_test750_y16_and_d16(bundle)
    eps_750 = torch.randn(z_src_750.shape, generator=torch.Generator().manual_seed(EVAL_SEED))
    dec = load_s10_decoder(device)

    sid_all = torch.tensor([t["source_style_id"] for t in tasks])
    tid_all = torch.tensor([t["target_style_id"] for t in tasks])
    same_all = sid_all == tid_all

    # 扫描矩阵: (arm_name, host_key, steps, full_steps_set, cfg_kw)
    arms_spec = [
        ("ICME_D2_s8_full", "V20_D2", 8, set(range(8)), D2_CFG),
        ("ICME_D2_s8_e3full", "V20_D2", 8, {0, 1, 2}, D2_CFG),
        ("ICME_D2_s6_full", "V20_D2", 6, set(range(6)), D2_CFG),
        ("ICME_D2_s6_e2full", "V20_D2", 6, {0, 1}, D2_CFG),
        ("ICME_D2_s6_l2full", "V20_D2", 6, {4, 5}, D2_CFG),
        ("ICME_D2_s6_csj", "V20_D2", 6, set(), D2_CFG),
        ("ICME_D2_s4_e2full", "V20_D2", 4, {0, 1}, D2_CFG),
        ("ICME_JPk_s6_csj", "V23_HiC_JPk", 6, set(), D5_CFG),
        ("ICME_TrT1_s6_csj", "V25_Tr_T1", 6, set(), D5_CFG),
    ]

    results = {"task": "ICME Step2: Closed-set host consolidation & few-step segmented CFG sweep",
               "updated_utc": utc(), "arms": {}}
    models_cache: dict[str, SLFUNet] = {}
    for arm_name, host_k, steps, full_set, cfg_kw in arms_spec:
        m_dir = OUT / "measured" / arm_name
        nfe = 2 * sum(4 if i in full_set else 2 for i in range(steps))
        if (m_dir / "summary_v27.json").exists():
            s = read_json(m_dir / "summary_v27.json")
        else:
            if host_k not in models_cache:
                m = SLFUNet(base_dim=144, u_dim=128).to(device).eval()
                m.load_state_dict(torch.load(HOST_CKPTS[host_k], map_location="cpu", weights_only=False)["ema"], strict=True)
                for p in m.parameters():
                    p.requires_grad_(False)
                models_cache[host_k] = m
            model = models_cache[host_k]
            wr = RawWriterV27(arm_name, dec=dec, device=device)
            t0 = time.time()
            for st in range(0, 750, 25):
                sl = slice(st, min(st + 25, 750))
                z_gen, _ = integrate_segmented(
                    model, z_t4[sl].to(device).float(), eps_750[sl].to(device).float(),
                    y16_750[sl].to(device).float(), d16_750[sl].to(device).float(),
                    tid_all[sl].to(device), bank, steps=steps, full_steps=full_set, cfg_kw=cfg_kw,
                )
                z_out = torch.where(same_all[sl].view(-1, 1, 1, 1).to(device),
                                    z_src_750[sl].to(device).float(), z_gen)
                wr.write_batch(z_out, tasks[sl])
            wr.finalize(host=host_k, steps=steps, full_steps=sorted(full_set), nfe=nfe, wall_s=round(time.time() - t0, 1))
            s = run_measure_v27(arm_name, device=device)
        c = s["cross600"]
        pt = s.get("per_target_cross120", {})
        # 计算 Imp→Min 单路线 Top-1
        rows = read_json(m_dir / "eval.json")["rows"]
        imp_min = [r for r in rows if r["src_style"] == "Impressionism" and r["tgt_style"] == "Minimalism"]
        imp_min_t1 = round(sum(int(r["target_top1"]) for r in imp_min) / max(len(imp_min), 1), 4)
        results["arms"][arm_name] = {
            "host": host_k, "steps": steps, "nfe": nfe, "full_steps": sorted(full_set),
            "csd_top1": round(c["csd_top1"], 4), "dino_s": round(c["dino_s"], 4),
            "dino_c": round(c["dino_c"], 4), "lpips": round(c["lpips"], 4),
            "style_affinity": round(c["style_affinity"], 4), "clip_s": round(c["clip_s"], 4),
            "same150_top1": round(s["same150"]["csd_top1"], 4),
            "min_top1": round(pt.get("Minimalism", {}).get("csd_top1", 0.0), 4),
            "imp_to_min_top1": imp_min_t1,
        }
        save_json(OUT / "icme_step2_accel.json", results)
        print(f"[Step2] {arm_name:20s} nfe={nfe:2d} | Top1={c['csd_top1']*100:.2f}% "
              f"D-S={c['dino_s']:.4f} D-C={c['dino_c']:.4f} Imp→Min={imp_min_t1*100:.1f}%", flush=True)

    _export_imp_min_strip([a[0] for a in arms_spec], tasks, OUT / "icme_step2_imp_min_strip.png")
    print(f"[Step2] DONE → {OUT / 'icme_step2_accel.json'}", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    sanitize_cuda_env()
    run_step2(args.pool, device=args.device)


if __name__ == "__main__":
    main()
