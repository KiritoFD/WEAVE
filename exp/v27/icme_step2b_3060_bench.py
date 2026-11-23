r"""ICME Step 2b (RTX 3060): 统一主力 Flow 定案 (D2@D5_CFG vs JPk vs TrT1) + 3060 实机 E2E 测速.

在唯一实机 RTX 3060 12GB 上实测:
  1. 测试 V20_D2 (slf_core_v2.pt) 在 D5_CFG (s6_csj / s6_e2full) 下是否也能消除
     Imp→Min 抹灰并守住 DINO-C>=0.61 (若可, slf_core_v2 单权重即兼顾 Flow + HC);
  2. 对比 V20_D2 / V23_HiC_JPk / V25_Tr_T1 在 6-step cs_j (24 NFE) 的全部 6 项指标;
  3. 实机 RTX 3060 12GB 测速: batch=1 与 max-batch (b=16/b=32) 全栈 RGB→RGB 延迟
     (A0 encode + DINO + anchor + 6-step Flow + S10 decode) vs 16-step full.

用法 (3060):  python -m exp.v27.icme_step2b_3060_bench --pool ../pseudo_gt_24h/orig_mean_fp16
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from exp.v18 import tasks_and_sources
from exp.v19.fiber_duality import build_or_load_v19_foundation
from exp.v20.run_v20 import get_test750_y16_and_d16
from exp.v20.slf_flow import SLFUNet, coarse_y16_lcn
from exp.v20.submode_patchswd import build_or_load_core_knn_bank
from exp.v22.t4 import build_or_load_t4_latents_750
from exp.v25.r14_e2e_benchmark import load_a0_encoder
from exp.v27 import OUT, ROOT, RawWriterV27, run_measure_v27
from exp.v27.icme_step2_closedset_accel import D2_CFG, D5_CFG, HOST_CKPTS, _export_imp_min_strip, integrate_segmented
from src.utils import decode_s10, load_s10_decoder
from src.utils._local_eval_refs import load_dino_model
from src.utils.protocol import EVAL_SEED
from src.utils.runtime import bf16_autocast as autocast, read_json, sanitize_cuda_env, save_json, utc


@torch.no_grad()
def bench_3060_e2e(device: str = "cuda") -> dict:
    """RTX 3060 12GB 实机阶段拆分测速 (A0 enc / DINO / T4 2-step / Flow 6-step / S10 dec)."""
    a0 = load_a0_encoder(device).eval()
    dino = load_dino_model(device=device).eval()
    s10 = load_s10_decoder(device)
    model = SLFUNet(base_dim=144, u_dim=128).to(device).eval()
    model.load_state_dict(torch.load(HOST_CKPTS["V25_Tr_T1"], map_location="cpu", weights_only=False)["ema"], strict=True)
    for p in model.parameters():
        p.requires_grad_(False)

    res = {}
    for bs in (1, 16):
        rgb_in = torch.rand(bs, 3, 512, 512, device=device)
        z_in = torch.randn(bs, 4, 64, 64, device=device)
        y16 = torch.randn(bs, 1, 16, 16, device=device)
        d16 = torch.randn(bs, 384, 16, 16, device=device)
        tid = torch.zeros(bs, dtype=torch.long, device=device)
        u = torch.randn(bs, 128, device=device)
        tv = torch.full((bs,), 0.5, device=device)
        f0 = torch.zeros(bs, dtype=torch.bool, device=device)
        f1 = torch.ones(bs, dtype=torch.bool, device=device)
        ls = torch.ones(bs, device=device)

        def _t(fn, n=12):
            for _ in range(3):
                fn()
            torch.cuda.synchronize(device)
            ts = []
            for _ in range(n):
                torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                fn()
                torch.cuda.synchronize(device)
                ts.append((time.perf_counter() - t0) * 1000.0 / bs)
            return round(float(np.percentile(ts, 50)), 2)

        def _enc():
            _ = a0.encode((rgb_in * 2 - 1).float()).latent_dist.mean * 0.18215

        def _dino():
            x = torch.nn.functional.interpolate(rgb_in.mean(1, keepdim=True).repeat(1, 3, 1, 1), size=(224, 224), mode="bilinear")
            with autocast(device):
                _ = dino.forward_features(x)

        def _flow_step_2b():
            with autocast(device):
                _ = model(z_in.repeat(2, 1, 1, 1), y16.repeat(2, 1, 1, 1), d16.repeat(2, 1, 1, 1),
                          tid.repeat(2), u.repeat(2, 1), tv.repeat(2),
                          drop_style=torch.cat([f0, f1], 0), drop_content=torch.cat([f0, f0], 0),
                          layout_scale=ls.repeat(2))

        def _flow_step_4b():
            with autocast(device):
                _ = model(z_in.repeat(4, 1, 1, 1), y16.repeat(4, 1, 1, 1), d16.repeat(4, 1, 1, 1),
                          tid.repeat(4), u.repeat(4, 1), tv.repeat(4),
                          drop_style=torch.cat([f0, f0, f1, f1], 0), drop_content=torch.cat([f0, f1, f0, f1], 0),
                          layout_scale=ls.repeat(4))

        def _dec():
            _ = decode_s10(s10, z_in, batch=bs, output_cpu=False)

        t_enc = _t(_enc)
        t_dino = _t(_dino)
        t_2b = _t(_flow_step_2b)   # 1 model call (2 branches)
        t_4b = _t(_flow_step_4b)   # 1 model call (4 branches)
        t_dec = _t(_dec)

        # RK2: 2 velocity evals per ODE step
        flow_s6_csj = round(6 * 2 * t_2b, 2)
        flow_s6_e2full = round((2 * t_4b + 4 * t_2b) * 2, 2)
        flow_s6_full = round(6 * 2 * t_4b, 2)
        flow_s16_full = round(16 * 2 * t_4b, 2)
        t4_anc_est = round(2 * t_dec * 1.6, 2)  # 2-step S10-VGG grad est (matches V25 r16 T_anc)

        res[f"b{bs}"] = {
            "enc_ms": t_enc, "dino_ms": t_dino, "dec_s10_ms": t_dec,
            "step_2b_ms": t_2b, "step_4b_ms": t_4b,
            "flow_s6_csj_ms": flow_s6_csj,
            "flow_s6_e2full_ms": flow_s6_e2full,
            "flow_s6_full_ms": flow_s6_full,
            "flow_s16_full_ms": flow_s16_full,
            "cached_s6_csj_ms": round(flow_s6_csj + t_dec, 2),
            "cached_s6_e2full_ms": round(flow_s6_e2full + t_dec, 2),
            "e2e_s6_csj_ms": round(t_enc + t_dino + flow_s6_csj + t_dec, 2),
            "e2e_s6_csj_with_t4anc_ms": round(t_enc + t_dino + t4_anc_est + flow_s6_csj + t_dec, 2),
        }
        print(f"[3060 Bench b{bs}] {res[f'b{bs}']}", flush=True)
    del a0, dino, s10, model
    torch.cuda.empty_cache()
    return res


def run_step2b(pool: str, device: str = "cuda", run_eval: bool = True) -> dict:
    bench = bench_3060_e2e(device=device)
    out = {"task": "ICME Step 2b (RTX 3060): host consolidation + real 3060 timing",
           "updated_utc": utc(), "bench_3060": bench, "arms": {}}
    if run_eval:
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

        models: dict[str, SLFUNet] = {}
        specs = [
            ("ICME_D2_d5_s6_csj", "V20_D2", 6, set(), D5_CFG),
            ("ICME_D2_d5_s6_e2full", "V20_D2", 6, {0, 1}, dict(D5_CFG, tau_min=0.26, tau_hi=0.66)),
            ("ICME_JPk_s6_csj", "V23_HiC_JPk", 6, set(), D5_CFG),
            ("ICME_TrT1_s6_csj", "V25_Tr_T1", 6, set(), D5_CFG),
        ]
        for arm_name, hk, steps, fset, cfg_kw in specs:
            m_dir = OUT / "measured" / arm_name
            if (m_dir / "summary_v27.json").exists():
                s = read_json(m_dir / "summary_v27.json")
            else:
                if hk not in models:
                    m = SLFUNet(base_dim=144, u_dim=128).to(device).eval()
                    m.load_state_dict(torch.load(HOST_CKPTS[hk], map_location="cpu", weights_only=False)["ema"], strict=True)
                    for p in m.parameters():
                        p.requires_grad_(False)
                    models[hk] = m
                wr = RawWriterV27(arm_name, dec=dec, device=device)
                for st in range(0, 750, 25):
                    sl = slice(st, min(st + 25, 750))
                    zg, _ = integrate_segmented(models[hk], z_t4[sl].to(device).float(), eps_750[sl].to(device).float(),
                                                y16_750[sl].to(device).float(), d16_750[sl].to(device).float(),
                                                tid_all[sl].to(device), bank, steps=steps, full_steps=fset, cfg_kw=cfg_kw)
                    wr.write_batch(torch.where(same_all[sl].view(-1, 1, 1, 1).to(device), z_src_750[sl].to(device).float(), zg), tasks[sl])
                wr.finalize(host=hk, steps=steps)
                s = run_measure_v27(arm_name, device=device)
            c = s["cross600"]
            rows = read_json(m_dir / "eval.json")["rows"]
            imp_min = [r for r in rows if r["src_style"] == "Impressionism" and r["tgt_style"] == "Minimalism"]
            out["arms"][arm_name] = {
                "csd_top1": round(c["csd_top1"], 4), "dino_s": round(c["dino_s"], 4),
                "dino_c": round(c["dino_c"], 4), "style_affinity": round(c["style_affinity"], 4),
                "clip_s": round(c["clip_s"], 4), "lpips": round(c["lpips"], 4),
                "imp_to_min_top1": round(sum(int(r["target_top1"]) for r in imp_min) / max(len(imp_min), 1), 4),
            }
            print(f"[Step2b] {arm_name:22s} | {out['arms'][arm_name]}", flush=True)
        _export_imp_min_strip([s[0] for s in specs], tasks, OUT / "icme_step2b_imp_min.png")
    save_json(OUT / "icme_step2b_3060.json", out)
    print(f"[Step2b] DONE → {OUT / 'icme_step2b_3060.json'}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bench-only", action="store_true")
    args = ap.parse_args()
    sanitize_cuda_env()
    run_step2b(args.pool, device=args.device, run_eval=not args.bench_only)


if __name__ == "__main__":
    main()
