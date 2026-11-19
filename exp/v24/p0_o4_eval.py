"""V24 P0 (3060): O4-A/B/C Source Anchor-Bridge 工作点基线 (M3_ab_cross50 @ srcA0 tau0=0.40/0.50/0.60)。

测量在提高 srcA0 起点 tau0 后 One-Shot 在 DINO-C 0.55~0.72 区间上的内容-风格前沿，为 O1/O2/O3 提供基线。

用法 (3060):
  python -m exp.v24.p0_o4_eval --pool ../pseudo_gt_24h/orig_mean_fp16
"""
from __future__ import annotations

import argparse
import time

import torch

from . import OUT, ROOT, RawWriterV24, autocast, config_hash, export_raw_pngs, file_sha, git_sha, read_json, run_measure_v24, sanitize_cuda_env, save_json, utc, validate_raw_meta
from exp.v23.p2_attn_cross import SLFUNetRefAttn, _sg5_ref_d16
from exp.v18 import tasks_and_sources
from exp.v19.fiber_duality import StyleFiberEncoder, build_or_load_v19_foundation
from exp.v20.slf_flow import coarse_y16_lcn
from exp.v22.ref_adapter import RefAdapter
from src.utils import load_s10_decoder

M3_CKPT = ROOT / "runs/v23/oneshot_attn/M3_ab_cross50.pt"


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    sanitize_cuda_env()
    bundle = build_or_load_v19_foundation(args.pool, device=args.device)
    c21 = torch.load(ROOT / "runs/v21/cache/v21_stiefel_ref_cache.pt", map_location="cpu", weights_only=False)
    tasks, z_src_750, _ = tasks_and_sources()
    dec = load_s10_decoder(args.device)

    ck = torch.load(M3_CKPT, map_location="cpu", weights_only=False)
    fm = SLFUNetRefAttn(base_dim=144, u_dim=128, use_attn=True, use_conf_gate=True).to(args.device).eval()
    fm.load_state_dict(ck["fm"], strict=True)
    ad = RefAdapter(dim=128).to(args.device).eval()
    ad.load_state_dict(ck["adapter"], strict=True)

    fiber_enc = StyleFiberEncoder(in_dim=bundle["fnd"]["tr_fib"].shape[1], dim=128).to(args.device).eval()
    fiber_enc.load_state_dict(bundle["fnd"]["fiber_state"])
    u_src_750 = fiber_enc.encode(bundle["fnd"]["test750_fib"].to(args.device).float()).cpu()
    del fiber_enc

    w_l = bundle["v18_cache"]["luma_proj"]["w"].float().view(1, 4, 1, 1)
    b_l = bundle["v18_cache"]["luma_proj"]["b"].float()
    y16_750 = coarse_y16_lcn(((z_src_750.float() * w_l).sum(1, keepdim=True) + b_l).clamp(0.0, 1.0)).half()
    eps_750 = torch.randn(z_src_750.shape, generator=torch.Generator().manual_seed(20261185))
    u_sg5 = c21["sg5"]["u"].float()
    d16_sg5 = _sg5_ref_d16(bundle, c21, device=args.device)

    results = []
    for tau0, wj, tag in ((0.40, 2.2, "O4A_t40"), (0.50, 2.2, "O4B_t50"), (0.60, 2.0, "O4C_t60")):
        cfg_hash = config_hash({"arm": tag, "tau0": tau0, "wj": wj, "ckpt": file_sha(M3_CKPT)[:12]})
        if (OUT / "raw" / tag / "meta.json").exists() and validate_raw_meta(read_json(OUT / "raw" / tag / "meta.json")):
            s = read_json(OUT / f"measured/{tag}/summary_v24.json")
        else:
            wr = RawWriterV24(tag, dec=dec, device=args.device)
            t0 = time.time()
            for st in range(0, 750, 25):
                sl = slice(st, min(st + 25, 750))
                grp = tasks[sl]
                B = len(grp)
                zs = z_src_750[sl].to(args.device).float()
                sid_p = torch.tensor([t["source_style_id"] for t in grp], device=args.device)
                tid_t = torch.tensor([t["target_style_id"] for t in grp], device=args.device)
                same = sid_p == tid_t

                u_style = ad(u_sg5[tid_t.cpu()].to(args.device), u_src_750[sl].to(args.device))
                d16_r = d16_sg5[tid_t.cpu()].to(args.device)
                y16 = y16_750[sl].to(args.device).float()
                d16 = bundle["test_struct"]["dino_16"][sl].to(args.device).float()
                z = (1.0 - tau0) * eps_750[sl].to(args.device).float() + tau0 * zs

                f0, f1 = torch.zeros(B, dtype=torch.bool, device=args.device), torch.ones(B, dtype=torch.bool, device=args.device)
                l1, sid0 = torch.ones(B, device=args.device), torch.zeros(B, dtype=torch.long, device=args.device)
                steps = 10
                dt = (1.0 - tau0) / float(steps)
                for i in range(steps):
                    def _v(zz, pr):
                        tv = torch.full((B,), tau0 + pr * (1.0 - tau0), device=args.device)
                        zz2 = torch.cat([zz, zz], 0)
                        with autocast(args.device):
                            v_sc, v_0c = fm(zz2, y16.repeat(2, 1, 1, 1), d16.repeat(2, 1, 1, 1),
                                            sid0.repeat(2), u_style.repeat(2, 1), tv.repeat(2),
                                            drop_style=torch.cat([f0, f1], 0), drop_content=torch.cat([f0, f0], 0),
                                            layout_scale=torch.cat([l1, l1], 0), d16_ref=d16_r.repeat(2, 1, 1, 1)).float().chunk(2, 0)
                        return v_0c + wj * (v_sc - v_0c)
                    v1 = _v(z, i / float(steps))
                    v2 = _v(z + 0.5 * dt * v1, (i + 0.5) / float(steps))
                    z = z + dt * v2
                z_out = torch.where(same.view(B, 1, 1, 1), zs, z)
                wr.write_batch(z_out, grp)
            meta = wr.finalize(cfg_hash=cfg_hash, tau0=tau0, wj=wj, wall_sec=round(time.time() - t0, 1))
            assert validate_raw_meta(meta)
            s = run_measure_v24(tag, device=args.device)
            export_raw_pngs(OUT / f"raw/{tag}", OUT / f"png_750/{tag}", max_workers=8)
        c = s["cross600"]
        results.append({"arm": tag, "dino_c": round(c["dino_c"], 5), "dino_s": round(c["dino_s"], 5),
                        "lpips": round(c["lpips"], 5), "style_affinity": round(c["style_affinity"], 5),
                        "csd_top1": round(c["csd_top1"], 5)})

    save_json(OUT / "p0_o4_eval.json", {"updated_utc": utc(), "results": results})
    print("\n=== V24 P0 O4-A/B/C 汇总 (目标: DINO-C >= 0.70, DINO-S >= 0.48-0.50, LPIPS <= 0.48) ===", flush=True)
    for r in results:
        print(f"{r['arm']:12s}: DINO-C={r['dino_c']:.4f} DINO-S={r['dino_s']:.4f} LPIPS={r['lpips']:.4f} Top1={r['csd_top1']*100:5.2f}%", flush=True)


if __name__ == "__main__":
    main()
