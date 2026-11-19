"""V24 P0 (4090): M0 (PRISM-1 λ=0.75/1.0 & V20 D6 重评), M1 (潜空间小幅混合), M2 (Haar DWT 频带混合)。

目标: 在 DINO-C >= 0.73 区间测定纯高内容基座 + 风格增量的物理可行界，为 M3/M4 残差网络定锚。

用法 (4090):
  python -m exp.v24.p0_m0_m2 --pool ../pseudo_gt_24h/orig_mean_fp16
"""
from __future__ import annotations

import argparse
import time

import torch

from . import OUT, EVAL_SEED, ROOT, RawWriterV24, config_hash, export_raw_pngs, file_sha, git_sha, read_json, run_measure_v24, sanitize_cuda_env, save_json, utc, validate_raw_meta
from exp.v22.core import CFGConfig, D6_SOLVER, GateConfig, PrismCore
from exp.v22.t4 import build_or_load_t4_latents_750
from exp.v18 import tasks_and_sources
from exp.v19.fiber_duality import StyleFiberEncoder, apply_latent_chroma_shift, build_or_load_v19_foundation
from exp.v19.routed_student import SpatiallyRoutedChordNet
from exp.v20.run_v20 import get_test750_y16_and_d16
from exp.v20.submode_patchswd import build_or_load_core_knn_bank
from src.utils import load_s10_decoder
from src.utils.structure import dwt2_haar, idwt2_haar

P1_CKPT = ROOT / "runs/v19/students/s2_prism1_v2.pt"
V20_CKPT = ROOT / "runs/v20/slf_flow/slf_core_v2.pt"


@torch.no_grad()
def compute_prism1_latents(bundle: dict, tasks: list[dict], z_src_750: torch.Tensor,
                           d16_750: torch.Tensor, lam: float, device: str = "cuda") -> torch.Tensor:
    fnd = bundle["fnd"]
    stu = SpatiallyRoutedChordNet(width=96, chord_dim=128, use_spatial_router=True).to(device).eval()
    stu.load_state_dict(torch.load(P1_CKPT, map_location="cpu", weights_only=False)["model"])
    fiber_enc = StyleFiberEncoder(in_dim=fnd["tr_fib"].shape[1], dim=128).to(device).eval()
    fiber_enc.load_state_dict(fnd["fiber_state"])
    u_src_all = fiber_enc.encode(fnd["test750_fib"].to(device).float())
    bary = fiber_enc.class_barycenters()
    priors = {k: v.to(device) if torch.is_tensor(v) else v for k, v in fnd["priors"].items()}
    P_Y = priors["P_Y"]
    c64_750 = bundle["test_struct"]["c_64"].float()
    z0_750 = torch.load(ROOT / "runs/v9_local_4070/cache/n2c_z.pt", map_location="cpu", weights_only=False)["z0"].float()

    sid_all = torch.tensor([t["source_style_id"] for t in tasks], device=device)
    tid_all = torch.tensor([t["target_style_id"] for t in tasks], device=device)
    same_t = sid_all == tid_all
    alpha_eff = min(0.45, 0.20 * lam)

    out = torch.zeros_like(z_src_750, dtype=torch.float32)
    for st in range(0, 750, 50):
        sl = slice(st, min(st + 50, 750))
        zs = z_src_750[sl].to(device).float()
        z_anc = apply_latent_chroma_shift(z0_750[sl].to(device), tid_all[sl], priors, alpha=alpha_eff)
        chord = bary[tid_all[sl]] - u_src_all[sl]
        z_luma = torch.einsum("ij,bjhw->bihw", P_Y, zs)
        delta = stu(z_anc, z_luma, c64_750[sl].to(device).float(), d16_750[sl].to(device).float(), chord, same_mask=same_t[sl])
        z_out = z_anc + lam * delta
        z_out[same_t[sl]] = zs[same_t[sl]]
        out[sl] = z_out.cpu()
    del stu, fiber_enc
    torch.cuda.empty_cache()
    return out


@torch.no_grad()
def compute_d6_latents(bundle: dict, bank: dict, tasks: list[dict], z_src_750: torch.Tensor,
                       z_t4: torch.Tensor, y16_750: torch.Tensor, d16_750: torch.Tensor,
                       eps_750: torch.Tensor, device: str = "cuda") -> torch.Tensor:
    core = PrismCore(bundle, bank, V20_CKPT, device=device)
    sid_all = torch.tensor([t["source_style_id"] for t in tasks])
    tid_all = torch.tensor([t["target_style_id"] for t in tasks])
    same_all = sid_all == tid_all
    out = torch.zeros_like(z_src_750, dtype=torch.float32)
    for st in range(0, 750, 25):
        sl = slice(st, min(st + 25, 750))
        z_gen, _ = core.integrate_flow(
            eps_750[sl], z_t4[sl].to(device), y16_750[sl], d16_750[sl], tid_all[sl].to(device),
            gate=GateConfig(mode="original"), cfg=CFGConfig(branches="full", rescale_phi=0.48), solver=dict(D6_SOLVER),
        )
        z_out = torch.where(same_all[sl].view(-1, 1, 1, 1).to(device), z_src_750[sl].to(device).float(), z_gen)
        out[sl] = z_out.cpu()
    del core
    torch.cuda.empty_cache()
    return out


def eval_latent_tensor(arm_name: str, z_750: torch.Tensor, tasks: list[dict], dec, device: str = "cuda") -> dict:
    cfg_hash = config_hash({"arm": arm_name, "v": 24})
    if (OUT / "raw" / arm_name / "meta.json").exists() and validate_raw_meta(read_json(OUT / "raw" / arm_name / "meta.json")):
        s = read_json(OUT / f"measured/{arm_name}/summary_v24.json")
    else:
        wr = RawWriterV24(arm_name, dec=dec, device=device)
        t0 = time.time()
        for st in range(0, 750, 25):
            sl = slice(st, min(st + 25, 750))
            wr.write_batch(z_750[sl].to(device), tasks[sl])
        meta = wr.finalize(cfg_hash=cfg_hash, git_sha=git_sha(), wall_sec=round(time.time() - t0, 1))
        assert validate_raw_meta(meta)
        s = run_measure_v24(arm_name, device=device)
        export_raw_pngs(OUT / f"raw/{arm_name}", OUT / f"png_750/{arm_name}", max_workers=8)
    c = s["cross600"]
    return {
        "arm": arm_name, "dino_c": round(c["dino_c"], 5), "dino_s": round(c["dino_s"], 5),
        "lpips": round(c["lpips"], 5), "style_affinity": round(c["style_affinity"], 5),
        "csd_top1": round(c["csd_top1"], 5), "same_top1": round(s["same150"]["csd_top1"], 5),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    sanitize_cuda_env()
    bundle = build_or_load_v19_foundation(args.pool, device=args.device)
    bank = build_or_load_core_knn_bank(bundle, k_sub=8, core_per_style=180, device=args.device)
    tasks, z_src_750, _ = tasks_and_sources()
    z_t4, _ = build_or_load_t4_latents_750(bundle, device=args.device)
    y16_750, d16_750 = get_test750_y16_and_d16(bundle)
    eps_750 = torch.randn(z_src_750.shape, generator=torch.Generator().manual_seed(EVAL_SEED))
    dec = load_s10_decoder(args.device)

    cache_p = OUT / "cache/p0_base_latents_750.pt"
    cache_p.parent.mkdir(parents=True, exist_ok=True)
    if cache_p.exists():
        d = torch.load(cache_p, map_location="cpu", weights_only=False)
        z_p1_075, z_p1_100, z_d6 = d["z_p1_075"], d["z_p1_100"], d["z_d6"]
    else:
        z_p1_075 = compute_prism1_latents(bundle, tasks, z_src_750, d16_750, 0.75, device=args.device)
        z_p1_100 = compute_prism1_latents(bundle, tasks, z_src_750, d16_750, 1.00, device=args.device)
        z_d6 = compute_d6_latents(bundle, bank, tasks, z_src_750, z_t4, y16_750, d16_750, eps_750, device=args.device)
        torch.save({"z_p1_075": z_p1_075, "z_p1_100": z_p1_100, "z_d6": z_d6}, cache_p)

    results = []
    # M0 基线核验
    results.append(eval_latent_tensor("M0_prism1_l075", z_p1_075, tasks, dec, device=args.device))
    results.append(eval_latent_tensor("M0_prism1_l100", z_p1_100, tasks, dec, device=args.device))
    results.append(eval_latent_tensor("M0_v20_d6", z_d6, tasks, dec, device=args.device))

    # M1: PRISM-1(l=1.0) 与 V20 D6 潜空间线性小幅混合
    for gamma in (0.25, 0.40, 0.55):
        z_mix = (1.0 - gamma) * z_p1_100 + gamma * z_d6
        results.append(eval_latent_tensor(f"M1_lin_g{int(gamma*100)}", z_mix, tasks, dec, device=args.device))

    # M2: Haar DWT 按频带混合 (LL 保内容基座, LH/HL/HH 高中频增风格)
    ll_p, lh_p, hl_p, hh_p = dwt2_haar(z_p1_100)
    ll_d, lh_d, hl_d, hh_d = dwt2_haar(z_d6)
    for w_ll, w_hf, tag in ((0.0, 0.50, "ll0_hf50"), (0.0, 0.85, "ll0_hf85"), (0.25, 0.85, "ll25_hf85")):
        z_band = idwt2_haar(
            (1.0 - w_ll) * ll_p + w_ll * ll_d,
            (1.0 - w_hf) * lh_p + w_hf * lh_d,
            (1.0 - w_hf) * hl_p + w_hf * hl_d,
            (1.0 - w_hf) * hh_p + w_hf * hh_d,
        )
        results.append(eval_latent_tensor(f"M2_dwt_{tag}", z_band, tasks, dec, device=args.device))

    save_json(OUT / "p0_m0_m2.json", {"updated_utc": utc(), "results": results})
    print("\n=== V24 P0 (M0–M2) 汇总 (目标: DINO-C >= 0.73, DINO-S >= 0.49-0.51, LPIPS <= 0.51) ===", flush=True)
    for r in results:
        print(f"{r['arm']:18s}: DINO-C={r['dino_c']:.4f} DINO-S={r['dino_s']:.4f} LPIPS={r['lpips']:.4f} Top1={r['csd_top1']*100:5.2f}%", flush=True)


if __name__ == "__main__":
    main()
