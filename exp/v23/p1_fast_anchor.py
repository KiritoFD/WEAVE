"""V23 P1 第一组 (4090): Peak1500 & T1 × 快速锚点整合 (T4 / PRISM-1 / Chroma-Shift)。

冻结两个工作点:
- hiS (高风格): g=0.45, wc=1.0
- hiC (高内容): g=0.50, wc=1.5

用法 (4090):
  python -m exp.v23.p1_fast_anchor --pool ../pseudo_gt_24h/orig_mean_fp16
"""
from __future__ import annotations

import argparse
import time

import torch

from . import OUT, EVAL_SEED, ROOT, RawWriterV23, config_hash, export_raw_pngs, file_sha, git_sha, read_json, run_measure_v23, sanitize_cuda_env, save_json, utc, validate_raw_meta
from exp.v22.core import CFGConfig, D5_SOLVER, GateConfig, PrismCore
from exp.v22.t4 import build_or_load_t4_latents_750
from exp.v18 import tasks_and_sources
from exp.v19.fiber_duality import apply_latent_chroma_shift, build_or_load_v19_foundation
from exp.v20.run_v20 import get_test750_y16_and_d16
from exp.v20.submode_patchswd import build_or_load_core_knn_bank

PEAK_CKPT = ROOT / "runs/v22/tmatrix/slf_T4Ssnap1500.pt"
T1_CKPT = ROOT / "runs/v22/tmatrix/slf_T1.pt"


@torch.no_grad()
def compute_prism1_anchor_750(bundle: dict, tasks: list[dict], z_src_750: torch.Tensor,
                              d16_750: torch.Tensor, device: str = "cuda") -> torch.Tensor:
    from exp.v19.routed_student import SpatiallyRoutedChordNet
    from exp.v19.fiber_duality import StyleFiberEncoder

    fnd = bundle["fnd"]
    stu = SpatiallyRoutedChordNet(width=96, chord_dim=128, use_spatial_router=True).to(device).eval()
    ck = torch.load(ROOT / "runs/v19/students/s2_prism1_v2.pt", map_location="cpu", weights_only=False)
    stu.load_state_dict(ck["model"])

    fiber_enc = StyleFiberEncoder(in_dim=fnd["tr_fib"].shape[1], dim=128).to(device).eval()
    fiber_enc.load_state_dict(fnd["fiber_state"])
    fib_750 = fnd["test750_fib"].float()
    u_src_all = fiber_enc.encode(fib_750.to(device))
    bary = fiber_enc.class_barycenters()

    priors = {k: v.to(device) if torch.is_tensor(v) else v for k, v in fnd["priors"].items()}
    P_Y = priors["P_Y"]
    c64_750 = bundle["test_struct"]["c_64"].float()
    sid_all = torch.tensor([t["source_style_id"] for t in tasks], device=device)
    tid_all = torch.tensor([t["target_style_id"] for t in tasks], device=device)
    same_t = sid_all == tid_all
    z0_750 = torch.load(ROOT / "runs/v9_local_4070/cache/n2c_z.pt", map_location="cpu", weights_only=False)["z0"].float()

    prism1_anchor = torch.zeros(750, 4, 64, 64)
    for st in range(0, 750, 50):
        sl = slice(st, min(st + 50, 750))
        zs = z_src_750[sl].to(device).float()
        z0_b = z0_750[sl].to(device)
        z_anc = apply_latent_chroma_shift(z0_b, tid_all[sl], priors, alpha=0.20)
        chord = bary[tid_all[sl]] - u_src_all[sl]
        z_luma = torch.einsum("ij,bjhw->bihw", P_Y, zs)
        delta = stu(z_anc, z_luma, c64_750[sl].to(device).float(), d16_750[sl].to(device).float(), chord, same_mask=same_t[sl])
        z_out = z_anc + delta
        z_out[same_t[sl]] = zs[same_t[sl]]
        prism1_anchor[sl] = z_out.detach().cpu().float()
    del stu, fiber_enc
    torch.cuda.empty_cache()
    return prism1_anchor


@torch.no_grad()
def run_anchor_arm(core: PrismCore, ckpt_path, ctx: dict, arm_name: str,
                   anchor_750: torch.Tensor, g_val: float, wc_val: float, device: str = "cuda") -> dict:
    cfg_hash = config_hash({"arm": arm_name, "ckpt": file_sha(ckpt_path)[:12], "g": g_val, "wc": wc_val})
    out_dir = OUT / "raw" / arm_name
    if out_dir.joinpath("meta.json").exists():
        old = read_json(out_dir / "meta.json")
        if old.get("cfg_hash") == cfg_hash and validate_raw_meta(old):
            s = read_json(OUT / f"measured/{arm_name}/summary_v23.json")
            c = s["cross600"]
            print(f"[V23 SKIP] {arm_name}: Top1={c['csd_top1']*100:.2f}% D-C={c['dino_c']:.4f}", flush=True)
            return {"arm": arm_name, "csd_top1": round(c["csd_top1"], 5), "dino_s": round(c["dino_s"], 5),
                    "dino_c": round(c["dino_c"], 5), "style_affinity": round(c["style_affinity"], 5),
                    "lpips": round(c["lpips"], 5)}

    tasks = ctx["tasks"]
    z_src_750 = ctx["z_src_750"]
    sid_all = torch.tensor([t["source_style_id"] for t in tasks])
    tid_all = torch.tensor([t["target_style_id"] for t in tasks])
    same_all = sid_all == tid_all
    eps_750 = ctx["eps_750"]
    y16_750, d16_750 = ctx["y16_750"], ctx["d16_750"]

    gate = GateConfig(mode="fixed", fixed_g=g_val)
    cfg = CFGConfig(branches="full", rescale_phi=0.48)
    solver = dict(D5_SOLVER, wc_fig=wc_val, wc_abs=wc_val)
    wr = RawWriterV23(arm_name, dec=ctx["dec"], device=device)
    t0 = time.time()

    for st in range(0, 750, 25):
        sl = slice(st, min(st + 25, 750))
        grp = tasks[sl]
        z_gen, _ = core.integrate_flow(
            eps_750[sl], anchor_750[sl].to(device), y16_750[sl], d16_750[sl],
            tid_all[sl].to(device), gate=gate, cfg=cfg, solver=solver,
        )
        z_out = torch.where(same_all[sl].view(-1, 1, 1, 1).to(device),
                            z_src_750[sl].to(device).float(), z_gen)
        wr.write_batch(z_out, grp)

    meta = wr.finalize(cfg_hash=cfg_hash, git_sha=git_sha(), ckpt_sha256=file_sha(ckpt_path),
                       tasks_sha256=file_sha(ROOT / "eval_protocol/tasks_750.json"),
                       seed=EVAL_SEED, g=g_val, wc=wc_val, wall_sec=round(time.time() - t0, 1))
    assert validate_raw_meta(meta)
    s = run_measure_v23(arm_name, device=device)
    export_raw_pngs(OUT / f"raw/{arm_name}", OUT / f"png_750/{arm_name}", max_workers=8)
    c = s["cross600"]
    return {"arm": arm_name, "csd_top1": round(c["csd_top1"], 5), "dino_s": round(c["dino_s"], 5),
            "dino_c": round(c["dino_c"], 5), "style_affinity": round(c["style_affinity"], 5),
            "lpips": round(c["lpips"], 5), "same_top1": round(s["same150"]["csd_top1"], 5)}


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
    z_prism1 = compute_prism1_anchor_750(bundle, tasks, z_src_750, get_test750_y16_and_d16(bundle)[1], device=args.device)
    priors_cpu = {k: v.cpu() if torch.is_tensor(v) else v for k, v in bundle["fnd"]["priors"].items()}
    tid_all = torch.tensor([t["target_style_id"] for t in tasks])
    z_chroma = apply_latent_chroma_shift(z_src_750.float(), tid_all, priors_cpu, alpha=0.25)

    y16_750, d16_750 = get_test750_y16_and_d16(bundle)
    eps_750 = torch.randn(z_src_750.shape, generator=torch.Generator().manual_seed(EVAL_SEED))
    from src.utils import load_s10_decoder
    dec = load_s10_decoder(args.device)
    ctx = {"tasks": tasks, "z_src_750": z_src_750, "y16_750": y16_750, "d16_750": d16_750,
           "eps_750": eps_750, "dec": dec}

    results = []
    # Peak1500 × {T4, PRISM-1, Chroma-Shift} × {hiS(0.45,1.0), hiC(0.50,1.5)}
    core_pk = PrismCore(bundle, bank, PEAK_CKPT, device=args.device)
    for anc_name, anc_t in (("A0_pk_t4", z_t4), ("A1_pk_prism1", z_prism1), ("A2_pk_chroma", z_chroma)):
        results.append(run_anchor_arm(core_pk, PEAK_CKPT, ctx, f"{anc_name}_hiS", anc_t, 0.45, 1.0, device=args.device))
        results.append(run_anchor_arm(core_pk, PEAK_CKPT, ctx, f"{anc_name}_hiC", anc_t, 0.50, 1.5, device=args.device))
    del core_pk
    torch.cuda.empty_cache()

    # T1 × {T4, PRISM-1} × {hiS, hiC}
    core_t1 = PrismCore(bundle, bank, T1_CKPT, device=args.device)
    results.append(run_anchor_arm(core_t1, T1_CKPT, ctx, "A3_t1_t4_hiC", z_t4, 0.50, 1.5, device=args.device))
    results.append(run_anchor_arm(core_t1, T1_CKPT, ctx, "A3_t1_prism1_hiS", z_prism1, 0.45, 1.0, device=args.device))
    results.append(run_anchor_arm(core_t1, T1_CKPT, ctx, "A3_t1_prism1_hiC", z_prism1, 0.50, 1.5, device=args.device))

    save_json(OUT / "p1_fast_anchor.json", {"updated_utc": utc(), "results": results})
    print("\n=== V23 P1 第一组（快速锚点整合）汇总 ===", flush=True)
    for r in results:
        print(f"{r['arm']:22s}: Top1={r['csd_top1']*100:5.2f}% DINO-S={r['dino_s']:.4f} DINO-C={r['dino_c']:.4f} LPIPS={r['lpips']:.4f}", flush=True)


if __name__ == "__main__":
    main()
