"""V22-P2 T 矩阵评测：T0(slf_final)/T1–T5 各自 × fixed-gate wc11/wc14 两档 → DINO-C 回收验证。

用法 (4090, 训练完成后):
  python -m exp.v22.eval_tmatrix --pool ../pseudo_gt_24h/orig_mean_fp16
"""
from __future__ import annotations

import argparse
import time

import torch

from . import OUT, EVAL_SEED, ROOT, config_hash, export_raw_pngs, file_sha, git_sha, read_json, run_measure_v22, sanitize_cuda_env, save_json, utc, validate_raw_meta
from .core import CFGConfig, D5_SOLVER, GateConfig, PrismCore, latent_change_rate
from .t4 import build_or_load_t4_latents_750
from exp.v18 import tasks_and_sources
from exp.v19.fiber_duality import build_or_load_v19_foundation
from exp.v20.run_v20 import get_test750_y16_and_d16
from exp.v20.submode_patchswd import build_or_load_core_knn_bank
from src.utils.protocol import pick_wrong_cross_tid  # noqa: F401
from src.utils.runtime import REPO as ROOT_  # noqa: F401

BASE_CKPT = ROOT / "runs/v20/slf_flow/slf_final.pt"


def eval_t_ckpt(core: PrismCore, ckpt_path, tag: str, ctx: dict, device: str = "cuda") -> list[dict]:
    """单个 T ckpt × fixed-gate wc11/wc14 两档评测。"""
    tasks = ctx["tasks"]
    z_src_750 = ctx["z_src_750"]
    sid_all = torch.tensor([t["source_style_id"] for t in tasks])
    tid_all = torch.tensor([t["target_style_id"] for t in tasks])
    same_all = sid_all == tid_all
    anchor_750 = ctx["z_t4"]
    eps_750 = ctx["eps_750"]
    y16_750, d16_750 = ctx["y16_750"], ctx["d16_750"]
    dec = ctx["dec"]

    gate = GateConfig(mode="fixed", fixed_g=0.5)
    results = []
    for wc_val, wc_tag in ((1.1, "wc11"), (1.4, "wc14")):
        name = f"{tag}_{wc_tag}"
        cfg_hash = config_hash({"tag": tag, "wc": wc_val, "gate": "fixed0.5"})
        out_dir = OUT / "raw" / name
        if out_dir.joinpath("meta.json").exists():
            old = read_json(out_dir / "meta.json")
            if old.get("cfg_hash") == cfg_hash and validate_raw_meta(old):
                print(f"[V22 TEVAL] SKIP {name}", flush=True)
                results.append({"arm": name, "skipped": True})
                continue

        from src.utils.raw_io import RawWriter
        from src.utils import decode_s10

        wr = RawWriter(name, raw_root=OUT / "raw", kind="v22")
        t0 = time.time()
        cfg = CFGConfig(branches="full", rescale_phi=0.48)
        solver = dict(D5_SOLVER, tau_min=0.28, tau_hi=0.66, steps=10,
                      ws_fig=4.2, wsp_fig_late=2.4, wsp_abs=5.2,
                      wc_fig=wc_val, wc_abs=wc_val, lscale_abs=0.34)

        with torch.no_grad():
            for st in range(0, 750, 25):
                sl = slice(st, min(st + 25, 750))
                grp = tasks[sl]
                z_gen, _ = core.integrate_flow(
                    eps_750[sl], anchor_750[sl].to(core.device),
                    y16_750[sl], d16_750[sl], tid_all[sl].to(core.device),
                    gate=gate, cfg=cfg, solver=solver,
                )
                z_out = torch.where(same_all[sl].view(-1, 1, 1, 1).to(core.device),
                                    z_src_750[sl].to(core.device).float(), z_gen)
                wr.append([t["task_id"] for t in grp],
                          decode_s10(dec, z_out, batch=16).clamp(0.0, 1.0), z_out.cpu())

        meta_path = wr.finish(cfg_hash=cfg_hash, git_sha=git_sha(), ckpt_sha256=file_sha(ckpt_path),
                              tasks_sha256=file_sha(ROOT / "eval_protocol/tasks_750.json"),
                              seed=EVAL_SEED, tag=tag, wc=wc_val,
                              wall_sec=round(time.time() - t0, 1))
        assert validate_raw_meta(read_json(meta_path / "meta.json"))
        run_measure_v22(name, device=device)
        export_raw_pngs(OUT / f"raw/{name}", OUT / f"png_750/{name}", max_workers=8)
        s = read_json(OUT / f"measured/{name}/summary_v22.json")
        c = s["cross600"]
        results.append({
            "arm": name, "tag": tag, "wc": wc_val,
            "csd_top1": round(c["csd_top1"], 5),
            "dino_s": round(c["dino_s"], 5),
            "style_affinity": round(c["style_affinity"], 5),
            "dino_c": round(c["dino_c"], 5),
            "lpips": round(c["lpips"], 5),
        })
        print(f"[V22 TEVAL] {name}: Top1={c['csd_top1']*100:.2f}% DINO-S={c['dino_s']:.4f} "
              f"DINO-C={c['dino_c']:.4f} LPIPS={c['lpips']:.4f}", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    sanitize_cuda_env()
    from src.utils import load_s10_decoder

    bundle = build_or_load_v19_foundation(args.pool, device=args.device)
    bank = build_or_load_core_knn_bank(bundle, k_sub=8, core_per_style=180, device=args.device)

    tasks, z_src_750, _ = tasks_and_sources()
    z_t4, t4_src = build_or_load_t4_latents_750(bundle, device=args.device)
    y16_750, d16_750 = get_test750_y16_and_d16(bundle)
    eps_750 = torch.randn(z_src_750.shape, generator=torch.Generator().manual_seed(EVAL_SEED))
    dec = load_s10_decoder(args.device)

    ctx = {
        "tasks": tasks, "z_src_750": z_src_750, "y16_750": y16_750, "d16_750": d16_750,
        "eps_750": eps_750, "z_t4": z_t4, "dec": dec,
    }

    all_results = []
    # T0 基线 = slf_final
    core = PrismCore(bundle, bank, BASE_CKPT, device=args.device)
    all_results.extend(eval_t_ckpt(core, BASE_CKPT, "T0_slf_final", ctx, device=args.device))

    # T1–T5
    for arm in ("T1", "T2", "T3", "T4", "T5"):
        ckpt = OUT / "tmatrix" / f"slf_{arm}.pt"
        if not ckpt.exists():
            print(f"[V22 TEVAL] SKIP {arm} (ckpt missing)", flush=True)
            continue
        core = PrismCore(bundle, bank, ckpt, device=args.device)
        all_results.extend(eval_t_ckpt(core, ckpt, arm, ctx, device=args.device))
        del core
        torch.cuda.empty_cache()

    save_json(OUT / "tmatrix_eval.json", {"updated_utc": utc(), "results": all_results})

    print("\n=== V22 T 矩阵汇总（fixed gate, Core-Simple 配置） ===", flush=True)
    print(f"{'臂':16s} {'wc11 Top1':>10s} {'wc11 D-C':>10s} {'wc14 Top1':>10s} {'wc14 D-C':>10s}", flush=True)
    by_tag = {}
    for r in all_results:
        by_tag.setdefault(r["tag"], {})[r["wc"]] = r
    for tag, d in sorted(by_tag.items()):
        r11 = d.get(1.1, {})
        r14 = d.get(1.4, {})
        t11 = f"{r11.get('csd_top1', 0)*100:.2f}%" if r11 else "—"
        d11 = f"{r11.get('dino_c', 0):.4f}" if r11 else "—"
        t14 = f"{r14.get('csd_top1', 0)*100:.2f}%" if r14 else "—"
        d14 = f"{r14.get('dino_c', 0):.4f}" if r14 else "—"
        print(f"{tag:16s} {t11:>10s} {d11:>10s} {t14:>10s} {d14:>10s}", flush=True)


if __name__ == "__main__":
    main()
