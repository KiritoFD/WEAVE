r"""ICME 收束第 1 步: 未剪枝 Full Decoder 同域微调 (D1/D2) 解码 60% 闭集 PRISM-Flow latent.

堵死质疑 3 (Table II Part B & Fig. 2b):
  1. 复用 V26-L2 (l2_decoder_matched.py) 完全相同的确定性配方 (seed=20266200,
     1275 张 d5 真图重建目标, 1000 steps, batch=1, lr=5e-5; D1 bf16+ckpt, D2 fp32),
     **首次落盘保存 D1_full_ft.pt / D2_s10_ft.pt**;
  2. 对 60.50% 闭集旗舰 V20_D2_core_slf_bal (runs/v20/raw/V20_D2_core_slf_bal 的
     750 输出 z_final) 分别以 {D0_full_frozen, D1_full_ft, D2_s10_ft, S10_as_is} 解码;
  3. 四裁判隔离进程评测 Cross600 (CSD Top-1/Margin/DINO-C/DINO-S/LPIPS/Same150)
     + 150 源图聚类配对 Bootstrap 95% CI.

用法:  python -m exp.v27.icme_step1_decoder_closedset --pool ../pseudo_gt_24h/orig_mean_fp16
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import torch

from exp.v26.gpu4090.l2_decoder_matched import (
    L2_STEPS,
    _build_ft_targets,
    _ft,
    _load_full_vae,
    build_or_load_bundle,
    wrap_decoder,
)
from exp.v27 import OUT, ROOT
from src.eval.runner import run_measure_subprocess
from src.utils import load_s10_decoder
from src.utils.raw_io import RawWriter
from src.utils.runtime import sanitize_cuda_env, save_json, seed_all, utc

V20_D2_RAW = ROOT / "runs/v20/raw/V20_D2_core_slf_bal"
CKPT_DIR = OUT / "icme_dec"


def _cluster_paired_ci(measured_dir_a: Path, measured_dir_b: Path,
                       n_boot: int = 2000, seed: int = 20261123) -> dict:
    """150 源图聚类配对 Bootstrap 95% CI (A − B, 跨 600 cross 任务)."""
    ra = {r["task_id"]: r for r in json.loads((measured_dir_a / "eval.json").read_text(encoding="utf-8"))["rows"] if r["is_cross"]}
    rb = {r["task_id"]: r for r in json.loads((measured_dir_b / "eval.json").read_text(encoding="utf-8"))["rows"] if r["is_cross"]}
    by_stem: dict[str, list[str]] = {}
    for tid, r in ra.items():
        by_stem.setdefault(r["src_stem"], []).append(tid)
    stems = sorted(by_stem)
    g = torch.Generator().manual_seed(seed)
    res = {}
    for key in ("target_top1", "target_margin", "lpips", "dino_s", "dino_c"):
        diffs = []
        for _ in range(n_boot):
            pick = torch.randint(0, len(stems), (len(stems),), generator=g).tolist()
            tids = [t for si in pick for t in by_stem[stems[si]]]
            va = sum(float(ra[t][key]) for t in tids) / len(tids)
            vb = sum(float(rb[t][key]) for t in tids) / len(tids)
            diffs.append(va - vb)
        d = torch.tensor(diffs)
        q = torch.quantile(d, torch.tensor([0.025, 0.975]))
        res[key] = {"mean_diff": round(float(d.mean()), 5),
                    "ci95": [round(float(q[0]), 5), round(float(q[1]), 5)]}
    return res


def run_step1(pool: str, device: str = "cuda") -> dict:
    seed_all(20266200)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    d1_p, d2_p = CKPT_DIR / "D1_full_ft.pt", CKPT_DIR / "D2_s10_ft.pt"

    bundle = build_or_load_bundle(pool, "cpu")
    tr_cache = bundle["v18_cache"]["train"]
    train_latents = tr_cache["z"][:2400].clone()
    ft_idx, ft_imgs = _build_ft_targets(tr_cache["ids"])
    assert len(ft_idx) >= 512, f"d5 真图不足: {len(ft_idx)}"
    print(f"[Step1] FT 目标: {len(ft_idx)} 张 d5 真图", flush=True)

    s10_asis = load_s10_decoder(device)
    for p in s10_asis.parameters():
        p.requires_grad_(False)

    full_ref = _load_full_vae(device)
    full = _load_full_vae(device)
    if d1_p.exists():
        print(f"[Step1] Reusing {d1_p}", flush=True)
        full.load_state_dict(torch.load(d1_p, map_location=device, weights_only=False))
    else:
        try:
            full.enable_gradient_checkpointing()
        except Exception:
            pass
        for n, p in full.named_parameters():
            p.requires_grad_(n.startswith("decoder") or n.startswith("post_quant_conv"))
        _ft(full, wrap_decoder("full_ft", full, device, kind="full"),
            train_latents, ft_imgs, ft_idx, device=device, tag="D1_full_ft")
        for p in full.parameters():
            p.requires_grad_(False)
        torch.save(full.state_dict(), d1_p)
        print(f"[Step1] Saved {d1_p}", flush=True)

    s10_ft = load_s10_decoder(device).to(torch.float32)
    if d2_p.exists():
        print(f"[Step1] Reusing {d2_p}", flush=True)
        s10_ft.load_state_dict(torch.load(d2_p, map_location=device, weights_only=False))
    else:
        _ft(s10_ft, wrap_decoder("s10_ft", s10_ft, device, kind="s10", enable_grad=True, dtype=torch.float32),
            train_latents, ft_imgs, ft_idx, device=device, tag="D2_s10_ft", amp=False)
        for p in s10_ft.parameters():
            p.requires_grad_(False)
        torch.save(s10_ft.state_dict(), d2_p)
        print(f"[Step1] Saved {d2_p}", flush=True)
    torch.cuda.empty_cache()

    dec_wraps = {
        "D0_full_frozen": wrap_decoder("D0", full_ref, device, kind="full"),
        "D1_full_ft": wrap_decoder("D1", full, device, kind="full"),
        "D2_s10_ft": wrap_decoder("D2", s10_ft, device, kind="s10", dtype=torch.float32),
        "S10_as_is": wrap_decoder("S10", s10_asis, device, kind="s10"),
    }

    lat_files = sorted(glob.glob(str(V20_D2_RAW / "batch_*.pt")))
    assert lat_files, f"V20_D2 raw latents missing: {V20_D2_RAW}"
    results = {"task": "ICME Step1: Closed-set V20_D2 (60.5%) decoded across 4 decoders",
               "source_latents": str(V20_D2_RAW), "steps": L2_STEPS, "n_ft_images": len(ft_idx),
               "updated_utc": utc(), "arms": {}, "paired_ci": {}}

    with torch.no_grad():
        for name, fn in dec_wraps.items():
            arm = f"ICME_V20D2_{name}"
            m_dir = OUT / "measured" / arm
            if (m_dir / "eval.json").exists():
                s = json.loads((m_dir / "summary_v27_dec.json").read_text(encoding="utf-8")) \
                    if (m_dir / "summary_v27_dec.json").exists() else \
                    run_measure_subprocess(OUT / "raw" / arm, m_dir, arm_name=arm, device=device)
            else:
                wr = RawWriter(arm, raw_root=OUT / "raw", kind="v27")
                for f in lat_files:
                    d = torch.load(f, map_location="cpu", weights_only=False)
                    rgb = fn(d["z_final"].float().to(device))
                    wr.append(d["ids"], rgb.cpu(), z_final=d["z_final"].cpu())
                wr.finish()
                s = run_measure_subprocess(OUT / "raw" / arm, m_dir, arm_name=arm, device=device)
                save_json(m_dir / "summary_v27_dec.json", s)
            c, sm = s["cross600"], s["same150"]
            results["arms"][name] = {
                "csd_top1": round(c["csd_top1"], 5),
                "csd_margin": round(c.get("target_margin", c.get("csd_margin", 0.0)), 5),
                "dino_s": round(c["dino_s"], 5), "dino_c": round(c["dino_c"], 5),
                "lpips": round(c["lpips"], 5), "same150_top1": round(sm["csd_top1"], 5),
            }
            print(f"[Step1 EVAL] {name:16s} | {results['arms'][name]}", flush=True)

    for cmp_name in ("D0_full_frozen", "D1_full_ft", "D2_s10_ft"):
        results["paired_ci"][f"S10_minus_{cmp_name}"] = _cluster_paired_ci(
            OUT / "measured" / "ICME_V20D2_S10_as_is",
            OUT / "measured" / f"ICME_V20D2_{cmp_name}",
        )
    out_p = OUT / "icme_step1_decoder_closedset.json"
    save_json(out_p, results)
    print(f"[Step1] DONE → {out_p}", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    sanitize_cuda_env()
    run_step1(args.pool, device=args.device)


if __name__ == "__main__":
    main()
