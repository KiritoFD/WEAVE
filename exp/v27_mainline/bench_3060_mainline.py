"""RTX 3060 12GB 实机耗时评测: 对比 Base_6k (z_t4), Stage 1A (A_p1 + s6_csj), 与 Dir1 4步/2步无锚点 Reflow.

用法 (3060):
  python -m exp.v27_mainline.bench_3060_mainline --ckpt runs/v27_mainline/d1_reflow.pt
"""
from __future__ import annotations

import argparse
import time

import torch

from exp.v20.submode_patchswd import build_or_load_core_knn_bank
from exp.v21.stiefel_ref_cache import build_or_load_v21_cache
from exp.v27_mainline import OUT, ROOT
from exp.v27_mainline.dir1_reflow import SLFUNetGamma, integrate_dir1_reflow
from src.utils import decode_s10, load_s10_decoder
from src.utils.runtime import sanitize_cuda_env, save_json, utc


@torch.no_grad()
def _time_fn(fn, warmup: int = 3, iters: int = 12) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / float(iters) * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--ckpt", default=str(OUT / "d1_reflow_v2.pt"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    sanitize_cuda_env()
    dev = args.device

    pack = build_or_load_v21_cache(args.pool, device=dev)
    bundle, c21 = pack["bundle"], pack["c21"]
    bank = build_or_load_core_knn_bank(bundle, k_sub=8, core_per_style=180, device=dev)
    priors = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in bundle["fnd"]["priors"].items()}

    dec = load_s10_decoder(dev)
    model = SLFUNetGamma(base_dim=144, u_dim=128).to(dev).eval()
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu", weights_only=False)["ema"], strict=True)

    # Load frozen encoders for E2E RGB->RGB measurement (same load_a0_encoder as ICME Step 2b)
    from exp.v25.r14_e2e_benchmark import load_a0_encoder
    from src.utils._local_eval_refs import load_dino_model
    from src.utils.structure import extract_whitened_dino_layout
    a0 = load_a0_encoder(dev).eval()
    dino = load_dino_model(device=dev).eval()

    out = {"task": "RTX 3060 12GB benchmark for Dir1 Anchor-Free 4-Step Q32-Reflow", "updated_utc": utc(), "batches": {}}
    for B in (1, 16):
        rgb = torch.rand(B, 3, 512, 512, device=dev)
        z_src = torch.randn(B, 4, 64, 64, device=dev)
        eps = torch.randn(B, 4, 64, 64, device=dev)
        y16 = torch.randn(B, 1, 16, 16, device=dev)
        d16 = torch.randn(B, 384, 16, 16, device=dev)
        q32 = torch.randn(B, 32, 16, 16, device=dev)
        tid = torch.randint(0, 5, (B,), device=dev)
        sm = torch.zeros(B, dtype=torch.bool, device=dev)

        enc_ms = _time_fn(lambda: a0.encode((rgb * 2.0 - 1.0).half()).latent_dist.mean * 0.18215) / B
        dino_ms = _time_fn(lambda: extract_whitened_dino_layout(dino, rgb, batch=B, device=dev)) / B
        dec_ms = _time_fn(lambda: decode_s10(dec, z_src, batch=min(B, 8), output_cpu=False)) / B
        e4_ms = _time_fn(lambda: integrate_dir1_reflow(model, z_src, eps, y16, d16, q32, tid, priors, bank, steps=4, solver="euler", gamma_val=1.0, same_mask=sm)) / B
        e2_ms = _time_fn(lambda: integrate_dir1_reflow(model, z_src, eps, y16, d16, q32, tid, priors, bank, steps=2, solver="euler", gamma_val=1.0, same_mask=sm)) / B
        rk2_4_ms = _time_fn(lambda: integrate_dir1_reflow(model, z_src, eps, y16, d16, q32, tid, priors, bank, steps=4, solver="rk2", gamma_val=1.0, same_mask=sm)) / B

        out["batches"][f"b{B}"] = {
            "enc_ms": round(enc_ms, 2),
            "dino_ms": round(dino_ms, 2),
            "dec_s10_ms": round(dec_ms, 2),
            "flow_e4_ms": round(e4_ms, 2),
            "flow_e2_ms": round(e2_ms, 2),
            "flow_rk2_4_ms": round(rk2_4_ms, 2),
            "cached_e4_ms": round(e4_ms + dec_ms, 2),
            "cached_e2_ms": round(e2_ms + dec_ms, 2),
            "e2e_e4_ms": round(enc_ms + dino_ms + e4_ms + dec_ms, 2),
            "e2e_e2_ms": round(enc_ms + dino_ms + e2_ms + dec_ms, 2),
            "e2e_rk2_4_ms": round(enc_ms + dino_ms + rk2_4_ms + dec_ms, 2),
            "peak_mem_gb": round(torch.cuda.max_memory_allocated(dev) / (1024 ** 3), 2),
        }
        print(f"[3060 b={B}] {out['batches'][f'b{B}']}", flush=True)

    save_json(OUT / "bench_3060_mainline.json", out)
    print("[3060 Bench] DONE →", OUT / "bench_3060_mainline.json", flush=True)


if __name__ == "__main__":
    main()
