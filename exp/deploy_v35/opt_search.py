"""Pure-PyTorch Fixed-Shape Infra Ablation (P1/P2) & Fast-Solver (P3) Sweep on RTX 3060."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from exp.deploy_v35 import BUNDLE_DEFAULT_DIR, OPT_OUT_DIR
from exp.deploy_v35.benchmark import (
    _make_batch_inputs,
    _summarize_times,
    _time_fn,
    collect_env_info,
    measure_process_cold,
    run_batch_sweep,
    run_stage_breakdown,
    run_torch_profiler_summary,
    select_stratified_tasks,
)
from exp.deploy_v35.parity import compare_pipelines
from exp.deploy_v35.pipeline import CleaveDeployPipeline, make_optimized_pipeline
from src.utils.runtime import save_json


def _bench_quick(pipe: CleaveDeployPipeline, mode: str, bs: int, *, host_e2e: bool = True) -> dict[str, Any]:
    tasks = select_stratified_tasks(max(bs, 8))
    u8_host, rgb_gpu, tids, eps = _make_batch_inputs(bs, tasks, pipe.device)
    out_pinned = torch.empty((bs, 3, 512, 512), dtype=torch.uint8, pin_memory=True)
    cached_src = None if host_e2e else pipe.prepare_source(rgb_gpu, need_y16=(mode == "e_to_p"))
    warmup = 6 if bs <= 2 else 4
    repeats = 18 if bs == 1 else (12 if bs <= 4 else 8)
    pipe.stats.reset()

    def _step():
        if host_e2e:
            rgb_in = u8_host.to(pipe.device, non_blocking=True).float().div_(255.0)
            out = pipe.run_e_only(rgb_in, tids, 0.60, eps) if mode == "e_only" else pipe.run_e_to_p(rgb_in, tids, 1.00, eps)
            out_pinned.copy_((out.clamp(0.0, 1.0) * 255.0).round().byte(), non_blocking=False)
        else:
            _ = pipe.run_e_only(rgb_gpu, tids, 0.60, eps, cached_src=cached_src) if mode == "e_only" else pipe.run_e_to_p(rgb_gpu, tids, 1.00, eps, cached_src=cached_src)

    res = _time_fn(_step, warmup=warmup, repeats=repeats)
    n_tot = warmup + repeats
    return {
        "gpu": _summarize_times(res["gpu_ms"], bs),
        "wall": _summarize_times(res["wall_ms"], bs),
        "peak_alloc_mb": res["peak_alloc_mb"],
        "peak_reserved_mb": res["peak_reserved_mb"],
        "stats_per_call": {k: v / n_tot for k, v in pipe.stats.as_dict().items()},
    }


def run_torch_infra_sweep(bundle_dir: Path, out_path: Path) -> dict[str, Any]:
    ref_p = CleaveDeployPipeline(bundle_dir=bundle_dir, load_pcore=True, profile="reference")
    base_e_b1 = _bench_quick(ref_p, "e_only", 1)
    base_e_b4 = _bench_quick(ref_p, "e_only", 4)
    base_p_b1 = _bench_quick(ref_p, "e_to_p", 1)
    base_p_b4 = _bench_quick(ref_p, "e_to_p", 4)

    arm_specs: list[tuple[str, dict[str, Any]]] = [
        ("p1_host_grid", {"profile": "optimized", "use_host_grid": True, "use_torch_opt": False}),
        ("p2_eager_opt", {
            "profile": "optimized", "use_host_grid": True, "use_torch_opt": True,
            "a0_dtype": torch.float16, "pcore_dtype": torch.float16, "ecore_dtype": torch.float16,
            "channels_last_s10": False, "compile_mode": "",
        }),
        ("p2_compiled_opt", {
            "profile": "optimized", "use_host_grid": True, "use_torch_opt": True,
            "a0_dtype": torch.float16, "pcore_dtype": torch.float16, "ecore_dtype": torch.float16,
            "channels_last_s10": True, "compile_mode": "reduce-overhead", "compile_batches": (1, 4),
        }),
        ("p3_fast_euler6", {
            "profile": "fast", "use_host_grid": True, "use_torch_opt": True,
            "a0_dtype": torch.float16, "pcore_dtype": torch.float16, "ecore_dtype": torch.float16,
            "channels_last_s10": True, "compile_mode": "reduce-overhead", "compile_batches": (1, 4),
            "solver_startup": "euler", "pcore_steps": 6,
        }),
        ("p3_fast_euler5", {
            "profile": "fast", "use_host_grid": True, "use_torch_opt": True,
            "a0_dtype": torch.float16, "pcore_dtype": torch.float16, "ecore_dtype": torch.float16,
            "channels_last_s10": True, "compile_mode": "reduce-overhead", "compile_batches": (1, 4),
            "solver_startup": "euler", "pcore_steps": 5,
        }),
    ]

    results: dict[str, Any] = {
        "env": collect_env_info(),
        "reference": {"e_b1": base_e_b1, "e_b4": base_e_b4, "p_b1": base_p_b1, "p_b4": base_p_b4},
        "arms": {},
    }

    for name, kw in arm_specs:
        cand = CleaveDeployPipeline(bundle_dir=bundle_dir, load_pcore=True, **kw)
        # Note: compiled arms are warmed on B=1, B=4 inside _bench_quick; run parity at B=4
        par = compare_pipelines(ref_p, cand, n_samples=16, batch_size=4)
        e_b1 = _bench_quick(cand, "e_only", 1)
        e_b4 = _bench_quick(cand, "e_only", 4)
        p_b1 = _bench_quick(cand, "e_to_p", 1)
        p_b4 = _bench_quick(cand, "e_to_p", 4)
        results["arms"][name] = {
            "parity": {
                "e_no_a3_mae255": par["e_only_no_a3"]["rgb"]["mae_255"],
                "e_no_a3_ze_max": par["e_only_no_a3"]["ze"]["max_abs"],
                "p_s100_mae255": par["p_knobs"]["s100"]["rgb"]["mae_255"],
                "p_s100_zo_max": par["p_knobs"]["s100"]["zo"]["max_abs"],
                "p_s085_mae255": par["p_knobs"]["s085"]["rgb"]["mae_255"],
                "z_s_max": par["source_tensors"]["z_s"]["max_abs"],
                "q16_cos": par["source_tensors"]["q16"]["cos_sim"],
            },
            "e_b1": e_b1, "e_b4": e_b4, "p_b1": p_b1, "p_b4": p_b4,
            "speedup_e_b1": base_e_b1["gpu"]["batch_p50_ms"] / e_b1["gpu"]["batch_p50_ms"],
            "speedup_e_b4": base_e_b4["gpu"]["batch_p50_ms"] / e_b4["gpu"]["batch_p50_ms"],
            "speedup_p_b1": base_p_b1["gpu"]["batch_p50_ms"] / p_b1["gpu"]["batch_p50_ms"],
            "speedup_p_b4": base_p_b4["gpu"]["batch_p50_ms"] / p_b4["gpu"]["batch_p50_ms"],
        }
        print(
            f"[{name}] e_b1={e_b1['gpu']['per_img_p50_ms']:.2f}ms ({results['arms'][name]['speedup_e_b1']:.2f}x) "
            f"e_b4={e_b4['gpu']['per_img_p50_ms']:.2f}ms ({results['arms'][name]['speedup_e_b4']:.2f}x) "
            f"p_b1={p_b1['gpu']['per_img_p50_ms']:.2f}ms ({results['arms'][name]['speedup_p_b1']:.2f}x) "
            f"p_b4={p_b4['gpu']['per_img_p50_ms']:.2f}ms ({results['arms'][name]['speedup_p_b4']:.2f}x) | "
            f"mae255(E={results['arms'][name]['parity']['e_no_a3_mae255']:.2f}, P100={results['arms'][name]['parity']['p_s100_mae255']:.2f})",
            flush=True,
        )
        del cand
        torch.cuda.empty_cache()

    del ref_p
    torch.cuda.empty_cache()
    save_json(out_path, results)
    return results


def run_full_optimized_profile(bundle_dir: Path, out_path: Path) -> dict[str, Any]:
    """Full B=1,2,4,8,12,16 batch sweep, hot vs host, stage breakdown, profiler, and cold start for Optimized."""
    # E-only isolated
    pipe_e = make_optimized_pipeline(bundle_dir, load_pcore=False, compile_mode="reduce-overhead", compile_batches=(1, 2, 4, 8, 12, 16))
    e_host = run_batch_sweep(pipe_e, "e_only", batch_sizes=(1, 2, 4, 8, 12, 16), host_e2e=True)
    e_hot = run_batch_sweep(pipe_e, "e_only", batch_sizes=(1, 4, 8), host_e2e=False)
    prof_e_b4 = run_torch_profiler_summary(pipe_e, "e_only", bs=4)
    del pipe_e
    torch.cuda.empty_cache()

    # Full E->P
    pipe_p = make_optimized_pipeline(bundle_dir, load_pcore=True, compile_mode="reduce-overhead", compile_batches=(1, 2, 4, 8, 12, 16))
    p_host = run_batch_sweep(pipe_p, "e_to_p", batch_sizes=(1, 2, 4, 8, 12, 16), host_e2e=True)
    p_hot = run_batch_sweep(pipe_p, "e_to_p", batch_sizes=(1, 4, 8), host_e2e=False)
    st_b1 = run_stage_breakdown(pipe_p, bs=1)
    st_b4 = run_stage_breakdown(pipe_p, bs=4)
    prof_p_b1 = run_torch_profiler_summary(pipe_p, "e_to_p", bs=1)
    prof_p_b4 = run_torch_profiler_summary(pipe_p, "e_to_p", bs=4)
    del pipe_p
    torch.cuda.empty_cache()

    cold_e = measure_process_cold(bundle_dir, "e_only", n_runs=3, profile="optimized")
    cold_p = measure_process_cold(bundle_dir, "e_to_p", n_runs=3, profile="optimized")

    out = {
        "env": collect_env_info(),
        "e_only": {"host_sweep": e_host, "hot_sweep": e_hot, "profiler_b4": prof_e_b4, "cold_start": cold_e},
        "e_to_p": {"host_sweep": p_host, "hot_sweep": p_hot, "stage_b1": st_b1, "stage_b4": st_b4, "profiler_b1": prof_p_b1, "profiler_b4": prof_p_b4, "cold_start": cold_p},
    }
    save_json(out_path, out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["ablate", "full_opt"], default="ablate")
    ap.add_argument("--bundle-dir", default=str(BUNDLE_DEFAULT_DIR))
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.mode == "ablate":
        out_p = Path(args.out) if args.out else (OPT_OUT_DIR / "p2_torch_ablation.json")
        run_torch_infra_sweep(Path(args.bundle_dir), out_p)
        print("TORCH_ABLATION_DONE", out_p)
    else:
        out_p = Path(args.out) if args.out else (OPT_OUT_DIR / "p2_optimized_profile.json")
        run_full_optimized_profile(Path(args.bundle_dir), out_p)
        print("OPTIMIZED_PROFILE_DONE", out_p)


if __name__ == "__main__":
    main()
