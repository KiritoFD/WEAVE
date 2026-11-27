"""RTX 3060 / 4090 Comprehensive Performance Profiler & Benchmark Harness for CLEAVE V35."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from exp.deploy_v35 import BUNDLE_DEFAULT_DIR, E_BENCH_KNOBS, OPT_OUT_DIR, P_BENCH_KNOBS
from exp.deploy_v35.parity import load_rgb01_from_rel, make_deterministic_eps, select_stratified_tasks
from exp.deploy_v35.pipeline import CleaveDeployPipeline, make_optimized_pipeline
from src.utils.runtime import REPO as ROOT, save_json, utc


def query_nvidia_smi() -> dict[str, Any]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,memory.used,temperature.gpu,clocks.sm,clocks.mem,power.draw,power.limit",
        "--format=csv,noheader,nounits",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=10)
        parts = [p.strip() for p in r.stdout.strip().splitlines()[0].split(",")]
        return {
            "name": parts[0],
            "driver_version": parts[1],
            "vram_total_mb": float(parts[2]),
            "vram_used_mb": float(parts[3]),
            "temp_c": float(parts[4]),
            "sm_clock_mhz": float(parts[5]),
            "mem_clock_mhz": float(parts[6]),
            "power_w": float(parts[7]) if parts[7] not in ("[N/A]", "N/A") else None,
            "power_limit_w": float(parts[8]) if parts[8] not in ("[N/A]", "N/A") else None,
        }
    except Exception as exc:
        return {"error": str(exc)}


def collect_env_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "utc": utc(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "gpu_torch_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "nvidia_smi": query_nvidia_smi(),
    }
    for pkg in ("tensorrt", "onnx", "onnxruntime", "triton"):
        try:
            m = __import__(pkg)
            info[pkg] = getattr(m, "__version__", "installed")
        except Exception:
            info[pkg] = None
    return info


def _summarize_times(ms_list: list[float], bs: int) -> dict[str, float]:
    arr = np.asarray(ms_list, dtype=np.float64)
    p50 = float(np.percentile(arr, 50))
    p95 = float(np.percentile(arr, 95))
    mean = float(np.mean(arr))
    return {
        "n_runs": len(ms_list),
        "batch_size": bs,
        "batch_p50_ms": p50,
        "batch_p95_ms": p95,
        "batch_mean_ms": mean,
        "per_img_p50_ms": p50 / bs,
        "per_img_p95_ms": p95 / bs,
        "per_img_mean_ms": mean / bs,
        "throughput_img_s": (bs * 1000.0) / p50,
    }


def _time_fn(fn, *, warmup: int, repeats: int) -> dict[str, Any]:
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    gpu_ms: list[float] = []
    wall_ms: list[float] = []
    for _ in range(repeats):
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        t0 = time.perf_counter()
        ev0.record()
        fn()
        ev1.record()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        gpu_ms.append(float(ev0.elapsed_time(ev1)))
        wall_ms.append((t1 - t0) * 1000.0)

    return {
        "gpu_ms": gpu_ms,
        "wall_ms": wall_ms,
        "peak_alloc_mb": float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)),
        "peak_reserved_mb": float(torch.cuda.max_memory_reserved() / (1024.0 * 1024.0)),
        "nvidia_smi_after": query_nvidia_smi(),
    }


def _make_batch_inputs(bs: int, tasks: list[dict[str, Any]], device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    picked = [tasks[i % len(tasks)] for i in range(bs)]
    rgb = torch.stack([load_rgb01_from_rel(t["source_rel_path"]) for t in picked])
    u8_pinned = (rgb.clamp(0.0, 1.0) * 255.0).round().byte().pin_memory()
    tids = torch.tensor([int(t["target_style_id"]) for t in picked], dtype=torch.long, device=device)
    eps = torch.stack([make_deterministic_eps(t["task_id"]) for t in picked]).to(device)
    return u8_pinned, rgb.to(device), tids, eps


def run_batch_sweep(
    pipe: CleaveDeployPipeline,
    mode: str,
    *,
    batch_sizes: tuple[int, ...] = (1, 2, 4, 8, 12, 16),
    e_lam: float = 0.60,
    p_knob: float = 1.00,
    host_e2e: bool = True,
) -> list[dict[str, Any]]:
    tasks = select_stratified_tasks(16)
    rows: list[dict[str, Any]] = []
    for bs in batch_sizes:
        warmup = 6 if bs == 1 else 4
        repeats = 20 if bs <= 2 else (12 if bs <= 4 else 8)
        if hasattr(pipe, "_compiled") and isinstance(pipe._compiled, dict):
            pipe._compiled.clear()
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        try:
            u8_host, rgb_gpu, tids, eps = _make_batch_inputs(bs, tasks, pipe.device)
            if not host_e2e:
                cached_src = pipe.prepare_source(rgb_gpu, need_y16=(mode == "e_to_p"))
            else:
                cached_src = None
                out_pinned = torch.empty((bs, 3, 512, 512), dtype=torch.uint8, pin_memory=True)

            pipe.stats.reset()

            def _step():
                if host_e2e:
                    rgb_in = u8_host.to(pipe.device, non_blocking=True).float().div_(255.0)
                    if mode == "e_only":
                        out = pipe.run_e_only(rgb_in, tids, e_lam, eps, use_a3=False)
                    else:
                        out = pipe.run_e_to_p(rgb_in, tids, p_knob, eps)
                    out_u8 = (out.clamp(0.0, 1.0) * 255.0).round().byte()
                    out_pinned.copy_(out_u8, non_blocking=False)
                else:
                    if mode == "e_only":
                        _ = pipe.run_e_only(rgb_gpu, tids, e_lam, eps, use_a3=False, cached_src=cached_src)
                    else:
                        _ = pipe.run_e_to_p(rgb_gpu, tids, p_knob, eps, cached_src=cached_src)

            res = _time_fn(_step, warmup=warmup, repeats=repeats)
            n_total_iters = warmup + repeats
            stats_per_call = {k: v / n_total_iters for k, v in pipe.stats.as_dict().items()}
            rows.append({
                "mode": mode,
                "host_e2e": host_e2e,
                "batch_size": bs,
                "status": "ok",
                "gpu_timing": _summarize_times(res["gpu_ms"], bs),
                "wall_timing": _summarize_times(res["wall_ms"], bs),
                "peak_alloc_mb": res["peak_alloc_mb"],
                "peak_reserved_mb": res["peak_reserved_mb"],
                "nvidia_smi_after": res["nvidia_smi_after"],
                "stats_per_call": stats_per_call,
            })
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            rows.append({
                "mode": mode,
                "host_e2e": host_e2e,
                "batch_size": bs,
                "status": "OOM",
                "error": str(exc).splitlines()[0],
            })
            break
    return rows


def run_stage_breakdown(pipe: CleaveDeployPipeline, *, bs: int = 1, p_knob: float = 1.00) -> dict[str, Any]:
    tasks = select_stratified_tasks(max(bs, 8))
    u8_host, rgb_gpu, tids, eps = _make_batch_inputs(bs, tasks, pipe.device)
    out_pinned = torch.empty((bs, 3, 512, 512), dtype=torch.uint8, pin_memory=True)

    # Pre-warm once
    rgb_in = u8_host.to(pipe.device, non_blocking=False).float().div_(255.0)
    src = pipe.prepare_source(rgb_in, need_y16=True)
    _ = pipe.run_e_only(rgb_in, tids, 0.60, eps, cached_src=src)
    _ = pipe.run_e_to_p(rgb_in, tids, p_knob, eps, cached_src=src)
    torch.cuda.synchronize()

    def _measure_stage(fn, reps: int = 15):
        return _time_fn(fn, warmup=4, repeats=reps)

    t_h2d = _measure_stage(lambda: u8_host.to(pipe.device, non_blocking=False).float().div_(255.0))
    t_a0 = _measure_stage(lambda: pipe._encode_a0(rgb_gpu))
    z_s = pipe._encode_a0(rgb_gpu)
    t_cond = _measure_stage(lambda: pipe._extract_conditioning(rgb_gpu, need_y16=True, z_s=z_s))
    src = pipe._extract_conditioning(rgb_gpu, need_y16=True, z_s=z_s)

    def _ecore_a3():
        from exp.v34_frontier.stage_a_startpoint import reconstruct_analytic_draft
        ze = pipe._run_ecore(src["z_s"], src["c64"], src["q16"], tids, 0.60, eps)
        return reconstruct_analytic_draft(ze, tids, pipe.priors_a, mode="A3")

    t_ecore = _measure_stage(_ecore_a3)
    ze_s = _ecore_a3()
    from exp.v32_nodino.cnx_layout import match_cnx_k8_submode
    from exp.v33_semantic_control.models import integrate_v33_trajectory
    from exp.v35_final.common import make_v35_schedule
    from exp.deploy_v35 import P8_LOCKED_CFG
    u_st = match_cnx_k8_submode(src["g16"], tids, pipe.cnx_bank)
    kw = make_v35_schedule(p_knob, ws_mul=P8_LOCKED_CFG["ws_mul"], wc_mul=P8_LOCKED_CFG["wc_mul"], dtau0=P8_LOCKED_CFG["dtau0"])
    kw.pop("draft_lambda", None)

    def _pcore_only():
        if pipe.profile == "reference" and not pipe.use_host_grid and not pipe.use_torch_opt:
            return integrate_v33_trajectory(pipe.pcore, ze_s, eps, src["y16"], src["q16"], tids, u_st, steps=pipe.pcore_steps, solver="dpm2m", **kw)
        return pipe._integrate_zero_sync(
            ze_s, eps, src["y16"], src["q16"], tids, u_st,
            tau0=float(kw["tau0"]), ws=float(kw["ws"]), wc=float(kw["wc"]), wsp=float(kw["wsp"]),
            rescale_phi=float(kw["rescale_phi"]), gamma=float(kw["gamma"]), lscale=float(kw["lscale"]),
            steps=pipe.pcore_steps, startup=pipe.solver_startup,
        )

    t_pcore = _measure_stage(_pcore_only, reps=10 if bs <= 4 else 6)
    zo = _pcore_only()
    t_s10 = _measure_stage(lambda: pipe._decode_s10(zo))
    rgb_out = pipe._decode_s10(zo)
    t_d2h = _measure_stage(lambda: out_pinned.copy_((rgb_out.clamp(0.0, 1.0) * 255.0).round().byte(), non_blocking=False))

    stages = {
        "h2d": _summarize_times(t_h2d["gpu_ms"], bs),
        "a0_encode": _summarize_times(t_a0["gpu_ms"], bs),
        "cnx_struct_stiefel_y16": _summarize_times(t_cond["gpu_ms"], bs),
        "ecore_a3": _summarize_times(t_ecore["gpu_ms"], bs),
        "pcore_trajectory": _summarize_times(t_pcore["gpu_ms"], bs),
        "s10b_decode": _summarize_times(t_s10["gpu_ms"], bs),
        "d2h": _summarize_times(t_d2h["gpu_ms"], bs),
    }
    return {"batch_size": bs, "p_knob": p_knob, "stages": stages}


def run_torch_profiler_summary(pipe: CleaveDeployPipeline, mode: str, *, bs: int = 1, top_n: int = 15) -> list[dict[str, Any]]:
    from torch.profiler import ProfilerActivity, profile
    tasks = select_stratified_tasks(max(bs, 8))
    _, rgb_gpu, tids, eps = _make_batch_inputs(bs, tasks, pipe.device)

    for _ in range(3):
        if mode == "e_only":
            _ = pipe.run_e_only(rgb_gpu, tids, 0.60, eps)
        else:
            _ = pipe.run_e_to_p(rgb_gpu, tids, 1.00, eps)
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
        if mode == "e_only":
            _ = pipe.run_e_only(rgb_gpu, tids, 0.60, eps)
        else:
            _ = pipe.run_e_to_p(rgb_gpu, tids, 1.00, eps)
        torch.cuda.synchronize()

    avgs = prof.key_averages()
    total_cuda_us = sum(getattr(e, "self_device_time_total", getattr(e, "self_cuda_time_total", 0.0)) for e in avgs)
    total_cpu_us = sum(e.self_cpu_time_total for e in avgs)
    sorted_evs = sorted(
        avgs,
        key=lambda e: getattr(e, "self_device_time_total", getattr(e, "self_cuda_time_total", 0.0)),
        reverse=True,
    )[:top_n]
    rows: list[dict[str, Any]] = []
    for e in sorted_evs:
        cuda_us = float(getattr(e, "self_device_time_total", getattr(e, "self_cuda_time_total", 0.0)))
        cpu_us = float(e.self_cpu_time_total)
        rows.append({
            "key": e.key,
            "count": int(e.count),
            "self_cuda_ms": cuda_us / 1000.0,
            "cuda_pct": (100.0 * cuda_us / total_cuda_us) if total_cuda_us > 0 else 0.0,
            "self_cpu_ms": cpu_us / 1000.0,
            "cpu_pct": (100.0 * cpu_us / total_cpu_us) if total_cpu_us > 0 else 0.0,
        })
    return rows


def run_cold_start_once(bundle_dir: str, mode: str, profile: str = "reference") -> dict[str, float]:
    t0 = time.perf_counter()
    tasks = select_stratified_tasks(1)
    rgb = load_rgb01_from_rel(tasks[0]["source_rel_path"]).unsqueeze(0).cuda()
    eps = make_deterministic_eps(tasks[0]["task_id"]).unsqueeze(0).cuda()
    tid = torch.tensor([int(tasks[0]["target_style_id"])], dtype=torch.long, device="cuda")
    t_init0 = time.perf_counter()
    if profile == "reference":
        pipe = CleaveDeployPipeline(bundle_dir=Path(bundle_dir), load_pcore=(mode == "e_to_p"), profile="reference")
    else:
        pipe = make_optimized_pipeline(bundle_dir=Path(bundle_dir), load_pcore=(mode == "e_to_p"), compile_batches=(1,))
    torch.cuda.synchronize()
    t_load = (time.perf_counter() - t_init0) * 1000.0
    t_inf0 = time.perf_counter()
    if mode == "e_only":
        _ = pipe.run_e_only(rgb, tid, 0.60, eps)
    else:
        _ = pipe.run_e_to_p(rgb, tid, 1.00, eps)
    torch.cuda.synchronize()
    t_first = (time.perf_counter() - t_inf0) * 1000.0
    t_total = (time.perf_counter() - t0) * 1000.0
    return {"load_ms": t_load, "first_infer_ms": t_first, "total_ms": t_total}


def measure_process_cold(bundle_dir: Path, mode: str, n_runs: int = 3, *, profile: str = "reference") -> dict[str, Any]:
    runs = []
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    for _ in range(n_runs):
        r = subprocess.run(
            [sys.executable, "-m", "exp.deploy_v35.benchmark", "--cold-once", mode, "--cold-profile", profile, "--bundle-dir", str(bundle_dir)],
            capture_output=True, text=True, check=True, cwd=str(ROOT), env=env,
        )
        runs.append(json.loads(r.stdout.strip().splitlines()[-1]))
    return {
        "mode": mode,
        "profile": profile,
        "runs": runs,
        "load_median_ms": float(np.median([x["load_ms"] for x in runs])),
        "first_infer_median_ms": float(np.median([x["first_infer_ms"] for x in runs])),
        "total_median_ms": float(np.median([x["total_ms"] for x in runs])),
    }


def run_p0_baseline(bundle_dir: Path, out_path: Path) -> dict[str, Any]:
    env = collect_env_info()
    # 1. E-only pipeline (true isolation: load_pcore=False)
    pipe_e = CleaveDeployPipeline(bundle_dir=bundle_dir, load_pcore=False, profile="reference")
    e_host_sweep = run_batch_sweep(pipe_e, "e_only", host_e2e=True)
    e_hot_sweep = run_batch_sweep(pipe_e, "e_only", batch_sizes=(1, 4, 8), host_e2e=False)
    e_knobs = []
    tasks = select_stratified_tasks(4)
    _, rgb_gpu, tids, eps = _make_batch_inputs(1, tasks, pipe_e.device)
    for lam in (0.0, *E_BENCH_KNOBS):
        pipe_e.stats.reset()
        r = _time_fn(lambda l=lam: pipe_e.run_e_only(rgb_gpu, tids, l, eps), warmup=4, repeats=15)
        e_knobs.append({"lam": lam, "b1": _summarize_times(r["gpu_ms"], 1), "stats": {k: v / 19.0 for k, v in pipe_e.stats.as_dict().items()}})
    prof_e_b1 = run_torch_profiler_summary(pipe_e, "e_only", bs=1)
    prof_e_b4 = run_torch_profiler_summary(pipe_e, "e_only", bs=4)
    del pipe_e
    torch.cuda.empty_cache()

    # 2. Full E -> P pipeline
    pipe_p = CleaveDeployPipeline(bundle_dir=bundle_dir, load_pcore=True, profile="reference")
    p_host_sweep = run_batch_sweep(pipe_p, "e_to_p", host_e2e=True)
    p_hot_sweep = run_batch_sweep(pipe_p, "e_to_p", batch_sizes=(1, 4, 8), host_e2e=False)
    p_knobs = []
    for sk in P_BENCH_KNOBS:
        pipe_p.stats.reset()
        r = _time_fn(lambda s=sk: pipe_p.run_e_to_p(rgb_gpu, tids, s, eps), warmup=3, repeats=10)
        p_knobs.append({"s": sk, "b1": _summarize_times(r["gpu_ms"], 1), "stats": {k: v / 13.0 for k, v in pipe_p.stats.as_dict().items()}})
    stage_b1 = run_stage_breakdown(pipe_p, bs=1)
    stage_b4 = run_stage_breakdown(pipe_p, bs=4)
    prof_p_b1 = run_torch_profiler_summary(pipe_p, "e_to_p", bs=1)
    prof_p_b4 = run_torch_profiler_summary(pipe_p, "e_to_p", bs=4)
    del pipe_p
    torch.cuda.empty_cache()

    cold_e = measure_process_cold(bundle_dir, "e_only", n_runs=3)
    cold_p = measure_process_cold(bundle_dir, "e_to_p", n_runs=3)

    summary = {
        "env": env,
        "e_only": {"host_sweep": e_host_sweep, "hot_sweep": e_hot_sweep, "knobs_b1": e_knobs, "cold_start": cold_e, "profiler_b1": prof_e_b1, "profiler_b4": prof_e_b4},
        "e_to_p": {"host_sweep": p_host_sweep, "hot_sweep": p_hot_sweep, "knobs_b1": p_knobs, "stage_b1": stage_b1, "stage_b4": stage_b4, "cold_start": cold_p, "profiler_b1": prof_p_b1, "profiler_b4": prof_p_b4},
    }
    save_json(out_path, summary)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-dir", default=str(BUNDLE_DEFAULT_DIR))
    ap.add_argument("--out", default=str(OPT_OUT_DIR / "p0_baseline.json"))
    ap.add_argument("--cold-once", choices=["e_only", "e_to_p"], default="")
    ap.add_argument("--cold-profile", default="reference")
    args = ap.parse_args()

    if args.cold_once:
        print(json.dumps(run_cold_start_once(args.bundle_dir, args.cold_once, args.cold_profile)))
        return
    run_p0_baseline(Path(args.bundle_dir), Path(args.out))
    print("P0_BASELINE_DONE", args.out)


if __name__ == "__main__":
    main()
