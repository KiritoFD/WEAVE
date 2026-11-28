"""F7: RTX 3060 Deployment Audit (S10B-TRT, A0+S10B TRT, Full-TRT, Cold/Warm Split, Full Test750 95% CI, and Stiefel NS10 Audit).

Zero training, zero weight modifications. Spawns isolated subprocesses per GPU stage so VRAM never leaks
between TensorRT engine deserialization, torch.compile warmup, and the frozen referee stack.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from exp.deploy_v35 import BUNDLE_DEFAULT_DIR, E_GATE_KNOBS, OPT_OUT_DIR, P_BENCH_KNOBS
from exp.deploy_v35.eval_gate import generate_test750_arm, paired_cluster_bootstrap
from exp.deploy_v35.parity import load_rgb01_from_rel, make_deterministic_eps, select_stratified_tasks
from exp.deploy_v35.pipeline import CleaveDeployPipeline, make_optimized_pipeline
from exp.deploy_v35.trt_runtime import TrtRunner
from exp.v21.stiefel_ref_cache import stiefel_polar_quotient
from src.eval.runner import run_measure_batch_subprocess
from src.utils.runtime import REPO, read_json, save_json, utc


F7_DIR = REPO / "runs/paper_audit/f7_3060"


def _prep_win_env() -> None:
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("PYTHONUTF8", "1")
    if os.name == "nt" and not os.environ.get("CC"):
        spec = importlib.util.find_spec("triton")
        if spec and spec.origin:
            cc = Path(spec.origin).parent / "runtime" / "tcc" / "tcc.exe"
            if cc.is_file():
                os.environ["CC"] = str(cc)


def _nvidia_smi_mem_mb() -> float | None:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=10,
        )
        return float(r.stdout.strip().splitlines()[0].strip())
    except Exception:
        return None


def _padded_call(runner: TrtRunner, *inputs: torch.Tensor, batch: int) -> torch.Tensor:
    n = inputs[0].shape[0]
    if n > batch:
        raise ValueError(f"Batch {n} exceeds fixed TRT shape {batch}")
    if n < batch:
        padded = [torch.cat([x, x[:1].expand(batch - n, *x.shape[1:])], dim=0) for x in inputs]
        return runner(*padded)[:n]
    return runner(*inputs)


def _install_trt_mode(pipe: CleaveDeployPipeline, batch: int, variant: str) -> dict[str, float]:
    """Attach requested TRT engines onto a pure-PyTorch Optimized pipeline and return load times (ms)."""
    trt_dir = OPT_OUT_DIR / "trt_engines"
    timings_ms: dict[str, float] = {}

    if variant in ("a0_s10_trt", "full_trt"):
        t0 = time.perf_counter()
        a0_r = TrtRunner(trt_dir / f"a0_b{batch}_fp16.plan", ["rgb"], ["z_s"])
        timings_ms["load_a0_trt_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
        pipe.fast_a0 = pipe.a0 = None

        def _enc_a0(x: torch.Tensor, _r=a0_r, _b=batch) -> torch.Tensor:
            pipe.stats.a0_calls += 1
            return _padded_call(_r, x.float(), batch=_b)

        pipe._encode_a0 = _enc_a0

    if variant in ("s10trt", "a0_s10_trt", "full_trt"):
        t0 = time.perf_counter()
        s10_r = TrtRunner(trt_dir / f"s10_b{batch}_fp16.plan", ["zo"], ["rgb_out"])
        timings_ms["load_s10_trt_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
        pipe.fast_s10 = pipe.s10 = None

        def _dec_s10(z: torch.Tensor, _r=s10_r, _b=batch) -> torch.Tensor:
            pipe.stats.s10_calls += 1
            return _padded_call(_r, z.float(), batch=_b)

        pipe._decode_s10 = _dec_s10

    if variant == "full_trt":
        t0 = time.perf_counter()
        ec_r = TrtRunner(trt_dir / f"ecore_b{batch}_fp16.plan", ["z_s", "c64", "q16", "tid", "lam", "eps"], ["ze"])
        timings_ms["load_ecore_trt_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
        pipe.fast_ecore = pipe.ecore = None

        def _run_ec(z_s, c64, q16, tid, lam, eps, _r=ec_r, _b=batch):
            pipe.stats.ecore_calls += 1
            lam_t = torch.full((z_s.shape[0],), float(lam), device=z_s.device, dtype=torch.float32)
            return _padded_call(_r, z_s.float(), c64.float(), q16.float(), tid.long(), lam_t, eps.float(), batch=_b)

        pipe._run_ecore = _run_ec

        if pipe.load_pcore:
            t0 = time.perf_counter()
            pc_r = TrtRunner(
                trt_dir / f"pcore_b{batch}_fp16.plan",
                ["z", "t_1b", "cb4", "gc1", "bc1", "gc2", "bc2"],
                ["v4b"],
            )
            timings_ms["load_pcore_trt_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
            orig_get = pipe._get_callable

            def _get_trt(kind: str, b_cur: int, fn_obj):
                if kind == "pcore" and b_cur == batch:
                    return lambda zz, t1b, cb4, gc1, bc1, gc2, bc2: pc_r(
                        zz.float(), t1b.float(), cb4.float(), gc1.float(), bc1.float(), gc2.float(), bc2.float()
                    )
                return orig_get(kind, b_cur, fn_obj)

            pipe._get_callable = _get_trt

    return timings_ms


@torch.inference_mode()
def bench_one(variant: str, mode: str, batch: int) -> dict[str, Any]:
    out_p = F7_DIR / "speed" / f"{variant}_{mode}_b{batch}.json"
    if out_p.exists():
        return read_json(out_p)

    t_init0 = time.perf_counter()
    pipe = make_optimized_pipeline(
        BUNDLE_DEFAULT_DIR,
        load_pcore=(mode == "p"),
        compile_mode="reduce-overhead",
        compile_batches=(batch,),
    )
    trt_load = _install_trt_mode(pipe, batch, variant) if variant != "optimized" else {}
    init_ms = (time.perf_counter() - t_init0) * 1000.0

    tasks = select_stratified_tasks(max(batch, 4))[:batch]
    cpu = torch.stack([load_rgb01_from_rel(t["source_rel_path"]) for t in tasks]).pin_memory()
    output = torch.empty_like(cpu, pin_memory=True)
    tid = torch.tensor([int(t["target_style_id"]) for t in tasks], device="cuda", dtype=torch.long)
    eps = torch.stack([make_deterministic_eps(t["task_id"]) for t in tasks], dim=0).cuda()

    def _step() -> None:
        rgb = cpu.cuda(non_blocking=True)
        y = pipe.run_e_only(rgb, tid, 0.60, eps) if mode == "e" else pipe.run_e_to_p(rgb, tid, 1.00, eps)
        output.copy_(y, non_blocking=True)

    # First call captures cold compile / first-execution latency separately
    t_first0 = time.perf_counter()
    _step()
    torch.cuda.synchronize()
    first_call_ms = (time.perf_counter() - t_first0) * 1000.0

    for _ in range(9):
        _step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    times: list[float] = []
    start = time.monotonic()
    while len(times) < 30 or time.monotonic() - start < 10.0:
        t0 = time.perf_counter()
        _step()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)

    times.sort()
    pipe.stats.reset()
    _step()
    torch.cuda.synchronize()
    med = float(times[len(times) // 2])
    res = {
        "variant": variant,
        "mode": mode,
        "batch": batch,
        "init_load_ms": round(init_ms, 2),
        "first_call_ms": round(first_call_ms, 2),
        "cold_plus_first_call_ms": round(init_ms + first_call_ms, 2),
        "trt_load_ms": trt_load,
        "p50_batch_ms": round(med, 3),
        "p95_batch_ms": round(float(times[int(0.95 * (len(times) - 1))]), 3),
        "ms_img": round(med / batch, 3),
        "ips": round(1000.0 * batch / med, 3),
        "samples": len(times),
        "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / (1024 ** 3), 4),
        "nvidia_smi_used_mb": _nvidia_smi_mem_mb(),
        "stats": pipe.stats.as_dict(),
        "hardware": torch.cuda.get_device_name(0),
        "scope": "warm resident host f32 RGB->RGB (H2D+D2H included; disk IO excluded) + explicit cold init/first-call split",
        "utc": utc(),
    }
    save_json(out_p, res)
    return res


def _arm_specs(variant: str) -> list[tuple[str, float, str, bool]]:
    specs: list[tuple[str, float, str, bool]] = []
    for k in E_GATE_KNOBS:
        specs.append(("e_only", k, f"{variant}_e_l{round(k * 100):03d}", False))
    for k in P_BENCH_KNOBS:
        specs.append(("e_to_p", k, f"{variant}_p_s{round(k * 100):03d}", False))
    specs.append(("e_to_p", 1.00, f"{variant}_p_s100_wrong", True))
    return specs


@torch.inference_mode()
def generate_variant_test750(variant: str, mode: str, batch: int = 4) -> None:
    raw_root = F7_DIR / "raw"
    tj = read_json(REPO / "eval_protocol/tasks_750.json")
    src = {r["source_rel_path"]: load_rgb01_from_rel(r["source_rel_path"]) for r in tj["sources"]}

    req = [(m, k, name, w) for m, k, name, w in _arm_specs(variant) if (mode == "e" and m == "e_only") or (mode == "p" and m == "e_to_p")]
    if all((raw_root / name / "meta.json").exists() for _, _, name, _ in req):
        print(f"[F7 gen {variant}:{mode}] all cached", flush=True)
        return

    pipe = make_optimized_pipeline(
        BUNDLE_DEFAULT_DIR,
        load_pcore=(mode == "p"),
        compile_mode="reduce-overhead",
        compile_batches=(batch,),
    )
    _install_trt_mode(pipe, batch, variant)
    for m, k, name, w in req:
        generate_test750_arm(
            pipe, name, raw_root, mode=m, knob=k, batch_size=batch, src_cpu_map=src, wrong_target=w,
        )


def measure_variant(variant: str, referee_batch: int = 4) -> dict[str, Any]:
    out_p = F7_DIR / f"{variant}_quality.json"
    if out_p.exists():
        return read_json(out_p)
    raw_root = F7_DIR / "raw"
    meas_root = F7_DIR / "measured"
    legacy = OPT_OUT_DIR / "test750_gate/measured"
    items: list[tuple[Path, Path, str]] = []
    tags: list[str] = []
    for _, _, name, w in _arm_specs(variant):
        items.append((raw_root / name, meas_root / name, name))
        if not w:
            tags.append(name.removeprefix(f"{variant}_"))

    summaries = run_measure_batch_subprocess(items, write_png=False, micro_batch=referee_batch)
    comparisons = {}
    for tag in tags:
        comparisons[tag] = paired_cluster_bootstrap(
            meas_root / f"{variant}_{tag}" / "eval.json",
            legacy / f"optimized_{tag}" / "eval.json",
        )
    wrong_ctrl = summaries.pop(f"{variant}_p_s100_wrong")
    rep = {
        "variant": variant,
        "utc": utc(),
        "summaries": summaries,
        "comparisons": comparisons,
        "wrong_target_control": wrong_ctrl,
        "all_knobs_pass": all(v["non_inferiority_pass"] for v in comparisons.values()),
    }
    save_json(out_p, rep)
    return rep


@torch.no_grad()
def run_stiefel_ns_audit(max_wikiart: int = 2000) -> dict[str, Any]:
    """Compare NS10 Stiefel (FastConditioner FP16 CNX) vs exact FP32 SVD across test150 + WikiArt."""
    out_p = F7_DIR / "stiefel_ns10_audit.json"
    if out_p.exists():
        return read_json(out_p)

    from PIL import Image
    pipe = make_optimized_pipeline(BUNDLE_DEFAULT_DIR, load_pcore=False, compile_mode="")
    fc = pipe.fast_cond.eval()
    tj = read_json(REPO / "eval_protocol/tasks_750.json")
    entries: list[tuple[str, Path]] = [
        (s["source_stem"], REPO / s["source_rel_path"]) for s in tj["sources"]
    ]
    wdir = REPO / "datasets/wikiart_new_512"
    if wdir.exists():
        wfiles = sorted(wdir.rglob("*.jpg"))
        step = max(1, len(wfiles) // max_wikiart)
        for p in wfiles[::step][:max_wikiart]:
            entries.append((f"wikiart::{p.stem}", p))

    gram_frob_ns, gram_spec_ns, gram_frob_svd = [], [], []
    cos_q, rmse_q, cond_ratio = [], [], []
    meta_records: list[tuple[str, float, float, float, float]] = []
    eye32 = torch.eye(32, device="cuda", dtype=torch.float32)

    for st in range(0, len(entries), 16):
        sl = entries[st:st + 16]
        imgs = []
        for _, path in sl:
            with Image.open(path) as im:
                arr = np.asarray(im.convert("RGB").resize((512, 512), Image.Resampling.BICUBIC), dtype=np.float32) / 255.0
            imgs.append(torch.from_numpy(arr).permute(2, 0, 1))
        rgb = torch.stack(imgs, 0).cuda()
        z_dummy = torch.zeros((len(sl), 4, 64, 64), device="cuda", dtype=torch.float32)
        _, g16_fp16, q_ns, _ = fc(rgb, z_dummy)
        q_svd = stiefel_polar_quotient(g16_fp16.float(), pipe.w32_st)

        # Scaled back by 1/16 to orthonormal matrix R in R^{32 x 256}
        r_ns = q_ns.flatten(2).float() / 16.0
        r_svd = q_svd.flatten(2).float() / 16.0
        fl = torch.bmm(fc.w32_t.unsqueeze(0).expand(len(sl), -1, -1), g16_fp16.flatten(2).float())
        fl = fl - fl.mean(dim=-1, keepdim=True)
        svs = torch.linalg.svdvals(fl)
        cr = (svs[:, 0] / svs[:, -1].clamp_min(1e-8)).cpu().tolist()

        for i in range(len(sl)):
            gn = r_ns[i] @ r_ns[i].t() - eye32
            gs = r_svd[i] @ r_svd[i].t() - eye32
            fn = float(torch.linalg.matrix_norm(gn, ord="fro").item())
            sn = float(torch.linalg.matrix_norm(gn, ord=2).item())
            fs = float(torch.linalg.matrix_norm(gs, ord="fro").item())
            cs = float(torch.nn.functional.cosine_similarity(q_ns[i].flatten(), q_svd[i].flatten(), dim=0).item())
            rm = float((q_ns[i] - q_svd[i]).pow(2).mean().sqrt().item())
            gram_frob_ns.append(fn)
            gram_spec_ns.append(sn)
            gram_frob_svd.append(fs)
            cos_q.append(cs)
            rmse_q.append(rm)
            cond_ratio.append(float(cr[i]))
            meta_records.append((sl[i][0], fn, sn, cs, float(cr[i])))

    meta_records.sort(key=lambda x: -x[1])
    arr_fn = np.asarray(gram_frob_ns)
    arr_sn = np.asarray(gram_spec_ns)
    arr_cs = np.asarray(cos_q)
    arr_rm = np.asarray(rmse_q)
    arr_cr = np.asarray(cond_ratio)
    rep = {
        "utc": utc(),
        "n_total": len(entries),
        "n_test150": len(tj["sources"]),
        "n_wikiart_sample": len(entries) - len(tj["sources"]),
        "ns10_gram_frob_err": {
            "mean": float(arr_fn.mean()), "p50": float(np.median(arr_fn)),
            "p95": float(np.percentile(arr_fn, 95)), "p99": float(np.percentile(arr_fn, 99)), "max": float(arr_fn.max()),
        },
        "ns10_gram_spectral_err": {
            "mean": float(arr_sn.mean()), "p95": float(np.percentile(arr_sn, 95)), "max": float(arr_sn.max()),
        },
        "svd_gram_frob_err_mean": float(np.mean(gram_frob_svd)),
        "q16_cosine_vs_svd": {"mean": float(arr_cs.mean()), "p05": float(np.percentile(arr_cs, 5)), "min": float(arr_cs.min())},
        "q16_rmse_vs_svd": {"mean": float(arr_rm.mean()), "p95": float(np.percentile(arr_rm, 95)), "max": float(arr_rm.max())},
        "sigma_ratio_s1_over_s32": {"mean": float(arr_cr.mean()), "p95": float(np.percentile(arr_cr, 95)), "max": float(arr_cr.max())},
        "worst_5_outliers": [
            {"stem": s, "gram_frob": round(fn, 6), "gram_spec": round(sn, 6), "cos_vs_svd": round(cs, 6), "cond_ratio": round(cr, 2)}
            for s, fn, sn, cs, cr in meta_records[:5]
        ],
    }
    save_json(out_p, rep)
    return rep


def _spawn(args_list: list[str]) -> None:
    cmd = [sys.executable, "-u", "-m", "exp.paper_audit.f7_3060_audit", *args_list]
    print("[F7 spawn]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def run_all() -> None:
    F7_DIR.mkdir(parents=True, exist_ok=True)
    for var in ("optimized", "s10trt", "a0_s10_trt", "full_trt"):
        for mode in ("e", "p"):
            for bs in (1, 4):
                _spawn(["--stage", "bench", "--variant", var, "--mode", mode, "--batch", str(bs)])
    for var in ("s10trt", "full_trt"):
        _spawn(["--stage", "gen", "--variant", var, "--mode", "e", "--batch", "4"])
        _spawn(["--stage", "gen", "--variant", var, "--mode", "p", "--batch", "4"])
        _spawn(["--stage", "measure", "--variant", var])
    _spawn(["--stage", "stiefel"])

    summary = {
        "utc": utc(),
        "speed": {p.stem: read_json(p) for p in sorted((F7_DIR / "speed").glob("*.json"))},
        "s10trt_quality": read_json(F7_DIR / "s10trt_quality.json"),
        "full_trt_quality": read_json(F7_DIR / "full_trt_quality.json"),
        "stiefel_ns10_audit": read_json(F7_DIR / "stiefel_ns10_audit.json"),
    }
    save_json(F7_DIR / "f7_summary.json", summary)
    print("[F7 DONE] saved", F7_DIR / "f7_summary.json", flush=True)


def main() -> None:
    _prep_win_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", "bench", "gen", "measure", "stiefel"])
    ap.add_argument("--variant", default="s10trt", choices=["optimized", "s10trt", "a0_s10_trt", "full_trt"])
    ap.add_argument("--mode", default="p", choices=["e", "p"])
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--referee-batch", type=int, default=4)
    a = ap.parse_args()
    if a.stage == "all":
        run_all()
    elif a.stage == "bench":
        res = bench_one(a.variant, a.mode, a.batch)
        print(json.dumps(res, indent=2), flush=True)
    elif a.stage == "gen":
        generate_variant_test750(a.variant, a.mode, a.batch)
    elif a.stage == "measure":
        measure_variant(a.variant, a.referee_batch)
    elif a.stage == "stiefel":
        res = run_stiefel_ns_audit()
        print(json.dumps(res, indent=2), flush=True)


if __name__ == "__main__":
    main()
