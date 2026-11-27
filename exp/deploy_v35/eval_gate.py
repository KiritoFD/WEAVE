"""Full Test750 evaluation & paired source-clustered bootstrap non-inferiority gate for CLEAVE V35."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from exp.deploy_v35 import (
    BUNDLE_DEFAULT_DIR,
    E_GATE_KNOBS,
    NON_INFERIORITY_LIMITS,
    OPT_OUT_DIR,
    P_BENCH_KNOBS,
)
from exp.deploy_v35.parity import load_rgb01_from_rel
from exp.deploy_v35.pipeline import CleaveDeployPipeline, make_optimized_pipeline
from src.eval.runner import run_measure_batch_subprocess, summarize_measured_arm
from src.utils.protocol import EVAL_SEED, STYLES
from src.utils.raw_io import RawWriter, save_png_batch_fast
from src.utils.runtime import REPO as ROOT, read_json, save_json, utc


def _eval_eps_750(n: int = 750) -> torch.Tensor:
    return torch.randn((n, 4, 64, 64), generator=torch.Generator().manual_seed(EVAL_SEED), dtype=torch.float32)


def paired_cluster_bootstrap(
    cand_eval_json: Path | str,
    ref_eval_json: Path | str,
    *,
    n_boot: int = 2000,
    seed: int = 20261127,
) -> dict[str, Any]:
    """Paired bootstrap clustered over the 150 unique sources (exact test750 dependency structure)."""
    cand_rows = read_json(cand_eval_json)["rows"]
    ref_rows = read_json(ref_eval_json)["rows"]
    by_id_c = {r["task_id"]: r for r in cand_rows}
    by_id_r = {r["task_id"]: r for r in ref_rows}
    common_ids = sorted(set(by_id_c) & set(by_id_r))
    if len(common_ids) != 750:
        raise ValueError(f"Expected 750 matched task_ids, got {len(common_ids)}")

    # Group rows by source_stem (150 clusters, each with 4 cross + 1 same)
    clusters: dict[str, tuple[list[dict], list[dict]]] = {}
    for tid in common_ids:
        rc, rr = by_id_c[tid], by_id_r[tid]
        st = rc["src_stem"]
        if st not in clusters:
            clusters[st] = ([], [])
        clusters[st][0].append(rc)
        clusters[st][1].append(rr)

    stems = sorted(clusters.keys())
    n_src = len(stems)
    rng = np.random.default_rng(seed)

    def _agg(rs_c: list[dict], rs_r: list[dict], is_cross_flag: bool) -> dict[str, np.ndarray]:
        c_sub = [x for x in rs_c if bool(x["is_cross"]) == is_cross_flag]
        r_sub = [x for x in rs_r if bool(x["is_cross"]) == is_cross_flag]
        return {
            "csd_top1_pp": np.array([100.0 * (c["target_top1"] - r["target_top1"]) for c, r in zip(c_sub, r_sub)]),
            "dino_s": np.array([c["dino_s"] - r["dino_s"] for c, r in zip(c_sub, r_sub)]),
            "dino_c": np.array([c["dino_c"] - r["dino_c"] for c, r in zip(c_sub, r_sub)]),
            "lpips": np.array([c["lpips"] - r["lpips"] for c, r in zip(c_sub, r_sub)]),
            "affinity": np.array([
                0.5 * ((c["dino_s"] + c["clip_scores_5"][c["tgt_style_id"]]) - (r["dino_s"] + r["clip_scores_5"][r["tgt_style_id"]]))
                for c, r in zip(c_sub, r_sub)
            ]),
        }

    # Pre-aggregate per-source cluster sums for fast exact bootstrap
    cross_metrics = ("csd_top1_pp", "dino_s", "dino_c", "lpips", "affinity")
    src_cross_sum = {m: np.zeros(n_src, dtype=np.float64) for m in cross_metrics}
    src_cross_cnt = np.zeros(n_src, dtype=np.float64)
    src_same_sum = np.zeros(n_src, dtype=np.float64)
    src_same_cnt = np.zeros(n_src, dtype=np.float64)
    for i, st in enumerate(stems):
        rc_l, rr_l = clusters[st]
        dc = _agg(rc_l, rr_l, True)
        ds = _agg(rc_l, rr_l, False)
        src_cross_cnt[i] = float(len(dc["dino_c"]))
        for m in cross_metrics:
            src_cross_sum[m][i] = float(dc[m].sum())
        src_same_cnt[i] = float(len(ds["csd_top1_pp"]))
        src_same_sum[i] = float(ds["csd_top1_pp"].sum())

    idx = rng.integers(0, n_src, size=(n_boot, n_src))
    out: dict[str, Any] = {"n_sources": n_src, "n_boot": n_boot, "cross600": {}, "same150": {}}

    w_cross = src_cross_cnt[idx].sum(axis=1)
    for m in cross_metrics:
        point = float(src_cross_sum[m].sum() / src_cross_cnt.sum())
        boots = src_cross_sum[m][idx].sum(axis=1) / w_cross
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out["cross600"][m] = {"delta": point, "ci95_lo": float(lo), "ci95_hi": float(hi)}

    w_same = src_same_cnt[idx].sum(axis=1)
    pt_same = float(src_same_sum.sum() / src_same_cnt.sum())
    b_same = src_same_sum[idx].sum(axis=1) / w_same
    lo_s, hi_s = np.percentile(b_same, [2.5, 97.5])
    out["same150"]["csd_top1_pp"] = {"delta": pt_same, "ci95_lo": float(lo_s), "ci95_hi": float(hi_s)}

    lim = NON_INFERIORITY_LIMITS
    checks = {
        "dino_c": out["cross600"]["dino_c"]["delta"] >= lim["dino_c_min_delta"],
        "dino_s": out["cross600"]["dino_s"]["delta"] >= lim["dino_s_min_delta"],
        "csd_top1": out["cross600"]["csd_top1_pp"]["delta"] >= lim["csd_top1_pp_min_delta"],
        "same150": out["same150"]["csd_top1_pp"]["delta"] >= lim["same150_pp_min_delta"],
        "lpips": out["cross600"]["lpips"]["delta"] <= lim["lpips_max_delta"],
    }
    out["non_inferiority_pass"] = all(checks.values())
    out["checks"] = checks
    return out


def interp_matched_c(points: list[tuple[float, float, float]], c_targets: tuple[float, ...] = (0.50, 0.56, 0.62, 0.68)) -> dict[str, dict[str, float]]:
    """Piecewise-linear interpolation of (DINO-S, Top-1%) along DINO-C across knob sweeps."""
    pts = sorted(points, key=lambda x: x[0])  # sorted by dino_c ascending
    res: dict[str, dict[str, float]] = {}
    for ct in c_targets:
        if ct < pts[0][0] - 1e-4 or ct > pts[-1][0] + 1e-4:
            continue
        for (c0, s0, t0), (c1, s1, t1) in zip(pts, pts[1:]):
            if c0 - 1e-4 <= ct <= c1 + 1e-4:
                u = (ct - c0) / max(c1 - c0, 1e-8)
                u = min(1.0, max(0.0, u))
                res[f"C={ct:.2f}"] = {
                    "dino_s": float((1.0 - u) * s0 + u * s1),
                    "csd_top1_pct": float((1.0 - u) * t0 + u * t1),
                }
                break
    return res


@torch.inference_mode()
def generate_test750_arm(
    pipe: CleaveDeployPipeline,
    arm_name: str,
    raw_root: Path,
    *,
    mode: str,
    knob: float,
    batch_size: int = 4,
    src_cpu_map: dict[str, torch.Tensor] | None = None,
    save_visual_dir: Path | None = None,
    visual_indices: tuple[int, ...] = (0, 62, 188, 315, 441, 612),
) -> Path:
    """Run full test750 inference using raw RGB images from data/test/ and save via RawWriter."""
    arm_raw = raw_root / arm_name
    if (arm_raw / "meta.json").exists():
        print(f"[gen {arm_name}] cached, skipping", flush=True)
        return arm_raw

    tasks = read_json(ROOT / "eval_protocol/tasks_750.json")["tasks"]
    eps_750 = _eval_eps_750(len(tasks))
    writer = RawWriter(arm_name, raw_root, kind=f"cleave_{mode}", expected_tasks=len(tasks))
    pipe.source_cache._store.clear()

    vis_set = set(visual_indices)
    vis_u8: list[torch.Tensor] = []
    vis_ids: list[str] = []

    print(f"[gen {arm_name}] start mode={mode} knob={knob} bs={batch_size}", flush=True)
    for s in range(0, len(tasks), batch_size):
        e = min(len(tasks), s + batch_size)
        sl = tasks[s:e]
        real_n = len(sl)
        rgb = torch.stack([
            src_cpu_map[t["source_rel_path"]] if src_cpu_map is not None else load_rgb01_from_rel(t["source_rel_path"])
            for t in sl
        ]).to(pipe.device)
        tids = torch.tensor([int(t["target_style_id"]) for t in sl], dtype=torch.long, device=pipe.device)
        eps = eps_750[s:e].to(pipe.device)
        if real_n < batch_size and getattr(pipe, "compile_mode", ""):
            pad_n = batch_size - real_n
            rgb_in = torch.cat([rgb, rgb[:1].expand(pad_n, -1, -1, -1)], dim=0)
            tid_in = torch.cat([tids, tids[:1].expand(pad_n)], dim=0)
            eps_in = torch.cat([eps, eps[:1].expand(pad_n, -1, -1, -1)], dim=0)
        else:
            rgb_in, tid_in, eps_in = rgb, tids, eps
        if mode == "e_only":
            rgb_out = pipe.run_e_only(rgb_in, tid_in, knob, eps_in, use_a3=False)[:real_n]
        else:
            rgb_out = pipe.run_e_to_p(rgb_in, tid_in, knob, eps_in)[:real_n]
        ids = [t["task_id"] for t in sl]
        writer.append(ids, rgb_out)
        if save_visual_dir is not None:
            for j in range(s, e):
                if j in vis_set:
                    u8 = (rgb_out[j - s].detach().cpu().float() * 255.0).round().clamp(0, 255).byte()
                    vis_u8.append(u8)
                    vis_ids.append(tasks[j]["task_id"])

    writer.finish(arm=arm_name, mode=mode, knob=knob, profile=pipe.profile, utc=utc())
    if save_visual_dir is not None and vis_u8:
        save_png_batch_fast(torch.stack(vis_u8, 0), vis_ids, save_visual_dir / arm_name)
    return arm_raw


def generate_all_arms(
    bundle_dir: Path,
    out_dir: Path,
    *,
    profiles: tuple[str, ...] = ("reference", "optimized", "fast"),
    batch_size: int = 4,
    fast_steps: int = 5,
) -> list[tuple[Path, Path, str]]:
    raw_root = out_dir / "raw"
    meas_root = out_dir / "measured"
    vis_root = out_dir / "visual_samples"
    measure_items: list[tuple[Path, Path, str]] = []

    tj = read_json(ROOT / "eval_protocol/tasks_750.json")
    src_cpu_map = {s["source_rel_path"]: load_rgb01_from_rel(s["source_rel_path"]) for s in tj["sources"]}

    for prof in profiles:
        is_ref = (prof == "reference")
        if prof != "fast":
            if is_ref:
                pipe_e = CleaveDeployPipeline(bundle_dir=bundle_dir, load_pcore=False, profile="reference")
            else:
                pipe_e = make_optimized_pipeline(bundle_dir=bundle_dir, load_pcore=False, compile_batches=(batch_size,), profile=prof)
            assert pipe_e.pcore is None and pipe_e.fast_pcore is None, "E-only isolation violated"
            sample_rgb = next(iter(src_cpu_map.values())).unsqueeze(0).to(pipe_e.device)
            e0_out = pipe_e.run_e_only(sample_rgb, torch.tensor([1], device=pipe_e.device), 0.0, torch.zeros(1, 4, 64, 64, device=pipe_e.device))
            assert float((e0_out - sample_rgb).abs().max().item()) == 0.0, f"{prof} lam=0 exact identity failed"
            for lam in E_GATE_KNOBS:
                tag = f"{prof}_e_l{int(round(lam * 100)):03d}"
                rdir = generate_test750_arm(pipe_e, tag, raw_root, mode="e_only", knob=lam, batch_size=batch_size, src_cpu_map=src_cpu_map, save_visual_dir=vis_root)
                measure_items.append((rdir, meas_root / tag, tag))
            del pipe_e
            torch.cuda.empty_cache()

        if is_ref:
            pipe_p = CleaveDeployPipeline(bundle_dir=bundle_dir, load_pcore=True, profile="reference")
        else:
            pipe_p = make_optimized_pipeline(
                bundle_dir=bundle_dir, load_pcore=True, compile_batches=(batch_size,),
                solver_startup=("euler" if prof == "fast" else "midpoint"),
                pcore_steps=(fast_steps if prof == "fast" else 6),
                profile=prof,
            )
        sample_rgb = next(iter(src_cpu_map.values())).unsqueeze(0).to(pipe_p.device)
        id_out = pipe_p.run_e_to_p(sample_rgb, torch.tensor([1], device=pipe_p.device), 0.0, torch.zeros(1, 4, 64, 64, device=pipe_p.device))
        assert float((id_out - sample_rgb).abs().mean().item()) < 0.08, f"{prof} s=0 roundtrip mae too large"
        for sk in P_BENCH_KNOBS:
            tag = f"{prof}_p_s{int(round(sk * 100)):03d}"
            rdir = generate_test750_arm(pipe_p, tag, raw_root, mode="e_to_p", knob=sk, batch_size=batch_size, src_cpu_map=src_cpu_map, save_visual_dir=vis_root)
            measure_items.append((rdir, meas_root / tag, tag))
        del pipe_p
        torch.cuda.empty_cache()

    return measure_items


def measure_and_bootstrap(
    out_dir: Path,
    *,
    profiles: tuple[str, ...] = ("reference", "optimized", "fast"),
) -> dict[str, Any]:
    raw_root = out_dir / "raw"
    meas_root = out_dir / "measured"
    measure_items: list[tuple[Path, Path, str]] = []
    for prof in profiles:
        if prof != "fast":
            for lam in E_GATE_KNOBS:
                tag = f"{prof}_e_l{int(round(lam * 100)):03d}"
                measure_items.append((raw_root / tag, meas_root / tag, tag))
        for sk in P_BENCH_KNOBS:
            tag = f"{prof}_p_s{int(round(sk * 100)):03d}"
            measure_items.append((raw_root / tag, meas_root / tag, tag))

    summaries = run_measure_batch_subprocess(measure_items, device="cuda", write_png=False)

    comparisons: dict[str, Any] = {}
    for prof in profiles:
        if prof == "reference":
            continue
        comparisons[prof] = {}
        if prof != "fast":
            for lam in E_GATE_KNOBS:
                ktag = f"e_l{int(round(lam * 100)):03d}"
                comparisons[prof][ktag] = paired_cluster_bootstrap(
                    meas_root / f"{prof}_{ktag}" / "eval.json",
                    meas_root / f"reference_{ktag}" / "eval.json",
                )
        for sk in P_BENCH_KNOBS:
            ktag = f"p_s{int(round(sk * 100)):03d}"
            comparisons[prof][ktag] = paired_cluster_bootstrap(
                meas_root / f"{prof}_{ktag}" / "eval.json",
                meas_root / f"reference_{ktag}" / "eval.json",
            )

    matched_c: dict[str, Any] = {}
    for prof in profiles:
        pts = []
        for sk in P_BENCH_KNOBS:
            c600 = summaries[f"{prof}_p_s{int(round(sk * 100)):03d}"]["cross600"]
            pts.append((c600["dino_c"], c600["dino_s"], c600["csd_top1"] * 100.0))
        matched_c[prof] = interp_matched_c(pts)

    report = {"summaries": summaries, "paired_bootstrap": comparisons, "matched_c": matched_c, "utc": utc()}
    save_json(out_dir / "quality_gate.json", report)
    return report


def run_quality_suite(
    bundle_dir: Path,
    out_dir: Path,
    *,
    profiles: tuple[str, ...] = ("reference", "optimized", "fast"),
    batch_size: int = 4,
    fast_steps: int = 5,
) -> dict[str, Any]:
    generate_all_arms(bundle_dir, out_dir, profiles=profiles, batch_size=batch_size, fast_steps=fast_steps)
    return measure_and_bootstrap(out_dir, profiles=profiles)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-dir", default=str(BUNDLE_DEFAULT_DIR))
    ap.add_argument("--out-dir", default=str(OPT_OUT_DIR / "test750_gate"))
    ap.add_argument("--profiles", nargs="+", default=["reference", "optimized", "fast"])
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--fast-steps", type=int, default=5)
    ap.add_argument("--stage", choices=["all", "generate", "measure"], default="all")
    args = ap.parse_args()
    if args.stage == "generate":
        generate_all_arms(Path(args.bundle_dir), Path(args.out_dir), profiles=tuple(args.profiles), batch_size=args.batch_size, fast_steps=args.fast_steps)
        print("QUALITY_GEN_DONE", args.out_dir)
    elif args.stage == "measure":
        measure_and_bootstrap(Path(args.out_dir), profiles=tuple(args.profiles))
        print("QUALITY_GATE_DONE", args.out_dir)
    else:
        run_quality_suite(Path(args.bundle_dir), Path(args.out_dir), profiles=tuple(args.profiles), batch_size=args.batch_size, fast_steps=args.fast_steps)
        print("QUALITY_GATE_DONE", args.out_dir)


if __name__ == "__main__":
    main()
