"""16-sample layer parity, source-preprocessing parity, and quality/non-inferiority suite."""
from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from exp.deploy_v35 import BUNDLE_DEFAULT_DIR, E_GATE_KNOBS, P_BENCH_KNOBS
from exp.deploy_v35.pipeline import CleaveDeployPipeline
from src.utils.runtime import REPO as ROOT, read_json, save_json


def make_deterministic_eps(task_id: str, seed: int = 42, shape=(4, 64, 64)) -> torch.Tensor:
    h = int.from_bytes(hashlib.blake2b(f"{seed}:{task_id}".encode("utf-8"), digest_size=8).digest(), "little")
    gen = torch.Generator(device="cpu").manual_seed(h & ((1 << 63) - 1))
    return torch.randn(shape, generator=gen, dtype=torch.float32)


def load_rgb01_from_rel(rel_path: str, size: int = 512) -> torch.Tensor:
    with Image.open(ROOT / rel_path) as im:
        im_r = im.convert("RGB")
        if im_r.size != (size, size):
            im_r = im_r.resize((size, size), Image.Resampling.BICUBIC)
        arr = np.asarray(im_r, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def select_stratified_tasks(n_samples: int = 16) -> list[dict[str, Any]]:
    data = read_json(ROOT / "eval_protocol/tasks_750.json")
    tasks = data["tasks"]
    cross = [t for t in tasks if t["is_cross"]]
    same = [t for t in tasks if not t["is_cross"]]
    picked: list[dict[str, Any]] = []
    # Pick evenly across all 5 target styles (mostly cross + 1 same)
    for sid in range(5):
        cand = [t for t in cross if t["target_style_id"] == sid]
        k = max(1, (n_samples - 1) // 5 + (1 if sid < ((n_samples - 1) % 5) else 0))
        step = max(1, len(cand) // k)
        picked.extend(cand[i * step] for i in range(k))
    if len(picked) < n_samples and same:
        picked.append(same[0])
    return picked[:n_samples]


def _tensor_diff(cand: torch.Tensor, ref: torch.Tensor) -> dict[str, float]:
    c = cand.detach().float()
    r = ref.detach().float()
    diff = c - r
    return {
        "max_abs": float(diff.abs().max().item()),
        "mean_abs": float(diff.abs().mean().item()),
        "rmse": float(diff.pow(2).mean().sqrt().item()),
        "cos_sim": float(F.cosine_similarity(c.flatten(1), r.flatten(1), dim=1).mean().item()),
    }


def _rgb_metrics(cand01: torch.Tensor, ref01: torch.Tensor) -> dict[str, float]:
    c = cand01.detach().float().clamp(0.0, 1.0)
    r = ref01.detach().float().clamp(0.0, 1.0)
    diff = (c - r).abs()
    mse = float((c - r).pow(2).mean().item())
    psnr = float(-10.0 * math.log10(max(mse, 1e-12)))
    # Fast global channel-wise SSIM approximation over 11x11 box window
    k = 11
    pad = k // 2
    mu_c = F.avg_pool2d(c, k, 1, pad)
    mu_r = F.avg_pool2d(r, k, 1, pad)
    sig_c = F.avg_pool2d(c * c, k, 1, pad) - mu_c * mu_c
    sig_r = F.avg_pool2d(r * r, k, 1, pad) - mu_r * mu_r
    sig_cr = F.avg_pool2d(c * r, k, 1, pad) - mu_c * mu_r
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu_c * mu_r + c1) * (2 * sig_cr + c2)) / ((mu_c ** 2 + mu_r ** 2 + c1) * (sig_c + sig_r + c2))
    return {
        "mae_255": float(diff.mean().item() * 255.0),
        "max_255": float(diff.max().item() * 255.0),
        "psnr": psnr,
        "ssim": float(ssim_map.mean().item()),
    }


def check_source_preprocessing_parity(pipe: CleaveDeployPipeline, *, limit: int = 16) -> dict[str, Any]:
    """Compare live JPG preprocessing against authoritative n2c_z.pt and test750_struct_cache.pt."""
    tj = read_json(ROOT / "eval_protocol/tasks_750.json")
    tasks = tj["tasks"]
    n2c = torch.load(ROOT / "runs/v9_local_4070/cache/n2c_z.pt", map_location="cpu", weights_only=False)
    id_to_idx = {tid: i for i, tid in enumerate(n2c["task_ids"])}
    sc_path = ROOT / "runs/v18/cache/test750_struct_cache.pt"
    sc = torch.load(sc_path, map_location="cpu", weights_only=False) if sc_path.exists() else None

    seen_stems: set[str] = set()
    chosen: list[tuple[int, dict[str, Any]]] = []
    for t in tasks:
        if t["source_stem"] in seen_stems:
            continue
        seen_stems.add(t["source_stem"])
        chosen.append((id_to_idx[t["task_id"]], t))
        if limit and len(chosen) >= limit:
            break

    z_live, c64_live, z_ref, c64_ref = [], [], [], []
    with torch.inference_mode():
        for i in range(0, len(chosen), 8):
            chunk = chosen[i:i + 8]
            rgb = torch.stack([load_rgb01_from_rel(t["source_rel_path"]) for _, t in chunk]).to(pipe.device)
            src = pipe.prepare_source(rgb, need_y16=False)
            z_live.append(src["z_s"].cpu())
            c64_live.append(src["c64"].cpu())
            idxs = [idx for idx, _ in chunk]
            z_ref.append(n2c["z_src"][idxs])
            if sc is not None:
                c64_ref.append(sc["c_64"][idxs].float())

    z_live_t = torch.cat(z_live, 0)
    z_ref_t = torch.cat(z_ref, 0)
    out = {
        "n_unique_sources": len(chosen),
        "z_s_vs_n2c_z": _tensor_diff(z_live_t, z_ref_t),
    }
    if c64_ref:
        out["c64_vs_struct_cache_fp16"] = _tensor_diff(torch.cat(c64_live, 0), torch.cat(c64_ref, 0))
    return out


def compare_pipelines(
    ref_pipe: CleaveDeployPipeline,
    cand_pipe: CleaveDeployPipeline,
    *,
    n_samples: int = 16,
    e_lam: float = 0.60,
    p_knobs: tuple[float, ...] = (0.85, 1.00),
    batch_size: int = 4,
) -> dict[str, Any]:
    """Run layer-by-layer + E2E parity between reference pipeline and candidate pipeline."""
    tasks = select_stratified_tasks(n_samples)
    rgbs = torch.stack([load_rgb01_from_rel(t["source_rel_path"]) for t in tasks])
    tids = torch.tensor([int(t["target_style_id"]) for t in tasks], dtype=torch.long)
    eps_all = torch.stack([make_deterministic_eps(t["task_id"]) for t in tasks])

    def _run_all(pipe: CleaveDeployPipeline):
        pipe.stats.reset()
        res: dict[str, Any] = {"src": [], "e_no_a3": [], "e_a3": [], "p": {k: [] for k in p_knobs}, "p_zero": [], "p_tau0": []}
        for s in range(0, len(tasks), batch_size):
            e = min(len(tasks), s + batch_size)
            rgb = rgbs[s:e].to(pipe.device)
            tid = tids[s:e].to(pipe.device)
            eps = eps_all[s:e].to(pipe.device)
            src = pipe.prepare_source(rgb, need_y16=True)
            res["src"].append({k: v.cpu() for k, v in src.items()})
            rgb_e0, inter_e0 = pipe.run_e_only(rgb, tid, e_lam, eps, use_a3=False, cached_src=src, return_intermediates=True)
            res["e_no_a3"].append((rgb_e0.cpu(), inter_e0["ze"].cpu()))
            rgb_ea3, inter_ea3 = pipe.run_e_only(rgb, tid, e_lam, eps, use_a3=True, cached_src=src, return_intermediates=True)
            res["e_a3"].append((rgb_ea3.cpu(), inter_ea3["z_out"].cpu()))
            rgb_z = pipe.run_e_to_p(rgb, tid, 0.0, eps, cached_src=src)
            res["p_zero"].append((rgb_z.cpu(), rgb.cpu()))
            for k in p_knobs:
                rgb_p, inter_p = pipe.run_e_to_p(rgb, tid, float(k), eps, cached_src=src, return_intermediates=True)
                res["p"][k].append((rgb_p.cpu(), inter_p["zo"].cpu(), inter_p["ze_s"].cpu() if inter_p["ze_s"] is not None else None))
            rgb_t0, inter_t0 = pipe.run_e_to_p(rgb, tid, 1.00, eps, cached_src=src, tau0_override=0.0, return_intermediates=True)
            res["p_tau0"].append((rgb_t0.cpu(), inter_t0["zo"].cpu(), inter_t0["ze"] is None))
        stats = pipe.stats.as_dict()
        return res, stats

    ref_res, ref_stats = _run_all(ref_pipe)
    cand_res, cand_stats = _run_all(cand_pipe)

    def _cat_src(blocks, key):
        return torch.cat([b[key] for b in blocks], 0)

    report: dict[str, Any] = {
        "n_samples": len(tasks),
        "stats_ref": ref_stats,
        "stats_cand": cand_stats,
        "source_tensors": {
            k: _tensor_diff(_cat_src(cand_res["src"], k), _cat_src(ref_res["src"], k))
            for k in ("z_s", "c64", "g16", "q16", "y16")
        },
        "e_only_no_a3": {
            "ze": _tensor_diff(torch.cat([x[1] for x in cand_res["e_no_a3"]], 0), torch.cat([x[1] for x in ref_res["e_no_a3"]], 0)),
            "rgb": _rgb_metrics(torch.cat([x[0] for x in cand_res["e_no_a3"]], 0), torch.cat([x[0] for x in ref_res["e_no_a3"]], 0)),
        },
        "e_only_a3": {
            "z_out": _tensor_diff(torch.cat([x[1] for x in cand_res["e_a3"]], 0), torch.cat([x[1] for x in ref_res["e_a3"]], 0)),
            "rgb": _rgb_metrics(torch.cat([x[0] for x in cand_res["e_a3"]], 0), torch.cat([x[0] for x in ref_res["e_a3"]], 0)),
        },
        "s0_identity": _rgb_metrics(torch.cat([x[0] for x in cand_res["p_zero"]], 0), torch.cat([x[0] for x in ref_res["p_zero"]], 0)),
        "p_tau0_s150": {
            "ecore_skipped": all(x[2] for x in cand_res["p_tau0"]),
            "zo": _tensor_diff(torch.cat([x[1] for x in cand_res["p_tau0"]], 0), torch.cat([x[1] for x in ref_res["p_tau0"]], 0)),
            "rgb": _rgb_metrics(torch.cat([x[0] for x in cand_res["p_tau0"]], 0), torch.cat([x[0] for x in ref_res["p_tau0"]], 0)),
        },
        "p_knobs": {},
    }
    for k in p_knobs:
        report["p_knobs"][f"s{int(round(k * 100)):03d}"] = {
            "zo": _tensor_diff(torch.cat([x[1] for x in cand_res["p"][k]], 0), torch.cat([x[1] for x in ref_res["p"][k]], 0)),
            "rgb": _rgb_metrics(torch.cat([x[0] for x in cand_res["p"][k]], 0), torch.cat([x[0] for x in ref_res["p"][k]], 0)),
        }
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-dir", default=str(BUNDLE_DEFAULT_DIR))
    ap.add_argument("--source-check-limit", type=int, default=16)
    ap.add_argument("--n-samples", type=int, default=16)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    ref_pipe = CleaveDeployPipeline(bundle_dir=Path(args.bundle_dir), profile="reference")
    cand_pipe = CleaveDeployPipeline(
        bundle_dir=Path(args.bundle_dir),
        profile="optimized",
        use_host_grid=True,
    )
    src_parity = check_source_preprocessing_parity(ref_pipe, limit=args.source_check_limit)
    pipe_parity = compare_pipelines(ref_pipe, cand_pipe, n_samples=args.n_samples)
    out = {"source_preprocessing": src_parity, "pipeline_parity": pipe_parity}
    if args.out:
        save_json(args.out, out)
    import json
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
