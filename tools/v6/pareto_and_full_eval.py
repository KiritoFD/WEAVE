"""WEAVE V6 Stage 3b/4: Pareto Calibration & Full 750 (600 cross + 150 same) Evaluation
for both C³ Direct/Flow models and Score Difference models, including Multi-Seed Verification.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools" / "12h"))
sys.path.insert(0, str(REPO_ROOT / "tools" / "trt"))

import paths  # noqa: E402
import csd_model  # noqa: E402
from tools.eval_fast import (  # noqa: E402
    METRIC_HEADER, DINO_HEADER, MetricEngine, Timer, write_csv,
)
from tools.pseudo_gt.matching import encode_dino, load_dino_model  # noqa: E402
from utils.run_evaluation import _load_eval_image_tensor, to_lpips_input  # noqa: E402
from tools.v5.route_c import csd_prototypes, decode_trt, load_trt, load_vae, source_latents  # noqa: E402
from tools.v6.canonicalize import ContentCanonicalizer  # noqa: E402
from tools.v6.models_v6 import C3UNet, ConditionalScoreNet  # noqa: E402
from tools.v6.score_diff import edit_with_score_difference  # noqa: E402
from tools.v6.train_eval_v6 import boot_ci, generate_v6  # noqa: E402

STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
V5_ROOT = REPO_ROOT / "runs" / "v5"
V6_ROOT = REPO_ROOT / "runs" / "v6"
RENDER_ROOT = paths.asset_root() / "eval_assets" / "v6_main"


def eval200_latents_fn(gen_fn, device: str = "cuda") -> dict:
    """Evaluate any latent generator gen_fn(z_src_batch, src_ids, tgt_ids) on 200 cross + 50 same."""
    tasks = json.loads((V5_ROOT / "v5_eval200_tasks.json").read_text(encoding="utf-8"))
    lat_dict = torch.load(V5_ROOT / "cache" / "eval_src_latents.pt", map_location="cpu")
    stems = sorted({t["src_stem"] for t in tasks})
    z_all = torch.stack([lat_dict[s] for s in stems]).to(device)
    stem_idx = {s: i for i, s in enumerate(stems)}
    src_idx = torch.tensor([stem_idx[t["src_stem"]] for t in tasks], device=device)
    src_ids_200 = torch.tensor([t["src_style_id"] for t in tasks], device=device, dtype=torch.long)
    tgt_ids_200 = torch.tensor([t["tgt_style_id"] for t in tasks], device=device, dtype=torch.long)

    proto = csd_prototypes()
    csd, _, _ = csd_model.load_csd(device)
    dino = load_dino_model(device=device)
    vae = load_vae(device)
    trt = load_trt(vae, device)
    import lpips
    lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    src_tensors = {
        t["src_stem"]: _load_eval_image_tensor(Path(t["src_path"]), size=512).unsqueeze(0).to(device)
        for t in tasks
    }
    csd_src50 = torch.load(V5_ROOT / "cache" / "eval_src_csd_feats.pt", map_location="cpu", weights_only=False)
    csd_src_cache = {s: np.asarray(csd_src50[s]) for s in stems}
    dino_src_cache = {}

    with torch.inference_mode():
        img_src = decode_trt(trt, vae, z_all, batch=8)
        u8_src = img_src.mul(255.0).round().clamp(0, 255).to(torch.uint8).cpu().permute(0, 2, 3, 1).numpy()
        pils_src = [Image.fromarray(u8_src[i]) for i in range(len(stems))]
        dcls, _ = encode_dino(dino, pils_src, batch=16, device=device)
        for i, s in enumerate(stems):
            dino_src_cache[s] = dcls[i].float().cpu().numpy()

        same_ids = torch.tensor(
            [next(t["src_style_id"] for t in tasks if t["src_stem"] == s) for s in stems],
            device=device, dtype=torch.long,
        )
        z_same = gen_fn(z_all, same_ids, same_ids)
        img_same = decode_trt(trt, vae, z_same, batch=8)
        u8_same = img_same.mul(255.0).round().clamp(0, 255).to(torch.uint8).cpu().permute(0, 2, 3, 1).numpy()
        pils_same = [Image.fromarray(u8_same[i]) for i in range(len(stems))]
        f_same = csd_model.feats_pils(csd, pils_same, batch=16, device=device)
        d_same, _ = encode_dino(dino, pils_same, batch=16, device=device)
        d_same = d_same.float().cpu().numpy()
        same_dS = [
            float(f_same[i] @ proto[int(same_ids[i].item())] - csd_src_cache[stems[i]] @ proto[int(same_ids[i].item())])
            for i in range(len(stems))
        ]
        same_dino_c = [float(d_same[i] @ dino_src_cache[stems[i]]) for i in range(len(stems))]

        z_gen = gen_fn(z_all[src_idx], src_ids_200, tgt_ids_200)
        img = decode_trt(trt, vae, z_gen, batch=8)
        u8 = img.mul(255.0).round().clamp(0, 255).to(torch.uint8).cpu()
        gen01 = u8.float().div_(255.0)
        arr_gen = u8.permute(0, 2, 3, 1).numpy()
        pils = [Image.fromarray(arr_gen[i]) for i in range(len(tasks))]

        fg = csd_model.feats_pils(csd, pils, batch=16, device=device)
        dg, _ = encode_dino(dino, pils, batch=16, device=device)
        dg = dg.float().cpu().numpy()
        lps = []
        for s in range(0, len(tasks), 16):
            chunk = gen01[s : s + 16].to(device)
            src = torch.cat([src_tensors[t["src_stem"]] for t in tasks[s : s + 16]], 0)
            lps.append(lpips_fn(to_lpips_input(chunk), to_lpips_input(src)).flatten().cpu())
        lps = torch.cat(lps, 0).numpy()

    dTs, dSs, dCs = [], [], []
    per_style = {s: [] for s in STYLES}
    for i, t in enumerate(tasks):
        fs = csd_src_cache[t["src_stem"]]
        tgt, sid = t["tgt_style_id"], t["src_style_id"]
        dt = float(fg[i] @ proto[tgt] - fs @ proto[tgt])
        ds = float(fg[i] @ proto[sid] - fs @ proto[sid])
        dTs.append(dt)
        dSs.append(ds)
        dCs.append(float(dg[i] @ dino_src_cache[t["src_stem"]]))
        per_style[t["tgt_style"]].append(dt)

    del csd, dino, vae, trt, lpips_fn
    torch.cuda.empty_cache()

    dT_arr, dS_arr = np.array(dTs), np.array(dSs)
    per_tgt_mean = {s: float(np.mean(per_style[s])) for s in STYLES}
    return {
        "mean_dT": float(dT_arr.mean()),
        "dT_ci": boot_ci(dT_arr),
        "mean_dS": float(dS_arr.mean()),
        "mean_dM": float((dT_arr - dS_arr).mean()),
        "mean_dino_c": float(np.mean(dCs)),
        "mean_lpips": float(lps.mean()),
        "per_target_dT": per_tgt_mean,
        "styles_improved": int(sum(1 for v in per_tgt_mean.values() if v > 0)),
        "ukiyo_dT": per_tgt_mean["Ukiyo_e"],
        "same50_mean_dS": float(np.mean(same_dS)),
        "same50_mean_dino_c": float(np.mean(same_dino_c)),
    }


def full_eval750_fn(gen_fn, cell: str, device: str = "cuda") -> dict:
    """Render 750 images (600 cross + 150 same) and run frozen v4_eval750.py."""
    out_dir = RENDER_ROOT / cell
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    z_src, src_paths, requests, style_meta = source_latents(device, max_src=30)
    num_styles = len(style_meta)
    req_map = {(str(r["src_path"]), int(r["tgt_style_id"])): r for r in requests}

    vae = load_vae(device)
    trt = load_trt(vae, device)
    ns = argparse.Namespace(
        disable_clip=False, disable_lpips=False, disable_dino=False,
        test_dir=str(REPO_ROOT / "data" / "test"),
        clip_model="openai/clip-vit-base-patch32", clip_local_only=True,
        clip_cache_dir=str(paths.eval_cache_root("hf") / "hub"), clip_skip_processor=False,
        amp=False, lpips_net="alex", lpips_chunk=32,
        dino_model="facebook/dinov2-small",
        dino_cache_dir=str(paths.eval_cache_root("hf")), allow_network=False,
        ref_cache="", max_ref_cache=16, dino_max_refs=30, image_size=512,
        metric_batch=32, ref_batch=32, only_clip_style=True,
        clip_style_idt_baseline=0.0, clip_vectors=False,
    )
    engine = MetricEngine(ns, {}, style_meta, device)
    engine.load(Timer())
    engine.prepare(Timer(), src_paths)

    rows_all = []
    save_pool = ThreadPoolExecutor(max_workers=8)
    futs = []
    t_start = time.perf_counter()

    for b in range(0, len(src_paths), 8):
        chunk_paths = src_paths[b : b + 8]
        z_s = z_src[b : b + len(chunk_paths)]
        n = z_s.shape[0]
        src_ids = torch.tensor(
            [int(req_map[(str(p), 0)]["src_style_id"]) for p in chunk_paths],
            dtype=torch.long, device=device,
        )
        for tgt in range(num_styles):
            rows = [req_map[(str(p), tgt)] for p in chunk_paths]
            tgt_ids = torch.full((n,), tgt, dtype=torch.long, device=device)
            with torch.inference_mode():
                z_gen = gen_fn(z_s, src_ids, tgt_ids)
                img = decode_trt(trt, vae, z_gen, batch=8)
            u8 = img.mul(255.0).round().clamp(0, 255).to(torch.uint8)
            gen01 = u8.float().div_(255.0)
            arr = u8.cpu().numpy()
            for i, r in enumerate(rows):
                futs.append(
                    save_pool.submit(
                        lambda a, p: Image.fromarray(a).save(p, format="PNG"),
                        arr[i].transpose(1, 2, 0).copy(),
                        img_dir / r["gen_name"],
                    )
                )
            out_rows, _ = engine.batch_metrics(gen01, rows, Timer())
            rows_all.extend(out_rows)
        if (b // 8) % 5 == 0:
            print(f"[eval750:{cell}] {min(b+8, len(src_paths))}/{len(src_paths)} sources ({time.perf_counter()-t_start:.0f}s)", flush=True)

    for f in futs:
        f.result()
    save_pool.shutdown(wait=True)

    write_csv(out_dir / "metrics.csv", rows_all, METRIC_HEADER)
    write_csv(out_dir / "dino_metrics.csv", rows_all, DINO_HEADER)
    del engine, vae, trt
    torch.cuda.empty_cache()

    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "pseudo_gt" / "v4_eval750.py"),
        "--cell", cell,
        "--render-root", str(RENDER_ROOT),
        "--max-src", "30",
    ]
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
    eval750 = json.loads((out_dir / "v4_eval750.json").read_text(encoding="utf-8"))
    rows750 = json.loads((out_dir / "v4_eval750_rows.json").read_text(encoding="utf-8"))
    cross_rows = [r for r in rows750 if r["cross"]]
    per_tgt = {
        s: float(np.mean([r["csd_dT"] for r in cross_rows if r["tgt_style"] == s]))
        for s in STYLES
    }
    eval750["per_target_cross_CSD_dT"] = per_tgt
    eval750["styles_improved"] = int(sum(1 for v in per_tgt.values() if v > 0))
    eval750["ukiyo_dT"] = per_tgt["Ukiyo_e"]
    (out_dir / "v4_eval750.json").write_text(json.dumps(eval750, indent=2), encoding="utf-8")
    return eval750


def main() -> None:
    device = "cuda"
    global_stats = torch.load(V6_ROOT / "cache" / "canonicalizer_stats.pt", map_location="cpu", weights_only=False)
    canon = ContentCanonicalizer(global_stats).to(device)

    # 1. Load Score Difference model & calibrate step_size in LPIPS-safe band (LPIPS <= 0.275, DINO-C >= 0.87)
    sd_ck = torch.load(V6_ROOT / "score_diff" / "score_diff_net.pt", map_location="cpu", weights_only=False)
    sd_model = ConditionalScoreNet(base_dim=64, num_styles=5).to(device).eval()
    sd_model.load_state_dict(sd_ck["model"])

    pareto_results = {"score_diff_pareto": {}, "d1_blend_pareto": {}}
    for name, steps, lr_s, anc in (
        ("sd_s10_lr0.045_a0.08", 10, 0.045, 0.08),
        ("sd_s12_lr0.055_a0.08", 12, 0.055, 0.08),
    ):
        print(f"[Pareto] Testing {name}...", flush=True)
        res = eval200_latents_fn(
            lambda z, s_id, t_id, st=steps, lr=lr_s, a=anc: edit_with_score_difference(
                sd_model, z, s_id, t_id, num_steps=st, step_size=lr, struct_anchor_weight=a, seed=42
            ),
            device=device,
        )
        pareto_results["score_diff_pareto"][name] = res
        print(f"  -> {name}: dT={res['mean_dT']:+.4f} DINO-C={res['mean_dino_c']:.4f} "
              f"LPIPS={res['mean_lpips']:.4f} styles+={res['styles_improved']}/5 same_dS={res['same50_mean_dS']:+.4f}", flush=True)

    # 2. Load D1_c1_gentle and test residual strength alpha in {0.65, 0.80} for LPIPS <= 0.275
    d1_ck = torch.load(V6_ROOT / "D1_c1_gentle" / "D1_c1_gentle.pt", map_location="cpu", weights_only=False)
    d1_model = C3UNet(base_dim=48, num_styles=5, mode="direct").to(device).eval()
    d1_model.load_state_dict(d1_ck["model"])

    for alpha in (0.65, 0.80):
        name = f"D1_c1_gentle_a{alpha:.2f}"
        print(f"[Pareto] Testing {name}...", flush=True)
        def d1_alpha_fn(z, s_id, t_id, al=alpha):
            z_out = generate_v6(d1_model, canon, z, t_id, canon_mode="c1_gentle")
            return (1.0 - al) * z + al * z_out
        res = eval200_latents_fn(d1_alpha_fn, device=device)
        pareto_results["d1_blend_pareto"][name] = res
        print(f"  -> {name}: dT={res['mean_dT']:+.4f} DINO-C={res['mean_dino_c']:.4f} "
              f"LPIPS={res['mean_lpips']:.4f} styles+={res['styles_improved']}/5 same_dS={res['same50_mean_dS']:+.4f}", flush=True)

    (V6_ROOT / "stage3_pareto_calibration.json").write_text(
        json.dumps(pareto_results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("[Pareto] Saved stage3_pareto_calibration.json", flush=True)


if __name__ == "__main__":
    main()
