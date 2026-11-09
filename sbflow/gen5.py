"""Generate the full 600 frozen cross-style transfers (or the Quick200 subset).

This is the stage-1 delivery gate from docs/bridgev10.md 1.5:
one five-style model must be able to produce all 600 cross-style results.

Writes PNGs plus a manifest so a separate CPU process can score them without
touching the GPU (`sbflow/score_cpu.py`).
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from PIL import Image

from .infer5 import load_model5, solve_transfer
from .vae import decode_latents, encode_images, load_sdvae

STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]


def load_tasks(repo: Path, subset: str = "full600") -> List[dict]:
    raw = json.loads((repo / "eval_protocol/tasks_750.json").read_text(encoding="utf-8"))
    tasks = [t for t in raw["tasks"] if t.get("is_cross", t["source_style"] != t["target_style"])]
    if subset == "quick200":
        # fixed 10 per directed pair, deterministic by sorted task id
        buckets: Dict[tuple, List[dict]] = {}
        for t in tasks:
            buckets.setdefault((t["source_style"], t["target_style"]), []).append(t)
        picked = []
        for key in sorted(buckets):
            picked.extend(sorted(buckets[key], key=lambda x: x["task_id"])[:10])
        tasks = picked
    elif subset == "same150":
        raw_all = raw["tasks"]
        tasks = [t for t in raw_all
                 if not t.get("is_cross", t["source_style"] != t["target_style"])]
    elif subset != "full600":
        raise ValueError(f"unknown subset {subset}")
    for t in tasks:
        t.setdefault("source_rel_path", f"data/test/{t['source_style']}/{t['source_filename']}")
    return tasks


def png_bytes(u8: np.ndarray) -> bytes:
    buf = BytesIO()
    Image.fromarray(u8).save(buf, format="PNG")
    return buf.getvalue()


def to_u8(im: torch.Tensor) -> np.ndarray:
    a = (im.detach().float().clamp(-1, 1).permute(0, 2, 3, 1).cpu().numpy() + 1) * 127.5
    return a.round().clip(0, 255).astype(np.uint8)


def write_png(u8: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png_bytes(u8))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--subset", choices=["full600", "quick200", "same150"], default="full600")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--method", choices=["euler", "heun"], default="heun")
    ap.add_argument("--t-floor", type=float, default=0.002)
    ap.add_argument("--weights", choices=["raw", "ema"], default="raw")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--save-images", action="store_true", default=True)
    ap.add_argument("--no-images", dest="save_images", action="store_false")
    ap.add_argument("--limit", type=int, default=0, help="debug: cap the number of tasks")
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    out = Path(args.out)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tasks = load_tasks(repo, args.subset)
    if args.limit:
        tasks = tasks[: args.limit]

    t0 = time.perf_counter()
    ck, model, stats = load_model5(args.ckpt, device, args.weights)
    vae, trt = load_sdvae(device)
    sf = float(ck["ae_scaling_factor"])
    gmean = stats.global_mean
    gstd = stats.global_std
    objective = ck.get("objective", "bidir")
    eps = float(ck["config"]["eps"])
    is_pivot = bool(ck["config"].get("pivot", False))
    n_real = int(stats.style_mean.shape[0] - 1) if is_pivot else len(STYLES)
    pivot_id = n_real if is_pivot else -1
    if is_pivot:
        from .pivot import pivot_transfer
    print(f"[gen5] ckpt={args.ckpt} objective={objective} eps={eps} tasks={len(tasks)} "
          f"subset={args.subset} steps={args.steps} {args.method} device={device} "
          f"pivot={is_pivot}", flush=True)

    # Cache the source encoding per unique source path.
    uniq = sorted({t["source_rel_path"] for t in tasks})
    zsrc: Dict[str, torch.Tensor] = {}
    pool = ThreadPoolExecutor(max_workers=args.workers) if args.save_images else None
    gen_dir = out / "generated"
    phase = {"encode": 0.0, "ode": 0.0, "decode": 0.0, "submit": 0.0}

    with torch.inference_mode():
        for s in range(0, len(uniq), args.batch):
            chunk = uniq[s:s + args.batch]
            t1 = time.perf_counter()
            x = torch.stack([
                torch.from_numpy(np.asarray(Image.open(repo / p).convert("RGB").resize((512, 512)),
                                            dtype=np.uint8).copy()).permute(2, 0, 1).float() / 255.0
                for p in chunk]).to(device) * 2 - 1
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                z = encode_images(vae, x, sf)
            z = (z - gmean[None, :, None, None]) / gstd[None, :, None, None]
            for i, p in enumerate(chunk):
                zsrc[p] = z[i:i + 1]
            phase["encode"] += time.perf_counter() - t1
        print(f"[gen5] encoded {len(uniq)} sources in {phase['encode']:.1f}s", flush=True)

        manifest: List[dict] = []
        for s in range(0, len(tasks), args.batch):
            group = tasks[s:s + args.batch]
            t1 = time.perf_counter()
            z = torch.cat([zsrc[t["source_rel_path"]] for t in group], 0)
            zout = torch.empty_like(z)
            for src_name in sorted({t["source_style"] for t in group}):
                for tgt_name in sorted({t["target_style"] for t in group}):
                    pos = [j for j, t in enumerate(group)
                           if t["source_style"] == src_name and t["target_style"] == tgt_name]
                    if not pos:
                        continue
                    idx = torch.tensor(pos, device=device)
                    si, ti = STYLES.index(src_name), STYLES.index(tgt_name)
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                        if is_pivot:
                            zout[idx] = pivot_transfer(
                                model, z[idx].clone(), si, ti, pivot_id, stats, eps,
                                args.steps, args.method, args.t_floor, objective)
                        else:
                            zout[idx] = solve_transfer(
                                model, z[idx].clone(), si, ti, stats, eps,
                                args.steps, args.method, args.t_floor, objective)
            phase["ode"] += time.perf_counter() - t1

            t1 = time.perf_counter()
            dec = decode_latents(vae, trt,
                                 (zout * gstd[None, :, None, None]
                                  + gmean[None, :, None, None]) / sf, batch=args.batch)
            u8 = to_u8(dec)
            phase["decode"] += time.perf_counter() - t1

            t1 = time.perf_counter()
            if pool is not None:
                for j, t in enumerate(group):
                    pool.submit(write_png, u8[j], gen_dir / f"{s + j:04d}.png")
            for j, t in enumerate(group):
                manifest.append({
                    "task_id": t["task_id"], "source_style": t["source_style"],
                    "target_style": t["target_style"], "source_rel_path": t["source_rel_path"],
                    "image": f"generated/{s + j:04d}.png" if pool is not None else None,
                    "latent_l2": float((zout[j] - z[j]).flatten().norm()),
                })
            phase["submit"] += time.perf_counter() - t1
            if s % (args.batch * 4) == 0:
                print(f"[gen5] {s + len(group)}/{len(tasks)}", flush=True)

    if pool is not None:
        pool.shutdown(wait=True)
    wall = time.perf_counter() - t0
    nfe = args.steps * (2 if args.method == "euler" else 4)
    info = {"checkpoint": str(args.ckpt), "objective": objective, "eps": eps,
            "subset": args.subset, "n_tasks": len(tasks), "solver": args.method,
            "solver_steps": args.steps, "network_nfe": nfe, "weights": args.weights,
            "batch": args.batch,
            "timing_sec": {k: round(v, 2) for k, v in phase.items()},
            "total_sec": round(wall, 2), "images_per_sec": round(len(tasks) / max(wall, 1e-9), 2)}
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps({"info": info, "tasks": manifest},
                                                  ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[gen5] done {len(tasks)} images in {wall:.1f}s "
          f"({info['images_per_sec']:.1f} img/s) -> {out}", flush=True)


if __name__ == "__main__":
    main()
