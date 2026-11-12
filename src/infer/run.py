"""CLI: run a registered model variant over the frozen task list, store raw output.

  python -m src.infer.run --ckpt runs/v6/Q2a_embed_style/Q2a_embed_style.pt \
      --kind direct --out runs/infer/q2a --batch 8

Output per run directory:
  meta.json        arguments, model kinds, batch, timing, per-batch file list
  batch_XXXX.pt    ids, z_src, z0, z_final, trace (and rgb_u8 unless --no-decode)

No PNG encoding, no DINO/CLIP/CSD/LPIPS. Those belong to src/eval.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from . import DEFAULT_BATCH, REPO, InferSpec, build, decode_padded, load_trt, load_vae


def load_tasks(path: Path, limit: int = 0):
    data = json.loads(path.read_text(encoding="utf-8"))
    tasks = data["tasks"]
    return tasks[:limit] if limit else tasks


def run(spec: InferSpec, out: Path, *, tasks_file: Path, src_cache: Path,
        batch: int = DEFAULT_BATCH, limit: int = 0, decode: bool = True,
        device: str = "cuda", resume: bool = True) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks(tasks_file, limit)
    src = torch.load(src_cache, map_location="cpu", weights_only=False)
    gen = build(spec, device)
    trt = vae = None
    if decode:
        vae = load_vae(device)
        trt = load_trt(vae, device)

    batches = []
    t0 = time.time()
    for begin in range(0, len(tasks), batch):
        group = tasks[begin:begin + batch]
        ids = [t["task_id"] for t in group]
        path = out / f"batch_{begin:04d}.pt"
        if resume and path.exists():
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if payload["ids"] != ids:
                raise ValueError(f"resume mismatch at offset {begin}")
        else:
            z = torch.cat([src["z_src"][t["source_stem"]] for t in group]).to(device)
            sid = torch.tensor([t["source_style_id"] for t in group], device=device)
            tid = torch.tensor([t["target_style_id"] for t in group], device=device)
            result = gen(z, sid, tid, group)
            payload = {
                "ids": ids,
                "z_src": z.detach().cpu(),
                "z0": result.get("z0", result["final"]).detach().cpu(),
                "z_final": result["final"].detach().cpu(),
                "trace": result.get("trace", []),
            }
            if decode:
                with torch.inference_mode():
                    rgb = decode_padded(trt, vae, result["final"].detach(), batch)
                    payload["rgb_u8"] = rgb.mul(255).round().clamp(0, 255).byte().cpu()
            tmp = path.with_suffix(".tmp")
            torch.save(payload, tmp)
            tmp.replace(path)
        batches.append({"offset": begin, "ids": ids, "file": path.name})
        if (begin // batch) % 10 == 0:
            print(f"  {begin + len(group)}/{len(tasks)} {time.time() - t0:.1f}s", flush=True)

    elapsed = time.time() - t0
    meta = {
        "kind": spec.kind, "ckpt": str(spec.ckpt) if spec.ckpt else None,
        "canon_mode": spec.canon_mode, "steps": spec.steps,
        "step_rms": spec.step_rms, "rho": spec.rho, "energy": spec.energy,
        "extra": spec.extra, "batch": batch, "decode": decode,
        "n_tasks": len(tasks), "wall_sec": round(elapsed, 2),
        "ms_per_img": round(elapsed * 1000 / max(len(tasks), 1), 2),
        "img_per_s": round(len(tasks) / elapsed, 2) if elapsed > 0 else None,
        "batches": batches,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--kind", default="direct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default=str(REPO / "eval_protocol/tasks_750.json"))
    ap.add_argument("--src-cache", default=str(REPO / "runs/v6_r2/cache/tasks750_source_features.pt"))
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--step-rms", type=float, default=0.08)
    ap.add_argument("--rho", type=float, default=0.16)
    ap.add_argument("--energy", default="")
    ap.add_argument("--canon-mode", default="c1_gentle")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-decode", action="store_true", help="store latents only")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--extra", default="{}", help="JSON dict for variant-specific options")
    a = ap.parse_args()

    spec = InferSpec(
        ckpt=Path(a.ckpt) if a.ckpt else None, kind=a.kind,
        canon_mode=a.canon_mode, steps=a.steps, step_rms=a.step_rms,
        rho=a.rho, energy=a.energy, extra=json.loads(a.extra))
    meta = run(spec, Path(a.out), tasks_file=Path(a.tasks), src_cache=Path(a.src_cache),
               batch=a.batch, limit=a.limit, decode=not a.no_decode,
               device=a.device, resume=not a.no_resume)
    print(f"INFER_DONE {meta['n_tasks']} tasks in {meta['wall_sec']}s "
          f"({meta['ms_per_img']} ms/img, batch={meta['batch']})", flush=True)


if __name__ == "__main__":
    main()
