"""Fast image production for V9 two-domain checkpoints (generation only, no metrics).

Design notes
------------
This is the GPU producer half of the evaluation pipeline. It deliberately does NOT
load CSD / DINOv2 / CLIP / LPIPS, so a generation pass pays no metric-model cost.
A separate consumer process can read the written PNGs or the dumped raw latents
and compute metrics without competing with this one for the GPU.

Three things make it faster than `eval_test600.py generate`:

1. **Source work is cached across checkpoints.** Source images are encoded once;
   for checkpoints 2..N the DCAE encode phase disappears entirely.
2. **No PNG round trip inside the timing-critical path.** Decoded uint8 pixels go
   straight to RAM, and PNG encoding/writing is submitted to a thread pool.
3. **The autoencoder runs in chunks.** DCAE at 512x512 is memory bound, so one
   batch of 60 is dramatically slower than 8 batches of 8.

The historical reference is `tools/eval_fast.py`, which established the same
pattern for WEAVE (single process, uint8 in RAM, async PNG writers).

Usage
-----
    python -m sbflow.generate --repo-root . \
        --ckpt runs/blog_sbflow/Mid8M_noOT/model_step1000.pt \
               runs/blog_sbflow/Mid8M_noOT/model_step1500.pt \
        --style-a Impressionism --style-b Ukiyo_e \
        --out runs/blog_sbflow/gen/Mid8M \
        --steps 4 --method heun --batch 8 --workers 4 --save-images
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image

from .cache import load_dcae
from .infer import load_bundle, solve_pfode


# --------------------------------------------------------------------------- #
# pixel helpers
# --------------------------------------------------------------------------- #
def to_u8(im: torch.Tensor) -> np.ndarray:
    """(B,3,H,W) in [-1,1] -> (B,H,W,3) uint8."""
    a = (im.detach().float().clamp(-1, 1).permute(0, 2, 3, 1).cpu().numpy() + 1) * 127.5
    return a.round().clip(0, 255).astype(np.uint8)


def write_png(u8: np.ndarray, path: Path) -> None:
    buf = BytesIO()
    Image.fromarray(u8).save(buf, format="PNG")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf.getvalue())


def load_rgb(path: Path, size: int) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)


def u8_to_tensor(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(a.astype(np.float32).copy()).permute(2, 0, 1) / 255.0


# --------------------------------------------------------------------------- #
# session
# --------------------------------------------------------------------------- #
class GenerateSession:
    """Holds DCAE and the cached source encoding; produces images for any checkpoint."""

    def __init__(self, repo: Path, src_style: str, tgt_style: str, device: str,
                 image_size: int = 512, batch: int = 8):
        from .eval_test600 import _tasks

        self.repo = repo
        self.src_style, self.tgt_style = src_style, tgt_style
        self.device = device
        self.size = image_size
        self.batch = batch
        self.tasks = _tasks(repo, src_style, tgt_style)
        self.ae = None
        self._prepared_for: Optional[str] = None
        self.timing: Dict[str, float] = {}

    # ---- one-time, checkpoint-independent -------------------------------- #
    def prepare(self, ck: dict) -> None:
        if self.ae is None:
            t0 = time.perf_counter()
            self.ae = load_dcae(ck["ae"], self.device)
            self.timing["load_dcae"] = time.perf_counter() - t0
        if self._prepared_for == ck["ae"]:
            return

        t0 = time.perf_counter()
        self.src_pils = [load_rgb(self.repo / t["source_rel_path"], self.size) for t in self.tasks]
        self.src_u8 = np.stack([np.asarray(p, dtype=np.uint8) for p in self.src_pils])
        lat, rec = [], []
        with torch.inference_mode():
            for s in range(0, len(self.tasks), self.batch):
                x = torch.stack([u8_to_tensor(a) for a in self.src_u8[s:s + self.batch]])
                x = x.to(self.device) * 2 - 1
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device == "cuda"):
                    z = self.ae.encode(x).latent.float()
                    lat.append(z)
                    rec.append(self.ae.decode(z).sample.float())
                del x
        self.src_latent_raw = torch.cat(lat, 0)
        self.noop_u8 = to_u8(torch.cat(rec, 0))
        del lat, rec
        self.timing["prepare_sources"] = time.perf_counter() - t0
        self._prepared_for = ck["ae"]

    # ---- per checkpoint --------------------------------------------------- #
    def generate(self, ckpt: Path, out_dir: Path, steps: int, method: str, t_floor: float,
                 weights: str, save_images: bool, workers: int,
                 dump_latents: Optional[Path], save_noop: bool) -> dict:
        t_start = time.perf_counter()
        ck, model, mom, mean, std = load_bundle(str(ckpt), self.device, weights)
        self.prepare(ck)

        sf = float(ck["ae_scaling_factor"])
        eps = float(ck["config"]["eps"])
        tag = f"{ckpt.parent.name}_{ckpt.stem}"
        gen_dir = out_dir / tag / "generated"
        lat_dir = (dump_latents / tag) if dump_latents else None
        if lat_dir:
            lat_dir.mkdir(parents=True, exist_ok=True)

        pool = ThreadPoolExecutor(max_workers=workers) if (save_images and workers > 0) else None
        phase = {"ode": 0.0, "decode": 0.0, "save_submit": 0.0, "dump": 0.0}
        manifest: List[dict] = []

        with torch.inference_mode():
            for start in range(0, len(self.tasks), self.batch):
                group = self.tasks[start:start + self.batch]
                sl = slice(start, start + len(group))
                z = (self.src_latent_raw[sl] * sf - mean[None, :, None, None]) / std[None, :, None, None]

                t0 = time.perf_counter()
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device == "cuda"):
                    zout = torch.empty_like(z)
                    for style, dir_name in ((self.src_style, "AtoB"), (self.tgt_style, "BtoA")):
                        pos = [j for j, t in enumerate(group) if t["source_style"] == style]
                        if pos:
                            idx = torch.tensor(pos, device=self.device)
                            zout[idx] = solve_pfode(model, z[idx].clone(), eps, mom, steps,
                                                    dir_name, t_floor, method)
                torch.cuda.synchronize()
                phase["ode"] += time.perf_counter() - t0

                if lat_dir is not None:
                    t0 = time.perf_counter()
                    np.save(lat_dir / f"batch_{start:04d}.npy", zout.float().cpu().numpy())
                    phase["dump"] += time.perf_counter() - t0

                t0 = time.perf_counter()
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device == "cuda"):
                    dec = self.ae.decode((zout * std[None, :, None, None]
                                          + mean[None, :, None, None]) / sf).sample.float()
                torch.cuda.synchronize()
                phase["decode"] += time.perf_counter() - t0
                u8 = to_u8(dec)
                del dec, zout, z

                if pool is not None:
                    t0 = time.perf_counter()
                    for j, t in enumerate(group):
                        pool.submit(write_png, u8[j], gen_dir / f"{start + j:04d}.png")
                    phase["save_submit"] += time.perf_counter() - t0

                for j, t in enumerate(group):
                    manifest.append({
                        "task_id": t["task_id"],
                        "source_style": t["source_style"],
                        "target_style": t["target_style"],
                        "source_rel_path": t["source_rel_path"],
                        "image": f"generated/{start + j:04d}.png" if pool is not None else None,
                    })

        if pool is not None:
            pool.shutdown(wait=True)
        t_save_done = time.perf_counter()

        if save_noop:
            noop_dir = out_dir / "_dcae_noop"
            if not noop_dir.exists():
                t0 = time.perf_counter()
                wpool = ThreadPoolExecutor(max_workers=max(1, workers))
                for j in range(len(self.tasks)):
                    wpool.submit(write_png, self.noop_u8[j], noop_dir / f"{j:04d}.png")
                wpool.shutdown(wait=True)
                self.timing["write_noop"] = time.perf_counter() - t0

        wall = time.perf_counter() - t_start
        nfe = steps * (2 if method == "euler" else 4)
        info = {
            "checkpoint": str(ckpt), "tag": tag, "weights": weights,
            "solver": method, "solver_steps": steps, "network_nfe": nfe,
            "n_tasks": len(self.tasks), "batch": self.batch,
            "source_style": self.src_style, "target_style": self.tgt_style,
            "timing_sec": {k: round(v, 3) for k, v in phase.items()},
            "timing_total_sec": round(wall, 2),
            "timing_save_wait_sec": round(t_save_done - t_start - sum(phase.values()), 2),
            "images_per_sec": round(len(self.tasks) / max(wall, 1e-9), 2),
            "saved_images": pool is not None,
            "dumped_latents": lat_dir is not None,
        }
        (out_dir / tag).mkdir(parents=True, exist_ok=True)
        (out_dir / tag / "manifest.json").write_text(
            json.dumps({"info": info, "tasks": manifest}, ensure_ascii=False, indent=2), encoding="utf-8")

        if lat_dir is not None:
            (lat_dir / "meta.json").write_text(json.dumps({
                "ae": ck["ae"], "ae_scaling_factor": sf,
                "latent_mean": ck["latent_mean"].tolist(),
                "latent_std": ck["latent_std"].tolist(),
                "source_style": self.src_style, "target_style": self.tgt_style,
                "image_size": self.size,
                "tasks": self.tasks,
            }, ensure_ascii=False), encoding="utf-8")

        print(f"  [{tag}] {len(self.tasks)} imgs in {wall:.1f}s "
              f"({info['images_per_sec']:.1f} img/s) ode={phase['ode']:.1f} "
              f"dec={phase['decode']:.1f} dump={phase['dump']:.1f} "
              f"submit={phase['save_submit']:.1f}", flush=True)
        return info


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--style-a", default="Impressionism")
    ap.add_argument("--style-b", default="Ukiyo_e")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--method", choices=["euler", "heun"], default="heun")
    ap.add_argument("--weights", choices=["raw", "ema"], default="raw")
    ap.add_argument("--t-floor", type=float, default=0.002)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--save-images", action="store_true", help="write generated PNGs")
    ap.add_argument("--workers", type=int, default=4, help="async PNG writer threads (0 = inline)")
    ap.add_argument("--dump-latents", default="", help="also write raw generated latents (.npy)")
    ap.add_argument("--save-noop", action="store_true", help="write the DCAE no-op baseline once")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    repo = Path(args.repo_root).resolve()
    out = Path(args.out)
    dump = Path(args.dump_latents) if args.dump_latents else None
    if dump:
        dump.mkdir(parents=True, exist_ok=True)

    t_all = time.perf_counter()
    sess = GenerateSession(repo, args.style_a, args.style_b, device, args.image_size, args.batch)
    print(f"[generate] tasks={len(sess.tasks)} device={device} batch={args.batch} "
          f"steps={args.steps} method={args.method} nfe={args.steps * (2 if args.method == 'euler' else 4)}",
          flush=True)

    infos = []
    for ckpt in args.ckpt:
        infos.append(sess.generate(Path(ckpt), out, args.steps, args.method, args.t_floor,
                                   args.weights, args.save_images, args.workers, dump, args.save_noop))

    total = time.perf_counter() - t_all
    print(f"[generate] {len(infos)} checkpoint(s) in {total:.1f}s "
          f"(one-time cached: { {k: round(v, 1) for k, v in sess.timing.items()} })", flush=True)
    summary = out / "summary.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps({"session_timing": {k: round(v, 3) for k, v in sess.timing.items()},
                                   "total_sec": round(total, 2),
                                   "checkpoints": infos}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[generate] summary -> {summary}", flush=True)


if __name__ == "__main__":
    main()
