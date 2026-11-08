"""Fast evaluation: one process, no PNG round trip, checkpoint-independent work cached.

Reference pattern: `tools/eval_fast.py` (historical WEAVE implementation) which keeps
generation + CLIP/LPIPS/DINO in one process, quantises once, keeps uint8 in RAM and
writes PNGs from a background thread pool.

What this module adds on top of `eval_test600.py`:

  1. DCAE + EvaluatorModels are loaded ONCE for all checkpoints (was 24 s per run).
  2. Source latents, source CSD features and the DCAE no-op baseline are computed
     ONCE (they do not depend on the checkpoint).
  3. Generation, decode, feature extraction and metrics stay in one process, so the
     PNG round trip disappears; uint8 pixels are kept in RAM.
  4. PNG writing is optional and asynchronous (`--save-images --workers N`).
  5. `--dump-latents DIR` writes raw generated latents so a *separate* process can
     decode/score without competing with the GPU during generation
     (`--score-latents DIR`), which is the non-blocking pipeline the user asked for.

Usage
  python -m sbflow.eval_fast --repo-root . \
      --ckpt runs/blog_sbflow/Mid8M_noOT/model_step1000.pt \
             runs/blog_sbflow/Mid8M_noOT/model_step1500.pt \
      --style-a Impressionism --style-b Ukiyo_e \
      --out runs/blog_sbflow/fast_eval/Mid8M --steps 4 --method heun

  python -m sbflow.eval_fast --repo-root . --score-latents runs/blog_sbflow/latents/Mid8M \
      --out runs/blog_sbflow/fast_eval/Mid8M_from_latents
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image

from .cache import image_tensor, load_dcae
from .eval_test600 import _mmd2, _tasks
from .infer import load_bundle, solve_pfode

STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
KEYS = ("csd_dT", "csd_dS", "target_margin", "target_top1", "lpips_to_input")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def png_bytes(u8: np.ndarray) -> bytes:
    buf = BytesIO()
    Image.fromarray(u8).save(buf, format="PNG")
    return buf.getvalue()


def write_png(u8: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png_bytes(u8))


def to_u8(im: torch.Tensor) -> np.ndarray:
    """(B,3,H,W) in [-1,1] -> (B,H,W,3) uint8."""
    a = ((im.detach().float().clamp(-1, 1).permute(0, 2, 3, 1).cpu().numpy() + 1) * 127.5)
    return a.round().clip(0, 255).astype(np.uint8)


def aggregate(rows: List[dict], refs: dict, gamma: float, src_style: str, tgt_style: str) -> dict:
    """Same schema as eval_test600.score so downstream tools keep working."""
    agg = {}
    for tag in ("generated", "dcae_noop"):
        one = [r for r in rows if r["variant"] == tag]
        agg[tag] = {}
        for style in (src_style, tgt_style):
            vals = [v for v in one if v["target"] == style]
            if not vals:
                continue
            feat = np.asarray([v["csd_embedding"] for v in vals], dtype=np.float32)
            agg[tag][style] = {k: float(np.mean([v[k] for v in vals])) for k in KEYS}
            agg[tag][style]["csd_mmd2_unbiased"] = _mmd2(feat, refs["styles"][style]["csd"], gamma)
        agg[tag]["cross_overall"] = {k: float(np.mean([v[k] for v in one])) for k in KEYS}
    return agg


# --------------------------------------------------------------------------- #
# core
# --------------------------------------------------------------------------- #
class FastEval:
    """Loads everything once and evaluates any number of checkpoints."""

    def __init__(self, repo: Path, src_style: str, tgt_style: str, device: str):
        self.repo = repo
        self.src_style, self.tgt_style = src_style, tgt_style
        self.device = device
        self.tasks = _tasks(repo, src_style, tgt_style)
        self.timing: Dict[str, float] = {}

        sys.path.insert(0, str(repo))
        from tools.evaluation_v2.features import EvaluatorModels, build_reference_features_cache

        t0 = time.perf_counter()
        self.ev = EvaluatorModels(device)
        self.refs = build_reference_features_cache(self.ev)
        self.gamma = float(self.refs["rbf_gamma"])
        proto = self.ev.proto_csd
        self.proto = proto.detach().float().cpu().numpy() if torch.is_tensor(proto) else np.asarray(proto, np.float32)
        self.timing["load_evaluator"] = time.perf_counter() - t0

        self.ae = None
        self.src_pils: List[Image.Image] = []
        self.src_u8: Optional[np.ndarray] = None
        self.src_csd: Optional[np.ndarray] = None
        self.src_values: Optional[np.ndarray] = None
        self.noop_u8: Optional[np.ndarray] = None
        self.z_src: Optional[torch.Tensor] = None
        self._prepared_for: Optional[str] = None

    # ---- checkpoint-independent preparation ------------------------------- #
    def prepare(self, ckpt: str, weights: str = "raw", batch: int = 8) -> dict:
        """Load DCAE + source latents + no-op baseline. Reuses work across checkpoints."""
        ck, model, mom, mean, std = load_bundle(ckpt, self.device, weights)
        if self.ae is None:
            t0 = time.perf_counter()
            self.ae = load_dcae(ck["ae"], self.device)
            self.timing["load_dcae"] = time.perf_counter() - t0

        size = 512
        if self._prepared_for != ck["ae"]:
            t0 = time.perf_counter()
            self.src_pils = [
                Image.open(self.repo / t["source_rel_path"]).convert("RGB").resize((size, size), Image.BICUBIC)
                for t in self.tasks
            ]
            self.src_u8 = np.stack([np.asarray(p, dtype=np.uint8) for p in self.src_pils])
            # Chunk the autoencoder work: DCAE at 512x512 is memory bound, so a single
            # batch of 60 costs far more than 8 batches of 8.
            lat, rec = [], []
            with torch.inference_mode():
                for s in range(0, len(self.tasks), batch):
                    x = torch.stack([image_tensor_from_u8(a) for a in self.src_u8[s:s + batch]])
                    x = x.to(self.device) * 2 - 1
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device == "cuda"):
                        z = self.ae.encode(x).latent.float()
                        lat.append(z)
                        rec.append(self.ae.decode(z).sample.float())
                    del x
            self.src_latent_raw = torch.cat(lat, 0)
            self.noop_u8 = to_u8(torch.cat(rec, 0))
            del lat, rec
            feats = self.ev.extract_image_features(self.src_pils)
            self.src_csd = feats["csd"].numpy()
            self.src_values = self.src_csd @ self.proto.T
            self.timing["prepare_sources"] = time.perf_counter() - t0
            self._prepared_for = ck["ae"]
        return {"ck": ck, "model": model, "mom": mom, "mean": mean, "std": std}

    # ---- per checkpoint ---------------------------------------------------- #
    def run(self, ckpt: str, weights: str, steps: int, method: str, t_floor: float,
            out_dir: Path, save_images: bool, workers: int, dump_latents: Optional[Path],
            batch: int) -> dict:
        from .infer import load_bundle

        bundle = self.prepare(ckpt, weights, batch)
        ck, model, mom, mean, std = (bundle[k] for k in ("ck", "model", "mom", "mean", "std"))
        sf = float(ck["ae_scaling_factor"])
        eps = float(ck["config"]["eps"])
        mean_t, std_t = mean, std

        pool = ThreadPoolExecutor(max_workers=workers) if (save_images and workers > 0) else None
        gen_dir = out_dir / "generated"
        tag = Path(ckpt).parent.name + "_" + Path(ckpt).stem
        lat_dir = (dump_latents / tag) if dump_latents else None
        if lat_dir:
            lat_dir.mkdir(parents=True, exist_ok=True)

        rows: List[dict] = []
        gen_u8_all = np.zeros_like(self.src_u8)
        t_gen = time.perf_counter()
        phase = {"ode": 0.0, "decode": 0.0, "feat": 0.0, "lpips": 0.0, "save": 0.0, "io": 0.0}

        with torch.inference_mode():
            for start in range(0, len(self.tasks), batch):
                group = self.tasks[start:start + batch]
                sl = slice(start, start + len(group))
                t0 = time.perf_counter()
                z = (self.src_latent_raw[sl] * sf - mean_t[None, :, None, None]) / std_t[None, :, None, None]
                phase["io"] += time.perf_counter() - t0

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
                    np.save(lat_dir / f"batch_{start:04d}.npy", zout.float().cpu().numpy())

                t0 = time.perf_counter()
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device == "cuda"):
                    dec = self.ae.decode((zout * std_t[None, :, None, None]
                                          + mean_t[None, :, None, None]) / sf).sample.float()
                torch.cuda.synchronize()
                phase["decode"] += time.perf_counter() - t0
                u8 = to_u8(dec)
                gen_u8_all[sl] = u8

                t0 = time.perf_counter()
                pils = [Image.fromarray(a) for a in u8]
                feats = self.ev.extract_image_features(pils)
                torch.cuda.synchronize()
                phase["feat"] += time.perf_counter() - t0

                t0 = time.perf_counter()
                csd = feats["csd"].numpy()
                values = csd @ self.proto.T
                for j, t in enumerate(group):
                    tid = STYLES.index(t["target_style"])
                    sid = STYLES.index(t["source_style"])
                    other = np.delete(values[j], tid)
                    a = torch.from_numpy(self.src_u8[sl][j].copy()).permute(2, 0, 1).to(self.device) / 127.5 - 1
                    b = torch.from_numpy(u8[j].copy()).permute(2, 0, 1).to(self.device) / 127.5 - 1
                    lps = float(self.ev.lpips(a[None], b[None]).mean().item())
                    key = f"{start + j:04d}.png"
                    rows.append({
                        "task_id": t["task_id"], "source": t["source_style"], "target": t["target_style"],
                        "variant": "generated",
                        "csd_dT": float(values[j, tid] - self.src_values[start + j, tid]),
                        "csd_dS": float(values[j, sid] - self.src_values[start + j, sid]),
                        "target_margin": float(values[j, tid] - other.max()),
                        "target_top1": int(values[j].argmax() == tid),
                        "lpips_to_input": lps,
                        "csd_embedding": csd[j].tolist(),
                        "image": key,
                    })
                phase["lpips"] += time.perf_counter() - t0

                if pool is not None:
                    t0 = time.perf_counter()
                    for j, t in enumerate(group):
                        pool.submit(write_png, u8[j], gen_dir / f"{start + j:04d}.png")
                    phase["save"] += time.perf_counter() - t0

        if pool is not None:
            pool.shutdown(wait=True)
        wall = time.perf_counter() - t_gen

        # ---- no-op baseline rows (checkpoint independent, cached pixels) ---- #
        t0 = time.perf_counter()
        noop_feats = self.ev.extract_image_features([Image.fromarray(a) for a in self.noop_u8])
        noop_values = noop_feats["csd"].numpy() @ self.proto.T
        noop_csd = noop_feats["csd"].numpy()
        noop_rows = []
        for j, t in enumerate(self.tasks):
            tid = STYLES.index(t["target_style"])
            sid = STYLES.index(t["source_style"])
            other = np.delete(noop_values[j], tid)
            a = torch.from_numpy(self.src_u8[j].copy()).permute(2, 0, 1).to(self.device) / 127.5 - 1
            b = torch.from_numpy(self.noop_u8[j].copy()).permute(2, 0, 1).to(self.device) / 127.5 - 1
            lps = float(self.ev.lpips(a[None], b[None]).mean().item())
            noop_rows.append({
                "task_id": t["task_id"], "source": t["source_style"], "target": t["target_style"],
                "variant": "dcae_noop",
                "csd_dT": float(noop_values[j, tid] - self.src_values[j, tid]),
                "csd_dS": float(noop_values[j, sid] - self.src_values[j, sid]),
                "target_margin": float(noop_values[j, tid] - other.max()),
                "target_top1": int(noop_values[j].argmax() == tid),
                "lpips_to_input": lps,
                "csd_embedding": noop_csd[j].tolist(),
            })
        self.timing["noop_baseline"] = self.timing.get("noop_baseline", 0.0) + (time.perf_counter() - t0)

        rows_all = rows + noop_rows
        agg = aggregate(rows_all, self.refs, self.gamma, self.src_style, self.tgt_style)
        nfe = steps * (2 if method == "euler" else 4)
        out = {
            "protocol": {
                "source": "WEAVE frozen Test600 (pair subset), fast single-process path",
                "n_tasks": len(self.tasks), "source_style": self.src_style,
                "target_style": self.tgt_style, "network_nfe": nfe,
                "mmd_gamma": self.gamma, "checkpoint": ckpt, "weights": weights,
                "solver": method, "solver_steps": steps,
                "timing_generate_sec": {k: round(v, 3) for k, v in phase.items()},
                "timing_generate_total_sec": round(wall, 2),
            },
            "aggregate": agg,
            "rows": rows_all,
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "style_scores.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                                   encoding="utf-8")
        g = agg["generated"]["cross_overall"]
        print(f"  [{tag}] top1={g['target_top1']:.4f} margin={g['target_margin']:+.4f} "
              f"dT={g['csd_dT']:+.4f} lpips={g['lpips_to_input']:.4f} "
              f"| gen={wall:.1f}s ode={phase['ode']:.1f} dec={phase['decode']:.1f} "
              f"feat={phase['feat']:.1f}", flush=True)
        return out


def image_tensor_from_u8(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(a.astype(np.float32).copy()).permute(2, 0, 1) / 255.0


# --------------------------------------------------------------------------- #
# latent-consuming scorer (the "other process" path)
# --------------------------------------------------------------------------- #
def score_latents(repo: Path, lat_root: Path, out_dir: Path, batch: int, device: str) -> None:
    """Decode + score previously dumped latents. Touches no generator state."""
    sys.path.insert(0, str(repo))
    from tools.evaluation_v2.features import EvaluatorModels, build_reference_features_cache

    ev = EvaluatorModels(device)
    refs = build_reference_features_cache(ev)
    gamma = float(refs["rbf_gamma"])
    proto = ev.proto_csd
    proto = proto.detach().float().cpu().numpy() if torch.is_tensor(proto) else np.asarray(proto, np.float32)

    meta = json.loads((lat_root / "meta.json").read_text(encoding="utf-8"))
    ae = load_dcae(meta["ae"], device)
    mean = torch.as_tensor(meta["latent_mean"], device=device)
    std = torch.as_tensor(meta["latent_std"], device=device)
    sf = float(meta["ae_scaling_factor"])
    tasks = meta["tasks"]
    src_u8 = np.stack([np.asarray(Image.open(repo / t["source_rel_path"]).convert("RGB").resize((512, 512)),
                                  dtype=np.uint8) for t in tasks])
    src_csd = ev.extract_image_features([Image.fromarray(a) for a in src_u8])["csd"].numpy()
    src_values = src_csd @ proto.T

    files = sorted(lat_root.glob("batch_*.npy"))
    z = np.concatenate([np.load(f) for f in files], axis=0)
    t0 = time.perf_counter()
    with torch.inference_mode():
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            dec = ae.decode((torch.as_tensor(z, device=device) * std[None, :, None, None]
                             + mean[None, :, None, None]) / sf).sample.float()
    u8 = to_u8(dec)
    feats = ev.extract_image_features([Image.fromarray(a) for a in u8])["csd"].numpy()
    values = feats @ proto.T
    rows = []
    for j, t in enumerate(tasks):
        tid = STYLES.index(t["target_style"])
        sid = STYLES.index(t["source_style"])
        other = np.delete(values[j], tid)
        a = torch.from_numpy(src_u8[j].copy()).permute(2, 0, 1).to(device) / 127.5 - 1
        b = torch.from_numpy(u8[j].copy()).permute(2, 0, 1).to(device) / 127.5 - 1
        lps = float(ev.lpips(a[None], b[None]).mean().item())
        rows.append({"task_id": t["task_id"], "source": t["source_style"], "target": t["target_style"],
                     "variant": "generated",
                     "csd_dT": float(values[j, tid] - src_values[j, tid]),
                     "csd_dS": float(values[j, sid] - src_values[j, sid]),
                     "target_margin": float(values[j, tid] - other.max()),
                     "target_top1": int(values[j].argmax() == tid),
                     "lpips_to_input": lps, "csd_embedding": feats[j].tolist()})
    wall = time.perf_counter() - t0
    agg = aggregate(rows, refs, gamma, meta["source_style"], meta["target_style"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "style_scores.json").write_text(json.dumps({
        "protocol": {"source": "latent-consumer path", "n_tasks": len(tasks),
                     "source_style": meta["source_style"], "target_style": meta["target_style"],
                     "mmd_gamma": gamma, "decode_score_sec": round(wall, 2)},
        "aggregate": agg, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    g = agg["generated"]["cross_overall"]
    print(f"  [latents] top1={g['target_top1']:.4f} margin={g['target_margin']:+.4f} "
          f"dT={g['csd_dT']:+.4f} decode+score={wall:.1f}s", flush=True)


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--ckpt", nargs="*", default=[])
    ap.add_argument("--style-a", default="Impressionism")
    ap.add_argument("--style-b", default="Ukiyo_e")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--method", choices=["euler", "heun"], default="heun")
    ap.add_argument("--weights", choices=["raw", "ema"], default="raw")
    ap.add_argument("--t-floor", type=float, default=0.002)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--save-images", action="store_true", help="also write generated PNGs")
    ap.add_argument("--workers", type=int, default=4, help="async PNG writer threads")
    ap.add_argument("--dump-latents", default="", help="write raw generated latents here")
    ap.add_argument("--score-latents", default="", help="consume dumped latents instead of generating")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    repo = Path(args.repo_root).resolve()

    if args.score_latents:
        score_latents(repo, Path(args.score_latents), Path(args.out), args.batch, device)
        return

    if not args.ckpt:
        raise SystemExit("--ckpt is required unless --score-latents is used")

    t_all = time.perf_counter()
    ev = FastEval(repo, args.style_a, args.style_b, device)
    print(f"[fast-eval] tasks={len(ev.tasks)} device={device} "
          f"load_evaluator={ev.timing['load_evaluator']:.1f}s", flush=True)

    dump_root = Path(args.dump_latents) if args.dump_latents else None
    if dump_root:
        dump_root.mkdir(parents=True, exist_ok=True)

    for ckpt in args.ckpt:
        t0 = time.perf_counter()
        out = ev.run(ckpt, args.weights, args.steps, args.method, args.t_floor,
                     Path(args.out), args.save_images, args.workers, dump_root, args.batch)
        print(f"  total {time.perf_counter() - t0:.1f}s", flush=True)

        if dump_root:
            ck = torch.load(ckpt, map_location="cpu", weights_only=False)
            (dump_root / (Path(ckpt).parent.name + "_" + Path(ckpt).stem) / "meta.json").write_text(
                json.dumps({
                    "ae": ck["ae"], "ae_scaling_factor": float(ck["ae_scaling_factor"]),
                    "latent_mean": ck["latent_mean"].tolist(), "latent_std": ck["latent_std"].tolist(),
                    "source_style": args.style_a, "target_style": args.style_b,
                    "tasks": ev.tasks,
                }, ensure_ascii=False), encoding="utf-8")

    print(f"[fast-eval] all done in {time.perf_counter() - t_all:.1f}s; "
          f"cached timing={ {k: round(v, 1) for k, v in ev.timing.items()} }", flush=True)


if __name__ == "__main__":
    main()
